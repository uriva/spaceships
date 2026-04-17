#!/usr/bin/env -S deno run --allow-read --allow-write
// train-follow.js — Train a TINY network to steer toward a mark.
// Uses [3, 4, 3] topology (only 31 params) for fast evolution,
// then embeds result into full [80,16,16,5] for the game.

const simCoreSrc = Deno.readTextFileSync(
  new URL("./sim-core.js", import.meta.url).pathname,
);
(new Function(simCoreSrc))();
const {
  V3,
  Q,
  PHYSICS,
  NeuralNetwork,
  createShipState,
  shipSimStep,
  applyNNOutputs,
  initPopulation,
  evolve,
} = globalThis.SimCore;

// ── Config ──
const TINY_TOPOLOGY = [3, 4, 3]; // localPos → pitch/yaw/burn
const POP_SIZE = 300;
const MUTATION_RATE = 0.20;
const MUTATION_STRENGTH = 0.5;
const OUTPUT_FILE = Deno.args[0] || "best-genome.json";
const ARENA_RADIUS = 50;
const MATCH_FRAMES = 600; // 10 seconds
const EVALS_PER_GENOME = 5; // more evals = less noise
const RUN_MINUTES = 30;
const CLOSE_THRESHOLD = 10;

// ── Temp vectors ──
const _fwd = [0, 0, 0];
const _right = [0, 0, 0];
const _up = [0, 0, 0];
const _rel = [0, 0, 0];
const _local = [0, 0, 0];
const _invQ = [0, 0, 0, 1];

// ── Compute local-frame relative position of mark ──
function getLocalRelPos(ship, mark) {
  _rel[0] = mark.pos[0] - ship.pos[0];
  _rel[1] = mark.pos[1] - ship.pos[1];
  _rel[2] = mark.pos[2] - ship.pos[2];
  Q.invertInto(ship.quat, _invQ);
  Q.applyToVec3Into(_invQ, _rel[0], _rel[1], _rel[2], _local);
  // Normalize to [-1, 1] range (divide by 50)
  return [_local[0] / 50, _local[1] / 50, _local[2] / 50];
}

// ── Apply tiny NN outputs as pitch/yaw/burn via the full physics ──
function applyTinyOutputs(ship, tinyOutputs) {
  // tinyOutputs: [pitch, yaw, burn] each in tanh range [-1, 1]
  // Build a full 5-output array: [pitch, yaw, burn, targetSelect=0, fire=0]
  const fullOutputs = [tinyOutputs[0], tinyOutputs[1], tinyOutputs[2], 0, 0];
  applyNNOutputs(ship, fullOutputs, [ship]); // no enemies to shoot
}

// ══════════════════════════════════════════════════════════════
function runEpisode(brain) {
  const ship = createShipState("alpha");
  Q.setRandomMut(ship.quat);

  // Stationary mark at random position, distance 15-40
  const dir = V3.normalize([
    Math.random() - 0.5,
    Math.random() - 0.5,
    Math.random() - 0.5,
  ]);
  const dist0 = 15 + Math.random() * 25;
  const markPos = [dir[0] * dist0, dir[1] * dist0, dir[2] * dist0];
  const mark = { pos: markPos }; // minimal mark object

  let proximityScore = 0;

  for (let frame = 0; frame < MATCH_FRAMES; frame++) {
    const localPos = getLocalRelPos(ship, mark);
    const outputs = brain.forward(localPos);
    applyTinyOutputs(ship, outputs);
    shipSimStep(ship);

    const dist = V3.distanceTo(ship.pos, markPos);
    proximityScore += Math.exp(-dist / CLOSE_THRESHOLD);
  }

  return proximityScore;
}

// ══════════════════════════════════════════════════════════════
function evaluate(genome) {
  const brain = new NeuralNetwork(TINY_TOPOLOGY);
  brain.fromGenome(genome);
  let total = 0;
  for (let i = 0; i < EVALS_PER_GENOME; i++) {
    total += runEpisode(brain);
  }
  return total / EVALS_PER_GENOME;
}

// ══════════════════════════════════════════════════════════════
// ██  Embed tiny [3,4,3] weights into full [80,16,16,5] genome
// ══════════════════════════════════════════════════════════════
function embedIntoFull(tinyGenome) {
  const FULL_TOPOLOGY = [80, 16, 16, 5];
  const fullNN = new NeuralNetwork(FULL_TOPOLOGY);
  const fullGenome = fullNN.toGenome();
  fullGenome.fill(0);

  // Tiny topology [3, 4, 3]:
  //   Layer 0: 3→4 weights (12) + 4 biases = 16
  //   Layer 1: 4→3 weights (12) + 3 biases = 15
  //   Total: 31 params

  // Full topology [80, 16, 16, 5]:
  //   Layer 0: 80→16 weights (1280) + 16 biases = 1296
  //   Layer 1: 16→16 weights (256) + 16 biases = 272
  //   Layer 2: 16→5 weights (80) + 5 biases = 85
  //   Total: 1653 params

  // The tiny NN maps: input[0,1,2] (local x,y,z) → output[0,1,2] (pitch,yaw,burn)
  // In the full NN: input[10,11,12] = enemy local position / 50
  //                 output[0,1,2] = pitch, yaw, burn

  // Strategy: wire tiny layer 0 into full layer 0 (inputs 10-12 → hidden 0-3),
  //           tiny layer 1 into full layer 2 (hidden 0-3 → outputs 0-2),
  //           full layer 1 passes hidden 0-3 through unchanged.

  const tinyArr = Array.from(tinyGenome);
  let tOff = 0;

  // ── Tiny Layer 0 (3→4): weights + biases → Full Layer 0 ──
  // Tiny weight[h][i] at tOff + h*3 + i (h=0..3, i=0..2)
  // Full Layer 0 weight[h][j] at h*80 + j (h=0..15, j=0..79)
  // Map: tiny input i → full input (10+i), tiny hidden h → full hidden h
  for (let h = 0; h < 4; h++) {
    for (let i = 0; i < 3; i++) {
      fullGenome[h * 80 + (10 + i)] = tinyArr[tOff + h * 3 + i];
    }
  }
  tOff += 12; // 3*4 weights
  // Tiny biases[h] at tOff + h → Full Layer 0 biases at 1280 + h
  for (let h = 0; h < 4; h++) {
    fullGenome[1280 + h] = tinyArr[tOff + h];
  }
  tOff += 4;

  // ── Full Layer 1: identity pass-through for hidden 0-3 ──
  const L1_OFF = 1296;
  for (let i = 0; i < 4; i++) {
    fullGenome[L1_OFF + i * 16 + i] = 5.0; // strong pass-through (tanh(5*tanh(x)) ≈ sign(x))
  }
  // Actually we need linear pass-through. tanh(5*x) saturates.
  // Use weight=1.0 for approximate identity through tanh.
  // tanh(1.0 * tanh(x)) ≈ tanh(x) * 0.76... not great.
  // Better: use weight=2.0 so tanh(2*tanh(x)) recovers more signal.
  for (let i = 0; i < 4; i++) {
    fullGenome[L1_OFF + i * 16 + i] = 2.0;
  }

  // ── Tiny Layer 1 (4→3): weights + biases → Full Layer 2 ──
  // Tiny weight[o][h] at tOff + o*4 + h (o=0..2, h=0..3)
  // Full Layer 2 weight[o][h] at L2_OFF + o*16 + h
  const L2_OFF = 1568;
  for (let o = 0; o < 3; o++) {
    for (let h = 0; h < 4; h++) {
      fullGenome[L2_OFF + o * 16 + h] = tinyArr[tOff + o * 4 + h];
    }
  }
  tOff += 12;
  // Tiny biases[o] at tOff + o → Full Layer 2 biases at L2_OFF + 80 + o
  for (let o = 0; o < 3; o++) {
    fullGenome[L2_OFF + 80 + o] = tinyArr[tOff + o];
  }

  return Array.from(fullGenome);
}

// ══════════════════════════════════════════════════════════════
// ██  Training loop
// ══════════════════════════════════════════════════════════════
let population = initPopulation(POP_SIZE, TINY_TOPOLOGY);

let globalBestFitness = -Infinity;
let globalBestGenome = null;

console.log("Follow-mark trainer (tiny network)");
console.log(`Pop: ${POP_SIZE}, evals/genome: ${EVALS_PER_GENOME}`);
console.log(
  `Topology: ${TINY_TOPOLOGY.join("→")}, params: ${
    new NeuralNetwork(TINY_TOPOLOGY).paramCount
  }`,
);
console.log(`Running for ${RUN_MINUTES} minutes...`);

const t0 = performance.now();
const timeLimitMs = RUN_MINUTES * 60 * 1000;
let gen = 0;

while (performance.now() - t0 < timeLimitMs) {
  const genStart = performance.now();
  gen++;

  for (const p of population) {
    p.fitness = evaluate(p.genome);
  }

  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
  const topFitness = sorted[0].fitness;
  const avgFitness = sorted.reduce((s, p) => s + p.fitness, 0) / POP_SIZE;

  if (topFitness > globalBestFitness) {
    globalBestFitness = topFitness;
    globalBestGenome = Array.from(sorted[0].genome);
  }

  population = evolve(population, {
    popSize: POP_SIZE,
    mutationRate: MUTATION_RATE,
    mutationStrength: MUTATION_STRENGTH,
  });

  const genMs = (performance.now() - genStart).toFixed(0);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(0);
  const remain = Math.max(0, (timeLimitMs - (performance.now() - t0)) / 1000)
    .toFixed(0);

  if (gen % 10 === 0 || gen <= 5) {
    console.log(
      `GEN ${String(gen).padStart(4)} | ` +
        `top: ${topFitness.toFixed(1).padStart(8)} | ` +
        `avg: ${avgFitness.toFixed(1).padStart(8)} | ` +
        `best: ${globalBestFitness.toFixed(1).padStart(8)} | ` +
        `${genMs}ms | ${elapsed}s/${remain}s left`,
    );
  }

  if (gen % 50 === 0 && globalBestGenome) {
    const fullGenome = embedIntoFull(globalBestGenome);
    const checkpoint = {
      topology: [80, 16, 16, 5],
      tinyTopology: TINY_TOPOLOGY,
      fitness: globalBestFitness,
      genome: fullGenome,
      tinyGenome: globalBestGenome,
      generation: gen,
      scenario: "follow-mark",
      config: {
        POP_SIZE,
        EVALS_PER_GENOME,
        MUTATION_RATE,
        MUTATION_STRENGTH,
        ARENA_RADIUS,
        MATCH_FRAMES,
      },
      trainedAt: new Date().toISOString(),
    };
    Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(checkpoint, null, 2));
  }
}

// Final save
if (globalBestGenome) {
  const fullGenome = embedIntoFull(globalBestGenome);
  const final = {
    topology: [80, 16, 16, 5],
    tinyTopology: TINY_TOPOLOGY,
    fitness: globalBestFitness,
    genome: fullGenome,
    tinyGenome: globalBestGenome,
    generation: gen,
    scenario: "follow-mark",
    config: {
      POP_SIZE,
      EVALS_PER_GENOME,
      MUTATION_RATE,
      MUTATION_STRENGTH,
      ARENA_RADIUS,
      MATCH_FRAMES,
    },
    trainedAt: new Date().toISOString(),
  };
  Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(final, null, 2));
}

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: ${gen} generations in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
