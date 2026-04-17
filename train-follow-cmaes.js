#!/usr/bin/env -S deno run --allow-read --allow-write
// train-follow-cmaes.js — sep-CMA-ES trainer for "follow" behavior
// Trains a ReLU network to fly a ship to a target point and stop.
// Inputs: local target pos (3) + local velocity (3) = 6
// Outputs: pitch, yaw, burn (3) — tanh output layer
// Optimizer: sep-CMA-ES (diagonal covariance, O(N) per generation)

const simCoreSrc = Deno.readTextFileSync(
  new URL("./sim-core.js", import.meta.url).pathname,
);
(new Function(simCoreSrc))();
const {
  V3,
  Q,
  PHYSICS,
  ReLUNetwork,
  createShipState,
  shipSimStep,
  applyNNOutputs,
} = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [6, 16, 16, 3];
const OUTPUT_FILE = Deno.args[0] || "follow-brain.json";
const MATCH_FRAMES = 600; // 10 sec at 60fps
const EVALS_PER_CANDIDATE = 16; // average over diverse scenarios (8 structured + 8 random)
const RUN_MINUTES = parseFloat(Deno.args[1] || "30");
const SEED_FILE = Deno.args[2] || "follow-brain.json"; // optional: seed from existing brain

// sep-CMA-ES hyperparameters
const nn = new ReLUNetwork(TOPOLOGY);
const N = nn.paramCount; // dimensionality
const LAMBDA = Math.max(16, 4 + Math.floor(3 * Math.log(N))); // population size
const MU = Math.floor(LAMBDA / 2); // number of parents

// Recombination weights (log-linear, truncation selection)
const rawWeights = [];
for (let i = 0; i < MU; i++) {
  rawWeights.push(Math.log(MU + 0.5) - Math.log(i + 1));
}
const wSum = rawWeights.reduce((a, b) => a + b, 0);
const weights = rawWeights.map((w) => w / wSum);
const mueff = 1.0 / weights.reduce((a, w) => a + w * w, 0);

// Adaptation parameters (sep-CMA-ES: same formulas, diagonal only)
const cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
const cs = (mueff + 2) / (N + mueff + 5);
const c1 = 2 / ((N + 1.3) ** 2 + mueff);
const cmu = Math.min(
  1 - c1,
  2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff),
);
const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (N + 1)) - 1) + cs;
const chiN = Math.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N));

// ══════════════════════════════════════════════════════════════
// ██  sep-CMA-ES State (diagonal covariance)
// ══════════════════════════════════════════════════════════════
const mean = new Float64Array(N); // distribution mean
let sigma = 0.5; // step size (adjusted below if seeding)
const pc = new Float64Array(N); // evolution path (covariance)
const ps = new Float64Array(N); // evolution path (sigma)
const diagC = new Float64Array(N); // diagonal of covariance matrix
diagC.fill(1.0);

// Initialize mean: seed from existing brain if available, else He init
let seeded = false;
try {
  const seedData = JSON.parse(Deno.readTextFileSync(SEED_FILE));
  if (
    seedData.genome && seedData.genome.length === N &&
    JSON.stringify(seedData.topology) === JSON.stringify(TOPOLOGY)
  ) {
    for (let i = 0; i < N; i++) mean[i] = seedData.genome[i];
    seeded = true;
    // Start with smaller sigma to refine around known-good solution
    sigma = seedData.optimizer?.sigma || 0.2;
    // But don't go too small — ensure exploration
    sigma = Math.max(sigma, 0.1);
    console.log(
      `Seeded from ${SEED_FILE} (fitness ${
        seedData.fitness?.toFixed(1)
      }, gen ${seedData.generation}, σ=${sigma.toFixed(4)})`,
    );
  }
} catch (_) { /* no seed file or wrong format */ }

if (!seeded) {
  let idx = 0;
  for (let li = 0; li < TOPOLOGY.length - 1; li++) {
    const fanIn = TOPOLOGY[li];
    const scale = Math.sqrt(2.0 / fanIn); // He init for ReLU
    const nW = TOPOLOGY[li] * TOPOLOGY[li + 1];
    for (let j = 0; j < nW; j++) mean[idx++] = (Math.random() * 2 - 1) * scale;
    const nB = TOPOLOGY[li + 1];
    for (let j = 0; j < nB; j++) mean[idx++] = 0; // zero biases
  }
  console.log("No seed found, using He initialization");
}

// ══════════════════════════════════════════════════════════════
// ██  Sampling (diagonal: just scale by sqrt(diagC))
// ══════════════════════════════════════════════════════════════
function sampleCandidate() {
  const z = new Float64Array(N);
  const y = new Float64Array(N);
  const x = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    // Box-Muller
    const u1 = Math.random(), u2 = Math.random();
    z[i] = Math.sqrt(-2 * Math.log(u1 + 1e-30)) * Math.cos(2 * Math.PI * u2);
    y[i] = Math.sqrt(diagC[i]) * z[i];
    x[i] = mean[i] + sigma * y[i];
  }

  return { x, y, z };
}

// ══════════════════════════════════════════════════════════════
// ██  Fitness evaluation: fly ship to target and stop
// ══════════════════════════════════════════════════════════════
const _invQ = [0, 0, 0, 1];
const _localPos = [0, 0, 0];
const _localVel = [0, 0, 0];

// Structured scenarios to ensure coverage of hard cases.
// Each: { target: [x,y,z], vel: [vx,vy,vz], quat: [x,y,z,w] or null for random }
const STRUCTURED_SCENARIOS = [
  // Pure lateral: target directly right
  { target: [30, 0, 0], vel: [0, 0, 0], quat: [0, 0, 0, 1] },
  // Pure lateral: target above+right
  { target: [20, 20, 0], vel: [0, 0, 0], quat: [0, 0, 0, 1] },
  // Behind with lateral velocity
  { target: [0, 0, -20], vel: [0.15, 0, 0], quat: [0, 0, 0, 1] },
  // Close target (needs precision, not overshoot)
  { target: [3, 3, 3], vel: [0, 0, 0], quat: [0, 0, 0, 1] },
  // Moving fast laterally, target ahead
  { target: [0, 0, 30], vel: [0.3, 0, 0], quat: [0, 0, 0, 1] },
  // Pure up, starting with downward velocity
  { target: [0, 25, 0], vel: [0, -0.1, 0], quat: [0, 0, 0, 1] },
  // Diagonal far, random orientation
  { target: [25, 25, 25], vel: [0, 0, 0], quat: null },
  // Close behind, high speed toward it
  { target: [0, 0, -10], vel: [0, 0, -0.2], quat: null },
];

function runFollowEpisode(brain, scenario) {
  const ship = createShipState("alpha");
  ship.fuel = PHYSICS.MAX_FUEL;

  let targetPos;
  if (scenario) {
    // Structured scenario
    if (scenario.quat) {
      ship.quat[0] = scenario.quat[0];
      ship.quat[1] = scenario.quat[1];
      ship.quat[2] = scenario.quat[2];
      ship.quat[3] = scenario.quat[3];
    } else {
      Q.setRandomMut(ship.quat);
    }
    ship.vel[0] = scenario.vel[0];
    ship.vel[1] = scenario.vel[1];
    ship.vel[2] = scenario.vel[2];
    targetPos = [scenario.target[0], scenario.target[1], scenario.target[2]];
  } else {
    // Random scenario
    Q.setRandomMut(ship.quat);
    const initSpeed = Math.random() * 0.5 * PHYSICS.MAX_SPEED;
    const vDir = V3.normalize([
      Math.random() - 0.5,
      Math.random() - 0.5,
      Math.random() - 0.5,
    ]);
    ship.vel[0] = vDir[0] * initSpeed;
    ship.vel[1] = vDir[1] * initSpeed;
    ship.vel[2] = vDir[2] * initSpeed;

    // Target at random distance 3-50 units, any direction
    const tDir = V3.normalize([
      Math.random() - 0.5,
      Math.random() - 0.5,
      Math.random() - 0.5,
    ]);
    const tDist = 3 + Math.random() * 47;
    targetPos = [tDir[0] * tDist, tDir[1] * tDist, tDir[2] * tDist];
  }

  let fitness = 0;
  const initDist = V3.distanceTo(ship.pos, targetPos);

  for (let frame = 0; frame < MATCH_FRAMES; frame++) {
    // Build inputs: local target pos (3) + local velocity (3)
    const rx = targetPos[0] - ship.pos[0];
    const ry = targetPos[1] - ship.pos[1];
    const rz = targetPos[2] - ship.pos[2];
    Q.invertInto(ship.quat, _invQ);
    Q.applyToVec3Into(_invQ, rx, ry, rz, _localPos);
    Q.applyToVec3Into(_invQ, ship.vel[0], ship.vel[1], ship.vel[2], _localVel);

    const inputs = [
      _localPos[0] / 50.0,
      _localPos[1] / 50.0,
      _localPos[2] / 50.0,
      _localVel[0] / PHYSICS.MAX_SPEED,
      _localVel[1] / PHYSICS.MAX_SPEED,
      _localVel[2] / PHYSICS.MAX_SPEED,
    ];

    const outputs = brain.forward(inputs);
    const fullOutputs = [outputs[0], outputs[1], outputs[2], 0, 0];
    applyNNOutputs(ship, fullOutputs, [ship]);
    shipSimStep(ship);

    const dist = V3.distanceTo(ship.pos, targetPos);
    const speed = V3.length(ship.vel);

    // Reward being close (exponential, broader scale for far targets)
    fitness += Math.exp(-dist / 10.0);

    // Bonus: reward being close AND slow (stopped at target)
    if (dist < 5.0) {
      fitness += Math.exp(-speed / 0.05) * 2.0;
    }

    // Penalize spinning (angular velocity waste)
    const angSpeed = V3.length(ship.angVel);
    fitness -= angSpeed * 0.5;
  }

  // Final state bonus: big reward for ending close and slow
  const finalDist = V3.distanceTo(ship.pos, targetPos);
  const finalSpeed = V3.length(ship.vel);
  fitness += Math.exp(-finalDist / 2.0) * 50;
  fitness += Math.exp(-finalSpeed / 0.02) * 30;

  // Progress bonus: reward reducing distance from initial
  const distReduction = Math.max(0, initDist - finalDist);
  fitness += distReduction * 0.5;

  // Fuel conservation bonus (small)
  fitness += (ship.fuel / ship.maxFuel) * 5;

  return fitness;
}

function evaluate(genome) {
  const brain = new ReLUNetwork(TOPOLOGY);
  brain.fromGenome(genome);
  let total = 0;

  // Run structured scenarios first
  for (let i = 0; i < STRUCTURED_SCENARIOS.length; i++) {
    total += runFollowEpisode(brain, STRUCTURED_SCENARIOS[i]);
  }

  // Fill remaining evals with random scenarios
  const nRandom = EVALS_PER_CANDIDATE - STRUCTURED_SCENARIOS.length;
  for (let i = 0; i < nRandom; i++) {
    total += runFollowEpisode(brain, null);
  }

  return total / EVALS_PER_CANDIDATE;
}

// ══════════════════════════════════════════════════════════════
// ██  sep-CMA-ES Training Loop
// ══════════════════════════════════════════════════════════════
console.log("sep-CMA-ES Follow Trainer");
console.log(`Topology: ${TOPOLOGY.join("→")}, params: ${N}`);
console.log(`Lambda: ${LAMBDA}, mu: ${MU}, mueff: ${mueff.toFixed(1)}`);
console.log(
  `Evals/candidate: ${EVALS_PER_CANDIDATE} (${STRUCTURED_SCENARIOS.length} structured + ${
    EVALS_PER_CANDIDATE - STRUCTURED_SCENARIOS.length
  } random)`,
);
console.log(`Running for ${RUN_MINUTES} minutes...\n`);

const t0 = performance.now();
const timeLimitMs = RUN_MINUTES * 60 * 1000;
let gen = 0;
let globalBestFitness = -Infinity;
let globalBestGenome = null;

while (performance.now() - t0 < timeLimitMs) {
  gen++;
  const genStart = performance.now();

  // Sample lambda candidates
  const candidates = [];
  for (let k = 0; k < LAMBDA; k++) {
    const { x, y } = sampleCandidate();
    const fitness = evaluate(x);
    candidates.push({ x, y, fitness });
  }

  // Sort by fitness (descending — best first)
  candidates.sort((a, b) => b.fitness - a.fitness);

  const topFitness = candidates[0].fitness;
  const avgFitness = candidates.reduce((s, c) => s + c.fitness, 0) / LAMBDA;

  if (topFitness > globalBestFitness) {
    globalBestFitness = topFitness;
    globalBestGenome = Array.from(candidates[0].x);
  }

  // ── Update mean ──
  const oldMean = new Float64Array(mean);
  mean.fill(0);
  for (let i = 0; i < MU; i++) {
    for (let j = 0; j < N; j++) {
      mean[j] += weights[i] * candidates[i].x[j];
    }
  }

  // ── Update evolution paths (diagonal sep-CMA-ES) ──
  const meanDiff = new Float64Array(N);
  for (let j = 0; j < N; j++) meanDiff[j] = (mean[j] - oldMean[j]) / sigma;

  // C^(-1/2) * meanDiff — diagonal: just divide by sqrt(diagC)
  const invsqrtCMd = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    invsqrtCMd[i] = meanDiff[i] / Math.sqrt(diagC[i]);
  }

  // ps update
  for (let i = 0; i < N; i++) {
    ps[i] = (1 - cs) * ps[i] + Math.sqrt(cs * (2 - cs) * mueff) * invsqrtCMd[i];
  }

  // Determine hsig
  let psNormSq = 0;
  for (let i = 0; i < N; i++) psNormSq += ps[i] * ps[i];
  const psNorm = Math.sqrt(psNormSq);
  const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN <
      1.4 + 2 / (N + 1)
    ? 1
    : 0;

  // pc update
  for (let i = 0; i < N; i++) {
    pc[i] = (1 - cc) * pc[i] +
      hsig * Math.sqrt(cc * (2 - cc) * mueff) * meanDiff[i];
  }

  // ── Update diagonal covariance ──
  const oldFactor = 1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc);
  for (let i = 0; i < N; i++) {
    let val = oldFactor * diagC[i];
    // Rank-1 update
    val += c1 * pc[i] * pc[i];
    // Rank-mu update
    for (let k = 0; k < MU; k++) {
      val += cmu * weights[k] * candidates[k].y[i] * candidates[k].y[i];
    }
    diagC[i] = Math.max(1e-20, val); // clamp positive
  }

  // ── Update sigma (step size) ──
  sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1));
  sigma = Math.max(1e-10, Math.min(sigma, 5.0));

  // ── Logging ──
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
        `σ: ${sigma.toFixed(4)} | ` +
        `${genMs}ms | ${elapsed}s/${remain}s`,
    );
  }

  // ── Checkpoint ──
  if (gen % 50 === 0 && globalBestGenome) {
    saveCheckpoint(gen);
  }
}

function saveCheckpoint(genNum) {
  const checkpoint = {
    topology: TOPOLOGY,
    activation: "relu+tanh",
    fitness: globalBestFitness,
    genome: globalBestGenome,
    generation: genNum,
    scenario: "follow-cmaes",
    optimizer: {
      name: "sep-CMA-ES",
      lambda: LAMBDA,
      mu: MU,
      sigma,
      N,
    },
    config: {
      MATCH_FRAMES,
      EVALS_PER_CANDIDATE,
    },
    trainedAt: new Date().toISOString(),
  };
  Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(checkpoint, null, 2));
}

// Final save
if (globalBestGenome) saveCheckpoint(gen);

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: ${gen} generations in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
