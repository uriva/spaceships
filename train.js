#!/usr/bin/env -S deno run --allow-read --allow-write
// train.js — focused 2v2 asteroid combat trainer
// Scenario: 2v2 in an asteroid field, ~60-unit arena
// Penalties: wandering off arena, running out of fuel
// Seeds from best-genome.json if available
// Run: deno run --allow-read --allow-write train.js [output.json]

const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const {
  V3, Q, PHYSICS, NeuralNetwork,
  createShipState, createAsteroid, checkAsteroidCollisions,
  shipSimStep, buildNNInputs, applyNNOutputs,
  scoreMatch, runMatch, initPopulation, evolve,
} = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [78, 16, 16, 6];
const POP_SIZE = 60;
const OPPONENTS_PER_EVAL = 3;
const MUTATION_RATE = 0.12;
const MUTATION_STRENGTH = 0.30;
const OUTPUT_FILE = Deno.args[0] || 'best-genome.json';
const ARENA_RADIUS = 80;          // ships penalized beyond this
const MATCH_FRAMES = 1500;        // ~25 seconds at 60fps
const FLEET_SIZE = 2;
const SEPARATION = 50;
const SPREAD = 15;
const RUN_MINUTES = 10;

// ── Seed from existing genome ──
function trySeedGenome() {
  try {
    const raw = Deno.readTextFileSync(new URL('./best-genome.json', import.meta.url).pathname);
    const data = JSON.parse(raw);
    if (data.genome && data.genome.length > 0) {
      console.log(`Seeding from best-genome.json (fitness: ${data.fitness?.toFixed(1) ?? '?'})`);
      return new Float64Array(data.genome);
    }
  } catch { /* no file or bad parse */ }
  return null;
}

// ── Arena asteroids (fixed per generation, regenerated each gen) ──
function generateArenaAsteroids(count = 12, radius = 60) {
  const asteroids = [];
  for (let i = 0; i < count; i++) {
    const dir = V3.normalize([Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5]);
    const pos = V3.scale(dir, radius * (0.2 + Math.random() * 0.8));
    const r = 4 + Math.random() * 12;
    asteroids.push(createAsteroid(pos, r));
  }
  return asteroids;
}

// ══════════════════════════════════════════════════════════════
// ██  Enhanced scoring: base scoreMatch + wandering/fuel penalties
// ══════════════════════════════════════════════════════════════
function scoreWithPenalties(allShips, matchFrames) {
  const base = scoreMatch(allShips, matchFrames);

  // Apply per-ship penalties to each team
  for (const teamKey of ['alpha', 'omega']) {
    const ships = allShips.filter(s => s.team === teamKey);
    let penalty = 0;

    for (const s of ships) {
      // ── Wandering penalty: per-frame accumulated distance from center ──
      penalty += (s._wanderPenalty || 0) * 2;

      // ── Fuel depletion penalty: punish ships that burn all fuel ──
      const fuelRatio = s.fuel / s.maxFuel;
      if (fuelRatio < 0.05) {
        penalty += 300;   // harsh: completely out of fuel
      } else if (fuelRatio < 0.2) {
        penalty += 100;   // moderate: very low fuel
      }

      // ── Idle penalty: ship never fired ──
      if ((s._shotsFired || 0) === 0) {
        penalty += 200;
      }
    }

    base[teamKey] -= penalty;
  }

  return base;
}

// ══════════════════════════════════════════════════════════════
// ██  Run match with wandering tracking
// ══════════════════════════════════════════════════════════════
function runMatchWithTracking(alphaBrain, omegaBrain, asteroids) {
  const allShips = [];
  for (let teamIdx = 0; teamIdx < 2; teamIdx++) {
    const team = teamIdx === 0 ? 'alpha' : 'omega';
    const brain = teamIdx === 0 ? alphaBrain : omegaBrain;
    const zSign = teamIdx === 0 ? 1 : -1;
    for (let i = 0; i < FLEET_SIZE; i++) {
      const s = createShipState(team);
      s.pos[0] = (Math.random() - 0.5) * SPREAD;
      s.pos[1] = (Math.random() - 0.5) * SPREAD;
      s.pos[2] = zSign * SEPARATION + (Math.random() - 0.5) * SPREAD;
      Q.setRandomMut(s.quat);
      s._brain = brain;
      s._framesInRange = 0;
      s._distanceClosed = 0;
      s._shotsFired = 0;
      s._prevMinDist = Infinity;
      s._wanderPenalty = 0;
      allShips.push(s);
    }
  }

  let frameCount = 0;
  for (let frame = 0; frame < MATCH_FRAMES; frame++) {
    frameCount++;
    let alphaAlive = false, omegaAlive = false;
    for (const s of allShips) {
      if (!s.alive) continue;
      if (s.team === 'alpha') alphaAlive = true;
      else omegaAlive = true;
      if (alphaAlive && omegaAlive) break;
    }
    if (!alphaAlive || !omegaAlive) break;

    for (const s of allShips) {
      if (!s.alive) continue;
      const inputs = buildNNInputs(s, allShips, asteroids);
      const outputs = s._brain.forward(inputs);
      const result = applyNNOutputs(s, outputs, allShips);
      shipSimStep(s);
      if (asteroids.length > 0) checkAsteroidCollisions(s, asteroids);

      // Track per-frame metrics
      if (result.firedAt) s._shotsFired++;
      let minDistToEnemy = Infinity;
      for (const e of allShips) {
        if (e === s || !e.alive || e.team === s.team) continue;
        const d = V3.distanceTo(s.pos, e.pos);
        if (d < minDistToEnemy) minDistToEnemy = d;
      }
      if (minDistToEnemy < PHYSICS.WEAPON_RANGE) s._framesInRange++;
      if (s._prevMinDist < Infinity && minDistToEnemy < Infinity) {
        const closed = s._prevMinDist - minDistToEnemy;
        if (closed > 0) s._distanceClosed += closed;
      }
      s._prevMinDist = minDistToEnemy;

      // Track wandering from arena center
      const distFromCenter = V3.length(s.pos);
      if (distFromCenter > ARENA_RADIUS) {
        s._wanderPenalty += (distFromCenter - ARENA_RADIUS) * 0.1;
      }
    }
  }

  allShips._matchFrames = frameCount;
  return allShips;
}

// ══════════════════════════════════════════════════════════════
// ██  Evaluate one genome adversarially
// ══════════════════════════════════════════════════════════════
function evaluate(genome, population) {
  const brain = new NeuralNetwork(TOPOLOGY);
  brain.fromGenome(genome);
  let totalFitness = 0;

  for (let opp = 0; opp < OPPONENTS_PER_EVAL; opp++) {
    const oppIdx = Math.floor(Math.random() * population.length);
    const oppBrain = new NeuralNetwork(TOPOLOGY);
    oppBrain.fromGenome(population[oppIdx].genome);

    const asteroids = generateArenaAsteroids(12, 60);
    const ships = runMatchWithTracking(brain, oppBrain, asteroids);
    const scores = scoreWithPenalties(ships, ships._matchFrames);
    totalFitness += scores.alpha;
  }

  return totalFitness / OPPONENTS_PER_EVAL;
}

// ══════════════════════════════════════════════════════════════
// ██  Training loop — time-based (runs for RUN_MINUTES)
// ══════════════════════════════════════════════════════════════
let population = initPopulation(POP_SIZE, TOPOLOGY);

// Seed best genome into first slot
const seedGenome = trySeedGenome();
if (seedGenome && seedGenome.length === population[0].genome.length) {
  population[0].genome = new Float64Array(seedGenome);
  // Also seed a few mutated copies
  for (let i = 1; i < Math.min(5, POP_SIZE); i++) {
    population[i].genome = new Float64Array(seedGenome);
    for (let j = 0; j < population[i].genome.length; j++) {
      if (Math.random() < MUTATION_RATE) {
        population[i].genome[j] += (Math.random() * 2 - 1) * MUTATION_STRENGTH;
      }
    }
  }
  console.log('Seeded 5 individuals from best-genome.json');
}

let globalBestFitness = -Infinity;
let globalBestGenome = null;

console.log(`2v2 asteroid combat trainer`);
console.log(`Pop: ${POP_SIZE}, opponents/eval: ${OPPONENTS_PER_EVAL}, arena: ${ARENA_RADIUS}u radius`);
console.log(`Topology: ${TOPOLOGY.join('→')}, params: ${new NeuralNetwork(TOPOLOGY).paramCount}`);
console.log(`Running for ${RUN_MINUTES} minutes...`);

const t0 = performance.now();
const timeLimitMs = RUN_MINUTES * 60 * 1000;
let gen = 0;

while (performance.now() - t0 < timeLimitMs) {
  const genStart = performance.now();
  gen++;

  // Evaluate every genome
  for (const p of population) {
    p.fitness = evaluate(p.genome, population);
  }

  // Track best
  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
  const topFitness = sorted[0].fitness;
  const avgFitness = sorted.reduce((s, p) => s + p.fitness, 0) / POP_SIZE;

  if (topFitness > globalBestFitness) {
    globalBestFitness = topFitness;
    globalBestGenome = Array.from(sorted[0].genome);
  }

  // Evolve
  population = evolve(population, {
    popSize: POP_SIZE,
    mutationRate: MUTATION_RATE,
    mutationStrength: MUTATION_STRENGTH,
  });

  const genMs = (performance.now() - genStart).toFixed(0);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(0);
  const remain = Math.max(0, (timeLimitMs - (performance.now() - t0)) / 1000).toFixed(0);

  if (gen % 5 === 0 || gen <= 3) {
    console.log(
      `GEN ${String(gen).padStart(4)} | ` +
      `top: ${topFitness.toFixed(1).padStart(8)} | ` +
      `avg: ${avgFitness.toFixed(1).padStart(8)} | ` +
      `best: ${globalBestFitness.toFixed(1).padStart(8)} | ` +
      `${genMs}ms | ${elapsed}s/${remain}s left`
    );
  }

  // Save checkpoint every 20 gens
  if (gen % 20 === 0 && globalBestGenome) {
    const checkpoint = {
      topology: TOPOLOGY,
      fitness: globalBestFitness,
      genome: globalBestGenome,
      generation: gen,
      scenario: '2v2-asteroids',
      config: { POP_SIZE, OPPONENTS_PER_EVAL, MUTATION_RATE, MUTATION_STRENGTH, ARENA_RADIUS, MATCH_FRAMES },
      trainedAt: new Date().toISOString(),
    };
    Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(checkpoint, null, 2));
  }
}

// Final save
if (globalBestGenome) {
  const final = {
    topology: TOPOLOGY,
    fitness: globalBestFitness,
    genome: globalBestGenome,
    generation: gen,
    scenario: '2v2-asteroids',
    config: { POP_SIZE, OPPONENTS_PER_EVAL, MUTATION_RATE, MUTATION_STRENGTH, ARENA_RADIUS, MATCH_FRAMES },
    trainedAt: new Date().toISOString(),
  };
  Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(final, null, 2));
}

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: ${gen} generations in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
