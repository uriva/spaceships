#!/usr/bin/env -S deno run --allow-read --allow-write
// train.js — Curriculum-based neuroevolution trainer
// Phases: navigate → asteroid field → shoot 1 → shoot 2
// Run: deno run --allow-read --allow-write train.js [output.json]

const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const {
  V3, Q, PHYSICS, NeuralNetwork,
  createShipState, createAsteroid, checkAsteroidCollisions,
  shipSimStep, buildNNInputs, applyNNOutputs,
  initPopulation, evolve,
} = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [78, 16, 16, 6];
const POP_SIZE = 20;
const EVALS_PER_GENOME = 3;
const MUTATION_RATE = 0.1;
const MUTATION_STRENGTH = 0.3;
const OUTPUT_FILE = Deno.args[0] || 'best-genome.json';

// ── Helpers ──
function randomDir() {
  return V3.normalize([Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5]);
}

function runSim(ship, allShips, asteroids, timeLimit) {
  for (let f = 0; f < timeLimit; f++) {
    if (!ship.alive) break;
    // Early exit if all enemies dead
    const enemiesAlive = allShips.some(s => s !== ship && s.alive && s.team !== ship.team);
    if (!enemiesAlive) break;
    const inputs = buildNNInputs(ship, allShips, asteroids);
    const outputs = ship._brain.forward(inputs);
    applyNNOutputs(ship, outputs, allShips);
    shipSimStep(ship);
    checkAsteroidCollisions(ship, asteroids);
  }
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 1: Navigate to a point (invulnerable target)
// ══════════════════════════════════════════════════════════════
function phase1(brain) {
  const ship = createShipState('alpha');
  ship.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
  );
  Q.setRandomMut(ship.quat);
  ship._brain = brain;

  // Invulnerable target 60-100 units away
  const target = createShipState('omega');
  const dist = 60 + Math.random() * 40;
  target.pos = V3.add(ship.pos, V3.scale(randomDir(), dist));
  target.hp = 1e6;
  target.maxHp = 1e6;

  const allShips = [ship, target];
  const startDist = V3.distanceTo(ship.pos, target.pos);
  let minDist = startDist;

  const TIME_LIMIT = 600;
  for (let f = 0; f < TIME_LIMIT; f++) {
    if (!ship.alive) break;
    const inputs = buildNNInputs(ship, allShips, []);
    const outputs = brain.forward(inputs);
    applyNNOutputs(ship, outputs, allShips);
    shipSimStep(ship);
    minDist = Math.min(minDist, V3.distanceTo(ship.pos, target.pos));
  }

  const endDist = V3.distanceTo(ship.pos, target.pos);
  const endSpeed = V3.length(ship.vel);

  let fitness = 0;
  fitness += (startDist - endDist) * 5;   // reward closing distance
  fitness -= endDist;                      // penalize remaining distance
  fitness -= endSpeed * 100;               // penalize speed (want to stop near target)
  if (minDist < 30) fitness += 200;
  if (minDist < 15) fitness += 300;
  if (minDist < 5) fitness += 500;
  return fitness;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 2: Navigate through asteroid field
// ══════════════════════════════════════════════════════════════
function generateAsteroidField() {
  // ~30 asteroids in a band from z=30 to z=120, radius 5-20
  const asteroids = [];
  for (let i = 0; i < 30; i++) {
    const pos = [
      (Math.random() - 0.5) * 120,     // wide x spread
      (Math.random() - 0.5) * 120,     // wide y spread
      30 + Math.random() * 90,          // z between 30-120
    ];
    const radius = 5 + Math.random() * 15;
    asteroids.push(createAsteroid(pos, radius));
  }
  return asteroids;
}

function phase2(brain) {
  const ship = createShipState('alpha');
  ship.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    0,
  );
  Q.setRandomMut(ship.quat);
  ship._brain = brain;

  // Invulnerable target on far side of asteroid field
  const target = createShipState('omega');
  target.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    160,  // beyond the field (field ends at z=120)
  );
  target.hp = 1e6;
  target.maxHp = 1e6;

  const asteroids = generateAsteroidField();
  const allShips = [ship, target];
  const startDist = V3.distanceTo(ship.pos, target.pos);
  let minDist = startDist;

  const TIME_LIMIT = 900;
  for (let f = 0; f < TIME_LIMIT; f++) {
    if (!ship.alive) break;
    const inputs = buildNNInputs(ship, allShips, asteroids);
    const outputs = brain.forward(inputs);
    applyNNOutputs(ship, outputs, allShips);
    shipSimStep(ship);
    checkAsteroidCollisions(ship, asteroids);
    minDist = Math.min(minDist, V3.distanceTo(ship.pos, target.pos));
  }

  if (!ship.alive) {
    // Died to asteroid — penalize heavily but still reward progress made
    const distMade = startDist - minDist;
    return distMade * 2 - 500;
  }

  const endDist = V3.distanceTo(ship.pos, target.pos);
  const endSpeed = V3.length(ship.vel);

  let fitness = 0;
  fitness += (startDist - endDist) * 5;
  fitness -= endDist;
  fitness -= endSpeed * 100;
  if (minDist < 30) fitness += 200;
  if (minDist < 15) fitness += 300;
  if (minDist < 5) fitness += 500;
  // Survival bonus — made it through the field alive
  fitness += 200;
  return fitness;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 3: Destroy 1 stationary enemy
// ══════════════════════════════════════════════════════════════
function phase3(brain) {
  const ship = createShipState('alpha');
  ship.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
  );
  Q.setRandomMut(ship.quat);
  ship._brain = brain;

  // Stationary enemy 100-200 units away
  const target = createShipState('omega');
  const dist = 100 + Math.random() * 100;
  target.pos = V3.add(ship.pos, V3.scale(randomDir(), dist));

  const allShips = [ship, target];
  const startDist = V3.distanceTo(ship.pos, target.pos);

  runSim(ship, allShips, [], 1200);

  const endDist = V3.distanceTo(ship.pos, target.pos);

  let fitness = 0;
  fitness += ship.neuralDamageDealt * 5;           // reward damage
  fitness += !target.alive ? 1000 : 0;             // kill bonus
  fitness += (startDist - endDist) * 2;            // approach reward
  fitness -= Math.min(endDist, 300) * 0.5;         // distance penalty (capped)
  return fitness;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 4: Destroy 2 stationary enemies
// ══════════════════════════════════════════════════════════════
function phase4(brain) {
  const ship = createShipState('alpha');
  ship.pos = V3.create(0, 0, 0);
  Q.setRandomMut(ship.quat);
  ship._brain = brain;

  // 2 enemies at different random directions, 80-140 units away
  const t1 = createShipState('omega');
  t1.pos = V3.add(ship.pos, V3.scale(randomDir(), 80 + Math.random() * 60));

  const t2 = createShipState('omega');
  t2.pos = V3.add(ship.pos, V3.scale(randomDir(), 80 + Math.random() * 60));

  const allShips = [ship, t1, t2];

  runSim(ship, allShips, [], 1800);

  const kills = (!t1.alive ? 1 : 0) + (!t2.alive ? 1 : 0);

  let fitness = 0;
  fitness += ship.neuralDamageDealt * 5;           // damage reward
  fitness += kills * 1000;                          // per-kill bonus
  if (kills === 2) fitness += 500;                  // both-killed bonus

  // Proximity incentive for remaining alive targets
  const alive = [t1, t2].filter(t => t.alive);
  if (alive.length > 0) {
    const avgDist = alive.reduce((s, t) => s + V3.distanceTo(ship.pos, t.pos), 0) / alive.length;
    fitness -= Math.min(avgDist, 300) * 0.3;
  }
  return fitness;
}

// ══════════════════════════════════════════════════════════════
// ██  Training loop
// ══════════════════════════════════════════════════════════════
const PHASES = [
  { name: 'Navigate to target',         gens: 200, run: phase1 },
  { name: 'Navigate through asteroids', gens: 200, run: phase2 },
  { name: 'Shoot 1 stationary',         gens: 100, run: phase3 },
  { name: 'Shoot 2 stationary',         gens: 100, run: phase4 },
];

let population = initPopulation(POP_SIZE, TOPOLOGY);
let globalBestFitness = -Infinity;
let globalBestGenome = null;

console.log(`Curriculum trainer: pop=${POP_SIZE}, evals/genome=${EVALS_PER_GENOME}`);
console.log(`Topology: ${TOPOLOGY.join('→')}, params=${new NeuralNetwork(TOPOLOGY).paramCount}`);
const t0 = performance.now();

for (const phase of PHASES) {
  console.log(`\n${'═'.repeat(50)}`);
  console.log(`  PHASE: ${phase.name} (${phase.gens} generations)`);
  console.log(`${'═'.repeat(50)}`);

  // Reset fitness for new phase (keep genomes)
  for (const p of population) { p.fitness = 0; p.fights = 0; }

  let phaseBest = -Infinity;

  for (let gen = 0; gen < phase.gens; gen++) {
    const genStart = performance.now();

    // Evaluate every genome
    for (const p of population) {
      const brain = new NeuralNetwork(TOPOLOGY);
      brain.fromGenome(p.genome);
      let total = 0;
      for (let e = 0; e < EVALS_PER_GENOME; e++) {
        total += phase.run(brain);
      }
      p.fitness = total / EVALS_PER_GENOME;
      p.fights = EVALS_PER_GENOME;
    }

    // Track best
    for (const p of population) {
      if (p.fitness > phaseBest) phaseBest = p.fitness;
      if (p.fitness > globalBestFitness) {
        globalBestFitness = p.fitness;
        globalBestGenome = Array.from(p.genome);
      }
    }

    // Evolve
    const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
    const topFitness = sorted[0].fitness;
    const avgFitness = sorted.reduce((s, p) => s + p.fitness, 0) / POP_SIZE;

    population = evolve(population, {
      popSize: POP_SIZE,
      mutationRate: MUTATION_RATE,
      mutationStrength: MUTATION_STRENGTH,
    });

    const genMs = (performance.now() - genStart).toFixed(0);
    console.log(
      `  GEN ${String(gen + 1).padStart(4)} | ` +
      `top: ${topFitness.toFixed(1).padStart(8)} | ` +
      `avg: ${avgFitness.toFixed(1).padStart(8)} | ` +
      `${genMs}ms`
    );
  }

  // Save after each phase
  const phaseOutput = {
    topology: TOPOLOGY,
    fitness: globalBestFitness,
    genome: globalBestGenome,
    phase: phase.name,
    config: { POP_SIZE, EVALS_PER_GENOME, MUTATION_RATE, MUTATION_STRENGTH },
    trainedAt: new Date().toISOString(),
  };
  Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(phaseOutput, null, 2));
  console.log(`  → Saved checkpoint (phase best: ${phaseBest.toFixed(1)})`);
}

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: all phases in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
