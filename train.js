#!/usr/bin/env -S deno run --allow-read --allow-write
// train.js — 6-phase curriculum neuroevolution trainer
// Phases: navigate → asteroids → 1v1 → 1v1+asteroids → 5v5 → 5v5+asteroids
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
const POP_SIZE = 40;
const EVALS_PER_GENOME = 3;
const MUTATION_RATE = 0.12;
const MUTATION_STRENGTH = 0.3;
const OUTPUT_FILE = Deno.args[0] || 'best-genome.json';

// ── Helpers ──
function randomDir() {
  return V3.normalize([Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5]);
}

function generateAsteroidField(count = 30, spreadXY = 120, zMin = 30, zMax = 120) {
  const asteroids = [];
  for (let i = 0; i < count; i++) {
    const pos = [
      (Math.random() - 0.5) * spreadXY,
      (Math.random() - 0.5) * spreadXY,
      zMin + Math.random() * (zMax - zMin),
    ];
    const radius = 5 + Math.random() * 15;
    asteroids.push(createAsteroid(pos, radius));
  }
  return asteroids;
}

// Generate a sphere of asteroids around the origin for combat arenas
function generateArenaAsteroids(count = 20, radius = 150) {
  const asteroids = [];
  for (let i = 0; i < count; i++) {
    const pos = V3.scale(randomDir(), radius * (0.3 + Math.random() * 0.7));
    const r = 5 + Math.random() * 20;
    asteroids.push(createAsteroid(pos, r));
  }
  return asteroids;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 1: Navigate to a point (invulnerable target)
// ══════════════════════════════════════════════════════════════
function phaseNavigate(brain) {
  const ship = createShipState('alpha');
  ship.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
  );
  Q.setRandomMut(ship.quat);
  ship._brain = brain;

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
  fitness += (startDist - endDist) * 5;
  fitness -= endDist;
  fitness -= endSpeed * 100;
  if (minDist < 30) fitness += 200;
  if (minDist < 15) fitness += 300;
  if (minDist < 5) fitness += 500;
  return fitness;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 2: Navigate through asteroid field
// ══════════════════════════════════════════════════════════════
function phaseAsteroids(brain) {
  const ship = createShipState('alpha');
  ship.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    0,
  );
  Q.setRandomMut(ship.quat);
  ship._brain = brain;

  const target = createShipState('omega');
  target.pos = V3.create(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    160,
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
  fitness += 200; // survival bonus
  return fitness;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 3: 1v1 combat (both NN-controlled)
// ══════════════════════════════════════════════════════════════
function phase1v1(brain) {
  const ships = runMatch(brain, brain, {
    fleetSize: 1,
    matchTimeLimit: 1200,
    separation: 80,
    spread: 20,
  });
  const scores = scoreMatch(ships);
  // Average both sides (symmetric — same brain controls both)
  return (scores.alpha + scores.omega) / 2;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 4: 1v1 combat in asteroid field
// ══════════════════════════════════════════════════════════════
function phase1v1Asteroids(brain) {
  const asteroids = generateArenaAsteroids(15, 100);

  // Manual 1v1 match with asteroids (runMatch doesn't support asteroids)
  const allShips = [];
  for (let teamIdx = 0; teamIdx < 2; teamIdx++) {
    const team = teamIdx === 0 ? 'alpha' : 'omega';
    const zSign = teamIdx === 0 ? 1 : -1;
    const s = createShipState(team);
    s.pos[0] = (Math.random() - 0.5) * 20;
    s.pos[1] = (Math.random() - 0.5) * 20;
    s.pos[2] = zSign * 80;
    Q.setRandomMut(s.quat);
    s._brain = brain;
    allShips.push(s);
  }

  const TIME_LIMIT = 1200;
  for (let frame = 0; frame < TIME_LIMIT; frame++) {
    let alphaAlive = false, omegaAlive = false;
    for (const s of allShips) {
      if (!s.alive) continue;
      if (s.team === 'alpha') alphaAlive = true;
      else omegaAlive = true;
    }
    if (!alphaAlive || !omegaAlive) break;

    for (const s of allShips) {
      if (!s.alive) continue;
      const inputs = buildNNInputs(s, allShips, asteroids);
      const outputs = s._brain.forward(inputs);
      applyNNOutputs(s, outputs, allShips);
      shipSimStep(s);
      checkAsteroidCollisions(s, asteroids);
    }
  }

  const scores = scoreMatch(allShips);
  // Survival bonus — don't die to asteroids
  let survivalBonus = 0;
  for (const s of allShips) {
    if (s.alive) survivalBonus += 50;
  }
  return (scores.alpha + scores.omega) / 2 + survivalBonus;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 5: 5v5 combat
// ══════════════════════════════════════════════════════════════
function phase5v5(brain) {
  const ships = runMatch(brain, brain, {
    fleetSize: 5,
    matchTimeLimit: 1800,
    separation: 50,
    spread: 30,
  });
  const scores = scoreMatch(ships);
  return (scores.alpha + scores.omega) / 2;
}

// ══════════════════════════════════════════════════════════════
// ██  Phase 6: 5v5 combat in asteroid field
// ══════════════════════════════════════════════════════════════
function phase5v5Asteroids(brain) {
  const asteroids = generateArenaAsteroids(25, 150);

  const allShips = [];
  for (let teamIdx = 0; teamIdx < 2; teamIdx++) {
    const team = teamIdx === 0 ? 'alpha' : 'omega';
    const zSign = teamIdx === 0 ? 1 : -1;
    for (let i = 0; i < 5; i++) {
      const s = createShipState(team);
      s.pos[0] = (Math.random() - 0.5) * 30;
      s.pos[1] = (Math.random() - 0.5) * 30;
      s.pos[2] = zSign * 50 + (Math.random() - 0.5) * 30;
      Q.setRandomMut(s.quat);
      s._brain = brain;
      allShips.push(s);
    }
  }

  const TIME_LIMIT = 1800;
  for (let frame = 0; frame < TIME_LIMIT; frame++) {
    let alphaAlive = false, omegaAlive = false;
    for (const s of allShips) {
      if (!s.alive) continue;
      if (s.team === 'alpha') alphaAlive = true;
      else omegaAlive = true;
    }
    if (!alphaAlive || !omegaAlive) break;

    for (const s of allShips) {
      if (!s.alive) continue;
      const inputs = buildNNInputs(s, allShips, asteroids);
      const outputs = s._brain.forward(inputs);
      applyNNOutputs(s, outputs, allShips);
      shipSimStep(s);
      checkAsteroidCollisions(s, asteroids);
    }
  }

  const scores = scoreMatch(allShips);
  let survivalBonus = 0;
  for (const s of allShips) {
    if (s.alive) survivalBonus += 30;
  }
  return (scores.alpha + scores.omega) / 2 + survivalBonus;
}

// ══════════════════════════════════════════════════════════════
// ██  Training loop
// ══════════════════════════════════════════════════════════════
const PHASES = [
  { name: 'Navigate to target',    gens: 150, run: phaseNavigate },
  { name: 'Navigate asteroids',    gens: 150, run: phaseAsteroids },
  { name: '1v1 combat',            gens: 100, run: phase1v1 },
  { name: '1v1 + asteroids',       gens: 100, run: phase1v1Asteroids },
  { name: '5v5 combat',            gens: 150, run: phase5v5 },
  { name: '5v5 + asteroids',       gens: 150, run: phase5v5Asteroids },
];

let population = initPopulation(POP_SIZE, TOPOLOGY);
let globalBestFitness = -Infinity;
let globalBestGenome = null;

console.log(`Curriculum trainer: pop=${POP_SIZE}, evals/genome=${EVALS_PER_GENOME}`);
console.log(`Topology: ${TOPOLOGY.join('→')}, params=${new NeuralNetwork(TOPOLOGY).paramCount}`);
console.log(`Phases: ${PHASES.map(p => p.name).join(' → ')}`);
const t0 = performance.now();

for (const phase of PHASES) {
  console.log(`\n${'═'.repeat(55)}`);
  console.log(`  PHASE: ${phase.name} (${phase.gens} generations)`);
  console.log(`${'═'.repeat(55)}`);

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
    if (gen % 10 === 0 || gen === phase.gens - 1) {
      console.log(
        `  GEN ${String(gen + 1).padStart(4)} | ` +
        `top: ${topFitness.toFixed(1).padStart(8)} | ` +
        `avg: ${avgFitness.toFixed(1).padStart(8)} | ` +
        `${genMs}ms`
      );
    }
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
  console.log(`  → Saved checkpoint (phase best: ${phaseBest.toFixed(1)}, global best: ${globalBestFitness.toFixed(1)})`);
}

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: all 6 phases in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
