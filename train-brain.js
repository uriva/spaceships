#!/usr/bin/env -S deno run --allow-read --allow-write
// train-brain.js — sep-CMA-ES trainer for unified ship brain (BRAIN.md Phase 1)
// Scenario: follow a moving target (no asteroid deaths — learn navigation first).
// Network: [43, 5] linear + tanh output. 220 params.
// Warm-start: hand-coded weights that approximately face and move toward target.
// Fitness: facing_reward * 50 + closing_distance * 100 (incremental, per-frame).
// Key: all candidates in one generation face IDENTICAL scenarios (seeded PRNG).

const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const {
  V3, Q, PHYSICS, ReLUNetwork,
  createShipState, createAsteroidWithMass,
  shipSimStep, applyGravity, buildBrainInputs, applyBrainOutputs,
} = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [43, 5];
const OUTPUT_FILE = Deno.args[0] || 'brain.json';
const RUN_MINUTES = parseFloat(Deno.args[1] || '60');
const FRESH = Deno.args.includes('--fresh');
const SEED_FILE = FRESH ? '__noseed__' : (Deno.args[2] || 'brain.json');
const EVALS_PER_CANDIDATE = 8;
const EPISODE_FRAMES = 3000; // 50 sec at 60fps

// sep-CMA-ES hyperparameters
const nn = new ReLUNetwork(TOPOLOGY);
const N = nn.paramCount;
const LAMBDA = 200;
const MU = Math.floor(LAMBDA / 2);

console.log(`Param count: ${N}`);

// Recombination weights
const rawWeights = [];
for (let i = 0; i < MU; i++) rawWeights.push(Math.log(MU + 0.5) - Math.log(i + 1));
const wSum = rawWeights.reduce((a, b) => a + b, 0);
const weights = rawWeights.map(w => w / wSum);
const mueff = 1.0 / weights.reduce((a, w) => a + w * w, 0);

const cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
const cs = (mueff + 2) / (N + mueff + 5);
const c1 = 2 / ((N + 1.3) ** 2 + mueff);
const cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff));
const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (N + 1)) - 1) + cs;
const chiN = Math.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N));

// ══════════════════════════════════════════════════════════════
// ██  Seeded PRNG (xoshiro128**)
// ══════════════════════════════════════════════════════════════
function makeRng(seed) {
  // Simple seed expansion from single integer
  let s0 = (seed ^ 0xDEADBEEF) >>> 0;
  let s1 = (seed * 1103515245 + 12345) >>> 0;
  let s2 = (s0 ^ (s1 << 13)) >>> 0;
  let s3 = (s1 ^ (s0 >> 7)) >>> 0;
  if ((s0 | s1 | s2 | s3) === 0) s0 = 1;

  return function () {
    const result = (((s1 * 5) << 7 | (s1 * 5) >>> 25) * 9) >>> 0;
    const t = (s1 << 9) >>> 0;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;
    s2 ^= t;
    s3 = ((s3 << 11) | (s3 >>> 21)) >>> 0;
    return result / 4294967296;
  };
}

// ══════════════════════════════════════════════════════════════
// ██  sep-CMA-ES State
// ══════════════════════════════════════════════════════════════
const mean = new Float64Array(N);
let sigma = 0.2;
const pc = new Float64Array(N);
const ps = new Float64Array(N);
const diagC = new Float64Array(N);
diagC.fill(1.0);

let seeded = false;
try {
  const seedData = JSON.parse(Deno.readTextFileSync(SEED_FILE));
  if (seedData.genome && seedData.genome.length === N &&
      JSON.stringify(seedData.topology) === JSON.stringify(TOPOLOGY)) {
    for (let i = 0; i < N; i++) mean[i] = seedData.genome[i];
    seeded = true;
    sigma = seedData.optimizer?.sigma || 0.3;
    sigma = Math.max(sigma, 0.05);
    console.log(`Seeded from ${SEED_FILE} (fitness ${seedData.fitness?.toFixed(1)}, gen ${seedData.generation}, sigma=${sigma.toFixed(4)})`);
  }
} catch (_) { /* no seed */ }

if (!seeded) {
  // Warm-start: hand-coded weights that approximately follow the target.
  // For [43, 5] topology, weights[0] is 5×43 = 215 weights, then 5 biases = 220 total.
  // Weight layout: output o, input i → index o*43 + i
  //
  // Input layout (from buildBrainInputs):
  //   [0-5]: own state (hull, battery, fuel, speed, vel_az, vel_el)
  //   [6]: target azimuth (local, /π)    — range [-1, 1]
  //   [7]: target elevation (local, /(π/2)) — range [-1, 1]
  //   [8]: target distance (1/(1+d/50)) — range (0, 1], higher=closer
  //   [9-17]: enemies, [18-26]: friends, [27-42]: asteroids
  //
  // Output layout (from applyBrainOutputs):
  //   [0]: desired azimuth (tanh → -π to π)
  //   [1]: desired elevation (tanh → -π/2 to π/2)
  //   [2]: desired speed (tanh → 0 to MAX_SPEED)
  //   [3]: fuel spend rate (tanh → 0 to 1)
  //   [4]: fire (>0 = shoot)
  //
  // Strategy: face the target direction, move toward it at moderate speed.
  mean.fill(0); // all zero baseline

  // Output 0 (azimuth) ← input 6 (target azimuth): direct pass-through
  mean[0 * 43 + 6] = 1.5;  // weight: target_az → desired_az (slightly over 1 for responsiveness)

  // Output 1 (elevation) ← input 7 (target elevation): direct pass-through
  mean[1 * 43 + 7] = 1.5;

  // Output 2 (desired speed): want speed when target is far (input 8 low), less when close
  // speed = tanh(bias - weight * closeness) → when far (closeness→0): tanh(bias) → moderate speed
  // when very close (closeness→1): tanh(bias - weight) → low speed
  const biasIdx = 43 * 5; // biases start after all weights
  mean[2 * 43 + 8] = -1.5;   // negative weight on closeness (closer = slower)
  mean[biasIdx + 2] = 1.0;   // positive bias (default: want to move)

  // Output 3 (fuel spend): moderate positive bias — willing to burn fuel
  mean[biasIdx + 3] = 0.5;

  // Output 4 (fire): negative bias — don't fire during follow
  mean[biasIdx + 4] = -1.0;

  console.log('No seed found, using warm-start initialization');
}

// ══════════════════════════════════════════════════════════════
// ██  Sampling
// ══════════════════════════════════════════════════════════════
function sampleCandidate() {
  const z = new Float64Array(N);
  const y = new Float64Array(N);
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const u1 = Math.random(), u2 = Math.random();
    z[i] = Math.sqrt(-2 * Math.log(u1 + 1e-30)) * Math.cos(2 * Math.PI * u2);
    y[i] = Math.sqrt(diagC[i]) * z[i];
    x[i] = mean[i] + sigma * y[i];
  }
  return { x, y, z };
}

// ══════════════════════════════════════════════════════════════
// ██  Scenario generation (deterministic from seed)
// ══════════════════════════════════════════════════════════════

function rngDir(rng) {
  const v = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (l < 1e-6) return [1, 0, 0];
  return [v[0] / l, v[1] / l, v[2] / l];
}

function generateAsteroidsDet(count, rng) {
  const asteroids = [];
  for (let i = 0; i < count; i++) {
    const dir = rngDir(rng);
    const dist = 20 + rng() * 200;
    const pos = [dir[0] * dist, dir[1] * dist, dir[2] * dist];
    const radius = 2 + rng() * 38;
    asteroids.push(createAsteroidWithMass(pos, radius));
  }
  return asteroids;
}

// Pre-compute target trajectory for an episode so all candidates see identical target movement
function generateTargetTrajectory(startDist, rng) {
  const dir = rngDir(rng);
  const startPos = [dir[0] * startDist, dir[1] * startDist, dir[2] * startDist];
  const speed = 0.01 + rng() * 0.09;
  let velDir = rngDir(rng);
  let vel = [velDir[0] * speed, velDir[1] * speed, velDir[2] * speed];
  let framesUntilTurn = 200 + Math.floor(rng() * 400);

  // Pre-compute all positions
  const positions = new Float64Array(EPISODE_FRAMES * 3);
  const pos = [startPos[0], startPos[1], startPos[2]];
  for (let f = 0; f < EPISODE_FRAMES; f++) {
    positions[f * 3] = pos[0];
    positions[f * 3 + 1] = pos[1];
    positions[f * 3 + 2] = pos[2];
    pos[0] += vel[0]; pos[1] += vel[1]; pos[2] += vel[2];
    framesUntilTurn--;
    if (framesUntilTurn <= 0) {
      velDir = rngDir(rng);
      vel = [velDir[0] * speed, velDir[1] * speed, velDir[2] * speed];
      framesUntilTurn = 200 + Math.floor(rng() * 400);
    }
  }
  return positions;
}

// Generate 8 episode configs (deterministic from master seed)
function generateEpisodeConfigs(masterSeed) {
  const rng = makeRng(masterSeed);
  const configs = [];

  // 4 short-range (target 10-50 units)
  for (let i = 0; i < 4; i++) {
    const targetDist = 10 + rng() * 40;
    const fuel = 2500;
    const asteroidCount = 2 + Math.floor(rng() * 4);
    const asteroids = generateAsteroidsDet(asteroidCount, rng);
    const targetPositions = generateTargetTrajectory(targetDist, rng);
    const shipQuat = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5];
    const ql = Math.sqrt(shipQuat[0] ** 2 + shipQuat[1] ** 2 + shipQuat[2] ** 2 + shipQuat[3] ** 2);
    for (let j = 0; j < 4; j++) shipQuat[j] /= ql;
    const initSpeed = rng() * 0.3 * PHYSICS.MAX_SPEED;
    const vDir = rngDir(rng);
    const shipVel = [vDir[0] * initSpeed, vDir[1] * initSpeed, vDir[2] * initSpeed];
    configs.push({ fuel, asteroids, targetPositions, shipQuat, shipVel });
  }
  // 4 long-range (target 80-250 units)
  for (let i = 0; i < 4; i++) {
    const targetDist = 80 + rng() * 170;
    const fuel = 2500;
    const asteroidCount = 2 + Math.floor(rng() * 4);
    const asteroids = generateAsteroidsDet(asteroidCount, rng);
    const targetPositions = generateTargetTrajectory(targetDist, rng);
    const shipQuat = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5];
    const ql = Math.sqrt(shipQuat[0] ** 2 + shipQuat[1] ** 2 + shipQuat[2] ** 2 + shipQuat[3] ** 2);
    for (let j = 0; j < 4; j++) shipQuat[j] /= ql;
    const initSpeed = rng() * 0.3 * PHYSICS.MAX_SPEED;
    const vDir = rngDir(rng);
    const shipVel = [vDir[0] * initSpeed, vDir[1] * initSpeed, vDir[2] * initSpeed];
    configs.push({ fuel, asteroids, targetPositions, shipQuat, shipVel });
  }
  return configs;
}

// ══════════════════════════════════════════════════════════════
// ██  Episode runner (deterministic — uses pre-computed config)
// ══════════════════════════════════════════════════════════════
const _targetPos = [0, 0, 0]; // reusable buffer

function runEpisode(brain, config) {
  const ship = createShipState('alpha');
  ship.fuel = config.fuel;
  ship.quat[0] = config.shipQuat[0]; ship.quat[1] = config.shipQuat[1];
  ship.quat[2] = config.shipQuat[2]; ship.quat[3] = config.shipQuat[3];
  ship.vel[0] = config.shipVel[0]; ship.vel[1] = config.shipVel[1]; ship.vel[2] = config.shipVel[2];

  const { asteroids, targetPositions } = config;

  // Fitness components (incremental, per-frame):
  // 1. facing: dot(ship.forward, dirToTarget) — rewards turning toward target [-1, 1]
  // 2. closing: (prevDist - curDist) — rewards reducing distance per frame
  // No asteroid death — learn navigation first, avoidance later.

  let facingSum = 0;
  let closingSum = 0;
  let prevDist = V3.distanceTo(ship.pos, [
    targetPositions[0], targetPositions[1], targetPositions[2]
  ]);

  for (let frame = 0; frame < EPISODE_FRAMES; frame++) {
    _targetPos[0] = targetPositions[frame * 3];
    _targetPos[1] = targetPositions[frame * 3 + 1];
    _targetPos[2] = targetPositions[frame * 3 + 2];

    const inputs = buildBrainInputs(ship, _targetPos, [], [], asteroids);
    const outputs = brain.forward(inputs);
    applyBrainOutputs(ship, outputs, []);
    applyGravity(ship, asteroids, 1);
    shipSimStep(ship);

    const dist = V3.distanceTo(ship.pos, _targetPos);

    // Facing reward: dot(forward, dirToTarget)
    if (dist > 0.01) {
      const fwd = Q.applyToVec3(ship.quat, [0, 0, 1]);
      const dx = _targetPos[0] - ship.pos[0];
      const dy = _targetPos[1] - ship.pos[1];
      const dz = _targetPos[2] - ship.pos[2];
      facingSum += (fwd[0] * dx + fwd[1] * dy + fwd[2] * dz) / dist;
    }

    // Closing reward: positive when getting closer
    closingSum += prevDist - dist;
    prevDist = dist;
  }

  // Normalize per frame, scale to nice numbers
  const facingAvg = facingSum / EPISODE_FRAMES; // range [-1, 1]
  const initDist = V3.distanceTo([0,0,0], [
    targetPositions[0], targetPositions[1], targetPositions[2]
  ]);
  const closingNorm = initDist > 1 ? closingSum / initDist : closingSum;

  // facing: 50 weight — turn toward target (learning step 1)
  // closing: 100 weight — actually get there (learning step 2)
  return facingAvg * 50 + closingNorm * 100;
}

// ══════════════════════════════════════════════════════════════
// ██  Evaluate a candidate (uses pre-generated episode configs)
// ══════════════════════════════════════════════════════════════
function evaluate(genome, episodeConfigs) {
  const brain = new ReLUNetwork(TOPOLOGY);
  brain.fromGenome(genome);
  let total = 0;
  for (let i = 0; i < episodeConfigs.length; i++) {
    total += runEpisode(brain, episodeConfigs[i]);
  }
  return total / episodeConfigs.length;
}

// ══════════════════════════════════════════════════════════════
// ██  Training loop
// ══════════════════════════════════════════════════════════════
console.log('sep-CMA-ES Brain Trainer (Phase 1: follow moving target)');
console.log(`Topology: ${TOPOLOGY.join('->')}, params: ${N}`);
console.log(`Lambda: ${LAMBDA}, mu: ${MU}, mueff: ${mueff.toFixed(1)}`);
console.log(`Episodes/candidate: ${EVALS_PER_CANDIDATE}, frames/episode: ${EPISODE_FRAMES}`);
console.log(`Scenarios: deterministic per generation (seeded PRNG)`);
console.log(`Running for ${RUN_MINUTES} minutes...\n`);

const t0 = performance.now();
const timeLimitMs = RUN_MINUTES * 60 * 1000;
let gen = 0;
let globalBestFitness = -Infinity;
let globalBestGenome = null;
let stagnationCount = 0;
let prevBest = -Infinity;

while (performance.now() - t0 < timeLimitMs) {
  gen++;
  const genStart = performance.now();

  // Generate episode configs for this generation (same for all candidates)
  const episodeConfigs = generateEpisodeConfigs(gen * 1000 + 42);

  // Sample and evaluate
  const candidates = [];
  for (let k = 0; k < LAMBDA; k++) {
    const { x, y } = sampleCandidate();
    const fitness = evaluate(x, episodeConfigs);
    candidates.push({ x, y, fitness });
  }

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

  // ── Evolution paths ──
  const meanDiff = new Float64Array(N);
  for (let j = 0; j < N; j++) meanDiff[j] = (mean[j] - oldMean[j]) / sigma;

  const invsqrtCMd = new Float64Array(N);
  for (let i = 0; i < N; i++) invsqrtCMd[i] = meanDiff[i] / Math.sqrt(diagC[i]);

  for (let i = 0; i < N; i++) {
    ps[i] = (1 - cs) * ps[i] + Math.sqrt(cs * (2 - cs) * mueff) * invsqrtCMd[i];
  }

  let psNormSq = 0;
  for (let i = 0; i < N; i++) psNormSq += ps[i] * ps[i];
  const psNorm = Math.sqrt(psNormSq);
  const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (N + 1) ? 1 : 0;

  for (let i = 0; i < N; i++) {
    pc[i] = (1 - cc) * pc[i] + hsig * Math.sqrt(cc * (2 - cc) * mueff) * meanDiff[i];
  }

  // ── Diagonal covariance update ──
  const oldFactor = 1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc);
  for (let i = 0; i < N; i++) {
    let val = oldFactor * diagC[i];
    val += c1 * pc[i] * pc[i];
    for (let k = 0; k < MU; k++) {
      val += cmu * weights[k] * candidates[k].y[i] * candidates[k].y[i];
    }
    diagC[i] = Math.max(1e-20, val);
  }

  // ── Sigma update ──
  sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1));
  sigma = Math.max(1e-10, Math.min(sigma, 5.0));

  // ── Stagnation check ──
  if (Math.abs(globalBestFitness - prevBest) < 0.01) {
    stagnationCount++;
  } else {
    stagnationCount = 0;
    prevBest = globalBestFitness;
  }

  if (sigma < 0.01 || stagnationCount >= 500) {
    console.log(`\nConverged: sigma=${sigma.toFixed(6)}, stagnation=${stagnationCount}`);
    break;
  }

  // ── Logging ──
  const genMs = (performance.now() - genStart).toFixed(0);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(0);
  const remain = Math.max(0, (timeLimitMs - (performance.now() - t0)) / 1000).toFixed(0);

  if (gen % 5 === 0 || gen <= 3) {
    console.log(
      `GEN ${String(gen).padStart(4)} | ` +
      `top: ${topFitness.toFixed(1).padStart(8)} | ` +
      `avg: ${avgFitness.toFixed(1).padStart(8)} | ` +
      `best: ${globalBestFitness.toFixed(1).padStart(8)} | ` +
      `sigma: ${sigma.toFixed(4)} | ` +
      `${genMs}ms | ${elapsed}s/${remain}s`
    );
  }

  // Checkpoint every 10 gens
  if (gen % 10 === 0 && globalBestGenome) {
    saveCheckpoint(gen);
  }
}

function saveCheckpoint(genNum) {
  const checkpoint = {
    topology: TOPOLOGY,
    activation: 'relu+tanh',
    fitness: globalBestFitness,
    genome: globalBestGenome,
    generation: genNum,
    scenario: 'brain-phase1-follow',
    optimizer: {
      name: 'sep-CMA-ES',
      lambda: LAMBDA,
      mu: MU,
      sigma,
      N,
    },
    config: {
      EPISODE_FRAMES,
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
