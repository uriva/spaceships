#!/usr/bin/env -S deno run --allow-read --allow-write
// train-brain.js — sep-CMA-ES trainer for simplified movement brain
// Goal: navigate to a static point and stop there.
// Network: [6, 3] linear + tanh output. 21 params.
// Inputs: target az/el/dist + speed + vel direction
// Outputs: face az/el + throttle

const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const {
  V3, Q, PHYSICS, ReLUNetwork,
  createShipState, shipSimStep,
  buildSimpleBrainInputs, applySimpleBrainOutputs,
} = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [6, 3];
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
let sigma = 0.3;
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
    sigma = Math.min(Math.max(sigma, 0.05), 0.3);
    console.log(`Seeded from ${SEED_FILE} (fitness ${seedData.fitness?.toFixed(1)}, gen ${seedData.generation}, sigma=${sigma.toFixed(4)})`);
  }
} catch (_) { /* no seed */ }

if (!seeded) {
  // Warm-start: hand-coded weights for "face target, thrust toward it, brake when close"
  // [6, 3] topology: 3×6 = 18 weights, then 3 biases = 21 total
  // Weight index: output_o, input_i → o*6 + i
  // Bias index: 18 + o
  //
  // Inputs:
  //   [0]: target azimuth (local, /π → [-1,1])
  //   [1]: target elevation (local, /(π/2) → [-1,1])
  //   [2]: target distance (min(dist/100, 1) → [0,1])
  //   [3]: speed (/MAX_SPEED → [0,1])
  //   [4]: velocity azimuth (local, /π)
  //   [5]: velocity elevation (local, /(π/2))
  //
  // Outputs:
  //   [0]: desired facing azimuth (×π)
  //   [1]: desired facing elevation (×π/2)
  //   [2]: throttle (-1 to +1)
  mean.fill(0);

  // Face where target is
  mean[0 * 6 + 0] = 1.0;  // target_az → face_az
  mean[1 * 6 + 1] = 1.0;  // target_el → face_el

  // Throttle: thrust when far, brake when fast
  mean[2 * 6 + 2] = 3.0;   // distance → throttle (far = more thrust)
  mean[2 * 6 + 3] = -3.0;  // speed → throttle (fast = brake)

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
// ██  Scenario generation
// ══════════════════════════════════════════════════════════════
function rngDir(rng) {
  const v = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (l < 1e-6) return [1, 0, 0];
  return [v[0] / l, v[1] / l, v[2] / l];
}

function generateEpisodeConfigs(masterSeed) {
  const rng = makeRng(masterSeed);
  const configs = [];

  // Stage 4: mix of "already here" + navigation scenarios
  // 4 near/at-target (0-3 units, some with drift), 2 short-range, 2 long-range
  const scenarios = [
    // Near target: distance 0-3, higher initial velocity to test braking
    { distRange: [0, 3], speedMul: 0.5 },
    { distRange: [0, 3], speedMul: 0.5 },
    { distRange: [0, 1], speedMul: 0.3 },
    { distRange: [0, 1], speedMul: 0.1 },
    // Navigation (must not regress)
    { distRange: [10, 50], speedMul: 0.2 },
    { distRange: [10, 50], speedMul: 0.2 },
    { distRange: [50, 200], speedMul: 0.2 },
    { distRange: [50, 200], speedMul: 0.2 },
  ];

  for (const { distRange: [lo, hi], speedMul } of scenarios) {
    const targetDist = lo + rng() * (hi - lo);
    const dir = rngDir(rng);
    const targetPos = [dir[0] * targetDist, dir[1] * targetDist, dir[2] * targetDist];
    // Random ship orientation
    const shipQuat = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5];
    const ql = Math.sqrt(shipQuat[0] ** 2 + shipQuat[1] ** 2 + shipQuat[2] ** 2 + shipQuat[3] ** 2);
    for (let j = 0; j < 4; j++) shipQuat[j] /= ql;
    // Initial velocity (higher for near-target to test braking)
    const initSpeed = rng() * speedMul * PHYSICS.MAX_SPEED;
    const vDir = rngDir(rng);
    const shipVel = [vDir[0] * initSpeed, vDir[1] * initSpeed, vDir[2] * initSpeed];
    configs.push({ targetPos, shipQuat, shipVel, nearTarget: lo < 5 });
  }
  return configs;
}

// ══════════════════════════════════════════════════════════════
// ██  Episode runner
// ══════════════════════════════════════════════════════════════
function runEpisode(brain, config) {
  const ship = createShipState('alpha');
  ship.fuel = 2500;
  ship.quat[0] = config.shipQuat[0]; ship.quat[1] = config.shipQuat[1];
  ship.quat[2] = config.shipQuat[2]; ship.quat[3] = config.shipQuat[3];
  ship.vel[0] = config.shipVel[0]; ship.vel[1] = config.shipVel[1]; ship.vel[2] = config.shipVel[2];

  const targetPos = config.targetPos;

  for (let frame = 0; frame < EPISODE_FRAMES; frame++) {
    const inputs = buildSimpleBrainInputs(ship, targetPos);
    const outputs = brain.forward(inputs);
    applySimpleBrainOutputs(ship, outputs);
    shipSimStep(ship);
  }

  const rawDist = V3.distanceTo(ship.pos, targetPos);
  const DEAD_ZONE = 5; // 500m — inside this, distance counts as 0
  const finalDist = Math.max(0, rawDist - DEAD_ZONE);
  const finalSpeed = V3.length(ship.vel);

  if (config.nearTarget) {
    // Stage 4 near-target: pure fuel + staying still
    // Max 100: fuel up to 70, stillness up to 30
    const fuelBonus = (ship.fuel / PHYSICS.MAX_FUEL) * 70;
    const stoppingBonus = Math.max(0, 1 - finalSpeed / (PHYSICS.MAX_SPEED * 0.1)) * 30;
    return fuelBonus + stoppingBonus;
  }

  // Stage 3 fitness for navigation episodes: arrive within dead zone AND stop efficiently
  const proximityBonus = 100 / (1 + finalDist);
  const fuelBonus = (ship.fuel / PHYSICS.MAX_FUEL) * 50;
  const stoppingBonus = rawDist < 10
    ? Math.max(0, 1 - finalSpeed / (PHYSICS.MAX_SPEED * 0.3)) * 30
    : 0;

  return proximityBonus + fuelBonus + stoppingBonus;
}

// ══════════════════════════════════════════════════════════════
// ██  Evaluate a candidate
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
console.log('sep-CMA-ES Stage 2: Navigate + Stop + Fuel Efficiency');
console.log(`Topology: ${TOPOLOGY.join('->')}, params: ${N}`);
console.log(`Lambda: ${LAMBDA}, mu: ${MU}, mueff: ${mueff.toFixed(1)}`);
console.log(`Episodes/candidate: ${EVALS_PER_CANDIDATE}, frames/episode: ${EPISODE_FRAMES}`);
console.log(`Fitness: proximity(100) + fuel(50) + stopping(30) = max 180`);
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

  const episodeConfigs = generateEpisodeConfigs(gen * 1000 + 42);

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

  if (gen % 10 === 0 && globalBestGenome) {
    saveCheckpoint(gen);
  }
}

function saveCheckpoint(genNum) {
  Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify({
    topology: TOPOLOGY,
    activation: 'relu+tanh',
    fitness: globalBestFitness,
    genome: globalBestGenome,
    generation: genNum,
    scenario: 'stage4-stay-still',
    optimizer: { name: 'sep-CMA-ES', lambda: LAMBDA, mu: MU, sigma, N },
    config: { EPISODE_FRAMES, EVALS_PER_CANDIDATE },
    trainedAt: new Date().toISOString(),
  }, null, 2));
}

if (globalBestGenome) saveCheckpoint(gen);

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: ${gen} generations in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
