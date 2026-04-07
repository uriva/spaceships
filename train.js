#!/usr/bin/env -S deno run --allow-read --allow-write
// train.js — Headless neuroevolution trainer using shared SimCore
// Run: deno run --allow-read --allow-write train.js [generations] [output.json]

// Load SimCore via eval (IIFE sets globalThis.SimCore)
const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const { NeuralNetwork, initPopulation, evolve, runMatch, scoreMatch } = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [62, 16, 16, 6];
const POP_SIZE = 20;
const MATCHES_PER_GEN = 10;
const MUTATION_RATE = 0.1;
const MUTATION_STRENGTH = 0.3;
const MATCH_TIME_LIMIT = 1800; // 30s at 60fps
const TARGET_GENS = parseInt(Deno.args[0]) || 100;
const OUTPUT_FILE = Deno.args[1] || 'best-genome.json';

// ── Training loop ──
let population = initPopulation(POP_SIZE, TOPOLOGY);
let bestFitness = -Infinity;
let bestGenome = null;

console.log(`Neuro trainer: pop=${POP_SIZE}, gens=${TARGET_GENS}, matches/gen=${MATCHES_PER_GEN}`);
console.log(`Topology: ${TOPOLOGY.join('→')}, params=${new NeuralNetwork(TOPOLOGY).paramCount}`);
console.log('');

const t0 = performance.now();

for (let gen = 0; gen < TARGET_GENS; gen++) {
  const genStart = performance.now();

  for (let m = 0; m < MATCHES_PER_GEN; m++) {
    // Pick 2 random distinct genomes
    const alphaIdx = Math.floor(Math.random() * POP_SIZE);
    let omegaIdx;
    do { omegaIdx = Math.floor(Math.random() * POP_SIZE); } while (omegaIdx === alphaIdx);

    // Create brains
    const alphaBrain = new NeuralNetwork(TOPOLOGY);
    alphaBrain.fromGenome(population[alphaIdx].genome);
    const omegaBrain = new NeuralNetwork(TOPOLOGY);
    omegaBrain.fromGenome(population[omegaIdx].genome);

    // Run match
    const allShips = runMatch(alphaBrain, omegaBrain, { matchTimeLimit: MATCH_TIME_LIMIT });
    const scores = scoreMatch(allShips);

    // Update running average fitness
    for (const [idx, score] of [[alphaIdx, scores.alpha], [omegaIdx, scores.omega]]) {
      const p = population[idx];
      p.fights++;
      p.fitness += (score - p.fitness) / p.fights;
    }
  }

  // Track best
  for (const p of population) {
    if (p.fitness > bestFitness) {
      bestFitness = p.fitness;
      bestGenome = Array.from(p.genome);
    }
  }

  // Evolve
  population = evolve(population, {
    popSize: POP_SIZE,
    mutationRate: MUTATION_RATE,
    mutationStrength: MUTATION_STRENGTH,
  });

  const genMs = (performance.now() - genStart).toFixed(0);
  const avgFitness = population.reduce((s, p) => s + (p.fights > 0 ? p.fitness : 0), 0) /
    Math.max(1, population.filter(p => p.fights > 0).length);

  // Progress report every generation
  console.log(
    `GEN ${String(gen + 1).padStart(4)} | ` +
    `best: ${bestFitness.toFixed(1).padStart(8)} | ` +
    `avg: ${avgFitness.toFixed(1).padStart(8)} | ` +
    `${genMs}ms`
  );
}

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: ${TARGET_GENS} generations in ${totalSec}s`);
console.log(`Best fitness: ${bestFitness.toFixed(2)}`);

// Save best genome
const output = {
  topology: TOPOLOGY,
  fitness: bestFitness,
  genome: bestGenome,
  config: { POP_SIZE, MATCHES_PER_GEN, MUTATION_RATE, MUTATION_STRENGTH, MATCH_TIME_LIMIT },
  trainedAt: new Date().toISOString(),
  generations: TARGET_GENS,
};
Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
console.log(`Saved to ${OUTPUT_FILE}`);
