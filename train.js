#!/usr/bin/env -S deno run --allow-read --allow-write
// train.js — 3v3 fleet trainer: focus fire + coordination
// Scenario: 3v3 fleet combat, no asteroids, 200-unit arena
// Scoring: kill-first + focus fire + team win/loss + fuel conservation
// Seeds from best-genome.json if available
// Run: deno run --allow-read --allow-write train.js [output.json]

const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const {
  V3, Q, PHYSICS, NeuralNetwork,
  createShipState, checkAsteroidCollisions,
  shipSimStep, buildNNInputs, applyNNOutputs,
  initPopulation, evolve,
} = globalThis.SimCore;

// ── Config ──
const TOPOLOGY = [80, 16, 16, 6];
const POP_SIZE = 80;
const OPPONENTS_PER_EVAL = 3;
const MUTATION_RATE = 0.12;
const MUTATION_STRENGTH = 0.30;
const OUTPUT_FILE = Deno.args[0] || 'best-genome.json';
const ARENA_RADIUS = 100;
const MATCH_FRAMES = 1500;        // ~25 seconds at 60fps
const FLEET_SIZE = 3;
const SEPARATION = 50;
const SPREAD = 15;
const RUN_MINUTES = 30;

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

// ══════════════════════════════════════════════════════════════
// ██  Scoring: kill-first + focus fire + team coordination
// ══════════════════════════════════════════════════════════════
function scoreTeamResult(allShips, matchFrames) {
  const teams = { alpha: { ships: [], alive: [] }, omega: { ships: [], alive: [] } };
  for (const s of allShips) {
    teams[s.team].ships.push(s);
    if (s.alive) teams[s.team].alive.push(s);
  }

  function scoreTeam(teamKey, enemyKey) {
    let score = 0;
    const myShips = teams[teamKey].ships;
    const myAlive = teams[teamKey].alive;
    const enemyShips = teams[enemyKey].ships;
    const enemyAlive = teams[enemyKey].alive;

    for (const s of myShips) {
      // ── Net approach: reward closing, penalize retreating ──
      const netApproach = (s._distanceClosed || 0) - (s._distanceOpened || 0);
      score += netApproach * 10;

      // ── Retreat penalty ──
      score -= (s._distanceOpened || 0) * 3;

      // ── Time in weapon range ──
      score += (s._framesInRange || 0) * 3;

      // ── Damage dealt: reduced base, hull still valued ──
      score += s.neuralDamageDealt * 2;
      score += (s.neuralHullDamageDealt || 0) * 5;

      // ── Focus fire reward: bonus for damage to lowest-HP enemy ──
      score += (s._focusDamage || 0) * 15;

      // ── Shots fired: reward aggression ──
      score += (s._shotsFired || 0) * 5;

      // ── Never-closed penalty ──
      if ((s._distanceClosed || 0) < 5) score -= 500;

      // ── Never-fired penalty ──
      if ((s._shotsFired || 0) === 0) score -= 300;

      // ── Fuel conservation ──
      const fuelRatio = s.fuel / s.maxFuel;
      score += fuelRatio * 300;
      if (s.fuel <= 0) score -= 500;
      if (fuelRatio < 0.2) score -= (0.2 - fuelRatio) * 1000;

      // ── Wandering penalty ──
      score -= (s._wanderPenalty || 0) * 2;
    }

    // ── Kills: massive bonus — THE primary objective ──
    const enemiesKilled = enemyShips.length - enemyAlive.length;
    score += enemiesKilled * 3000;

    // ── Survival bonus ──
    score += myAlive.length * 100;

    // ── Team outcome: win/loss bonus ──
    const myKilled = myShips.length - myAlive.length;
    if (enemiesKilled > myKilled) score += 1000;       // winning team
    else if (enemiesKilled < myKilled) score -= 500;    // losing team

    // ── Full wipe bonus: annihilated the enemy fleet ──
    if (enemyAlive.length === 0) score += 2000;

    // ── Final proximity: penalty for distance at match end ──
    if (myAlive.length > 0 && enemyAlive.length > 0) {
      let totalDist = 0;
      for (const s of myAlive) {
        let minDist = Infinity;
        for (const e of enemyAlive) minDist = Math.min(minDist, V3.distanceTo(s.pos, e.pos));
        totalDist += minDist;
      }
      score -= (totalDist / myAlive.length) * 15;
    }

    return score;
  }

  return {
    alpha: scoreTeam('alpha', 'omega'),
    omega: scoreTeam('omega', 'alpha'),
  };
}

// ══════════════════════════════════════════════════════════════
// ██  Run match with tracking (no asteroids)
// ══════════════════════════════════════════════════════════════
function runMatchWithTracking(alphaBrain, omegaBrain) {
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
      s._distanceOpened = 0;
      s._shotsFired = 0;
      s._focusDamage = 0;
      s._prevMinDist = Infinity;
      s._wanderPenalty = 0;
      allShips.push(s);
    }
  }

  const NO_ASTEROIDS = [];
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
      const inputs = buildNNInputs(s, allShips, NO_ASTEROIDS);
      const outputs = s._brain.forward(inputs);
      const result = applyNNOutputs(s, outputs, allShips);
      shipSimStep(s);

      // Track per-frame metrics
      if (result.firedAt) {
        s._shotsFired++;
        // Focus fire: bonus if this shot hit the lowest-HP enemy
        let lowestHp = Infinity, lowestEnemy = null;
        for (const e of allShips) {
          if (e === s || !e.alive || e.team === s.team) continue;
          if (e.hp < lowestHp) { lowestHp = e.hp; lowestEnemy = e; }
        }
        if (result.firedAt === lowestEnemy) {
          s._focusDamage += PHYSICS.LASER_DAMAGE;
        }
      }
      let minDistToEnemy = Infinity;
      for (const e of allShips) {
        if (e === s || !e.alive || e.team === s.team) continue;
        const d = V3.distanceTo(s.pos, e.pos);
        if (d < minDistToEnemy) minDistToEnemy = d;
      }
      if (minDistToEnemy < PHYSICS.WEAPON_RANGE) s._framesInRange++;
      if (s._prevMinDist < Infinity && minDistToEnemy < Infinity) {
        const delta = s._prevMinDist - minDistToEnemy;
        if (delta > 0) s._distanceClosed += delta;
        else s._distanceOpened += (-delta);
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

    const ships = runMatchWithTracking(brain, oppBrain);
    const scores = scoreTeamResult(ships, ships._matchFrames);
    totalFitness += scores.alpha;
  }

  return totalFitness / OPPONENTS_PER_EVAL;
}

// ══════════════════════════════════════════════════════════════
// ██  Training loop — time-based (runs for RUN_MINUTES)
// ══════════════════════════════════════════════════════════════
let population = initPopulation(POP_SIZE, TOPOLOGY);

// Seed best genome into first slot + mutated copies
const seedGenome = trySeedGenome();
if (seedGenome && seedGenome.length === population[0].genome.length) {
  population[0].genome = new Float64Array(seedGenome);
  for (let i = 1; i < Math.min(10, POP_SIZE); i++) {
    population[i].genome = new Float64Array(seedGenome);
    for (let j = 0; j < population[i].genome.length; j++) {
      if (Math.random() < MUTATION_RATE) {
        population[i].genome[j] += (Math.random() * 2 - 1) * MUTATION_STRENGTH;
      }
    }
  }
  console.log('Seeded 10 individuals from best-genome.json');
}

let globalBestFitness = -Infinity;
let globalBestGenome = null;

console.log(`3v3 fleet trainer: focus fire + coordination`);
console.log(`Pop: ${POP_SIZE}, opponents/eval: ${OPPONENTS_PER_EVAL}, arena: ${ARENA_RADIUS}u radius`);
console.log(`Topology: ${TOPOLOGY.join('→')}, params: ${new NeuralNetwork(TOPOLOGY).paramCount}`);
console.log(`Running for ${RUN_MINUTES} minutes...`);

const t0 = performance.now();
const timeLimitMs = RUN_MINUTES * 60 * 1000;
let gen = 0;

while (performance.now() - t0 < timeLimitMs) {
  const genStart = performance.now();
  gen++;

  for (const p of population) {
    p.fitness = evaluate(p.genome, population);
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

  if (gen % 20 === 0 && globalBestGenome) {
    const checkpoint = {
      topology: TOPOLOGY,
      fitness: globalBestFitness,
      genome: globalBestGenome,
      generation: gen,
      scenario: '3v3-focusfire',
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
    scenario: '3v3-focusfire',
    config: { POP_SIZE, OPPONENTS_PER_EVAL, MUTATION_RATE, MUTATION_STRENGTH, ARENA_RADIUS, MATCH_FRAMES },
    trainedAt: new Date().toISOString(),
  };
  Deno.writeTextFileSync(OUTPUT_FILE, JSON.stringify(final, null, 2));
}

const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\nDone: ${gen} generations in ${totalSec}s`);
console.log(`Global best fitness: ${globalBestFitness.toFixed(2)}`);
console.log(`Saved to ${OUTPUT_FILE}`);
