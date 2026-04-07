// sim-core.js — shared neuroevolution simulation core
// Works in browser (<script>) and Deno (eval/new Function)
// No Three.js dependency — pure math on plain arrays
(function () {
  'use strict';

  // ══════════════════════════════════════════════════════════════
  // ██  Vector3 utilities — operate on [x, y, z] arrays
  // ══════════════════════════════════════════════════════════════
  const V3 = {
    create: (x = 0, y = 0, z = 0) => [x, y, z],
    clone: (v) => [v[0], v[1], v[2]],
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    scale: (v, s) => [v[0] * s, v[1] * s, v[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    lengthSq: (v) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
    length: (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]),
    normalize: (v) => {
      const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      return l > 1e-12 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
    },
    distanceTo: (a, b) => {
      const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    },
    addMut: (a, b) => { a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; return a; },
    scaleMut: (v, s) => { v[0] *= s; v[1] *= s; v[2] *= s; return v; },
    set: (v, x, y, z) => { v[0] = x; v[1] = y; v[2] = z; return v; },
    copy: (dst, src) => { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; return dst; },
  };

  // ══════════════════════════════════════════════════════════════
  // ██  Quaternion utilities — operate on [x, y, z, w] arrays
  // ══════════════════════════════════════════════════════════════
  const Q = {
    create: () => [0, 0, 0, 1],
    clone: (q) => [q[0], q[1], q[2], q[3]],
    normalize: (q) => {
      const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      if (l < 1e-12) return [0, 0, 0, 1];
      return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
    },
    normalizeMut: (q) => {
      const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      if (l < 1e-12) { q[0] = 0; q[1] = 0; q[2] = 0; q[3] = 1; return q; }
      q[0] /= l; q[1] /= l; q[2] /= l; q[3] /= l;
      return q;
    },
    // a * b (Hamilton product)
    multiply: (a, b) => [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ],
    invert: (q) => {
      const d = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
      return d > 1e-12 ? [-q[0] / d, -q[1] / d, -q[2] / d, q[3] / d] : [0, 0, 0, 1];
    },
    fromAxisAngle: (axis, angle) => {
      const ha = angle * 0.5, s = Math.sin(ha);
      return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(ha)];
    },
    // Rotate vector v by quaternion q
    applyToVec3: (q, v) => {
      const x = v[0], y = v[1], z = v[2];
      const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
      // t = 2 * cross(q.xyz, v)
      const tx = 2 * (qy * z - qz * y);
      const ty = 2 * (qz * x - qx * z);
      const tz = 2 * (qx * y - qy * x);
      return [
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
      ];
    },
    // Set quaternion from random values and normalize — for random orientation
    setRandomMut: (q) => {
      q[0] = Math.random() - 0.5;
      q[1] = Math.random() - 0.5;
      q[2] = Math.random() - 0.5;
      q[3] = Math.random() - 0.5;
      return Q.normalizeMut(q);
    },
  };

  // ══════════════════════════════════════════════════════════════
  // ██  Physics constants
  // ══════════════════════════════════════════════════════════════
  const PHYSICS = {
    THRUST: 0.015,
    TORQUE: 0.006,
    TORQUE_FUEL_COST: 0.05,
    FUEL_COST_PER_FRAME: 0.4,
    MAX_SPEED: 0.5,
    MAX_FUEL: 2500,
    MAX_HP: 50,
    MAX_BATTERY: 100,
    WEAPON_COST: 8,
    LASER_DAMAGE: 10,
    BATTERY_RECHARGE: 0.02,
    MAX_ANG_SPEED: 0.1,         // rad/frame cap for neural torque
    WEAPON_RANGE: 200,          // units (~20 km)
  };

  // ══════════════════════════════════════════════════════════════
  // ██  NeuralNetwork — fixed topology, fast forward pass
  // ══════════════════════════════════════════════════════════════
  class NeuralNetwork {
    constructor(layers) {
      this.layers = layers;
      this.weights = [];
      this.biases = [];
      for (let i = 0; i < layers.length - 1; i++) {
        this.weights.push(new Float64Array(layers[i] * layers[i + 1]));
        this.biases.push(new Float64Array(layers[i + 1]));
      }
    }
    get paramCount() {
      let n = 0;
      for (let i = 0; i < this.layers.length - 1; i++) {
        n += this.layers[i] * this.layers[i + 1] + this.layers[i + 1];
      }
      return n;
    }
    fromGenome(genome) {
      let idx = 0;
      for (let i = 0; i < this.layers.length - 1; i++) {
        const nW = this.layers[i] * this.layers[i + 1];
        for (let j = 0; j < nW; j++) this.weights[i][j] = genome[idx++];
        const nB = this.layers[i + 1];
        for (let j = 0; j < nB; j++) this.biases[i][j] = genome[idx++];
      }
    }
    toGenome() {
      const g = new Float64Array(this.paramCount);
      let idx = 0;
      for (let i = 0; i < this.layers.length - 1; i++) {
        for (let j = 0; j < this.weights[i].length; j++) g[idx++] = this.weights[i][j];
        for (let j = 0; j < this.biases[i].length; j++) g[idx++] = this.biases[i][j];
      }
      return g;
    }
    forward(inputs) {
      let activation = Float64Array.from(inputs);
      for (let i = 0; i < this.layers.length - 1; i++) {
        const nIn = this.layers[i];
        const nOut = this.layers[i + 1];
        const next = new Float64Array(nOut);
        for (let o = 0; o < nOut; o++) {
          let sum = this.biases[i][o];
          for (let j = 0; j < nIn; j++) {
            sum += activation[j] * this.weights[i][o * nIn + j];
          }
          next[o] = Math.tanh(sum);
        }
        activation = next;
      }
      return activation;
    }
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Ship state — plain object, no Three.js
  // ══════════════════════════════════════════════════════════════
  function createShipState(team) {
    return {
      pos: V3.create(),
      vel: V3.create(),
      quat: Q.create(),
      angVel: V3.create(),
      hp: PHYSICS.MAX_HP,
      maxHp: PHYSICS.MAX_HP,
      battery: PHYSICS.MAX_BATTERY,
      maxBattery: PHYSICS.MAX_BATTERY,
      fuel: PHYSICS.MAX_FUEL,
      maxFuel: PHYSICS.MAX_FUEL,
      team,                  // 'alpha' or 'omega'
      alive: true,
      neuralDamageDealt: 0,
      neuralDamageTaken: 0,
      weaponsDepleted: false,
      shieldFlash: 0,
      isAccelerating: false,
      isBraking: false,
    };
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Per-frame ship physics (angular velocity → quat, position, recharge)
  // ══════════════════════════════════════════════════════════════
  function shipSimStep(ship) {
    // Angular velocity → quaternion integration
    const angSpeed = V3.length(ship.angVel);
    if (angSpeed > 1e-6) {
      const axis = V3.normalize(ship.angVel);
      const dq = Q.fromAxisAngle(axis, angSpeed);
      // premultiply: dq * ship.quat
      const newQ = Q.multiply(dq, ship.quat);
      ship.quat[0] = newQ[0]; ship.quat[1] = newQ[1];
      ship.quat[2] = newQ[2]; ship.quat[3] = newQ[3];
      Q.normalizeMut(ship.quat);
    }

    // Position integration
    V3.addMut(ship.pos, ship.vel);

    // Battery recharge
    if (ship.battery < ship.maxBattery) {
      ship.battery = Math.min(ship.maxBattery, ship.battery + PHYSICS.BATTERY_RECHARGE);
      if (ship.battery > PHYSICS.WEAPON_COST && ship.weaponsDepleted) {
        ship.weaponsDepleted = false;
      }
    }

    // Shield flash decay
    if (ship.shieldFlash > 0) ship.shieldFlash--;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Asteroid helpers
  // ══════════════════════════════════════════════════════════════
  function createAsteroid(pos, radius) {
    return { pos: V3.clone(pos), radius };
  }

  function checkAsteroidCollisions(ship, asteroids) {
    for (const a of asteroids) {
      const dist = V3.distanceTo(ship.pos, a.pos);
      if (dist < a.radius) {
        ship.alive = false;
        ship.hp = 0;
        return true;
      }
    }
    return false;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Build 78-element NN input vector
  // ══════════════════════════════════════════════════════════════
  // 8 own state + 54 (6 ships × 9) + 16 (4 asteroids × 4) = 78
  function buildNNInputs(ship, allShips, asteroids) {
    const inputs = new Float64Array(78);

    // Own state (8 inputs)
    inputs[0] = ship.angVel[0];
    inputs[1] = ship.angVel[1];
    inputs[2] = ship.angVel[2];

    const speed = V3.length(ship.vel);
    inputs[3] = speed / PHYSICS.MAX_SPEED;

    // forward · velocity (alignment)
    const fwd = Q.applyToVec3(ship.quat, [0, 0, 1]);
    inputs[4] = speed > 1e-4 ? V3.dot(fwd, V3.normalize(ship.vel)) : 0;

    inputs[5] = ship.battery / ship.maxBattery;
    inputs[6] = ship.hp / ship.maxHp;
    inputs[7] = ship.fuel / ship.maxFuel;

    // Nearest 6 visible ships (9 each = 54 inputs)
    const invQuat = Q.invert(ship.quat);
    const others = [];
    for (const other of allShips) {
      if (other === ship || !other.alive) continue;
      const relPos = V3.sub(other.pos, ship.pos);
      const dist = V3.length(relPos);
      others.push({ ship: other, dist, relPos });
    }
    others.sort((a, b) => a.dist - b.dist);

    for (let i = 0; i < 6; i++) {
      const base = 8 + i * 9;
      if (i < others.length) {
        const o = others[i];
        const localPos = Q.applyToVec3(invQuat, o.relPos);
        inputs[base + 0] = localPos[0] / 500;
        inputs[base + 1] = localPos[1] / 500;
        inputs[base + 2] = localPos[2] / 500;
        const relVel = V3.sub(o.ship.vel, ship.vel);
        const localVel = Q.applyToVec3(invQuat, relVel);
        inputs[base + 3] = localVel[0] / PHYSICS.MAX_SPEED;
        inputs[base + 4] = localVel[1] / PHYSICS.MAX_SPEED;
        inputs[base + 5] = localVel[2] / PHYSICS.MAX_SPEED;
        inputs[base + 6] = o.ship.team === ship.team ? 1 : -1;
        inputs[base + 7] = o.ship.hp / o.ship.maxHp;
        inputs[base + 8] = o.ship.battery / o.ship.maxBattery;
      }
      // else: stays 0
    }

    // Nearest 4 asteroids (4 each: local pos xyz / 500, radius / 50 = 16 inputs)
    if (asteroids && asteroids.length > 0) {
      const astDist = [];
      for (const a of asteroids) {
        const relPos = V3.sub(a.pos, ship.pos);
        const dist = V3.length(relPos);
        astDist.push({ asteroid: a, dist, relPos });
      }
      astDist.sort((a, b) => a.dist - b.dist);

      for (let i = 0; i < 4; i++) {
        const base = 62 + i * 4;
        if (i < astDist.length) {
          const a = astDist[i];
          const localPos = Q.applyToVec3(invQuat, a.relPos);
          inputs[base + 0] = localPos[0] / 500;
          inputs[base + 1] = localPos[1] / 500;
          inputs[base + 2] = localPos[2] / 500;
          inputs[base + 3] = a.asteroid.radius / 50;
        }
      }
    }

    return inputs;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Apply NN outputs to ship — returns { firedAt: ship|null }
  // ══════════════════════════════════════════════════════════════
  function applyNNOutputs(ship, outputs, allShips) {
    const hasFuel = ship.fuel > 0;
    let firedAt = null;

    // outputs[0..2]: body torque (xyz)
    if (hasFuel) {
      ship.angVel[0] += outputs[0] * PHYSICS.TORQUE;
      ship.angVel[1] += outputs[1] * PHYSICS.TORQUE;
      ship.angVel[2] += outputs[2] * PHYSICS.TORQUE;
      // Cap angular velocity
      const angSpeed = V3.length(ship.angVel);
      if (angSpeed > PHYSICS.MAX_ANG_SPEED) {
        V3.scaleMut(ship.angVel, PHYSICS.MAX_ANG_SPEED / angSpeed);
      }
      ship.fuel = Math.max(0, ship.fuel - PHYSICS.TORQUE_FUEL_COST);
    }

    // outputs[3]: engine burn along forward axis
    const burn = outputs[3];
    if (Math.abs(burn) > 0.1 && hasFuel) {
      const fwd = Q.applyToVec3(ship.quat, [0, 0, 1]);
      const thrust = V3.scale(fwd, burn * PHYSICS.THRUST);
      V3.addMut(ship.vel, thrust);
      const speed = V3.length(ship.vel);
      if (speed > PHYSICS.MAX_SPEED) {
        V3.scaleMut(ship.vel, PHYSICS.MAX_SPEED / speed);
      }
      ship.fuel = Math.max(0, ship.fuel - PHYSICS.FUEL_COST_PER_FRAME * Math.abs(burn));
      ship.isAccelerating = burn > 0;
      ship.isBraking = burn < 0;
    } else {
      ship.isAccelerating = false;
      ship.isBraking = false;
    }

    // outputs[4]: target selection, outputs[5]: fire
    const enemies = [];
    for (const s of allShips) {
      if (s !== ship && s.alive && s.team !== ship.team) enemies.push(s);
    }
    // Sort by distance
    enemies.sort((a, b) => V3.distanceTo(ship.pos, a.pos) - V3.distanceTo(ship.pos, b.pos));

    if (enemies.length > 0) {
      const idx = Math.min(
        Math.floor((outputs[4] + 1) / 2 * enemies.length),
        enemies.length - 1
      );
      const target = enemies[Math.max(0, idx)];

      // Fire
      if (outputs[5] > 0 && ship.battery >= PHYSICS.WEAPON_COST) {
        const dist = V3.distanceTo(ship.pos, target.pos);
        if (dist < PHYSICS.WEAPON_RANGE) {
          // Deduct battery from shooter
          ship.battery = Math.max(0, ship.battery - PHYSICS.WEAPON_COST);

          // Apply damage to target
          if (target.battery > 0) {
            target.battery = Math.max(0, target.battery - PHYSICS.LASER_DAMAGE);
            target.shieldFlash = 45;
          } else {
            target.hp -= PHYSICS.LASER_DAMAGE;
          }

          // Track damage
          ship.neuralDamageDealt += PHYSICS.LASER_DAMAGE;
          target.neuralDamageTaken += PHYSICS.LASER_DAMAGE;

          // Kill check
          if (target.hp <= 0) {
            target.alive = false;
          }

          firedAt = target;
        }
      }
    }

    return { firedAt };
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Score a completed match — returns { alpha: number, omega: number }
  // ══════════════════════════════════════════════════════════════
  function scoreMatch(allShips) {
    const teams = { alpha: { ships: [], alive: [] }, omega: { ships: [], alive: [] } };
    for (const s of allShips) {
      teams[s.team].ships.push(s);
      if (s.alive) teams[s.team].alive.push(s);
    }

    function scoreTeam(teamKey, enemyKey) {
      let score = 0;
      // Damage dealt ×3, damage taken ×0.5
      for (const s of teams[teamKey].ships) {
        score += s.neuralDamageDealt * 3;
        score -= s.neuralDamageTaken * 0.5;
      }
      // Kill bonus: 150 per enemy killed
      const enemiesKilled = teams[enemyKey].ships.length - teams[enemyKey].alive.length;
      score += enemiesKilled * 150;
      // Proximity penalty
      const alive = teams[teamKey].alive;
      const enemyAlive = teams[enemyKey].alive;
      if (alive.length > 0 && enemyAlive.length > 0) {
        let totalDist = 0;
        for (const s of alive) {
          let minDist = Infinity;
          for (const e of enemyAlive) minDist = Math.min(minDist, V3.distanceTo(s.pos, e.pos));
          totalDist += minDist;
        }
        score -= Math.min(totalDist / alive.length, 200);
      }
      return score;
    }

    return {
      alpha: scoreTeam('alpha', 'omega'),
      omega: scoreTeam('omega', 'alpha'),
    };
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Evolution — selection, crossover, mutation
  // ══════════════════════════════════════════════════════════════

  function crossover(g1, g2) {
    const child = new Float64Array(g1.length);
    for (let i = 0; i < child.length; i++) {
      child[i] = Math.random() < 0.5 ? g1[i] : g2[i];
    }
    return child;
  }

  function mutate(genome, rate, strength) {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < rate) {
        genome[i] += (Math.random() * 2 - 1) * strength;
      }
    }
  }

  function evolve(population, config) {
    const { popSize, mutationRate, mutationStrength, elitism = 2 } = config;
    const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
    const parents = sorted.slice(0, Math.floor(popSize / 2));
    const newPop = [];

    // Elitism
    for (let i = 0; i < elitism && i < parents.length; i++) {
      newPop.push({
        genome: new Float64Array(parents[i].genome),
        fitness: 0,
        fights: 0,
      });
    }

    // Fill with crossover + mutation
    while (newPop.length < popSize) {
      const p1 = parents[Math.floor(Math.random() * parents.length)];
      const p2 = parents[Math.floor(Math.random() * parents.length)];
      const child = crossover(p1.genome, p2.genome);
      mutate(child, mutationRate, mutationStrength);
      newPop.push({ genome: child, fitness: 0, fights: 0 });
    }

    return newPop;
  }

  function initPopulation(size, topology) {
    const nn = new NeuralNetwork(topology);
    const nParams = nn.paramCount;
    const pop = [];
    for (let i = 0; i < size; i++) {
      const genome = new Float64Array(nParams);
      let idx = 0;
      for (let li = 0; li < topology.length - 1; li++) {
        const fanIn = topology[li];
        const scale = 1 / Math.sqrt(fanIn);
        const nW = topology[li] * topology[li + 1];
        for (let j = 0; j < nW; j++) genome[idx++] = (Math.random() * 2 - 1) * scale;
        const nB = topology[li + 1];
        for (let j = 0; j < nB; j++) genome[idx++] = 0;
      }
      pop.push({ genome, fitness: 0, fights: 0 });
    }
    return pop;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Run a complete match — returns array of all ship states
  // ══════════════════════════════════════════════════════════════
  function runMatch(alphaBrain, omegaBrain, config = {}) {
    const fleetSize = config.fleetSize || 5;
    const matchTimeLimit = config.matchTimeLimit || 1800;
    const separation = config.separation || 50;
    const spread = config.spread || 30;

    // Create ships
    const allShips = [];
    for (let teamIdx = 0; teamIdx < 2; teamIdx++) {
      const team = teamIdx === 0 ? 'alpha' : 'omega';
      const brain = teamIdx === 0 ? alphaBrain : omegaBrain;
      const zSign = teamIdx === 0 ? 1 : -1;
      for (let i = 0; i < fleetSize; i++) {
        const s = createShipState(team);
        s.pos[0] = (Math.random() - 0.5) * spread;
        s.pos[1] = (Math.random() - 0.5) * spread;
        s.pos[2] = zSign * separation + (Math.random() - 0.5) * spread;
        Q.setRandomMut(s.quat);
        s._brain = brain;
        allShips.push(s);
      }
    }

    // Simulate
    for (let frame = 0; frame < matchTimeLimit; frame++) {
      // Check early termination
      let alphaAlive = false, omegaAlive = false;
      for (const s of allShips) {
        if (!s.alive) continue;
        if (s.team === 'alpha') alphaAlive = true;
        else omegaAlive = true;
        if (alphaAlive && omegaAlive) break;
      }
      if (!alphaAlive || !omegaAlive) break;

      // NN + physics for each living ship
      for (const s of allShips) {
        if (!s.alive) continue;
        const inputs = buildNNInputs(s, allShips, []);
        const outputs = s._brain.forward(inputs);
        applyNNOutputs(s, outputs, allShips);
        shipSimStep(s);
      }
    }

    return allShips;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Export
  // ══════════════════════════════════════════════════════════════
  const SimCore = {
    V3, Q, PHYSICS,
    NeuralNetwork,
    createShipState,
    createAsteroid,
    checkAsteroidCollisions,
    shipSimStep,
    buildNNInputs,
    applyNNOutputs,
    scoreMatch,
    crossover,
    mutate,
    evolve,
    initPopulation,
    runMatch,
  };

  if (typeof globalThis !== 'undefined') globalThis.SimCore = SimCore;
  if (typeof module !== 'undefined' && module.exports) module.exports = SimCore;
})();
