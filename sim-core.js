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
    // Rotate (vx,vy,vz) by quaternion q, write result into out[3]
    applyToVec3Into: (q, vx, vy, vz, out) => {
      const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
      const tx = 2 * (qy * vz - qz * vy);
      const ty = 2 * (qz * vx - qx * vz);
      const tz = 2 * (qx * vy - qy * vx);
      out[0] = vx + qw * tx + qy * tz - qz * ty;
      out[1] = vy + qw * ty + qz * tx - qx * tz;
      out[2] = vz + qw * tz + qx * ty - qy * tx;
    },
    // Invert quaternion, write into pre-allocated out[4]
    invertInto: (q, out) => {
      const d = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
      if (d > 1e-12) { out[0] = -q[0]/d; out[1] = -q[1]/d; out[2] = -q[2]/d; out[3] = q[3]/d; }
      else { out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1; }
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
    MAX_ANG_SPEED: 0.05,         // rad/frame cap for neural torque
    WEAPON_RANGE: 200,          // units (~20 km)
    WEAPON_CONE_COS: Math.cos(5 * Math.PI / 180),  // cos(5°) half-angle cone
    // Gravity: G_real converted to game units (gu distance, tonnes mass, frames time)
    // G_game = G_real × 1000 / 1e6 / 3600 = ~1.854e-17
    G_GAME: 6.674e-11 * 1000 / 1e6 / 3600,
    ROCK_DENSITY: 2500,       // kg/m³ — typical stony asteroid
  };

  // ══════════════════════════════════════════════════════════════
  // ██  NeuralNetwork — fixed topology, fast forward pass
  // ══════════════════════════════════════════════════════════════
  class NeuralNetwork {
    constructor(layers) {
      this.layers = layers;
      this.weights = [];
      this.biases = [];
      this._bufs = [];   // pre-allocated activation buffers per layer
      for (let i = 0; i < layers.length - 1; i++) {
        this.weights.push(new Float64Array(layers[i] * layers[i + 1]));
        this.biases.push(new Float64Array(layers[i + 1]));
        this._bufs.push(new Float64Array(layers[i + 1]));
      }
      this._inputBuf = new Float64Array(layers[0]);
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
      // Copy inputs into pre-allocated buffer (avoid Float64Array.from allocation)
      const inp = this._inputBuf;
      for (let k = 0; k < inp.length; k++) inp[k] = inputs[k];
      let activation = inp;
      for (let i = 0; i < this.layers.length - 1; i++) {
        const nIn = this.layers[i];
        const nOut = this.layers[i + 1];
        const next = this._bufs[i]; // pre-allocated
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
  // ██  ReLUNetwork — ReLU hidden layers, tanh output layer
  // ══════════════════════════════════════════════════════════════
  class ReLUNetwork {
    constructor(layers) {
      this.layers = layers;
      this.weights = [];
      this.biases = [];
      this._bufs = [];
      for (let i = 0; i < layers.length - 1; i++) {
        this.weights.push(new Float64Array(layers[i] * layers[i + 1]));
        this.biases.push(new Float64Array(layers[i + 1]));
        this._bufs.push(new Float64Array(layers[i + 1]));
      }
      this._inputBuf = new Float64Array(layers[0]);
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
      const inp = this._inputBuf;
      for (let k = 0; k < inp.length; k++) inp[k] = inputs[k];
      let activation = inp;
      const lastLayer = this.layers.length - 2;
      for (let i = 0; i <= lastLayer; i++) {
        const nIn = this.layers[i];
        const nOut = this.layers[i + 1];
        const next = this._bufs[i];
        const isOutput = (i === lastLayer);
        for (let o = 0; o < nOut; o++) {
          let sum = this.biases[i][o];
          for (let j = 0; j < nIn; j++) {
            sum += activation[j] * this.weights[i][o * nIn + j];
          }
          next[o] = isOutput ? Math.tanh(sum) : Math.max(0, sum); // ReLU hidden, tanh output
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
      neuralHullDamageDealt: 0,
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
    // Angular damping — ships naturally stop spinning (~84% decay/sec)
    V3.scaleMut(ship.angVel, 0.97);

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
  // ██  Build 80-element NN input vector
  // ══════════════════════════════════════════════════════════════
  // 10 own state + 54 (6 ships × 9) + 16 (4 asteroids × 4) = 80
  // Own state: angVel(3), speed, velAlign, battery, hp, fuel, damageTaken, damageDealt
  // Pre-allocated scratch buffers to avoid per-call allocations
  const _nnInputs = new Float64Array(80);
  const _fwdTmp = [0, 0, 0];
  const _rightTmp = [0, 0, 0];
  const _upTmp = [0, 0, 0];
  const _relPosTmp = [0, 0, 0];
  const _relVelTmp = [0, 0, 0];
  const _localTmp = [0, 0, 0];
  const _invQuatTmp = [0, 0, 0, 1];
  const _othersBuf = []; // reused sort array; entries are { ship, dist, r0, r1, r2 }

  function buildNNInputs(ship, allShips, asteroids) {
    const inputs = _nnInputs;
    inputs.fill(0);

    // Own state (8 inputs)
    inputs[0] = ship.angVel[0];
    inputs[1] = ship.angVel[1];
    inputs[2] = ship.angVel[2];

    const speed = V3.length(ship.vel);
    inputs[3] = speed / PHYSICS.MAX_SPEED;

    // forward · velocity (alignment)
    Q.applyToVec3Into(ship.quat, 0, 0, 1, _fwdTmp);
    inputs[4] = speed > 1e-4 ? V3.dot(_fwdTmp, V3.normalize(ship.vel)) : 0;

    inputs[5] = ship.battery / ship.maxBattery;
    inputs[6] = ship.hp / ship.maxHp;
    inputs[7] = ship.fuel / ship.maxFuel;
    inputs[8] = ship.neuralDamageTaken / ship.maxHp;   // how hurt am I (0→1+)
    inputs[9] = Math.min(ship.neuralDamageDealt / 100, 2); // how effective am I (scaled)

    // Nearest 6 visible ships (9 each = 54 inputs)
    Q.invertInto(ship.quat, _invQuatTmp);
    let nOthers = 0;
    for (let si = 0; si < allShips.length; si++) {
      const other = allShips[si];
      if (other === ship || !other.alive) continue;
      const r0 = other.pos[0] - ship.pos[0];
      const r1 = other.pos[1] - ship.pos[1];
      const r2 = other.pos[2] - ship.pos[2];
      const dist = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2);
      if (nOthers < _othersBuf.length) {
        const e = _othersBuf[nOthers];
        e.ship = other; e.dist = dist; e.r0 = r0; e.r1 = r1; e.r2 = r2;
      } else {
        _othersBuf.push({ ship: other, dist, r0, r1, r2 });
      }
      nOthers++;
    }
    // Sort only the populated portion
    const sortSlice = _othersBuf.length > nOthers ? _othersBuf.slice(0, nOthers) : _othersBuf;
    if (sortSlice.length !== nOthers) { sortSlice.length = nOthers; }
    // In-place partial sort: we only need top-6, but full sort is fine for <=20 ships
    for (let i = 0; i < nOthers - 1; i++) {
      for (let j = i + 1; j < nOthers; j++) {
        if (_othersBuf[j].dist < _othersBuf[i].dist) {
          const tmp = _othersBuf[i];
          _othersBuf[i] = _othersBuf[j];
          _othersBuf[j] = tmp;
        }
      }
      if (i >= 5) break; // only need top 6
    }

    for (let i = 0; i < 6; i++) {
      const base = 10 + i * 9;
      if (i < nOthers) {
        const o = _othersBuf[i];
        Q.applyToVec3Into(_invQuatTmp, o.r0, o.r1, o.r2, _localTmp);
        inputs[base + 0] = _localTmp[0] / 50;
        inputs[base + 1] = _localTmp[1] / 50;
        inputs[base + 2] = _localTmp[2] / 50;
        const vr0 = o.ship.vel[0] - ship.vel[0];
        const vr1 = o.ship.vel[1] - ship.vel[1];
        const vr2 = o.ship.vel[2] - ship.vel[2];
        Q.applyToVec3Into(_invQuatTmp, vr0, vr1, vr2, _localTmp);
        inputs[base + 3] = _localTmp[0] / PHYSICS.MAX_SPEED;
        inputs[base + 4] = _localTmp[1] / PHYSICS.MAX_SPEED;
        inputs[base + 5] = _localTmp[2] / PHYSICS.MAX_SPEED;
        inputs[base + 6] = o.ship.team === ship.team ? 1 : -1;
        inputs[base + 7] = o.ship.hp / o.ship.maxHp;
        inputs[base + 8] = o.ship.battery / o.ship.maxBattery;
      }
    }

    // Nearest 4 asteroids (4 each: local pos xyz / 200, radius / 50 = 16 inputs)
    if (asteroids && asteroids.length > 0) {
      // Simple insertion-sort for nearest 4 — avoid allocating temp arrays
      const a4 = [null, null, null, null];
      const a4d = [Infinity, Infinity, Infinity, Infinity];
      const a4r = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]];
      for (let ai = 0; ai < asteroids.length; ai++) {
        const a = asteroids[ai];
        const r0 = a.pos[0] - ship.pos[0];
        const r1 = a.pos[1] - ship.pos[1];
        const r2 = a.pos[2] - ship.pos[2];
        const dist = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2);
        // Insert into sorted top-4
        for (let k = 0; k < 4; k++) {
          if (dist < a4d[k]) {
            // Shift down
            for (let m = 3; m > k; m--) { a4[m] = a4[m-1]; a4d[m] = a4d[m-1]; a4r[m] = a4r[m-1]; }
            a4[k] = a; a4d[k] = dist; a4r[k] = [r0, r1, r2];
            break;
          }
        }
      }

      for (let i = 0; i < 4; i++) {
        const base = 64 + i * 4;
        if (a4[i]) {
          Q.applyToVec3Into(_invQuatTmp, a4r[i][0], a4r[i][1], a4r[i][2], _localTmp);
          inputs[base + 0] = _localTmp[0] / 200;
          inputs[base + 1] = _localTmp[1] / 200;
          inputs[base + 2] = _localTmp[2] / 200;
          inputs[base + 3] = a4[i].radius / 50;
        }
      }
    }

    return inputs;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Gravity — apply asteroid gravitational acceleration
  // ══════════════════════════════════════════════════════════════
  // asteroids must have .pos [x,y,z] and .mass (tonnes)
  // gravityMultiplier defaults to 1 (real gravity)
  function applyGravity(ship, asteroids, gravityMultiplier) {
    if (!asteroids || asteroids.length === 0) return;
    const gm = (gravityMultiplier || 1);
    const G = PHYSICS.G_GAME * gm;
    for (let i = 0; i < asteroids.length; i++) {
      const a = asteroids[i];
      const dx = a.pos[0] - ship.pos[0];
      const dy = a.pos[1] - ship.pos[1];
      const dz = a.pos[2] - ship.pos[2];
      const distSq = dx * dx + dy * dy + dz * dz;
      const dist = Math.sqrt(distSq);
      if (dist < 3) continue; // avoid singularity at overlap
      const strength = G * a.mass / distSq;
      const invDist = 1 / dist;
      ship.vel[0] += dx * invDist * strength;
      ship.vel[1] += dy * invDist * strength;
      ship.vel[2] += dz * invDist * strength;
    }
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Build 43-element brain input vector (BRAIN.md spec)
  // ══════════════════════════════════════════════════════════════
  // All entity positions as ship-local polar (azimuth, elevation, distance).
  // targetPos: [x,y,z] world position of current target.
  // enemies/friends: arrays of ship states (can be empty).
  // asteroids: array of {pos, radius, mass} (can be empty).
  const _brainInputs = new Float64Array(43);
  const _brainInvQ = [0, 0, 0, 1];
  const _brainLocal = [0, 0, 0];
  const _brainFwd = [0, 0, 0];

  // Convert world-space vector to ship-local polar (azimuth, elevation, distance)
  // Returns [azimuth/π, elevation/(π/2), 1/(1+dist/50)]
  // If distance is 0 (no entity), returns [0, 0, 0]
  function _worldToLocalPolar(shipQuat, invQuat, worldDelta, out) {
    const dist = Math.sqrt(worldDelta[0] * worldDelta[0] + worldDelta[1] * worldDelta[1] + worldDelta[2] * worldDelta[2]);
    if (dist < 1e-6) { out[0] = 0; out[1] = 0; out[2] = 0; return; }
    Q.applyToVec3Into(invQuat, worldDelta[0], worldDelta[1], worldDelta[2], _brainLocal);
    const lx = _brainLocal[0], ly = _brainLocal[1], lz = _brainLocal[2];
    const horizDist = Math.sqrt(lx * lx + lz * lz);
    const azimuth = Math.atan2(lx, lz);          // -π to π, 0 = forward
    const elevation = Math.atan2(ly, horizDist);  // -π/2 to π/2
    out[0] = azimuth / Math.PI;
    out[1] = elevation / (Math.PI / 2);
    out[2] = 1 / (1 + dist / 50);
  }

  function buildBrainInputs(ship, targetPos, enemies, friends, asteroids) {
    const inp = _brainInputs;
    inp.fill(0);

    Q.invertInto(ship.quat, _brainInvQ);

    // ── Own state (6) ──
    inp[0] = ship.hp / ship.maxHp;
    inp[1] = ship.battery / ship.maxBattery;
    inp[2] = ship.fuel / ship.maxFuel;
    const speed = V3.length(ship.vel);
    inp[3] = speed / PHYSICS.MAX_SPEED;

    // Velocity direction in local frame (azimuth, elevation)
    if (speed > 1e-4) {
      Q.applyToVec3Into(_brainInvQ, ship.vel[0], ship.vel[1], ship.vel[2], _brainLocal);
      const lx = _brainLocal[0], ly = _brainLocal[1], lz = _brainLocal[2];
      const horizDist = Math.sqrt(lx * lx + lz * lz);
      inp[4] = Math.atan2(lx, lz) / Math.PI;             // vel azimuth
      inp[5] = Math.atan2(ly, horizDist) / (Math.PI / 2); // vel elevation
    }

    // ── Target (3) ──
    if (targetPos) {
      const delta = [targetPos[0] - ship.pos[0], targetPos[1] - ship.pos[1], targetPos[2] - ship.pos[2]];
      _worldToLocalPolar(ship.quat, _brainInvQ, delta, _brainLocal);
      inp[6] = _brainLocal[0]; inp[7] = _brainLocal[1]; inp[8] = _brainLocal[2];
    }

    // ── Closest 3 enemies (9) ──
    if (enemies && enemies.length > 0) {
      // Sort by distance (only need top 3)
      const sorted = [];
      for (let i = 0; i < enemies.length; i++) {
        if (!enemies[i].alive) continue;
        const d = V3.distanceTo(ship.pos, enemies[i].pos);
        sorted.push({ s: enemies[i], d });
      }
      sorted.sort((a, b) => a.d - b.d);
      const n = Math.min(3, sorted.length);
      for (let i = 0; i < n; i++) {
        const base = 9 + i * 3;
        const e = sorted[i].s;
        const delta = [e.pos[0] - ship.pos[0], e.pos[1] - ship.pos[1], e.pos[2] - ship.pos[2]];
        _worldToLocalPolar(ship.quat, _brainInvQ, delta, _brainLocal);
        inp[base] = _brainLocal[0]; inp[base + 1] = _brainLocal[1]; inp[base + 2] = _brainLocal[2];
      }
    }

    // ── Closest 3 friends (9) ──
    if (friends && friends.length > 0) {
      const sorted = [];
      for (let i = 0; i < friends.length; i++) {
        if (!friends[i].alive || friends[i] === ship) continue;
        const d = V3.distanceTo(ship.pos, friends[i].pos);
        sorted.push({ s: friends[i], d });
      }
      sorted.sort((a, b) => a.d - b.d);
      const n = Math.min(3, sorted.length);
      for (let i = 0; i < n; i++) {
        const base = 18 + i * 3;
        const f = sorted[i].s;
        const delta = [f.pos[0] - ship.pos[0], f.pos[1] - ship.pos[1], f.pos[2] - ship.pos[2]];
        _worldToLocalPolar(ship.quat, _brainInvQ, delta, _brainLocal);
        inp[base] = _brainLocal[0]; inp[base + 1] = _brainLocal[1]; inp[base + 2] = _brainLocal[2];
      }
    }

    // ── Closest 4 asteroids (16) — includes diameter ──
    if (asteroids && asteroids.length > 0) {
      const sorted = [];
      for (let i = 0; i < asteroids.length; i++) {
        const d = V3.distanceTo(ship.pos, asteroids[i].pos);
        sorted.push({ a: asteroids[i], d });
      }
      sorted.sort((a, b) => a.d - b.d);
      const n = Math.min(4, sorted.length);
      for (let i = 0; i < n; i++) {
        const base = 27 + i * 4;
        const a = sorted[i].a;
        const delta = [a.pos[0] - ship.pos[0], a.pos[1] - ship.pos[1], a.pos[2] - ship.pos[2]];
        _worldToLocalPolar(ship.quat, _brainInvQ, delta, _brainLocal);
        inp[base] = _brainLocal[0]; inp[base + 1] = _brainLocal[1]; inp[base + 2] = _brainLocal[2];
        inp[base + 3] = 1 / (1 + (a.radius * 2) / 100); // diameter normalization
      }
    }

    return inp;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Apply brain outputs — low-level controller (BRAIN.md spec)
  // ══════════════════════════════════════════════════════════════
  // outputs[0]: azimuth (-1→1 mapped to -π→π) — desired facing direction in ship-local
  // outputs[1]: elevation (-1→1 mapped to -π/2→π/2) — desired facing direction in ship-local
  // outputs[2]: desired speed (tanh remapped to 0–1, then × MAX_SPEED)
  // outputs[3]: fuel spend rate (tanh remapped to 0–1)
  // outputs[4]: fire (>0 = shoot forward cannon)
  const _brainTargetDir = [0, 0, 0];
  const _brainCross = [0, 0, 0];

  function applyBrainOutputs(ship, outputs, enemies) {
    const hasFuel = ship.fuel > 0;

    // ── Decode outputs ──
    const azimuth = outputs[0] * Math.PI;               // -π to π in local frame
    const elevation = outputs[1] * (Math.PI / 2);       // -π/2 to π/2 in local frame
    const desiredSpeed = (outputs[2] + 1) / 2 * PHYSICS.MAX_SPEED;  // tanh → 0–1 → 0–MAX_SPEED
    const fuelSpend = (outputs[3] + 1) / 2;             // tanh → 0–1
    const fireCmd = outputs[4];

    // ── 1. Compute desired direction in local frame from azimuth/elevation ──
    // Local frame: +Z = forward, +X = right, +Y = up
    const cosEl = Math.cos(elevation);
    _brainTargetDir[0] = Math.sin(azimuth) * cosEl;   // local X
    _brainTargetDir[1] = Math.sin(elevation);          // local Y
    _brainTargetDir[2] = Math.cos(azimuth) * cosEl;   // local Z (forward)

    // ── 2. Turn toward desired direction ──
    // Current forward is [0,0,1] in local frame.
    // Angular error = cross(forward, targetDir) gives rotation axis, asin(|cross|) gives angle
    // cross([0,0,1], targetDir) = [-targetDir.y, targetDir.x, 0]
    const crossX = -_brainTargetDir[1];
    const crossY = _brainTargetDir[0];
    // crossZ = 0 (forward × something in forward plane)
    const sinAngle = Math.sqrt(crossX * crossX + crossY * crossY);
    const dotFwd = _brainTargetDir[2]; // dot([0,0,1], targetDir) = targetDir.z

    if (sinAngle > 1e-6 && hasFuel) {
      const angle = Math.atan2(sinAngle, dotFwd); // always positive

      // Proportional torque: scale angular velocity by error, cap at MAX_ANG_SPEED
      // Use sqrt for gentler response near target to reduce oscillation
      const angSpeedTarget = Math.min(angle * 0.3, PHYSICS.MAX_ANG_SPEED);

      // Rotation axis in local frame (normalized): [crossX/sinAngle, crossY/sinAngle, 0]
      const axisLocalX = crossX / sinAngle;
      const axisLocalY = crossY / sinAngle;

      // Convert rotation axis from local frame to world frame
      const axisWorld = Q.applyToVec3(ship.quat, [axisLocalX, axisLocalY, 0]);

      // Set angular velocity directly (controller replaces, doesn't accumulate)
      ship.angVel[0] = axisWorld[0] * angSpeedTarget;
      ship.angVel[1] = axisWorld[1] * angSpeedTarget;
      ship.angVel[2] = axisWorld[2] * angSpeedTarget;

      // Torque fuel cost (proportional to how hard we're turning)
      ship.fuel = Math.max(0, ship.fuel - PHYSICS.TORQUE_FUEL_COST * (angSpeedTarget / PHYSICS.MAX_ANG_SPEED));
    } else if (sinAngle <= 1e-6) {
      // Already facing target direction — kill angular velocity
      ship.angVel[0] = 0; ship.angVel[1] = 0; ship.angVel[2] = 0;
    }

    // ── 3. Thrust: match desired speed using fuel_spend as aggressiveness ──
    const currentSpeed = V3.length(ship.vel);
    const speedError = desiredSpeed - currentSpeed;

    if (fuelSpend > 0.01 && hasFuel) {
      // thrust = speedError * fuelSpend, clamped to engine limits
      const thrustMag = Math.max(-PHYSICS.THRUST, Math.min(PHYSICS.THRUST, speedError * fuelSpend));

      if (Math.abs(thrustMag) > 1e-6) {
        // Get forward direction in world frame
        Q.applyToVec3Into(ship.quat, 0, 0, 1, _brainFwd);
        ship.vel[0] += _brainFwd[0] * thrustMag;
        ship.vel[1] += _brainFwd[1] * thrustMag;
        ship.vel[2] += _brainFwd[2] * thrustMag;

        // Speed cap
        const newSpeed = V3.length(ship.vel);
        if (newSpeed > PHYSICS.MAX_SPEED) {
          V3.scaleMut(ship.vel, PHYSICS.MAX_SPEED / newSpeed);
        }

        // Fuel cost proportional to actual burn
        ship.fuel = Math.max(0, ship.fuel - PHYSICS.FUEL_COST_PER_FRAME * Math.abs(thrustMag) / PHYSICS.THRUST);
        ship.isAccelerating = thrustMag > 0;
        ship.isBraking = thrustMag < 0;
      } else {
        ship.isAccelerating = false;
        ship.isBraking = false;
      }
    } else {
      // Coast — no fuel used
      ship.isAccelerating = false;
      ship.isBraking = false;
    }

    // ── 4. Forward cannon ──
    let firedAt = null;
    if (fireCmd > 0 && ship.battery >= PHYSICS.WEAPON_COST && enemies && enemies.length > 0) {
      Q.applyToVec3Into(ship.quat, 0, 0, 1, _brainFwd);
      let bestTarget = null;
      let bestDist = PHYSICS.WEAPON_RANGE;
      for (let i = 0; i < enemies.length; i++) {
        const e = enemies[i];
        if (!e.alive) continue;
        const dx = e.pos[0] - ship.pos[0];
        const dy = e.pos[1] - ship.pos[1];
        const dz = e.pos[2] - ship.pos[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > PHYSICS.WEAPON_RANGE || dist < 1e-6) continue;
        // Check cone: dot(forward, dirToEnemy) > cos(5°)
        const dot = (_brainFwd[0] * dx + _brainFwd[1] * dy + _brainFwd[2] * dz) / dist;
        if (dot > PHYSICS.WEAPON_CONE_COS && dist < bestDist) {
          bestTarget = e;
          bestDist = dist;
        }
      }
      if (bestTarget) {
        ship.battery = Math.max(0, ship.battery - PHYSICS.WEAPON_COST);
        if (bestTarget.battery > 0) {
          bestTarget.battery = Math.max(0, bestTarget.battery - PHYSICS.LASER_DAMAGE);
          bestTarget.shieldFlash = 15;
        } else {
          bestTarget.hp -= PHYSICS.LASER_DAMAGE;
          ship.neuralHullDamageDealt = (ship.neuralHullDamageDealt || 0) + PHYSICS.LASER_DAMAGE;
        }
        ship.neuralDamageDealt = (ship.neuralDamageDealt || 0) + PHYSICS.LASER_DAMAGE;
        bestTarget.neuralDamageTaken = (bestTarget.neuralDamageTaken || 0) + PHYSICS.LASER_DAMAGE;
        if (bestTarget.hp <= 0) bestTarget.alive = false;
        firedAt = bestTarget;
      }
    }

    return firedAt;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Create asteroid with mass (for training compatibility)
  // ══════════════════════════════════════════════════════════════
  function createAsteroidWithMass(pos, radius) {
    // Mass from volume assuming sphere + ROCK_DENSITY
    // radius is in game units (1gu = 100m), so radius_m = radius * 100
    const radiusM = radius * 100;
    const volumeM3 = (4 / 3) * Math.PI * radiusM * radiusM * radiusM;
    const massTonnes = volumeM3 * PHYSICS.ROCK_DENSITY / 1000;
    return { pos: V3.clone(pos), radius, mass: massTonnes };
  }

  // ══════════════════════════════════════════════════════════════
  // ██  PD Steering Controller — analytical steering toward a target
  // ══════════════════════════════════════════════════════════════
  // Returns [pitch, yaw, burn] in [-1, 1].
  // Sign convention: +pitch = nose down, +yaw = nose right,
  //                  +burn = thrust forward, -burn = thrust backward.
  const _steerInvQ = [0, 0, 0, 1];
  const _steerLocal = [0, 0, 0];
  const _steerLocalAngVel = [0, 0, 0];

  const _steerVelLocal = [0, 0, 0];

  function steerToward(ship, targetPos) {
    // 1. Compute target position in ship-local frame
    const rx = targetPos[0] - ship.pos[0];
    const ry = targetPos[1] - ship.pos[1];
    const rz = targetPos[2] - ship.pos[2];
    Q.invertInto(ship.quat, _steerInvQ);
    Q.applyToVec3Into(_steerInvQ, rx, ry, rz, _steerLocal);
    const lx = _steerLocal[0], ly = _steerLocal[1], lz = _steerLocal[2];

    // 2. Angular error: atan2 gives angle from forward (+Z) axis
    const distXZ = Math.sqrt(lx * lx + lz * lz);
    const dist = Math.sqrt(lx * lx + ly * ly + lz * lz);
    const yawError = Math.atan2(lx, lz);       // +X => positive yaw
    const pitchError = Math.atan2(-ly, distXZ); // +Y target => negative pitch

    // 3. Angular velocity in local frame (derivative term)
    Q.applyToVec3Into(_steerInvQ, ship.angVel[0], ship.angVel[1], ship.angVel[2], _steerLocalAngVel);
    const pitchRate = _steerLocalAngVel[0];
    const yawRate = _steerLocalAngVel[1];

    // 4. PD control for rotation
    const Kp = 4.0;
    const Kd = 8.0;
    let pitchCmd = Math.max(-1, Math.min(1, Kp * pitchError - Kd * pitchRate));
    let yawCmd   = Math.max(-1, Math.min(1, Kp * yawError   - Kd * yawRate));

    // 5. Burn control — velocity-error approach
    //    Compute ideal velocity toward target, then correct the error.
    //    This naturally handles approach, braking, AND lateral drift.

    const speed = V3.length(ship.vel);

    // Ideal velocity: point toward target at safe approach speed
    // Safe speed ramps down as we get close: v = sqrt(2*a*d) * safety
    const safeSpeed = dist < 0.5 ? 0
      : Math.sqrt(2 * PHYSICS.THRUST * Math.max(0, dist - 0.5)) * 0.35;
    const cappedSafe = Math.min(safeSpeed, PHYSICS.MAX_SPEED * 0.8);

    // Ideal velocity vector (world frame): toward target at safe speed
    let idealVx = 0, idealVy = 0, idealVz = 0;
    if (dist > 0.5) {
      const s = cappedSafe / dist;
      idealVx = rx * s;
      idealVy = ry * s;
      idealVz = rz * s;
    }

    // Velocity error (world frame): what we need to subtract from current vel
    const errVx = ship.vel[0] - idealVx;
    const errVy = ship.vel[1] - idealVy;
    const errVz = ship.vel[2] - idealVz;
    const errSpeed = Math.sqrt(errVx * errVx + errVy * errVy + errVz * errVz);

    // 5b. Blended correction: deflect aim by velocity error to kill lateral drift
    //     No hard mode switch — smooth blend avoids oscillation.
    let burnCmd = 0;

    if (errSpeed > 0.005) {
      // Transform velocity error into ship-local frame
      Q.applyToVec3Into(_steerInvQ, errVx, errVy, errVz, _steerVelLocal);
      const errLocalX = _steerVelLocal[0];
      const errLocalY = _steerVelLocal[1];
      const errLocalZ = _steerVelLocal[2];

      // Deflect aim direction by lateral velocity error:
      //   drift right (errLocalX > 0) → aim left (subtract from yaw)
      //   drift up    (errLocalY > 0) → aim down (add to pitch, since +pitch = nose down)
      const corrGain = 3.0;
      const corrYaw   = yawError   - corrGain * errLocalX;
      const corrPitch = pitchError + corrGain * errLocalY;
      pitchCmd = Math.max(-1, Math.min(1, Kp * corrPitch - Kd * pitchRate));
      yawCmd   = Math.max(-1, Math.min(1, Kp * corrYaw   - Kd * yawRate));

      // Burn based on forward component of velocity error
      // errLocalZ > 0 → going too fast forward → brake (negative burn)
      // errLocalZ < 0 → going too slow → thrust (positive burn)
      burnCmd = Math.max(-1, Math.min(1, -errLocalZ * 10));
    }
    // else: errSpeed < 0.005 → velocity ~= ideal, face target, no burn
    //       (pitchCmd/yawCmd already steer toward target from step 4)

    return [pitchCmd, yawCmd, burnCmd];
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Apply NN outputs to ship — returns { firedAt: ship|null }
  // ══════════════════════════════════════════════════════════════
  const _enemyBuf = [];   // reused buffer for enemy list in applyNNOutputs
  const _firedResult = { firedAt: null };  // reused return object

  function applyNNOutputs(ship, outputs, allShips) {
    const hasFuel = ship.fuel > 0;
    _firedResult.firedAt = null;

    // outputs[0]: pitch (nose up/down), outputs[1]: yaw (nose left/right)
    // Computed in ship-local frame then applied as world-space angular velocity
    if (hasFuel) {
      const pitch = Math.abs(outputs[0]) > 0.1 ? outputs[0] : 0;
      const yaw   = Math.abs(outputs[1]) > 0.1 ? outputs[1] : 0;
      if (pitch !== 0 || yaw !== 0) {
        Q.applyToVec3Into(ship.quat, 1, 0, 0, _rightTmp);  // local right
        Q.applyToVec3Into(ship.quat, 0, 1, 0, _upTmp);     // local up
        // pitch = rotate around right axis, yaw = rotate around up axis
        ship.angVel[0] += (_rightTmp[0] * pitch + _upTmp[0] * yaw) * PHYSICS.TORQUE;
        ship.angVel[1] += (_rightTmp[1] * pitch + _upTmp[1] * yaw) * PHYSICS.TORQUE;
        ship.angVel[2] += (_rightTmp[2] * pitch + _upTmp[2] * yaw) * PHYSICS.TORQUE;
        const angSpeed = V3.length(ship.angVel);
        if (angSpeed > PHYSICS.MAX_ANG_SPEED) {
          V3.scaleMut(ship.angVel, PHYSICS.MAX_ANG_SPEED / angSpeed);
        }
        ship.fuel = Math.max(0, ship.fuel - PHYSICS.TORQUE_FUEL_COST);
      }
    }

    // outputs[2]: engine burn along forward axis
    const burn = outputs[2];
    if (Math.abs(burn) > 0.1 && hasFuel) {
      Q.applyToVec3Into(ship.quat, 0, 0, 1, _fwdTmp);
      ship.vel[0] += _fwdTmp[0] * burn * PHYSICS.THRUST;
      ship.vel[1] += _fwdTmp[1] * burn * PHYSICS.THRUST;
      ship.vel[2] += _fwdTmp[2] * burn * PHYSICS.THRUST;
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

    // outputs[3]: target selection, outputs[4]: fire
    let nEnemies = 0;
    for (let i = 0; i < allShips.length; i++) {
      const s = allShips[i];
      if (s !== ship && s.alive && s.team !== ship.team) {
        if (nEnemies < _enemyBuf.length) _enemyBuf[nEnemies] = s;
        else _enemyBuf.push(s);
        nEnemies++;
      }
    }
    // Sort only populated portion by distance
    for (let i = 0; i < nEnemies - 1; i++) {
      for (let j = i + 1; j < nEnemies; j++) {
        const di = V3.distanceTo(ship.pos, _enemyBuf[i].pos);
        const dj = V3.distanceTo(ship.pos, _enemyBuf[j].pos);
        if (dj < di) { const tmp = _enemyBuf[i]; _enemyBuf[i] = _enemyBuf[j]; _enemyBuf[j] = tmp; }
      }
    }

    if (nEnemies > 0) {
      const idx = Math.min(
        Math.floor((outputs[3] + 1) / 2 * nEnemies),
        nEnemies - 1
      );
      const target = _enemyBuf[Math.max(0, idx)];

      // Fire
      if (outputs[4] > 0 && ship.battery >= PHYSICS.WEAPON_COST) {
        const dist = V3.distanceTo(ship.pos, target.pos);
        if (dist < PHYSICS.WEAPON_RANGE) {
          // Deduct battery from shooter
          ship.battery = Math.max(0, ship.battery - PHYSICS.WEAPON_COST);

          // Apply damage to target
          if (target.battery > 0) {
            target.battery = Math.max(0, target.battery - PHYSICS.LASER_DAMAGE);
            target.shieldFlash = 15;
          } else {
            target.hp -= PHYSICS.LASER_DAMAGE;
            ship.neuralHullDamageDealt += PHYSICS.LASER_DAMAGE;
          }

          // Track damage
          ship.neuralDamageDealt += PHYSICS.LASER_DAMAGE;
          target.neuralDamageTaken += PHYSICS.LASER_DAMAGE;

          // Kill check
          if (target.hp <= 0) {
            target.alive = false;
          }

          _firedResult.firedAt = target;
        }
      }
    }

    return _firedResult;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Score a completed match — returns { alpha: number, omega: number }
  // ══════════════════════════════════════════════════════════════
  // Ships must track: _framesInRange, _distanceClosed, _shotsFired (set by runMatch)
  function scoreMatch(allShips, matchFrames) {
    const teams = { alpha: { ships: [], alive: [] }, omega: { ships: [], alive: [] } };
    for (const s of allShips) {
      teams[s.team].ships.push(s);
      if (s.alive) teams[s.team].alive.push(s);
    }
    const totalFrames = matchFrames || 1800;

    function scoreTeam(teamKey, enemyKey) {
      let score = 0;
      const myShips = teams[teamKey].ships;
      const myAlive = teams[teamKey].alive;
      const enemyShips = teams[enemyKey].ships;
      const enemyAlive = teams[enemyKey].alive;

      // ── Damage dealt: base + hull bonus ──
      for (const s of myShips) score += s.neuralDamageDealt * 2;
      for (const s of myShips) score += (s.neuralHullDamageDealt || 0) * 5;

      // ── Focus fire: bonus for damage to lowest-HP enemy ──
      for (const s of myShips) score += (s._focusDamage || 0) * 15;

      // ── Kills: massive bonus — primary objective ──
      const enemiesKilled = enemyShips.length - enemyAlive.length;
      score += enemiesKilled * 3000;

      // ── Full wipe bonus ──
      if (enemyAlive.length === 0) score += 2000;

      // ── Team outcome ──
      const myKilled = myShips.length - myAlive.length;
      if (enemiesKilled > myKilled) score += 1000;
      else if (enemiesKilled < myKilled) score -= 500;

      // ── Pursuit: reward closing distance ──
      for (const s of myShips) score += (s._distanceClosed || 0) * 3;

      // ── Time in weapon range ──
      for (const s of myShips) score += (s._framesInRange || 0) * 1.0;

      // ── Shots fired ──
      for (const s of myShips) score += (s._shotsFired || 0) * 10;

      // ── Final proximity ──
      if (myAlive.length > 0 && enemyAlive.length > 0) {
        let totalDist = 0;
        for (const s of myAlive) {
          let minDist = Infinity;
          for (const e of enemyAlive) minDist = Math.min(minDist, V3.distanceTo(s.pos, e.pos));
          totalDist += minDist;
        }
        score -= (totalDist / myAlive.length) * 5;
      }

      // ── Survival bonus ──
      score += myAlive.length * 100;

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
    const asteroids = config.asteroids || [];

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
        // Per-frame tracking for scoring
        s._framesInRange = 0;
        s._distanceClosed = 0;
        s._shotsFired = 0;
        s._prevMinDist = Infinity;
        allShips.push(s);
      }
    }

    // Simulate
    let frameCount = 0;
    for (let frame = 0; frame < matchTimeLimit; frame++) {
      frameCount++;
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
      }
    }

    allShips._matchFrames = frameCount;
    return allShips;
  }

  // ══════════════════════════════════════════════════════════════
  // ██  Export
  // ══════════════════════════════════════════════════════════════
  const SimCore = {
    V3, Q, PHYSICS,
    NeuralNetwork, ReLUNetwork,
    createShipState,
    createAsteroid,
    createAsteroidWithMass,
    checkAsteroidCollisions,
    shipSimStep,
    applyGravity,
    buildNNInputs,
    buildBrainInputs,
    applyNNOutputs,
    applyBrainOutputs,
    steerToward,
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
