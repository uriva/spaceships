#!/usr/bin/env -S deno run --allow-read
// test-follow-brain.js — Diagnostic tests for trained follow brain
// Tests the specific failure cases from before + general scenarios

const simCoreSrc = Deno.readTextFileSync(new URL('./sim-core.js', import.meta.url).pathname);
(new Function(simCoreSrc))();
const {
  V3, Q, PHYSICS, ReLUNetwork,
  createShipState, shipSimStep, applyNNOutputs,
} = globalThis.SimCore;

const BRAIN_FILE = Deno.args[0] || 'follow-brain.json';
const brainData = JSON.parse(Deno.readTextFileSync(BRAIN_FILE));
const brain = new ReLUNetwork(brainData.topology);
brain.fromGenome(brainData.genome);

console.log(`Loaded brain: fitness=${brainData.fitness.toFixed(1)}, gen=${brainData.generation}, σ=${brainData.optimizer?.sigma?.toFixed(4)}`);
console.log(`Topology: ${brainData.topology.join('→')}, params: ${brain.paramCount}\n`);

const MATCH_FRAMES = 600; // 10s at 60fps
const _invQ = [0, 0, 0, 1];
const _localPos = [0, 0, 0];
const _localVel = [0, 0, 0];

function runTest(name, target, vel, quat) {
  const ship = createShipState('alpha');
  ship.fuel = PHYSICS.MAX_FUEL;

  if (quat) {
    ship.quat[0] = quat[0]; ship.quat[1] = quat[1];
    ship.quat[2] = quat[2]; ship.quat[3] = quat[3];
  } else {
    Q.setRandomMut(ship.quat);
  }
  ship.vel[0] = vel[0]; ship.vel[1] = vel[1]; ship.vel[2] = vel[2];

  const initDist = V3.distanceTo(ship.pos, target);
  let minDist = initDist;

  // Track trajectory at key frames
  const snapshots = [];
  
  for (let frame = 0; frame < MATCH_FRAMES; frame++) {
    const rx = target[0] - ship.pos[0];
    const ry = target[1] - ship.pos[1];
    const rz = target[2] - ship.pos[2];
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

    const dist = V3.distanceTo(ship.pos, target);
    if (dist < minDist) minDist = dist;

    if (frame === 59 || frame === 179 || frame === 359 || frame === 599) {
      const speed = V3.length(ship.vel);
      const angSpeed = V3.length(ship.angVel);
      snapshots.push({ t: ((frame + 1) / 60).toFixed(1), dist: dist.toFixed(2), speed: speed.toFixed(4), angSpeed: angSpeed.toFixed(4) });
    }
  }

  const finalDist = V3.distanceTo(ship.pos, target);
  const finalSpeed = V3.length(ship.vel);
  const finalAngSpeed = V3.length(ship.angVel);
  const fuelUsed = (PHYSICS.MAX_FUEL - ship.fuel).toFixed(1);

  // Pass criteria: within 2 units AND nearly stopped
  const closeEnough = finalDist < 2.0;
  const stopped = finalSpeed < 0.02;
  const pass = closeEnough && stopped;

  const status = pass ? 'PASS' : (closeEnough ? 'CLOSE' : 'FAIL');
  console.log(`[${status}] ${name}`);
  console.log(`  Init: dist=${initDist.toFixed(1)}, vel=[${vel.map(v=>v.toFixed(3)).join(',')}]`);
  console.log(`  Final: dist=${finalDist.toFixed(2)}, speed=${finalSpeed.toFixed(4)}, angSpeed=${finalAngSpeed.toFixed(4)}`);
  console.log(`  Min dist reached: ${minDist.toFixed(2)}, fuel used: ${fuelUsed}`);
  console.log(`  Timeline:`);
  for (const s of snapshots) {
    console.log(`    t=${s.t}s: dist=${s.dist}, speed=${s.speed}, angSpeed=${s.angSpeed}`);
  }
  console.log();

  return { name, pass, finalDist, finalSpeed, minDist };
}

console.log('═══════════════════════════════════════════════════');
console.log('  DIAGNOSTIC TESTS — Follow Brain');
console.log('═══════════════════════════════════════════════════\n');

const results = [];

// Tests that previously FAILED
console.log('── Previously-failing cases ──\n');
results.push(runTest('Off-axis: pure lateral [30,0,0]', [30, 0, 0], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Off-axis: above+right [20,20,0]', [20, 20, 0], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Lateral velocity [0.3,0,0] + forward target', [0, 0, 30], [0.3, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Close target [3,3,3]', [3, 3, 3], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Behind with lateral vel', [0, 0, -20], [0.15, 0, 0], [0, 0, 0, 1]));

// Tests that previously PASSED (regression check)
console.log('── Regression checks (should still pass) ──\n');
results.push(runTest('Forward target [0,0,30]', [0, 0, 30], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Forward close [0,0,10]', [0, 0, 10], [0, 0, 0], [0, 0, 0, 1]));

// New challenging scenarios
console.log('── Additional challenges ──\n');
results.push(runTest('Pure up [0,25,0]', [0, 25, 0], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Pure up + downward vel', [0, 25, 0], [0, -0.1, 0], [0, 0, 0, 1]));
results.push(runTest('Diagonal far [25,25,25]', [25, 25, 25], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Behind + fast toward [0,0,-10]', [0, 0, -10], [0, 0, -0.2], [0, 0, 0, 1]));
results.push(runTest('Random quat + diagonal far', [25, 25, 25], [0, 0, 0], null));
results.push(runTest('Random quat + lateral vel', [20, 0, 20], [0.2, 0.1, 0], null));
results.push(runTest('Very close [1,1,1]', [1, 1, 1], [0, 0, 0], [0, 0, 0, 1]));
results.push(runTest('Far [0,0,50]', [0, 0, 50], [0, 0, 0], [0, 0, 0, 1]));

// Summary
console.log('═══════════════════════════════════════════════════');
console.log('  SUMMARY');
console.log('═══════════════════════════════════════════════════');
const passed = results.filter(r => r.pass).length;
const close = results.filter(r => !r.pass && r.finalDist < 5).length;
const failed = results.filter(r => !r.pass && r.finalDist >= 5).length;
console.log(`  PASS: ${passed}/${results.length}`);
console.log(`  CLOSE (dist<5 but not stopped): ${close}`);
console.log(`  FAIL (dist>=5): ${failed}`);
console.log();

if (failed > 0) {
  console.log('  Failed tests:');
  for (const r of results.filter(r => !r.pass && r.finalDist >= 5)) {
    console.log(`    - ${r.name}: dist=${r.finalDist.toFixed(2)}, speed=${r.finalSpeed.toFixed(4)}`);
  }
}
