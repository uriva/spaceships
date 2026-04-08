# Ship Brain — Design Document

## Overview

One neural network architecture for all ship behaviors. The brain outputs
high-level intentions (desired direction, speed, aggression, fire). A
deterministic low-level controller translates intentions into thrust and torque.

Behaviors (follow, attack, defend, conserve) are not separate brains — they
emerge from training on diverse scenarios with different objectives.

## Outputs (5)

| # | Name | Range | Meaning |
|---|------|-------|---------|
| 0 | azimuth | -1 to 1 | Desired facing direction, horizontal. Mapped to -π to π. |
| 1 | elevation | -1 to 1 | Desired facing direction, vertical. Mapped to -π/2 to π/2. |
| 2 | speed | 0 to 1 | Desired speed as fraction of MAX_SPEED. (tanh → remap to 0–1) |
| 3 | fuel spend | 0 to 1 | Acceleration aggressiveness. 0 = coast/gentle, 1 = max burn. |
| 4 | fire | >0 = shoot | Forward cannon fires along ship's facing direction. |

The azimuth/elevation define a **world-space direction** the ship should face.
The low-level controller turns the ship toward it and applies thrust to reach
the desired speed at the given fuel spend rate.

### Low-level controller

Each frame:
1. Compute target quaternion from (azimuth, elevation).
2. Compute angular error between current orientation and target.
3. Apply torque to reduce error (proportional to error magnitude, capped at
   MAX_ANG_SPEED). This is deterministic — no NN needed.
4. Compute thrust: `thrust = (desired_speed - current_speed) * fuel_spend`.
   Positive = accelerate along forward, negative = brake. Clamp to engine
   limits. Only burn fuel when thrusting.

This means:
- fuel_spend = 0 → coast, no fuel used regardless of speed mismatch.
- fuel_spend = 1 → match desired speed as fast as the engine allows.
- Turning always costs torque fuel (TORQUE_FUEL_COST) but is handled by the
  controller, not the brain.

## Inputs (43)

All entity positions are in **ship-local polar coordinates**: azimuth angle,
elevation angle, distance. This is rotation-invariant — the brain doesn't need
to know its absolute orientation.

### Own state (6)

| # | Input | Normalization |
|---|-------|---------------|
| 0 | hull | hp / maxHp |
| 1 | energy | battery / maxBattery |
| 2 | fuel | fuel / maxFuel |
| 3 | speed | speed / MAX_SPEED |
| 4 | velocity azimuth | angle / π (−1 to 1) |
| 5 | velocity elevation | angle / (π/2) (−1 to 1) |

Velocity direction is the azimuth/elevation of the current velocity vector in
ship-local frame. Tells the brain which way it's drifting.

### Target (3)

| # | Input | Normalization |
|---|-------|---------------|
| 6 | azimuth | angle / π |
| 7 | elevation | angle / (π/2) |
| 8 | distance | 1 / (1 + dist/50) — asymptotic, never saturates |

Distance uses inverse normalization so it's always in (0, 1]. Close targets →
near 1, far targets → near 0. No saturation at any distance.

### Closest 3 enemies (9)

For each enemy (i = 0, 1, 2):

| # | Input | Normalization |
|---|-------|---------------|
| 9+3i | azimuth | angle / π |
| 10+3i | elevation | angle / (π/2) |
| 11+3i | distance | 1 / (1 + dist/50) |

If fewer than 3 enemies exist, remaining slots are zeroed (distance=0 signals
"no entity").

### Closest 3 friends (9)

Same layout as enemies, indices 18–26.

### Closest 4 asteroids (16)

For each asteroid (i = 0, 1, 2, 3):

| # | Input | Normalization |
|---|-------|---------------|
| 27+4i | azimuth | angle / π |
| 28+4i | elevation | angle / (π/2) |
| 29+4i | distance | 1 / (1 + dist/50) |
| 30+4i | diameter | 1 / (1 + diameter/100) |

Total: 6 + 3 + 9 + 9 + 16 = **43 inputs**.

## Weapon — forward cannon

Single fixed laser cannon along the ship's forward axis (+Z local).

- **Cone**: 5° half-angle from forward vector.
- **Range**: WEAPON_RANGE (200 units).
- **Cost**: WEAPON_COST (8 energy) per shot.
- **Damage**: LASER_DAMAGE (10) to battery first, then hull.
- **Hit logic**: When fire output > 0 and battery ≥ WEAPON_COST, raycast forward.
  Hit the closest enemy within the 5° cone and range.

Tactical consequence: to shoot, you must face the enemy. To travel, you must
face your destination. You can't do both unless they align.

## Network architecture

- **Topology**: [43, 32, 32, 5] — ReLU hidden layers, tanh output layer.
- **Parameters**: ~(43×32+32) + (32×32+32) + (32×5+5) = 1408 + 1056 + 165 = **2629 params**.
- **Activation**: ReLU hidden, tanh output. Outputs remapped: speed and fuel
  spend from tanh (−1,1) → (0,1) via `(x+1)/2`.

## Training — CMA-ES

### Phase 1: Follow moving target in asteroid field

Start with one scenario type. Get it working, then add combat/evasion later.

**Scenario setup:**
- 3–8 asteroids with randomized positions, sizes, and gravity.
- Target starts 10–300 units away, random direction.
- Target **moves**: random velocity 0.01–0.1 units/frame, changes direction
  occasionally (every 200–600 frames) to simulate a moving crosshair or entity.
- Ship starts with random orientation and 0–50% of MAX_SPEED in a random
  direction.
- Fuel: 300–2500 (varied). Brain must learn to be efficient when low.
- Episode: 3000 frames (50 sec).

**Fitness:**
- **Death → 0 points.** Collide with asteroid = dead = zero fitness for the
  entire episode. This forces asteroid avoidance as a hard constraint.
- **Proximity reward**: each frame, accumulate `1 / (1 + dist)`. Closer =
  more points. Summed over all 3000 frames, so sustained proximity matters
  more than a brief flyby.
- **Fuel efficiency bonus**: `(fuel_remaining / fuel_initial) × F` where F is
  a tuning weight. Rewards finishing with fuel left. A ship that tracks the
  target AND conserves fuel scores higher than one that burns everything.

```
if dead:
  fitness = 0
else:
  proximity = Σ 1/(1 + dist)  over all frames
  fuel_bonus = (fuel_remaining / fuel_initial) × 50
  fitness = proximity + fuel_bonus
```

Simple, clean, no competing objectives. Proximity drives following, fuel bonus
drives efficiency, death zeroes everything so asteroid avoidance is mandatory.

### Trainer config

- **Algorithm**: sep-CMA-ES (diagonal covariance).
- **Population**: 200 (larger pop for 2629 params).
- **σ₀**: 0.5.
- **Episodes per eval**: 12 (4 short-range + 4 long-range + 4 low-fuel).
  All have asteroids and moving targets.
- **Convergence**: σ < 0.01 or fitness plateau for 500 generations.
