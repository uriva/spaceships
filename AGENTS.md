# Spaceships — 3D Space Fleet Command Game

## Overview

A browser-based 3D real-time strategy game where you command a fleet of
spaceships. Select ships, issue movement/attack/patrol/orbit commands via
right-click or natural language chat (backed by Gemini LLM). Built with Three.js
(no build step) and a Deno backend.

## Architecture

### Frontend — `index.html`

The entire game is a single ~4600-line HTML file with inline CSS and JavaScript.
No framework, no bundler. Key systems:

- **Rendering**: Three.js scene with starfield, spaceships (team-colored),
  asteroids (displaced icosahedrons with multi-octave terrain), engine particle
  effects. Engagement sphere drawn with manual lat/long lines (24 meridians + 12
  parallels).
- **Physics engine** (`updateSpaceships()`): The single authority for ship
  movement. Applies gravitational acceleration from asteroids, then handles
  engine thrust (acceleration/braking toward `targetVelocity`). One engine per
  ship — braking is reverse thrust at the same power and fuel cost as
  acceleration. Fuel is consumed only by engine thrust. Position is updated from
  velocity each frame.
- **Behavior engine** (`updateShipBehaviors()`): Runs each frame before physics.
  Each ship has a `behavior` object (type:
  move/brake/orbit/patrol/follow/attack). Behaviors compute the desired
  `targetVelocity` and set `isAccelerating`/`isBraking` flags. They never
  directly modify velocity or position — the physics engine does that.
- **Gravity**: Real G constant `G_REAL = 6.674e-11` converted to game units,
  with tunable `gravityMultiplier`. All asteroids have mass proportional to
  volume — no special-cased dense asteroids. Power-law size distribution
  naturally produces rare massive bodies.
- **Asteroids**: All use displaced icosahedrons with multi-octave terrain
  (ridges, craters, coarse + fine noise), unique seed per asteroid. Power-law
  size distribution: `scale = 15 / Math.pow(u + 0.02, 1/3)` clamped to [15,
  8000]. No special-cased asteroids — mass comes from volume naturally. HUD
  shows orange mass color for asteroids >= 1 Gt. Low-fuel ships auto-seek orbit
  around nearest asteroid with mass >= 1e9 tonnes.
- **Selection**: Click to select, box-drag for multi-select, Ctrl+click to
  toggle. Control groups via Ctrl+1/2/3 (save) and 1/2/3 (recall). Tab cycles
  ships only (not asteroids).
- **3D Crosshair/Cursor**: Pole-free direction vector (`cursorDir` +
  `cursorUp` + `cursorRadius`). Two orbs: inner orb at cursor position, surface
  orb on sphere surface. Three colored beams (X=red, Z=green, Y=blue) as full
  cross lines, with 6 axis endpoint orbs where beams hit the sphere. Shift+WASD
  moves cursor on sphere, Shift+Z/X controls radius.
- **Camera**: Orbit mode (default) revolves around `camOrbitCenter` using
  pole-free `camSphereDir`/`camSphereUp`/`camSphereRadius`. Alt+WASD = strafe
  fly. Space+WASD = look rotation. Camera only updates when keys are pressed —
  idle frames don't overwrite state. L = focus on selected (double-tap to zoom),
  C = focus on crosshair (double-tap to zoom).
- **HUD**: Selection info, fuel display, ship group dots (1/2/3), asteroid mass
  display (orange for >= 1 Gt).
- **Chat**: Text input (press `/` to focus) sends natural language to backend,
  which returns DSL commands executed by `executeDSL()`. AI context sends
  `Player at [x,y,z]` (crosshair position only, no camera position). Game keys
  are disabled while typing in chat input.

### Backend — `backend/main.ts`

Deno server that proxies natural language commands to Google Gemini LLM. The
system prompt defines a DSL (MOVE, STOP, ORBIT, PATROL, FOLLOW, ATTACK, STATUS)
and instructs Gemini to translate casual English (including slang, typos,
informal references like "boys" or "guys") into structured commands. All DSL
positions must be numeric `[x,y,z]`.

Run with: `deno run --allow-net --allow-env backend/main.ts`

Requires `GEMINI_API_KEY` environment variable.

### Backend config — `backend/deno.json`

Deno workspace config with import map.

## Key design principles

1. **Physics is law**: All movement goes through the physics engine. Behaviors
   set intentions (targetVelocity, flags), physics applies forces. No
   teleporting, no instant velocity changes. No artificial drag.
2. **Fuel matters**: Engine thrust costs fuel (`FUEL_COST_PER_FRAME = 0.4t`).
   Gravity is free. Ships near heavy asteroids can orbit without fuel
   expenditure.
3. **One engine, one thrust**: Braking is reverse thrust — same power
   (`THRUST = 0.015`), same fuel cost as forward acceleration. No separate
   deceleration constant.
4. **No pole singularities**: Both crosshair and camera look rotation use local
   up vectors (`cursorUp`, `cameraUpVector`) and `applyAxisAngle` rotation. No
   reference to world Y for computing rotation axes. Motion passes smoothly
   through poles.
5. **AI knows the game**: AI context includes all ships, asteroids, and player
   crosshair position. The crosshair IS where the player "is."
6. **No special-cased objects**: All asteroids come from the same distribution.
   No hardcoded sizes, positions, or masses. Rare giants emerge naturally from
   the power-law distribution.

## Controls

| Input       | Action                                                   |
| ----------- | -------------------------------------------------------- |
| WASD        | Orbit camera: W/S=pitch, A/D=yaw around `camOrbitCenter` |
| Z/X         | Orbit: radius in/out (ramps 2→40)                        |
| Alt+WASD    | Strafe: linear fly in camera-relative directions         |
| Space+WASD  | Look rotation: pitch/yaw (no position change)            |
| Shift+WASD  | Move crosshair on sphere surface                         |
| Shift+Z/X   | Crosshair radius in/out                                  |
| L           | Focus on selected (double-tap = zoom close)              |
| C           | Focus on crosshair (double-tap = zoom close)             |
| Tab         | Cycle through ships (Shift+Tab = reverse)                |
| /           | Focus chat input (Escape to blur)                        |
| Click       | Select ship/asteroid                                     |
| Box drag    | Multi-select                                             |
| Ctrl+click  | Toggle selection                                         |
| Ctrl+1-9    | Save control group                                       |
| 1-9         | Recall control group                                     |
| Right-click | Move selected ships to point                             |

## Important code locations (index.html)

| What                                                               | Approx. lines |
| ------------------------------------------------------------------ | ------------- |
| Physics constants (THRUST, FUEL_COST_PER_FRAME, G_REAL)            | ~605-615      |
| Cursor variables (cursorDir, cursorUp, cursorRadius)               | ~629          |
| Camera variables (camOrbitCenter, camSphereDir, etc.)              | ~678-692      |
| Camera accel constant (CAMERA_ACCEL_FRAMES = 45)                   | ~699          |
| init() — beams, orbs, scene setup                                  | ~786-840      |
| createAsteroids() — power-law distribution, displaced icosahedrons | ~1920-2080    |
| onKeyDown() — with chat input guard                                | ~2208         |
| onKeyUp() + clearAllKeys()                                         | ~2335-2365    |
| focusOnCrosshair()                                                 | ~2615         |
| focusOnSelected()                                                  | ~2635         |
| applyCameraTransition() — syncs camOrbitCenter                     | ~2655         |
| HUD updates (fuel, ship dots, asteroid info)                       | ~3130-3210    |
| Behavior engine (updateShipBehaviors)                              | ~3060-3180    |
| Space+WASD look-around                                             | ~3415         |
| Alt+WASD strafe                                                    | ~3445         |
| Orbit mode — camOrbitCenter, only when keys pressed                | ~3490         |
| buildContext() — AI context with player position                   | ~4450         |
