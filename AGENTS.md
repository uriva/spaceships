# Spaceships — 3D Space Fleet Command Game

## Overview

A browser-based 3D real-time strategy game where you command a fleet of spaceships. Select ships, issue movement/attack/patrol/orbit commands via right-click or natural language chat (backed by Gemini LLM). Built with Three.js (no build step) and a Deno backend.

## Architecture

### Frontend — `index.html`

The entire game is a single ~4200-line HTML file with inline CSS and JavaScript. No framework, no bundler. Key systems:

- **Rendering**: Three.js scene with starfield, spaceships (team-colored), asteroids (some massive with orange glow), engine particle effects. Engagement sphere drawn with manual lat/long lines (24 meridians + 12 parallels).
- **Physics engine** (`updateSpaceships()`): The single authority for ship movement. Applies gravitational acceleration from asteroids, then handles engine thrust (acceleration/braking toward `targetVelocity`). One engine per ship — braking is reverse thrust at the same power and fuel cost as acceleration. Fuel is consumed only by engine thrust. Position is updated from velocity each frame.
- **Behavior engine** (`updateShipBehaviors()`): Runs each frame before physics. Each ship has a `behavior` object (type: move/brake/orbit/patrol/follow/attack). Behaviors compute the desired `targetVelocity` and set `isAccelerating`/`isBraking` flags. They never directly modify velocity or position — the physics engine does that.
- **Gravity**: Real G constant `G_REAL = 6.674e-11` converted to game units, with tunable `gravityMultiplier`. All asteroids have mass proportional to volume. ~1/8 are super-dense (200x, marked `isMassive`).
- **Selection**: Click to select, box-drag for multi-select, Ctrl+click to toggle. Control groups via Ctrl+1/2/3 (save) and 1/2/3 (recall).
- **3D Crosshair/Cursor**: Pole-free direction vector (`cursorDir` + `cursorUp` + `cursorRadius`). Two orbs: inner orb at cursor position, surface orb on sphere surface. Three colored beams (X=red, Z=green, Y=blue) as full cross lines, with 6 axis endpoint orbs where beams hit the sphere. Shift+WASD moves cursor on sphere, Shift+Z/X controls radius.
- **Camera**: WASD = up/down/left/right fly, Z/X = forward/back, Space+WASD = look rotation (pitch/yaw using local up vector, pole-free), L = focus on selected. Movement accelerates on sustained key press (0.8 → 6.0 over 90 frames). Constrained to boundary sphere.
- **HUD**: Selection info, fuel display, ship group dots (1/2/3), asteroid mass display (orange for massive).
- **Chat**: Text input sends natural language to backend, which returns DSL commands executed by `executeDSL()`. AI context sends `Player at [x,y,z]` (crosshair position only, no camera position).

### Backend — `backend/main.ts`

Deno server that proxies natural language commands to Google Gemini LLM. The system prompt defines a DSL (MOVE, STOP, ORBIT, PATROL, FOLLOW, ATTACK, STATUS) and instructs Gemini to translate casual English (including slang, typos, informal references like "boys" or "guys") into structured commands. All DSL positions must be numeric `[x,y,z]`.

Run with: `deno run --allow-net --allow-env backend/main.ts`

Requires `GEMINI_API_KEY` environment variable.

### Backend config — `backend/deno.json`

Deno workspace config with import map.

## Key design principles

1. **Physics is law**: All movement goes through the physics engine. Behaviors set intentions (targetVelocity, flags), physics applies forces. No teleporting, no instant velocity changes. No artificial drag.
2. **Fuel matters**: Engine thrust costs fuel (`FUEL_COST_PER_FRAME = 0.4t`). Gravity is free. Ships near massive asteroids can orbit without fuel expenditure.
3. **One engine, one thrust**: Braking is reverse thrust — same power (`THRUST = 0.015`), same fuel cost as forward acceleration. No separate deceleration constant.
4. **No pole singularities**: Both crosshair and camera look rotation use local up vectors (`cursorUp`, `cameraUpVector`) and `applyAxisAngle` rotation. No reference to world Y for computing rotation axes. Motion passes smoothly through poles.
5. **AI knows the game**: AI context includes all ships, asteroids, and player crosshair position. The crosshair IS where the player "is."

## Controls

| Input | Action |
|-------|--------|
| WASD | Fly camera: W=up, S=down, A=left, D=right |
| Z/X | Fly camera: Z=forward, X=back |
| Space+WASD | Look rotation: W/S=pitch, A/D=yaw (no position change) |
| Shift+WASD | Move crosshair on sphere surface |
| Shift+Z/X | Crosshair radius in/out |
| L | Focus camera on selected object |
| Click | Select ship/asteroid |
| Box drag | Multi-select |
| Ctrl+click | Toggle selection |
| Ctrl+1-9 | Save control group |
| 1-9 | Recall control group |
| Right-click | Move selected ships to point |

## Important code locations (index.html)

| What | Approx. lines |
|------|---------------|
| Physics constants (THRUST, FUEL_COST_PER_FRAME, G_REAL) | ~605-615 |
| Cursor variables (cursorDir, cursorUp, cursorRadius) | ~636-641 |
| Camera speed constants (MIN/MAX, ACCEL_FRAMES) | ~700-706 |
| init() — beams, orbs, scene setup | ~786-840 |
| createBoundarySphere() — manual lat/long lines | ~1990-2020 |
| updateCameraPosition() | ~2040-2045 |
| updateKeyboardControls() — crosshair + camera + fly | ~3216-3316 |
| updateCursorBeams() — beam/orb positioning | ~3385-3455 |
| buildContext() — AI context with player position | ~4060-4075 |
| Selection state, controlGroups | ~644-648 |
| Ship userData init (velocity, fuel, mass, behavior) | ~1523-1543 |
| Asteroid userData init (mass, isMassive, glow) | ~1707-1733 |
| Behavior engine (updateShipBehaviors) | ~3118-3242 |
| executeDSL (chat command dispatch) | ~3058-3113 |
| HUD updates (fuel, ship dots, asteroid info) | ~2540-2660 |
