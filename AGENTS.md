# Spaceships — 3D Space Fleet Command Game

## Overview

A browser-based 3D real-time strategy game where you command a fleet of spaceships. Select ships, issue movement/attack/patrol/orbit commands via right-click or natural language chat (backed by Gemini LLM). Built with Three.js (no build step) and a Deno backend.

## Architecture

### Frontend — `index.html`

The entire game is a single ~3400-line HTML file with inline CSS and JavaScript. No framework, no bundler. Key systems:

- **Rendering**: Three.js scene with grid plane, starfield, spaceships (team-colored), asteroids (some massive with orange glow), engine particle effects.
- **Physics engine** (`updateSpaceships()`): The single authority for ship movement. Applies gravitational acceleration from asteroids, then handles engine thrust (acceleration/braking toward `targetVelocity`). One engine per ship — braking is reverse thrust at the same power and fuel cost as acceleration. Fuel is consumed only by engine thrust. Position is updated from velocity each frame.
- **Behavior engine** (`updateShipBehaviors()`): Runs each frame before physics. Each ship has a `behavior` object (type: move/brake/orbit/patrol/follow/attack). Behaviors compute the desired `targetVelocity` and set `isAccelerating`/`isBraking` flags. They never directly modify velocity or position — the physics engine does that.
- **Gravity**: Constant `G=0.0001`. All asteroids have mass proportional to volume. ~1/8 are super-dense (200x, marked `isMassive`). Gravitational acceleration is applied to ships each frame before engine physics (no fuel cost).
- **Selection**: Click to select, box-drag for multi-select, Ctrl+click to toggle. Control groups via Ctrl+1/2/3 (save) and 1/2/3 (recall).
- **Camera**: WASD fly, Space+WASD pan/orbit, Z/X zoom, L to focus on selected, mouse drag to orbit. Constrained to boundary sphere.
- **HUD**: Selection info, fuel display, ship group dots (1/2/3), asteroid mass display (orange for massive).
- **Chat**: Text input sends natural language to backend, which returns DSL commands executed by `executeDSL()`.

### Backend — `backend/main.ts`

Deno server that proxies natural language commands to Google Gemini LLM. The system prompt defines a DSL (MOVE, STOP, ORBIT, PATROL, FOLLOW, ATTACK, STATUS) and instructs Gemini to translate casual English (including slang, typos, informal references like "boys" or "guys") into structured commands.

Run with: `deno run --allow-net --allow-env backend/main.ts`

Requires `GEMINI_API_KEY` environment variable.

### Backend config — `backend/deno.json`

Deno workspace config with import map.

## Key design principles

1. **Physics is law**: All movement goes through the physics engine. Behaviors set intentions (targetVelocity, flags), physics applies forces. No teleporting, no instant velocity changes.
2. **Fuel matters**: Engine thrust costs fuel. Gravity is free. Ships near massive asteroids can orbit without fuel expenditure.
3. **One engine, one thrust**: Braking is reverse thrust — same power, same fuel cost as forward acceleration. No separate deceleration constant.

## Important code locations (index.html)

| What | Approx. lines |
|------|---------------|
| Physics constants (THRUST, FUEL_COST_PER_FRAME, G) | ~605-611 |
| Selection state, controlGroups | ~582-587 |
| Ship userData init (velocity, fuel, mass, behavior) | ~1523-1543 |
| Asteroid userData init (mass, isMassive, glow) | ~1707-1733 |
| Keyboard handler | ~1790-1830 |
| addToSelection / deselectAll | ~1987-2278 |
| Physics engine (gravity + thrust + position) | ~2429-2451 |
| Behavior engine (updateShipBehaviors) | ~3118-3242 |
| executeDSL (chat command dispatch) | ~3058-3113 |
| HUD updates (fuel, ship dots, asteroid info) | ~2540-2660 |
