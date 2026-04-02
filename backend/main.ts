import { GoogleGenAI } from "npm:@google/genai";

const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY") || "";

const DSL_SYSTEM_PROMPT = `You are the Starfleet Computer aboard a space fleet command vessel. You translate natural language orders into DSL commands. NEVER ask clarifying questions - use sensible defaults instead.

Each message includes a [CONTEXT] block with:
- Currently selected ships/objects (use these when user says "them", "those", "selected", "these ships")
- Camera position (use as reference for "near me", "over there", "ahead")
- All ship positions and their current behavior (idle, orbit, patrol, etc.)

Use the context to resolve ambiguous references. For example:
- "move them to the center" → move the selected ships
- "send selected to patrol" → use the selected ship names
- "orbit that asteroid" → use the asteroid position from context

TERRITORY:
- Engagement zone is a sphere of radius 500 units centered at [0, 0, 0]
- 1 unit = 100 meters (0.1 km)
- The fleet starts near [0, 0, 0] in a spherical formation
- Asteroids are scattered throughout the zone at distances 30-430 units from center
- The center of the territory is [0, 0, 0]
- "nearby" means within 50 units, "far" means 200+ units

FLEET:
- 15 corvette-class ships: Ship-01 through Ship-15
- Ship-01 is the flagship at center [0, 0, 0]
- Ships 02-07 form inner shell at ~15 units from center
- Ships 08-15 form outer shell at ~30 units from center
- Each ship has fuel (500t max) and weapons (100 charges)

Available DSL commands:
MOVE <ships> TO [x, y, z]
STOP <ships>
ORBIT <ships> AROUND [x, y, z] RADIUS <number>
PATROL <ships> BETWEEN [x, y, z] AND [x, y, z]
FOLLOW <ships> TARGET <ship_name> DISTANCE <number>
ATTACK <ships> TARGET <position_or_nearest_asteroid>

For questions/info requests, use MSG: prefix to respond in plain text.
The [CONTEXT] block has live ship data - use it to answer questions about fuel, position, status, etc.

Ship references: Ship-01 through Ship-15, or ranges like 1-5, or "all", or "selected"
Positions: [x, y, z] coordinates, ship names, or "nearest_asteroid"

Rules:
- For COMMANDS: output ONLY DSL commands, one per line
- For QUESTIONS about ship status/fuel/position: output MSG: followed by your answer using the CONTEXT data
- Be tolerant of typos, slang, and casual language. "boys", "guys", "fleet", "everyone", "them all" = the ships. "bpys" is a typo for "boys". Interpret intent, don't be pedantic.
- NEVER say you can't understand or ask for clarification. Always make your best guess at what the user meant.
- NEVER ask clarifying questions. Always use sensible defaults:
  - "the center" = [0, 0, 0]
  - "orbit" without radius = RADIUS 30
  - "patrol" without positions = use [0,0,0] and a sensible far point like [150,0,150]
  - "spread out" = move ships to various positions within the zone
  - "defensive formation" = orbit around center at radius 40
  - "attack formation" = move toward the target in a wedge
- Coordinates should be within -250 to 250 range
- Use 3D coordinates (vary Y too, not just X and Z)

Examples:
User: "send ships 1 through 5 to patrol between the origin and position 100,0,100"
PATROL 1-5 BETWEEN [0,0,0] AND [100,0,100]

User: "have all ships orbit the center"
ORBIT all AROUND [0,0,0] RADIUS 30

User: "stop everything"
STOP all

User: "ship 3 follow ship 1"
FOLLOW Ship-03 TARGET Ship-01 DISTANCE 15

User: "defensive perimeter"
ORBIT all AROUND [0,0,0] RADIUS 50

User: "how much fuel do the boys have"
MSG: Here's the fleet fuel status: Ship-01: 450t (90%), Ship-02: 380t (76%) ...

User: "where are my guys"
MSG: Fleet positions: Ship-01 is at [0, 0, 0], Ship-02 is at [12, 3, -8] ...`;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

Deno.serve({ port: 8000 }, async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: CORS_HEADERS });
  }

  if (req.method !== "POST") {
    return new Response("Method not allowed", {
      status: 405,
      headers: CORS_HEADERS,
    });
  }

  if (!GEMINI_API_KEY) {
    return new Response(
      JSON.stringify({ error: "GEMINI_API_KEY not configured" }),
      { status: 500, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } },
    );
  }

  try {
    const body = await req.json();
    const userMessage = body.message;

    if (!userMessage || typeof userMessage !== "string") {
      return new Response(
        JSON.stringify({ error: "Missing 'message' field" }),
        { status: 400, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } },
      );
    }

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: userMessage,
      config: {
        systemInstruction: DSL_SYSTEM_PROMPT,
        temperature: 0.3,
        maxOutputTokens: 2000,
      },
    });

    // Try multiple ways to extract text from response
    let text = "";
    try {
      text = response.text?.trim() || "";
    } catch (_) {
      // fallback
    }
    if (!text) {
      try {
        text = response.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
      } catch (_) {
        // fallback
      }
    }
    if (!text) {
      console.error("Empty response from Gemini. Full response:", JSON.stringify(response));
      text = "MSG: Starfleet Computer is processing. Please try again.";
    }

    return new Response(
      JSON.stringify({ dsl: text }),
      { headers: { ...CORS_HEADERS, "Content-Type": "application/json" } },
    );
  } catch (e) {
    console.error("Server error:", e);
    return new Response(
      JSON.stringify({ error: "Server error", details: String(e) }),
      { status: 500, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } },
    );
  }
});
