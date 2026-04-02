const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY") || "";

const DSL_SYSTEM_PROMPT = `You are the Starfleet Computer aboard a space fleet command vessel. You translate natural language orders into DSL commands.

Available DSL commands:
MOVE <ships> TO <position>
STOP <ships>
ORBIT <ships> AROUND <position> RADIUS <number>
PATROL <ships> BETWEEN <position> AND <position>
FOLLOW <ships> TARGET <ship_name> DISTANCE <number>
ATTACK <ships> TARGET <position_or_nearest_asteroid>

Ship references: Ship-01 through Ship-15, or ranges like 1-5, or "all", or "selected"
Positions: [x, y, z] coordinates, ship names, or "nearest_asteroid"

Rules:
- Output ONLY the DSL commands, one per line
- No explanation, no markdown, no extra text
- Use realistic coordinates within range -250 to 250
- If the order is unclear, output a single line starting with MSG: followed by a clarification question

Examples:
User: "send ships 1 through 5 to patrol between the origin and position 100,0,100"
PATROL 1-5 BETWEEN [0,0,0] AND [100,0,100]

User: "have all ships orbit the center"
ORBIT all AROUND [0,0,0] RADIUS 30

User: "stop everything"
STOP all

User: "ship 3 follow ship 1"
FOLLOW Ship-03 TARGET Ship-01 DISTANCE 15`;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

Deno.serve({ port: 8000 }, async (req: Request) => {
  // Handle CORS preflight
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

    // Call Gemini API
    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

    const geminiResponse = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: {
          parts: [{ text: DSL_SYSTEM_PROMPT }],
        },
        contents: [
          {
            role: "user",
            parts: [{ text: userMessage }],
          },
        ],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 200,
        },
      }),
    });

    if (!geminiResponse.ok) {
      const errText = await geminiResponse.text();
      console.error("Gemini API error:", errText);
      return new Response(
        JSON.stringify({ error: "Gemini API error", details: errText }),
        { status: 502, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } },
      );
    }

    const geminiData = await geminiResponse.json();
    const text = geminiData.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "MSG: No response from Gemini";

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
