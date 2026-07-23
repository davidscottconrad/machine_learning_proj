import { NextResponse } from "next/server";

export const runtime = "nodejs";

// In production this is injected by Vercel's service binding (see repo-root vercel.json,
// the frontend service's `bindings` entry targeting the `predictor` service) -- never a
// hardcoded hostname. The localhost fallback is just for running `uvicorn app:app --port
// 8000` locally without going through `vercel dev`.
const PREDICTOR_URL = process.env.PREDICTOR_URL ?? "http://localhost:8000";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const homeTeam = body?.home_team;
  const awayTeam = body?.away_team;
  const targetYear = body?.target_year;
  const apiKey = body?.api_key;

  if (typeof homeTeam !== "string" || typeof awayTeam !== "string" || !homeTeam || !awayTeam) {
    return NextResponse.json({ error: "home_team and away_team are required" }, { status: 400 });
  }
  if (typeof targetYear !== "number") {
    return NextResponse.json({ error: "target_year is required" }, { status: 400 });
  }
  if (apiKey !== undefined && (typeof apiKey !== "string" || !apiKey.trim())) {
    return NextResponse.json({ error: "api_key must be a non-empty string" }, { status: 400 });
  }
  // Every visitor supplies their own key; this app never ships a shared one, so fail fast
  // instead of calling the predictor service with nothing to authenticate with.
  if (!apiKey) {
    return NextResponse.json(
      { error: "Enter your Anthropic API key above to run predictions." },
      { status: 400 },
    );
  }

  try {
    const res = await fetch(new URL("/predict", PREDICTOR_URL), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-Api-Key": apiKey } : {}),
      },
      body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, target_year: targetYear }),
    });

    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const message = typeof payload.detail === "string" ? payload.detail : `Predictor request failed (${res.status})`;
      return NextResponse.json({ error: message }, { status: 500 });
    }
    return NextResponse.json(payload);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Prediction failed";
    return NextResponse.json({ error: `Could not reach predictor service: ${message}` }, { status: 500 });
  }
}
