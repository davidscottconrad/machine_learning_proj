import { execFile } from "node:child_process";
import path from "node:path";
import { existsSync } from "node:fs";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const PYTHON_DIR = path.resolve(process.cwd(), "..", "python");
const SCRIPT_PATH = path.join(PYTHON_DIR, "predict.py");

function resolvePythonExecutable(): string {
  if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;
  const venvPython =
    process.platform === "win32"
      ? path.join(PYTHON_DIR, ".venv", "Scripts", "python.exe")
      : path.join(PYTHON_DIR, ".venv", "bin", "python");
  return existsSync(venvPython) ? venvPython : "python";
}

/** Python tracebacks are long; pull out the last non-empty line ("ExceptionType: message") for display. */
function summarizeError(raw: string): string {
  const lines = raw.trim().split("\n").filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : raw.trim();
}

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
  // No key from the request and none configured on the server (e.g. python/.env for local
  // dev) means the Python subprocess has nothing to authenticate with — fail fast instead of
  // spawning it. Every visitor supplies their own key; this server never ships one.
  if (!apiKey && !process.env.ANTHROPIC_API_KEY) {
    return NextResponse.json(
      { error: "Enter your Anthropic API key above to run predictions." },
      { status: 400 },
    );
  }

  const python = resolvePythonExecutable();
  const payload = JSON.stringify({ home_team: homeTeam, away_team: awayTeam, target_year: targetYear });
  // Passed as an env var (not a CLI arg or the JSON payload) so it never shows up in process
  // listings or logs of the arguments. python-dotenv's load_dotenv() defaults to override=False,
  // so this takes precedence over any key in python/.env.
  const env = apiKey ? { ...process.env, ANTHROPIC_API_KEY: apiKey } : process.env;

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        python,
        [SCRIPT_PATH, payload],
        { cwd: PYTHON_DIR, env, timeout: 120_000, maxBuffer: 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            reject(new Error(stderr?.trim() ? summarizeError(stderr) : error.message, { cause: stderr }));
            return;
          }
          resolve(stdout);
        },
      );
    });

    const result = JSON.parse(stdout.trim().split("\n").pop() ?? "{}");
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Prediction failed";
    const details = err instanceof Error && typeof err.cause === "string" ? err.cause : undefined;
    return NextResponse.json({ error: message, details }, { status: 500 });
  }
}
