"use client";

import type { Match, ModelProbs } from "@/lib/types";

function TeamRow({ name, isWinner }: { name: string | null; isWinner: boolean }) {
  return (
    <div
      className={`truncate rounded px-2 py-1 text-sm ${
        name ? (isWinner ? "bg-emerald-600/20 font-semibold text-emerald-400" : "text-slate-200") : "italic text-slate-500"
      }`}
    >
      {name ?? "TBD"}
    </div>
  );
}

function ModelBreakdown({
  label,
  probs,
  homeTeam,
  awayTeam,
}: {
  label: string;
  probs: ModelProbs;
  homeTeam: string;
  awayTeam: string;
}) {
  const topIsHome = probs.home_win >= probs.away_win;
  return (
    <div className="text-xs">
      <p className="mb-0.5 font-semibold text-slate-400">{label}</p>
      <div className="flex justify-between text-slate-300">
        <span className={topIsHome ? "font-semibold text-emerald-400" : ""}>
          {homeTeam} {Math.round(probs.home_win * 100)}%
        </span>
        <span>Draw {Math.round(probs.draw * 100)}%</span>
        <span className={!topIsHome ? "font-semibold text-emerald-400" : ""}>
          {awayTeam} {Math.round(probs.away_win * 100)}%
        </span>
      </div>
    </div>
  );
}

export function MatchCard({ match, onRun }: { match: Match; onRun?: () => void }) {
  const p = match.prediction;
  const canRun = !!match.home && !!match.away && !match.loading;

  return (
    <div className="w-56 rounded-lg border border-slate-700 bg-slate-900 p-3 shadow">
      <div className="space-y-1">
        <TeamRow name={match.home} isWinner={match.winner === match.home} />
        <TeamRow name={match.away} isWinner={match.winner === match.away} />
      </div>

      {onRun && (
        <button
          className="mt-2 w-full rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!canRun}
          onClick={onRun}
          title="Run just this match (debug)"
        >
          {match.loading ? "Running…" : "Run this card"}
        </button>
      )}

      {match.loading && <p className="mt-2 text-xs text-indigo-400">Predicting…</p>}
      {match.error && <p className="mt-2 text-xs text-red-400">{match.error}</p>}

      {p && (
        <div className="mt-3 space-y-2 border-t border-slate-700 pt-2">
          <ModelBreakdown label="Claude" probs={p.claude} homeTeam={p.home_team} awayTeam={p.away_team} />
          {p.random_forest && (
            <ModelBreakdown label="Random Forest" probs={p.random_forest} homeTeam={p.home_team} awayTeam={p.away_team} />
          )}
          {p.logistic_regression && (
            <ModelBreakdown
              label="Logistic Regression"
              probs={p.logistic_regression}
              homeTeam={p.home_team}
              awayTeam={p.away_team}
            />
          )}
        </div>
      )}
    </div>
  );
}
