"use client";

import { useEffect, useRef, useState } from "react";
import { TOURNAMENTS, type TournamentId } from "@/lib/tournaments";
import type { Match, PredictionResult } from "@/lib/types";
import { MatchCard } from "./MatchCard";
import { ResultsTab } from "./ResultsTab";

function cloneRounds(rounds: Match[][]): Match[][] {
  return rounds.map((round) => round.map((m) => ({ ...m })));
}

type Tab = TournamentId | "results";

export function TournamentApp() {
  const [statesByTournament, setStatesByTournament] = useState<Record<TournamentId, Match[][]>>(() => {
    const init = {} as Record<TournamentId, Match[][]>;
    for (const t of TOURNAMENTS) init[t.id] = t.buildInitial();
    return init;
  });
  const [simulating, setSimulating] = useState<Record<TournamentId, boolean>>(() => {
    const init = {} as Record<TournamentId, boolean>;
    for (const t of TOURNAMENTS) init[t.id] = false;
    return init;
  });
  const [activeTab, setActiveTab] = useState<Tab>("2026");
  const [banner, setBanner] = useState<string | null>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const [headerHeight, setHeaderHeight] = useState(0);

  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => setHeaderHeight(entries[0].contentRect.height));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Always merges into the *latest* state for just one match. Concurrent calls (two cards
  // running at once, or a card running mid-simulate) each only ever touch their own slot,
  // so one finishing can't stomp on another's in-progress update.
  function applyPatch(id: TournamentId, roundIdx: number, matchIdx: number, patch: Partial<Match>) {
    setStatesByTournament((prev) => {
      const next = cloneRounds(prev[id]);
      next[roundIdx][matchIdx] = { ...next[roundIdx][matchIdx], ...patch };
      return { ...prev, [id]: next };
    });
  }

  async function simulateTournament(id: TournamentId) {
    const tDef = TOURNAMENTS.find((t) => t.id === id)!;
    setBanner(null);
    setSimulating((s) => ({ ...s, [id]: true }));
    // Local bookkeeping only (tracks this call's own round-to-round propagation) —
    // never written back to React state wholesale; every UI update goes through applyPatch.
    const working = cloneRounds(statesByTournament[id]);

    for (let r = 0; r < working.length; r++) {
      const round = working[r];
      const playable = round
        .map((_, i) => i)
        .filter((i) => round[i].home && round[i].away && !round[i].winner);
      if (playable.length === 0) continue;

      playable.forEach((i) => {
        working[r][i] = { ...working[r][i], loading: true, error: null };
        applyPatch(id, r, i, { loading: true, error: null });
      });

      await Promise.all(
        playable.map(async (i) => {
          const match = working[r][i];
          try {
            const res = await fetch("/api/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ home_team: match.home, away_team: match.away, target_year: tDef.year }),
            });
            if (!res.ok) {
              const errBody = await res.json().catch(() => ({}));
              throw new Error(errBody.error ?? `Request failed (${res.status})`);
            }
            const prediction: PredictionResult = await res.json();
            working[r][i] = { ...working[r][i], loading: false, prediction, winner: prediction.winner_team };
            applyPatch(id, r, i, { loading: false, prediction, winner: prediction.winner_team });

            // Only the first round's matchup is the real draw; every later round is
            // whichever team the model actually predicted to advance.
            if (r + 1 < working.length) {
              const nextMatchIdx = Math.floor(i / 2);
              const side: "home" | "away" = i % 2 === 0 ? "home" : "away";
              working[r + 1][nextMatchIdx] = {
                ...working[r + 1][nextMatchIdx],
                [side]: prediction.winner_team,
              };
              applyPatch(id, r + 1, nextMatchIdx, { [side]: prediction.winner_team } as Partial<Match>);
            }
          } catch (err) {
            const message = err instanceof Error ? err.message : "Prediction failed";
            working[r][i] = { ...working[r][i], loading: false, error: message };
            applyPatch(id, r, i, { loading: false, error: message });
            setBanner(`${match.home} vs ${match.away}: ${message}`);
          }
        }),
      );
    }

    setSimulating((s) => ({ ...s, [id]: false }));
  }

  async function predictOne(id: TournamentId, roundIdx: number, matchIdx: number) {
    const tDef = TOURNAMENTS.find((t) => t.id === id)!;
    const match = statesByTournament[id][roundIdx][matchIdx];
    if (!match.home || !match.away) return;

    setBanner(null);
    applyPatch(id, roundIdx, matchIdx, { loading: true, error: null });

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home_team: match.home, away_team: match.away, target_year: tDef.year }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error ?? `Request failed (${res.status})`);
      }
      const prediction: PredictionResult = await res.json();
      applyPatch(id, roundIdx, matchIdx, { loading: false, prediction, winner: prediction.winner_team });

      if (roundIdx + 1 < tDef.roundLabels.length) {
        const nextMatchIdx = Math.floor(matchIdx / 2);
        const side: "home" | "away" = matchIdx % 2 === 0 ? "home" : "away";
        applyPatch(id, roundIdx + 1, nextMatchIdx, { [side]: prediction.winner_team } as Partial<Match>);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prediction failed";
      applyPatch(id, roundIdx, matchIdx, { loading: false, error: message });
      setBanner(`${match.home} vs ${match.away}: ${message}`);
    }
  }

  async function simulateAll() {
    for (const t of TOURNAMENTS) {
      await simulateTournament(t.id);
    }
  }

  function resetTournament(id: TournamentId) {
    const tDef = TOURNAMENTS.find((t) => t.id === id)!;
    setStatesByTournament((prev) => ({ ...prev, [id]: tDef.buildInitial() }));
  }

  const anySimulating = Object.values(simulating).some(Boolean);
  const activeTournament = activeTab !== "results" ? TOURNAMENTS.find((t) => t.id === activeTab)! : null;

  return (
    <div>
      <div ref={headerRef} className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950">
        <div className="px-6 py-4">
          <h1 className="text-xl font-bold">World Cup Bracket Predictor</h1>
          <p className="text-sm text-slate-400">
            2026/2022/2018/2014 knockout brackets, backtested against real results — each matchup
            predicted by a Claude agent backed by the project&apos;s Elo / Random Forest model.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 px-6 pb-4">
          <button
            className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-500"
            disabled={anySimulating}
            onClick={simulateAll}
          >
            {anySimulating ? "Simulating…" : "Run All Tournaments"}
          </button>
          <div className="ml-2 flex gap-1">
            {TOURNAMENTS.map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`rounded px-3 py-1.5 text-sm ${
                  activeTab === t.id ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"
                }`}
              >
                {t.label}
              </button>
            ))}
            <button
              onClick={() => setActiveTab("results")}
              className={`rounded px-3 py-1.5 text-sm ${
                activeTab === "results" ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              Results
            </button>
          </div>
        </div>
        {activeTournament && (
          <div className="flex items-center gap-3 px-6 pb-4">
            <button
              className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-500"
              disabled={simulating[activeTournament.id]}
              onClick={() => simulateTournament(activeTournament.id)}
            >
              {simulating[activeTournament.id] ? "Simulating…" : `Simulate ${activeTournament.label}`}
            </button>
            <button
              className="rounded border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={simulating[activeTournament.id]}
              onClick={() => resetTournament(activeTournament.id)}
            >
              Reset
            </button>
          </div>
        )}
        {banner && (
          <div className="flex items-start justify-between gap-3 border-t border-red-900 bg-red-950/60 px-6 py-2 text-sm text-red-300">
            <span className="break-all">{banner}</span>
            <button
              className="shrink-0 text-red-400 hover:text-red-200"
              onClick={() => setBanner(null)}
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {activeTab === "results" || !activeTournament ? (
        <ResultsTab statesByTournament={statesByTournament} />
      ) : (
        <div className="overflow-auto px-6" style={{ height: `calc(100vh - ${headerHeight}px)` }}>
          <div className="flex gap-8">
            {statesByTournament[activeTournament.id].map((round, roundIdx) => (
              <div key={roundIdx} className="flex shrink-0 flex-col">
                <h2 className="sticky top-0 z-10 bg-slate-950 py-4 text-center text-sm font-semibold uppercase tracking-wide text-slate-400">
                  {activeTournament.roundLabels[roundIdx]}
                </h2>
                <div className="flex flex-1 flex-col justify-around gap-6 pb-6">
                  {round.map((match, matchIdx) => (
                    <MatchCard
                      key={match.id}
                      match={match}
                      onRun={() => predictOne(activeTournament.id, roundIdx, matchIdx)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
