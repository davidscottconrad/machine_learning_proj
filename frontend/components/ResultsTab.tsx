import { TOURNAMENTS, type TournamentId } from "@/lib/tournaments";
import type { Match } from "@/lib/types";

const HISTORICAL_IDS: TournamentId[] = ["2026", "2022", "2018", "2014"];

export function ResultsTab({ statesByTournament }: { statesByTournament: Record<TournamentId, Match[][]> }) {
  return (
    <div className="space-y-10 p-6">
      <p className="text-sm text-slate-400">
        Backtest: for each real past tournament, the model predicts every knockout match using only data
        available in principle before that match — compare against what actually happened. Beyond the
        first round, the model advances its own predicted winners (like a real bracket pick&apos;em), so once
        a pick is wrong the rest of that branch is scored against the real matchup it never actually played.
      </p>
      {HISTORICAL_IDS.map((id) => {
        const tDef = TOURNAMENTS.find((t) => t.id === id)!;
        const rounds = statesByTournament[id];
        const rows = rounds.flatMap((round, rIdx) =>
          round.map((m) => ({
            round: tDef.roundLabels[rIdx],
            home: m.actualHome ?? m.home,
            away: m.actualAway ?? m.away,
            predicted: m.prediction?.winner_team ?? null,
            actual: m.actualWinner ?? null,
          })),
        );
        const scored = rows.filter((r) => r.predicted);
        const correct = scored.filter((r) => r.predicted === r.actual).length;

        return (
          <div key={id}>
            <h2 className="mb-2 text-lg font-semibold text-slate-100">
              {tDef.label}
              {scored.length > 0 && (
                <span className="ml-2 text-sm font-normal text-slate-400">
                  {correct}/{scored.length} correct ({Math.round((100 * correct) / scored.length)}%)
                </span>
              )}
            </h2>
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left text-slate-400">
                  <th className="py-1 pr-4 font-medium">Round</th>
                  <th className="py-1 pr-4 font-medium">Match</th>
                  <th className="py-1 pr-4 font-medium">Predicted</th>
                  <th className="py-1 pr-4 font-medium">Actual</th>
                  <th className="py-1 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => (
                  <tr key={idx} className="border-b border-slate-800">
                    <td className="py-1 pr-4 text-slate-500">{r.round}</td>
                    <td className="py-1 pr-4 text-slate-200">
                      {r.home} vs {r.away}
                    </td>
                    <td className="py-1 pr-4 text-slate-300">{r.predicted ?? "—"}</td>
                    <td className="py-1 pr-4 text-slate-300">{r.actual}</td>
                    <td className="py-1">
                      {r.predicted ? (r.predicted === r.actual ? "✅" : "❌") : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}
