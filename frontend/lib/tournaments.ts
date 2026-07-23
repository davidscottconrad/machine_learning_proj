import type { Match } from "./types";

import data2014 from "./historical/2014.json";
import data2018 from "./historical/2018.json";
import data2022 from "./historical/2022.json";
import data2026 from "./historical/2026.json";

export type TournamentId = "2026" | "2022" | "2018" | "2014";

interface HistoricalMatch {
  home: string;
  away: string;
  winner: string;
}

interface HistoricalData {
  year: number;
  roundLabels: string[];
  rounds: HistoricalMatch[][];
}

export interface TournamentDef {
  id: TournamentId;
  label: string;
  year: number;
  roundLabels: string[];
  buildInitial: () => Match[][];
}

// Only the first round (the real draw) is fixed. Every later round starts empty and gets
// filled in by whichever team the model actually predicts to advance — the real historical
// matchup/winner for that slot is preserved separately (`actualHome`/`actualAway`/`actualWinner`)
// so the Results tab can still show what really happened, independent of the model's own path.
function buildHistorical(data: HistoricalData): Match[][] {
  return data.rounds.map((round, rIdx) =>
    round.map((m, i) => ({
      id: `h-${data.year}-r${rIdx}-${i}`,
      home: rIdx === 0 ? m.home : null,
      away: rIdx === 0 ? m.away : null,
      winner: null,
      prediction: null,
      loading: false,
      error: null,
      actualHome: m.home,
      actualAway: m.away,
      actualWinner: m.winner,
    })),
  );
}

export const TOURNAMENTS: TournamentDef[] = [
  {
    id: "2026",
    label: "2026",
    year: 2026,
    roundLabels: (data2026 as HistoricalData).roundLabels,
    buildInitial: () => buildHistorical(data2026 as HistoricalData),
  },
  {
    id: "2022",
    label: "2022",
    year: 2022,
    roundLabels: (data2022 as HistoricalData).roundLabels,
    buildInitial: () => buildHistorical(data2022 as HistoricalData),
  },
  {
    id: "2018",
    label: "2018",
    year: 2018,
    roundLabels: (data2018 as HistoricalData).roundLabels,
    buildInitial: () => buildHistorical(data2018 as HistoricalData),
  },
  {
    id: "2014",
    label: "2014",
    year: 2014,
    roundLabels: (data2014 as HistoricalData).roundLabels,
    buildInitial: () => buildHistorical(data2014 as HistoricalData),
  },
];
