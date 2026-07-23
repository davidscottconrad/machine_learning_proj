export interface ModelProbs {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface ClaudeProbs extends ModelProbs {
  reasoning: string;
}

export interface PredictionResult {
  home_team: string;
  away_team: string;
  target_year: number;
  winner: "home" | "away";
  winner_team: string;
  claude: ClaudeProbs;
  random_forest?: ModelProbs;
  logistic_regression?: ModelProbs;
  model?: string;
  usage?: {
    input_tokens: number;
    output_tokens: number;
    estimated_cost_usd: number;
  };
}

export interface Match {
  id: string;
  home: string | null;
  away: string | null;
  winner: string | null;
  prediction: PredictionResult | null;
  loading: boolean;
  error: string | null;
  /**
   * The real historical matchup/winner for this bracket slot — always the actual teams that
   * played here in reality, independent of `home`/`away` (which for rounds after the first
   * reflect whichever team the model predicted to advance, not necessarily who really did).
   * Used by the Results tab so "Actual" always shows what really happened.
   */
  actualHome?: string | null;
  actualAway?: string | null;
  actualWinner?: string | null;
}

export interface BracketState {
  rounds: Match[][];
}
