import type { TradeDecision } from "../api";
import { parseJsonRecord } from "./format";

export type DecisionSummary = {
  shortStrike?: number;
  longStrike?: number;
  credit?: number;
  gexNet?: number;
  zeroGamma?: number;
  vix?: number;
};

export function getDecisionSummary(row: TradeDecision): DecisionSummary | null {
  const data = parseJsonRecord(row.chosen_legs_json as Record<string, unknown> | string | null);
  if (!data) return null;

  const short = data["short"] as Record<string, unknown> | undefined;
  const long = data["long"] as Record<string, unknown> | undefined;
  const credit = data["credit"] as number | undefined;
  const context = data["context"] as Record<string, unknown> | undefined;

  if (!short || !long) return null;

  return {
    shortStrike: short["strike"] as number | undefined,
    longStrike: long["strike"] as number | undefined,
    credit: typeof credit === "number" ? credit : undefined,
    gexNet: typeof context?.["gex_net"] === "number" ? (context["gex_net"] as number) : undefined,
    zeroGamma: typeof context?.["zero_gamma_level"] === "number" ? (context["zero_gamma_level"] as number) : undefined,
    vix: typeof context?.["vix"] === "number" ? (context["vix"] as number) : undefined,
  };
}
