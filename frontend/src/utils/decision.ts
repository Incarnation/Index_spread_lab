import type { TradeDecision } from "../api";
import { parseJsonRecord } from "./format";

export type DecisionSummary = {
  shortStrike?: number;
  longStrike?: number;
  spreadSide?: "put" | "call";
  credit?: number;
  gexNet?: number;
  zeroGamma?: number;
  vix?: number;
};

/**
 * Infer spread side from explicit payload fields or option-right fallbacks.
 */
function inferSpreadSide(data: Record<string, unknown>, shortLeg: Record<string, unknown> | undefined): "put" | "call" | undefined {
  const explicitSide = data["spread_side"];
  if (explicitSide === "put" || explicitSide === "call") return explicitSide;
  const optionRight = shortLeg?.["option_right"];
  if (optionRight === "P") return "put";
  if (optionRight === "C") return "call";
  return undefined;
}

/**
 * Parse the decision JSON payload into table-friendly numeric fields.
 *
 * Returns null when payload is absent or malformed so callers can render
 * placeholder values safely without throwing.
 */
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
    spreadSide: inferSpreadSide(data, short),
    credit: typeof credit === "number" ? credit : undefined,
    gexNet: typeof context?.["gex_net"] === "number" ? (context["gex_net"] as number) : undefined,
    zeroGamma: typeof context?.["zero_gamma_level"] === "number" ? (context["zero_gamma_level"] as number) : undefined,
    vix: typeof context?.["vix"] === "number" ? (context["vix"] as number) : undefined,
  };
}
