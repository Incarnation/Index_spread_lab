import type { GexSnapshot } from "../api";
import { formatDateIsoInTimezone } from "../utils/format";

export const GEX_UNDERLYING_OPTIONS = ["SPX", "SPY", "VIX"] as const;
export type GexUnderlying = (typeof GEX_UNDERLYING_OPTIONS)[number];
export const GEX_SOURCE_OPTIONS = ["all", "TRADIER", "CBOE"] as const;
export type GexSource = (typeof GEX_SOURCE_OPTIONS)[number];

export const GEX_TRADING_TIMEZONE = "America/New_York";
export const GEX_ZERO_DTE_ONLY_SENTINEL = "__ZERO_DTE_ONLY__";

/**
 * Resolve a snapshot timestamp into its trading-session date (`YYYY-MM-DD`).
 *
 * We normalize to New York time because index/ETF option expirations in this
 * dashboard are aligned to US market sessions.
 */
export function getSnapshotTradingDateIso(snapshot: Pick<GexSnapshot, "ts"> | null): string | null {
  if (!snapshot) return null;
  return formatDateIsoInTimezone(snapshot.ts, GEX_TRADING_TIMEZONE);
}
