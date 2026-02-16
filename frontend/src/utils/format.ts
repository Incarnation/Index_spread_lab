/**
 * Truncate long text values for compact table/code-cell display.
 */
export function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength)}…`;
}

/**
 * Format an ISO timestamp string into local human-readable time.
 *
 * If parsing fails, returns the original input so debugging data is preserved.
 */
export function formatTs(ts: string): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return ts;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

/**
 * Safely coerce JSON-like values into an object payload.
 *
 * Accepts object input directly or parses a string payload; returns null for
 * empty/invalid input so callers can handle absence explicitly.
 */
export function parseJsonRecord(value: Record<string, unknown> | string | null): Record<string, unknown> | null {
  if (!value) return null;
  if (typeof value === "string") {
    try {
      return JSON.parse(value) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  return value;
}
