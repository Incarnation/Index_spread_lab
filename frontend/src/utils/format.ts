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
 * Convert one timestamp into an ISO date string (`YYYY-MM-DD`) in a target timezone.
 *
 * This is used for strict expiration-day matching (for example 0DTE views) where
 * local browser timezone should not affect filtering behavior.
 */
export function formatDateIsoInTimezone(ts: string, timeZone: string): string | null {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) return null;
  return `${year}-${month}-${day}`;
}

/**
 * Format one timestamp as `YYYY-MM-DD HH:mm:ss` in the requested timezone.
 *
 * Returns null when the input cannot be parsed or timezone parts are missing.
 */
export function formatDateTimeInTimezone(ts: string, timeZone: string): string | null {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  const hour = parts.find((part) => part.type === "hour")?.value;
  const minute = parts.find((part) => part.type === "minute")?.value;
  const second = parts.find((part) => part.type === "second")?.value;
  if (!year || !month || !day || !hour || !minute || !second) return null;
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

/**
 * Format one timestamp as a rounded slot label (`HH:mm`) in timezone.
 *
 * Callers can pick the slot interval (for example 15 minutes for RTH polling).
 * Returns null for unparseable timestamps, invalid intervals, or missing parts.
 */
export function formatTimeSlotInTimezone(ts: string, timeZone: string, slotMinutes = 1): string | null {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime()) || !Number.isFinite(slotMinutes) || slotMinutes <= 0) return null;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const hour = parts.find((part) => part.type === "hour")?.value;
  const minute = parts.find((part) => part.type === "minute")?.value;
  const second = parts.find((part) => part.type === "second")?.value;
  if (!hour || !minute || !second) return null;

  const totalMinutes = Number(hour) * 60 + Number(minute) + Number(second) / 60;
  const roundedSlotMinutes = Math.round(totalMinutes / slotMinutes) * slotMinutes;
  const normalizedMinutes = ((roundedSlotMinutes % (24 * 60)) + 24 * 60) % (24 * 60);
  const slotHour = Math.floor(normalizedMinutes / 60);
  const slotMinute = normalizedMinutes % 60;
  return `${String(slotHour).padStart(2, "0")}:${String(slotMinute).padStart(2, "0")}`;
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
