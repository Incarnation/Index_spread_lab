import { describe, it, expect } from "vitest";
import { formatCurrency, formatPercent, formatDate, formatDateTime, timeAgo } from "./utils";

describe("formatCurrency", () => {
  it("formats positive values with + sign", () => {
    expect(formatCurrency(1234.5)).toBe("+$1,234.50");
  });

  it("formats negative values with - sign", () => {
    expect(formatCurrency(-500)).toBe("-$500.00");
  });

  it("formats zero without sign", () => {
    expect(formatCurrency(0)).toBe("$0.00");
  });

  it("returns dash for null/undefined", () => {
    expect(formatCurrency(null)).toBe("—");
    expect(formatCurrency(undefined)).toBe("—");
  });
});

describe("formatPercent", () => {
  it("formats decimal as percentage", () => {
    expect(formatPercent(0.5)).toBe("50.0%");
    expect(formatPercent(0.123)).toBe("12.3%");
  });

  it("returns dash for null/undefined", () => {
    expect(formatPercent(null)).toBe("—");
    expect(formatPercent(undefined)).toBe("—");
  });
});

describe("formatDate", () => {
  it("formats ISO date string", () => {
    const result = formatDate("2026-04-08T12:00:00Z");
    expect(result).toContain("Apr");
    expect(result).toContain("2026");
  });

  it("returns dash for null/undefined/empty", () => {
    expect(formatDate(null)).toBe("—");
    expect(formatDate(undefined)).toBe("—");
    expect(formatDate("")).toBe("—");
  });
});

describe("formatDateTime", () => {
  it("formats ISO datetime with time component", () => {
    const result = formatDateTime("2026-04-08T14:30:00Z");
    expect(result).toContain("Apr");
  });

  it("returns dash for null/undefined/empty", () => {
    expect(formatDateTime(null)).toBe("—");
    expect(formatDateTime(undefined)).toBe("—");
  });
});

describe("timeAgo", () => {
  it("returns 'just now' for very recent timestamps", () => {
    const now = new Date().toISOString();
    expect(timeAgo(now)).toBe("just now");
  });

  it("returns minutes ago for recent timestamps", () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    expect(timeAgo(fiveMinAgo)).toBe("5m ago");
  });

  it("returns hours ago for older timestamps", () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60_000).toISOString();
    expect(timeAgo(twoHoursAgo)).toBe("2h ago");
  });

  it("returns days ago for very old timestamps", () => {
    const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60_000).toISOString();
    expect(timeAgo(threeDaysAgo)).toBe("3d ago");
  });

  it("returns dash for null/undefined", () => {
    expect(timeAgo(null)).toBe("—");
    expect(timeAgo(undefined)).toBe("—");
  });
});
