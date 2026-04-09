/**
 * Tests for api.ts: URL builder, auth header injection, 401 handling,
 * and representative fetch helpers.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as authStorage from "./auth";
import { apiUrl, UNAUTHORIZED_EVENT } from "./api";

beforeEach(() => {
  authStorage.clearToken();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("apiUrl", () => {
  it("prepends API_BASE to a path", () => {
    const url = apiUrl("/api/health");
    expect(url).toMatch(/\/api\/health$/);
  });
});

describe("fetchWithAuth behavior", () => {
  it("attaches Bearer token when present", async () => {
    authStorage.setToken("test-jwt");

    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );

    const { fetchGexSnapshots } = await import("./api");
    // fetchGexSnapshots uses fetchWithAuth internally
    // It may fail parsing, but we care about the fetch call
    try {
      await fetchGexSnapshots(1);
    } catch {
      // response shape may not match exactly; we just verify fetch was called
    }

    expect(spy).toHaveBeenCalled();
    const [, init] = spy.mock.calls[0];
    expect((init?.headers as Record<string, string>)?.Authorization).toBe(
      "Bearer test-jwt",
    );
  });

  it("clears token and dispatches event on 401", async () => {
    authStorage.setToken("expired-jwt");

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("Unauthorized", { status: 401 }),
    );

    const events: Event[] = [];
    const listener = (e: Event) => events.push(e);
    window.addEventListener(UNAUTHORIZED_EVENT, listener);

    const { fetchTrades } = await import("./api");
    await expect(fetchTrades(10)).rejects.toThrow("Unauthorized");

    expect(authStorage.getToken()).toBeNull();
    expect(events.length).toBe(1);

    window.removeEventListener(UNAUTHORIZED_EVENT, listener);
  });

  it("throws on non-OK non-401 responses", async () => {
    authStorage.setToken("good-jwt");

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("Server Error", { status: 500 }),
    );

    const { runQuotesNow } = await import("./api");
    await expect(runQuotesNow()).rejects.toThrow("HTTP 500");
  });
});
