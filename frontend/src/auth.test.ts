import { describe, it, expect, beforeEach } from "vitest";
import { getToken, setToken, clearToken } from "./auth";

describe("auth token storage", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it("returns null when no token is stored", () => {
    expect(getToken()).toBeNull();
  });

  it("stores and retrieves a token", () => {
    setToken("test-jwt-token");
    expect(getToken()).toBe("test-jwt-token");
  });

  it("clears the token", () => {
    setToken("test-jwt-token");
    clearToken();
    expect(getToken()).toBeNull();
  });

  it("overwrites previous token on set", () => {
    setToken("first");
    setToken("second");
    expect(getToken()).toBe("second");
  });
});
