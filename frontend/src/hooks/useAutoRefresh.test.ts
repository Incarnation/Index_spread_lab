import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAutoRefresh } from "./useAutoRefresh";

describe("useAutoRefresh", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts with tick=0 and paused=false", () => {
    const { result } = renderHook(() => useAutoRefresh(30_000, true));
    expect(result.current.tick).toBe(0);
    expect(result.current.paused).toBe(false);
  });

  it("increments tick on interval when alwaysActive", () => {
    const { result } = renderHook(() => useAutoRefresh(1000, true));
    expect(result.current.tick).toBe(0);

    act(() => { vi.advanceTimersByTime(1000); });
    expect(result.current.tick).toBe(1);

    act(() => { vi.advanceTimersByTime(1000); });
    expect(result.current.tick).toBe(2);
  });

  it("does not increment when paused", () => {
    const { result } = renderHook(() => useAutoRefresh(1000, true));

    act(() => { result.current.setPaused(true); });
    act(() => { vi.advanceTimersByTime(3000); });

    expect(result.current.tick).toBe(0);
  });

  it("resumes incrementing after unpause", () => {
    const { result } = renderHook(() => useAutoRefresh(1000, true));

    act(() => { result.current.setPaused(true); });
    act(() => { vi.advanceTimersByTime(2000); });
    expect(result.current.tick).toBe(0);

    act(() => { result.current.setPaused(false); });
    act(() => { vi.advanceTimersByTime(1000); });
    expect(result.current.tick).toBe(1);
  });
});
