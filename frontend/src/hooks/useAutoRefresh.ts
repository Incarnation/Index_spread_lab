import { useState, useEffect, useCallback } from "react";

/**
 * Auto-refresh hook with market-hours awareness.
 * Returns a tick counter that increments on each refresh, plus controls.
 *
 * During US market hours (Mon-Fri, 9:30-16:00 ET), polls at the given interval.
 * Outside market hours, polling is paused unless `alwaysActive` is set.
 */
export function useAutoRefresh(intervalMs = 30_000, alwaysActive = false) {
  const [tick, setTick] = useState(0);
  const [paused, setPaused] = useState(false);

  const isMarketHours = useCallback(() => {
    const now = new Date();
    const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
    const day = et.getDay();
    if (day === 0 || day === 6) return false;
    const h = et.getHours();
    const m = et.getMinutes();
    const mins = h * 60 + m;
    return mins >= 570 && mins <= 960; // 9:30 - 16:00 ET
  }, []);

  useEffect(() => {
    if (paused) return;
    const id = setInterval(() => {
      if (alwaysActive || isMarketHours()) {
        setTick((t) => t + 1);
      }
    }, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, paused, alwaysActive, isMarketHours]);

  return { tick, paused, setPaused, isMarketHours: isMarketHours() };
}
