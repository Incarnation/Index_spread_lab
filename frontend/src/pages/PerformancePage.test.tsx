/**
 * Page-level tests for PerformancePage: KPI rendering, mode/lookback switching, error display.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

vi.mock("@/hooks/useAutoRefresh", () => ({
  useAutoRefresh: () => ({ tick: 0, paused: false, setPaused: vi.fn(), isMarketHours: true }),
}));

vi.mock("recharts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("recharts")>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

const mockPerf = {
  lookback_days: 90,
  mode: "realized" as const,
  window_start_utc: null,
  as_of_utc: null,
  snapshot: null,
  summary: {
    trade_count: 25,
    win_count: 16,
    loss_count: 9,
    net_pnl: 1250,
    realized_net_pnl: 1250,
    win_rate: 0.64,
    profit_factor: 2.1,
    avg_win: 120,
    avg_loss: -80,
    max_drawdown: -350,
    sharpe: 1.3,
  },
  equity_curve: [
    { date: "2026-03-01", daily_pnl: 50, cumulative_pnl: 50, drawdown: 0, trade_count: 1 },
    { date: "2026-03-02", daily_pnl: -20, cumulative_pnl: 30, drawdown: -20, trade_count: 1 },
  ],
  breakdowns: {
    side: [{ bucket: "put", trade_count: 25, win_count: 16, loss_count: 9, net_pnl: 1250, avg_pnl: 50, profit_factor: 2.1, win_rate: 0.64 }],
    dte_bucket: [],
    delta_bucket: [],
    weekday: [],
    hour: [],
    source: [],
  },
};

vi.mock("@/api", () => ({
  fetchPerformanceAnalytics: vi.fn(),
}));

import * as api from "@/api";
import { PerformancePage } from "./PerformancePage";

function renderPage() {
  return render(
    <MemoryRouter>
      <PerformancePage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("PerformancePage", () => {
  it("renders KPI cards and equity curve", async () => {
    vi.mocked(api.fetchPerformanceAnalytics).mockResolvedValue(mockPerf as never);
    renderPage();

    await waitFor(() => {
      expect(screen.getByText("Performance")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getAllByText("+$1,250.00").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText("64.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("25").length).toBeGreaterThan(0);
  });

  it("switches lookback period and refetches", async () => {
    vi.mocked(api.fetchPerformanceAnalytics).mockResolvedValue(mockPerf as never);
    renderPage();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByText("Performance")).toBeInTheDocument();
    });

    await user.click(screen.getByText("30d"));

    await waitFor(() => {
      expect(vi.mocked(api.fetchPerformanceAnalytics)).toHaveBeenCalledWith(
        30,
        expect.any(String),
        expect.any(Object),
      );
    });
  });

  it("switches mode and refetches", async () => {
    vi.mocked(api.fetchPerformanceAnalytics).mockResolvedValue(mockPerf as never);
    renderPage();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByText("Performance")).toBeInTheDocument();
    });

    await user.click(screen.getByText("combined"));

    await waitFor(() => {
      expect(vi.mocked(api.fetchPerformanceAnalytics)).toHaveBeenCalledWith(
        expect.any(Number),
        "combined",
        expect.any(Object),
      );
    });
  });

  it("shows error banner on fetch failure", async () => {
    vi.mocked(api.fetchPerformanceAnalytics).mockRejectedValue(new Error("API down"));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/API down/)).toBeInTheDocument();
    });
  });
});
