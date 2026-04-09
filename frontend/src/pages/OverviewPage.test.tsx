/**
 * Page-level tests for OverviewPage: KPI rendering, error handling, unrealized PnL.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("@/hooks/useAutoRefresh", () => ({
  useAutoRefresh: () => ({ tick: 0, paused: false, setPaused: vi.fn(), isMarketHours: true }),
}));

const mockTrades = [
  { trade_id: 1, status: "OPEN", current_pnl: 120, realized_pnl: null, entry_time: "2026-04-08T10:00:00Z", trade_source: "scheduled" },
  { trade_id: 2, status: "OPEN", current_pnl: -50, realized_pnl: null, entry_time: "2026-04-08T11:00:00Z", trade_source: "event" },
  { trade_id: 3, status: "CLOSED", current_pnl: null, realized_pnl: 200, entry_time: "2026-04-07T10:00:00Z", trade_source: "scheduled" },
];

const mockPerf = {
  summary: { net_pnl: 500, win_rate: 0.65, trade_count: 10, win_count: 6, loss_count: 4, profit_factor: 1.8, avg_win: 100, max_drawdown: -200 },
  equity_curve: [],
  breakdowns: {},
};

vi.mock("@/api", () => ({
  fetchTrades: vi.fn(),
  fetchPerformanceAnalytics: vi.fn(),
  fetchPortfolioStatus: vi.fn(),
  fetchPortfolioHistory: vi.fn(),
  fetchPipelineStatus: vi.fn(),
}));

import * as api from "@/api";
import { OverviewPage } from "./OverviewPage";

function renderPage() {
  return render(
    <MemoryRouter>
      <OverviewPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("OverviewPage", () => {
  it("renders KPI cards after data loads", async () => {
    vi.mocked(api.fetchTrades).mockResolvedValue(mockTrades as never);
    vi.mocked(api.fetchPerformanceAnalytics).mockResolvedValue(mockPerf as never);
    vi.mocked(api.fetchPortfolioStatus).mockRejectedValue(new Error("not configured"));
    vi.mocked(api.fetchPortfolioHistory).mockRejectedValue(new Error("not configured"));
    vi.mocked(api.fetchPipelineStatus).mockResolvedValue({ warnings: [] } as never);

    renderPage();

    await waitFor(() => {
      expect(screen.getByText("Overview")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("+$500.00")).toBeInTheDocument();
    });
  });

  it("shows error banner when core fetch fails", async () => {
    vi.mocked(api.fetchTrades).mockRejectedValue(new Error("Network error"));
    vi.mocked(api.fetchPerformanceAnalytics).mockRejectedValue(new Error("Network error"));
    vi.mocked(api.fetchPortfolioStatus).mockRejectedValue(new Error("down"));
    vi.mocked(api.fetchPortfolioHistory).mockRejectedValue(new Error("down"));
    vi.mocked(api.fetchPipelineStatus).mockRejectedValue(new Error("down"));

    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeInTheDocument();
    });
  });

  it("displays unrealized PnL from open trades", async () => {
    vi.mocked(api.fetchTrades).mockResolvedValue(mockTrades as never);
    vi.mocked(api.fetchPerformanceAnalytics).mockResolvedValue(mockPerf as never);
    vi.mocked(api.fetchPortfolioStatus).mockRejectedValue(new Error("off"));
    vi.mocked(api.fetchPortfolioHistory).mockRejectedValue(new Error("off"));
    vi.mocked(api.fetchPipelineStatus).mockResolvedValue({ warnings: [] } as never);

    renderPage();

    await waitFor(() => {
      expect(screen.getByText("Unrealized PnL")).toBeInTheDocument();
    });

    // 120 + (-50) = 70
    await waitFor(() => {
      expect(screen.getByText("+$70.00")).toBeInTheDocument();
    });
  });
});
