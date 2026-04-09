/**
 * Page-level tests for TradesPage: trade table rendering, source filter, error display.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

vi.mock("@/hooks/useAutoRefresh", () => ({
  useAutoRefresh: () => ({ tick: 0, paused: false, setPaused: vi.fn(), isMarketHours: true }),
}));

const mockTrades = [
  {
    trade_id: 1, status: "OPEN", trade_source: "scheduled_portfolio",
    entry_time: "2026-04-08T10:00:00Z", expiration: "2026-04-11",
    target_dte: 3, contracts: 1, entry_credit: 1.5, current_pnl: 80, realized_pnl: null,
    legs: [{ option_right: "put" }],
  },
  {
    trade_id: 2, status: "CLOSED", trade_source: "event_vix_spike",
    entry_time: "2026-04-07T10:00:00Z", expiration: "2026-04-10",
    target_dte: 3, contracts: 2, entry_credit: 2.0, current_pnl: null, realized_pnl: -120,
    exit_reason: "stop_loss", legs: [{ option_right: "put" }],
  },
];

vi.mock("@/api", () => ({
  fetchTrades: vi.fn(),
}));

import * as api from "@/api";
import { TradesPage } from "./TradesPage";

function renderPage() {
  return render(
    <MemoryRouter>
      <TradesPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("TradesPage", () => {
  it("renders trade table with open/closed tabs", async () => {
    vi.mocked(api.fetchTrades).mockResolvedValue(mockTrades as never);
    renderPage();

    await waitFor(() => {
      expect(screen.getByText("Trades")).toBeInTheDocument();
    });

    expect(screen.getByText("Open (1)")).toBeInTheDocument();
    expect(screen.getByText("Closed (1)")).toBeInTheDocument();
  });

  it("filters trades by source", async () => {
    vi.mocked(api.fetchTrades).mockResolvedValue(mockTrades as never);
    renderPage();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByText(/Event \(1\)/)).toBeInTheDocument();
    });

    await user.click(screen.getByText(/Event \(1\)/));

    expect(screen.getByText("Open (0)")).toBeInTheDocument();
    expect(screen.getByText("Closed (1)")).toBeInTheDocument();
  });

  it("shows error banner on fetch failure", async () => {
    vi.mocked(api.fetchTrades).mockRejectedValue(new Error("Server error"));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/Server error/)).toBeInTheDocument();
    });
  });
});
