/**
 * Page-level tests for GexPage: snapshot rendering, empty state, error display.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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

const mockSnapshots = [
  { snapshot_id: 100, ts: "2026-04-08T14:30:00Z", underlying: "SPX", source: "CBOE", spot_price: 5200.5, gex_net: 1.5e9, zero_gamma_level: 5180 },
  { snapshot_id: 99, ts: "2026-04-08T14:00:00Z", underlying: "SPX", source: "CBOE", spot_price: 5198.0, gex_net: 1.4e9, zero_gamma_level: 5175 },
];

const mockCurve = [
  { strike: 5100, gex_net: 1e8, gex_calls: 2e8, gex_puts: -1e8 },
  { strike: 5200, gex_net: 5e8, gex_calls: 6e8, gex_puts: -1e8 },
];

vi.mock("@/api", () => ({
  fetchGexSnapshots: vi.fn(),
  fetchGexDtes: vi.fn(),
  fetchGexCurve: vi.fn(),
}));

import * as api from "@/api";
import { GexPage } from "./GexPage";

function renderPage() {
  return render(
    <MemoryRouter>
      <GexPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("GexPage", () => {
  it("renders snapshot selector and info after loading", async () => {
    vi.mocked(api.fetchGexSnapshots).mockResolvedValue(mockSnapshots as never);
    vi.mocked(api.fetchGexDtes).mockResolvedValue([0, 1, 3] as never);
    vi.mocked(api.fetchGexCurve).mockResolvedValue(mockCurve as never);

    renderPage();

    await waitFor(() => {
      expect(screen.getByText("GEX / Market")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("5200.50")).toBeInTheDocument();
    });
  });

  it("handles empty snapshots gracefully", async () => {
    vi.mocked(api.fetchGexSnapshots).mockResolvedValue([]);
    vi.mocked(api.fetchGexDtes).mockResolvedValue([]);
    vi.mocked(api.fetchGexCurve).mockResolvedValue([]);

    renderPage();

    await waitFor(() => {
      expect(screen.getByText("GEX / Market")).toBeInTheDocument();
    });

    expect(screen.queryByText("Spot:")).not.toBeInTheDocument();
  });

  it("shows error banner on fetch failure", async () => {
    vi.mocked(api.fetchGexSnapshots).mockRejectedValue(new Error("Timeout"));
    vi.mocked(api.fetchGexDtes).mockResolvedValue([]);
    vi.mocked(api.fetchGexCurve).mockResolvedValue([]);

    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/Timeout/)).toBeInTheDocument();
    });
  });
});
