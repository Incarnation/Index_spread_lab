import React from "react";
import {
  fetchGexCurve,
  fetchGexDtes,
  fetchGexExpirations,
  fetchGexSnapshots,
  type GexCurvePoint,
  type GexExpirationItem,
  type GexSnapshot,
} from "../api";

type UseGexDataArgs = {
  onError: (message: string) => void;
};

type UseGexDataResult = {
  gexSnapshots: GexSnapshot[];
  selectedGexSnapshot: GexSnapshot | null;
  setSelectedGexSnapshot: (snapshot: GexSnapshot | null) => void;
  gexDtes: number[];
  gexExpirations: GexExpirationItem[];
  selectedDte: string;
  selectedCustomExpirations: string[];
  setSelectedCustomExpirations: (values: string[]) => void;
  handleSelectedDteChange: (value: string) => void;
  gexCurve: GexCurvePoint[];
  gexLoading: boolean;
};

/**
 * Manage all GEX panel state: snapshots, DTE filters, custom expirations, curve.
 *
 * This hook coordinates dependent API calls so the chart always reflects the
 * currently selected snapshot and expiration filtering mode.
 */
export function useGexData({ onError }: UseGexDataArgs): UseGexDataResult {
  const [gexSnapshots, setGexSnapshots] = React.useState<GexSnapshot[]>([]);
  const [selectedGexSnapshot, setSelectedGexSnapshot] = React.useState<GexSnapshot | null>(null);
  const [gexDtes, setGexDtes] = React.useState<number[]>([]);
  const [gexExpirations, setGexExpirations] = React.useState<GexExpirationItem[]>([]);
  const [selectedDte, setSelectedDte] = React.useState<string>("all");
  const [selectedCustomExpirations, setSelectedCustomExpirations] = React.useState<string[]>([]);
  const [gexCurve, setGexCurve] = React.useState<GexCurvePoint[]>([]);
  const [gexLoading, setGexLoading] = React.useState<boolean>(false);

  /**
   * Load recent GEX snapshots once on mount and default-select the latest row.
   */
  React.useEffect(() => {
    let cancelled = false;
    setGexLoading(true);
    fetchGexSnapshots(20)
      .then((rows) => {
        if (cancelled) return;
        setGexSnapshots(rows);
        if (rows.length > 0) {
          setSelectedGexSnapshot(rows[0]);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) onError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setGexLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [onError]);

  /**
   * Load DTE and expiration options whenever selected snapshot changes.
   */
  React.useEffect(() => {
    if (!selectedGexSnapshot) return;
    let cancelled = false;
    Promise.all([fetchGexDtes(selectedGexSnapshot.snapshot_id), fetchGexExpirations(selectedGexSnapshot.snapshot_id)])
      .then(([dteRows, expirationRows]) => {
        if (cancelled) return;
        setGexDtes(dteRows);
        setGexExpirations(expirationRows);
        setSelectedDte((prev) => {
          if (prev === "all" || prev === "custom") return prev;
          return dteRows.includes(Number(prev)) ? prev : "all";
        });
        setSelectedCustomExpirations((prev) =>
          prev.filter((value) => expirationRows.some((row) => row.expiration === value)),
        );
      })
      .catch((e: unknown) => {
        if (!cancelled) onError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [onError, selectedGexSnapshot]);

  /**
   * Load the curve points for current selection mode (all, one DTE, custom dates).
   */
  React.useEffect(() => {
    if (!selectedGexSnapshot) return;
    let cancelled = false;
    setGexLoading(true);

    if (selectedDte === "custom" && selectedCustomExpirations.length === 0) {
      setGexCurve([]);
      setGexLoading(false);
      return;
    }

    const dteVal = selectedDte === "all" || selectedDte === "custom" ? undefined : Number(selectedDte);
    fetchGexCurve(
      selectedGexSnapshot.snapshot_id,
      dteVal,
      selectedDte === "custom" ? selectedCustomExpirations : undefined,
    )
      .then((points) => {
        if (!cancelled) setGexCurve(points);
      })
      .catch((e: unknown) => {
        if (!cancelled) onError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setGexLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [onError, selectedCustomExpirations, selectedDte, selectedGexSnapshot]);

  /**
   * Update DTE mode and clear stale custom selections when leaving custom mode.
   */
  const handleSelectedDteChange = React.useCallback((value: string) => {
    setSelectedDte(value);
    if (value !== "custom") {
      setSelectedCustomExpirations([]);
    }
  }, []);

  return {
    gexSnapshots,
    selectedGexSnapshot,
    setSelectedGexSnapshot,
    gexDtes,
    gexExpirations,
    selectedDte,
    selectedCustomExpirations,
    setSelectedCustomExpirations,
    handleSelectedDteChange,
    gexCurve,
    gexLoading,
  };
}
