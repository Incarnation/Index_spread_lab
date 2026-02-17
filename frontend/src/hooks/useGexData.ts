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

const GEX_DTE_STORAGE_KEY = "dashboard.gex.selectedDte";
const GEX_CUSTOM_EXP_STORAGE_KEY = "dashboard.gex.selectedCustomExpirations";
const GEX_SNAPSHOT_STORAGE_KEY = "dashboard.gex.selectedSnapshotId";

/**
 * Read one localStorage string value, guarding browsers where storage is unavailable.
 */
function readStorageString(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * Read one JSON string-array value from localStorage.
 */
function readStorageStringArray(key: string): string[] {
  const raw = readStorageString(key);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
}

/**
 * Persist one string value to localStorage.
 */
function writeStorageString(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage write errors (e.g., private mode limits).
  }
}

/**
 * Persist one string-array value to localStorage.
 */
function writeStorageStringArray(key: string, value: string[]): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage write errors (e.g., private mode limits).
  }
}

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
  const [selectedDte, setSelectedDte] = React.useState<string>(() => readStorageString(GEX_DTE_STORAGE_KEY) ?? "all");
  const [selectedCustomExpirations, setSelectedCustomExpirations] = React.useState<string[]>(() =>
    readStorageStringArray(GEX_CUSTOM_EXP_STORAGE_KEY),
  );
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
          const persistedSnapshotIdRaw = readStorageString(GEX_SNAPSHOT_STORAGE_KEY);
          const persistedSnapshotId = persistedSnapshotIdRaw ? Number(persistedSnapshotIdRaw) : null;
          const preferredSnapshot =
            persistedSnapshotId != null && Number.isFinite(persistedSnapshotId)
              ? rows.find((row) => row.snapshot_id === persistedSnapshotId) ?? null
              : null;
          setSelectedGexSnapshot(preferredSnapshot ?? rows[0]);
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
   * Persist key GEX selector state so analysts keep context after page refresh.
   */
  React.useEffect(() => {
    writeStorageString(GEX_DTE_STORAGE_KEY, selectedDte);
  }, [selectedDte]);

  React.useEffect(() => {
    writeStorageStringArray(GEX_CUSTOM_EXP_STORAGE_KEY, selectedCustomExpirations);
  }, [selectedCustomExpirations]);

  React.useEffect(() => {
    if (!selectedGexSnapshot) return;
    writeStorageString(GEX_SNAPSHOT_STORAGE_KEY, String(selectedGexSnapshot.snapshot_id));
  }, [selectedGexSnapshot]);

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
