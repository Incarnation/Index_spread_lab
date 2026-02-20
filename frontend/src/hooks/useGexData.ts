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
import {
  GEX_SOURCE_OPTIONS,
  GEX_UNDERLYING_OPTIONS,
  GEX_ZERO_DTE_ONLY_SENTINEL,
  getSnapshotTradingDateIso,
  type GexSource,
  type GexUnderlying,
} from "../constants/gex";

type UseGexDataArgs = {
  onError: (message: string) => void;
};

const GEX_DTE_STORAGE_KEY = "dashboard.gex.selectedDte";
const GEX_CUSTOM_EXP_STORAGE_KEY = "dashboard.gex.selectedCustomExpirations";
const GEX_SNAPSHOT_STORAGE_KEY = "dashboard.gex.selectedSnapshotId";
const GEX_UNDERLYING_STORAGE_KEY = "dashboard.gex.selectedUnderlying";
const GEX_SOURCE_STORAGE_KEY = "dashboard.gex.selectedSource";

/**
 * Normalize one underlying symbol to uppercase without surrounding whitespace.
 */
function normalizeUnderlying(value: string | null | undefined): string {
  return (value ?? "").trim().toUpperCase();
}

/**
 * Return true when a snapshot row belongs to the selected underlying.
 */
function isMatchingUnderlying(snapshot: GexSnapshot, selectedUnderlying: GexUnderlying): boolean {
  return normalizeUnderlying(snapshot.underlying) === selectedUnderlying;
}

/**
 * Return true when a snapshot row belongs to the selected source filter.
 */
function isMatchingSource(snapshot: GexSnapshot, selectedSource: GexSource): boolean {
  if (selectedSource === "all") return true;
  return (snapshot.source ?? "").trim().toUpperCase() === selectedSource;
}

/**
 * Normalize custom expiration selection values and enforce exclusive 0DTE mode.
 */
function normalizeCustomExpirationSelection(values: string[]): string[] {
  const cleaned = Array.from(
    new Set(values.map((value) => value.trim()).filter((value) => value.length > 0)),
  );
  if (cleaned.includes(GEX_ZERO_DTE_ONLY_SENTINEL)) {
    return [GEX_ZERO_DTE_ONLY_SENTINEL];
  }
  return cleaned;
}

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

/**
 * Normalize persisted underlying values to one supported symbol.
 */
function readStoredGexUnderlying(): GexUnderlying {
  const raw = normalizeUnderlying(readStorageString(GEX_UNDERLYING_STORAGE_KEY));
  if ((GEX_UNDERLYING_OPTIONS as readonly string[]).includes(raw)) {
    return raw as GexUnderlying;
  }
  return "SPX";
}

/**
 * Normalize persisted source values to one supported source filter option.
 */
function readStoredGexSource(): GexSource {
  const raw = (readStorageString(GEX_SOURCE_STORAGE_KEY) ?? "").trim();
  if ((GEX_SOURCE_OPTIONS as readonly string[]).includes(raw)) {
    return raw as GexSource;
  }
  return "all";
}

type UseGexDataResult = {
  gexSnapshots: GexSnapshot[];
  selectedGexSnapshot: GexSnapshot | null;
  setSelectedGexSnapshot: (snapshot: GexSnapshot | null) => void;
  selectedUnderlying: GexUnderlying;
  handleSelectedUnderlyingChange: (value: string | null) => void;
  selectedSource: GexSource;
  handleSelectedSourceChange: (value: string | null) => void;
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
  const [selectedGexSnapshotState, setSelectedGexSnapshotState] = React.useState<GexSnapshot | null>(null);
  const [selectedUnderlying, setSelectedUnderlying] = React.useState<GexUnderlying>(() => readStoredGexUnderlying());
  const [selectedSource, setSelectedSource] = React.useState<GexSource>(() => readStoredGexSource());
  const [gexDtes, setGexDtes] = React.useState<number[]>([]);
  const [gexExpirations, setGexExpirations] = React.useState<GexExpirationItem[]>([]);
  const [selectedDte, setSelectedDte] = React.useState<string>(() => readStorageString(GEX_DTE_STORAGE_KEY) ?? "all");
  const [selectedCustomExpirationsState, setSelectedCustomExpirationsState] = React.useState<string[]>(() =>
    normalizeCustomExpirationSelection(readStorageStringArray(GEX_CUSTOM_EXP_STORAGE_KEY)),
  );
  const [gexCurve, setGexCurve] = React.useState<GexCurvePoint[]>([]);
  const [gexLoading, setGexLoading] = React.useState<boolean>(false);

  const selectedGexSnapshot = selectedGexSnapshotState;
  const selectedCustomExpirations = selectedCustomExpirationsState;

  /**
   * Enforce symbol consistency whenever the selected snapshot is set manually.
   */
  const setSelectedGexSnapshot = React.useCallback(
    (snapshot: GexSnapshot | null) => {
      if (snapshot && (!isMatchingUnderlying(snapshot, selectedUnderlying) || !isMatchingSource(snapshot, selectedSource))) {
        return;
      }
      setSelectedGexSnapshotState(snapshot);
    },
    [selectedSource, selectedUnderlying],
  );

  /**
   * Normalize custom expiration values before storing selection state.
   */
  const setSelectedCustomExpirations = React.useCallback((values: string[]) => {
    setSelectedCustomExpirationsState(normalizeCustomExpirationSelection(values));
  }, []);

  /**
   * Load recent GEX snapshots for the selected underlying and pick a default batch.
   */
  React.useEffect(() => {
    let cancelled = false;
    setGexLoading(true);
    fetchGexSnapshots(20, selectedUnderlying, selectedSource === "all" ? undefined : selectedSource)
      .then((rows) => {
        if (cancelled) return;
        const symbolRows = rows.filter(
          (row) => isMatchingUnderlying(row, selectedUnderlying) && isMatchingSource(row, selectedSource),
        );
        setGexSnapshots(symbolRows);
        if (symbolRows.length > 0) {
          const persistedSnapshotIdRaw = readStorageString(GEX_SNAPSHOT_STORAGE_KEY);
          const persistedSnapshotId = persistedSnapshotIdRaw ? Number(persistedSnapshotIdRaw) : null;
          const preferredSnapshot =
            persistedSnapshotId != null && Number.isFinite(persistedSnapshotId)
              ? symbolRows.find((row) => row.snapshot_id === persistedSnapshotId) ?? null
              : null;
          setSelectedGexSnapshotState(preferredSnapshot ?? symbolRows[0]);
        } else {
          setSelectedGexSnapshotState(null);
          setGexDtes([]);
          setGexExpirations([]);
          setGexCurve([]);
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
  }, [onError, selectedSource, selectedUnderlying]);

  /**
   * Guard against stale UI state by forcing selected snapshot symbol to match filter.
   */
  React.useEffect(() => {
    if (!selectedGexSnapshot) return;
    if (isMatchingUnderlying(selectedGexSnapshot, selectedUnderlying) && isMatchingSource(selectedGexSnapshot, selectedSource)) {
      return;
    }
    const fallbackSnapshot =
      gexSnapshots.find((row) => isMatchingUnderlying(row, selectedUnderlying) && isMatchingSource(row, selectedSource)) ?? null;
    setSelectedGexSnapshotState(fallbackSnapshot);
  }, [gexSnapshots, selectedGexSnapshot, selectedSource, selectedUnderlying]);

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
        setSelectedCustomExpirationsState((prev) =>
          prev.filter(
            (value) =>
              value === GEX_ZERO_DTE_ONLY_SENTINEL || expirationRows.some((row) => row.expiration === value),
          ),
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
    const hasZeroDteOnlySelection =
      selectedDte === "custom" && selectedCustomExpirations.includes(GEX_ZERO_DTE_ONLY_SENTINEL);
    const selectedSnapshotTradingDate = getSnapshotTradingDateIso(selectedGexSnapshot);
    const selectedRealExpirations = selectedCustomExpirations.filter((value) => value !== GEX_ZERO_DTE_ONLY_SENTINEL);
    if (hasZeroDteOnlySelection && !selectedSnapshotTradingDate) {
      setGexCurve([]);
      setGexLoading(false);
      return;
    }
    const expirationsForRequest =
      hasZeroDteOnlySelection && selectedSnapshotTradingDate
        ? [selectedSnapshotTradingDate]
        : selectedRealExpirations;
    fetchGexCurve(
      selectedGexSnapshot.snapshot_id,
      dteVal,
      selectedDte === "custom" ? expirationsForRequest : undefined,
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

  React.useEffect(() => {
    writeStorageString(GEX_UNDERLYING_STORAGE_KEY, selectedUnderlying);
  }, [selectedUnderlying]);

  React.useEffect(() => {
    writeStorageString(GEX_SOURCE_STORAGE_KEY, selectedSource);
  }, [selectedSource]);

  /**
   * Update selected underlying and reset stale snapshot-specific selections.
   */
  const handleSelectedUnderlyingChange = React.useCallback((value: string | null) => {
    const normalized = normalizeUnderlying(value);
    if (!(GEX_UNDERLYING_OPTIONS as readonly string[]).includes(normalized)) {
      return;
    }
    setSelectedUnderlying(normalized as GexUnderlying);
    setSelectedGexSnapshotState(null);
    setSelectedCustomExpirationsState([]);
    setGexDtes([]);
    setGexExpirations([]);
    setGexCurve([]);
  }, []);

  /**
   * Update selected source filter and reset stale snapshot-specific selections.
   */
  const handleSelectedSourceChange = React.useCallback((value: string | null) => {
    const raw = (value ?? "").trim();
    if (!(GEX_SOURCE_OPTIONS as readonly string[]).includes(raw)) {
      return;
    }
    setSelectedSource(raw as GexSource);
    setSelectedGexSnapshotState(null);
    setSelectedCustomExpirationsState([]);
    setGexDtes([]);
    setGexExpirations([]);
    setGexCurve([]);
  }, []);

  /**
   * Update DTE mode and clear stale custom selections when leaving custom mode.
   */
  const handleSelectedDteChange = React.useCallback((value: string) => {
    setSelectedDte(value);
    if (value !== "custom") {
      setSelectedCustomExpirationsState([]);
    }
  }, []);

  return {
    gexSnapshots,
    selectedGexSnapshot,
    setSelectedGexSnapshot,
    selectedUnderlying,
    handleSelectedUnderlyingChange,
    selectedSource,
    handleSelectedSourceChange,
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
