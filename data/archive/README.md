# `data/archive/` -- supersession tombstones

Historical optimizer / pipeline outputs that are no longer the
canonical version but are kept on disk for forensics, regression
diffing, or "what did v1 look like before we changed X" investigations.

The whole subtree is **whitelisted** in `.gitignore` (see
`!data/archive/**` near the bottom of the file), so anything you drop
here will be tracked by git. Keep entries small unless the historical
content itself is the artifact (the v1 optimizer CSV below is the only
current exception).

## Contents

| File                            | Superseded by                                                                | Date archived | Why kept                                                                                                                            |
| ------------------------------- | ---------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `optimizer_event_only_v1.csv`   | `data/optimizer_event_only_v2.csv` + `data/optimizer_event_only_v2_explore.csv` | 2026-04-13    | Audit L3 -- 48 MB v1 optimizer output kept for cross-version Pareto comparison; not referenced by any `.py`. Safe to delete locally if disk is tight. |

## Adding a new tombstone

1. Move (don't copy) the stale artifact into `data/archive/<file>`.
2. Add a row to the table above with **what superseded it** and **why
   it's still useful**.
3. Update the corresponding row in `data/README.md` so the top-level
   index still resolves "where did `<file>` go?".
4. If the artifact is large (> ~10 MB), prefer leaving the data on the
   producer's local disk and committing only a marker file here that
   says "v1 optimizer ran on YYYY-MM-DD with config X, size Y MB,
   sha256 Z" -- the marker is the tombstone, the data does not need to
   travel through git.
