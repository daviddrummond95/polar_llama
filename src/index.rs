//! Persistent, incrementally updatable HNSW index (issue #82).
//!
//! `instant-distance`'s `HnswMap` (the same type `src/ann.rs`'s stateless
//! `knn_hnsw` expression builds fresh on every call) has no incremental
//! `insert`/`remove` -- it is an immutable graph produced once by
//! `Builder::build`. This module wraps that immutable graph in a small
//! LSM-like layer so the *index as a whole* supports add/remove without a
//! full rebuild on every call:
//!
//! - **main**: an immutable `HnswMap<EmbeddingPoint, u32>` (the same
//!   `EmbeddingPoint`/cosine-distance metric as `src/ann.rs`, reused
//!   verbatim -- `src/ann.rs` itself is untouched). Its `u32` values are
//!   *our* stable internal ids, not `instant_distance::PointId` (which is
//!   re-numbered on every rebuild).
//! - **staged**: a brute-force `Vec<(internal_id, EmbeddingPoint)>` holding
//!   points added since the last compaction. New points are queryable
//!   immediately (linear-scanned and merged with the `main` graph's
//!   results at query time), without touching `main`.
//! - **tombstones**: a `HashSet<u32>` of internal ids that used to be live
//!   (removed via `.remove()`, or superseded by a newer upsert of the same
//!   external id) but may still physically exist inside `main` or
//!   `staged`. Query time filters them out and over-fetches
//!   `k' = min(k + tombstones.len(), ef_search)` from `main` so genuinely
//!   live results still surface even when some of `main`'s nearest
//!   candidates turn out to be dead.
//! - **compaction**: rebuilds `main` from scratch out of every currently
//!   live point (`main`'s survivors, via `HnswMap::iter()`, plus `staged`),
//!   then clears `staged`/`tombstones`. Runs automatically
//!   (`auto_compact`, default on) once `staged.len()` or `tombstones.len()`
//!   crosses `max(compact_*_min, compact_*_ratio * main.len())`, or
//!   on-demand via `.compact()`.
//!
//! External ids are arbitrary, caller-supplied strings, stable across
//! add/remove/compact/save/load. Internally they're mapped to a private,
//! monotonically increasing `u32` internal-id space (`id_to_internal` /
//! `internal_to_id`) that both `main`'s values and `staged` are keyed by --
//! this is what lets a point move from `staged` into `main` at compaction
//! time (or get tombstoned) without disturbing any external-facing id.
//!
//! Persistence (`.save()`/`.load()`) serializes the whole
//! `PersistentIndex` (main graph, staged buffer, tombstones, id maps,
//! dimension, and the pinned build parameters -- including the HNSW
//! `seed`, so a reloaded-then-compacted index rebuilds deterministically)
//! with `bincode` 1.x behind an 8-byte magic + `u32` format-version header.
//! `bincode` is the *only* new dependency this feature adds (see
//! `Cargo.toml`) -- pure-Rust, serde-only, no unsafe transmute tricks.
//! `instant-distance`'s `with-serde` feature (already enabled) supplies
//! `Serialize`/`Deserialize` for `HnswMap`/`EmbeddingPoint` for free.
//!
//! The `#[pyclass]` boundary (`PyHnswIndex` / Python name `_HnswIndexCore`)
//! takes and returns `pyo3_polars::{PySeries, PyDataFrame}` directly --
//! ids and embeddings cross as real Polars columns (zero-copy Arrow, no
//! JSON shuttle), and batch queries release the GIL
//! (`Python::detach`, pyo3 0.27's current name for `allow_threads`) while
//! the actual HNSW/brute-force search runs. `polar_llama/index.py` wraps
//! this core with the DataFrame-native `HnswIndex` Python API (`.build`,
//! `.add`, `.remove`, `.query`, `.query_one`, `.save`/`.load`, `.knn`).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use instant_distance::{Builder, HnswMap, Point, Search};
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use serde::{Deserialize, Serialize};

use crate::ann::EmbeddingPoint;

/// File-format magic bytes, written at the very start of every `.save()`d
/// file. Guards against loading an unrelated file (or a future,
/// incompatible format) as an index.
const MAGIC: &[u8; 8] = b"PLHNSWI\0";
/// Bumped whenever the on-disk `bincode` layout of `PersistentIndex`
/// changes in a way that isn't forward/backward compatible.
const FORMAT_VERSION: u32 = 1;

fn default_ef_construction() -> usize {
    200
}
fn default_ef_search() -> usize {
    200
}
fn default_seed() -> u64 {
    42
}
fn default_compact_min() -> usize {
    1000
}
fn default_compact_ratio() -> f64 {
    0.10
}

/// Build-time / policy parameters, persisted alongside the graph so a
/// reloaded index compacts deterministically and with the same policy it
/// was built with.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexParams {
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
    /// Pinned `instant_distance::Builder` seed -- fixed (not re-randomized
    /// per build/compact) so the graph -- and therefore query results -- is
    /// reproducible for a given point set.
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_auto_compact")]
    pub auto_compact: bool,
    #[serde(default = "default_compact_min")]
    pub compact_staged_min: usize,
    #[serde(default = "default_compact_ratio")]
    pub compact_staged_ratio: f64,
    #[serde(default = "default_compact_min")]
    pub compact_tombstone_min: usize,
    #[serde(default = "default_compact_ratio")]
    pub compact_tombstone_ratio: f64,
}

fn default_auto_compact() -> bool {
    true
}

impl Default for IndexParams {
    fn default() -> Self {
        IndexParams {
            ef_construction: default_ef_construction(),
            ef_search: default_ef_search(),
            seed: default_seed(),
            auto_compact: default_auto_compact(),
            compact_staged_min: default_compact_min(),
            compact_staged_ratio: default_compact_ratio(),
            compact_tombstone_min: default_compact_min(),
            compact_tombstone_ratio: default_compact_ratio(),
        }
    }
}

/// A single point sitting in the brute-force staging buffer, keyed by our
/// own stable internal id (not `instant_distance::PointId`).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct StagedPoint {
    internal_id: u32,
    point: EmbeddingPoint,
}

/// The whole index: an immutable `main` HNSW graph, a brute-force `staged`
/// buffer of points added since the last compaction, a `tombstones` set of
/// internal ids that are no longer live, and the external<->internal id
/// maps. See the module docs for the full design.
#[derive(Serialize, Deserialize)]
pub struct PersistentIndex {
    pub dim: usize,
    pub params: IndexParams,
    main: Option<HnswMap<EmbeddingPoint, u32>>,
    staged: Vec<StagedPoint>,
    tombstones: HashSet<u32>,
    id_to_internal: HashMap<String, u32>,
    internal_to_id: HashMap<u32, String>,
    next_internal_id: u32,
}

impl PersistentIndex {
    /// Fresh build from a full `(id, embedding)` batch. All ids are
    /// assumed live going in; last-value-wins on a duplicate id within
    /// `pairs` (matching `.add()`'s upsert semantics -- internal ids are
    /// still assigned in first-occurrence order, but that's an
    /// implementation detail: the *set* of live points and every query
    /// result is identical to inserting the same rows one at a time via
    /// `.add()`).
    pub fn build(pairs: Vec<(String, Vec<f64>)>, params: IndexParams) -> Result<Self, String> {
        if pairs.is_empty() {
            return Err("build requires at least one (id, embedding) row".to_string());
        }
        let dim = pairs[0].1.len();
        if dim == 0 {
            return Err("embedding vectors must be non-empty".to_string());
        }
        for (i, (_, v)) in pairs.iter().enumerate() {
            if v.len() != dim {
                return Err(format!(
                    "embedding dimension mismatch at row {i}: expected dim={dim} \
                     (inferred from row 0), got dim={}",
                    v.len()
                ));
            }
        }

        // Last-id-wins dedup within the batch, preserving first-occurrence
        // order for internal-id assignment.
        let mut order: Vec<String> = Vec::with_capacity(pairs.len());
        let mut by_id: HashMap<String, Vec<f64>> = HashMap::with_capacity(pairs.len());
        for (id, v) in pairs {
            if !by_id.contains_key(&id) {
                order.push(id.clone());
            }
            by_id.insert(id, v);
        }

        let mut points = Vec::with_capacity(order.len());
        let mut values = Vec::with_capacity(order.len());
        let mut id_to_internal = HashMap::with_capacity(order.len());
        let mut internal_to_id = HashMap::with_capacity(order.len());
        for (internal_id, id) in order.into_iter().enumerate() {
            let v = by_id
                .remove(&id)
                .expect("id was just pushed into `order`, must still be in `by_id`");
            let internal_id = internal_id as u32;
            points.push(EmbeddingPoint(v));
            values.push(internal_id);
            id_to_internal.insert(id.clone(), internal_id);
            internal_to_id.insert(internal_id, id);
        }
        let next_internal_id = points.len() as u32;

        let main = Builder::default()
            .seed(params.seed)
            .ef_construction(params.ef_construction)
            .ef_search(params.ef_search)
            .build(points, values);

        Ok(PersistentIndex {
            dim,
            params,
            main: Some(main),
            staged: Vec::new(),
            tombstones: HashSet::new(),
            id_to_internal,
            internal_to_id,
            next_internal_id,
        })
    }

    /// Upsert `pairs` into the staging buffer (queryable immediately,
    /// merged with `main` at query time). An id that already exists (in
    /// `main` or `staged`) has its old internal id tombstoned and gets a
    /// fresh one -- `main` is immutable, so an upsert can never mutate a
    /// point already baked into it. Duplicate ids *within* `pairs` are
    /// resolved the same way, in order (last one wins). Returns the number
    /// of rows processed. Auto-compacts afterward if `params.auto_compact`
    /// and a threshold is crossed.
    pub fn add(&mut self, pairs: Vec<(String, Vec<f64>)>) -> Result<usize, String> {
        for (i, (_, v)) in pairs.iter().enumerate() {
            if v.is_empty() {
                return Err(format!("embedding vectors must be non-empty (row {i})"));
            }
            if v.len() != self.dim {
                return Err(format!(
                    "embedding dimension mismatch at row {i}: index dim={}, got dim={}",
                    self.dim,
                    v.len()
                ));
            }
        }

        let n = pairs.len();
        for (id, vec) in pairs {
            if let Some(&old_internal) = self.id_to_internal.get(&id) {
                self.tombstones.insert(old_internal);
            }
            let internal_id = self.next_internal_id;
            self.next_internal_id += 1;
            self.staged.push(StagedPoint {
                internal_id,
                point: EmbeddingPoint(vec),
            });
            self.id_to_internal.insert(id.clone(), internal_id);
            self.internal_to_id.insert(internal_id, id);
        }

        if self.params.auto_compact && self.should_compact() {
            self.compact();
        }
        Ok(n)
    }

    /// Tombstone every id in `ids` that is currently live. Ids that aren't
    /// currently live (unknown, or already removed) are silently skipped.
    /// Returns the number actually removed. Auto-compacts afterward if
    /// `params.auto_compact` and a threshold is crossed.
    pub fn remove(&mut self, ids: &[String]) -> usize {
        let mut n = 0usize;
        for id in ids {
            if let Some(internal_id) = self.id_to_internal.remove(id) {
                self.internal_to_id.remove(&internal_id);
                self.tombstones.insert(internal_id);
                n += 1;
            }
        }
        if self.params.auto_compact && self.should_compact() {
            self.compact();
        }
        n
    }

    /// Policy compaction trigger: `staged` or `tombstones` has grown past
    /// `max(min, ratio * main.len())`.
    pub fn should_compact(&self) -> bool {
        let main_len = self.main.as_ref().map(|m| m.values.len()).unwrap_or(0) as f64;
        let staged_threshold = (self.params.compact_staged_min as f64)
            .max(main_len * self.params.compact_staged_ratio) as usize;
        let tomb_threshold = (self.params.compact_tombstone_min as f64)
            .max(main_len * self.params.compact_tombstone_ratio) as usize;
        self.staged.len() > staged_threshold || self.tombstones.len() > tomb_threshold
    }

    /// Rebuild `main` from every currently live point (`main`'s survivors
    /// plus `staged`), then clear `staged`/`tombstones` and prune
    /// `internal_to_id` down to only-live entries. "Live" is derived from
    /// `id_to_internal`'s current values, independent of the (possibly
    /// stale, pre-compaction) `tombstones` set -- so this is correct even
    /// if `tombstones` under- or over-counts relative to reality.
    pub fn compact(&mut self) {
        let live: HashSet<u32> = self.id_to_internal.values().copied().collect();

        let mut points: Vec<EmbeddingPoint> = Vec::with_capacity(live.len());
        let mut values: Vec<u32> = Vec::with_capacity(live.len());

        if let Some(main) = &self.main {
            for (i, (_pid, point)) in main.iter().enumerate() {
                let internal_id = main.values[i];
                if live.contains(&internal_id) {
                    points.push(point.clone());
                    values.push(internal_id);
                }
            }
        }
        for sp in &self.staged {
            if live.contains(&sp.internal_id) {
                points.push(sp.point.clone());
                values.push(sp.internal_id);
            }
        }

        self.main = if points.is_empty() {
            None
        } else {
            Some(
                Builder::default()
                    .seed(self.params.seed)
                    .ef_construction(self.params.ef_construction)
                    .ef_search(self.params.ef_search)
                    .build(points, values),
            )
        };
        self.staged.clear();
        self.tombstones.clear();
        self.internal_to_id.retain(|k, _| live.contains(k));
    }

    /// k-nearest-neighbor search for one query vector: over-fetches from
    /// `main` to absorb tombstoned candidates, brute-force scans `staged`,
    /// merges, sorts by distance, and truncates to `k`. Returns
    /// `(external_id, distance)` pairs, nearest first.
    pub fn query_one(&self, query: &[f64], k: usize) -> Vec<(String, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let point = EmbeddingPoint(query.to_vec());
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        if let Some(main) = &self.main {
            let main_len = main.values.len();
            if main_len > 0 {
                let ef = self.params.ef_search.max(1);
                let over_fetch = k.saturating_add(self.tombstones.len());
                let k_main = over_fetch.min(ef).min(main_len);
                let mut search = Search::default();
                for item in main.search(&point, &mut search).take(k_main) {
                    if !self.tombstones.contains(item.value) {
                        candidates.push((*item.value, item.distance));
                    }
                }
            }
        }

        for sp in &self.staged {
            if self.tombstones.contains(&sp.internal_id) {
                continue;
            }
            candidates.push((sp.internal_id, point.distance(&sp.point)));
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        candidates
            .into_iter()
            .filter_map(|(internal_id, dist)| {
                self.internal_to_id
                    .get(&internal_id)
                    .map(|id| (id.clone(), dist))
            })
            .collect()
    }

    /// Batch form of `query_one`: `None` entries (null embedding rows)
    /// produce an empty result vector for that row, never an error.
    pub fn query_batch(&self, queries: &[Option<Vec<f64>>], k: usize) -> Vec<Vec<(String, f32)>> {
        queries
            .iter()
            .map(|q| match q {
                Some(v) => self.query_one(v, k),
                None => Vec::new(),
            })
            .collect()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.id_to_internal.contains_key(id)
    }

    /// Number of currently live points (survives add/remove/compact).
    pub fn len(&self) -> usize {
        self.id_to_internal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_internal.is_empty()
    }

    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    pub fn tombstone_len(&self) -> usize {
        self.tombstones.len()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::with_capacity(MAGIC.len() + 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        let body =
            bincode::serialize(self).map_err(|e| format!("failed to serialize index: {e}"))?;
        buf.extend_from_slice(&body);
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let header_len = MAGIC.len() + 4;
        if bytes.len() < header_len {
            return Err("not a valid HnswIndex file (too short)".to_string());
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err("not a valid HnswIndex file (bad magic bytes)".to_string());
        }
        let version = u32::from_le_bytes(
            bytes[MAGIC.len()..header_len]
                .try_into()
                .expect("slice of exactly 4 bytes"),
        );
        if version != FORMAT_VERSION {
            return Err(format!(
                "unsupported HnswIndex file format version {version} \
                 (this build supports version {FORMAT_VERSION})"
            ));
        }
        bincode::deserialize(&bytes[header_len..])
            .map_err(|e| format!("failed to deserialize index: {e}"))
    }

    /// Atomic write: serialize to a temp file in the same directory, then
    /// rename over the destination -- a crash mid-write can never leave a
    /// truncated/corrupt file at `path` (same convention as
    /// `polar_llama/checkpoint.py`'s store writes).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        let dir: PathBuf = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        fs::create_dir_all(&dir).map_err(|e| format!("failed to create directory {dir:?}: {e}"))?;
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("hnsw_index");
        let tmp_path = dir.join(format!(".{file_name}.tmp-{}", std::process::id()));
        fs::write(&tmp_path, &bytes)
            .map_err(|e| format!("failed to write {tmp_path:?}: {e}"))?;
        fs::rename(&tmp_path, path).map_err(|e| format!("failed to finalize {path:?}: {e}"))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;
        Self::from_bytes(&bytes)
    }
}

// ============================================================================
// Polars <-> Rust extraction helpers
// ============================================================================

/// Cast to `String` and collect, erroring (naming the offending row) on any
/// null.
fn extract_string_vec(series: &Series) -> PolarsResult<Vec<String>> {
    let casted = series.cast(&DataType::String)?;
    let ca = casted.str()?;
    let mut out = Vec::with_capacity(ca.len());
    for (i, v) in ca.into_iter().enumerate() {
        match v {
            Some(s) => out.push(s.to_string()),
            None => {
                return Err(PolarsError::ComputeError(
                    format!("id column contains a null value at row {i}").into(),
                ));
            }
        }
    }
    Ok(out)
}

/// Extract a `List[Float64]` (or any numeric list dtype, cast to Float64)
/// column into `Option<Vec<f64>>` per row -- `None` for a null row or an
/// empty vector. A vector with a null *inside* it is an error (naming the
/// row), since a partially-null embedding is never valid input.
fn extract_embeddings(series: &Series) -> PolarsResult<Vec<Option<Vec<f64>>>> {
    let list = series.list()?;
    let mut out = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        match list.get_as_series(i) {
            None => out.push(None),
            Some(s) => {
                if s.is_empty() {
                    out.push(None);
                    continue;
                }
                let casted = s.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                if ca.null_count() > 0 {
                    return Err(PolarsError::ComputeError(
                        format!("embedding vector at row {i} contains a null value").into(),
                    ));
                }
                out.push(Some(ca.into_no_null_iter().collect()));
            }
        }
    }
    Ok(out)
}

/// Same as `extract_embeddings`, but a null/empty row is an error (naming
/// the row) rather than `None` -- used for `.build()`/`.add()`, where every
/// row must contribute a real point.
fn extract_embeddings_required(series: &Series) -> PolarsResult<Vec<Vec<f64>>> {
    extract_embeddings(series)?
        .into_iter()
        .enumerate()
        .map(|(i, o)| {
            o.ok_or_else(|| {
                PolarsError::ComputeError(
                    format!("embedding column contains a null/empty value at row {i}").into(),
                )
            })
        })
        .collect()
}

fn extract_id_embedding_pairs(
    ids: &Series,
    embeddings: &Series,
) -> PolarsResult<Vec<(String, Vec<f64>)>> {
    if ids.len() != embeddings.len() {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "id column has {} rows, embedding column has {} rows",
                ids.len(),
                embeddings.len()
            )
            .into(),
        ));
    }
    let id_vals = extract_string_vec(ids)?;
    let emb_vals = extract_embeddings_required(embeddings)?;
    Ok(id_vals.into_iter().zip(emb_vals).collect())
}

fn build_query_result_df(results: &[Vec<(String, f32)>]) -> DataFrame {
    let mut query_ids: Vec<i64> = Vec::new();
    let mut neighbor_ids: Vec<String> = Vec::new();
    let mut distances: Vec<f64> = Vec::new();
    let mut ranks: Vec<u32> = Vec::new();

    for (qi, row) in results.iter().enumerate() {
        for (rank, (id, dist)) in row.iter().enumerate() {
            query_ids.push(qi as i64);
            neighbor_ids.push(id.clone());
            distances.push(*dist as f64);
            ranks.push((rank + 1) as u32);
        }
    }

    let height = query_ids.len();
    let columns = vec![
        Series::from_vec(PlSmallStr::from_static("query_id"), query_ids).into_column(),
        Series::new(PlSmallStr::from_static("neighbor_id"), neighbor_ids.as_slice())
            .into_column(),
        Series::from_vec(PlSmallStr::from_static("distance"), distances).into_column(),
        Series::from_vec(PlSmallStr::from_static("rank"), ranks).into_column(),
    ];
    DataFrame::new(height, columns).expect("columns constructed with matching lengths")
}

fn build_single_query_result_df(results: &[(String, f32)]) -> DataFrame {
    let mut neighbor_ids: Vec<String> = Vec::with_capacity(results.len());
    let mut distances: Vec<f64> = Vec::with_capacity(results.len());
    let mut ranks: Vec<u32> = Vec::with_capacity(results.len());
    for (rank, (id, dist)) in results.iter().enumerate() {
        neighbor_ids.push(id.clone());
        distances.push(*dist as f64);
        ranks.push((rank + 1) as u32);
    }

    let height = neighbor_ids.len();
    let columns = vec![
        Series::new(PlSmallStr::from_static("neighbor_id"), neighbor_ids.as_slice())
            .into_column(),
        Series::from_vec(PlSmallStr::from_static("distance"), distances).into_column(),
        Series::from_vec(PlSmallStr::from_static("rank"), ranks).into_column(),
    ];
    DataFrame::new(height, columns).expect("columns constructed with matching lengths")
}

fn to_pyerr(e: PolarsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ============================================================================
// PyO3 boundary
// ============================================================================

/// `#[pyclass]` core backing `polar_llama.index.HnswIndex`. Ids and
/// embeddings cross the Python boundary as real `pyo3_polars`
/// `PySeries`/`PyDataFrame` (zero-copy Arrow, no JSON shuttle); batch
/// queries release the GIL around the actual search.
#[pyclass(name = "_HnswIndexCore")]
pub struct PyHnswIndex {
    inner: PersistentIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[staticmethod]
    #[pyo3(signature = (
        ids,
        embeddings,
        ef_construction=200,
        ef_search=200,
        seed=42,
        auto_compact=true,
        compact_staged_min=1000,
        compact_staged_ratio=0.10,
        compact_tombstone_min=1000,
        compact_tombstone_ratio=0.10,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        ids: PySeries,
        embeddings: PySeries,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
        auto_compact: bool,
        compact_staged_min: usize,
        compact_staged_ratio: f64,
        compact_tombstone_min: usize,
        compact_tombstone_ratio: f64,
    ) -> PyResult<Self> {
        let pairs = extract_id_embedding_pairs(&ids.0, &embeddings.0).map_err(to_pyerr)?;
        let params = IndexParams {
            ef_construction,
            ef_search,
            seed,
            auto_compact,
            compact_staged_min,
            compact_staged_ratio,
            compact_tombstone_min,
            compact_tombstone_ratio,
        };
        let inner = PersistentIndex::build(pairs, params).map_err(PyValueError::new_err)?;
        Ok(PyHnswIndex { inner })
    }

    fn add(&mut self, ids: PySeries, embeddings: PySeries) -> PyResult<usize> {
        let pairs = extract_id_embedding_pairs(&ids.0, &embeddings.0).map_err(to_pyerr)?;
        self.inner.add(pairs).map_err(PyValueError::new_err)
    }

    fn remove(&mut self, ids: PySeries) -> PyResult<usize> {
        let ids = extract_string_vec(&ids.0).map_err(to_pyerr)?;
        Ok(self.inner.remove(&ids))
    }

    fn compact(&mut self) {
        self.inner.compact();
    }

    fn should_compact(&self) -> bool {
        self.inner.should_compact()
    }

    fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    fn query(&self, py: Python<'_>, embeddings: PySeries, k: usize) -> PyResult<PyDataFrame> {
        let queries = extract_embeddings(&embeddings.0).map_err(to_pyerr)?;
        // Validate every non-null query row's dimension up front. Without this,
        // a wrong-dim query silently returns garbage (EmbeddingPoint::distance
        // zips over the shorter length) on the primary DataFrame-native API --
        // mirror the single-vector query_one check. Null rows stay empty.
        for (row, q) in queries.iter().enumerate() {
            if let Some(v) = q {
                if v.len() != self.inner.dim {
                    return Err(PyValueError::new_err(format!(
                        "query vector at row {row} has dim={}, index dim={}",
                        v.len(),
                        self.inner.dim
                    )));
                }
            }
        }
        let inner = &self.inner;
        let results = py.detach(|| inner.query_batch(&queries, k));
        Ok(PyDataFrame(build_query_result_df(&results)))
    }

    fn query_one(&self, py: Python<'_>, vector: Vec<f64>, k: usize) -> PyResult<PyDataFrame> {
        if vector.is_empty() || vector.len() != self.inner.dim {
            return Err(PyValueError::new_err(format!(
                "query vector has dim={}, index dim={}",
                vector.len(),
                self.inner.dim
            )));
        }
        let inner = &self.inner;
        let results = py.detach(|| inner.query_one(&vector, k));
        Ok(PyDataFrame(build_single_query_result_df(&results)))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(Path::new(path))
            .map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = PersistentIndex::load(Path::new(path)).map_err(PyValueError::new_err)?;
        Ok(PyHnswIndex { inner })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn staged_len(&self) -> usize {
        self.inner.staged_len()
    }

    #[getter]
    fn tombstone_len(&self) -> usize {
        self.inner.tombstone_len()
    }

    #[getter]
    fn auto_compact(&self) -> bool {
        self.inner.params.auto_compact
    }

    #[setter]
    fn set_auto_compact(&mut self, value: bool) {
        self.inner.params.auto_compact = value;
    }
}

// ============================================================================
// Unit tests (pure Rust, no pyo3 -- runnable with plain `cargo test`)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(cx: f64, cy: f64, n: usize, spread: f64, seed: u64) -> Vec<Vec<f64>> {
        // Tiny deterministic LCG so tests don't need the `rand` crate.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64)
        };
        (0..n)
            .map(|_| {
                let dx = (next() - 0.5) * spread;
                let dy = (next() - 0.5) * spread;
                vec![cx + dx, cy + dy]
            })
            .collect()
    }

    fn brute_force_nearest(points: &[(String, Vec<f64>)], query: &[f64], k: usize) -> Vec<String> {
        let q = EmbeddingPoint(query.to_vec());
        let mut scored: Vec<(String, f32)> = points
            .iter()
            .map(|(id, v)| (id.clone(), q.distance(&EmbeddingPoint(v.clone()))))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn small_dataset() -> Vec<(String, Vec<f64>)> {
        let mut pairs = Vec::new();
        for (i, v) in blob(10.0, 0.0, 20, 0.5, 1).into_iter().enumerate() {
            pairs.push((format!("a{i}"), v));
        }
        for (i, v) in blob(0.0, 10.0, 20, 0.5, 2).into_iter().enumerate() {
            pairs.push((format!("b{i}"), v));
        }
        pairs
    }

    #[test]
    fn test_build_and_query_matches_brute_force_top1() {
        let pairs = small_dataset();
        let idx = PersistentIndex::build(pairs.clone(), IndexParams::default()).unwrap();

        let query = vec![10.0, 0.0];
        let got = idx.query_one(&query, 5);
        let want = brute_force_nearest(&pairs, &query, 5);

        assert_eq!(got.len(), 5);
        // Top-1 should agree (approximate search on a tiny, well-separated
        // dataset finds the exact nearest neighbor).
        assert_eq!(got[0].0, want[0]);
        // Every returned id came from the "a" blob (nearest cluster).
        assert!(got.iter().all(|(id, _)| id.starts_with('a')));
    }

    #[test]
    fn test_add_is_queryable_immediately_without_compaction() {
        let pairs = small_dataset();
        let mut idx = PersistentIndex::build(pairs, IndexParams::default()).unwrap();
        idx.params.auto_compact = false;

        idx.add(vec![("new1".to_string(), vec![10.0, 0.0])]).unwrap();
        assert_eq!(idx.staged_len(), 1);

        let got = idx.query_one(&[10.0, 0.0], 1);
        assert_eq!(got[0].0, "new1");
        assert!((got[0].1).abs() < 1e-6);
    }

    #[test]
    fn test_upsert_tombstones_old_internal_id() {
        // A second, always-live point near the origin so "the old position
        // is no longer reachable as x" is actually falsifiable -- with only
        // one point in the whole index, querying near its old position
        // would trivially still return it (nothing else to return).
        let mut idx = PersistentIndex::build(
            vec![
                ("x".to_string(), vec![0.0, 0.0]),
                ("other".to_string(), vec![0.1, 0.1]),
            ],
            IndexParams::default(),
        )
        .unwrap();
        idx.params.auto_compact = false;

        idx.add(vec![("x".to_string(), vec![100.0, 100.0])]).unwrap();
        assert_eq!(idx.tombstone_len(), 1);
        assert_eq!(idx.len(), 2);

        let got = idx.query_one(&[100.0, 100.0], 1);
        assert_eq!(got[0].0, "x");
        assert!((got[0].1).abs() < 1e-3);

        // The old position must no longer be reachable as "x" -- "other"
        // (still live near the origin) wins instead.
        let near_origin = idx.query_one(&[0.0, 0.0], 1);
        assert_eq!(near_origin[0].0, "other");
    }

    #[test]
    fn test_remove_excludes_id_and_k_still_returned() {
        let pairs = small_dataset();
        let mut idx = PersistentIndex::build(pairs, IndexParams::default()).unwrap();
        idx.params.auto_compact = false;

        let removed = idx.remove(&["a0".to_string(), "a1".to_string()]);
        assert_eq!(removed, 2);
        assert_eq!(idx.tombstone_len(), 2);
        assert!(!idx.contains("a0"));

        let got = idx.query_one(&[10.0, 0.0], 5);
        assert_eq!(got.len(), 5); // over-fetch backfills the removed slots
        assert!(got.iter().all(|(id, _)| id != "a0" && id != "a1"));
    }

    #[test]
    fn test_remove_unknown_id_is_a_noop() {
        let mut idx = PersistentIndex::build(
            vec![("x".to_string(), vec![0.0, 0.0])],
            IndexParams::default(),
        )
        .unwrap();
        assert_eq!(idx.remove(&["does-not-exist".to_string()]), 0);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_compaction_shrinks_staged_and_tombstones_and_preserves_query_results() {
        let pairs = small_dataset();
        let mut idx = PersistentIndex::build(pairs.clone(), IndexParams::default()).unwrap();
        idx.params.auto_compact = false;

        idx.add(vec![("c0".to_string(), vec![10.1, 0.1])]).unwrap();
        idx.remove(&["a0".to_string()]);
        assert_eq!(idx.staged_len(), 1);
        assert_eq!(idx.tombstone_len(), 1);

        let before = idx.query_one(&[10.0, 0.0], 5);

        idx.compact();
        assert_eq!(idx.staged_len(), 0);
        assert_eq!(idx.tombstone_len(), 0);
        assert!(!idx.contains("a0"));
        assert!(idx.contains("c0"));

        let after = idx.query_one(&[10.0, 0.0], 5);
        assert_eq!(before, after);
    }

    #[test]
    fn test_auto_compact_triggers_past_threshold() {
        let params = IndexParams {
            compact_staged_min: 3,
            ..Default::default()
        };
        let mut idx =
            PersistentIndex::build(vec![("x".to_string(), vec![0.0, 0.0])], params).unwrap();

        for i in 0..4 {
            idx.add(vec![(format!("y{i}"), vec![1.0, 1.0])]).unwrap();
        }
        // Should have auto-compacted at least once, so staged shouldn't
        // have been allowed to grow past the threshold indefinitely.
        assert!(idx.staged_len() <= 3);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn test_save_load_round_trip_preserves_query_results_and_pending_state() {
        let pairs = small_dataset();
        let mut idx = PersistentIndex::build(pairs, IndexParams::default()).unwrap();
        idx.params.auto_compact = false;
        idx.add(vec![("pending1".to_string(), vec![10.0, 0.05])]).unwrap();
        idx.remove(&["a0".to_string()]);

        let query = vec![10.0, 0.0];
        let before = idx.query_one(&query, 6);

        let dir = std::env::temp_dir().join(format!(
            "polar_llama_hnsw_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bin");
        idx.save(&path).unwrap();

        let loaded = PersistentIndex::load(&path).unwrap();
        assert_eq!(loaded.dim, idx.dim);
        assert_eq!(loaded.staged_len(), idx.staged_len());
        assert_eq!(loaded.tombstone_len(), idx.tombstone_len());
        assert_eq!(loaded.len(), idx.len());

        let after = loaded.query_one(&query, 6);
        assert_eq!(before, after);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_rejects_bad_magic() {
        let dir = std::env::temp_dir().join(format!(
            "polar_llama_hnsw_test_badmagic_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not_an_index.bin");
        std::fs::write(&path, b"not an hnsw index file at all").unwrap();

        let err = match PersistentIndex::load(&path) {
            Err(e) => e,
            Ok(_) => panic!("expected loading a bad-magic file to fail"),
        };
        assert!(err.contains("bad magic"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_build_rejects_empty_input() {
        let err = match PersistentIndex::build(Vec::new(), IndexParams::default()) {
            Err(e) => e,
            Ok(_) => panic!("expected building from an empty batch to fail"),
        };
        assert!(err.contains("at least one"));
    }

    #[test]
    fn test_build_rejects_dimension_mismatch() {
        let err = match PersistentIndex::build(
            vec![
                ("a".to_string(), vec![1.0, 2.0]),
                ("b".to_string(), vec![1.0, 2.0, 3.0]),
            ],
            IndexParams::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected a dimension mismatch to fail"),
        };
        assert!(err.contains("dimension mismatch"));
    }

    #[test]
    fn test_build_last_id_wins_on_duplicate() {
        let idx = PersistentIndex::build(
            vec![
                ("dup".to_string(), vec![0.0, 0.0]),
                ("dup".to_string(), vec![50.0, 50.0]),
            ],
            IndexParams::default(),
        )
        .unwrap();
        assert_eq!(idx.len(), 1);
        let got = idx.query_one(&[50.0, 50.0], 1);
        assert_eq!(got[0].0, "dup");
        assert!((got[0].1).abs() < 1e-3);
    }

    #[test]
    fn test_query_batch_null_row_returns_empty_not_error() {
        let idx = PersistentIndex::build(
            vec![("x".to_string(), vec![0.0, 0.0])],
            IndexParams::default(),
        )
        .unwrap();
        let out = idx.query_batch(&[Some(vec![0.0, 0.0]), None], 1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1);
        assert!(out[1].is_empty());
    }
}
