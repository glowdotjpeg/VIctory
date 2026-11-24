//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
}

#[derive(Default)]
struct Before {
    snapshot: Sparse,
    unchanged: UnchangedPages,
}

/// The copy's bytes before the submission ran, for the union of the ranges
/// its shaders write: sparse runs covering exactly the declared ranges
/// (merged where they meet or overlap), so disjoint views of a big heap do
/// not copy the gaps between them. Each byte is read once, before the
/// submission, and kept to tell later what the GPU wrote.
#[derive(Default)]
struct Sparse {
    /// `(start offset, bytes)` runs in offset order, disjoint and not
    /// adjacent.
    runs: Vec<(usize, Vec<u8>)>,
}

impl Sparse {
    /// Adds `[from, to)` (offsets in the copy) to the snapshot, reading only
    /// the newly covered bytes through `host`; bytes a run already holds are
    /// kept as first read, and runs that meet or overlap are merged.
    fn insert<'a>(
        &mut self,
        from: usize,
        to: usize,
        host: impl Fn(usize, usize) -> &'a [u8],
    ) {
        if from >= to {
            return;
        }
        // Already within one run: nothing new to read or merge (the world
        // declares the same ranges dozens of times a submission).
        if self.covering(from, to).is_some() {
            return;
        }
        // Runs ending before `from` stay put; the rest (through the first
        // starting after `to`) meet `[from, to)` and merge with it.
        let first = self.runs.partition_point(|&(at, ref bytes)| at + bytes.len() < from);
        let last = self.runs.partition_point(|&(at, _)| at <= to);
        let new_start = if first < last {
            from.min(self.runs[first].0)
        } else {
            from
        };
        let new_end = if first < last {
            let (at, ref bytes) = self.runs[last - 1];
            to.max(at + bytes.len())
        } else {
            to
        };
        let mut merged = pooled(new_end - new_start);
        let mut at = new_start;
        // Each run's bytes are kept; only the gaps around and between the
        // runs are read from the copy.
        for &(run_at, ref run) in &self.runs[first..last] {
            merged.extend_from_slice(host(at, run_at));
            merged.extend_from_slice(run);
            at = run_at + run.len();
        }
        merged.extend_from_slice(host(at, new_end));
        for (_, bytes) in self.runs.drain(first..last) {
            give_back(bytes);
        }
        self.runs.insert(first, (new_start, merged));
    }

    /// The run wholly covering `[from, to)`: its start offset and bytes.
    fn covering(&self, from: usize, to: usize) -> Option<(usize, &[u8])> {
        self.runs
            .iter()
            .find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            .map(|(at, bytes)| (*at, &bytes[..]))
    }

    /// The bounding offsets of every run (first start, last end), if any.
    fn span(&self) -> Option<(usize, usize)> {
        let (at, _) = self.runs.first()?;
        let (last, bytes) = self.runs.last()?;
        Some((*at, *last + bytes.len()))
    }

    /// Returns every run's bytes to the pool.
    fn clear(&mut self) {
        for (_, bytes) in self.runs.drain(..) {
            give_back(bytes);
        }
    }
}

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        let page = address / PAGE;
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if current != before {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + current.len() == ((page + 1) * PAGE).min(self.end);
        false
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

/// Ascending pages as inclusive-exclusive runs of neighbours (a copy back
/// queries each run's span, or the whole range when the runs are too many).
fn page_runs(pages: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &page in pages {
        match runs.last_mut() {
            Some((_, end)) if *end == page => *end = page + 1,
            _ => runs.push((page, page + 1)),
        }
    }
    runs
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as title addresses:
    /// what the copy stands for.
    pub(crate) fn clamped(&self, address: usize, len: usize) -> (usize, usize) {
        let (from, to) = self.inside(address, len);
        (self.start as usize + from, self.start as usize + to)
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, as sparse runs over exactly the
    /// declared ranges (the world declares dozens of disjoint views of its
    /// 174 MB heap; one bounding vector over them copied the gaps too), in
    /// reused memory (fresh snapshots cost ~2.3 million page faults per
    /// world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        self.before
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
            .insert(from, to, host);
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(
        &self,
        address: usize,
        len: usize,
        woken: &mut crate::fence::Woken,
    ) -> usize {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if from < to
            && let Some((taken, bytes)) = snapshot.covering(from, to)
        {
            if let Some((at, end)) = snapshot.span() {
                unchanged.begin(start + at, start + end);
            }
            let before = &bytes[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                if unchanged.differs(start + at, host, old)
                    && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                    && crate::host::read_into((start + at) as *const u8, title)
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        woken.add(crate::fence::write_and_collect(start + at, n, || {
                            crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        }));
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            // Asking the kernel about the whole range for this would walk
            // every page of it and clear the title's bits on the pages
            // nobody synced; asking about the synced pages' runs walks only
            // them, and what the title wrote elsewhere keeps its bit until
            // the next refresh asks (the journal has no record of a write
            // that has not been reported yet, so the bit is all there is).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last && self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
                synced.retain(|&p| (first..last).contains(&p));
                if !synced.is_empty() {
                    let runs = page_runs(&synced);
                    let base = start / PAGE;
                    if crate::watch::narrow_wins(runs.len(), synced.len(), last - first) {
                        let epochs = runs
                            .iter()
                            .map(|&(a, b)| {
                                crate::watch::writes(a * PAGE, b * PAGE, true).map(|(n, _)| n)
                            })
                            .collect::<Option<Vec<_>>>();
                        // Each page with the epoch of the query that covered
                        // it: an entry another user journals for it later,
                        // between the runs or after them, stays above. A
                        // failed query leaves the old epochs in place.
                        if let Some(epochs) = epochs {
                            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                            for ((a, b), epoch) in runs.into_iter().zip(epochs) {
                                copied[a - base..b - base].fill(epoch);
                            }
                        }
                    } else if let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
                    {
                        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                        for &page in &synced {
                            copied[page - base] = now;
                        }
                    }
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() {
            unchanged.clear();
            snapshot.clear();
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(all(test, feature = "diagnostics"))]
mod memory_wait_tests {
    use super::*;
    use crate::fence::{MemoryWait, Operand, Predicate, Woken};
    use crate::timeline::{Op, Timeline};
    use std::sync::atomic::AtomicU64;

    #[test]
    fn copy_back_keeps_a_transient_match_until_after_submission() {
        let (buffer, data) = host()
            .expect("D3D12")
            .shared_buffer(8)
            .expect("test buffer");
        let title = AtomicU64::new(0);
        let at = title.as_ptr() as usize;
        let shadow = Shadow::new(buffer, data, 0, at as u64, 8);
        let queue = Arc::new(Timeline::default());
        let (tx, rx) = std::sync::mpsc::channel();
        queue.submit_batch([
            Op::Memory(MemoryWait::new(Predicate {
                address: at,
                operand: Operand::Qword,
                comparison: 3,
                reference: 1,
            })),
            Op::Run(Box::new(move || tx.send(()).unwrap())),
        ]);
        let mut woken = Woken::default();
        // SAFETY: this test exclusively owns the mapped host buffer.
        unsafe { data.cast::<u64>().write(0) };
        for value in [1, 2] {
            shadow.writing(at, 8);
            // SAFETY: simulate the completed shader store into our host buffer.
            unsafe { data.cast::<u64>().write(value) };
            assert_eq!(shadow.copy_back(at, 8, &mut woken), 8);
        }
        assert_eq!(title.load(Ordering::Acquire), 2);
        assert!(rx.try_recv().is_err());
        woken.run();
        rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    #[test]
    fn neighbouring_pages_make_one_run() {
        assert_eq!(page_runs(&[]), []);
        assert_eq!(page_runs(&[7]), [(7, 8)]);
        assert_eq!(
            page_runs(&[5, 6, 7, 9, 20, 21]),
            [(5, 8), (9, 10), (20, 22)]
        );
    }

    /// The copy back only ever reads the snapshot's bytes for a declared
    /// range; proving those are identical to the old bounding-box snapshot's
    /// bytes (a single vector over the min..max range) proves the copy back
    /// writes identical bytes, for separated and overlapping declarations.
    #[test]
    fn sparse_matches_the_bounding_snapshot_byte_for_byte() {
        fn bounding(host: &[u8], ranges: &[Range<usize>]) -> (usize, Vec<u8>) {
            let from = ranges.iter().map(|r| r.start).min().unwrap();
            let to = ranges.iter().map(|r| r.end).max().unwrap();
            (from, host[from..to].to_vec())
        }

        let mut host = vec![0u8; 4 * PAGE];
        for (i, b) in host.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        for ranges in [
            // Separated declarations.
            vec![0..PAGE, 3 * PAGE..4 * PAGE],
            // Overlapping and adjacent declarations.
            vec![
                0..PAGE,
                PAGE / 2..PAGE + 100,
                PAGE..2 * PAGE,
                2 * PAGE..2 * PAGE + 16,
            ],
        ] {
            let mut snapshot = Sparse::default();
            for r in &ranges {
                snapshot.insert(r.start, r.end, |a, b| &host[a..b]);
            }
            let (from, bytes) = bounding(&host, &ranges);
            assert_eq!(snapshot.span(), Some((from, from + bytes.len())));
            // Every declared range reads the bounding box's exact bytes.
            for r in &ranges {
                let (at, run) = snapshot.covering(r.start, r.end).expect("covered");
                assert_eq!(
                    &run[r.start - at..r.end - at],
                    &bytes[r.start - from..r.end - from],
                    "range {r:?} differs from the bounding snapshot"
                );
            }
        }
    }

    /// Disjoint views of a big heap must not store the gaps between them:
    /// the runs cover the union of the declared ranges, not their bounding
    /// box.
    #[test]
    fn sparse_snapshot_does_not_copy_the_gaps() {
        let host = vec![0u8; 4 * PAGE];
        let ranges = [
            0..PAGE,
            PAGE / 2..PAGE,
            3 * PAGE..3 * PAGE + 500,
            3 * PAGE + 256..3 * PAGE + 999,
        ];
        let mut snapshot = Sparse::default();
        for r in &ranges {
            snapshot.insert(r.start, r.end, |a, b| &host[a..b]);
        }
        assert_eq!(snapshot.runs.len(), 2);
        assert_eq!(snapshot.span(), Some((0, 3 * PAGE + 999)));
        // Only the union is stored: two runs, no 2-page gap.
        let stored: usize = snapshot.runs.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(stored, PAGE + 999);
    }

    /// A later declaration that overlaps a run keeps the first read: it
    /// never re-reads bytes the snapshot already holds.
    #[test]
    fn overlapping_ranges_preserve_the_first_read() {
        let mut host = [0u8; 200];
        let mut snapshot = Sparse::default();
        snapshot.insert(0, 100, |a, b| &host[a..b]);
        // The copy changes before a later overlapping declaration (a
        // concurrent write); the first read is what must be kept.
        host[40..140].fill(0xFF);
        snapshot.insert(40, 120, |a, b| &host[a..b]);
        let (at, bytes) = snapshot.covering(0, 120).unwrap();
        assert_eq!(at, 0);
        assert_eq!(&bytes[0..40], &[0u8; 40]);
        assert_eq!(&bytes[40..100], &[0u8; 60], "overlap keeps the first read");
        assert_eq!(&bytes[100..120], &[0xFF; 20], "newly covered reads fresh");
    }

    /// Ranges that touch merge into one run; a declaration already covered
    /// reads nothing new.
    #[test]
    fn adjacent_ranges_merge_and_repeats_read_nothing() {
        let host = (0..200u8).collect::<Vec<_>>();
        let mut snapshot = Sparse::default();
        snapshot.insert(0, 100, |a, b| &host[a..b]);
        snapshot.insert(100, 200, |a, b| &host[a..b]);
        assert_eq!(snapshot.runs.len(), 1);
        let (_, bytes) = snapshot.covering(0, 200).unwrap();
        assert_eq!(bytes, &host[..]);
        // Re-declaring an already-covered range is a no-op.
        snapshot.insert(50, 150, |a, b| &host[a..b]);
        assert_eq!(snapshot.runs.len(), 1);
        assert_eq!(snapshot.covering(0, 200).unwrap().1, &host[..]);
    }

    /// An empty declaration (a write at the buffer's very end) must not add
    /// a far, zero-length run that pulls the coverage out to it.
    #[test]
    fn an_empty_range_does_not_grow_the_snapshot() {
        let host = vec![0u8; 2 * PAGE];
        let mut snapshot = Sparse::default();
        snapshot.insert(2 * PAGE, 2 * PAGE, |a, b| &host[a..b]);
        snapshot.insert(0, 64, |a, b| &host[a..b]);
        assert_eq!(snapshot.span(), Some((0, 64)));
        assert_eq!(snapshot.runs.len(), 1);
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }

    /// Times the old one-bounding-vector snapshot against the sparse one
    /// over `ranges`, five runs each, and prints the medians. Both use the
    /// shared pool, as the code under test does.
    fn bench_snapshot(label: &str, host: &[u8], ranges: &[Range<usize>]) {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        let mut baseline = [Duration::ZERO; 5];
        let mut sparse = [Duration::ZERO; 5];
        for (baseline, sparse) in baseline.iter_mut().zip(&mut sparse) {
            // The old `writing`: one bounding vector grown over every range.
            let started = Instant::now();
            let mut taken: Option<(usize, Vec<u8>)> = None;
            for r in ranges {
                let (from, to) = black_box((r.start, r.end));
                let (start, end) = match &taken {
                    Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
                    None => (from, to),
                };
                if let Some((at, bytes)) = &taken
                    && (*at, at + bytes.len()) == (start, end)
                {
                    continue;
                }
                let mut grown = pooled(end - start);
                match taken.take() {
                    Some((at, bytes)) => {
                        grown.extend_from_slice(&host[start..at]);
                        grown.extend_from_slice(&bytes);
                        grown.extend_from_slice(&host[at + bytes.len()..end]);
                        give_back(bytes);
                    }
                    None => grown.extend_from_slice(&host[start..end]),
                }
                taken = Some((start, grown));
            }
            black_box(&taken);
            *baseline = started.elapsed();
            if let Some((_, bytes)) = taken {
                give_back(bytes);
            }

            let started = Instant::now();
            let mut snapshot = Sparse::default();
            for r in ranges {
                snapshot.insert(black_box(r.start), black_box(r.end), |a, b| &host[a..b]);
            }
            black_box(&snapshot);
            *sparse = started.elapsed();
        }
        baseline.sort_unstable();
        sparse.sort_unstable();
        println!(
            "median of five runs, {label}: bounding {:?} -> sparse {:?} ({:.1}x)",
            baseline[2],
            sparse[2],
            baseline[2].as_secs_f64() / sparse[2].as_secs_f64(),
        );
    }

    // cargo test --release -p victory-gpu benchmark_disjoint_snapshot -- --ignored --nocapture
    /// The sparse snapshot against the old one-bounding-vector snapshot: a
    /// handful of small disjoint views of a 174 MiB heap. The old version
    /// copies the whole gap between the views every time one is declared;
    /// the sparse version reads only the views.
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_disjoint_snapshot() {
        const SIZE: usize = 174 << 20;
        let host = vec![0xA5u8; SIZE];
        let ranges: Vec<_> = (0..24)
            .map(|i| {
                let at = i * (SIZE / 24) + i;
                at..at + 256
            })
            .collect();
        bench_snapshot("24 disjoint 256 B views across 174 MiB", &host, &ranges);
    }

    // cargo test --release -p victory-gpu benchmark_contiguous_snapshot -- --ignored --nocapture
    /// The sparse snapshot's least favourable case: contiguous ranges that
    /// keep growing one run. Both schemes copy the accumulated bytes each
    /// time, so sparse's run lookup and merge overhead shows here; it should
    /// stay within noise of the bounding baseline, not get worse.
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_contiguous_snapshot() {
        let host = vec![0xA5u8; 24 * PAGE];
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE..(i + 1) * PAGE).collect();
        bench_snapshot("24 contiguous 64 KiB ranges", &host, &ranges);
    }
}

/// Copy back against real write-watched title memory (a device is needed
/// for the copy itself, so the diagnostics build and, without a GPU,
/// `VICTORY_D3D12_WARP=1`).
#[cfg(all(test, feature = "diagnostics"))]
mod copy_back_tests {
    use super::*;
    use crate::fence::Woken;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    // 0x20_0000, `MEM_WRITE_WATCH` (in SystemServices in windows-sys).
    const WATCH: u32 = 0x20_0000;

    #[test]
    fn copy_back_keeps_the_title_writes_of_untouched_pages() {
        let Some(host) = host() else { return };
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(title, 0);
        let (buffer, data) = host
            .shared_buffer(4 * PAGE as u64)
            .expect("test buffer");
        let shadow = Shadow::new(buffer, data, 0, title as u64, 4 * PAGE as u64);
        let mut woken = Woken::default();

        shadow.writing(title, 4 * PAGE);
        // The shaders write page 1 of the copy while the list runs.
        // SAFETY: the test owns this mapping.
        unsafe { data.add(PAGE + 8).write(0xFF) };
        // The title's threads write pages 2 and 3 of title memory meanwhile.
        // SAFETY: committed pages of ours.
        unsafe {
            (title as *mut u8).add(2 * PAGE + 8).write(0x22);
            (title as *mut u8).add(3 * PAGE + 8).write(0x11);
        }

        let written = shadow.copy_back(title, 4 * PAGE, &mut woken);
        assert_eq!(written, PAGE, "only the shaders' page is copied back");
        // SAFETY: committed pages of ours.
        assert_eq!(unsafe { (title as *const u8).add(PAGE + 8).read() }, 0xFF);
        // SAFETY: the test owns this mapping.
        assert_eq!(unsafe { data.add(2 * PAGE + 8).read() }, 0, "untouched");
        // The title's page 3 write was not reported by the copy back's
        // narrow queries, so its bit must still be there for the next ask.
        let (_, reported) =
            crate::watch::writes(title + 3 * PAGE, title + 4 * PAGE, false).expect("watched");
        assert_eq!(reported.len(), 1, "page 3's write is still reported");
        assert_eq!(reported[0].0, title / PAGE + 3);

        // The next refresh takes the title's page 3 over (its bit, or the
        // journal a broader query left).
        assert_eq!(shadow.update(title + 3 * PAGE, PAGE), PAGE);
        // SAFETY: the test owns this mapping.
        assert_eq!(unsafe { data.add(3 * PAGE + 8).read() }, 0x11);
        // The synced page is current: the next refresh does not recopy it.
        assert_eq!(shadow.update(title + PAGE, PAGE), 0);

        // Another copy of the same title memory: the journal tells it what
        // the copy back wrote and the kernel what the title wrote.
        let (other, other_data) = host
            .shared_buffer(4 * PAGE as u64)
            .expect("test buffer");
        let other = Shadow::new(other, other_data, 0, title as u64, 4 * PAGE as u64);
        let copied = other.update(title, 4 * PAGE);
        // The copy back's page (through the journal) and the title's two
        // (one through the journal, one through the kernel's bit); the
        // journal may also carry older entries for pages a freed
        // allocation had at these addresses, so page 0 may come along.
        assert!(copied >= 3 * PAGE);
        // SAFETY: the test owns this mapping; the reads are of pages the
        // copy back and the title wrote.
        let (a, b, c) = unsafe {
            (
                other_data.add(PAGE + 8).read(),
                other_data.add(2 * PAGE + 8).read(),
                other_data.add(3 * PAGE + 8).read(),
            )
        };
        assert_eq!(a, 0xFF);
        assert_eq!(b, 0x22);
        assert_eq!(c, 0x11);

        // SAFETY: our allocation.
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
    }

    /// Two separated declared ranges (and a concurrent title write between
    /// them) copy back exactly their own pages through the sparse snapshot;
    /// the pages the shaders did not touch keep the title's writes.
    #[test]
    fn copy_back_matches_for_separated_ranges() {
        let Some(host) = host() else { return };
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                8 * PAGE,
                MEM_RESERVE | MEM_COMMIT | WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(title, 0);
        let (buffer, data) = host
            .shared_buffer(8 * PAGE as u64)
            .expect("test buffer");
        let shadow = Shadow::new(buffer, data, 0, title as u64, 8 * PAGE as u64);
        let mut woken = Woken::default();

        shadow.writing(title + PAGE, PAGE);
        shadow.writing(title + 5 * PAGE, PAGE);
        // The shaders write one byte of each declared page.
        // SAFETY: the test owns this mapping.
        unsafe {
            data.add(PAGE + 8).write(0xAA);
            data.add(5 * PAGE + 8).write(0xCC);
        }
        // The title's threads write pages 2 and 6 meanwhile (untouched).
        // SAFETY: committed pages of ours.
        unsafe {
            (title as *mut u8).add(2 * PAGE + 8).write(0x22);
            (title as *mut u8).add(6 * PAGE + 8).write(0x11);
        }

        assert_eq!(shadow.copy_back(title + PAGE, PAGE, &mut woken), PAGE);
        assert_eq!(shadow.copy_back(title + 5 * PAGE, PAGE, &mut woken), PAGE);
        // SAFETY: the title pages the copy back wrote.
        assert_eq!(unsafe { (title as *const u8).add(PAGE + 8).read() }, 0xAA);
        // SAFETY: the title pages the copy back wrote.
        assert_eq!(unsafe { (title as *const u8).add(5 * PAGE + 8).read() }, 0xCC);
        // The pages the shaders did not write keep the title's bytes.
        // SAFETY: committed pages of ours, untouched by the copy back.
        assert_eq!(unsafe { (title as *const u8).add(2 * PAGE + 8).read() }, 0x22);
        // SAFETY: committed pages of ours, untouched by the copy back.
        assert_eq!(unsafe { (title as *const u8).add(6 * PAGE + 8).read() }, 0x11);

        // SAFETY: our allocation.
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
    }

    /// `gpu_writes` clamps a declared length to the bytes left at `address`
    /// before it is deferred, so a declaration past the buffer's end becomes
    /// length 0 and one cut short by it keeps only the inside bytes. The
    /// pending range is keyed by that clamped `(address, address + len)` on
    /// both sides (`writing` pushes it, `copy_back` retains it), so every
    /// declaration is cleared exactly once however far out of range it is.
    #[test]
    fn a_clamped_declaration_clears_its_pending_range() {
        let Some(host) = host() else { return };
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(title, 0);
        let (buffer, data) = host
            .shared_buffer(2 * PAGE as u64)
            .expect("test buffer");
        let shadow = Shadow::new(buffer, data, 0, title as u64, 2 * PAGE as u64);
        let mut woken = Woken::default();

        let pending = || {
            shadow
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
        };

        // `gpu_writes(title + 2 * PAGE, PAGE)` clamps to `left(...) = 0`;
        // the deferred range and its pending key are `(title + 2 * PAGE, ...)`.
        let past = title + 2 * PAGE;
        shadow.writing(past, 0);
        assert_eq!(pending(), 1);
        assert_eq!(shadow.copy_back(past, 0, &mut woken), 0);
        assert_eq!(pending(), 0, "a length-0 declaration leaves no pending key");

        // `gpu_writes(title + PAGE, 2 * PAGE)` clamps to the `PAGE` bytes
        // inside; the pending key uses the clamped length.
        let tail = title + PAGE;
        shadow.writing(tail, PAGE);
        assert_eq!(shadow.copy_back(tail, PAGE, &mut woken), 0);
        assert_eq!(pending(), 0, "a cut-short declaration clears its key");

        // SAFETY: our allocation.
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
    }
}
