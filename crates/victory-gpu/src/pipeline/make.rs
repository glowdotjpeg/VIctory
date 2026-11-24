//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The workers make what a draw or dispatch waits for before what was
//! queued speculatively at the title's creation, so the loading-time
//! starts of a whole world's pipelines cannot leave a first use waiting
//! behind them.
//!
//! A first pass can miss the budget. If its pipeline is made later, it is
//! still saved as a candidate to start earlier on the next run.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    /// A use reached the frame's wait path. Written and read under `DONE`.
    waited: AtomicBool,
    /// Queued for a use (not still in the speculation): later uses do not
    /// look for it in the speculation again.
    demanded: AtomicBool,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            waited: AtomicBool::new(false),
            demanded: AtomicBool::new(false),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        let made = self
            .made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let (lock, done) = &DONE;
        let used = {
            let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
            done.notify_all();
            made.is_some() && self.waited.load(Ordering::Relaxed)
        };
        if used {
            self.remember_use();
        }
    }

    fn remember_use(&self) {
        if !self.used.swap(true, Ordering::Relaxed) {
            super::library::note_used(&self.key);
        }
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(500);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() {
        p.remember_use();
    }
    made
}

/// Queues `p` for the workers, once. A first use by a draw or dispatch
/// (`demand`) is made before the speculative starts; one already queued
/// speculatively is moved ahead of the rest of them.
fn start(p: &Arc<Pending>, demand: bool) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(
            WAITING.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );
        if demand {
            p.demanded.store(true, Ordering::Relaxed);
        }
        queue(p.clone());
    } else if demand && !p.demanded.swap(true, Ordering::Relaxed) {
        // Started speculatively at the title's creation: make it next.
        promote(p);
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let left = BUDGET.saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    p.waited.store(true, Ordering::Relaxed);
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// The workers' queues: first the pipelines a draw or dispatch waits for,
/// then those started speculatively at the title's creation, each in the
/// order queued. A pipeline is in at most one of the two, so a worker
/// makes it exactly once.
struct Queues {
    demand: VecDeque<Arc<Pending>>,
    spec: VecDeque<Arc<Pending>>,
}

impl Queues {
    const fn new() -> Queues {
        Queues {
            demand: VecDeque::new(),
            spec: VecDeque::new(),
        }
    }

    /// Classifies under the queue lock, including a first use that tried
    /// to promote this pipeline before the speculative start queued it.
    fn push(&mut self, p: Arc<Pending>) {
        if p.demanded.load(Ordering::Relaxed) {
            self.demand.push_back(p);
        } else {
            self.spec.push_back(p);
        }
    }

    /// The next pipeline to make: a waited-for one before a speculative
    /// one.
    fn pop(&mut self) -> Option<Arc<Pending>> {
        self.demand.pop_front().or_else(|| self.spec.pop_front())
    }

    /// A draw or dispatch waits for `p`, queued speculatively: it is made
    /// next, ahead of the rest of the speculation. `false` if it is not
    /// queued any more (made already, or in a worker's hands, which a
    /// first use then waits for as before). Called once per pipeline at
    /// most; a pipeline leaves the speculation only here.
    fn promote(&mut self, p: &Pending) -> bool {
        let Some(at) = self
            .spec
            .iter()
            .position(|q| std::ptr::eq(Arc::as_ptr(q), p as *const Pending))
        else {
            return false;
        };
        let p = self.spec.remove(at).unwrap();
        self.demand.push_back(p);
        true
    }
}

/// The workers' queues, shared by the title threads and the workers.
static QUEUES: (Mutex<Queues>, Condvar) = (Mutex::new(Queues::new()), Condvar::new());

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>) {
    static SPAWNED: OnceLock<usize> = OnceLock::new();
    let workers = *SPAWNED.get_or_init(spawn_workers);
    let (queues, not_empty) = &QUEUES;
    if workers == 0 {
        // No worker: made here instead.
        p.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
        return;
    }
    let mut held = queues.lock().unwrap_or_else(|e| e.into_inner());
    held.push(p);
    not_empty.notify_one();
}

/// Starts the workers, below the title's own threads (loading,
/// recording); returns how many were started.
fn spawn_workers() -> usize {
    let single_warp_worker = cfg!(feature = "diagnostics")
        && std::env::var_os("VICTORY_D3D12_WARP").is_some()
        && std::env::var_os("VICTORY_WARP_SERIAL_PSO").is_some();
    let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
    let workers = if single_warp_worker {
        tracing::info!("WARP pipeline creation limited to one worker");
        1
    } else {
        workers
    };
    let mut spawned = 0;
    for i in 0..workers {
        let spawned_ok = std::thread::Builder::new()
            .name(format!("victory-pipelines-{i}"))
            .spawn(worker);
        if spawned_ok.is_ok() {
            spawned += 1;
        } else {
            tracing::warn!(worker = i, "no pipeline worker");
        }
    }
    spawned
}

/// What a worker does: makes the next pipeline, forever.
fn worker() {
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    loop {
        let p = take();
        p.make();
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The next pipeline to make, waiting until there is one.
fn take() -> Arc<Pending> {
    let (queues, not_empty) = &QUEUES;
    let mut held = queues.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = held.pop() {
            return p;
        }
        held = not_empty.wait(held).unwrap_or_else(|e| e.into_inner());
    }
}

/// A draw or dispatch waits for `p`, started speculatively: hands it to a
/// worker next, ahead of the rest of the speculation.
fn promote(p: &Pending) -> bool {
    let (queues, not_empty) = &QUEUES;
    let mut held = queues.lock().unwrap_or_else(|e| e.into_inner());
    let promoted = held.promote(p);
    if promoted {
        not_empty.notify_one();
    }
    promoted
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending() -> Arc<Pending> {
        Arc::new(Pending::new(Vec::new(), None))
    }

    fn demanded() -> Arc<Pending> {
        let p = pending();
        p.demanded.store(true, Ordering::Relaxed);
        p
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
        assert!(!p.used.load(Ordering::Relaxed));
    }

    /// The speculation keeps its order, and a first use jumps it.
    #[test]
    fn a_waited_for_pipeline_is_made_before_the_speculative_ones() {
        let mut q = Queues::new();
        let spec: Vec<_> = (0..3).map(|_| pending()).collect();
        for p in &spec {
            q.push(p.clone());
        }
        let demanded = demanded();
        q.push(demanded.clone());
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &demanded));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[0]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[1]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[2]));
        assert!(q.pop().is_none());
    }

    /// Promoting a speculatively started pipeline makes it next, without
    /// dropping or duplicating any of the queues.
    #[test]
    fn promotion_jumps_the_speculation_without_touching_the_rest() {
        let mut q = Queues::new();
        let spec: Vec<_> = (0..4).map(|_| pending()).collect();
        for p in &spec {
            q.push(p.clone());
        }
        assert!(q.promote(&spec[1]));
        let demanded = demanded();
        q.push(demanded.clone());
        // Both waited-for ones before the rest of the speculation, in the
        // order they reached the demand queue: the promoted one first,
        // then the one queued straight into it.
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[1]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &demanded));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[0]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[2]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[3]));
        assert!(q.pop().is_none());
    }

    /// A pipeline a worker already holds (popped) or that is promoted
    /// twice cannot be queued again: promotion is a no-op, and no pipeline
    /// is ever in the queues twice.
    #[test]
    fn a_pipeline_in_a_workers_hands_or_promoted_twice_is_left_alone() {
        let mut q = Queues::new();
        let spec: Vec<_> = (0..3).map(|_| pending()).collect();
        for p in &spec {
            q.push(p.clone());
        }
        // A worker takes the head of the speculation.
        let in_flight = q.pop().unwrap();
        assert!(Arc::ptr_eq(&in_flight, &spec[0]));
        // A first use of it waits out that one compile; promotion finds
        // nothing to move.
        assert!(!q.promote(&spec[0]));
        // Another first use promotes the second; a third (same pipeline
        // again) finds it already moved.
        assert!(q.promote(&spec[1]));
        assert!(!q.promote(&spec[1]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[1]));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec[2]));
        assert!(q.pop().is_none());
    }

    #[test]
    fn a_first_use_before_speculative_insertion_keeps_its_priority() {
        let mut q = Queues::new();
        let spec = pending();
        q.push(spec.clone());

        let requested = pending();
        requested.queued.store(true, Ordering::Relaxed);
        requested.demanded.store(true, Ordering::Relaxed);
        assert!(!q.promote(&requested));
        q.push(requested.clone());

        assert!(Arc::ptr_eq(&q.pop().unwrap(), &requested));
        assert!(Arc::ptr_eq(&q.pop().unwrap(), &spec));
    }

    /// With workers draining the queues while first uses arrive over the
    /// speculation, every pipeline is handed out exactly once: promotion
    /// moves one, it does not copy it.
    #[test]
    fn promotion_holds_under_contention() {
        const N: u64 = 400;
        const DEMANDS_PER_THREAD: u64 = 200;
        let q = Arc::new((Mutex::new(Queues::new()), Condvar::new()));
        let spec: Vec<_> = (0..N).map(|_| pending()).collect();
        for p in &spec {
            q.0.lock().unwrap().push(p.clone());
        }
        // A worker busy with the head of the speculation: it cannot be
        // promoted or handed out again.
        let in_flight = q.0.lock().unwrap().pop().unwrap();
        assert!(Arc::ptr_eq(&in_flight, &spec[0]));

        let pushed = Arc::new(AtomicU64::new(0));
        let popped = Arc::new(Mutex::new(Vec::new()));
        let mut threads = Vec::new();
        // First-use threads: each queues new pipelines as demand and, at
        // the same time, promotes a pipeline from the middle of the
        // speculation (the second promotion of one finds nothing to move,
        // as in a run).
        for t in 0..2 {
            let q = q.clone();
            let pushed = pushed.clone();
            let spec = spec.clone();
            threads.push(std::thread::spawn(move || {
                for _ in 0..DEMANDS_PER_THREAD {
                    let mut held = q.0.lock().unwrap();
                    held.push(demanded());
                    pushed.fetch_add(1, Ordering::Relaxed);
                    held.promote(&spec[100 + t]);
                    drop(held);
                    q.1.notify_one();
                }
            }));
        }
        // Workers: hand out pipelines until the queues are empty and no
        // more are coming.
        let stop = Arc::new(AtomicBool::new(false));
        for _ in 0..2 {
            let q = q.clone();
            let popped = popped.clone();
            let stop = stop.clone();
            threads.push(std::thread::spawn(move || {
                loop {
                    let next = {
                        let mut held = q.0.lock().unwrap();
                        loop {
                            if let Some(p) = held.pop() {
                                break Some(p);
                            }
                            if stop.load(Ordering::Relaxed) {
                                break None;
                            }
                            held = q.1.wait(held).unwrap();
                        }
                    };
                    match next {
                        Some(p) => popped.lock().unwrap().push(p),
                        None => break,
                    }
                }
            }));
        }
        // No more first uses: what the workers have not handed out yet is
        // handed out here (the queues are drained under the same lock the
        // workers pop under).
        for t in threads.drain(0..2) {
            t.join().unwrap();
        }
        {
            let mut held = q.0.lock().unwrap();
            let mut handed = popped.lock().unwrap();
            while let Some(p) = held.pop() {
                handed.push(p);
            }
            stop.store(true, Ordering::Relaxed);
        }
        q.1.notify_all();
        for t in threads {
            t.join().unwrap();
        }
        let handed = popped.lock().unwrap();
        // The pipeline in the worker's hands is not handed out again.
        assert!(!handed.iter().any(|p| Arc::ptr_eq(p, &in_flight)));
        // Everything queued is handed out exactly once.
        let mut keys: Vec<*const Pending> = handed.iter().map(Arc::as_ptr).collect();
        keys.sort_unstable();
        let distinct = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), distinct, "a pipeline was handed out twice");
        assert_eq!(
            distinct,
            (N - 1) as usize + pushed.load(Ordering::Relaxed) as usize,
            "handed out fewer or more than queued"
        );
        assert_eq!(pushed.load(Ordering::Relaxed), 2 * DEMANDS_PER_THREAD);
    }
}
