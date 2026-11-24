//! Fences: a 64-bit value and the events and queues waiting for it. Three
//! things move a value: the title's CPU signals, our queues' signals (a
//! queue's work runs to completion before its signal, so a queue not held
//! back by a wait signals at once) and bottom-of-pipe writes a command list
//! makes at the fence's GPU address. All three wake the waits, so an
//! event, a blocking wait or a queue's wait ends once the value it asked
//! for is reached, whichever moved it.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;
use crate::timeline::Timeline;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
    /// Wakes blocking waits (`SetEventOnCompletion` with a null event)
    /// whenever the value moves.
    moved: Condvar,
    /// Queues holding work back until the value reaches theirs
    /// (`ID3D12CommandQueue::Wait`). Locked after `waiters`, never before.
    queues: Mutex<Vec<(u64, Weak<Timeline>)>>,
}

impl Drop for Fence {
    fn drop(&mut self) {
        live().remove(&((&raw const self.value) as usize));
    }
}

/// Live fences by the address of their value (their GPU address), for
/// bottom-of-pipe writes that land on one. Locked before a fence's
/// `waiters`, never after.
static LIVE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn live() -> MutexGuard<'static, BTreeMap<usize, usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// How often a blocking wait looks at the value without being woken (a
/// value moved by a path we do not see), and how long it waits before
/// saying it is stuck.
const RECHECK: Duration = Duration::from_millis(100);
const STALL: Duration = Duration::from_secs(5);
/// How often the background watcher re-checks live fences while at least one
/// event or queue wait is pending (for direct CPU/shader memory writes).
const WATCH_POLL: Duration = Duration::from_millis(10);

struct Watcher {
    /// Incremented whenever an event or queue wait is added. Pure leaf lock:
    /// never held while acquiring `LIVE`, fence locks, or a timeline lock.
    generation: Mutex<u64>,
    added: Condvar,
}

static WATCHER: Watcher = Watcher {
    generation: Mutex::new(0),
    added: Condvar::new(),
};

fn watch_notify() {
    static THREAD: std::sync::Once = std::sync::Once::new();
    {
        let mut g = WATCHER.generation.lock().unwrap_or_else(|e| e.into_inner());
        *g = g.wrapping_add(1);
        WATCHER.added.notify_one();
    }
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-fence-watch".into())
            .spawn(watch_loop);
    });
}

/// The two list methods have distinct operand widths. Only Wait32 takes a mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operand {
    Dword(u32),
    Qword,
}

impl Operand {
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::Dword(_) => 4,
            Self::Qword => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Predicate {
    pub address: usize,
    pub operand: Operand,
    pub comparison: u32,
    pub reference: u64,
}

impl Predicate {
    fn matches(self, value: u64) -> bool {
        let value = match self.operand {
            Operand::Dword(mask) => u64::from(value as u32 & mask),
            Operand::Qword => value,
        };
        // D3D12_COMPARISON_FUNC, using unsigned operands of the guest width.
        match self.comparison {
            1 => false,
            2 => value < self.reference,
            3 => value == self.reference,
            4 => value <= self.reference,
            5 => value > self.reference,
            6 => value != self.reference,
            7 => value >= self.reference,
            8 => true,
            _ => unreachable!("comparison validated at record time"),
        }
    }

    /// The reachable masked values are exactly the subsets of the mask.
    /// Classify only predicates for which no possible word can satisfy the wait.
    fn can_match(self) -> bool {
        let max = match self.operand {
            Operand::Dword(mask) => u64::from(mask),
            Operand::Qword => u64::MAX,
        };
        match self.comparison {
            1 => false,
            2 => self.reference > 0,
            3 => self.reference & !max == 0,
            4 | 8 => true,
            5 => max > self.reference,
            6 => max != 0 || self.reference != 0,
            7 => max >= self.reference,
            _ => unreachable!("comparison validated at record time"),
        }
    }

    fn reached(self, fences: &MutexGuard<'_, BTreeMap<usize, usize>>) -> bool {
        // Under LIVE the fence cannot finish dropping. Read its owned atomic,
        // including either dword, without probing a guest pointer.
        let base = self.address & !(size_of::<u64>() - 1);
        if let Some(&ptr) = fences.get(&base) {
            // SAFETY: the registry guard covers the entire load.
            let word = unsafe { &*(ptr as *const Fence) }
                .value
                .load(Ordering::Acquire);
            return self.matches(word >> (8 * (self.address - base)));
        }
        #[cfg(windows)]
        {
            // The title may unmap a pending operand. Read through the OS, as
            // AndyLatimer did in a9a7985, while preserving the wait's width.
            let mut bytes = [0; 8];
            crate::host::read_into(
                self.address as *const u8,
                &mut bytes[..self.operand.bytes()],
            ) && self.matches(u64::from_le_bytes(bytes))
        }
        #[cfg(all(not(windows), test))]
        {
            // Non-Windows has no GPU list producer. The unit tests use live,
            // naturally aligned atomic words until their wait completes.
            let value = unsafe {
                match self.operand {
                    Operand::Dword(_) => u64::from(
                        (&*(self.address as *const std::sync::atomic::AtomicU32))
                            .load(Ordering::Acquire),
                    ),
                    Operand::Qword => {
                        (&*(self.address as *const AtomicU64)).load(Ordering::Acquire)
                    }
                }
            };
            self.matches(value)
        }
        #[cfg(all(not(windows), not(test)))]
        {
            false
        }
    }
}

/// Latch per execution, allocated only if the queue actually has to wait.
pub(crate) struct MemoryWait {
    predicate: Predicate,
    observed: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// Submitted work outlives the guest queue handle. An impossible predicate has no possible
/// producer, so only its diagnostic needs to follow the queue's lifetime.
enum WaitingQueue {
    Reachable(Arc<Timeline>),
    Impossible(Weak<Timeline>),
}

struct Pending {
    predicate: Predicate,
    observed: Arc<std::sync::atomic::AtomicBool>,
    queue: WaitingQueue,
    since: Instant,
    warned: bool,
}

static MEMORY: Mutex<Vec<Pending>> = Mutex::new(Vec::new());

#[cfg(feature = "diagnostics")]
pub(crate) fn trace_memory_waits() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_TRACE_MEMORY_WAITS").is_some())
}

impl MemoryWait {
    pub(crate) fn new(predicate: Predicate) -> Self {
        Self {
            predicate,
            observed: None,
        }
    }

    pub(crate) fn is_fresh_repeat(&self, predicate: Predicate) -> bool {
        self.observed.is_none() && self.predicate == predicate
    }

    pub(crate) fn register(
        &mut self,
        queue: &Arc<Timeline>,
        on_fresh_match: impl FnOnce(Predicate),
    ) -> bool {
        if let Some(observed) = &self.observed {
            let reached = observed.load(Ordering::Acquire);
            #[cfg(feature = "diagnostics")]
            if reached && trace_memory_waits() {
                tracing::info!(
                    timeline = ?Arc::as_ptr(queue),
                    wait = ?self.predicate,
                    "title queue resumed after its memory wait"
                );
            }
            return reached;
        }
        let fences = live();
        let mut pending = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let possible = self.predicate.can_match();
        if possible && self.predicate.reached(&fences) {
            on_fresh_match(self.predicate);
            return true;
        }
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.observed = Some(observed.clone());
        pending.push(Pending {
            predicate: self.predicate,
            observed,
            queue: if possible {
                WaitingQueue::Reachable(queue.clone())
            } else {
                WaitingQueue::Impossible(Arc::downgrade(queue))
            },
            since: Instant::now(),
            warned: false,
        });
        drop(pending);
        drop(fences);
        #[cfg(feature = "diagnostics")]
        if trace_memory_waits() {
            tracing::info!(
                timeline = ?Arc::as_ptr(queue),
                wait = ?self.predicate,
                "title queue paused for an unsatisfied memory wait"
            );
        }
        watch_notify();
        false
    }
}

fn scan_memory(
    fences: &MutexGuard<'_, BTreeMap<usize, usize>>,
    pending: &mut Vec<Pending>,
    range: Option<std::ops::Range<usize>>,
) -> (Vec<Arc<Timeline>>, bool) {
    let mut ready = Vec::new();
    pending.retain_mut(|entry| {
        let wait = entry.predicate;
        let overlaps = range.as_ref().is_none_or(|r| {
            r.start < wait.address.saturating_add(wait.operand.bytes()) && wait.address < r.end
        });
        if let WaitingQueue::Impossible(queue) = &entry.queue
            && queue.strong_count() == 0
        {
            return false;
        }
        if let WaitingQueue::Reachable(queue) = &entry.queue
            && overlaps && wait.reached(fences)
        {
            entry.observed.store(true, Ordering::Release);
            ready.push(queue.clone());
            return false;
        }
        if range.is_none() && !entry.warned && entry.since.elapsed() >= STALL {
            tracing::warn!(?wait, elapsed = ?entry.since.elapsed(), "GPU queue still waiting for memory");
            entry.warned = true;
        }
        true
    });
    (ready, !pending.is_empty())
}

fn watch_loop() {
    loop {
        let seen = *WATCHER.generation.lock().unwrap_or_else(|e| e.into_inner());
        let (mut woken, fence_waiting) = scan_live_fences();
        let (memory_ready, memory_waiting) = {
            let fences = live();
            let mut pending = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            scan_memory(&fences, &mut pending, None)
        };
        woken.extend(memory_ready);
        let any_waiting = fence_waiting || memory_waiting;
        if !woken.is_empty() {
            run_held(woken);
            continue;
        }
        let g = WATCHER.generation.lock().unwrap_or_else(|e| e.into_inner());
        if *g != seen {
            continue;
        }
        if any_waiting {
            drop(
                WATCHER
                    .added
                    .wait_timeout(g, WATCH_POLL)
                    .unwrap_or_else(|e| e.into_inner()),
            );
        } else {
            drop(WATCHER.added.wait(g).unwrap_or_else(|e| e.into_inner()));
        }
    }
}

/// Checks every live fence for event or queue waits whose value has been
/// reached outside `Signal` / bottom-of-pipe writes (for example a CPU write
/// to the fence's memory).
fn scan_live_fences() -> (Vec<Arc<Timeline>>, bool) {
    let live = live();
    let mut woken = Vec::new();
    let mut any_waiting = false;
    for &ptr in live.values() {
        // SAFETY: registered fences stay alive until `Fence::drop`, which
        // takes the `LIVE` lock we hold here.
        let fence = unsafe { &*(ptr as *const Fence) };
        let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
        let mut queues = fence.queues.lock().unwrap_or_else(|e| e.into_inner());
        if waiters.is_empty() && queues.is_empty() {
            continue;
        }
        let (more, moved) = wake_locked(fence, &mut waiters, &mut queues);
        if moved {
            fence.moved.notify_all();
        }
        woken.extend(more);
        any_waiting |= !waiters.is_empty() || !queues.is_empty();
    }
    (woken, any_waiting)
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
        moved: Condvar::new(),
        queues: Mutex::new(Vec::new()),
    });
    let at = (&raw const fence.value) as usize;
    live().insert(at, (&raw const *fence) as usize);
    // Dropped (and so unregistered) here if the title asked for an
    // interface a fence does not have.
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if !event.is_null() {
        waiters.push((value, event as usize));
        drop(waiters);
        watch_notify();
        return S_OK;
    }
    // A null event: block until the value is reached, as D3D12 does.
    let started = Instant::now();
    let mut warned = Duration::ZERO;
    while fence.value.load(Ordering::Acquire) < value {
        waiters = fence
            .moved
            .wait_timeout(waiters, RECHECK)
            .unwrap_or_else(|e| e.into_inner())
            .0;
        let waited = started.elapsed();
        if waited >= warned + STALL {
            warned = waited;
            tracing::warn!(
                value,
                completed = fence.value.load(Ordering::Acquire),
                seconds = waited.as_secs(),
                "blocking fence wait still waiting"
            );
        }
    }
    S_OK
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    let at = fence.value.as_ptr() as usize;
    write_and_collect(at, size_of::<u64>(), || {
        fence.value.store(value, Ordering::Release)
    })
    .run();
    S_OK
}

/// Wakes any events and queues on `fence` whose value has been reached;
/// returns the timelines to resume once locks are dropped and whether
/// anything changed.
#[must_use]
fn wake_locked(
    fence: &Fence,
    waiters: &mut Vec<(u64, usize)>,
    queues: &mut Vec<(u64, Weak<Timeline>)>,
) -> (Vec<Arc<Timeline>>, bool) {
    let value = fence.value.load(Ordering::Acquire);
    let mut moved = false;
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
            moved = true;
        }
        !reached
    });
    let mut woken = Vec::new();
    queues.retain(|(wanted, queue)| {
        if *wanted <= value {
            woken.extend(queue.upgrade());
            moved = true;
            false
        } else {
            queue.strong_count() > 0
        }
    });
    (woken, moved)
}

/// Sets the events whose value the fence has reached, wakes the blocking
/// waits and returns the queues to wake, for [`run_held`] once no fence
/// lock is held.
#[must_use]
fn wake(fence: &Fence, mut waiters: MutexGuard<'_, Vec<(u64, usize)>>) -> Vec<Arc<Timeline>> {
    let mut queues = fence.queues.lock().unwrap_or_else(|e| e.into_inner());
    let (woken, _) = wake_locked(fence, &mut waiters, &mut queues);
    fence.moved.notify_all();
    woken
}

/// Runs the work queues held back until a value now reached. With no fence
/// lock held: the work may move fences itself.
fn run_held(queues: Vec<Arc<Timeline>>) {
    for queue in queues {
        queue.wake();
    }
}

/// Queues a bottom-of-pipe write woke, to run once the writer holds no
/// lock: their work submits lists itself (`Host::execute`'s lock).
#[derive(Default)]
#[must_use = "run the woken queues once no lock is held"]
pub(crate) struct Woken(Vec<Arc<Timeline>>);

impl Woken {
    pub(crate) fn add(&mut self, other: Woken) {
        self.0.extend(other.0);
    }

    pub(crate) fn run(self) {
        run_held(self.0);
    }
}

/// A reference to a fence held by a queue's work, given back on drop.
pub(crate) struct Held(object::Held);

impl Held {
    pub(crate) fn new(this: This) -> Held {
        Held(object::Held::new(this))
    }

    pub(crate) fn reached(&self, value: u64) -> bool {
        fence(self.0.get()).value.load(Ordering::Acquire) >= value
    }

    pub(crate) fn signal(&self, value: u64) {
        signal(self.0.get(), value);
    }

    /// Whether the value is reached; if not, `queue` is woken once it is.
    /// Checked under the lock a signal takes, so no signal is missed.
    pub(crate) fn wait_on_gpu(&self, value: u64, queue: &Arc<Timeline>) -> bool {
        let fence = fence(self.0.get());
        let mut queues = fence.queues.lock().unwrap_or_else(|e| e.into_inner());
        if fence.value.load(Ordering::Acquire) >= value {
            return true;
        }
        let queue = Arc::downgrade(queue);
        if !queues.iter().any(|(v, q)| *v == value && q.ptr_eq(&queue)) {
            queues.push((value, queue));
        }
        drop(queues);
        watch_notify();
        false
    }
}

/// Commit the write and remove satisfied waiters under the same registry
/// locks used by the polling thread. Otherwise the watcher can steal a wake
/// between the store and notification while Host::execute still holds its lock.
pub(crate) fn write_and_collect(at: usize, bytes: usize, store: impl FnOnce()) -> Woken {
    if bytes == 0 {
        store();
        return Woken::default();
    }
    let end = at.saturating_add(bytes);
    let live = live();
    let mut pending = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    store();
    let (mut ready, _) = scan_memory(&live, &mut pending, Some(at..end));
    drop(pending);
    // A copied range may cover more than one fence.
    for (_, &ptr) in live.range(at.saturating_sub(size_of::<u64>() - 1)..end) {
        // SAFETY: the LIVE lock prevents deregistration during collection.
        let fence = unsafe { &*(ptr as *const Fence) };
        ready.extend(wake(
            fence,
            fence.waiters.lock().unwrap_or_else(|e| e.into_inner()),
        ));
    }
    Woken(ready)
}

#[cfg(test)]
fn written(at: usize, bytes: usize) -> Woken {
    write_and_collect(at, bytes, || {})
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn memory_predicates_use_unsigned_width_and_mask_only_the_operand() {
        let matches = |operand, comparison, value, reference| {
            Predicate {
                address: 0,
                operand,
                comparison,
                reference,
            }
            .matches(value)
        };
        for (comparison, less, equal, greater) in [
            (1, false, false, false),
            (2, true, false, false),
            (3, false, true, false),
            (4, true, true, false),
            (5, false, false, true),
            (6, true, false, true),
            (7, false, true, true),
            (8, true, true, true),
        ] {
            for (operand, maximum) in [
                (Operand::Dword(u32::MAX), u64::from(u32::MAX)),
                (Operand::Qword, u64::MAX),
            ] {
                assert_eq!(matches(operand, comparison, 0, maximum), less);
                assert_eq!(matches(operand, comparison, maximum, maximum), equal);
                assert_eq!(matches(operand, comparison, maximum, 0), greater);
            }
        }
        assert!(matches(Operand::Dword(0xff), 3, 0x1_8000_ab10, 0x10));
        assert!(!matches(Operand::Dword(0xff), 3, 0xab10, 0xab10));
        assert!(matches(Operand::Dword(0), 3, u64::MAX, 0));
        assert!(matches(Operand::Qword, 5, 1 << 63, (1 << 63) - 1));
    }

    #[test]
    fn never_wait_releases_its_timeline_and_warns_without_resuming() {
        let timeline = Arc::new(Timeline::default());
        let weak = Arc::downgrade(&timeline);
        let word = AtomicU64::new(1);
        let predicate = Predicate {
            address: word.as_ptr() as usize,
            operand: Operand::Qword,
            comparison: 1,
            reference: 1,
        };
        let mut wait = MemoryWait::new(predicate);
        assert!(!wait.register(&timeline, |_| {}));
        let mut pending = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let ours = |entry: &Pending| entry.predicate.address == predicate.address;
        let index = pending.iter().position(ours).unwrap();
        // Use the real scan and threshold, without sleeping or scanning other tests' entries.
        let mut isolated = vec![pending.swap_remove(index)];
        drop(pending);
        isolated[0].since = Instant::now() - STALL;
        assert!(scan_memory(&live(), &mut isolated, None).0.is_empty());
        assert!(isolated[0].warned);
        assert!(!wait.register(&timeline, |_| {}));
        drop(timeline);
        assert!(weak.upgrade().is_none());
        assert!(!scan_memory(&live(), &mut isolated, None).1);
    }

    #[test]
    fn impossibility_depends_only_on_the_mask_and_unsigned_domain() {
        for mask in 0..16 {
            for reference in 0..=16 {
                for comparison in 1..=8 {
                    let p = Predicate {
                        operand: Operand::Dword(mask),
                        reference,
                        comparison,
                        address: 0,
                    };
                    assert_eq!(p.can_match(), (0..16).any(|value| p.matches(value)));
                }
            }
        }
        let p = Predicate {
            operand: Operand::Qword,
            reference: u64::MAX,
            comparison: 5,
            address: 0,
        };
        assert!(!p.can_match());
        assert!(
            Predicate {
                reference: u64::MAX - 1,
                ..p
            }
            .can_match()
        );
        assert!(
            !Predicate {
                comparison: 2,
                reference: 0,
                ..p
            }
            .can_match()
        );
    }

    #[test]
    fn a_previous_latch_cannot_satisfy_the_next_identical_wait() {
        use crate::timeline::Op;
        let word = AtomicU64::new(0);
        let predicate = Predicate {
            address: word.as_ptr() as usize,
            operand: Operand::Qword,
            comparison: 3,
            reference: 1,
        };
        let queue = Arc::new(Timeline::default());
        let (tx, rx) = std::sync::mpsc::channel();
        queue.submit_batch([
            Op::Memory(MemoryWait::new(predicate)),
            Op::Memory(MemoryWait::new(predicate)),
            Op::Run(Box::new(move || tx.send(()).unwrap())),
        ]);
        let write = |value| crate::queue::Written::Value(value).write(word.as_ptr().cast(), 8);
        let first = write(1);
        write(2).run();
        first.run();
        assert!(matches!(
            rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        write(1).run();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn unreadable_operand_stays_pending_until_remapped() {
        use crate::timeline::Op;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        // SAFETY: we own this reservation throughout the test.
        let page = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4096,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!page.is_null());
        let p = Predicate {
            address: page as usize,
            operand: Operand::Qword,
            comparison: 3,
            reference: 1,
        };
        let q = Arc::new(Timeline::default());
        let (tx, rx) = std::sync::mpsc::channel();
        q.submit_batch([
            Op::Memory(MemoryWait::new(p)),
            Op::Run(Box::new(move || tx.send(()).unwrap())),
        ]);
        // SAFETY: decommit keeps the same address reserved.
        assert_ne!(unsafe { VirtualFree(page, 4096, MEM_DECOMMIT) }, 0);
        for operand in [Operand::Dword(u32::MAX), Operand::Qword] {
            assert!(
                !Predicate {
                    operand,
                    reference: 0,
                    ..p
                }
                .reached(&live())
            );
        }
        assert!(rx.try_recv().is_err());
        assert_eq!(
            // SAFETY: recommit our own reserved page before writing it.
            unsafe { VirtualAlloc(page, 4096, MEM_COMMIT, PAGE_READWRITE) },
            page
        );
        crate::queue::Written::Value(1).write(page.cast(), 8).run();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // SAFETY: the wait has completed and no thread reads this reservation.
        assert_ne!(unsafe { VirtualFree(page, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn live_fence_operands_read_the_selected_dword() {
        let fence = new_fence(0xaabbccdd_11223344);
        let at = address(fence);
        for (address, operand, reference) in [
            (at, Operand::Qword, 0xaabbccdd_11223344),
            (at, Operand::Dword(0xffff), 0x3344),
            (at + 4, Operand::Dword(0xffff), 0xccdd),
        ] {
            assert!(
                Predicate {
                    address,
                    operand,
                    comparison: 3,
                    reference
                }
                .reached(&live())
            );
        }
        release_fence(fence);
    }

    #[cfg(windows)]
    #[test]
    fn cpu_and_queue_signals_latch_address_waits_and_allow_reentry() {
        use crate::timeline::Op;
        for queued in [false, true] {
            let f = new_fence(0);
            let (producer, signal_slot) = crate::queue::test_queue();
            let consumer = Arc::new(Timeline::default());
            let again = Held::new(f);
            let (tx, rx) = std::sync::mpsc::channel();
            consumer.submit_batch([
                Op::Memory(MemoryWait::new(Predicate {
                    address: address(f),
                    operand: Operand::Qword,
                    comparison: 3,
                    reference: 1,
                })),
                Op::Run(Box::new(move || {
                    again.signal(2);
                    tx.send(()).unwrap();
                })),
            ]);
            if queued {
                let signal =
                    slot::<extern "system" fn(This, This, u64) -> HRESULT>(producer, signal_slot);
                assert_eq!(signal(producer, f, 1), S_OK);
                assert_eq!(signal(producer, f, 3), S_OK);
            } else {
                assert_eq!(signal(f, 1), S_OK);
                assert_eq!(signal(f, 3), S_OK);
            }
            rx.recv_timeout(Duration::from_secs(2))
                .expect("missed fence signal");
            slot::<extern "system" fn(This) -> u32>(producer, 0x10)(producer);
            release_fence(f);
        }
    }

    #[test]
    fn racing_memory_write_and_registration_resume_exactly_once() {
        use crate::timeline::Op;
        use std::sync::Barrier;
        for _ in 0..128 {
            let timeline = Arc::new(Timeline::default());
            let word = AtomicU64::new(0);
            let start = Barrier::new(2);
            let (tx, rx) = std::sync::mpsc::channel();
            let wait = MemoryWait::new(Predicate {
                address: word.as_ptr() as usize,
                operand: Operand::Qword,
                comparison: 3,
                reference: u64::MAX,
            });
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    start.wait();
                    crate::queue::Written::Value(u64::MAX)
                        .write(word.as_ptr().cast(), 8)
                        .run();
                });
                start.wait();
                timeline.submit_batch([
                    Op::Memory(wait),
                    Op::Run(Box::new(move || tx.send(()).unwrap())),
                ]);
                rx.recv_timeout(Duration::from_secs(2)).expect("lost wake");
            });
            assert!(matches!(
                rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Disconnected)
            ));
        }
    }

    type Wait = extern "system" fn(This, u64, *mut c_void) -> HRESULT;
    type Signal = extern "system" fn(This, u64) -> HRESULT;

    fn new_fence(initial: u64) -> This {
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), initial, 0, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        f
    }

    fn release_fence(f: This) {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    fn address(f: This) -> usize {
        slot::<extern "system" fn(This) -> u64>(f, 0x60)(f) as usize
    }

    /// A fence pointer that may cross threads (the fence is `Sync`).
    #[derive(Clone, Copy)]
    struct Shared(This);
    // SAFETY: fences are made for use from many threads.
    unsafe impl Send for Shared {}

    /// Runs a blocking wait for `value` on another thread; the receiver
    /// gets its result once it returns.
    fn blocking_wait(f: This, value: u64) -> std::sync::mpsc::Receiver<HRESULT> {
        let (done, result) = std::sync::mpsc::channel();
        let f = Shared(f);
        std::thread::spawn(move || {
            let f = f;
            let wait = slot::<Wait>(f.0, 0x48);
            let _ = done.send(wait(f.0, value, std::ptr::null_mut()));
        });
        result
    }

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(1);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<Wait>(f, 0x48);
        let signal = slot::<Signal>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // Grand Theft Auto VI reads the value at the fence's address.
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(address(f) as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[test]
    fn blocking_wait_returns_at_once_when_reached() {
        let f = new_fence(7);
        let wait = slot::<Wait>(f, 0x48);
        assert_eq!(wait(f, 7, std::ptr::null_mut()), S_OK);
        assert_eq!(wait(f, 3, std::ptr::null_mut()), S_OK);
        release_fence(f);
    }

    /// Used to answer `E_NOTIMPL` at once, so the title went on before the
    /// value was reached.
    #[test]
    fn blocking_wait_blocks_until_signalled() {
        let f = new_fence(0);
        let result = blocking_wait(f, 2);
        let short = Duration::from_millis(200);
        assert!(
            result.recv_timeout(short).is_err(),
            "returned before any signal"
        );
        let signal = slot::<Signal>(f, 0x50);
        assert_eq!(signal(f, 1), S_OK);
        assert!(
            result.recv_timeout(short).is_err(),
            "returned below its value"
        );
        assert_eq!(signal(f, 2), S_OK);
        assert_eq!(result.recv_timeout(Duration::from_secs(5)), Ok(S_OK));
        release_fence(f);
    }

    #[test]
    fn bottom_of_pipe_write_releases_blocking_wait() {
        let f = new_fence(0);
        let result = blocking_wait(f, 0x1_0000_0003);
        std::thread::sleep(Duration::from_millis(50));
        // The low half alone does not reach the value.
        crate::queue::Written::Value(3)
            .write(address(f) as *mut u8, 4)
            .run();
        assert!(result.recv_timeout(Duration::from_millis(200)).is_err());
        crate::queue::Written::Value(0x1_0000_0003)
            .write(address(f) as *mut u8, 8)
            .run();
        assert_eq!(result.recv_timeout(Duration::from_secs(5)), Ok(S_OK));
        release_fence(f);
    }

    #[cfg(windows)]
    #[test]
    fn bottom_of_pipe_write_sets_waiting_event() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(0);
        // SAFETY: plain manual-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        assert_eq!(slot::<Wait>(f, 0x48)(f, 9, event), S_OK);
        crate::queue::Written::Value(9)
            .write(address(f) as *mut u8, 8)
            .run();
        // SAFETY: a valid event.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[cfg(windows)]
    #[test]
    fn direct_memory_write_sets_waiting_event() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(0);
        // SAFETY: plain manual-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        assert_eq!(slot::<Wait>(f, 0x48)(f, 11, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // Direct write to the fence's GPU address, bypassing `Signal` and
        // `Written::write`.
        // SAFETY: `address(f)` is `Fence::value`, an `AtomicU64`.
        unsafe { (*(address(f) as *const AtomicU64)).store(11, Ordering::Release) };
        // SAFETY: a valid event.
        assert_eq!(unsafe { WaitForSingleObject(event, 2_000) }, WAIT_OBJECT_0);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[test]
    fn released_fences_leave_the_registry() {
        let f = new_fence(0);
        let at = address(f);
        assert!(live().contains_key(&at));
        release_fence(f);
        assert!(!live().contains_key(&at));
        // A write next to (not on) a live fence's value wakes nothing and
        // does not fault.
        let g = new_fence(0);
        let mut other = [0u64; 2];
        written(other.as_mut_ptr() as usize, 16).run();
        written(address(g) + 8, 8).run();
        release_fence(g);
    }

    /// A bottom-of-pipe write at a fence a held queue waits for does not
    /// run that queue's work inside the write: the writer (`Host::execute`)
    /// still holds its submission lock, which the work takes again. The old
    /// code ran it there (a deadlock on the real lock).
    #[test]
    fn bottom_of_pipe_writes_leave_woken_work_to_the_writer() {
        use crate::timeline::{Op, Timeline};
        let f = new_fence(0);
        let submission = Arc::new(Mutex::new(()));
        let ran = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let queue = Arc::new(Timeline::default());
        queue.submit(Op::Wait(Held::new(f), 1));
        let (lock, seen) = (submission.clone(), ran.clone());
        queue.submit(Op::Run(Box::new(move || {
            let free = lock.try_lock().is_ok();
            seen.store(if free { 1 } else { 2 }, Ordering::SeqCst);
        })));
        let held = submission.lock().unwrap();
        let woken = crate::queue::Written::Value(1).write(address(f) as *mut u8, 8);
        assert_eq!(ran.load(Ordering::SeqCst), 0, "ran inside the write");
        drop(held);
        woken.run();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        release_fence(f);
    }
}
