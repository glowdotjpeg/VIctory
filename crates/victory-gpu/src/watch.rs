//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

/// Mean cost of a query of a small span, nanoseconds. One query's cost is a
/// fixed part plus a walk over its pages, so callers choosing between many
/// narrow queries and one broad one need both this and the per page walk.
static SMALL_NS: AtomicU64 = AtomicU64::new(2_000);
/// Mean walk cost per page of a query of a big span, nanoseconds.
static BIG_NS: AtomicU64 = AtomicU64::new(85);

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let span = (last - first) * PAGE;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let took = started.elapsed();
    note_watch_cost(took, span);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, took.as_micros() as u64);
    }
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        Vec::new()
    };
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Records a query's cost: one average for small spans (mostly the fixed
/// part of a call), one per page for big spans (mostly the walk), so that
/// [`narrow_wins`] can compare splitting a query against walking it whole.
fn note_watch_cost(took: Duration, span: usize) {
    let ns = took.as_nanos() as u64;
    if span < 2 << 20 {
        ewma(&SMALL_NS, ns);
    } else if span >= 16 << 20 {
        let pages = (span / PAGE) as u64;
        ewma(&BIG_NS, ns / pages);
    }
}

/// Moves an average a sixteenth of the way to `sample`.
fn ewma(slot: &AtomicU64, sample: u64) {
    let _ = slot.fetch_update(Relaxed, Relaxed, |old| {
        Some(old.saturating_sub(old / 16) + sample.min(u64::MAX / 16) / 16)
    });
}

/// Whether querying `runs` separate spans that together cover `synced`
/// pages costs less than one query of `span` pages, by the costs measured
/// here lately. Used by a copy back that only needs the kernel's bits of
/// the pages it synced: asking for the whole declared range would walk
/// every page of it and clear the title's bits on the rest.
pub(crate) fn narrow_wins(runs: usize, synced: usize, span: usize) -> bool {
    narrow_wins_on(
        runs as u64,
        synced as u64,
        span as u64,
        SMALL_NS.load(Relaxed),
        BIG_NS.load(Relaxed),
    )
}

/// [`narrow_wins`] with the costs given: pages of each, `small` and `big`
/// in nanoseconds (a call, and its walk per page).
fn narrow_wins_on(runs: u64, synced: u64, span: u64, small: u64, big: u64) -> bool {
    runs * small + synced * big < small + span * big
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.Type != MEM_MAPPED {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    let mut g = guards();
    let blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .copied()
        .collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Drops the journal's entries for `[from, to)`: freed allocations
    /// leave theirs behind, and a fresh one at the same pages must not
    /// inherit them.
    fn forget(from: usize, to: usize) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<usize> = j
            .written
            .range(from / PAGE..to.div_ceil(PAGE))
            .map(|(&p, _)| p)
            .collect();
        for key in keys {
            j.written.remove(&key);
        }
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        forget(start, end);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// Querying a span leaves the kernel's bits of pages outside it: a copy
    /// back that queries only the pages it synced must not clear what the
    /// title wrote elsewhere in the range (the journal is no substitute; the
    /// kernel's bit is the only record of a write nobody has copied yet).
    #[test]
    fn a_narrow_query_leaves_neighbouring_bits_set() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        forget(start, end);
        writes(start, end, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + 3 * PAGE + 8) as *mut u8).write(1) };
        // The synced span only: what a copy back would query.
        let (now, _) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        let (later, pages) = writes(p + 3 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert!(later > now);
        // The journal can only date the write to the asking that found it.
        assert_eq!(pages, [(p / PAGE + 3, later)]);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Splitting a query beats walking the whole span when the runs' fixed
    /// costs are less than the walk the split saves; one run always wins,
    /// many expensive calls lose.
    #[test]
    fn narrow_queries_win_by_the_measured_costs() {
        let (small, big) = (2_000, 85);
        // One run of one page in a four page span: always less to walk.
        assert!(narrow_wins_on(1, 1, 4, small, big));
        // A dozen runs in a 174 MiB span, the world's copy back: cheap.
        assert!(narrow_wins_on(12, 12, 2_784, small, big));
        // Sixty-four runs of one page each are still less than walking
        // 174 MiB here.
        assert!(narrow_wins_on(64, 64, 2_784, small, big));
        // With calls costing a millisecond apiece, three narrow calls lose
        // to one broad one whose walk costs 100 ns a page.
        assert!(!narrow_wins_on(3, 3, 2_784, 1_000_000, 100));
        // And they win once the walk is dear enough per page.
        assert!(narrow_wins_on(3, 3, 2_784, 1_000_000, 100_000));
    }

    // Informational: whether a kernel write (`WriteProcessMemory`, what a
    // copy back does) leaves a write-watch bit. Print, do not assert.
    #[test]
    fn write_process_memory_leaves_a_bit_or_not() {
        use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        forget(p, p + 2 * PAGE);
        writes(p, p + 2 * PAGE, false).expect("watched");
        let mut done = 0;
        let bytes = [7u8; 16];
        // SAFETY: committed pages of ours.
        let ok = unsafe {
            WriteProcessMemory(
                GetCurrentProcess(),
                (p + PAGE) as _,
                bytes.as_ptr().cast(),
                bytes.len(),
                &mut done,
            )
        };
        let (_, pages) = writes(p, p + 2 * PAGE, false).expect("watched");
        println!(
            "WriteProcessMemory ok={ok} done={done}: write watch reports {} pages {pages:?}",
            pages.len()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    // cargo test --release -p victory-gpu benchmark_write_watch -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU benchmark, commits ~10 GiB"]
    fn benchmark_write_watch() {
        use std::hint::black_box;
        use std::time::Instant;

        // 0x20_0000, `MEM_WRITE_WATCH` (in SystemServices in windows-sys).
        const WATCH: u32 = 0x20_0000;
        // The title keeps 9-13 GiB write-watched; the walk's cost may depend
        // on the process's page tables, not just the span.
        const POOL: usize = 10 << 30;
        // SAFETY: committed and freed at the end of the test.
        let pool = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                POOL,
                MEM_RESERVE | MEM_COMMIT | WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(pool, 0);
        // Touch it all: the title's watched memory is committed and written.
        let started = Instant::now();
        for at in (0..POOL).step_by(4096) {
            // SAFETY: committed pages of ours.
            unsafe { *((pool + at) as *mut u8) = 1 };
        }
        println!("touched 10 GiB in {:?}", started.elapsed());
        // A first query of everything, so later ones measure steady state.
        let started = Instant::now();
        let (_, n) = writes(pool, pool + POOL, false).expect("watched");
        println!(
            "first whole query: {:?}, {} pages",
            started.elapsed(),
            n.len()
        );

        let median = |mut times: Vec<u128>| {
            times.sort_unstable();
            times[times.len() / 2]
        };
        // Per span: the walk's cost, with a few dirty pages to report.
        for mb in [1usize, 4, 16, 64, 174, 256] {
            let bytes = mb << 20;
            let (from, to) = (pool, pool + bytes);
            for page in (0..bytes / PAGE).step_by(bytes / PAGE / 8) {
                // SAFETY: committed pages of ours.
                unsafe { *((from + page * PAGE) as *mut u8) = 2 };
            }
            let mut calls = Vec::new();
            for _ in 0..21 {
                let started = Instant::now();
                black_box(writes(from, to, false).expect("watched"));
                calls.push(started.elapsed().as_micros());
            }
            let us = median(calls);
            println!(
                "span {mb:>4} MiB: {us:>5} us/call, {:.1} us/MiB, dirty 8 pages",
                us as f64 / mb as f64,
            );
        }
        // Dirty-page count at a fixed span.
        for dirty in [0usize, 64, 1024] {
            let (from, to) = (pool, pool + (64 << 20));
            for page in (0..dirty).map(|i| i * (64 << 20) / dirty.max(1) / PAGE) {
                // SAFETY: committed pages of ours.
                unsafe { *((from + page * PAGE) as *mut u8) = 3 };
            }
            let mut calls = Vec::new();
            for _ in 0..21 {
                let started = Instant::now();
                black_box(writes(from, to, false).expect("watched"));
                calls.push(started.elapsed().as_micros());
            }
            println!(
                "span 64 MiB, {} dirty pages: {} us/call",
                dirty,
                median(calls)
            );
        }
        // A copy back's choice: one broad query of the declared range, or
        // narrow ones over the pages it synced (here 12 pages in 3 runs of a
        // 174 MiB range, the world's big heap).
        let bytes = 174 << 20;
        let (from, to) = (pool, pool + bytes);
        let synced = [
            7usize, 8, 9, 600, 601, 1200, 1300, 1301, 1302, 1303, 2000, 2650,
        ];
        for page in synced {
            // SAFETY: committed pages of ours.
            unsafe { *((from + page * PAGE) as *mut u8) = 4 };
        }
        let mut broad = Vec::new();
        for _ in 0..21 {
            let started = Instant::now();
            black_box(writes(from, to, false).expect("watched"));
            broad.push(started.elapsed().as_micros());
        }
        let mut narrow = Vec::new();
        for _ in 0..21 {
            let started = Instant::now();
            let mut runs = Vec::new();
            let mut run: Option<(usize, usize)> = None;
            for &page in &synced {
                match run {
                    Some((a, b)) if b == page => run = Some((a, page + 1)),
                    Some(done) => {
                        runs.push(done);
                        run = Some((page, page + 1));
                    }
                    None => run = Some((page, page + 1)),
                }
            }
            runs.extend(run);
            for (a, b) in runs {
                black_box(writes(from + a * PAGE, from + b * PAGE, false).expect("watched"));
            }
            narrow.push(started.elapsed().as_micros());
        }
        println!(
            "copy back over 174 MiB, 12 synced pages in 3 runs: broad {} us, narrow {} us",
            median(broad),
            median(narrow),
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(pool as *mut _, 0, MEM_RELEASE) };
    }
}
