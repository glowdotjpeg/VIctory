//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Vec<u8>>>,
    /// Shared with submissions still to run (queues held back by a wait).
    #[cfg(windows)]
    host: std::sync::Mutex<Option<std::sync::Arc<crate::host::HostList>>>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
    #[cfg(windows)]
    prefixes: std::sync::Mutex<Vec<Prefix>>,
    #[cfg(windows)]
    recorded: std::sync::atomic::AtomicBool,
}

#[cfg(windows)]
#[derive(Clone)]
pub(crate) struct Prefix {
    /// None for a leading wait or a wait immediately following another.
    pub list: Option<std::sync::Arc<crate::host::HostList>>,
    pub predicate: crate::fence::Predicate,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

#[cfg(windows)]
fn recycle_prefixes(prefixes: &mut Vec<Prefix>) {
    for prefix in prefixes.drain(..) {
        if let Some(segment) = prefix.list
            && let Ok(segment) = std::sync::Arc::try_unwrap(segment)
            && let Some(host) = crate::host::host()
        {
            host.recycle_list(segment);
        }
    }
}

#[cfg(windows)]
impl Drop for List {
    fn drop(&mut self) {
        recycle_prefixes(self.prefixes.get_mut().unwrap_or_else(|e| e.into_inner()));
        let uploads = std::mem::take(self.uploads.get_mut().unwrap_or_else(|e| e.into_inner()));
        if let Some(pending) = self
            .host
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            && std::sync::Arc::strong_count(pending) > 1
        {
            pending.keep_uploads(uploads);
            let dynamic = self
                .state
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_dynamic();
            pending.keep_dynamic(dynamic);
        }
    }
}

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(&list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// Get the native list for recording, conservatively marking its segment nonempty.
#[cfg(windows)]
pub(crate) fn host_list(this: This) -> Option<std::sync::Arc<crate::host::HostList>> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    // Native commands can be recorded through this handle. A wait after
    // even an opaque native command must preserve the complete prefix.
    list.recorded
        .store(true, std::sync::atomic::Ordering::Relaxed);
    list.host.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    _pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    _pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, iid, out)
}

fn new_list(kind: u32, allocator: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        host: std::sync::Mutex::new(crate::host::host().and_then(|h| match h.acquire_list() {
            Ok(list) => Some(std::sync::Arc::new(list)),
            Err(e) => {
                tracing::warn!(error = %e, "no host command list");
                None
            }
        })),
        #[cfg(windows)]
        state: Default::default(),
        #[cfg(windows)]
        prefixes: Default::default(),
        #[cfg(windows)]
        recorded: Default::default(),
    });
    object::hand_out(LIST, list, iid, &LIST_IIDS, out)
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    #[cfg(windows)]
    recycle_prefixes(&mut list.prefixes.lock().unwrap_or_else(|e| e.into_inner()));
    let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
    // A submission of the list still to run (its queue waits for a fence)
    // keeps the host list, its upload memory and its dynamic descriptor
    // slots; recording goes on in a pooled or new list, open already.
    #[cfg(windows)]
    let (fresh, keep_pending) = {
        let mut host = list.host.lock().unwrap_or_else(|e| e.into_inner());
        match host.as_ref() {
            Some(pending) if std::sync::Arc::strong_count(pending) > 1 => {
                pending.keep_uploads(uploads);
                let dynamic = list
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take_dynamic();
                pending.keep_dynamic(dynamic);
                match crate::host::host().map(|h| h.acquire_list()) {
                    Some(Ok(new)) => {
                        *host = Some(std::sync::Arc::new(new));
                        (true, false)
                    }
                    _ => {
                        tracing::warn!("no new host list: a list to run is reset");
                        (false, true)
                    }
                }
            }
            _ => (false, false),
        }
    };
    #[cfg(windows)]
    with_state(this, |host, state| {
        if !fresh && !keep_pending {
            host.reset();
        }
        state.reset(pipeline);
    });
    #[cfg(windows)]
    list.recorded
        .store(false, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(windows))]
    let _ = (this, pipeline);
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    data: BarrierData,
}

#[repr(C)]
#[derive(Clone, Copy)]
union BarrierData {
    transition: TransitionBarrier,
    aliasing: [This; 2],
    uav: This,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TransitionBarrier {
    resource: This,
    subresource: u32,
    before: u32,
    after: u32,
}

/// Appends the host barriers for this title barrier.
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier, out: &mut Vec<Barrier>) {
    use windows::core::Interface;
    let host = |r: This| {
        if r.is_null() {
            Some(r)
        } else {
            crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast())
        }
    };
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return;
            }
            // SAFETY: kind 0 selects the transition member.
            let transition = unsafe { b.data.transition };
            let Some(copy) = crate::resource::host_copy(transition.resource) else {
                return;
            };
            let wanted = crate::host::texture_state(transition.after);
            list.transition(copy, transition.subresource, wanted, |texture, index, before, after| {
                out.push(Barrier {
                    kind: 0,
                    flags: 0,
                    data: BarrierData {
                        transition: TransitionBarrier {
                            resource: texture.as_raw().cast_const().cast(),
                            subresource: index,
                            before,
                            after,
                        },
                    },
                });
            });
        }
        1 => {
            // SAFETY: kind 1 selects the aliasing member.
            let [before, after] = unsafe { b.data.aliasing };
            let (Some(before), Some(after)) = (host(before), host(after)) else { return };
            out.push(Barrier { data: BarrierData { aliasing: [before, after] }, ..*b });
        }
        2 => {
            // SAFETY: kind 2 selects the UAV member.
            let Some(uav) = host(unsafe { b.data.uav }) else { return };
            out.push(Barrier { data: BarrierData { uav }, ..*b });
        }
        _ => {}
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    data: BarrierData { uav: std::ptr::null() },
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host = Vec::new();
        for b in barriers {
            host_barrier(&list, b, &mut host);
        }
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        let has_buffer_barriers = barriers
            .iter()
            .filter_map(|b| {
                // SAFETY: only read the tagged union member.
                unsafe {
                    match b.kind {
                        0 => Some(b.data.transition.resource),
                        2 => Some(b.data.uav),
                        _ => None,
                    }
                }
            })
            .any(|resource| !resource.is_null() && crate::resource::buffer_bytes(resource).is_some());
        if has_buffer_barriers {
            host.push(UAV_BARRIER);
        }
        #[cfg(feature = "diagnostics")]
        if crate::capture::listed(crate::queue::presented() + 1) {
            use windows::core::Interface;
            tracing::debug!(
                target: "victory_gpu::capture::passes",
                list = ?list.list.as_raw(), count, emitted = host.len(), has_buffer_barriers,
                "captured title barriers"
            );
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ensure_subresource_state(list, resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state);
}

/// Moves a single host subresource, or the whole texture, to `state`.
#[cfg(windows)]
pub(crate) fn ensure_subresource_state(
    list: &crate::host::HostList,
    resource: This,
    subresource: u32,
    state: u32,
) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES;
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    list.transition(copy, subresource, state, |texture, index, before, after| {
        let barrier = crate::host::transition_subresource(
            texture,
            index,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(after as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    });
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(from) {
                crate::texture::upload(&list, src, from);
            }
            ensure_state(&list, dst, COPY_DEST);
            ensure_state(&list, src, COPY_SOURCE);
            crate::texture::host_written(dst);
            #[cfg(feature = "diagnostics")]
            {
                crate::capture::trace_transfer("copy resource", src, dst);
                list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            }
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(to, from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    data: CopyLocationData,
}

#[repr(C)]
#[derive(Clone, Copy)]
union CopyLocationData {
    footprint: crate::resource::PlacedFootprint,
    subresource: u32,
}

/// Extent of the selected mip/array/plane subresource on the host.
#[cfg(windows)]
fn readback_extent(
    texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    subresource: u32,
) -> Option<[u32; 3]> {
    let host = crate::host::host()?;
    use windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT;
    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: the texture is live and both outputs hold one subresource's
    // result.
    unsafe {
        host.device.GetCopyableFootprints(
            &texture.GetDesc(),
            subresource,
            1,
            0,
            Some(&mut layout),
            None,
            None,
            Some(&mut total),
        );
    }
    let fp = layout.Footprint;
    (total != u64::MAX).then_some([fp.Width, fp.Height, fp.Depth])
}

/// Texture to texture on the GPU, buffer to texture through an upload
/// buffer filled at submission, and texture to buffer through a readback
/// buffer copied into title memory once the submission completes.
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const SUBRESOURCE: u32 = 0;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        if dst.kind == FOOTPRINT {
            // Texture to buffer: the GPU copies the source subresource into
            // a mapped readback buffer, and the submission writes the rows
            // on into title memory once it completes.
            if src.kind != SUBRESOURCE {
                return;
            }
            let (Some(host), Some(from_tex)) = (
                crate::host::host(),
                crate::resource::host_copy(src.resource),
            ) else {
                return;
            };
            // SAFETY: only the subresource tag can reach the source.
            let src_subresource = unsafe { src.data.subresource };
            if src_subresource >= from_tex.subresource_count() {
                return;
            }
            // SAFETY: the footprint tag selects this member (dst.kind == FOOTPRINT).
            let footprint = unsafe { dst.data.footprint };
            let Some((lines, row_bytes, total)) = footprint.layout() else {
                return;
            };
            if total == 0 || lines == 0 || row_bytes == 0 {
                return;
            }
            let host_format = crate::resource::pc_format(footprint.format);
            if crate::resource::element(host_format) != crate::resource::element(footprint.format) {
                // Console formats such as 118..=120 (3-byte D16S8) map to
                // 4-byte PC substitutes (45..=47): raw readback would
                // mis-size the staging buffer and write the wrong texel
                // layout.
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        format = footprint.format,
                        host_format,
                        "texture-to-buffer readback skipped: host format has a different texel size",
                    );
                });
                return;
            }
            let Some((element, block)) = crate::resource::element(footprint.format) else {
                return;
            };
            // SAFETY: the source texture is live.
            let source_desc = unsafe { from_tex.texture.GetDesc() };
            if source_desc.SampleDesc.Count != 1
                || crate::resource::family(source_desc.Format.0 as u32)
                    != crate::resource::family(host_format)
                || !u64::from(x).is_multiple_of(block)
                || !u64::from(y).is_multiple_of(block)
            {
                return;
            }
            let Some([src_width, src_height, src_depth]) =
                readback_extent(&from_tex.texture, src_subresource)
            else {
                return;
            };
            // SAFETY: the optional source box is read during this call.
            let [width, height, depth] = match unsafe { source_box.as_ref() } {
                Some(&[left, top, front, right, bottom, back]) => {
                    if left >= right || top >= bottom || front >= back
                        || right > src_width || bottom > src_height || back > src_depth
                        || !u64::from(left).is_multiple_of(block)
                        || !u64::from(top).is_multiple_of(block)
                    {
                        return;
                    }
                    [right - left, bottom - top, back - front]
                }
                None => [src_width, src_height, src_depth],
            };
            // A BC mip's reported extent can include its final partial
            // block. Measure the footprint and the copied area in the same
            // storage units.
            let rows_per_slice = u64::from(footprint.height).div_ceil(block);
            let row_in_slice = u64::from(y) / block;
            let first_row = u64::from(z) * rows_per_slice + row_in_slice;
            let column_bytes = (u64::from(x) / block) * element;
            let copy_rows = u64::from(height).div_ceil(block);
            let copy_bytes = u64::from(width).div_ceil(block) * element;
            if copy_bytes == 0
                || copy_rows == 0
                || depth == 0
                || column_bytes + copy_bytes > row_bytes
                || row_in_slice + copy_rows > rows_per_slice
                || u64::from(z) + u64::from(depth) > u64::from(footprint.depth)
            {
                return;
            }
            let dst_pitch = u64::from(footprint.row_pitch);
            if lines > 1 && dst_pitch < row_bytes {
                return;
            }
            let Some(target) = buffer_range(dst.resource, footprint.offset, total) else {
                return;
            };
            // D3D12 requires placed footprints' row pitch to be 256-byte
            // aligned.
            let Ok(row_bytes_u32) = u32::try_from(row_bytes) else {
                return;
            };
            let Some(host_pitch) = footprint
                .row_pitch
                .max(row_bytes_u32)
                .checked_next_multiple_of(256)
            else {
                return;
            };
            let Some(staging_bytes) = u64::from(host_pitch)
                .checked_mul(lines - 1)
                .and_then(|n| n.checked_add(row_bytes))
            else {
                return;
            };
            let Some((buffer, staging)) = list.readback(host, staging_bytes) else {
                return;
            };
            // A texture the host GPU does not write takes its contents
            // from title memory; one the host wrote (a cleared render
            // target) keeps them.
            if crate::texture::title_written(&from_tex.texture) {
                crate::texture::upload(&list, src.resource, &from_tex.texture);
            }
            ensure_subresource_state(&list, src.resource, src_subresource, COPY_SOURCE);
            let mut to_location = dst;
            to_location.resource = buffer.as_raw().cast_const().cast();
            // Writing a union member is safe; only reads are not.
            to_location.data.footprint.offset = 0;
            to_location.data.footprint.format = host_format;
            to_location.data.footprint.row_pitch = host_pitch;
            let mut from_location = src;
            from_location.resource = from_tex.texture.as_raw().cast_const().cast();
            let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (from_tex.texture.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "readback to {:?} footprint {:?} at {x},{y},{z} from {:?} {}x{} f{} sub {} box {b:?}",
                    dst.resource,
                    footprint,
                    src.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    src_subresource,
                )
            });
            // SAFETY: an open list; both locations in the host's layout,
            // naming live resources; the box is the title's (same layout).
            unsafe {
                list.list.CopyTextureRegion(
                    (&raw const to_location).cast(),
                    x,
                    y,
                    z,
                    (&raw const from_location).cast(),
                    host_box,
                )
            };
            // The GPU only populated the copied rectangle in staging.
            // Preserve the rest of each destination row and the gaps
            // between slices.
            for slice in 0..u64::from(depth) {
                let row = first_row + slice * rows_per_slice;
                let dst_offset = (row * dst_pitch + column_bytes) as usize;
                let src_offset = (row * u64::from(host_pitch) + column_bytes) as usize;
                list.defer_readback(
                    target.wrapping_add(dst_offset),
                    staging.wrapping_add(src_offset),
                    dst_pitch as usize,
                    host_pitch as usize,
                    copy_rows as usize,
                    copy_bytes as usize,
                );
            }
            return;
        }
        let Some(to) = crate::resource::host_copy(dst.resource) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("texture to buffer copies not implemented"));
            return;
        };
        if dst.kind != SUBRESOURCE || !matches!(src.kind, SUBRESOURCE | FOOTPRINT) {
            return;
        }
        // SAFETY: the location tag selects the initialized member.
        let dst_subresource = unsafe { dst.data.subresource };
        if dst_subresource >= to.subresource_count() {
            return;
        }
        let src_footprint = if src.kind == FOOTPRINT {
            // SAFETY: the footprint tag selects this member.
            Some(unsafe { src.data.footprint })
        } else {
            None
        };
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if let Some(mut footprint) = src_footprint {
            let (Some(host), Some(bytes)) = (crate::host::host(), footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, footprint.offset, bytes) else {
                return;
            };
            let Some((buffer, staging)) = list.upload(host, bytes) else {
                return;
            };
            // The list owns this mapped D3D12 upload buffer. Wine can report
            // its address as MEM_FREE, so use the host-upload path.
            list.defer_fill(staging, data, bytes as usize);
            from.resource = buffer.as_raw().cast_const().cast();
            footprint.offset = 0;
            from.data = CopyLocationData { footprint };
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() }, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_copy(src.resource) else {
                return;
            };
            // SAFETY: only the subresource tag can reach this branch.
            let src_subresource = unsafe { src.data.subresource };
            if src_subresource >= texture.subresource_count()
                || (src_subresource == dst_subresource && std::sync::Arc::ptr_eq(&texture, &to))
            {
                return;
            }
            if crate::texture::title_written(&texture.texture) {
                crate::texture::upload(&list, src.resource, &texture.texture);
            }
            ensure_subresource_state(&list, src.resource, src_subresource, COPY_SOURCE);
            from.resource = texture.texture.as_raw().cast_const().cast();
        }
        ensure_subresource_state(&list, dst.resource, dst_subresource, COPY_DEST);
        crate::texture::host_written(dst.resource);
        let mut to_location = dst;
        to_location.resource = to.texture.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            crate::capture::trace_transfer("copy region", src.resource, dst.resource);
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.texture.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} subresource {dst_subresource} at {x},{y},{z} from {:?} kind {} footprint {src_footprint:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    src.resource,
                    src.kind,
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        const _: () = assert!(
            std::mem::offset_of!(CopyLocation, data)
                == std::mem::offset_of!(D3D12_TEXTURE_COPY_LOCATION, Anonymous)
        );
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_copy(dst),
            crate::resource::host_copy(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        if dst_sub >= to.subresource_count()
            || src_sub >= from.subresource_count()
            || (dst_sub == src_sub && std::sync::Arc::ptr_eq(&to, &from))
        {
            return;
        }
        ensure_subresource_state(&list, dst, dst_sub, RESOLVE_DEST);
        ensure_subresource_state(&list, src, src_sub, RESOLVE_SOURCE);
        crate::texture::host_written(dst);
        #[cfg(feature = "diagnostics")]
        {
            crate::capture::trace_transfer("resolve", src, dst);
            list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        }
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (to, from) = (&to.texture, &from.texture);
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        ensure_state(&list, record.resource, RENDER_TARGET);
        let rtv = host.rtv(texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(&list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        list.defer(dst as usize, len, move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones (time spans come out as the submission's length).
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    if offset + u64::from(count) * bytes > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do.
extern "system" fn sync(_this: This, _flags: u32, _control: u32, _address: u64, _size: u64) {
    #[cfg(all(windows, feature = "diagnostics"))]
    if crate::capture::listed(crate::queue::presented() + 1)
        && let Some(list) = host_list(_this)
    {
        use windows::core::Interface;
        tracing::debug!(
            target: "victory_gpu::capture::passes",
            list = ?list.list.as_raw(), flags = _flags, control = _control,
            address = format_args!("{_address:#x}"), size = _size, "captured title pipeline sync"
        );
    }
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. Grand Theft Auto VI's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let mut block = vec![0u8; len];
    let start = block.as_mut_ptr();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    // Moving the vector keeps its heap block where it is.
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        with_state(this, |list, state| {
            state.set_viewports(list, items(viewports, count))
        });
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, state| {
            state.set_scissors(list, items(rects, count))
        });
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        with_state(this, |list, state| {
            // SAFETY: the title passes four floats or null.
            state.set_blend(list, unsafe { factor.as_ref() })
        });
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, state| state.set_stencil(list, reference));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let buffer = crate::resource::buffer_bytes(arguments);
    let address = buffer.and_then(|(p, _)| (p as u64).checked_add(offset));
    let counter = crate::resource::buffer_bytes(count).and_then(|(p, width)| {
        if width.checked_sub(count_offset)? < 4 {
            return None;
        }
        (p as u64).checked_add(count_offset)
    });
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        let available = buffer.and_then(|(_, width)| width.checked_sub(offset));
        if let (Some(available), Some(required)) =
            (available, crate::indirect::required_bytes(kind, stride, max))
            && required <= available
        {
            with_state(this, |list, s| {
                ran = s.execute_indirect(list, (kind, stride, max), address, counter);
            });
        } else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why(format!(
                "indirect kind {kind}: unsupported layout or records outside the argument buffer"
            ));
        }
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

#[cfg(windows)]
pub(crate) fn prefixes(this: This) -> Vec<Prefix> {
    // SAFETY: submitted guest list, live for the duration of the call.
    let list = unsafe { &*this.cast::<List>() };
    list.prefixes
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Wait32BitValueX masks the loaded dword, leaving the reference unchanged.
/// See AlanQiang0418's independent implementation in VIctory issue #700.
extern "system" fn wait32(
    this: This,
    address: u64,
    comparison: u32,
    reference: u32,
    mask: u32,
    flags: u32,
) {
    record_wait(
        this,
        address,
        comparison,
        u64::from(reference),
        crate::fence::Operand::Dword(mask),
        flags,
    );
}

extern "system" fn wait64(this: This, address: u64, comparison: u32, reference: u64, flags: u32) {
    record_wait(
        this,
        address,
        comparison,
        reference,
        crate::fence::Operand::Qword,
        flags,
    );
}

fn record_wait(
    this: This,
    address: u64,
    comparison: u32,
    reference: u64,
    operand: crate::fence::Operand,
    flags: u32,
) {
    #[cfg(windows)]
    {
        use std::sync::atomic::Ordering::Relaxed;
        if flags != 0 {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, Relaxed) {
                tracing::warn!(
                    flags,
                    "GPU memory wait flags unimplemented; applying comparison"
                );
            }
        }
        if !(1..=8).contains(&comparison) {
            tracing::warn!(comparison, "unsupported GPU memory comparison");
            return;
        }
        if comparison == 8 {
            return; // ALWAYS is independent of memory at execution time.
        }
        if address == 0 || !address.is_multiple_of(operand.bytes() as u64) {
            tracing::warn!(address, "invalid GPU wait address");
            return;
        }
        // SAFETY: the guest records a list through its own method table.
        let list = unsafe { &*this.cast::<List>() };
        let prefix = if list.recorded.swap(false, Relaxed) {
            let mut current = list.host.lock().unwrap_or_else(|e| e.into_inner());
            let Some(previous) = current.as_ref() else {
                return;
            };
            let host = crate::host::host().expect("a native list has a host");
            let next = host
                .acquire_list()
                .unwrap_or_else(|e| crate::host::fail(format_args!("wait segment: {e}")));
            let mut state = list.state.lock().unwrap_or_else(|e| e.into_inner());
            previous.close();
            previous.keep_dynamic(state.take_dynamic());
            state.resume_on(&next);
            current.replace(std::sync::Arc::new(next))
        } else {
            // Leading and adjacent predicates have no intervening native work.
            // Still enqueue each predicate: values may change before execution.
            None
        };
        list.prefixes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Prefix {
                list: prefix,
                predicate: crate::fence::Predicate {
                    address: address as usize,
                    operand,
                    comparison,
                    reference,
                },
            });
    }
    #[cfg(not(windows))]
    let _ = (this, address, comparison, reference, operand, flags);
}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        #[cfg(feature = "diagnostics")]
        if crate::capture::listed(crate::queue::presented() + 1) {
            use windows::core::Interface;
            tracing::debug!(
                target: "victory_gpu::capture::passes",
                list = ?list.list.as_raw(), count, "captured title enhanced barriers"
            );
        }
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now (no lock held).
    what.write(address as *mut u8, bytes).run();
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, ignored_state as *const ()),             // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait32 as *const ()),                    // 0x238, 32 bits
    (72, wait64 as *const ()),                    // 0x240, 64 bits
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 9] = [
    (28, query_mark as *const ()),             // 0xe0 BeginQuery
    (29, query_mark as *const ()),             // 0xe8 EndQuery
    (30, resolve_queries as *const ()),        // 0xf0
    (78, dispatch_x as *const ()),             // 0x270
    (98, state::dispatch_mesh_x as *const ()), // 0x310
    (99, driver_hint as *const ()),            // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    #[ignore = "requires D3D12"]
    fn adjacent_waits_sample_at_execution_and_hold_after_queue_release() {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
        let _host = crate::host::host().expect("D3D12 host");
        let null = std::ptr::null();
        let (queue, _) = crate::queue::test_queue();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        // Read the high dword of a pair, aligned to four bytes.
        #[repr(align(8))]
        struct Pair([AtomicU32; 2]);
        let words = Pair([AtomicU32::new(u32::MAX), AtomicU32::new(0xab10)]);
        let at = words.0[1].as_ptr() as u64;
        let second = AtomicU64::new(1 << 63);
        let done = AtomicU64::new(0);
        type Wait32 = extern "system" fn(This, u64, u32, u32, u32, u32);
        // SAFETY: this list was created above and starts with its method table.
        let table = unsafe { *list.cast::<*const *const ()>() };
        let index = if table == NEWER_LIST_FUNCTIONS.0.as_ptr() {
            newer_slot(71)
        } else {
            71
        };
        slot::<Wait32>(list, index * 8)(list, at, 3, 0x10, 0xff, 0);
        wait64(list, second.as_ptr() as u64, 5, (1 << 63) - 1, 0);
        let waits = prefixes(list);
        assert_eq!(waits.len(), 2);
        assert!(waits.iter().all(|prefix| prefix.list.is_none()));
        write64(list, done.as_ptr() as u64, 1, 0, 0);
        assert_eq!(close(list), S_OK);
        words.0[1].store(0xab00, Ordering::Release);
        second.store(0, Ordering::Release);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(done.load(Ordering::Acquire), 0);
        for object in [list, allocator, queue] {
            slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
        }
        Written::Value(0xcd10).write(at as *mut u8, 4).run();
        assert_eq!(done.load(Ordering::Acquire), 0);
        Written::Value(1 << 63)
            .write(second.as_ptr().cast(), 8)
            .run();
        assert_eq!(done.load(Ordering::Acquire), 1);
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    #[ignore = "requires D3D12; set VICTORY_D3D12_WARP=1 for the software adapter"]
    fn wait64_orders_two_guest_queues() {
        for producer_first in [true, false] {
            wait_case(7, 0, 1, 2, producer_first, false, false);
        }
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    #[ignore = "requires D3D12; set VICTORY_D3D12_WARP=1 for the software adapter"]
    fn wait64_in_the_middle_preserves_prefix_and_compares_full_words() {
        let reference = 0x1_0000_0001;
        for producer_first in [true, false] {
            for separate_submit in [true, false] {
                // Equality must reject a larger word with identical low bits.
                wait_case(
                    3,
                    0x2_0000_0001,
                    reference,
                    reference,
                    producer_first,
                    true,
                    separate_submit,
                );
                // A greater low word alone is insufficient. Publishing a value
                // with bit 63 set must satisfy the unsigned comparison.
                wait_case(
                    7,
                    u64::from(u32::MAX),
                    reference,
                    1 << 63,
                    producer_first,
                    true,
                    separate_submit,
                );
            }
        }
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    fn wait_case(
        cmp: u32,
        initial: u64,
        reference: u64,
        publish: u64,
        producer_first: bool,
        mid_list: bool,
        separate_submit: bool,
    ) {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{Duration, Instant};
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIAdapter1, IDXGIFactory4,
        };
        let host = crate::host::host().expect("D3D12 host initialization failed");
        // SAFETY: query the DXGI adapter of the actual device used for submission.
        let desc = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(Default::default()).unwrap();
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByLuid(host.device.GetAdapterLuid())
                .unwrap();
            adapter.GetDesc1().unwrap()
        };
        let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        assert_eq!(software, std::env::var_os("VICTORY_D3D12_WARP").is_some());
        let name_len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        eprintln!(
            "D3D12: {}",
            String::from_utf16_lossy(&desc.Description[..name_len])
        );
        type Write = extern "system" fn(This, u64, u64, u32, u32);
        type Wait = extern "system" fn(This, u64, u32, u64, u32);
        type CopyBytes = extern "system" fn(This, u64, u64, u64, u32);
        type Submit = extern "system" fn(This, u32, *const This);
        type Close = extern "system" fn(This) -> HRESULT;
        type Release = extern "system" fn(This) -> u32;
        let null = std::ptr::null();
        let make_pair = || {
            let (mut queue, mut allocator, mut list) = (null, null, null);
            // QueueDescX has four u32 fields. All zero selects a direct queue.
            let direct = [0u32; 4];
            assert_eq!(
                crate::queue::create_x(null, direct.as_ptr().cast(), &PAGEABLE_IIDS[0], &mut queue),
                S_OK
            );
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
                S_OK
            );
            assert_eq!(
                create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
                S_OK
            );
            assert!(host_list(list).is_some(), "no host list");
            (queue, allocator, list)
        };
        let offset = |list: This, index: usize| {
            // SAFETY: our list starts with its method table.
            let table = unsafe { *list.cast::<*const *const ()>() };
            if table == NEWER_LIST_FUNCTIONS.0.as_ptr() {
                newer_slot(index) * 8
            } else {
                index * 8
            }
        };

        let producer = make_pair();
        let consumer = make_pair();
        let mut follower = null;
        assert_eq!(
            create_list(null, 0, 0, consumer.1, null, &LIST_IIDS[1], &mut follower),
            S_OK
        );
        const SENTINEL: u64 = 0xface_feed;
        let (mut payload, mut counter, mut result) = (0u64, initial, SENTINEL);
        let mut copied_marker = SENTINEL;
        let following = AtomicU64::new(0);
        let (source, source_ptr) = host.upload_buffer(8).unwrap();
        let (native_result, native_ptr) = host.shared_buffer(16).unwrap();
        // SAFETY: both resources are mapped here and outlive the final completion marker.
        unsafe {
            source_ptr.cast::<u64>().write(0x1234);
            native_ptr.write_bytes(0, 16);
        }
        let gpu_copy = |offset| {
            let segment = host_list(consumer.2).unwrap();
            // SAFETY: both buffers are live and contain the full 8-byte range.
            unsafe {
                segment
                    .list
                    .CopyBufferRegion(&native_result, offset, &source, 0, 8)
            };
        };
        let before = AtomicU64::new(0);
        let after = AtomicU64::new(0);
        let write = slot::<Write>(producer.2, offset(producer.2, 70));
        write(producer.2, (&raw mut payload) as u64, 0x5a55a11c, 0, 0);
        write(producer.2, (&raw mut counter) as u64, publish, 0, 0);
        let consumer_write = slot::<Write>(consumer.2, offset(consumer.2, 70));
        if mid_list {
            gpu_copy(0);
            consumer_write(consumer.2, before.as_ptr() as u64, 1, 0, 0);
        }
        let wait = slot::<Wait>(consumer.2, offset(consumer.2, 72));
        wait(consumer.2, (&raw mut counter) as u64, cmp, reference, 0);
        if mid_list {
            gpu_copy(8);
        }
        slot::<CopyBytes>(consumer.2, offset(consumer.2, 60))(
            consumer.2,
            (&raw mut result) as u64,
            (&raw mut payload) as u64,
            8,
            0,
        );
        consumer_write(consumer.2, after.as_ptr() as u64, 1, 0, 0);
        // This copy is performed during host preparation. Its input is written
        // only when the consumer's native suffix has completed.
        slot::<CopyBytes>(follower, offset(follower, 60))(
            follower,
            (&raw mut copied_marker) as u64,
            after.as_ptr() as u64,
            8,
            0,
        );
        slot::<Write>(follower, offset(follower, 70))(follower, following.as_ptr() as u64, 1, 0, 0);
        assert_eq!(slot::<Close>(follower, 0x48)(follower), S_OK);
        assert_eq!(slot::<Close>(producer.2, 0x48)(producer.2), S_OK);
        assert_eq!(slot::<Close>(consumer.2, 0x48)(consumer.2), S_OK);
        let submit = |queue, lists: &[This]| {
            slot::<Submit>(queue, 0x50)(queue, lists.len() as u32, lists.as_ptr());
        };
        let submit_consumer = || {
            if separate_submit {
                submit(consumer.0, &[consumer.2]);
                submit(consumer.0, &[follower]);
            } else {
                submit(consumer.0, &[consumer.2, follower]);
            }
        };
        let mut while_waiting = None;
        let mut native_before_signal = None;
        if producer_first {
            submit(producer.0, &[producer.2]);
            submit_consumer();
        } else {
            submit_consumer();
            // Both guest lists and all buffers remain live past completion.
            while_waiting = Some((
                before.load(Ordering::Acquire),
                result,
                after.load(Ordering::Acquire),
                following.load(Ordering::Acquire),
                copied_marker,
            ));
            if mid_list {
                // SAFETY: the producer has not been submitted; the prefix has finished.
                native_before_signal = Some(unsafe { native_ptr.cast::<[u64; 2]>().read() });
                // The original suffix must survive an immediate guest reset.
                assert_eq!(reset(consumer.2, consumer.1, null), S_OK);
            }
            submit(producer.0, &[producer.2]);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while following.load(Ordering::Acquire) != 1 {
            assert!(Instant::now() < deadline, "consumer did not finish");
            std::thread::yield_now();
        }
        slot::<Release>(follower, 0x10)(follower);
        for (queue, allocator, list) in [consumer, producer] {
            for object in [list, allocator, queue] {
                slot::<Release>(object, 0x10)(object);
            }
        }
        if mid_list {
            // SAFETY: the end-of-pipe marker above follows the native copy.
            assert_eq!(unsafe { native_ptr.cast::<[u64; 2]>().read() }, [0x1234; 2]);
        }
        if let Some(prefix) = native_before_signal {
            assert_eq!(prefix, [0x1234, 0]);
        }
        if let Some(snapshot) = while_waiting {
            assert_eq!(snapshot, (u64::from(mid_list), SENTINEL, 0, 0, SENTINEL));
        }
        assert_eq!(after.load(Ordering::Acquire), 1);
        assert_eq!(
            copied_marker, 1,
            "following list prepared before suffix completed"
        );
        assert_eq!((counter, payload), (publish, 0x5a55a11c));
        assert_eq!(result, payload, "producer_first={producer_first}");
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn deferred_copy_latches_a_wait_before_a_later_overwrite() {
        use crate::fence::{MemoryWait, Operand, Predicate};
        use crate::timeline::{Op, Timeline};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};
        crate::host::host().expect("D3D12");
        for memory_to_memory in [false, true] {
            let (queue, _) = crate::queue::test_queue();
            let null = std::ptr::null();
            let (mut allocator, mut list) = (null, null);
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
                S_OK
            );
            assert_eq!(
                create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
                S_OK
            );
            let word = AtomicU64::new(0);
            let first = AtomicU64::new(1);
            let second = AtomicU64::new(2);
            let consumer = Arc::new(Timeline::default());
            let (tx, rx) = std::sync::mpsc::channel();
            consumer.submit_batch([
                Op::Memory(MemoryWait::new(Predicate {
                    address: word.as_ptr() as usize,
                    operand: Operand::Qword,
                    comparison: 3,
                    reference: 1,
                })),
                Op::Run(Box::new(move || tx.send(()).unwrap())),
            ]);
            for src in [&first, &second] {
                if memory_to_memory {
                    copy_memory(list, word.as_ptr() as u64, src.as_ptr() as u64, 8, 0);
                } else {
                    host_list(list).unwrap().defer_copy(
                        word.as_ptr().cast(),
                        src.as_ptr().cast(),
                        8,
                    );
                }
            }
            assert_eq!(close(list), S_OK);
            slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
            assert_eq!(word.load(Ordering::Acquire), 2);
            rx.recv_timeout(std::time::Duration::from_secs(2))
                .expect("copy lost EQ(1)");
            for o in [list, allocator, queue] {
                slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
            }
        }
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// Resetting or dropping a list whose submission is held behind a fence
    /// wait keeps its upload blocks alive until the submission runs, and
    /// recycles the detached host list back into the host's pool.
    #[cfg(windows)]
    #[test]
    fn pending_list_reset_and_drop_keep_uploads_and_recycle_host_list() {
        use windows::core::Interface;

        let Some(host) = crate::host::host() else {
            return;
        };
        type Sync = extern "system" fn(This, This, u64) -> HRESULT;
        type Execute = extern "system" fn(This, u32, *const This);
        let null = std::ptr::null();
        let (waits, sig_slot) = crate::queue::test_queue();
        let wait = slot::<Sync>(waits, sig_slot + 8);
        let signal = slot::<Sync>(waits, sig_slot);
        let execute = slot::<Execute>(waits, 0x50);
        let completed = |f: This| slot::<extern "system" fn(This) -> u64>(f, 0x40)(f);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);

        let fence_iid = &crate::fence::FENCE_IIDS[0];
        let (mut ready, mut done) = (null, null);
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut ready),
            S_OK
        );
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut done), S_OK);

        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );

        // Hold the queue, record a copy from list upload memory, submit, then
        // reset the list while its submission is still waiting.
        assert_eq!(wait(waits, ready, 1), S_OK);
        let src1 = allocate_upload(list, 64, 16);
        assert!(!src1.is_null());
        // SAFETY: 64 bytes handed out above.
        unsafe { src1.write_bytes(0x3c, 64) };
        let mut dst1 = [0u8; 64];
        copy_memory(list, dst1.as_mut_ptr() as u64, src1 as u64, 64, 0);
        assert_eq!(close(list), S_OK);
        let raw1 = host_list(list).unwrap().list.as_raw() as usize;
        execute(waits, 1, &list);
        assert_eq!(signal(waits, done, 1), S_OK);
        assert_eq!(reset(list, allocator, null), S_OK);
        let raw2 = host_list(list).unwrap().list.as_raw() as usize;
        assert_ne!(raw2, raw1, "reset must detach from the pending host list");
        assert_eq!(dst1, [0u8; 64]);

        let reused_before = host.reused_lists();
        assert_eq!(crate::fence::signal(ready, 1), S_OK);
        assert_eq!(completed(done), 1);
        assert_eq!(dst1, [0x3c; 64], "kept upload memory reached the copy");

        // Now hold behind ready == 2: reset `list` again (reusing a pooled
        // host list) and also drop a second list while its submission waits.
        assert_eq!(wait(waits, ready, 2), S_OK);
        assert_eq!(close(list), S_OK);
        execute(waits, 1, &list);
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(
            host.reused_lists() > reused_before,
            "detached host list should be recycled from the pool"
        );

        let mut list2 = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list2),
            S_OK
        );
        let src2 = allocate_upload(list2, 64, 16);
        assert!(!src2.is_null());
        // SAFETY: 64 bytes handed out above.
        unsafe { src2.write_bytes(0x5a, 64) };
        let mut dst2 = [0u8; 64];
        copy_memory(list2, dst2.as_mut_ptr() as u64, src2 as u64, 64, 0);
        assert_eq!(close(list2), S_OK);
        execute(waits, 1, &list2);
        assert_eq!(signal(waits, done, 2), S_OK);
        // Drop `list2` while its submission is still held back.
        assert_eq!(release(list2), 0);
        assert_eq!(dst2, [0u8; 64]);

        assert_eq!(crate::fence::signal(ready, 2), S_OK);
        assert_eq!(completed(done), 2);
        assert_eq!(
            dst2, [0x5a; 64],
            "dropping a pending list must keep its upload memory until run"
        );

        for object in [list, allocator, waits, ready, done] {
            assert_eq!(release(object), 0);
        }
    }
}
