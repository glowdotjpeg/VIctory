//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. A compute shader copies and clamps the
//! active records into list-owned scratch just before execution. It also
//! snapshots the optional count, leaving both title buffers untouched.
//! Non-indexed rectangle-list draws follow the clamp with GPU-side
//! rectangle expansion in [`crate::rectangles`].
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{BUFFER_READ, BUFFER_WRITE, HostList, host};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint payload;
    uint args_offset;
    uint has_count;
    uint src_bias;
    uint count_bias;
};
ByteAddressBuffer args : register(t0);
ByteAddressBuffer counter : register(t1);
RWByteAddressBuffer scratch : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint active = records;
    if (has_count != 0) {
        active = min(active, counter.Load(count_bias));
        if (id.x == 0) scratch.Store(0, active);
    }
    if (id.x >= active) return;
    uint src = src_bias + id.x * stride;
    uint dst = args_offset + id.x * payload;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(src), 1u << 24);
        uint instances = min(args.Load(src + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        scratch.Store2(dst, uint2(count, instances));
        if (kind == 0) {
            scratch.Store2(dst + 8, args.Load2(src + 8));
        } else {
            scratch.Store3(dst + 8, args.Load3(src + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(src), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        scratch.Store3(dst, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(src);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        scratch.Store3(dst, g);
    }
    uint used;
    budget.InterlockedCompareExchange(0, 0, 0, used);
    bool allowed = cost == 0;
    [allow_uav_condition]
    while (!allowed && used < BUDGET && cost <= BUDGET - used) {
        uint observed;
        budget.InterlockedCompareExchange(0, used, used + cost, observed);
        allowed = observed == used;
        used = observed;
    }
    if (!allowed) {
        // Draws keep their layout with no vertices; groups become zero.
        scratch.Store(dst, 0);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_compute(device, SOURCE, s!("clamp"))
}

fn make_compute(
    device: &ID3D12Device,
    source: &str,
    name: windows::core::PCSTR,
) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Packed record size of a supported one-argument PC signature.
pub(crate) fn payload_size(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Source span of a supported layout. Only a record's payload must exist
/// at the end of the buffer. The clamp uses a one-dimensional dispatch
/// and 32-bit byte offsets relative to the source GPU address.
pub(crate) fn required_bytes(kind: u32, stride: u32, records: u32) -> Option<u64> {
    let payload = payload_size(kind)?;
    if stride < payload || !stride.is_multiple_of(4) || records.div_ceil(64) > 65535 {
        return None;
    }
    if records == 0 {
        return Some(0);
    }
    let bytes = u64::from(records - 1)
        .checked_mul(u64::from(stride))?
        .checked_add(u64::from(payload))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

const ARGS_OFFSET: u32 = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT;

/// Copies active records to packed scratch in unordered access, and orders
/// previous GPU writes to the source buffers and budget. Callers rebind
/// their compute root signature and pipeline before doing further work.
pub(crate) fn bound(
    list: &HostList,
    (src, offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<Bounded> {
    let bytes = required_bytes(kind, stride, records)?;
    if bytes == 0 || !offset.is_multiple_of(4) || count.is_some_and(|(_, at)| !at.is_multiple_of(4))
    {
        return None;
    }
    // The guest offsets are dword-aligned. Keep the root SRVs aligned to
    // raw view alignment and pass any remaining bytes to the shader.
    let alignment = u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT);
    let src_bias = offset % alignment;
    let count_bias = count.map_or(0, |(_, at)| at % alignment);
    if bytes + src_bias > u64::from(u32::MAX) {
        return None;
    }
    let payload = payload_size(kind)?;
    let (c, budget) = (clamp()?, list.budget()?);
    let scratch =
        list.indirect_scratch(u64::from(ARGS_OFFSET) + u64::from(payload) * u64::from(records))?;
    // SAFETY: the caller verified that the aligned source ranges fit the live buffers.
    let (src_gpu, count_gpu, dst_gpu) = unsafe {
        let src_gpu = src.GetGPUVirtualAddress() + (offset - src_bias);
        (
            src_gpu,
            count.map_or(src_gpu, |(buffer, at)| {
                buffer.GetGPUVirtualAddress() + (at - count_bias)
            }),
            scratch.GetGPUVirtualAddress(),
        )
    };
    list.buffer_state(src, BUFFER_READ);
    if let Some((buffer, _)) = count {
        list.buffer_state(buffer, BUFFER_READ);
    }
    list.buffer_state(&scratch, BUFFER_WRITE);
    let l = &list.list;
    let counted = count.is_some();
    let constants = [
        kind,
        stride,
        records,
        payload,
        ARGS_OFFSET,
        u32::from(counted),
        src_bias as u32,
        count_bias as u32,
    ];
    // SAFETY: an open list with live source, scratch and budget buffers.
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    Some(Bounded {
        scratch,
        records,
        counted,
    })
}

/// Scratch for one indirect operation. Consume it before the next clamp
/// on the same list, which may reuse the buffer.
pub(crate) struct Bounded {
    scratch: ID3D12Resource,
    records: u32,
    counted: bool,
}

impl Bounded {
    /// Packed argument address and optional count address for compute passes.
    pub(crate) fn gpu(&self) -> (u64, Option<u64>) {
        // SAFETY: the scratch buffer is live.
        let base = unsafe { self.scratch.GetGPUVirtualAddress() };
        (base + u64::from(ARGS_OFFSET), self.counted.then_some(base))
    }

    /// Executes the packed records and optional count from the same scratch buffer.
    pub(crate) fn execute(self, list: &HostList, signature: &ID3D12CommandSignature) {
        list.buffer_state(&self.scratch, BUFFER_READ);
        // SAFETY: an open list with the pipeline bound, list-owned scratch,
        // and a cached signature whose stride is the packed record size.
        unsafe {
            list.list.ExecuteIndirect(
                signature,
                self.records,
                &self.scratch,
                u64::from(ARGS_OFFSET),
                self.counted.then_some(&self.scratch),
                0,
            )
        };
    }
}

pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_indirect_source_span() {
        assert_eq!(required_bytes(2, 12, 2), Some(24));
        assert_eq!(required_bytes(0, 32, 2), Some(48));
        assert_eq!(required_bytes(1, 20, 1), Some(20));
        assert_eq!(required_bytes(10, 12, 0), Some(0));
        assert_eq!(required_bytes(8, 32, 1), None);
        assert_eq!(required_bytes(1, 16, 1), None);
        assert_eq!(required_bytes(2, 13, 1), None);
        assert_eq!(required_bytes(2, 12, 65535 * 64 + 1), None);
        assert_eq!(required_bytes(2, u32::MAX - 3, 2), None);
    }

    #[test]
    fn rejected_records_do_not_wrap_the_indirect_work_budget() {
        let Some(host) = host() else { return };
        const LARGE_RECORDS: u32 = 64;
        const STRIDE: u32 = 16;
        let (buffer, data) = host
            .shared_buffer(u64::from((LARGE_RECORDS + 1) * STRIDE))
            .unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: the buffer's CPU mapping contains every record. The GPU
        // reads it after submission and the CPU checks it after the fence.
        unsafe {
            for i in 0..=LARGE_RECORDS {
                let count = if i < LARGE_RECORDS { 1 << 24 } else { 3 };
                data.add((i * STRIDE) as usize)
                    .cast::<[u32; 4]>()
                    .write([count, 4, 0, 0]);
            }
        }
        let bytes = u64::from((LARGE_RECORDS + 1) * STRIDE);
        let (back, mapped) = list.readback(host, bytes).unwrap();
        let first = bound(&list, (&buffer, 0), None, (0, STRIDE, LARGE_RECORDS)).unwrap();
        let first_bytes = u64::from(LARGE_RECORDS * STRIDE);
        list.buffer_state(&first.scratch, BUFFER_READ);
        // SAFETY: an open list and live buffers of sufficient size.
        unsafe {
            list.list.CopyBufferRegion(
                &back,
                0,
                &first.scratch,
                u64::from(ARGS_OFFSET),
                first_bytes,
            );
        }
        let last = bound(&list, (&buffer, first_bytes), None, (0, STRIDE, 1)).unwrap();
        list.buffer_state(&last.scratch, BUFFER_READ);
        // SAFETY: as above. The second clamp may reuse the scratch of the first.
        unsafe {
            list.list.CopyBufferRegion(
                &back,
                first_bytes,
                &last.scratch,
                u64::from(ARGS_OFFSET),
                u64::from(STRIDE),
            );
        }
        list.close();
        host.execute(&[&list], "test indirect argument budget");
        // SAFETY: the copies are complete and their readback mapping is live.
        let vertex_count = |i: u32| unsafe {
            mapped
                .add((i * STRIDE) as usize)
                .cast::<u32>()
                .read_volatile()
        };
        assert_eq!(
            (0..LARGE_RECORDS).filter(|&i| vertex_count(i) != 0).count(),
            2
        );
        assert_eq!(vertex_count(LARGE_RECORDS), 0);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }
}

#[cfg(all(test, feature = "diagnostics"))]
#[path = "indirect/warp_tests.rs"]
mod warp_tests;
