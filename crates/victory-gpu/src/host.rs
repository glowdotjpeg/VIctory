//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

mod states;
#[cfg(feature = "diagnostics")]
mod removal_trace;
pub(crate) use states::HostTexture;
use states::RecordedTexture;

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;
/// Maximum number of detached host lists kept for reuse after pending resets.
const LIST_POOL_CAP: usize = 64;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// Detached host lists whose submissions finished, ready to reuse.
    free_lists: Mutex<Vec<HostList>>,
    #[cfg(test)]
    reused_lists: std::sync::atomic::AtomicU64,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        #[cfg(feature = "diagnostics")]
        removal_trace::log();
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        #[cfg(feature = "diagnostics")]
        if debug && std::env::var_os("VICTORY_D3D12_REMOVAL_STACK").is_some() {
            removal_trace::register(&device);
        }
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            free_lists: Mutex::default(),
            #[cfg(test)]
            reused_lists: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

// The WARP ordering tests are shared with `list.rs`'s: its helper fns
// are pub(crate) for it.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    fn a_recorded_list_keeps_the_descriptor_heap_owner_until_reset() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let heap = Arc::new(crate::descriptor::host::HostHeap::new(false, 1).unwrap());
        let weak_heap = Arc::downgrade(&heap);
        list.keep_descriptor_heap(&heap);
        drop(heap);
        assert!(weak_heap.upgrade().is_some());

        list.close();
        list.reset();
        assert!(weak_heap.upgrade().is_none());
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn dred_can_match_a_list_by_name_without_its_pointer() {
        let Some(host) = host() else { return };
        let a = host.new_list().unwrap();
        let b = host.new_list().unwrap();
        let name = windows::core::HSTRING::from(format!("{DRED_LIST_PREFIX}{}", b.dred_id));
        let runs = [&a, &b];
        assert_eq!(breadcrumb_list(&runs, None, PCWSTR(name.as_ptr())), Some(1));
        assert_eq!(breadcrumb_list(&runs, Some(a.list.as_raw()), PCWSTR::null()), Some(0));
    }

    #[test]
    fn buffer_states_transition_between_writes_and_reads() {
        let mut s = BufferStates::default();
        // First uses: the prelude's.
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), None);
        assert_eq!(s.update(2, 'b', BUFFER_READ), None);
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), None);
        assert_eq!(s.update(1, 'a', BUFFER_READ), Some(BUFFER_WRITE));
        assert_eq!(s.update(1, 'a', BUFFER_READ), None);
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), Some(BUFFER_READ));
        let sorted = |mut v: Vec<(char, u32)>| {
            v.sort_unstable();
            v
        };
        let firsts = sorted(s.firsts().map(|(r, f)| (*r, f)).collect());
        assert_eq!(firsts, [('a', BUFFER_WRITE), ('b', BUFFER_READ)]);
        let lasts = sorted(s.lasts().map(|(r, l)| (*r, l)).collect());
        assert_eq!(lasts, [('a', BUFFER_WRITE), ('b', BUFFER_READ)]);
        s.clear();
        assert_eq!(s.firsts().count(), 0);
    }

    /// Set in the child process `in_child` starts.
    const CHILD: &str = "VICTORY_TEST_DEBUG_LAYER_CHILD";

    /// Runs the test `name` of this module alone in a child process and
    /// checks that it passed; false in that child. The debug layer is
    /// process-wide and must be on before the host device is made.
    fn in_child(name: &str) -> bool {
        if std::env::var_os(CHILD).is_some() {
            return false;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([&format!("host::tests::{name}"), "--exact", "--nocapture"])
            .env(CHILD, "1")
            .env_remove("VICTORY_D3D12_DEBUG")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{text}");
        assert!(out.status.success(), "the child test failed");
        assert!(text.contains("1 passed"), "the child ran no test");
        true
    }

    pub(crate) fn compile(source: &str, entry: windows::core::PCSTR, target: windows::core::PCSTR) -> Vec<u8> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names live through the call; out pointers
        // ours; the blobs' own bytes are read.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                entry,
                target,
                0,
                0,
                &mut code,
                Some(&mut errors),
            );
            if let Some(e) = &errors {
                let text = std::slice::from_raw_parts(
                    e.GetBufferPointer().cast::<u8>(),
                    e.GetBufferSize(),
                );
                assert!(made.is_ok(), "{}", String::from_utf8_lossy(text));
            }
            made.unwrap();
            let code = code.unwrap();
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
                .to_vec()
        }
    }

    /// A buffer a shader writes and draws read as vertices, as the world's
    /// 174 MB heap is (#199, #264): the argument clamp writes it and
    /// indirect and plain draws read it, in two lists submitted together.
    /// The debug layer checks every barrier's state before against what
    /// the lists did (id 527, which #264 reported); it does not check
    /// vertex buffer states at draws on every system, so this cannot show
    /// #199's vertex buffer in unordered access.
    #[test]
    fn debug_layer_accepts_buffers_written_then_read_as_vertices() {
        use windows::core::s;
        if in_child("debug_layer_accepts_buffers_written_then_read_as_vertices") {
            return;
        }
        let mut layer: Option<ID3D12Debug> = None;
        // SAFETY: plain queries, before any device exists in this process.
        if unsafe { D3D12GetDebugInterface(&mut layer) }.is_err() {
            eprintln!("no D3D12 debug layer; skipped");
            return;
        }
        // SAFETY: as above.
        unsafe { layer.unwrap().EnableDebugLayer() };
        let Some(host) = host() else {
            eprintln!("no host device; skipped");
            return;
        };
        let messages: ID3D12InfoQueue = host.device.cast().unwrap();

        const SOURCE: &str = r#"
            #define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)"
            [RootSignature(RS)] float4 vs(float4 p : POSITION) : SV_Position { return p; }
            [RootSignature(RS)] float4 ps() : SV_Target { return float4(1, 0, 0, 1); }
        "#;
        let (vs, ps) = (
            compile(SOURCE, s!("vs"), s!("vs_5_1")),
            compile(SOURCE, s!("ps"), s!("ps_5_1")),
        );
        let element = D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        // SAFETY: plain object creation from valid descriptions; the
        // shader bytes outlive the calls.
        let (root, pso, target) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &vs).unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    FrontFace: stencil,
                    BackFace: stencil,
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: &element,
                    NumElements: 1,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let pso: ID3D12PipelineState = host.device.CreateGraphicsPipelineState(&desc).unwrap();
            let mut target: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 4,
                        Height: 4,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .unwrap();
            (root, pso, target.unwrap())
        };

        // Three vertices at 0, a draw's arguments (3 vertices, 1 instance)
        // at 256: one buffer, as title memory is.
        let (buffer, data) = host.shared_buffer(1 << 16).unwrap();
        let vertices: [f32; 12] = [
            -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0,
        ];
        // SAFETY: inside the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(vertices.as_ptr().cast::<u8>(), data, 48);
            std::ptr::copy_nonoverlapping([3u32, 1, 0, 0].as_ptr().cast::<u8>(), data.add(256), 16);
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let rtv = host.rtv(&target, None);
        // SAFETY: one draw argument, no root arguments.
        let signature: ID3D12CommandSignature = unsafe {
            let argument = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                ..Default::default()
            };
            let mut signature = None;
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 16,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None::<&ID3D12RootSignature>,
                    &mut signature,
                )
                .unwrap();
            signature.unwrap()
        };
        // The draw state, with the buffer's vertices bound as `draw.rs`
        // binds them.
        let bind = |list: &HostList| {
            list.buffer_state(&buffer, BUFFER_READ);
            // SAFETY: an open list; our objects, alive until the lists ran.
            unsafe {
                let l = &list.list;
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pso);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 4.0,
                    Height: 4.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[RECT {
                    right: 4,
                    bottom: 4,
                    ..Default::default()
                }]);
                l.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                l.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: gpu,
                        SizeInBytes: 48,
                        StrideInBytes: 16,
                    }]),
                );
            }
        };
        let draw = |list: &HostList| {
            bind(list);
            // SAFETY: an open list with the draw state set.
            unsafe { list.list.DrawInstanced(3, 1, 0, 0) };
        };
        // As `draw.rs` executes indirect draws: clamp, bind, execute.
        let indirect = |list: &HostList| {
            let bounded = crate::indirect::bound(list, (&buffer, 256), None, (0, 16, 1)).unwrap();
            bind(list);
            bounded.execute(list, &signature);
        };
        // A list that draws, draws indirectly, draws; then one that starts
        // with an indirect draw, submitted together.
        let (a, b) = (host.new_list().unwrap(), host.new_list().unwrap());
        draw(&a);
        indirect(&a);
        draw(&a);
        a.close();
        indirect(&b);
        draw(&b);
        b.close();
        // SAFETY: plain queries.
        unsafe { messages.ClearStoredMessages() };
        host.execute(&[&a, &b], "synthetic buffer transitions");

        let mut errors = Vec::new();
        // SAFETY: each message is read into a buffer of the size the queue
        // reports.
        unsafe {
            for i in 0..messages.GetNumStoredMessages() {
                let mut size = 0;
                if messages.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut bytes = vec![0u64; size.div_ceil(8)];
                let m = bytes.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if messages.GetMessage(i, Some(m), &mut size).is_ok()
                    && (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0
                {
                    let text =
                        std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength);
                    errors.push(format!(
                        "id {}: {}",
                        (*m).ID.0,
                        String::from_utf8_lossy(text)
                    ));
                }
            }
        }
        assert!(errors.is_empty(), "debug layer errors: {errors:#?}");
    }

    /// The batching rule: a list's CPU copy of title memory splits the
    /// submission only when its source overlaps a range an earlier list's
    /// shaders write; disjoint and merely touching ranges (one ends where
    /// the other begins) stay in one submission.
    #[test]
    fn only_overlapping_copy_sources_split_the_submission() {
        let writes = [(0x1000, 0x1100), (0x2000, 0x3000)];
        assert!(overlaps_write(&writes, 0x1080, 8), "inside a write");
        assert!(overlaps_write(&writes, 0x0ff0, 0x20), "across its start");
        assert!(overlaps_write(&writes, 0x10f0, 0x20), "across its end");
        assert!(overlaps_write(&writes, 0x2080, 0x800), "inside the other");
        assert!(overlaps_write(&writes, 0x0ff0, 0x2100), "across both");
        assert!(!overlaps_write(&writes, 0x1100, 8), "touching after it");
        assert!(!overlaps_write(&writes, 0x0f00, 0x100), "touching before it");
        assert!(!overlaps_write(&writes, 0x3000, 8), "touching after the last");
        assert!(!overlaps_write(&writes, 0x1180, 0x100), "between them");
        assert!(!overlaps_write(&[], 0x1000, 8), "no writes, no split");
        assert!(!overlaps_write(&writes, 0x1080, 0), "nothing read, no split");
    }

    /// The same rule for a copy recorded behind the writes in its own
    /// recording (`copy_intersects_write`): only a shader write, a
    /// readback or a completion write the recording already declared
    /// splits it.
    #[test]
    fn only_a_copy_of_a_written_range_splits_its_recording() {
        if host().is_none() {
            eprintln!("no host device; skipped");
            return;
        }
        let mut title = vec![0u8; 1 << 16];
        let at = title.as_ptr() as u64;
        let shadow = title_buffer(&mut title);
        let written = [Step::GpuWrite(shadow.clone(), at as usize, 256)];
        let none: [Readback; 0] = [];
        let no_words: [(usize, crate::queue::Written, usize); 0] = [];
        assert!(
            copy_intersects_write(&written, &none, &no_words, at as usize + 64, 8),
            "inside a write"
        );
        assert!(
            copy_intersects_write(&written, &none, &no_words, at as usize - 8, 16),
            "across its start"
        );
        assert!(
            copy_intersects_write(&written, &none, &no_words, at as usize + 256 - 8, 16),
            "across its end"
        );
        assert!(
            !copy_intersects_write(&written, &none, &no_words, at as usize + 256, 8),
            "touching after it"
        );
        assert!(
            !copy_intersects_write(&written, &none, &no_words, at as usize + 4096, 256),
            "beside it"
        );
        assert!(
            !copy_intersects_write(&written, &none, &no_words, at as usize, 0),
            "nothing read, no split"
        );
        // A readback's rows fill its destination range once the list has
        // run, so a copy out of them splits too.
        let filled = [Readback {
            dst: at as usize + 8192,
            src: 0,
            dst_pitch: 512,
            src_pitch: 512,
            lines: 2,
            row_bytes: 256,
        }];
        assert!(
            copy_intersects_write(&[], &filled, &no_words, at as usize + 8192 + 512, 8),
            "inside the rows"
        );
        assert!(
            !copy_intersects_write(
                &[],
                &filled,
                &no_words,
                at as usize + 8192 + 512 + 256,
                8
            ),
            "between the rows"
        );
        // A bottom-of-pipe completion write recorded earlier lands in
        // title memory once the list has run; a copy out of it splits
        // too.
        let words = [(at as usize + 16384, crate::queue::Written::Value(7), 8)];
        assert!(
            copy_intersects_write(&written, &none, &words, at as usize + 16384, 8),
            "inside the write"
        );
        assert!(
            copy_intersects_write(&written, &none, &words, at as usize + 16384 - 4, 8),
            "across its start"
        );
        assert!(
            !copy_intersects_write(&written, &none, &words, at as usize + 16392, 8),
            "touching after it"
        );
        crate::shadow::unregister(1, at);
    }

    /// Set in the child process `in_warp_child` starts.
    const WARP_CHILD: &str = "VICTORY_TEST_WARP_CHILD";

    /// Runs the test `test` (its full path in the test binary) alone in a
    /// child process with the host device on the software adapter (the
    /// ordering shows on any adapter; WARP keeps the test off hardware).
    /// One test at a time: the host device is global and made before the
    /// first use.
    pub(crate) fn in_warp_child(test: &str) -> bool {
        if std::env::var_os(WARP_CHILD).is_some() {
            return false;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test, "--exact", "--nocapture"])
            .env(WARP_CHILD, "1")
            .env("VICTORY_D3D12_WARP", "1")
            .env_remove("VICTORY_D3D12_DEBUG")
            .env_remove("VICTORY_D3D12_GBV")
            .env_remove("VICTORY_D3D12_DRED")
            .env_remove("VICTORY_GPU_TIMING")
            .env_remove("VICTORY_COPY_MODE")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{text}");
        assert!(out.status.success(), "the child test failed");
        assert!(text.contains("1 passed"), "the child ran no test");
        true
    }

    /// Whether the host device is on the software adapter: without it the
    /// WARP setting is ignored (diagnostics builds) and the test would run
    /// on hardware.
    pub(crate) fn on_warp(host: &Host) -> bool {
        // SAFETY: plain DXGI and D3D12 queries.
        unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return false;
            };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter1>() else {
                return false;
            };
            let Ok(desc) = warp.GetDesc1() else { return false };
            let device = host.device.GetAdapterLuid();
            (device.LowPart, device.HighPart) == (desc.AdapterLuid.LowPart, desc.AdapterLuid.HighPart)
        }
    }

    /// What the test's dispatch writes, dword `i` of the range.
    pub(crate) const WRITE_PATTERN: u32 = 0xC0DE_0000;

    /// A shader whose dispatch writes 64 dwords (256 bytes) at the start
    /// of the buffer it is given (as a shader's unordered access does).
    const WRITE_SOURCE: &str = r#"
        #define RS "RootFlags(0), UAV(u0)"
        RWByteAddressBuffer out_buffer : register(u0);
        [RootSignature(RS)] [numthreads(64, 1, 1)]
        void cs(uint i : SV_DispatchThreadID) {
            out_buffer.Store(i * 4, 0xC0DE0000u + i);
        }
    "#;

    /// The pipeline of [`WRITE_SOURCE`].
    pub(crate) fn write_pipeline(host: &Host) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::core::s;
        let cs = compile(WRITE_SOURCE, s!("cs"), s!("cs_5_1"));
        // SAFETY: the shader bytes live through the call; our objects.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &cs).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.as_ptr().cast(),
                        BytecodeLength: cs.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// Records into `a`: the refresh that gives the shaders the title's
    /// data, a dispatch whose shaders write the range, and the copy-back
    /// of what they wrote once the list has run.
    pub(crate) fn record_gpu_write(
        a: &HostList,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        at: u64,
        (root, pso): (&ID3D12RootSignature, &ID3D12PipelineState),
    ) {
        shadow.refresh(a, at, 256);
        shadow.gpu_writes(a, at, 256);
        // SAFETY: a live buffer.
        let gpu = unsafe { shadow.buffer.GetGPUVirtualAddress() } + shadow.offset(at);
        a.buffer_state(&shadow.buffer, BUFFER_WRITE);
        // SAFETY: an open list; our objects, alive until the lists ran.
        unsafe {
            let l = &a.list;
            l.SetComputeRootSignature(root);
            l.SetPipelineState(pso);
            l.SetComputeRootUnorderedAccessView(0, gpu);
            l.Dispatch(1, 1, 1);
        }
    }

    /// A title buffer for the synthetic ordering tests: the memory the
    /// test owns (`title`), with its host copy.
    pub(crate) fn title_buffer(title: &mut [u8]) -> std::sync::Arc<crate::shadow::Shadow> {
        let at = title.as_mut_ptr() as u64;
        crate::shadow::register(1, at, title.len() as u64);
        crate::shadow::find(at, title.len() as u64).unwrap()
    }

    /// List A's shaders write a range of a title buffer's host copy; list
    /// B, submitted in the same execute, CPU-copies title memory out of
    /// that range (as a title buffer copy whose source is there). In title
    /// order the copy runs after the shaders: the submission must split,
    /// so the copy reads what they wrote once it went back. Without the
    /// split both lists run before the write lands and the copy keeps the
    /// old bytes.
    #[test]
    fn a_copy_of_memory_an_earlier_list_wrote_waits_for_the_write() {
        if in_warp_child("host::tests::a_copy_of_memory_an_earlier_list_wrote_waits_for_the_write")
        {
            return;
        }
        let Some(host) = host() else {
            eprintln!("no host device; skipped");
            return;
        };
        if !on_warp(host) {
            eprintln!("not on the software adapter; skipped");
            return;
        }
        // Title memory: 64 KiB the test owns (the copies read it through
        // the kernel, as they do the title's). The write at the start, the
        // copy out of it, its result past the middle.
        let mut title = vec![0u8; 1 << 16];
        let at = title.as_ptr() as u64;
        let shadow = title_buffer(&mut title);
        let (root, pso) = write_pipeline(host);

        let (a, b) = (host.new_list().unwrap(), host.new_list().unwrap());
        record_gpu_write(&a, &shadow, at, (&root, &pso));
        a.close();
        let to = title[4096..].as_mut_ptr();
        let from = title.as_ptr();
        b.defer_copy(to, from, 256);
        b.close();

        let before = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        host.execute(&[&a, &b], "synthetic copy after a GPU write");
        let after = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            after - before,
            2,
            "the copy must wait for the earlier list's write: one submission each"
        );
        for (i, &got) in title[4096..4096 + 256].as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(got),
                WRITE_PATTERN + i as u32,
                "the copy read what the shaders wrote"
            );
        }
        for (i, &got) in title[..256].as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(got),
                WRITE_PATTERN + i as u32,
                "the write went back into title memory"
            );
        }
        crate::shadow::unregister(1, at);
    }

    /// As [`a_copy_of_memory_an_earlier_list_wrote_waits_for_the_write`],
    /// but the copy's source lies outside the range the shaders write:
    /// it reads none of what they write, so the lists keep batching in one
    /// submission (and the copy reads what was there when it ran).
    #[test]
    fn a_nonoverlapping_copy_stays_batched_with_the_write() {
        if in_warp_child("host::tests::a_nonoverlapping_copy_stays_batched_with_the_write") {
            return;
        }
        let Some(host) = host() else {
            eprintln!("no host device; skipped");
            return;
        };
        if !on_warp(host) {
            eprintln!("not on the software adapter; skipped");
            return;
        }
        let mut title = vec![0u8; 1 << 16];
        let at = title.as_ptr() as u64;
        let shadow = title_buffer(&mut title);
        let (root, pso) = write_pipeline(host);
        // The copy's source, written by the test: outside the write's
        // range.
        title[8192..8192 + 256].fill(0xAA);

        let (a, b) = (host.new_list().unwrap(), host.new_list().unwrap());
        record_gpu_write(&a, &shadow, at, (&root, &pso));
        a.close();
        let to = title[4096..].as_mut_ptr();
        let from = title[8192..].as_ptr();
        b.defer_copy(to, from, 256);
        b.close();

        let before = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        host.execute(&[&a, &b], "synthetic copy next to a GPU write");
        let after = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            after - before,
            1,
            "the copy reads none of the write: one submission"
        );
        assert!(
            title[4096..4096 + 256].iter().all(|&b| b == 0xAA),
            "the copy read what was there when it ran"
        );
        for (i, &got) in title[..256].as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(got),
                WRITE_PATTERN + i as u32,
                "the write went back into title memory"
            );
        }
        crate::shadow::unregister(1, at);
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Initial and final states of the host subresources this list uses.
    /// Lists are recorded on many threads and run in submission order,
    /// so `prelude` reconciles the states at submission. These entries
    /// also keep the textures alive until the list is reset.
    states: Mutex<States>,
    /// The states of title buffer copies and indirect scratch the list uses.
    buffers: Mutex<BufferStates<ID3D12Resource>>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// Memory handed out for the list's recording, kept until it is finished
    /// when the title reset or dropped the list while a submission of it was
    /// still to run.
    kept_uploads: Mutex<Vec<Vec<u8>>>,
    /// Dynamic descriptor slots kept alongside `kept_uploads` until the pending
    /// submission completes.
    kept_dynamic: Mutex<Vec<(Arc<crate::descriptor::host::HostHeap>, u32, u32)>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Packed indirect arguments and count, reused in command order.
    indirect_scratch: Mutex<Option<(ID3D12Resource, u64)>>,
    #[cfg(feature = "diagnostics")]
    dred_id: u64,
    /// GPU-writable scratch for indirect rectangle expansion (`rectangles`):
    /// expanded vertices, rewritten draw arguments and per-draw atomic
    /// counters, reused across draws and across resets.
    rect_scratch: Mutex<RectScratch>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

impl Drop for HostList {
    fn drop(&mut self) {
        self.release_kept();
    }
}

// SAFETY: D3D12 lists and allocators may be used from any thread, one at
// a time: the title records a list on one thread at a time, and a
// submission runs it after its recording ended (a reset while one is to
// run moves recording to a new list). The rest is locked.
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<usize, RecordedTexture>;

/// Every read of a host copy of title memory (vertex, index, constant and
/// shader-resource data, indirect arguments).
pub(crate) const BUFFER_READ: u32 = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
/// Shader writes to a host copy of title memory.
pub(crate) const BUFFER_WRITE: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;

/// The states of tracked buffers one list uses, by resource: the state it
/// starts in and the one it is in now. A buffer is in
/// the common state between lists: the list's prelude moves it to its first
/// state and the list moves it back before it closes. A buffer a shader
/// writes and a later draw reads as vertices (#199, #264) is transitioned in
/// between; the implicit promotion from common that it relied on allows
/// either use in a list, not both.
///
/// shortcut: buffers read only through directly indexed heaps are not
/// tracked, and a command that both writes a buffer and reads it as
/// vertices, indices or arguments leaves it in the read state.
pub(crate) struct BufferStates<R> {
    by_copy: std::collections::HashMap<usize, (u32, u32, R)>,
}

impl<R> Default for BufferStates<R> {
    fn default() -> Self {
        BufferStates {
            by_copy: std::collections::HashMap::new(),
        }
    }
}

impl<R> BufferStates<R> {
    /// The list uses `copy` (`resource`) in `wanted`: the state it leaves,
    /// if a transition must be recorded now. A first use records nothing:
    /// the prelude moves the buffer there.
    pub(crate) fn update(&mut self, copy: usize, resource: R, wanted: u32) -> Option<u32> {
        match self.by_copy.entry(copy) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, resource));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// The prelude's transitions: each buffer from common to its first state.
    pub(crate) fn firsts(&self) -> impl Iterator<Item = (&R, u32)> {
        self.by_copy.values().map(|(first, _, r)| (r, *first))
    }

    /// The list's closing transitions: each buffer from its last state back
    /// to common.
    pub(crate) fn lasts(&self) -> impl Iterator<Item = (&R, u32)> {
        self.by_copy.values().map(|(_, last, r)| (r, *last))
    }

    pub(crate) fn clear(&mut self) {
        self.by_copy.clear();
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

#[cfg(feature = "diagnostics")]
const DRED_LIST_PREFIX: &str = "VIctory command list ";

fn breadcrumb_list(
    runs: &[&HostList],
    object: Option<*mut std::ffi::c_void>,
    debug_name: PCWSTR,
) -> Option<usize> {
    let index = runs.iter().position(|r| Some(r.list.as_raw()) == object);
    #[cfg(feature = "diagnostics")]
    return index.or_else(|| {
        if debug_name.is_null() {
            return None;
        }
        // SAFETY: the DRED query owns its debug names while the device lives.
        let name = unsafe { debug_name.to_string().ok()? };
        let id: u64 = name.strip_prefix(DRED_LIST_PREFIX)?.parse().ok()?;
        runs.iter().position(|r| r.dred_id == id)
    });
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = debug_name;
        index
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// Per-list GPU-writable buffers for indirect rectangle expansion
/// (`crate::rectangles`): one vertex scratch buffer (reused per draw after
/// its VB -> UAV barrier), one rewritten-arguments buffer (grown if a draw
/// has more records), and chunks of 4-byte atomic counters zeroed before
/// submission.
#[derive(Default)]
struct RectScratch {
    vertices: Option<ID3D12Resource>,
    args: Option<(ID3D12Resource, u64)>,
    counters: Vec<(ID3D12Resource, usize, u64)>,
    counter_chunk: usize,
    counter_used: u32,
}

// SAFETY: D3D12 resources are free-threaded; the counter pointers are
// persistent mappings written only before submission.
unsafe impl Send for RectScratch {}

/// Counters per chunk (4 KiB = 1024 words).
const RECT_COUNTER_CHUNK: u32 = 1024;
/// Initial rewritten-arguments buffer size (64 KiB = 4096 draw records).
const RECT_ARGS_MIN: u64 = 64 << 10;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(usize, usize, Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// A texture-to-buffer readback copied from a mapped readback buffer into
/// title memory once the GPU submission completes. Only the rows, columns
/// and depth slices the GPU copied are written: `row_bytes` of each of
/// `lines` destination rows, pitched `dst_pitch` apart, from the staging
/// rows pitched `src_pitch` apart. Title pixels outside the copied region
/// keep their bytes.
struct Readback {
    dst: usize,
    src: usize,
    dst_pitch: usize,
    src_pitch: usize,
    lines: usize,
    row_bytes: usize,
}

impl Readback {
    fn title_range(&self) -> TitleRange {
        let size = self.dst_pitch.saturating_mul(self.lines.saturating_sub(1));
        let end = self.dst.saturating_add(size).saturating_add(self.row_bytes);
        (self.dst, end)
    }
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload/readback buffers its GPU commands name, kept until the list
/// is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Post-submission readbacks from GPU textures into title buffers.
    readbacks: Vec<Readback>,
    #[cfg(feature = "diagnostics")]
    probes: Vec<Box<dyn Fn() + Send + Sync>>,
    /// Host objects the list's commands name (pipelines, root signatures):
    /// the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Retain each bound heap and the resources its slots own.
    descriptor_heaps: Vec<Arc<crate::descriptor::host::HostHeap>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// A recycled host list (reset and open), or a new one when the pool is
    /// empty.
    pub(crate) fn acquire_list(&self) -> windows::core::Result<HostList> {
        let pooled = self
            .free_lists
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop();
        if let Some(list) = pooled {
            #[cfg(test)]
            self.reused_lists.fetch_add(1, Ordering::Relaxed);
            list.reset();
            return Ok(list);
        }
        self.new_list()
    }

    #[cfg(test)]
    pub(crate) fn reused_lists(&self) -> u64 {
        self.reused_lists.load(Ordering::Relaxed)
    }

    /// Returns a finished, unreferenced host list to the pool up to
    /// [`LIST_POOL_CAP`], releasing any kept upload blocks and dynamic slots
    /// immediately.
    pub(crate) fn recycle_list(&self, list: HostList) {
        list.release_kept();
        let mut pool = self.free_lists.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < LIST_POOL_CAP {
            pool.push(list);
        }
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            #[cfg(feature = "diagnostics")]
            let dred_id = {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                let id = NEXT.fetch_add(1, Ordering::Relaxed);
                name(&list, || format!("{DRED_LIST_PREFIX}{id}"));
                id
            };
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                buffers: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                kept_uploads: Mutex::default(),
                kept_dynamic: Mutex::default(),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                indirect_scratch: Mutex::default(),
                rect_scratch: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                dred_id,
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList], operation: &str) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let woken = {
            let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
            self.execute_locked(runs, &mut submitted, operation)
        };
        woken.run();
    }

    /// Records and executes a standalone host read, restoring the source's states.
    /// Keep the submission lock from the state snapshot through execution.
    pub(crate) fn execute_texture_read<T>(
        &self,
        list: &HostList,
        source: &HostTexture,
        subresource: u32,
        operation: &str,
        record: impl FnOnce(&ID3D12GraphicsCommandList) -> T,
    ) -> T {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let mut barriers = Vec::new();
        let mut transition = |restore| {
            barriers.clear();
            source.visit_states(subresource, |index, state| {
                let state = D3D12_RESOURCE_STATES(state as i32);
                let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
                if state != copy {
                    let (before, after) = if restore { (copy, state) } else { (state, copy) };
                    barriers.push(transition_subresource(&source.texture, index, before, after));
                }
            });
            if !barriers.is_empty() {
                // SAFETY: an open list and a live source.
                unsafe { list.list.ResourceBarrier(&barriers) };
            }
        };
        transition(false);
        let result = record(&list.list);
        transition(true);
        list.close();
        let woken = self.execute_locked(&[list], &mut submitted, operation);
        drop(submitted);
        woken.run();
        result
    }

    fn execute_locked(
        &self,
        runs: &[&HostList],
        submitted: &mut u64,
        operation: &str,
    ) -> crate::fence::Woken {
        // SAFETY: a status query on our device.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            self.submission_failed(
                operation,
                runs,
                format_args!("device removed before preparation, last fence {submitted}"),
            );
        }
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        if tracing::enabled!(tracing::Level::TRACE) {
            for (index, &run) in runs.iter().enumerate() {
                let state_count = run.states.lock().unwrap_or_else(|e| e.into_inner()).len();
                let (copies, writes) = {
                    let deferred = run.deferred();
                    let frame = crate::queue::presented() + 1;
                    if crate::capture::listed(frame) {
                        let shader_writes: Vec<_> = deferred
                            .copies
                            .iter()
                            .filter_map(|step| match step {
                                Step::GpuWrite(_, at, len) => Some((*at, *len)),
                                _ => None,
                            })
                            .collect();
                        if !shader_writes.is_empty() {
                            tracing::trace!(
                                frame, index, next_fence = *submitted + 1, ?shader_writes,
                                "captured shader write ranges"
                            );
                        }
                    }
                    (deferred.copies.len(), deferred.writes.len())
                };
                let pipelines = run.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone();
                tracing::trace!(
                    operation,
                    next_fence = *submitted + 1,
                    index,
                    lists = runs.len(),
                    state_count,
                    copies,
                    writes,
                    ?pipelines,
                    notes = %last_notes(&[run]),
                    "preparing host list"
                );
            }
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each of the submission's runs' lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        // The runs of the submission being prepared.
        let mut batch: Vec<usize> = Vec::with_capacity(runs.len());
        let mut woken = crate::fence::Woken::default();
        let mut batcher = Batch::default();
        // Title ranges still awaiting GPU copy-back or completion writes.
        let mut writing: Vec<TitleRange> = Vec::new();
        // Where the last submission ended (the time between it and the
        // next one is preparing, `SUBMISSION_US[0]`).
        let mut done = started;
        for (index, list) in runs.iter().enumerate() {
            let (copies, own_writes, self_overlap) = list.copy_plan();
            if !batch.is_empty() && copies.iter().any(|&(at, len)| overlaps_write(&writing, at, len))
            {
                // Complete the earlier lists' writes before reading them.
                // Copies from unrelated ranges can still share a batch.
                batcher.title_written();
                #[cfg(feature = "diagnostics")]
                tracing::debug!(
                    operation,
                    next_fence = *submitted + 1,
                    index,
                    writes = ?writing,
                    "the submission splits: a copy reads a range an earlier list writes"
                );
                self.run_batch(
                    runs,
                    &batch,
                    &lists,
                    &firsts,
                    submitted,
                    operation,
                    &mut woken,
                    &mut done,
                );
                lists.clear();
                batch.clear();
                firsts.clear();
                writing.clear();
            }
            if self_overlap {
                // One list: the copy is prepared before the list runs,
                // whatever the recorded order; only a split between lists
                // can restore title order.
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        "a host list copies title memory a dispatch of its own wrote: \
                         the copy is prepared before the list runs"
                    );
                });
            }
            list.run_deferred(&mut batcher, &mut woken);
            writing.extend(own_writes);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
            batch.push(index);
        }
        batcher.title_written();
        self.run_batch(
            runs,
            &batch,
            &lists,
            &firsts,
            submitted,
            operation,
            &mut woken,
            &mut done,
        );
        woken
    }

    /// Runs the submission's lists and waits for them, then copies the
    /// ranges its shaders wrote back into title memory and applies the
    /// lists' bottom-of-pipe writes, so the next submission's preparation
    /// (its CPU copies of title memory) sees what they wrote.
    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        runs: &[&HostList],
        batch: &[usize],
        lists: &[Option<ID3D12CommandList>],
        firsts: &[usize],
        submitted: &mut u64,
        operation: &str,
        woken: &mut crate::fence::Woken,
        done: &mut std::time::Instant,
    ) {
        if lists.is_empty() {
            return;
        }
        let prepared = std::time::Instant::now();
        SUBMISSION_US[0].fetch_add(
            (prepared - *done).as_micros() as u64,
            Ordering::Relaxed,
        );
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![lists]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [lists];
        #[cfg(not(feature = "diagnostics"))]
        let _ = firsts;
        for (i, group) in groups.iter().enumerate() {
            // SAFETY: a status query on our device.
            if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                self.submission_failed(
                    operation,
                    runs,
                    format_args!("device removed before execution, last fence {submitted}"),
                );
            }
            *submitted += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            tracing::trace!(operation, fence = *submitted, host_lists = group.len(), "submitting host lists");
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                if let Err(error) = self.queue.Signal(&self.fence, *submitted) {
                    self.submission_failed(
                        operation,
                        runs,
                        format_args!("signal host fence: {error}"),
                    );
                }
                if let Err(error) = self.fence.SetEventOnCompletion(*submitted, self.event) {
                    self.submission_failed(
                        operation,
                        runs,
                        format_args!("register host fence event: {error}"),
                    );
                }
                let waited = WaitForSingleObject(self.event, INFINITE);
                if waited != WAIT_OBJECT_0 {
                    self.submission_failed(
                        operation,
                        runs,
                        format_args!("host fence wait returned {:#x}", waited.0),
                    );
                }
                if self.device.GetDeviceRemovedReason().is_err() {
                    self.submission_failed(operation, runs, format_args!("device removed"));
                }
            }
            #[cfg(feature = "diagnostics")]
            tracing::trace!(operation, fence = *submitted, "host fence completed");
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[batch[i]]
                    .notes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[batch[i]].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let gpu = prepared.elapsed();
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = batch
                .iter()
                .flat_map(|&r| {
                    runs[r].pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone()
                })
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = batch.len(), ?pipelines, "slow host submission");
        }
        // From here on: what the title must see before the next
        // preparation, and the time between submissions is copying, not
        // preparing.
        *done = std::time::Instant::now();
        // What the submission's shaders wrote goes back before the
        // bottom-of-pipe writes, which tell the title the GPU is done; each
        // range once per submission, however many lists declared it (the
        // world's passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for &r in batch {
            for step in &runs[r].deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && !back
                        .iter()
                        .any(|(s, a, l)| std::sync::Arc::ptr_eq(s, shadow) && (*a, *l) == (*address, *len))
                {
                    back.push((shadow.clone(), *address, *len));
                }
            }
        }
        for (shadow, address, len) in &back {
            let (address, len) = (*address, *len);
            let written = shadow.copy_back(address, len, woken);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        // Texture readbacks land in title memory before the completion
        // writes: a title thread woken by a completion write must see the
        // bytes the GPU copied. Per destination row: only the copied
        // columns are written, so the rest of each row and the gaps
        // between slices keep the title's pixels.
        for &r in batch {
            let deferred = runs[r].deferred();
            #[cfg(feature = "diagnostics")]
            for probe in &deferred.probes {
                probe();
            }
            for rb in &deferred.readbacks {
                let written = if rb.dst_pitch == rb.row_bytes && rb.src_pitch == rb.row_bytes {
                    // Packed rows: one contiguous range.
                    let bytes = rb.row_bytes.saturating_mul(rb.lines);
                    crate::watch::writing(rb.dst, bytes);
                    woken.add(crate::fence::write_and_collect(rb.dst, bytes, || {
                        write_mapped(rb.dst as *mut u8, rb.src as *const u8, bytes);
                    }));
                    bytes
                } else {
                    let mut n = 0;
                    for line in 0..rb.lines {
                        let to = rb.dst + line * rb.dst_pitch;
                        let from = rb.src + line * rb.src_pitch;
                        crate::watch::writing(to, rb.row_bytes);
                        woken.add(crate::fence::write_and_collect(to, rb.row_bytes, || {
                            write_mapped(to as *mut u8, from as *const u8, rb.row_bytes);
                        }));
                        n += rb.row_bytes;
                    }
                    n
                };
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Queues these writes release run their work once no lock is held:
        // it submits lists itself (this lock, the lists' `deferred`).
        for &r in batch {
            for &(at, what, bytes) in &runs[r].deferred().writes {
                woken.add(what.write(at as *mut u8, bytes));
                #[cfg(feature = "diagnostics")]
                if crate::fence::trace_memory_waits() {
                    tracing::trace!(
                        frame = crate::queue::presented() + 1,
                        fence = *submitted,
                        address = format_args!("{at:#x}"),
                        ?what,
                        bytes,
                        "applied title command list completion write"
                    );
                }
            }
        }
        self.log_messages();
    }

    fn submission_failed(
        &self,
        operation: &str,
        runs: &[&HostList],
        reason: std::fmt::Arguments<'_>,
    ) -> ! {
        fail(format_args!(
            "host GPU {operation} submission failed ({reason}, {} lists){}{}",
            runs.len(),
            self.removal_data(runs),
            last_notes(runs)
        ));
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = breadcrumb_list(runs, ours, p.pCommandListDebugNameW);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = breadcrumb_list(runs, list, p.pCommandListDebugNameW)
                        .map(|index| runs[index]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

/// A range of title memory: its start and its end.
type TitleRange = (usize, usize);

/// Whether a CPU copy of title memory `[at, at + len)` reads a range the
/// submission's shaders write on the GPU (`writing`: the title ranges of
/// the GPU writes declared so far): the two overlap, so in title order the
/// copy runs after the shaders and must not read before their lists ran
/// and the writes went back. Disjoint and merely touching ranges (one ends
/// where the other begins) do not overlap.
fn overlaps_write(writing: &[TitleRange], at: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let end = at.saturating_add(len);
    writing.iter().any(|&(from, to)| from < to && at < to && from < end)
}

/// The copy-order rule within one recording: a copy of title memory
/// splits only when its source intersects a range the recording fills
/// after it has run — a shader write (`GpuWrite`, as title addresses), a
/// readback, or a bottom-of-pipe completion write — all of which land in
/// title memory only once the list has run, while the copy is prepared
/// before it. Disjoint and merely touching ranges (one ends where the
/// other begins) do not overlap, and an empty read reads nothing.
fn copy_intersects_write(
    steps: &[Step],
    readbacks: &[Readback],
    writes: &[(usize, crate::queue::Written, usize)],
    at: usize,
    len: usize,
) -> bool {
    if len == 0 {
        return false;
    }
    let end = at.saturating_add(len);
    let inside = |(from, to): (usize, usize)| from < to && at < to && from < end;
    steps.iter().any(|step| match step {
        Step::GpuWrite(shadow, address, written) => {
            let (from, to) = shadow.clamped(*address, *written);
            inside((from, to))
        }
        _ => false,
    }) || readbacks.iter().map(Readback::title_range).any(inside)
        || writes
            .iter()
            .map(|&(at, _, bytes)| (at, at.saturating_add(bytes)))
            .any(inside)
}

impl HostList {
    /// Keeps `blocks` (memory handed out for the recording) until the list
    /// finishes executing and is recycled or dropped.
    pub(crate) fn keep_uploads(&self, blocks: Vec<Vec<u8>>) {
        if !blocks.is_empty() {
            self.kept_uploads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(blocks);
        }
    }

    /// Keeps dynamic descriptor slots alive until this host list finishes
    /// executing and is recycled or dropped.
    pub(crate) fn keep_dynamic(
        &self,
        slots: Vec<(Arc<crate::descriptor::host::HostHeap>, u32, u32)>,
    ) {
        if !slots.is_empty() {
            self.kept_dynamic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(slots);
        }
    }

    /// Frees kept upload blocks and returns kept dynamic descriptor slots.
    fn release_kept(&self) {
        self.kept_uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let dynamic =
            std::mem::take(&mut *self.kept_dynamic.lock().unwrap_or_else(|e| e.into_inner()));
        for (heap, first, n) in dynamic {
            heap.release(first, n);
        }
    }

    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // The buffers it used back to common, where the next list's prelude
        // expects them.
        let back: Vec<D3D12_RESOURCE_BARRIER> = self
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .lasts()
            .map(|(b, last)| {
                transition(
                    b,
                    D3D12_RESOURCE_STATES(last as i32),
                    D3D12_RESOURCE_STATE_COMMON,
                )
            })
            .collect();
        if !back.is_empty() {
            // SAFETY: an open list and buffers the list holds.
            unsafe { self.list.ResourceBarrier(&back) };
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.release_kept();
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        {
            let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
            (rect.counter_chunk, rect.counter_used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves host subresources to `wanted` in this list, emitting inline
    /// barriers for states that are already known and need to change.
    pub(crate) fn transition(
        &self,
        copy: Arc<HostTexture>,
        subresource: u32,
        wanted: u32,
        emit: impl FnMut(&ID3D12Resource, u32, u32, u32),
    ) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .entry(Arc::as_ptr(&copy) as usize)
            .or_insert_with(|| RecordedTexture::new(copy))
            .record(subresource, wanted, emit);
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn recorded_state(&self, copy: &Arc<HostTexture>, subresource: u32) -> Option<u32> {
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .get(&(Arc::as_ptr(copy) as usize))
            .and_then(|state| state.last(subresource))
    }

    /// Moves a tracked buffer to `wanted` ([`BUFFER_READ`] or [`BUFFER_WRITE`])
    /// in this list and retains it. See [`BufferStates`].
    pub(crate) fn buffer_state(&self, buffer: &ID3D12Resource, wanted: u32) {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(before) = buffers.update(buffer.as_raw() as usize, buffer.clone(), wanted) {
            let barrier = transition(
                buffer,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a buffer the list holds.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// At submission: reconcile used subresources and prepare the buffers
    /// for the initial states the list expects.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let mut barriers = Vec::new();
        for recorded in states.values() {
            recorded.reconcile(|index, before, after| {
                barriers.push(transition_subresource(
                    &recorded.copy.texture,
                    index,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATES(after as i32),
                ));
            });
        }
        barriers.extend(buffers.firsts().map(|(buffer, first)| {
            transition(buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATES(first as i32))
        }));
        if barriers.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                list.ResourceBarrier(&barriers);
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    /// Whether a copy of title memory at `[src, src + len)` reads a range
    /// an earlier step of this recording fills once the list has run: a
    /// shader write (`GpuWrite`), a texture readback, or a bottom-of-pipe
    /// completion write. The copy is prepared before the list runs, so
    /// recording it here would read the old bytes; the recording splits
    /// at the copy (see `list.rs`'s `list_for_title_copy`).
    pub(crate) fn copy_reads_earlier_write(&self, src: *const u8, len: usize) -> bool {
        let deferred = self.deferred();
        copy_intersects_write(
            &deferred.copies,
            &deferred.readbacks,
            &deferred.writes,
            src as usize,
            len,
        )
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the list needs from the submission it goes into: the title
    /// ranges its CPU copies read (the sources of its title-to-title
    /// copies and of its fills of upload memory), the title ranges its
    /// shaders write (`GpuWrite` steps, cut to their buffers), and whether
    /// a copy of its own reads one of its own earlier GPU writes (out of
    /// title order within one list, which no split can fix).
    fn copy_plan(&self) -> (Vec<TitleRange>, Vec<TitleRange>, bool) {
        let deferred = self.deferred();
        let mut sources = Vec::new();
        let mut writes = Vec::new();
        let mut self_overlap = false;
        for step in &deferred.copies {
            match step {
                Step::Copy(_, src, len) | Step::Fill(_, src, len) => {
                    if overlaps_write(&writes, *src, *len) {
                        self_overlap = true;
                    }
                    sources.push((*src, *len));
                }
                Step::GpuWrite(shadow, address, len) => {
                    writes.push(shadow.clamped(*address, *len));
                }
                _ => {}
            }
        }
        writes.extend(deferred.readbacks.iter().map(Readback::title_range));
        writes.extend(
            deferred
                .writes
                .iter()
                .map(|&(at, _, bytes)| (at, at.saturating_add(bytes))),
        );
        (sources, writes, self_overlap)
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Copies a GPU texture readback from `src` into title memory at `dst`
    /// once the list has finished running on the host GPU: `lines` rows of
    /// `row_bytes`, pitched apart on both sides.
    pub(crate) fn defer_readback(
        &self,
        dst: *mut u8,
        src: *const u8,
        dst_pitch: usize,
        src_pitch: usize,
        lines: usize,
        row_bytes: usize,
    ) {
        self.deferred().readbacks.push(Readback {
            dst: dst as usize,
            src: src as usize,
            dst_pitch,
            src_pitch,
            lines,
            row_bytes,
        });
    }

    /// Reads a list-owned diagnostic result after the batch's fence passes.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn defer_probe(&self, probe: impl Fn() + Send + Sync + 'static) {
        self.deferred().probes.push(Box::new(probe));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset (a reset waits for the submission, so the GPU's copy into it
    /// is done and the deferred readbacks have run before the memory is
    /// reused).
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let (buffer, data) = host.readback_buffer(size)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// Each recorded use enters `BufferStates`, which keeps earlier buffers
    /// alive even when this slot grows or recording continues on a new segment.
    pub(crate) fn indirect_scratch(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut scratch = self.indirect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, capacity)) = &*scratch
            && *capacity >= bytes
        {
            return Some(buffer.clone());
        }
        let capacity = bytes.next_multiple_of(64 << 10);
        let buffer = host()?.default_buffer(capacity)?;
        name(&buffer, || format!("indirect scratch {capacity:#x}"));
        *scratch = Some((buffer.clone(), capacity));
        Some(buffer)
    }

    /// The list's GPU-writable scratch buffer for expanded indirect
    /// rectangle vertices (`capacity` bytes), created on first use.
    pub(crate) fn rect_vertices(&self, capacity: u64) -> Option<ID3D12Resource> {
        let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if rect.vertices.is_none() {
            let buffer = host()?.default_buffer(capacity)?;
            name(&buffer, || format!("rect scratch {capacity:#x}"));
            rect.vertices = Some(buffer);
        }
        rect.vertices.clone()
    }

    /// A GPU-writable buffer of at least `bytes` for rewritten indirect
    /// draw arguments, kept alive until the list is reset.
    pub(crate) fn rect_args(&self, bytes: u64) -> Option<ID3D12Resource> {
        let want = bytes.max(RECT_ARGS_MIN).next_multiple_of(256);
        let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buf, len)) = &rect.args
            && *len >= want
        {
            return Some(buf.clone());
        }
        let buffer = host()?.default_buffer(want)?;
        name(&buffer, || format!("rect args {want:#x}"));
        let old = rect.args.replace((buffer.clone(), want));
        drop(rect);
        if let Some((old, _)) = old {
            self.keep(&old);
        }
        Some(buffer)
    }

    /// The host GPU address of a 4-byte atomic counter for one indirect
    /// rectangle draw, zeroed at each submission before the list runs.
    pub(crate) fn rect_counter(&self) -> Option<u64> {
        let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if rect.counter_used == RECT_COUNTER_CHUNK {
            rect.counter_chunk += 1;
            rect.counter_used = 0;
        }
        if rect.counter_chunk == rect.counters.len() {
            let bytes = u64::from(RECT_COUNTER_CHUNK) * 4;
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            name(&buffer, || "rect counters".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            rect.counters.push((buffer, data as usize, gpu));
            rect.counter_used = 0;
        }
        let (_, data, gpu) = rect.counters[rect.counter_chunk];
        let slot = rect.counter_used;
        rect.counter_used += 1;
        drop(rect);
        if slot == 0 {
            let bytes = RECT_COUNTER_CHUNK as usize * 4;
            // SAFETY: the list's own mapped chunk, zeroed before it runs.
            self.defer_host(move || unsafe {
                std::ptr::write_bytes(data as *mut u8, 0, bytes);
            });
        }
        Some(gpu + u64::from(slot) * 4)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    pub(crate) fn keep_descriptor_heap(&self, heap: &Arc<crate::descriptor::host::HostHeap>) {
        self.deferred().descriptor_heaps.push(heap.clone());
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, dst: usize, len: usize, work: impl Fn() + Send + 'static) {
        self.deferred()
            .copies
            .push(Step::Run(dst, len, Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch, woken: &mut crate::fence::Woken) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written_over(dst, dst.saturating_add(len));
                    woken.add(crate::fence::write_and_collect(dst, len, || {
                        copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    }));
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(_, _, work) | Step::Host(work) => {
                    if let Step::Run(dst, len, _) = step {
                        batch.title_written_over(*dst, (*dst).saturating_add(*len));
                        woken.add(crate::fence::write_and_collect(*dst, *len, work));
                    } else {
                        work();
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this
                    // buffer's refreshes the range's pages overlap are
                    // done before the snapshot takes them.
                    batch.gpu_write(shadow, *address, *len);
                    shadow.writing(*address, *len);
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step whose title write their
/// pages overlap, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

/// Whether a refresh's hull `[a, b)` and a title write `[from, to)`
/// share a 64 KiB page: a refresh's write-watch query rounds to whole
/// pages (`crate::watch::writes`), so its reset takes a page's kernel
/// bits together even when the ranges themselves are disjoint bytes, and
/// a write into a page the query touches must see it done first.
fn shares_page(a: usize, b: usize, from: usize, to: usize) -> bool {
    const PAGE: usize = crate::watch::PAGE;
    a < b && from < to && a / PAGE <= (to - 1) / PAGE && from / PAGE <= (b - 1) / PAGE
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        match self
            .refreshes
            .iter_mut()
            .find(|(s, ..)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that may write title memory anywhere (a submission
    /// split, or the end of the batch, which must leave nothing pending):
    /// every refresh is done first, and textures are checked again after.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    /// Before a step that writes title memory `[from, to)`: refreshes
    /// sharing a page with the write are done first (their query must
    /// take the bytes before the write does), and textures are checked
    /// again after it. Refreshes of other pages stay pending: the write
    /// cannot change the bytes they copy, and keeping them merges their
    /// queries — a `GetWriteWatch` call costs tens of microseconds mostly
    /// regardless of its span (more the bigger the watched pool), so one
    /// query of a buffer's hull per submission beats several of its
    /// pieces.
    fn title_written_over(&mut self, from: usize, to: usize) {
        let mut i = 0;
        while i < self.refreshes.len() {
            let (_, a, b) = &self.refreshes[i];
            if shares_page(*a, *b, from, to) {
                let (shadow, a, b) = self.refreshes.remove(i);
                self.update(shadow, a, b);
            } else {
                i += 1;
            }
        }
        self.textures.clear();
    }

    /// Before a step that gives shaders `[address, address + len)` of a
    /// buffer copy: that buffer's refreshes sharing a page with the range
    /// are done first, so the snapshot `Shadow::writing` takes (the
    /// shaders start from the title's data) has the refreshed bytes.
    /// Refreshes of other buffers read title memory, which the shaders'
    /// writes reach only at copy-back — after the lists ran, with this
    /// batch's end flush before that — so they stay pending; so do the
    /// other pages of this buffer.
    fn gpu_write(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let to = address.saturating_add(len.max(1));
        let mut i = 0;
        while i < self.refreshes.len() {
            let (s, a, b) = &self.refreshes[i];
            if std::sync::Arc::ptr_eq(s, shadow) && shares_page(*a, *b, address, to) {
                let (shadow, a, b) = self.refreshes.remove(i);
                self.update(shadow, a, b);
            } else {
                i += 1;
            }
        }
    }

    /// Runs every pending refresh, in recorded order.
    fn flush(&mut self) {
        for (shadow, from, to) in std::mem::take(&mut self.refreshes) {
            self.update(shadow, from, to);
        }
    }

    /// Runs one pending refresh: `[from, to)` of `shadow`'s copy, timed.
    fn update(&mut self, shadow: std::sync::Arc<crate::shadow::Shadow>, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 || title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe {
        ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0
            && done == len
    }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer in the readback heap, mapped for reading: the GPU copies a
    /// texture into it, and the submission's readback steps copy the rows
    /// on to title memory.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows the source as frame `frame` (saved if captures are on).
    /// Returns `None` if nothing was shown. In diagnostics builds it reports
    /// whether the frame was black.
    pub(crate) fn present(&self, source: &HostTexture, frame: u64) -> Option<bool> {
        let texture = &source.texture;
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            #[cfg(feature = "diagnostics")]
            let (mut capture, mut probe) = (None, None);
            self.execute_texture_read(
                &p.copy,
                source,
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                "presentation copy",
                |list| {
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_PRESENT,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    list.CopyResource(&back, texture);
                    #[cfg(feature = "diagnostics")]
                    {
                        capture = crate::capture::wanted(frame)
                            .then(|| crate::capture::read_back(&self.device, list, texture))
                            .flatten();
                        probe = crate::capture::probe(&self.device, list, texture);
                    }
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                },
            );
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            if let Err(error) = p.swap_chain.Present(0, DXGI_PRESENT(0)).ok() {
                if self.device.GetDeviceRemovedReason().is_err() {
                    fail(format_args!("host GPU swap-chain present failed: {error}"));
                }
                tracing::warn!(%error, "host swap-chain present failed");
                return None;
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            // The window may belong to a title thread. Let that thread maximize
            // it without waiting here or resizing the presentation buffers.
            if !IsZoomed(hwnd).as_bool() {
                let _ = ShowWindowAsync(hwnd, SW_MAXIMIZE);
            }
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A GPU-only default-heap buffer that shaders can write (unordered
    /// access). Starts in the common state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a default-heap buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no default buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    transition_subresource(resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after)
}

pub(crate) fn transition_subresource(
    resource: &ID3D12Resource,
    subresource: u32,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: subresource,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::{Batch, shares_page};
    use std::sync::Arc;

    #[test]
    fn flush_overlap_is_page_rounded() {
        const PAGE: usize = crate::watch::PAGE;
        // Overlapping ranges share a page.
        assert!(shares_page(0, PAGE, 8, 16));
        // Disjoint bytes on one page: a refresh's query takes the page's
        // kernel bits together, so the write must see it done first.
        assert!(shares_page(8, 24, 40, 56));
        // Neighbouring pages do not: the hull's query stops at its last
        // page's end.
        assert!(!shares_page(0, PAGE, PAGE, PAGE + 16));
        assert!(!shares_page(PAGE, PAGE + 16, 0, PAGE));
        // Far apart.
        assert!(!shares_page(0, PAGE, 12 * PAGE, 12 * PAGE + 16));
        assert!(!shares_page(8, 24, 40, 40));
        assert!(shares_page(usize::MAX - 16, usize::MAX, usize::MAX - 8, usize::MAX));
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}

/// The batch's flush around real write-watched title memory: which steps
/// query a hull and in what order, and that the copy ends up with the
/// title's bytes whatever the interleaving. A device is needed for the
/// copies; without one, `VICTORY_D3D12_WARP=1`.
#[cfg(all(test, feature = "diagnostics"))]
mod refresh_flush {
    use super::{Batch, host};
    use crate::watch::PAGE;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    // 0x20_0000, `MEM_WRITE_WATCH` (in SystemServices in windows-sys).
    const WATCH: u32 = 0x20_0000;

    /// A fresh write-watched allocation, freed when dropped.
    struct Watched {
        at: usize,
        len: usize,
    }
    impl Watched {
        fn new(pages: usize) -> Self {
            // SAFETY: committed here, freed in `Drop`.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    pages * PAGE,
                    MEM_RESERVE | MEM_COMMIT | WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(at, 0);
            Watched { at, len: pages * PAGE }
        }
    }
    impl Drop for Watched {
        fn drop(&mut self) {
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
        }
    }

    static RESOURCE: AtomicUsize = AtomicUsize::new(7001);

    /// A shadow over a fresh watched allocation, registered so that
    /// `find` makes its copy exactly the allocation's size.
    fn make_shadow(w: &Watched) -> Arc<crate::shadow::Shadow> {
        let resource = RESOURCE.fetch_add(1, Relaxed);
        crate::shadow::register(resource, w.at as u64, w.len as u64);
        let shadow = crate::shadow::find(w.at as u64, w.len as u64)
            .expect("a device for the shadow (set VICTORY_D3D12_WARP=1)");
        // A first query of everything: fresh allocations can inherit a
        // freed one's journal entries; from here the pages this test
        // dirties are the only stale ones.
        crate::watch::writes(w.at, w.at + w.len, false).expect("write-watched");
        shadow
    }

    /// The shadow's copy mapped for the test to inspect, and to stand in
    /// for the GPU's writes to it (the shadow keeps its own mapping).
    struct Mapped<'a> {
        buffer: &'a ID3D12Resource,
        data: *mut u8,
    }
    impl<'a> Mapped<'a> {
        fn new(shadow: &'a crate::shadow::Shadow) -> Self {
            let mut data = std::ptr::null_mut();
            // SAFETY: a CPU-visible buffer, mapped again.
            unsafe { shadow.buffer.Map(0, None, Some(&mut data)) }.expect("map the copy");
            Mapped { buffer: &shadow.buffer, data: data.cast() }
        }
        fn byte(&self, at: usize) -> u8 {
            // SAFETY: inside the mapping, which spans the shadow.
            unsafe { *self.data.add(at) }
        }
        fn write(&self, at: usize, value: u8) {
            // SAFETY: inside the mapping.
            unsafe { *self.data.add(at) = value };
        }
    }
    impl Drop for Mapped<'_> {
        fn drop(&mut self) {
            // SAFETY: our mapping.
            unsafe { self.buffer.Unmap(0, None) };
        }
    }

    /// Whether the title's write to `at` is still unreported by any query
    /// (its kernel bit is alive): only a query that finds it written
    /// journals it with that asking's number.
    fn bit_alive(at: usize) -> bool {
        let (now, pages) = crate::watch::writes(at, at + PAGE, false).expect("write-watched");
        pages.iter().any(|&(p, w)| p == at / PAGE && w == now)
    }

    /// Writes the watched allocation the way the title's threads do (a
    /// plain write, which leaves a kernel bit).
    fn title_write(at: usize, value: u8) {
        // SAFETY: committed pages of ours.
        unsafe { (at as *mut u8).write(value) };
    }

    /// Reads the watched allocation back (what the title's threads read).
    fn title_read(at: usize) -> u8 {
        // SAFETY: committed pages of ours.
        unsafe { (at as *const u8).read() }
    }

    #[test]
    fn disjoint_title_copies_keep_the_hulls_query_pending() {
        if host().is_none() {
            return;
        }
        let w = Watched::new(16);
        let at = w.at;
        let shadow = make_shadow(&w);
        let mapped = Mapped::new(&shadow);
        let mut batch = Batch::default();
        // The first list's draws read the hull's four pages; the title
        // wrote each of them.
        for page in 0..4usize {
            title_write(at + page * PAGE + 8, 0x40 + page as u8);
        }
        batch.add(&shadow, at, 4 * PAGE);
        // A copy step writing title memory on another page cannot change
        // the bytes the hull copies: the hull stays pending, nothing
        // queries it (a full flush here would walk the hull for nothing).
        batch.title_written_over(at + 12 * PAGE, at + 12 * PAGE + 16);
        assert_eq!(batch.refreshes.len(), 1);
        assert!(bit_alive(at + PAGE + 8), "the disjoint copy left the hull unqueried");
        // A second list reads the same hull; again a disjoint copy.
        for page in 0..4usize {
            title_write(at + page * PAGE + 40, 0x50 + page as u8);
        }
        batch.add(&shadow, at, 4 * PAGE);
        assert_eq!(batch.refreshes.len(), 1, "one list's hulls merge");
        batch.title_written_over(at + 13 * PAGE, at + 13 * PAGE + 16);
        assert!(bit_alive(at + 2 * PAGE + 8), "still nothing queried the hull");
        // The end of the batch flushes everything before the lists run:
        // one query of the hull covers both lists' writes.
        batch.title_written();
        assert!(batch.refreshes.is_empty());
        for page in 0..4usize {
            assert_eq!(mapped.byte(page * PAGE + 8), 0x40 + page as u8);
            assert_eq!(mapped.byte(page * PAGE + 40), 0x50 + page as u8);
        }
    }

    #[test]
    fn a_write_sharing_the_hulls_page_flushes_it_before_itself() {
        if host().is_none() {
            return;
        }
        let w = Watched::new(4);
        let at = w.at;
        let shadow = make_shadow(&w);
        let mapped = Mapped::new(&shadow);
        let mut batch = Batch::default();
        // The title wrote bytes the hull covers ...
        title_write(at + 8, 0xAA);
        title_write(at + 23, 0xAA);
        batch.add(&shadow, at + 8, 16);
        // ... and its copy step writes other bytes of the same 64 KiB
        // page. The pages meet, so the flush comes first.
        batch.title_written_over(at + 40, at + 56);
        assert_eq!(mapped.byte(8), 0xAA, "refreshed before the write");
        assert_eq!(mapped.byte(23), 0xAA);
        // The write lands after the flush's query: its bit is alive (a
        // query after the write would have taken it).
        title_write(at + 40, 0xBB);
        title_write(at + 55, 0xBB);
        assert!(bit_alive(at + 40), "the write came after the flush's query");
        // A later refresh of the whole page takes the write: the page's
        // rounded reset cost the refresh nothing.
        batch.add(&shadow, at, PAGE);
        batch.title_written();
        assert_eq!(mapped.byte(40), 0xBB, "the CPU write is not lost");
        assert_eq!(mapped.byte(55), 0xBB);
        assert_eq!(mapped.byte(8), 0xAA, "and the copied bytes stand");
        assert_eq!(mapped.byte(23), 0xAA);
    }

    #[test]
    fn a_gpu_write_flushes_only_its_own_shadows_overlapping_refresh() {
        if host().is_none() {
            return;
        }
        // Two buffers; each has a pending refresh at submission time.
        let a = Watched::new(4);
        let b = Watched::new(4);
        let s1 = make_shadow(&a);
        let s2 = make_shadow(&b);
        let m1 = Mapped::new(&s1);
        let m2 = Mapped::new(&s2);
        let mut batch = Batch::default();
        title_write(a.at + 8, 0x60);
        title_write(b.at + 8, 0x70);
        batch.add(&s1, a.at, PAGE);
        batch.add(&s2, b.at, PAGE);
        let texture = Arc::new(0u32);
        assert!(batch.texture_due(&texture), "sampled this submission");
        // A step whose shaders write page 0 of the first buffer: its
        // refresh is done first (the snapshot takes the refreshed bytes).
        batch.gpu_write(&s1, a.at, PAGE);
        assert_eq!(m1.byte(8), 0x60, "refreshed before the snapshot");
        assert_eq!(batch.refreshes.len(), 1, "the other buffer's stays pending");
        assert!(std::sync::Arc::ptr_eq(&batch.refreshes[0].0, &s2));
        assert!(bit_alive(b.at + 8), "no query of the other buffer");
        // Shaders do not write textures: no invalidation (as before).
        assert!(!batch.texture_due(&texture));
        // The end of the batch flushes the rest before the lists run.
        batch.title_written();
        assert_eq!(m2.byte(8), 0x70);
        assert!(batch.refreshes.is_empty());
        assert!(batch.texture_due(&texture), "a title write makes them due");
        // The shaders write a byte of the copy after the snapshot; copy
        // back puts just it into title memory (the bytes the snapshot saw
        // refreshed stay alone).
        s1.writing(a.at, PAGE);
        m1.write(16, 0x61);
        let mut woken = crate::fence::Woken::default();
        assert_eq!(s1.copy_back(a.at, PAGE, &mut woken), PAGE);
        woken.run();
        assert_eq!(title_read(a.at + 16), 0x61, "the GPU's byte went back");
        assert_eq!(title_read(a.at + 8), 0x60, "nothing else was written");
        // A later refresh takes the copy-back's bytes and any the title's
        // threads wrote meanwhile.
        title_write(a.at + 24, 0x63);
        batch.add(&s1, a.at, PAGE);
        batch.title_written();
        assert_eq!(m1.byte(16), 0x61, "the copy keeps what it wrote back");
        assert_eq!(m1.byte(24), 0x63, "and takes the title's new bytes");
    }

    /// The flush of a submission's refreshes: one full flush per title
    /// copy (a hull re-walked per copy, the old behaviour) against
    /// flushing only what the copy's pages overlap, with the same bytes
    /// copied in the end. Run with
    /// `cargo test --release -p victory-gpu --features diagnostics
    /// -- --ignored --nocapture flush_per_title_copy`.
    #[test]
    #[ignore = "benchmark; set VICTORY_D3D12_WARP=1"]
    fn flush_per_title_copy_against_overlap_only() {
        if host().is_none() {
            return;
        }
        const BUFFERS: usize = 8;
        const PAGES: usize = 128; // 8 MiB each
        const LISTS: usize = 28;
        const COPIES: usize = 5;
        let buffers: Vec<Watched> = (0..BUFFERS).map(|_| Watched::new(PAGES)).collect();
        let shadows: Vec<_> = buffers.iter().map(make_shadow).collect();
        let maps: Vec<_> = shadows.iter().map(|s| Mapped::new(s)).collect();
        // The copies go to their own memory, as the title's copies between
        // buffers do.
        let scratch = Watched::new(4);
        let run = |overlap_only: bool, mark: u8| {
            let mut batch = Batch::default();
            let started = std::time::Instant::now();
            for list in 0..LISTS {
                for (b, shadow) in shadows.iter().enumerate() {
                    // Each list reads a different stretch of each buffer.
                    let from = (list * 3 + b * 5) % (PAGES - 16);
                    title_write(buffers[b].at + from * PAGE, mark + b as u8);
                    batch.add(shadow, buffers[b].at + from * PAGE, 16 * PAGE);
                }
                for copy in 0..COPIES {
                    let dst = scratch.at + ((list * COPIES + copy) % 4) * PAGE;
                    title_write(dst, mark);
                    if overlap_only {
                        batch.title_written_over(dst, dst + PAGE);
                    } else {
                        batch.title_written();
                    }
                }
            }
            batch.title_written();
            started.elapsed()
        };
        let _ = run(false, 1); // warm up
        let whole = run(false, 2);
        // Both patterns copied the same bytes: every hull is up to date.
        for (b, map) in maps.iter().enumerate() {
            for list in 0..LISTS {
                let from = (list * 3 + b * 5) % (PAGES - 16);
                assert_eq!(map.byte(from * PAGE), 2 + b as u8);
            }
        }
        let over = run(true, 3);
        for (b, map) in maps.iter().enumerate() {
            for list in 0..LISTS {
                let from = (list * 3 + b * 5) % (PAGES - 16);
                assert_eq!(map.byte(from * PAGE), 3 + b as u8, "the last run's bytes");
            }
        }
        println!(
            "{LISTS} lists, {BUFFERS} buffers of {} MiB, {COPIES} title copies per list: \
             full flush per copy {whole:?}, overlap only {over:?} ({:.1}x)",
            PAGES as f64 * PAGE as f64 / 1048576.0,
            whole.as_secs_f64() / over.as_secs_f64(),
        );
    }

    /// The same A/B at the world's sizes: buffers a world heap's ~64 MiB
    /// (the run logs' top hull is 182 MB), a hull per buffer spanning it,
    /// read by every list. The fixed cost of a `GetWriteWatch` call grows
    /// with the watched pool (the title keeps 9-13 GiB), so the ratio
    /// below understates production.
    #[test]
    #[ignore = "benchmark; set VICTORY_D3D12_WARP=1"]
    fn flush_per_title_copy_world_sized() {
        if host().is_none() {
            return;
        }
        const BUFFERS: usize = 4;
        const PAGES: usize = 1024; // 64 MiB each
        const LISTS: usize = 28;
        const COPIES: usize = 3;
        let buffers: Vec<Watched> = (0..BUFFERS).map(|_| Watched::new(PAGES)).collect();
        let shadows: Vec<_> = buffers.iter().map(make_shadow).collect();
        let maps: Vec<_> = shadows.iter().map(|s| Mapped::new(s)).collect();
        let scratch = Watched::new(4);
        let run = |overlap_only: bool, mark: u8| {
            let mut batch = Batch::default();
            let started = std::time::Instant::now();
            for list in 0..LISTS {
                for (b, shadow) in shadows.iter().enumerate() {
                    let from = (list * 37 + b * 11) % (PAGES - 16);
                    title_write(buffers[b].at + from * PAGE, mark + b as u8);
                    batch.add(shadow, buffers[b].at, PAGES * PAGE);
                }
                for copy in 0..COPIES {
                    let dst = scratch.at + ((list * COPIES + copy) % 4) * PAGE;
                    title_write(dst, mark);
                    if overlap_only {
                        batch.title_written_over(dst, dst + PAGE);
                    } else {
                        batch.title_written();
                    }
                }
            }
            batch.title_written();
            started.elapsed()
        };
        let _ = run(false, 1); // warm up
        let whole = run(false, 2);
        for (b, map) in maps.iter().enumerate() {
            for list in 0..LISTS {
                let from = (list * 37 + b * 11) % (PAGES - 16);
                assert_eq!(map.byte(from * PAGE), 2 + b as u8);
            }
        }
        let over = run(true, 3);
        for (b, map) in maps.iter().enumerate() {
            for list in 0..LISTS {
                let from = (list * 37 + b * 11) % (PAGES - 16);
                assert_eq!(map.byte(from * PAGE), 3 + b as u8, "the last run's bytes");
            }
        }
        println!(
            "{LISTS} lists, {BUFFERS} buffers of {} MiB, {COPIES} title copies per list: \
             full flush per copy {whole:?}, overlap only {over:?} ({:.1}x)",
            PAGES as f64 * PAGE as f64 / 1048576.0,
            whole.as_secs_f64() / over.as_secs_f64(),
        );
    }
}
