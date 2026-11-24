use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, PCSTR};
use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleFileNameW, GetModuleHandleExW,
};
use windows_sys::Win32::System::Threading::GetCurrentThreadId;

const DEPTH: usize = 24;

struct Removal {
    message: i32,
    thread: u32,
    frame: u64,
    count: usize,
    addresses: [usize; DEPTH],
}

static REMOVAL: OnceLock<Removal> = OnceLock::new();

pub(super) fn register(device: &ID3D12Device) {
    let result = device.cast::<ID3D12InfoQueue1>().and_then(|queue| {
        let mut cookie = 0;
        // SAFETY: the callback uses static storage and the host lives for
        // the process. Direct3D owns the callback registration.
        unsafe {
            queue.RegisterMessageCallback(
                Some(capture),
                D3D12_MESSAGE_CALLBACK_IGNORE_FILTERS,
                null_mut(),
                &mut cookie,
            )
        }
    });
    match result {
        Ok(()) => tracing::info!("capturing the first D3D12 device removal stack"),
        Err(error) => tracing::warn!(%error, "D3D12 removal stack capture unavailable"),
    }
}

unsafe extern "system" fn capture(
    _: D3D12_MESSAGE_CATEGORY,
    _: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    _: PCSTR,
    _: *mut c_void,
) {
    if !matches!(
        id,
        D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_AT_FAULT
            | D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_POSSIBLY_AT_FAULT
            | D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_NOT_AT_FAULT
    ) || REMOVAL.get().is_some()
    {
        return;
    }

    // This callback may run under Direct3D's locks. Save addresses here,
    // and resolve them and log them outside Direct3D.
    let mut addresses = [null_mut(); DEPTH];
    // SAFETY: the output array has room for DEPTH addresses.
    let count =
        unsafe { RtlCaptureStackBackTrace(1, DEPTH as u32, addresses.as_mut_ptr(), null_mut()) };
    let _ = REMOVAL.set(Removal {
        message: id.0,
        // SAFETY: reads the calling thread's identifier.
        thread: unsafe { GetCurrentThreadId() },
        frame: crate::queue::presented(),
        count: usize::from(count),
        addresses: addresses.map(|address| address as usize),
    });
}

pub(super) fn log() {
    let Some(removal) = REMOVAL.get() else { return };
    let stack: Vec<_> = removal.addresses[..removal.count]
        .iter()
        .copied()
        .map(describe)
        .collect();
    tracing::error!(
        message_id = removal.message,
        thread = removal.thread,
        presented_frame = removal.frame,
        ?stack,
        "D3D12 device removal notification stack"
    );
}

fn describe(address: usize) -> String {
    let mut module = null_mut();
    let mut name = [0u16; 260];
    // SAFETY: the lookup interprets the value as an address, and the
    // filename buffer is writable. Loaded graphics modules are stable.
    let length = unsafe {
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address as *const u16,
            &mut module,
        ) == 0
        {
            return format!("{address:#x}");
        }
        GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32)
    };
    let name = String::from_utf16_lossy(&name[..length as usize]);
    let name = name.rsplit('\\').next().unwrap_or_default();
    format!("{name}+{:#x}", address - module as usize)
}
