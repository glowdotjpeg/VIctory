//! 64-bit image atomics (`image_atomic_*` with two data dwords per value, on
//! the console's R32G32 view) run on the host GPU and WARP, on an
//! R32G32_UINT texture UAV through the host's 64-bit typed atomics.
//!
//! The view's size is queried first in the same block, so a handle of the
//! same view with another element type exists before the atomics' own.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};
use windows::Win32::Graphics::Direct3D12::*;

const S_ENDPGM: u32 = 0xBF81_0000;
/// Source operands: an inline constant `n` (0..=64), -1, a literal.
const fn int(n: u8) -> u16 {
    128 + n as u16
}
const MINUS_ONE: u16 = 193;
const LITERAL: u16 = 255;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx8 s[dst:dst+7], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx8(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0`.
const fn v_mov(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// MIMG `op vdata, vaddr, s[srsrc:srsrc+7] dmask:dmask dim:2D unorm`.
const fn mimg_2d(op: u8, vdata: u8, vaddr: u8, srsrc: u8, dmask: u8) -> [u32; 2] {
    [
        0x3C << 26 | (op as u32 & 0x7F) << 18 | 1 << 12 | (dmask as u32 & 0xF) << 8 | 1 << 3,
        vaddr as u32 | (vdata as u32) << 8 | (srsrc as u32 >> 2) << 16,
    ]
}

const IMAGE_GET_RESINFO: u8 = 0x0E;
const IMAGE_ATOMIC_CMPSWAP: u8 = 0x10;
const IMAGE_ATOMIC_ADD: u8 = 0x11;
const IMAGE_ATOMIC_SUB: u8 = 0x12;
const IMAGE_ATOMIC_UMAX: u8 = 0x17;

/// One thread: a size query of the texture UAV at heap index 0, then on
/// texels (x, 0) of it:
/// - 0: add 0xFFFFFFFF, add 1 (the carry into the high dword);
/// - 1: cmpswap 0x22222222_11111111 against 0;
/// - 2: umax 0x00000001_00000005;
/// - 3: add 0x00000001_00000000, sub 1 (the borrow from it).
fn build() -> HostShader {
    let mut w = vec![s_mov_b32(1, 0x80)];
    w.extend(s_load_dwordx8(8, 0));
    // v20: the queried level; v[10:13] its size, unused.
    w.push(v_mov(20, int(0)));
    w.extend(mimg_2d(IMAGE_GET_RESINFO, 10, 20, 8, 0xF));
    // v[0:1]: the texel.
    w.push(v_mov(1, int(0)));
    let add = |w: &mut Vec<u32>, op: u8, x: u8, lo: u16, hi: u16| {
        w.push(v_mov(0, int(x)));
        w.push(v_mov(2, lo));
        w.push(v_mov(3, hi));
        w.extend(mimg_2d(op, 2, 0, 8, 0x3));
    };
    add(&mut w, IMAGE_ATOMIC_ADD, 0, MINUS_ONE, int(0));
    add(&mut w, IMAGE_ATOMIC_ADD, 0, int(1), int(0));
    // cmpswap: data (new, compare), two dwords each.
    w.push(v_mov(0, int(1)));
    w.extend([
        v_mov(4, LITERAL),
        0x1111_1111,
        v_mov(5, LITERAL),
        0x2222_2222,
    ]);
    w.extend([v_mov(6, int(0)), v_mov(7, int(0))]);
    w.extend(mimg_2d(IMAGE_ATOMIC_CMPSWAP, 4, 0, 8, 0xF));
    add(&mut w, IMAGE_ATOMIC_UMAX, 2, int(5), int(1));
    add(&mut w, IMAGE_ATOMIC_ADD, 3, int(0), int(1));
    add(&mut w, IMAGE_ATOMIC_SUB, 3, int(1), int(0));
    w.push(S_ENDPGM);

    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 24,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 1,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // 1 user SGPR: s0
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Whether the device runs 64-bit atomics on typed views in the heap.
fn atomics64(host: &Host) -> bool {
    let mut o9 = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
    let mut o11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
    // SAFETY: out structures of the sizes passed.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS9,
                (&raw mut o9).cast(),
                size_of_val(&o9) as u32,
            )
            .is_ok()
            && host
                .device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS11,
                    (&raw mut o11).cast(),
                    size_of_val(&o11) as u32,
                )
                .is_ok()
            && o9.AtomicInt64OnTypedResourceSupported.as_bool()
            && o11.AtomicInt64OnDescriptorHeapResourceSupported.as_bool()
    }
}

#[test]
fn image_atomics_64_bit_add_sub_cmpswap_umax() {
    let shader = build();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        if !atomics64(&host) {
            eprintln!("{label}: no 64-bit atomics on typed heap views; skipped");
            continue;
        }
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::Uint64TextureUav(4, 1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want = [
            0,
            1, // 0xFFFFFFFF + 1
            0x1111_1111,
            0x2222_2222, // swapped in
            5,
            1, // umax
            0xFFFF_FFFF,
            0, // 1 << 32, minus 1
        ];
        assert_eq!(out[0], want, "{label}: texels (low, high)");
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no device with 64-bit typed atomics; skipped");
    }
}
