use super::{Lower, Parts};
use crate::emit::host::{Dispatch, Host, Root, View};
use crate::emit::{HostShader, validator::Validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const WAIT_LGKM: u32 = 0xBF8C_C07F;

fn fixture() -> Vec<u32> {
    vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0x2C08_0085, // v_lshrrev_b32 v4, 5, v0 (1 from lanes 32..63)
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
        0xD838_0100,
        0x0001_0101, // ds_write2_b32 v1, v1, v1 offset1:1
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
        WAIT_LGKM,
        0xD828_0000,
        0x0000_0001, // ds_or_b32 v1, v0
        0xD828_0004,
        0x0000_0401, // ds_or_b32 v1, v4 offset:4
        WAIT_LGKM,
        0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
        0xBF88_0005, // s_cbranch_execz to s_endpgm
        0xD8DC_0100,
        0x0200_0001, // ds_read2_b32 v[2:3], v1 offset1:1
        WAIT_LGKM,
        0xE074_0000,
        0x8001_0200, // buffer_store_dwordx2 v[2:3], off, s[4:7], 0
        0xBF81_0000, // s_endpgm
    ]
}

fn lower_fixture(words: &[u32], threads: u32, planned: bool, fixed_wave: bool) -> HostShader {
    lower_case(words, threads, planned, false, fixed_wave)
}

fn lower_case(words: &[u32], threads: u32, planned: bool, counted: bool, fixed_wave: bool) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (1 << 15) | (1 << 1),
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let parts = Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    };
    let bad = std::collections::HashMap::new();
    let mut lower = Lower::new(&parts, &bad, true);
    assert_eq!(lower.lds_reduction.is_some(), planned);
    assert_eq!(lower.counted_reduction.is_some(), counted);
    if let Some([first, last]) = lower.lds_reduction {
        assert_eq!(
            [first, last].map(|i| p.blocks[0].insts[i].machine.offset),
            [11 * 4, 13 * 4]
        );
        assert_eq!(lower.s.wave_size, Some(64));
    }
    if !fixed_wave {
        // The fixture uses thread IDs for both contributions and branch
        // predicates. WARP can test its barriers with its own wave size.
        lower.s.wave_size = None;
    }
    let lowered = lower.run().ok().unwrap();
    assert!(!lowered.wave_ops);
    let mut dxil = lowered.dxil;
    if let Ok(mut validator) = Validator::load(Validator::default_path().as_deref()) {
        validator.validate(&mut dxil).unwrap();
    }
    HostShader {
        stage: HwStage::Cs,
        dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: vec![],
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    }
}

fn counted_fixture() -> Vec<u32> {
    let base = fixture();
    let mut words = base[..9].to_vec();
    words.extend([
        0xD838_0302, 0x0001_0101, // clear the two counters too
        base[9], WAIT_LGKM,
        0xBF82_0000, // jump to a separate reduction block
        0xD828_0000, 0x0000_0001, // OR mask
        0xD800_0008, 0x0000_0001, // add to counter at 8
        0xD800_000C, 0x0000_0401, // add to counter at 12
        0xD828_0004, 0x0000_0401, // OR second mask
        base[9], WAIT_LGKM,
    ]);
    words.extend_from_slice(&base[16..]);
    words
}

#[test]
fn counted_reduction_across_blocks_uses_both_native_waits() {
    let words = counted_fixture();
    lower_case(&words, 64, false, true, true);
    lower_case(&words, 128, false, false, true);
    for (index, word) in words.iter().enumerate().filter(|(_, word)| **word == WAIT_LGKM).take(2) {
        let mut missing_wait = words.clone();
        missing_wait[index] = 0xBF8C_C17F;
        lower_case(&missing_wait, 64, false, false, true);
        assert_eq!(*word, WAIT_LGKM);
    }
    let Ok(host) = Host::new(true) else {
        eprintln!("WARP does not expose SM 6.6");
        return;
    };
    let shader = lower_case(&words, 64, false, true, false);
    let _one = crate::emit::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let output = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords() as usize]))],
            views: &[View::RawUav(2)],
            samplers: &[],
            groups: [1, 1, 1],
        },
    ).unwrap();
    assert_eq!(output, vec![vec![63, 1]]);
}

#[test]
#[ignore = "requires the locally captured mask producer"]
fn captured_mask_producer_uses_both_group_barriers() {
    use crate::{Container, FourCC, native_programs};
    let path = std::env::var_os("VICTORY_TEST_MASK_PRODUCER").expect("local shader path");
    let input = std::fs::read(path).unwrap();
    let container = Container::parse(&input).unwrap();
    let [native] = native_programs(container.part(FourCC::DXIL).unwrap().data)
        .unwrap().try_into().unwrap();
    let mut p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    crate::sgpr_spill::forward(&mut p);
    let st = structure::structure(&p).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let parts = Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 0,
    };
    let bad = std::collections::HashMap::new();
    let lower = Lower::new(&parts, &bad, true);
    let sites = lower.counted_reduction.expect("both waits must be recognized");
    assert_eq!(sites.map(|(b, i)| p.blocks[b].insts[i].machine.offset), [0xa4, 0x1d98]);
    let mut dxil = lower.run().ok().unwrap().dxil;
    let mut validator = Validator::load(Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut dxil).unwrap();
}

#[test]
fn fences_single_wave_lds_reduction_only_with_both_full_waits() {
    let words = fixture();
    lower_fixture(&words, 64, true, true);
    lower_fixture(&words, 128, false, true);
    for (at, replacement) in [(10, 0xBF8C_3F70), (15, 0xBF8C_C17F)] {
        let mut missing_wait = words.clone();
        missing_wait[at] = replacement;
        lower_fixture(&missing_wait, 64, false, true);
    }
    for (wait, exec) in [(10, 9), (15, 16)] {
        let mut moved_wait = words.clone();
        moved_wait.swap(wait, exec);
        lower_fixture(&moved_wait, 64, false, true);
    }
}

#[test]
fn single_wave_lds_reduction_fences_work_with_warp_wave_size() {
    let Ok(host) = Host::new(true) else {
        eprintln!("WARP does not expose SM 6.6");
        return;
    };
    let shader = lower_fixture(&fixture(), 64, true, false);
    let _one = crate::emit::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let output = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                )],
                views: &[View::RawUav(2)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    assert_eq!(output, vec![vec![63, 1]]);
}
