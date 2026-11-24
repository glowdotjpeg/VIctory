use super::{HwStage, Lower, NativeProgram, Parts, SigElement, StageIo, lower};
use crate::{io, ir, native::MeshIo, resource, structure};

fn mesh_words(counts: u32) -> Vec<u32> {
    vec![
        0xBEFE_04C1, // s_mov_b64 exec, -1
        0xD765_0000,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD766_0000,
        0x0002_00C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x9381_FF03,
        0x0004_0018, // s_bfe_u32 s1, s3, {24, 4}
        0xD746_0000,
        0x0401_0C01, // v_lshl_add_u32 v0, s1, 6, v0
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        0xBF88_0008, // s_cbranch_execz to the join
        0xBEFC_03FF,
        counts,      // s_mov_b32 m0, counts
        0xBF90_0009, // s_sendmsg GS_ALLOC_REQ
        0x7E04_7081, // v_bfrev_b32 v2, 1
        0xF800_08C0,
        0, // exp pos0, off, off, off, off done
        0xF800_0941,
        2,           // exp prim, v2 (culled), off, off, off done
        0xBEFE_04C1, // join: s_mov_b64 exec, -1
        0x7E1C_02FF,
        (-1.0_f32).to_bits(), // v_mov_b32 v14, -1.0
        0x7E1E_02FF,
        3.0_f32.to_bits(), // v_mov_b32 v15, 3.0
        0x7D84_0081,       // v_cmp_eq_u32 vcc, 1, v0
        0xD501_000A,
        0x01AA_1F0E, // v_cndmask_b32 v10, v14, v15, vcc
        0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
        0xD501_000B,
        0x01AA_1F0E, // v_cndmask_b32 v11, v14, v15, vcc
        0x7E18_0280, // v_mov_b32 v12, 0
        0x7E1A_02F2, // v_mov_b32 v13, 1.0
        0x7E12_02FF,
        (2 << 20) | (1 << 10), // v_mov_b32 v9, packed (0, 1, 2)
        0xF800_08CF,
        0x0D0C_0B0A, // exp pos0, v10, v11, v12, v13 done
        0xF800_0941,
        9,           // exp prim, v9, off, off, off done
        0xBF81_0000, // s_endpgm
    ]
}

fn lower_mesh(words: &[u32], planned: bool, unavailable: bool) -> Vec<u8> {
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: 128,
            max_vertices: 128,
            max_primitives: 128,
            topology: 2,
            lds_bytes: 0,
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
        outputs: vec![SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }],
        prims: vec![],
        mesh_stride: 0,
    };
    let bad = std::collections::HashMap::new();
    let candidate = Lower::new(&parts, &bad, true);
    assert_eq!(candidate.mesh_counts_plan.is_some(), planned);
    let lowered = lower(&parts).unwrap().dxil;
    let fallback = matches!(candidate.run(), Err(super::Stop::MeshCounts));
    assert_eq!(fallback, unavailable);
    if fallback || !planned {
        let fallback = Lower::new(&parts, &bad, false).run().ok().unwrap().dxil;
        assert_eq!(lowered, fallback);
    }
    lowered
}

fn validate(dxil: &mut [u8]) {
    assert!(dxil.starts_with(b"DXBC"));
    #[cfg(windows)]
    {
        use super::super::validator::Validator;
        if let Ok(mut validator) = Validator::load(Validator::default_path().as_deref()) {
            validator.validate(dxil).unwrap();
        }
    }
}

#[test]
fn mesh_counts_require_thread_zero_and_no_early_real_exports() {
    for counts in [0, 0x1001, 0x1003, 0x00FF_FFFF] {
        let mut dxil = lower_mesh(&mesh_words(counts), true, false);
        validate(&mut dxil);
    }
    for (before, after) in [(0x7E04_7081, 0x7E04_0280), (0x7DA4_0080, 0x7DA4_0081)] {
        let mut words = mesh_words(0x1003);
        *words.iter_mut().find(|word| **word == before).unwrap() = after;
        let mut dxil = lower_mesh(&words, false, false);
        validate(&mut dxil);
    }
}

#[test]
fn mesh_counts_fall_back_when_m0_is_unavailable() {
    let mut words = mesh_words(0x1003);
    let mov = words.iter().position(|word| *word == 0xBEFC_03FF).unwrap();
    words[mov] = 0xBEFC_0368; // s_mov_b32 m0, s104
    words[mov + 1] = 0xBF80_0000; // s_nop 0
    let mut dxil = lower_mesh(&words, true, true);
    validate(&mut dxil);
}

#[cfg(windows)]
#[test]
fn mesh_counts_rasterize_only_declared_vertices_on_warp() {
    use super::super::{
        HostShader,
        bitcode::{Attr, Term},
        host::{Draw, Host},
        shader::Shader,
    };
    let Ok(host) = Host::new(true) else {
        eprintln!("WARP does not expose SM 6.6");
        return;
    };
    let host_shader = |stage, mut dxil: Vec<u8>| {
        validate(&mut dxil);
        HostShader {
            stage,
            dxil,
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
        }
    };
    let mut pixel = Shader::new(true);
    pixel.outputs.push(SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    });
    let zero = pixel.m.i32(0);
    for (channel, value) in [1.0_f32, 0.0, 0.0, 1.0].into_iter().enumerate() {
        let channel = pixel.m.i8(channel as u8);
        let value = pixel.m.f32(value.to_bits());
        pixel.op(
            pixel.entry,
            "storeOutput.f32",
            Attr::NoUnwind,
            pixel.t.void,
            5,
            &[zero, zero, channel, value],
        );
    }
    pixel.m.terminate(pixel.entry, Term::Ret);
    let pixel = host_shader(HwStage::Ps, pixel.finish());
    let draw = Draw {
        roots: vec![],
        textures: &[],
        samplers: &[],
        vertices: &[],
        stride: 0,
        size: (8, 8),
        mesh_groups: 1,
    };
    for (counts, visible) in [
        (0, false),
        (0x1000, false),
        (0x1001, false),
        (0x1003, true),
        (0x00FF_FFFF, true),
    ] {
        let ms = host_shader(HwStage::Ms, lower_mesh(&mesh_words(counts), true, false));
        let (root, pso) = host.mesh_pipeline(&ms, &pixel).unwrap();
        let pixels = host.draw(&root, &pso, &draw).unwrap();
        assert_eq!(pixels.iter().any(|p| p[0] > 0.5), visible, "m0={counts:#x}");
    }
}
