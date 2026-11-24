use super::*;
use windows::core::s;

fn texture(
    host: &Host,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: the descriptions and output pointer are valid.
    unsafe {
        host.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut resource,
            )
            .unwrap();
    }
    resource.unwrap()
}

#[test]
fn a_float_depth_mirror_drives_d24_depth_tests_without_changing_stencil() {
    if tests::in_warp_child(
        "host::depth_uav_tests::a_float_depth_mirror_drives_d24_depth_tests_without_changing_stencil",
    ) {
        return;
    }
    let host = host().expect("WARP device");
    assert!(tests::on_warp(host));
    let mirror = texture(
        host,
        DXGI_FORMAT_R32_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let depth = texture(
        host,
        DXGI_FORMAT_D24_UNORM_S8_UINT,
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
    );
    let target = texture(
        host,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    let dsv = host.dsv(&depth, None);
    let rtv = host.rtv(&target, None);

    const SOURCE: &str = r#"
        #define RS "DescriptorTable(SRV(t0))"
        Texture2D<float> input_depth : register(t0);
        RWTexture2D<float> output_depth : register(u0);
        [RootSignature("DescriptorTable(UAV(u0))")]
        [numthreads(4, 4, 1)] void write_depth(uint3 id : SV_DispatchThreadID) {
            output_depth[id.xy] = id.x < 2 ? 0.25 : 0.75;
        }
        [RootSignature(RS)] float4 vs(uint i: SV_VertexID) : SV_Position {
            return float4(i == 2 ? 3 : -1, i == 1 ? -3 : 1, 0.5, 1);
        }
        [RootSignature(RS)] float copy_depth(float4 p : SV_Position) : SV_Depth {
            return input_depth.Load(int3(int2(p.xy), 0));
        }
        [RootSignature(RS)] float4 color() : SV_Target { return float4(1, 0, 0, 1); }
    "#;
    let vs = tests::compile(SOURCE, s!("vs"), s!("vs_5_1"));
    let copy = tests::compile(SOURCE, s!("copy_depth"), s!("ps_5_1"));
    let color = tests::compile(SOURCE, s!("color"), s!("ps_5_1"));
    let write = tests::compile(SOURCE, s!("write_depth"), s!("cs_5_1"));
    let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    };

    let list = host.new_list().unwrap();
    // SAFETY: all resources and descriptors are held through the fence.
    unsafe {
        let write_root: ID3D12RootSignature = host.device.CreateRootSignature(0, &write).unwrap();
        let write_pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&write_root),
                CS: code(&write),
                ..Default::default()
            })
            .unwrap();
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &vs).unwrap();
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_EQUAL,
        };
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: code(&vs),
            PS: code(&copy),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                FrontFace: stencil,
                BackFace: stencil,
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            DSVFormat: DXGI_FORMAT_D24_UNORM_S8_UINT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let copy_pso: ID3D12PipelineState = host.device.CreateGraphicsPipelineState(&desc).unwrap();
        desc.PS = code(&color);
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        desc.DepthStencilState.StencilEnable = true.into();
        desc.DepthStencilState.StencilReadMask = 0xff;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: 15,
            ..Default::default()
        };
        let color_pso: ID3D12PipelineState =
            host.device.CreateGraphicsPipelineState(&desc).unwrap();
        let heap: ID3D12DescriptorHeap = host
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();
        host.device.CreateShaderResourceView(
            &mirror,
            None,
            heap.GetCPUDescriptorHandleForHeapStart(),
        );
        let increment = host
            .device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let mut uav_cpu = heap.GetCPUDescriptorHandleForHeapStart();
        uav_cpu.ptr += increment as usize;
        host.device
            .CreateUnorderedAccessView(&mirror, None::<&ID3D12Resource>, None, uav_cpu);
        let mut uav_gpu = heap.GetGPUDescriptorHandleForHeapStart();
        uav_gpu.ptr += u64::from(increment);
        let l = &list.list;
        l.SetDescriptorHeaps(&[Some(heap.clone())]);
        l.SetComputeRootSignature(&write_root);
        l.SetComputeRootDescriptorTable(0, uav_gpu);
        l.SetPipelineState(&write_pso);
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[transition(
            &mirror,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )]);
        l.ClearDepthStencilView(
            dsv,
            D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
            0.0,
            7,
            None,
        );
        l.ClearRenderTargetView(rtv, &[0.0; 4], None);
        l.SetGraphicsRootSignature(&root);
        l.SetGraphicsRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
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
        l.SetPipelineState(&copy_pso);
        l.OMSetRenderTargets(0, None, false, Some(&dsv));
        l.DrawInstanced(3, 1, 0, 0);
        l.SetPipelineState(&color_pso);
        l.OMSetStencilRef(7);
        l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
        l.DrawInstanced(3, 1, 0, 0);
        l.ResourceBarrier(&[transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        let (readback, footprint) = crate::capture::read_back(&host.device, l, &target).unwrap();
        list.close();
        host.execute(&[&list], "depth UAV mirror proof");
        let mut data = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut data)).unwrap();
        for y in 0..4 {
            let row = data
                .cast::<u8>()
                .add(y * footprint.Footprint.RowPitch as usize);
            assert_eq!(
                std::slice::from_raw_parts(row, 16),
                &[0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255]
            );
        }
        readback.Unmap(0, None);
    }
}
