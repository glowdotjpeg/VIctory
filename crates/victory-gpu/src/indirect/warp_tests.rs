use super::*;
use windows::core::Interface;

const PRODUCER: &str = r#"
#define RS "UAV(u0), UAV(u1)"
RWByteAddressBuffer padded : register(u0);
RWByteAddressBuffer tight : register(u1);
[RootSignature(RS)] [numthreads(1, 1, 1)]
void main() {
    padded.Store2(0, uint2(0, 2));
    padded.Store4(8, uint4(1, 1, 1, 111));
    padded.Store4(24, uint4(0xffffffffu, 0xffffffffu, 1, 222));
    padded.Store4(40, uint4(1024, 1024, 1, 333));
    padded.Store3(56, uint3(1024, 1024, 1));
    tight.Store3(0, uint3(2, 1, 1));
}
"#;

const WORKER: &str = r#"
#define RS "UAV(u0), UAV(u1)"
RWByteAddressBuffer result : register(u0);
RWByteAddressBuffer budget : register(u1);
[RootSignature(RS)] [numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint previous;
    result.InterlockedAdd(0, 1, previous);
    if (id.x == 0) result.Store(4, budget.Load(0));
}
"#;

#[test]
#[ignore = "requires WARP and D3D12 Graphics Tools"]
fn gpu_arguments_survive_repeated_counted_and_uncounted_executes() {
    const CHILD: &str = "VICTORY_TEST_INDIRECT_WARP_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "indirect::warp_tests::gpu_arguments_survive_repeated_counted_and_uncounted_executes",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_D3D12_WARP", "1")
            .env_remove("VICTORY_D3D12_DEBUG")
            .env_remove("VICTORY_D3D12_DRED")
            .env_remove("VICTORY_GPU_TIMING")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}\n{stderr}"
        );
        return;
    }

    assert!(std::env::var_os("VICTORY_D3D12_WARP").is_some());
    // Enable validation directly so Host::execute leaves its messages in the queue.
    // SAFETY: this child has not created a D3D12 device yet.
    unsafe {
        let mut debug: Option<ID3D12Debug1> = None;
        D3D12GetDebugInterface(&mut debug).expect("D3D12 Graphics Tools");
        let debug = debug.unwrap();
        debug.EnableDebugLayer();
        debug.SetEnableGPUBasedValidation(true);
    }
    let host = host().expect("WARP host device");
    let queue: ID3D12InfoQueue = host.device.cast().unwrap();
    let producer = make_compute(&host.device, PRODUCER, s!("produce_indirect")).unwrap();
    let worker = make_compute(&host.device, WORKER, s!("consume_indirect")).unwrap();
    let signature = crate::draw::host_signature(2).unwrap();
    let list = host.new_list().unwrap();
    let expected = [
        0u32,
        2,
        1,
        1,
        1,
        111,
        u32::MAX,
        u32::MAX,
        1,
        222,
        1024,
        1024,
        1,
        333,
        1024,
        1024,
        1,
    ];
    let (padded, padded_data) = host.shared_buffer(size_of_val(&expected) as u64).unwrap();
    let (tight, tight_data) = host.shared_buffer(12).unwrap();
    let (wide, _) = host.shared_buffer(6000 * 12).unwrap();
    let (result, result_data) = host.shared_buffer(8).unwrap();
    let budget = list.budget().unwrap();

    for buffer in [&padded, &tight, &result] {
        list.buffer_state(buffer, BUFFER_WRITE);
    }
    // SAFETY: the list and mappings are live, and all buffer ranges fit.
    unsafe {
        result_data.cast::<[u32; 2]>().write([0; 2]);
        list.list.SetComputeRootSignature(&producer.root);
        list.list.SetPipelineState(&producer.pso);
        list.list
            .SetComputeRootUnorderedAccessView(0, padded.GetGPUVirtualAddress());
        list.list
            .SetComputeRootUnorderedAccessView(1, tight.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
    }

    // Two counted calls use the same GPU-written arguments. A zero count skips
    // them, and an uncounted call uses a separate, exactly 12-byte resource.
    for (args, counter, spec) in [
        ((&padded, 8), Some((&padded, 4)), (2, 16, 4)),
        ((&padded, 8), Some((&padded, 0)), (2, 16, 4)),
        // Grow scratch while the earlier calls still refer to the old buffer.
        ((&wide, 0), Some((&padded, 0)), (2, 12, 6000)),
        ((&tight, 0), None, (2, 12, 1)),
        ((&padded, 8), Some((&padded, 4)), (2, 16, 4)),
    ] {
        let bounded = bound(&list, args, counter, spec).unwrap();
        // SAFETY: the worker pipeline and buffers live through execution.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list
                .SetComputeRootUnorderedAccessView(0, result.GetGPUVirtualAddress());
            list.list.SetComputeRootUnorderedAccessView(1, budget);
        }
        bounded.execute(&list, &signature);
    }
    list.close();
    host.execute(&[&list], "test GPU-written indirect scratch");

    // SAFETY: the fence completed, and the mappings hold the stated arrays.
    unsafe {
        assert_eq!(padded_data.cast::<[u32; 17]>().read(), expected);
        assert_eq!(tight_data.cast::<[u32; 3]>().read(), [2, 1, 1]);
        assert_eq!(result_data.cast::<[u32; 2]>().read(), [4, 4 * 64]);
    }
    assert_no_errors(&queue);
}

fn assert_no_errors(queue: &ID3D12InfoQueue) {
    let mut errors = Vec::new();
    // SAFETY: each message is read into aligned storage sized by the debug queue.
    unsafe {
        for index in 0..queue.GetNumStoredMessages() {
            let mut size = 0;
            queue.GetMessage(index, None, &mut size).unwrap();
            let mut storage = vec![0u64; size.div_ceil(8)];
            let message = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
            queue.GetMessage(index, Some(message), &mut size).unwrap();
            let message = &*message;
            if message.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                let description =
                    std::slice::from_raw_parts(message.pDescription, message.DescriptionByteLength);
                errors.push(
                    String::from_utf8_lossy(description)
                        .trim_end_matches('\0')
                        .to_owned(),
                );
            }
        }
    }
    assert!(errors.is_empty(), "D3D12 validation errors: {errors:#?}");
}
