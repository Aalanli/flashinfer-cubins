
from harness import enumerate_workloads
import torch

lo, hi = torch.cuda.get_device_capability()
arch = f"sm_{lo}{hi}"
for w in enumerate_workloads(arch):
    print(w.name)

    workload = w.from_arch(arch)
    
    cases = workload.get_cases()
    print(cases[0])

    inp = workload.get_inputs(cases[0])
    launch, _ = workload.prepare(inp)

    out = launch()
    ref = workload.get_reference(inp)
    workload.validate(ref, out)

