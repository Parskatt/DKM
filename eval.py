from dkm import dkm_base
from dkm.benchmarks import HpatchesHomogBenchmark

model = dkm_base(pretrained=True, version="v11")
homog_benchmark = HpatchesHomogBenchmark("data/hpatches")
homog_benchmark.benchmark_hpatches(model)