from dkm.benchmarks import HpatchesDenseBenchmark

if __name__ == "__main__":
    from dkm import DKMv3_outdoor
    benchmark = HpatchesDenseBenchmark("data/hpatches")
    model = DKMv3_outdoor(device = "cuda")
    model.upsample_preds = False
    model.symmetric = False
    model.h_resized = 660
    model.w_resized = 880
    model.upsample_res = (864, 1152)
    benchmark.benchmark(model)
    