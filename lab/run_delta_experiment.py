"""V7: compile donor deltas and evaluate on untouched final corpus slice."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from safetensors.torch import load_file,save_file
from engine.wsp_delta import factorize,compile_delta

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

root=Path("models")
base_path=root/"nanowsp_lm_v6_random/model.safetensors"
donor_path=root/"nanowsp_lm_v6_trained/model.safetensors"
base,donor=load_file(str(base_path)),load_file(str(donor_path))
factors=factorize(base,donor)
arms={"full":{}, "r8":{"rank":8}, "r16":{"rank":16}, "r32":{"rank":32},
      "r64":{"rank":64}, "placebo32":{"rank":32,"placebo":True},
      "attention":{"groups":{"attention"}}, "mlp":{"groups":{"mlp"}},
      "embedding_norm":{"groups":{"embedding_norm"}},
      "without_attention":{"groups":{"mlp","embedding_norm"}},
      "without_mlp":{"groups":{"attention","embedding_norm"}},
      "without_embedding_norm":{"groups":{"attention","mlp"}}}
report={"provenance":"donor_assisted","source_sha256":sha(base_path),
        "donor_sha256":sha(donor_path),"arms":{}}
for arm,opts in arms.items():
    state,stats=compile_delta(base,factors,**opts)
    if arm=="full":
        assert all(__import__("torch").allclose(state[k],donor[k],atol=1e-7,rtol=1e-6) for k in donor)
    path=root/f"nanowsp_lm_v7_{arm}/model.safetensors"
    path.parent.mkdir(parents=True,exist_ok=True)
    save_file(state,str(path),metadata={"provenance":"donor_assisted_delta",
        "source_sha256":report["source_sha256"],"donor_sha256":report["donor_sha256"],
        "arm":arm,"surgery_backward_calls":"0","donor_backward_calls":"1000"})
    energy=sum(t["energy"] for t in stats.values())
    total=sum(t["total_energy"] for t in stats.values())
    report["arms"][arm]={"sha256":sha(path),"energy_fraction":energy/total,
        "factor_values":sum(t["factor_values"] for t in stats.values()),
        "dense_values":sum(t["dense_values"] for t in stats.values()),"tensors":stats}
    print(arm,report["arms"][arm]["energy_fraction"],flush=True)
Path("lab/runs/nanowsp_lm_v7_delta_manifest.json").write_text(json.dumps(report,indent=2))
