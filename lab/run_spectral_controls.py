"""Matched 3M-parameter controls; train reference for a fixed 1000 updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import wsp_arch, wsp_train, wsp_spectral

data=wsp_train.char_dataset(Path("lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8"))
arch=Path("programs/arch/nanowsp_lm.wsparch")
def init():
    return wsp_arch.load_arch_file(arch,overrides=dict(wsp_spectral.ARCH,vocab=len(data.chars)),seed=1337)
for arm in ("random","trained"):
    path=Path(f"models/nanowsp_lm_v6_{arm}/model.safetensors")
    path.parent.mkdir(parents=True,exist_ok=True)
    m=init()
    if arm=="random":
        m.save(path,metadata={"provenance":"random_initialization","backward_calls":"0"})
    else:
        cut=int(len(data.train)*.9)
        split=wsp_train.CharDataset(data.chars,data.train[:cut],data.train[cut:])
        cfg=wsp_train.TrainConfig(device="cpu",dtype="float32",batch_size=8,
            gradient_accumulation_steps=1,max_iters=1000,eval_interval=250,
            eval_iters=10,log_interval=100,always_save_checkpoint=True,
            learning_rate=.001,warmup_iters=50,lr_decay_iters=1000,min_lr=.0001)
        wsp_train.train(m,split,cfg,path)
