import json,torch,time
from safetensors import safe_open
from engine import delta_pack as dp
pre='model.layers.0.'
names=[pre+s for s in ('linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight','linear_attn.out_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')]
fb=safe_open('/tmp/probe_cache/base_layer0.safetensors','pt'); fm=safe_open('/tmp/probe_cache/mimo_layer0.safetensors','pt')
meta={"base_repo":"Qwen/Qwen3.5-9B","base_revision":"c202236235762e1c871ad0ccb60c8ee5ba337b9a","base_license":"Apache-2.0","model_repo":"XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B","model_revision":"1f09ddcc1a0ea45751ae0dfe2af20d533168c46f","model_license":"MIT"}
allm={'entries':[]}
for n in names:
    t=time.time()
    base={n:fb.get_tensor(n)}; mimo={n:fm.get_tensor(n)}
    factors,man=dp.build_pack(base,mimo,[n],16,meta)
    patched=dp.apply_pack(base,factors,man,1.0)[n]
    side,v=torch.load(f'/tmp/probe_cache/basis_{n}.pt'); v=v[:,:16]
    d=mimo[n].float()-base[n].float()
    ref=base[n].float()+(d@v@v.T if side=='right' else v@(v.T@d))
    err=(patched.float()-ref).abs().max().item(); rel=(patched.float()-ref).norm().item()/d.norm().item()
    print(n.split('layers.0.')[1],'kept',round(man['tensors'][0]['delta_energy_kept'],4),'max|pack-probe|',f'{err:.2e}','rel to |delta|',f'{rel:.2e}','verify base',dp.verify_tensors(base,man,'base'),round(time.time()-t),'s',flush=True)
    allm['entries']+=man['tensors']
man['tensors']=allm['entries']; json.dump(man,open('/tmp/demo_manifest_layer0.json','w'),indent=1); print('entries',len(man['tensors']))
