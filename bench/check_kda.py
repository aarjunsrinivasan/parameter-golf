"""Gate for GDN_MIXER=kda.

1. Op level: chunk_kda with a per-channel gate that is constant across K must
   reproduce chunk_gated_delta_rule with the equivalent per-head gate.
2. Layer level: GDNLayer in kda mode runs forward + backward at the training
   shape (head_dim 42, 32k tokens) with finite outputs and grads.
"""
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda

torch.manual_seed(0)
dev = "cuda"
B, T, H, K, V = 2, 2048, 8, 42, 84

q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16) * 2.0
a_head = torch.randn(B, T, H, device=dev, dtype=torch.float32)
A_log = torch.empty(H, device=dev).uniform_(1, 16).log()
dt_head = torch.randn(H, device=dev) * 0.5 - 3.0

g_head = -A_log.exp() * F.softplus(a_head + dt_head)
o_gdn, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g_head, beta=beta, use_qk_l2norm_in_kernel=True)

a_chan = a_head[..., None].expand(B, T, H, K).contiguous()
dt_chan = dt_head.repeat_interleave(K)
o_kda, _ = chunk_kda(
    q=q, k=k, v=v, g=a_chan, beta=beta, A_log=A_log, dt_bias=dt_chan,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
)
err = (o_kda.float() - o_gdn.float()).abs()
rel = err.max().item() / o_gdn.float().abs().max().item()
print(f"[op] kda(constant-over-K) vs gdn: max_abs={err.max().item():.4e} mean_abs={err.mean().item():.4e} rel={rel:.4e}")
ok_op = rel < 2e-2

# Layer level, imported from the record script with GDN_MIXER=kda
os.environ["GDN_MIXER"] = "kda"
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("tg", path)
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)
layer = tg.GDNLayer(448, 8, 2.0, 4, 0.75).to(dev)
n = sum(p.numel() for p in layer.parameters())
print(f"[layer] head_dim={layer.head_dim} head_v_dim={layer.head_v_dim} params={n} dt_bias={tuple(layer.dt_bias.shape)}")
x = torch.randn(1, 32768, 448, device=dev, requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = layer(x)
y.float().square().mean().backward()
finite = torch.isfinite(y).all().item() and all(
    p.grad is None or torch.isfinite(p.grad).all().item() for p in layer.parameters()
)
no_grad = [nm for nm, p in layer.named_parameters() if p.grad is None]
print(f"[layer] out={tuple(y.shape)} finite={finite} params_without_grad={no_grad}")
# w_out is zero-initialised, so only it (and dt/A_log via chain) get grads at step 0 -- report, don't fail
ok = ok_op and finite
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
