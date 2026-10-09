"""Extract the committable model from a reservoir checkpoint: everything EXCEPT the large, deterministic
sparse connectome layer (``body.layer.*``, ~3.86M edges) which is rebuilt byte-identically from
artifacts/pruned/k1 at load time.

IMPORTANT: this KEEPS ``body.input_projection`` (and the small index buffers). That projection is
RANDOMLY initialized and frozen -- NOT reproducible from artifacts -- so the readout, trained against
that specific random projection, only works if it is committed too. (Stripping all ``body.*`` gave a
fresh random projection on reload -> 0% completion.)

Reload: build K1Reservoir()/K1ReservoirAug(E), then load_state_dict(torch.load(model_readout.pt),
strict=False) -- body.layer.* stays as the rebuilt sparse layer; everything else is overridden.
Usage: save_readout.py <full_ckpt.pt> <out_readout.pt>"""
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
keep = {k: v for k, v in sd.items() if not k.startswith("body.layer.")}
torch.save(keep, sys.argv[2])
n = sum(v.numel() for v in keep.values())
print(f"saved {len(keep)} tensors ({n/1e3:.1f}k params) -> {sys.argv[2]}")
print("keys:", ", ".join(keep.keys()))
