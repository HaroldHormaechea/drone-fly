"""Extract just the TRAINABLE readout (+ tap_index buffer) from a reservoir checkpoint -> small,
committable model file. The frozen K1 body (177MB) is reconstructed deterministically from
artifacts/pruned/k1 at load time, so only the readout needs committing.

Reload: build K1Reservoir()/K1ReservoirAug(E), then load_state_dict(torch.load(model_readout.pt),
strict=False). Usage: save_readout.py <full_ckpt.pt> <out_readout.pt>"""
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
readout = {k: v for k, v in sd.items() if not k.startswith("body.")}
torch.save(readout, sys.argv[2])
n = sum(v.numel() for v in readout.values())
print(f"saved {len(readout)} tensors ({n/1e3:.1f}k params) -> {sys.argv[2]}")
print("keys:", ", ".join(readout.keys()))
