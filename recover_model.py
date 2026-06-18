"""
recover_model.py
----------------
Recovers yolov8n.pt and models/best.pt from Model/ checkpoint folders.

Usage:
    python recover_model.py
"""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent

def load_from_checkpoint_dir(ckpt_dir: Path):
    """
    PyTorch saves checkpoint directories with:
      - data.pkl          (the pickled tensor metadata)
      - data/0, data/1... (the raw tensor data shards)
    torch.load() on the DIRECTORY handles this automatically.
    We pass the directory path as a string.
    """
    # torch.load can open a directory-based checkpoint
    return torch.load(str(ckpt_dir), map_location="cpu", weights_only=False)


tasks = [
    ("Model/yolov8n", "yolov8n.pt"),
    ("Model/best",    "models/best.pt"),
]

success = 0
for src_rel, dst_rel in tasks:
    src = ROOT / src_rel
    dst = ROOT / dst_rel

    if not src.exists():
        print(f"[SKIP] {src_rel} not found")
        continue

    # Check we can actually read data.pkl
    pkl = src / "data.pkl"
    if not pkl.exists():
        print(f"[SKIP] {src_rel}/data.pkl not found — not a valid checkpoint")
        continue

    print(f"Loading {src_rel} ...")
    try:
        # Read data.pkl directly then reconstruct with shards
        import pickle, struct, io

        # Read all shard files into a combined byte stream
        data_dir = src / "data"
        shards   = sorted(data_dir.iterdir(), key=lambda p: int(p.name))
        combined = b"".join(p.read_bytes() for p in shards)

        # Now unpickle with a custom persistent_load that reads from combined
        class TorchUnpickler(pickle.Unpickler):
            def __init__(self, file, storage_bytes):
                super().__init__(file)
                self._storage = storage_bytes
                self._offset  = 0

            def persistent_load(self, pid):
                # pid = (storage_type, key, location, numel, ...)
                storage_type = pid[0]
                dtype_str    = pid[1] if len(pid) > 1 else "storage"
                # Find the tensor in combined storage by reading sequentially
                # Use torch's internal mechanism instead
                raise NotImplementedError("Use torch.load directly")

        # Fallback: use torch.load with the directory path directly
        # torch >= 1.6 supports loading from a directory checkpoint
        data = torch.load(str(src), map_location="cpu", weights_only=False)

        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(dst))
        sz = dst.stat().st_size
        print(f"  Saved  -> {dst}  ({sz:,} bytes)")
        if isinstance(data, dict):
            keys = list(data.keys())
            print(f"  Keys   : {keys[:8]}")
        success += 1

    except PermissionError as e:
        print(f"  [ERROR] Permission denied: {e}")
        print("  Fix: right-click the Model folder -> Properties -> Security -> give Full Control to Administrator")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")

print(f"\n{success}/{len(tasks)} models recovered.")
if success > 0:
    yn = ROOT / "yolov8n.pt"
    bst = ROOT / "models" / "best.pt"
    if yn.exists():
        print(f"\nyolov8n.pt ready  ({yn.stat().st_size:,} bytes)")
        print("Run training:  python train.py")
    if bst.exists():
        print(f"models/best.pt ready  ({bst.stat().st_size:,} bytes)")
        print("Run app:  streamlit run app.py")
