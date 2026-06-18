"""
download_weights.py
-------------------
Downloads yolov8n.pt with SSL verification disabled.
Handles corporate proxy / firewall environments.

Usage:
    python download_weights.py
"""

import os
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
DEST = ROOT / "yolov8n.pt"

URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
]


def ssl_ctx():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def try_download(url: str, dest: Path) -> bool:
    print(f"  Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx(), timeout=180) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done*100//total}%  ({done:,}/{total:,} bytes)  ", end="", flush=True)
        print(f"\r  100%  ({done:,} bytes)                ")
        return dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as e:
        print(f"\n  Error: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def main():
    # Already exists and valid?
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"yolov8n.pt already present  ({DEST.stat().st_size:,} bytes).")
        print("Nothing to do. Run:  python train.py")
        return

    print("=" * 55)
    print("  Downloading yolov8n.pt")
    print("=" * 55)

    for url in URLS:
        if try_download(url, DEST):
            print(f"\nSaved to: {DEST}")
            print(f"Size    : {DEST.stat().st_size:,} bytes")
            print("\nNext step:  python train.py")
            return

    # All failed
    print("\n" + "=" * 55)
    print("  Automatic download FAILED")
    print("  (Corporate proxy / firewall detected)")
    print("=" * 55)
    print("\nManual download — do this in your browser:")
    print()
    print("  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt")
    print()
    print(f"Save the file here:")
    print(f"  {DEST}")
    print()
    print("Then run:  python train.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
