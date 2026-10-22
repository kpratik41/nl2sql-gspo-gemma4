"""Fresh network download of original BIRD dev JSON and official train-6601.

HTTP ranges extract only JSON from the official 2024 archive; DBs stay local.
"""
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
TRAIN_REPO = "birdsql/bird23-train-filtered"


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.headers = dict(r.headers)
            self.size = int(r.headers["Content-Length"])

    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos + n if whence == 1 else self.size + n
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else n
        if not n:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"})
        with urllib.request.urlopen(request, timeout=120) as r:
            if r.status != 206:
                raise RuntimeError("Server must support byte ranges; refusing to download databases")
            data = r.read()
        self.pos += len(data)
        return data


def main():
    root = Path("data/raw")
    root.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip(DEV_URL)
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "dev_url": DEV_URL,
                "archive_headers": remote.headers, "files": {}}
    with zipfile.ZipFile(remote) as archive:
        for name in ("dev.json", "dev_tables.json"):
            member = f"dev_20240627/{name}"
            data = archive.read(member)
            (root / name).write_bytes(data)
            manifest["files"][name] = {"source": DEV_URL + "#" + member, "sha256": hashlib.sha256(data).hexdigest()}
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{TRAIN_REPO}", timeout=60) as r:
        revision = json.load(r)["sha"]
    url = f"https://huggingface.co/datasets/{TRAIN_REPO}/resolve/{revision}/data/train-00000-of-00001.jsonl"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    (root / "train-6601.jsonl").write_bytes(data)
    manifest["files"]["train-6601.jsonl"] = {"source": url, "sha256": hashlib.sha256(data).hexdigest()}
    dev = json.loads((root / "dev.json").read_text())
    train = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert len(dev) == 1534, len(dev)
    assert len(train) == 6601, len(train)
    assert len({x["question_id"] for x in dev}) == 1534
    manifest["counts"] = {"dev": len(dev), "train": len(train),
                          "california_schools": sum(x["db_id"] == "california_schools" for x in dev)}
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
