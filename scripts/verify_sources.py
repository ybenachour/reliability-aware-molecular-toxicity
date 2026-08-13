from __future__ import annotations

import json
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
config = yaml.safe_load((ROOT / "configs/data_config.yaml").read_text())
results = []
for source_key, spec in config["sources"].items():
    for url_type in ["official_url", "download_url"]:
        url = spec.get(url_type)
        if not url:
            continue
        try:
            response = requests.get(url, timeout=30, stream=True, headers={"User-Agent": "toxicity-screening-source-check/0.1"})
            results.append({"source": source_key, "url_type": url_type, "url": url, "status_code": response.status_code, "ok": response.ok, "content_type": response.headers.get("content-type")})
            response.close()
        except Exception as exc:
            results.append({"source": source_key, "url_type": url_type, "url": url, "status_code": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
path = ROOT / "data/metadata/source_url_check.json"
path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(path)
if not all(row["ok"] for row in results):
    raise SystemExit("One or more configured source URLs failed; inspect the saved report before acquisition.")
