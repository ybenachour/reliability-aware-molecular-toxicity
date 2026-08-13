from pathlib import Path
import pandas as pd
from toxicity_screening.utils import sha256_file

root = Path(__file__).resolve().parents[1]
files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p != root / "data/metadata/source_manifest.csv"]
frame = pd.DataFrame({
    "path": [str(p.relative_to(root)) for p in files],
    "size_bytes": [p.stat().st_size for p in files],
    "sha256": [sha256_file(p) for p in files],
})
output = root / "data/metadata/source_manifest.csv"
frame.to_csv(output, index=False)
print(output)
