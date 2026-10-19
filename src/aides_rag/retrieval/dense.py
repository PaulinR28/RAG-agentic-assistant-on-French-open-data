"""Dense (embedding) search over the chunks, on CPU with sentence-transformers.

Each chunk text is turned into a normalised vector once; the vectors are cached in indexes/
(git-ignored), keyed by model and chunk content. A query is embedded the same way and scored
by cosine similarity (dot product of normalised vectors) against every chunk: 865 chunks is
small enough for exact search, no approximate index needed.

Candidate models (all multilingual, open weights). E5 models expect "query: " / "passage: " prefixes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repo root, so paths work from any working directory
CACHE_DIR = ROOT / "indexes"
LOCAL_MODELS = ROOT / "models"  # optional manual downloads: models/<alias>/ with a REVISION file
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


DEFAULT_DENSE = "bge-m3"  # chosen on dev Recall@1 (0.91 vs 0.74 for e5); see eval/retrieval_dense.json

MODELS = {
    "e5-small": ModelSpec("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ModelSpec("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ModelSpec("BAAI/bge-m3", max_seq_length=1024),
}


def local_dir(alias: str) -> Path | None:
    path = LOCAL_MODELS / alias
    return path if (path / "REVISION").exists() else None


def model_revision(name: str, alias: str | None = None) -> str | None:
    """Commit hash of the local model (for the README: exact versions)."""
    if alias and local_dir(alias):
        return (local_dir(alias) / "REVISION").read_text().strip()
    ref = HF_CACHE / f"models--{name.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


class DenseIndex:
    def __init__(self, alias: str, texts: list[str], cache_dir: Path = CACHE_DIR, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        self.alias = alias
        self.spec = MODELS[alias]
        source = local_dir(alias) or self.spec.name
        self.model = SentenceTransformer(str(source), device="cpu")
        self.model.max_seq_length = self.spec.max_seq_length
        self.revision = model_revision(self.spec.name, alias)

        key = hashlib.sha256("\n".join([self.spec.name, self.spec.passage_prefix, str(self.spec.max_seq_length),
                                        *texts]).encode("utf-8")).hexdigest()[:16]
        path = cache_dir / f"{alias}-{key}.npy"
        meta = path.with_suffix(".json")  # which model revision made the vectors (committed next to them)
        self.from_cache = path.exists()
        if self.from_cache:
            made_by = json.loads(meta.read_text(encoding="utf-8")).get("revision") if meta.exists() else None
            if made_by and self.revision and made_by != self.revision:
                # queries and chunks would be embedded by different weights: scores silently meaningless
                raise RuntimeError(f"{path.name} was built with {self.spec.name}@{made_by}, but the local model is "
                                   f"@{self.revision}. Download the pinned revision (README, Models) or delete "
                                   f"the .npy to re-embed.")
            self.vectors = np.load(path)
        else:
            self.vectors = self.encode([self.spec.passage_prefix + t for t in texts], batch_size)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, self.vectors)
            meta.write_text(json.dumps({"model": self.spec.name, "revision": self.revision}) + "\n",
                            encoding="utf-8", newline="\n")

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        return self.vectors @ self.encode([self.spec.query_prefix + query])[0]
