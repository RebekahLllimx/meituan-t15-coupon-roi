import json
import os
import pickle
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


def _stable_hash(obj: Any) -> str:
    """Hash params deterministically for caching keys."""
    try:
        payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = repr(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class CacheMeta:
    run_id: str
    run_type: str
    created_at: float
    params_hash: str
    params: Dict[str, Any]
    summary: Dict[str, Any]


class ResultCache:
    """
    Lightweight result cache for Streamlit runs.

    - Stores a JSON meta index + a pickle artifact per run_id
    - Provides list/load/export for previously computed results
    """

    def __init__(self, cache_dir: str = ".cache/results"):
        self.cache_dir = cache_dir
        self.meta_dir = os.path.join(cache_dir, "meta")
        self.artifact_dir = os.path.join(cache_dir, "artifacts")
        _ensure_dir(self.meta_dir)
        _ensure_dir(self.artifact_dir)

    def _paths(self, run_id: str) -> Tuple[str, str]:
        return (
            os.path.join(self.meta_dir, f"{run_id}.json"),
            os.path.join(self.artifact_dir, f"{run_id}.pkl"),
        )

    def save(
        self,
        run_type: str,
        params: Dict[str, Any],
        summary: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> CacheMeta:
        created_at = time.time()
        params_hash = _stable_hash(params)
        run_id = f"{run_type}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(created_at))}-{params_hash}"
        meta = CacheMeta(
            run_id=run_id,
            run_type=run_type,
            created_at=created_at,
            params_hash=params_hash,
            params=params,
            summary=summary,
        )

        meta_path, artifact_path = self._paths(run_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

        with open(artifact_path, "wb") as f:
            pickle.dump(artifacts or {}, f, protocol=pickle.HIGHEST_PROTOCOL)

        return meta

    def list(self, run_type: Optional[str] = None, limit: int = 50) -> List[CacheMeta]:
        metas: List[CacheMeta] = []
        for name in os.listdir(self.meta_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.meta_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                meta = CacheMeta(**raw)
                if run_type and meta.run_type != run_type:
                    continue
                metas.append(meta)
            except Exception:
                continue
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas[: max(1, int(limit))]

    def load(self, run_id: str) -> Tuple[CacheMeta, Dict[str, Any]]:
        meta_path, artifact_path = self._paths(run_id)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = CacheMeta(**json.load(f))
        with open(artifact_path, "rb") as f:
            artifacts = pickle.load(f)
        return meta, artifacts

    def export_summary_json(self, run_id: str) -> str:
        meta, _ = self.load(run_id)
        return json.dumps(asdict(meta), ensure_ascii=False, indent=2)

