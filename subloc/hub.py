"""More translator models: search Hugging Face for GGUF models, pick the right file for this PC, download it checked.

Only when the user searches (Settings or Setup → Find more models, `subloc models --search`): Subloc asks Hugging
Face's public API, nothing else is sent. A download is checked against the SHA-256 Hugging Face publishes for the
file (its LFS id), like Subloc's own models. Gated models (they need a Hugging Face account) and models split into
several files are left out. These models aren't tested with Subloc: quality, in Arabic especially, can vary.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from .catalog import TranslatorModel
from .config import models_dir
from .download import download
from .errors import SublocError
from .hardware import Hardware

API = "https://huggingface.co/api/models"
PAGE = "https://huggingface.co/{repo}"
# Quantizations in the order Subloc prefers them: 4-bit K-quants keep nearly all the quality at a quarter of the
# size; bigger ones are only a little better and much slower, smaller ones lose quality quickly.
PREFERRED = ["Q4_K_M", "Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q5_K_M", "Q4_0", "Q5_K_S", "Q5_K_XL", "Q6_K",
             "Q3_K_M", "Q3_K_XL", "Q8_0"]  # fmt: skip
QUANT = re.compile(r"(?:UD-)?(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_[A-Z]+)?)$", re.IGNORECASE)
SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


@dataclass(frozen=True)
class Repo:
    id: str  # "owner/name"
    downloads: int
    likes: int
    updated: str  # ISO date
    licence: str  # "" when the repository doesn't say

    @property
    def page(self) -> str:
        return PAGE.format(repo=self.id)


@dataclass(frozen=True)
class File:
    repo: str
    path: str  # inside the repository
    size: int  # bytes
    sha256: str

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def quant(self) -> str:
        m = QUANT.search(Path(self.path).stem)
        return m.group(1).upper() if m else ""

    def as_model(self) -> TranslatorModel:
        """As a catalog entry (what Setup and the download code already handle), saved under models/llm/hub."""
        local = f"hub/{self.repo.replace('/', '--')}/{self.name}"
        label = f"{self.repo.split('/')[-1].removesuffix('-GGUF').removesuffix('-gguf')} {self.quant}".strip()
        return TranslatorModel(f"hf:{self.repo}/{self.path}", label, self.repo, self.path, self.size / 1e9,
                               self.sha256, need_gb(self.size), "from Hugging Face", local)  # fmt: skip


def _get(url: str, params: dict | None = None, timeout: float = 15.0):
    try:
        r = httpx.get(url, params=params, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as e:
        raise SublocError("Couldn't reach Hugging Face. Check the internet connection.", str(e)) from e
    if r.status_code == 404:
        raise SublocError("That model isn't on Hugging Face (or is private).")
    if r.status_code != 200:
        raise SublocError(f"Hugging Face answered {r.status_code}.", r.text[:300])
    try:
        return r.json()
    except ValueError as e:
        raise SublocError("Hugging Face sent an answer Subloc couldn't read.", r.text[:300]) from e


def parse_repos(data: list) -> list[Repo]:
    out = []
    for m in data or []:
        if not isinstance(m, dict) or m.get("gated") not in (None, False) or m.get("private") or m.get("disabled"):
            continue  # gated models need a Hugging Face account to download
        tags = [str(t) for t in m.get("tags") or []]
        if any(t in ("sentence-similarity", "feature-extraction") for t in tags) or m.get("pipeline_tag") in (
            "feature-extraction", "sentence-similarity", "text-to-speech", "automatic-speech-recognition"
        ):  # fmt: skip
            continue  # embeddings and speech models can't translate
        licence = next((t.removeprefix("license:") for t in tags if t.startswith("license:")), "")
        out.append(Repo(str(m.get("id") or ""), int(m.get("downloads") or 0), int(m.get("likes") or 0),
                        str(m.get("lastModified") or m.get("createdAt") or "")[:10], licence))  # fmt: skip
    return [r for r in out if r.id]


def search(query: str = "", limit: int = 40) -> list[Repo]:
    """GGUF models on Hugging Face matching `query` (all of them when empty), most downloaded first."""
    params: list[tuple[str, str | int]] = [("filter", "gguf"), ("sort", "downloads"), ("direction", -1),
                                           ("limit", limit)]  # fmt: skip
    if query.strip():
        params.append(("search", query.strip()))
    for field in ("gated", "downloads", "likes", "lastModified", "tags", "pipeline_tag"):
        params.append(("expand[]", field))
    return parse_repos(_get(API, params))  # type: ignore[arg-type]


def parse_files(repo: str, tree: list) -> list[File]:
    """The model files of a repository that Subloc can run: one-file GGUFs, not the vision/speculative helpers."""
    out = []
    for e in tree or []:
        if not isinstance(e, dict) or e.get("type") != "file":
            continue
        path = str(e.get("path") or "")
        name = Path(path).name.lower()
        if not name.endswith(".gguf") or SPLIT.search(name) or name.startswith(("mmproj", "mtp-")):
            continue
        if path.lower().startswith("mtp/"):
            continue
        sha = str((e.get("lfs") or {}).get("oid") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            continue  # nothing to check the download against
        out.append(File(repo, path, int((e.get("lfs") or {}).get("size") or e.get("size") or 0), sha))
    return sorted(out, key=lambda f: f.size)


def files(repo: str) -> list[File]:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise SublocError("A Hugging Face model is written as owner/name, e.g. lmstudio-community/gemma-4-E4B-it-GGUF.")
    return parse_files(repo, _get(f"{API}/{repo}/tree/main", {"recursive": "true"}))


# ------------------------------------------------------------------------------------------------ this PC


def need_gb(size: int) -> float:
    """Memory to run a model file: the weights plus room for the conversation (about 15 % and 1 GB)."""
    return size / 1e9 * 1.15 + 1.0


def fit(size: int, hw: Hardware | None) -> str:
    """gpu (fits the NVIDIA card) | partly (part on the card, part on the processor: slower) | cpu (processor or
    built-in graphics: slow) | too_big (more than this PC's memory) | unknown (no hardware check)."""
    if hw is None:
        return "unknown"
    need = need_gb(size)
    if hw.nvidia and hw.vram_gb >= need:
        return "gpu"
    spare_ram = max(0.0, hw.ram_gb - 3.0)  # Windows and Subloc itself
    if hw.nvidia and hw.vram_gb + spare_ram >= need:
        return "partly"
    if spare_ram >= need:
        return "cpu"
    return "too_big"


FIT_TEXT = {
    "gpu": "fits your graphics card",
    "partly": "partly on the graphics card: slower",
    "cpu": "on the processor: slow",
    "too_big": "too big for this PC",
    "unknown": "",
}


def recommend(choices: list[File], hw: Hardware | None) -> File | None:
    """The file to suggest: the most preferred quantization that runs well here, else the most preferred that runs
    at all, else the smallest."""
    if not choices:
        return None
    rank = {q: i for i, q in enumerate(PREFERRED)}
    usable = [f for f in choices if f.quant in rank]
    for ok in (("gpu", "unknown"), ("partly", "cpu")):
        good = [f for f in usable if fit(f.size, hw) in ok]
        if good:
            return min(good, key=lambda f: (rank[f.quant], -f.size))
    return min(choices, key=lambda f: f.size)


# ------------------------------------------------------------------------------------------------ download


def local_path(m: TranslatorModel) -> Path:
    return models_dir() / "llm" / (m.local or m.file)


def fetch(f: File, on_progress: Callable[[int, int], None] = lambda d, t: None,
          cancel: threading.Event | None = None) -> Path:  # fmt: skip
    """Download a model file (resumable), checked against Hugging Face's SHA-256. Returns where it was saved."""
    m = f.as_model()
    path = local_path(m)
    download(m.url, path, m.sha256, on_progress, cancel)
    return path
