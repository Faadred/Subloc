"""Updates for the built-in translator's engine, llama.cpp (llama-server), so new kinds of models keep loading.

Only when the user asks (Settings → Translator engine, `subloc engine --update`): Subloc asks GitHub for
llama.cpp's newest *stable* release (its numbered builds are nightly pre-releases; a stable release names the build
it stands for in nightly-tag.txt), downloads the Windows builds Subloc uses (NVIDIA: CUDA 12, the libraries Subloc
already has; others: Vulkan), checked against the SHA-256 GitHub publishes, keeps llama-server and its libraries,
makes sure it starts, then switches to it. The build it replaces is kept for Undo. Builds live in the data folder
(llama/<backend>, version in version.txt); the engine uses the newest build it finds there or bundled with the app.
"""

import hashlib
import re
import shutil
import subprocess
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import proc
from .config import data_dir
from .cuda import LLAMA_BUILD, keep_llama_file
from .download import download
from .errors import SublocError

REPO = "ggml-org/llama.cpp"
API = f"https://api.github.com/repos/{REPO}"
ASSET = {
    "cuda": re.compile(r"^llama-(b\d+)-bin-win-cuda-12\.\d+-x64\.zip$"),  # CUDA 12: the libraries Subloc ships
    "vulkan": re.compile(r"^llama-(b\d+)-bin-win-vulkan-x64\.zip$"),
}
POINTER = "nightly-tag.txt"
VERSION_FILE = "version.txt"


@dataclass(frozen=True)
class Asset:
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Build:
    tag: str  # "b11960"
    release: str  # the stable release's name, e.g. "v0.8.0"
    page: str  # its notes
    assets: dict[str, Asset] = field(default_factory=dict)  # backend → Windows build


def number(tag: str) -> int:
    m = re.fullmatch(r"b(\d+)", tag.strip())
    return int(m.group(1)) if m else 0


def _digest(a: dict) -> str:
    d = str(a.get("digest") or "")
    sha = d.removeprefix("sha256:") if d.startswith("sha256:") else ""
    return sha if re.fullmatch(r"[0-9a-f]{64}", sha) else ""


def parse_assets(release: dict) -> tuple[str, dict[str, Asset]]:
    """(build tag, backend → asset) from a release's JSON; only assets GitHub publishes a SHA-256 for."""
    tag, found = "", {}
    for a in release.get("assets") or []:
        name = str(a.get("name") or "")
        for backend, pattern in ASSET.items():
            m = pattern.match(name)
            sha = _digest(a)
            if m and sha:
                tag = m.group(1)
                found[backend] = Asset(str(a.get("browser_download_url") or ""), int(a.get("size") or 0), sha)
    return tag, found


def _get(url: str, timeout: float):
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"Accept": "application/vnd.github+json"})
    except httpx.HTTPError as e:
        raise SublocError("Couldn't reach GitHub. Check the internet connection.", str(e)) from e
    if r.status_code != 200:
        raise SublocError(f"GitHub answered {r.status_code}.", r.text[:300])
    return r


def latest(timeout: float = 10.0) -> Build | None:
    """llama.cpp's newest stable build with Windows downloads Subloc can use, or None."""
    rel = _get(f"{API}/releases/latest", timeout).json()
    name, page = str(rel.get("tag_name") or ""), str(rel.get("html_url") or "")
    tag, assets = parse_assets(rel)  # a release that carries the builds itself
    if not assets:  # a stable release that points at its build
        pointer = next((a for a in rel.get("assets") or [] if a.get("name") == POINTER), None)
        if pointer is None:
            return None
        r = _get(str(pointer.get("browser_download_url") or ""), timeout)
        if _digest(pointer) and hashlib.sha256(r.content).hexdigest() != _digest(pointer):
            raise SublocError("llama.cpp's release didn't match its checksum; try again later.")
        tag = r.text.strip().split()[0] if r.text.strip() else ""
        if not number(tag):
            return None
        build_tag, assets = parse_assets(_get(f"{API}/releases/tags/{tag}", timeout).json())
        if build_tag != tag:
            return None
    return Build(tag, name or tag, page, assets) if assets else None


# ------------------------------------------------------------------------------------------------ installed builds


def folder(backend: str) -> Path:
    return data_dir() / "llama" / backend


def version_of(d: Path) -> str:
    """The build in folder d: its version.txt, else the one bundled with this Subloc."""
    try:
        tag = (d / VERSION_FILE).read_text(encoding="utf-8").strip()
        return tag if number(tag) else LLAMA_BUILD
    except OSError:
        return LLAMA_BUILD


def current() -> dict[str, str]:
    """Backend → the build the engine uses now."""
    from .engine import _backend_dir, available_backends

    out = {}
    for b in available_backends():
        d = _backend_dir(b)
        if d is not None:
            out[b] = version_of(d)
    return out


def _bundled(backend: str) -> bool:
    from .engine import _vendor

    return (_vendor() / backend / "llama-server.exe").exists()


def previous(backend: str) -> str:
    """The build Undo goes back to ("" when there is none): the one the last update replaced, or the one bundled
    with the app when the update was the first."""
    d = folder(backend).with_name(f"{backend}.previous")
    if (d / "llama-server.exe").exists():
        return version_of(d)
    if (folder(backend) / VERSION_FILE).exists() and _bundled(backend):
        return LLAMA_BUILD
    return ""


def newer(build: Build | None, have: dict[str, str]) -> list[str]:
    """The backends `build` would update (installed here, and older than it)."""
    if build is None:
        return []
    return [b for b, tag in have.items() if b in build.assets and number(build.tag) > number(tag)]


def _check(d: Path, tag: str) -> None:
    """The new llama-server starts and says which build it is."""
    try:
        out = proc.run([str(d / "llama-server.exe"), "--version"], capture_output=True, text=True, timeout=60,
                       cwd=str(d))  # fmt: skip
    except (OSError, subprocess.SubprocessError) as e:
        raise SublocError("The new translator engine didn't start; Subloc keeps the one it has.", str(e)) from e
    text = (out.stdout or "") + (out.stderr or "")
    if str(number(tag)) not in text:
        raise SublocError("The new translator engine didn't start; Subloc keeps the one it has.", text[-600:])


def _swap(backend: str, new: Path) -> None:
    cur, old = folder(backend), folder(backend).with_name(f"{backend}.previous")
    try:
        shutil.rmtree(old, ignore_errors=True)
        if cur.exists():
            cur.rename(old)
        new.rename(cur)
    except OSError as e:
        raise SublocError("The translator engine is in use. Close Subloc's window (or wait for it to finish), then "
                          "try again.", str(e)) from e  # fmt: skip


def update(backend: str, build: Build, on_progress: Callable[[int, int], None] = lambda d, t: None,
           cancel: threading.Event | None = None) -> str:  # fmt: skip
    """Download `build` for `backend` (checked), make sure it starts, switch to it. Returns its tag."""
    a = build.assets.get(backend)
    if a is None:
        raise SublocError(f"llama.cpp {build.tag} has no {backend} build for Windows.")
    archive = data_dir() / "downloads" / Path(a.url).name
    download(a.url, archive, a.sha256, on_progress, cancel)
    new = folder(backend).with_name(f"{backend}.new")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                name = Path(info.filename).name
                if not info.is_dir() and keep_llama_file(name):
                    (new / name).write_bytes(z.read(info))
        if not (new / "llama-server.exe").exists():
            raise SublocError(f"llama.cpp {build.tag}'s download has no llama-server.")
        (new / VERSION_FILE).write_text(build.tag, encoding="utf-8")
        _check(new, build.tag)
        _swap(backend, new)
    finally:
        shutil.rmtree(new, ignore_errors=True)
    archive.unlink(missing_ok=True)
    return build.tag


def undo(backend: str) -> str:
    """Go back to the build the last update replaced. Returns its tag."""
    tag = previous(backend)
    if not tag:
        raise SublocError("There is no earlier translator engine to go back to.")
    old, cur = folder(backend).with_name(f"{backend}.previous"), folder(backend)
    spare = cur.with_name(f"{backend}.undone")
    try:
        shutil.rmtree(spare, ignore_errors=True)
        if cur.exists():
            cur.rename(spare)
        if (old / "llama-server.exe").exists():
            old.rename(cur)  # else: the build bundled with the app is used again
        shutil.rmtree(spare, ignore_errors=True)
    except OSError as e:
        raise SublocError("The translator engine is in use. Close Subloc's window (or wait for it to finish), then "
                          "try again.", str(e)) from e  # fmt: skip
    return tag
