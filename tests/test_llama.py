"""Updates for the built-in translator's engine, llama.cpp (subloc/llama.py). No network: GitHub is faked."""

import hashlib
import json
import zipfile

import pytest

from subloc import cli, engine, llama
from subloc.cuda import LLAMA_BUILD
from subloc.errors import SublocError

SHA = "8d9d819261de10cc2f03921b8b79c779c5c458e21edb60c170ee30a7d462d684"


def asset(name, digest=f"sha256:{SHA}", size=1000):
    a = {"name": name, "size": size, "browser_download_url": f"https://example.invalid/{name}"}
    if digest:
        a["digest"] = digest
    return a


BUILD_ASSETS = [asset("llama-b11960-bin-win-cuda-12.4-x64.zip"), asset("llama-b11960-bin-win-cuda-13.4-x64.zip"),
                asset("llama-b11960-bin-win-vulkan-x64.zip"), asset("llama-b11960-bin-win-cpu-x64.zip"),
                asset("cudart-llama-bin-win-cuda-12.4-x64.zip")]  # fmt: skip


def test_the_windows_builds_subloc_uses_are_found_with_their_checksums():
    tag, found = llama.parse_assets({"assets": BUILD_ASSETS})
    assert tag == "b11960" and sorted(found) == ["cuda", "vulkan"]
    assert found["cuda"].url.endswith("cuda-12.4-x64.zip") and found["cuda"].sha256 == SHA
    _, none = llama.parse_assets({"assets": [asset("llama-b11960-bin-win-vulkan-x64.zip", digest=None)]})
    assert none == {}  # no checksum published: never used


class Answer:
    def __init__(self, data=None, content=b""):
        self._data, self.content, self.text = data, content, content.decode() if content else json.dumps(data)

    def json(self):
        return self._data


def fake_github(monkeypatch, pointer_text=b"b11960\n", digest=None):
    digest = digest or "sha256:" + hashlib.sha256(pointer_text).hexdigest()
    answers = {
        f"{llama.API}/releases/latest": Answer(
            {"tag_name": "v0.8.0", "html_url": "https://x/v0.8.0", "assets": [asset("nightly-tag.txt", digest)]}
        ),
        "https://example.invalid/nightly-tag.txt": Answer(content=pointer_text),
        f"{llama.API}/releases/tags/b11960": Answer({"tag_name": "b11960", "assets": BUILD_ASSETS}),
    }
    monkeypatch.setattr(llama, "_get", lambda url, timeout: answers[url])


def test_the_newest_stable_release_is_followed_to_its_build(monkeypatch):
    fake_github(monkeypatch)
    b = llama.latest()
    assert (b.tag, b.release, sorted(b.assets)) == ("b11960", "v0.8.0", ["cuda", "vulkan"])
    assert llama.newer(b, {"cuda": LLAMA_BUILD, "vulkan": "b11960"}) == ["cuda"]
    assert llama.newer(None, {"cuda": LLAMA_BUILD}) == []


def test_a_pointer_that_doesnt_match_its_checksum_is_refused(monkeypatch):
    fake_github(monkeypatch, digest="sha256:" + "0" * 64)
    with pytest.raises(SublocError, match="checksum"):
        llama.latest()


def make_zip(path, tag="b11960"):
    with zipfile.ZipFile(path, "w") as z:
        for name in ("llama-server.exe", "llama-server-impl.dll", "ggml-cuda.dll", "llama-cli.exe",
                     "llama-cli-impl.dll", "LICENSE-curl"):  # fmt: skip
            z.writestr(f"build/{name}", f"{tag} {name}")


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setattr(llama, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(engine, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(engine, "_vendor", lambda: tmp_path / "vendor")
    return tmp_path


def fake_download(monkeypatch, tag="b11960"):
    def download(url, dest, sha256, on_progress, cancel):
        dest.parent.mkdir(parents=True, exist_ok=True)
        make_zip(dest, tag)
        return dest

    monkeypatch.setattr(llama, "download", download)


def build(tag="b11960"):
    return llama.Build(tag, "v0.8.0", "page", llama.parse_assets({"assets": BUILD_ASSETS})[1])


def install_old(d, tag=None):
    d.mkdir(parents=True)
    (d / "llama-server.exe").write_text("old")
    if tag:
        (d / "version.txt").write_text(tag)


def test_an_update_is_checked_switched_to_and_can_be_undone(data, monkeypatch):
    install_old(data / "llama" / "cuda")  # Setup's download, before version.txt existed: the bundled build
    fake_download(monkeypatch)
    checked = []
    monkeypatch.setattr(llama, "_check", lambda d, tag: checked.append((sorted(p.name for p in d.iterdir()), tag)))
    assert llama.update("cuda", build()) == "b11960"
    cur = data / "llama" / "cuda"
    assert sorted(p.name for p in cur.iterdir()) == ["LICENSE-curl", "ggml-cuda.dll", "llama-server-impl.dll",
                                                     "llama-server.exe", "version.txt"]  # fmt: skip
    assert checked[0][1] == "b11960" and "llama-server.exe" in checked[0][0]
    assert llama.version_of(cur) == "b11960" and llama.previous("cuda") == LLAMA_BUILD
    assert not (data / "downloads" / "llama-b11960-bin-win-cuda-12.4-x64.zip").exists()
    assert llama.undo("cuda") == LLAMA_BUILD
    assert (cur / "llama-server.exe").read_text() == "old" and llama.previous("cuda") == ""


def test_a_build_that_doesnt_start_is_never_used(data, monkeypatch):
    install_old(data / "llama" / "cuda", "b11457")
    fake_download(monkeypatch)

    def broken(d, tag):
        raise SublocError("The new translator engine didn't start; Subloc keeps the one it has.")

    monkeypatch.setattr(llama, "_check", broken)
    with pytest.raises(SublocError, match="didn't start"):
        llama.update("cuda", build())
    assert (data / "llama" / "cuda" / "llama-server.exe").read_text() == "old"
    assert not (data / "llama" / "cuda.new").exists() and llama.previous("cuda") == ""


def test_the_engine_uses_the_newest_build_and_undo_goes_back_to_the_bundled_one(data, monkeypatch):
    install_old(data / "vendor" / "vulkan")  # bundled with the app (no version.txt: this Subloc's build)
    assert engine._backend_dir("vulkan") == data / "vendor" / "vulkan"
    install_old(data / "llama" / "vulkan", "b11960")  # an update
    assert engine._backend_dir("vulkan") == data / "llama" / "vulkan"
    assert llama.current() == {"vulkan": "b11960"} and llama.previous("vulkan") == LLAMA_BUILD
    llama.undo("vulkan")
    assert engine._backend_dir("vulkan") == data / "vendor" / "vulkan" and llama.current() == {"vulkan": LLAMA_BUILD}
    (data / "llama").mkdir(exist_ok=True)
    install_old(data / "llama" / "cuda", "b10000")  # older than the bundled build: not used over it
    install_old(data / "vendor" / "cuda")
    assert engine._backend_dir("cuda") == data / "vendor" / "cuda"


def test_cli_engine_reports_checks_and_refuses_while_busy(data, monkeypatch, capsys):
    install_old(data / "vendor" / "cuda")
    code = cli.run_cli(["engine", "--json"])
    assert code == 0 and json.loads(capsys.readouterr().out)["builds"] == {"cuda": LLAMA_BUILD}
    monkeypatch.setattr(llama, "latest", lambda: build())
    code = cli.run_cli(["engine", "--check", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and (out["latest"], out["newer"]) == ("b11960", ["cuda"])
    monkeypatch.setattr(cli, "holder", lambda: "Subloc's window: Episode 3")
    monkeypatch.setattr(llama, "update", lambda *a, **k: pytest.fail("must not update while busy"))
    assert cli.run_cli(["engine", "--update"]) == cli.BUSY
    assert cli.run_cli(["engine", "--undo"]) == 1  # nothing to go back to
