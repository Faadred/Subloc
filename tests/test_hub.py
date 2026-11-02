"""More translator models from Hugging Face (subloc/hub.py). No network: Hugging Face's answers are faked."""

import json

import pytest

from subloc import cli, hub
from subloc.config import Config
from subloc.errors import SublocError
from subloc.hardware import Hardware

SHA = "0ffb122c8b6921f13cbc34186e052524d0b5803b17f4867b7197a561400b3770"
GB = 1_000_000_000


def entry(path, size_gb, sha=SHA):
    e = {"type": "file", "path": path, "size": int(size_gb * GB)}
    if sha:
        e["lfs"] = {"oid": sha, "size": int(size_gb * GB)}
    return e


TREE = [
    {"type": "directory", "path": "BF16"},
    entry(".gitattributes", 0.000002, sha=None),
    entry("README.md", 0.000002, sha=None),
    entry("model-Q8_0.gguf", 8.0),
    entry("model-UD-Q4_K_M.gguf", 5.3),
    entry("model-Q2_K.gguf", 2.1),
    entry("mmproj-BF16.gguf", 1.0),
    entry("MTP/mtp-model-Q8_0.gguf", 0.5),
    entry("BF16/model-BF16-00001-of-00002.gguf", 49.0),
    entry("model-Q6_K.gguf", 6.2, sha="not-a-sha"),
]
RTX24 = Hardware("RTX", 24.0, [], 64.0, 16)
RTX4 = Hardware("RTX", 4.0, [], 32.0, 8)
NO_GPU = Hardware(None, 0.0, [], 16.0, 8)


def test_only_files_subloc_can_run_and_check_are_listed():
    files = hub.parse_files("owner/model-GGUF", TREE)
    assert [f.path for f in files] == ["model-Q2_K.gguf", "model-UD-Q4_K_M.gguf", "model-Q8_0.gguf"]
    assert [f.quant for f in files] == ["Q2_K", "Q4_K_M", "Q8_0"]
    assert all(f.sha256 == SHA for f in files)


def test_search_results_leave_out_gated_and_non_translating_models():
    data = [{"id": "a/ok-GGUF", "downloads": 5, "likes": 1, "lastModified": "2026-07-17T00:00:00Z",
             "tags": ["gguf", "license:apache-2.0"], "gated": False},
            {"id": "b/gated-GGUF", "gated": "manual"},
            {"id": "c/embed-GGUF", "tags": ["gguf", "sentence-similarity"]},
            {"id": "d/whisper-GGUF", "pipeline_tag": "automatic-speech-recognition"}]  # fmt: skip
    repos = hub.parse_repos(data)
    assert [r.id for r in repos] == ["a/ok-GGUF"]
    assert (repos[0].licence, repos[0].updated, repos[0].page) == ("apache-2.0", "2026-07-17",
                                                                   "https://huggingface.co/a/ok-GGUF")  # fmt: skip


def test_the_fit_for_this_pc():
    assert hub.fit(5 * GB, RTX24) == "gpu"
    assert hub.fit(8 * GB, RTX4) == "partly"
    assert hub.fit(5 * GB, NO_GPU) == "cpu"
    assert hub.fit(40 * GB, NO_GPU) == "too_big"
    assert hub.fit(5 * GB, None) == "unknown"


def test_the_suggested_file_runs_well_here():
    files = hub.parse_files("owner/model-GGUF", TREE)
    assert hub.recommend(files, RTX24).quant == "Q4_K_M"  # preferred, fits the card
    assert hub.recommend(files, RTX4).quant == "Q4_K_M"  # nothing fits fully: the preferred one that runs at all
    tiny = Hardware(None, 0.0, [], 6.0, 4)
    assert hub.recommend(files, tiny).quant == "Q2_K"  # nothing runs well: the smallest
    assert hub.recommend([], RTX24) is None


def test_a_found_model_is_a_catalog_entry_saved_under_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBLOC_DATA", str(tmp_path))
    f = hub.parse_files("owner/Model-7B-GGUF", TREE)[1]
    m = f.as_model()
    assert m.url == "https://huggingface.co/owner/Model-7B-GGUF/resolve/main/model-UD-Q4_K_M.gguf"
    assert m.label == "Model-7B Q4_K_M" and m.sha256 == SHA
    assert hub.local_path(m) == tmp_path / "models" / "llm" / "hub" / "owner--Model-7B-GGUF" / "model-UD-Q4_K_M.gguf"


def test_downloads_are_checked_against_hugging_faces_checksum(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBLOC_DATA", str(tmp_path))
    seen = {}

    def fake_download(url, dest, sha256, on_progress, cancel):
        seen.update(url=url, sha=sha256)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"gguf")
        return dest

    monkeypatch.setattr(hub, "download", fake_download)
    path = hub.fetch(hub.parse_files("owner/m-GGUF", TREE)[1])
    assert path.exists() and seen["sha"] == SHA and seen["url"].endswith("/model-UD-Q4_K_M.gguf")


class Answer:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


def test_search_asks_for_gguf_models_most_downloaded_first(monkeypatch):
    calls = []
    monkeypatch.setattr(hub.httpx, "get", lambda url, params=None, **kw: calls.append((url, params)) or Answer(200, []))
    hub.search("qwen")
    url, params = calls[-1]
    assert url == hub.API and ("filter", "gguf") in params and ("sort", "downloads") in params
    assert ("search", "qwen") in params and ("expand[]", "gated") in params
    hub.search("")
    assert not any(k == "search" for k, _ in calls[-1][1])
    monkeypatch.setattr(hub.httpx, "get", lambda url, params=None, **kw: Answer(404, {}))
    with pytest.raises(SublocError):
        hub.files("owner/missing")
    with pytest.raises(SublocError, match="owner/name"):
        hub.files("not a repo")


def run(argv, capsys):
    code = cli.run_cli(argv)
    return code, capsys.readouterr().out


def test_cli_searches_and_gets_a_model(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SUBLOC_DATA", str(tmp_path))
    monkeypatch.setattr(hub, "search", lambda q, limit=20: hub.parse_repos([{"id": "owner/m-GGUF", "downloads": 9}]))
    code, out = run(["models", "--search", "m", "--json"], capsys)
    assert code == 0 and json.loads(out)["models"][0]["id"] == "owner/m-GGUF"

    monkeypatch.setattr(hub, "files", lambda repo: hub.parse_files(repo, TREE))
    monkeypatch.setattr("subloc.hardware.detect", lambda: RTX24)
    got = []

    def fake_fetch(f, progress=lambda d, t: None, cancel=None):
        got.append(f)
        progress(1, 1)
        return hub.local_path(f.as_model())

    monkeypatch.setattr(hub, "fetch", fake_fetch)
    code, out = run(["models", "--get", "owner/m-GGUF", "--json"], capsys)
    assert code == 0 and got[-1].quant == "Q4_K_M" and json.loads(out)["in_use"] is True
    cfg = Config.load()
    assert cfg.provider == "builtin" and cfg.builtin_model.endswith("model-UD-Q4_K_M.gguf")
    code, _ = run(["models", "--get", "owner/m-GGUF", "--file", "Q8_0"], capsys)
    assert code == 0 and got[-1].quant == "Q8_0"
    code, _ = run(["models", "--get", "owner/m-GGUF", "--file", "Q3_K_S"], capsys)
    assert code == 1
