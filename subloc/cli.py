"""Subloc from the command line: the same work as the app, with the app's settings (any of them can be overridden).

    subloc translate VIDEO_OR_FOLDER...    subtitles in your language (folders: every subfolder too)
    subloc transcribe VIDEO_OR_FOLDER...   subtitles in the spoken language only
    subloc info VIDEO                      tracks, subtitle files, progress
    subloc lines VIDEO                     the subtitle lines (id, time, original, translation)
    subloc edit VIDEO LINE_ID "text"       correct a translated line
    subloc retranslate VIDEO LINE_ID...    translate some lines again (after fixing a name; --to woman for "you")
    subloc burn VIDEO                      a copy of the video with the subtitles drawn in
    subloc folder FOLDER                   what still needs subtitles under a folder (quick)
    subloc library [list|add|remove|refresh|watch]
    subloc queue [list|add|clear]          the app's queue (start it in the app)
    subloc status | models | styles
    subloc models --search WORDS | --get OWNER/NAME   models from Hugging Face for the built-in translator
    subloc engine [--check|--update|--undo]  the built-in translator's engine (llama.cpp)
    subloc update [--check]                a newer Subloc? download and install it (installed app only)
    subloc mcp                             MCP server for AI agents (stdin/stdout)

Every command takes --json: one JSON document on stdout, progress on stderr.
Exit codes: 0 done · 1 failed · 2 wrong usage · 3 busy (another Subloc is using the graphics card and --no-wait).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import library, service
from .busy import holder

COMMANDS = {"translate", "transcribe", "info", "lines", "edit", "retranslate", "burn", "folder", "library", "queue",
            "status", "models", "styles", "update", "engine", "mcp"}  # fmt: skip
BUSY = 3


def _compat(argv: list[str]) -> list[str]:
    """The first command line (subloc VIDEO --to ar, --info, --models) still works."""
    if not argv or argv[0] in COMMANDS or argv[0] in ("-h", "--help", "--version"):
        return argv
    if "--models" in argv:
        return ["models"]
    if "--info" in argv:
        return ["info", *[a for a in argv if a != "--info"]]
    return ["translate", *argv]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subloc",
        description="Local subtitles: listen, translate, fix, burn in.",
        epilog="Every command takes --json. Agents: see 'subloc mcp' and docs/agents.md.",
    )
    p.add_argument("--version", action="version", version=f"subloc {service.version()}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def cmd(name: str, help: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help, description=help)
        c.add_argument("--json", action="store_true", help="print one JSON document (progress goes to stderr)")
        return c

    for name, help in (("translate", "make subtitles in your language"),
                       ("transcribe", "make subtitles in the spoken language only")):  # fmt: skip
        c = cmd(name, help)
        c.add_argument("paths", nargs="+", type=Path, metavar="VIDEO_OR_FOLDER")
        if name == "translate":
            c.add_argument("--to", help="subtitle language, e.g. ar, en (default: the app's choice)")
            c.add_argument("--model", help="translator model id on your server (default: the app's choice)")
            c.add_argument("--no-names", action="store_true", help="skip the name pass (faster, less consistent)")
            c.add_argument(
                "--parallel", type=int, choices=range(1, 5), metavar="1-4", help="translation requests at once"
            )
            c.add_argument("--style", help="a style preset for the .ass file, or 'none' (default: the app's choice)")
            c.add_argument("--retranslate", action="store_true", help="translate again even if already done")
        c.add_argument("--from", dest="source", help="spoken language code, or 'mixed' (default: detect)")
        c.add_argument("--quality", choices=sorted(service.SPEECH_QUALITY), help="speech recognition model size")
        c.add_argument("--listen", action="store_true", help="always listen, even when the file has subtitles inside")
        c.add_argument("--subs", type=int, metavar="N", help="translate the file's subtitle track N (1 = first)")
        c.add_argument("--audio-track", type=int, metavar="N", help="audio track N (1 = first)")
        c.add_argument("--retranscribe", action="store_true", help="listen again even if done before")
        c.add_argument("--no-wait", action="store_true", help="exit (code 3) instead of waiting when Subloc is busy")

    c = cmd("info", "a video's tracks, subtitle files and progress")
    c.add_argument("video", type=Path)
    c.add_argument("--to", help="language to report progress for")
    c = cmd("lines", "the subtitle lines of a video")
    c.add_argument("video", type=Path)
    c.add_argument("--offset", type=int, default=0)
    c.add_argument("--limit", type=int, default=100000)
    c = cmd("edit", "correct one translated line (the subtitle files are written again)")
    c.add_argument("video", type=Path)
    c.add_argument("line", type=int, metavar="LINE_ID")
    c.add_argument("text")
    c = cmd("retranslate", "translate some lines again, with their neighbours and the name list")
    c.add_argument("video", type=Path)
    c.add_argument("lines", type=int, nargs="+", metavar="LINE_ID")
    c.add_argument(
        "--to",
        dest="listener",
        choices=["man", "woman", "several"],
        help="who these lines are spoken to (fixes he/she/plural 'you'; kept for later)",
    )
    c.add_argument("--no-wait", action="store_true")
    c = cmd("burn", "make a copy of the video with the subtitles drawn in (plays everywhere)")
    c.add_argument("video", type=Path)
    c.add_argument("--style", help="style preset (default: the app's style)")
    c.add_argument("--out", type=Path, help="output file (default: VIDEO.<lang>.subtitled.mp4)")
    c.add_argument("--no-wait", action="store_true")
    c = cmd("folder", "every video under a folder and whether it still needs subtitles (quick: files only)")
    c.add_argument("folder", type=Path)
    c.add_argument("--to", help="language to check for")
    c.add_argument("--missing", action="store_true", help="only the ones that still need subtitles")
    c = cmd("library", "the app's library of folders")
    c.add_argument("action", nargs="?", default="list", choices=["list", "add", "remove", "refresh", "watch"])
    c.add_argument("arg", nargs="?", help="a folder (add, remove, refresh) or on/off (watch)")
    c.add_argument("--to", help="language to check for")
    c.add_argument("--missing", action="store_true", help="list only videos that still need subtitles")
    c = cmd("queue", "the app's queue (the app runs it)")
    c.add_argument("action", nargs="?", default="list", choices=["list", "add", "clear"])
    c.add_argument("paths", nargs="*", type=Path)
    cmd("status", "settings, translator and whether Subloc is busy")
    c = cmd("models", "translator models: in use; search Hugging Face; download one for the built-in translator")
    c.add_argument("--search", metavar="WORDS", nargs="?", const="", help="search Hugging Face for GGUF models")
    c.add_argument("--limit", type=int, default=20, help="how many search results (default 20)")
    c.add_argument("--get", metavar="OWNER/NAME", help="download a model (the file that suits this PC), use it")
    c.add_argument("--file", help="with --get: this file (name or quantization, e.g. Q4_K_M) instead")
    c = cmd("engine", "the built-in translator's engine (llama.cpp): version, updates")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="is there a newer stable llama.cpp?")
    g.add_argument("--update", action="store_true", help="download it (checked), test it, switch to it")
    g.add_argument("--undo", action="store_true", help="go back to the engine the last update replaced")
    cmd("styles", "subtitle style presets")
    c = cmd("update", "is there a newer Subloc? download and install it (the installed app; from source: git)")
    c.add_argument("--check", action="store_true", help="only say whether there is one")
    sub.add_parser("mcp", help="run as an MCP server for AI agents (on stdin/stdout)")
    return p


def main(argv: list[str] | None = None) -> None:
    sys.exit(run_cli(_compat(list(sys.argv[1:] if argv is None else argv))))


def run_cli(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Arabic, Japanese… in names and lines
    a = _parser().parse_args(argv)
    if a.cmd == "mcp":
        from .mcp_server import serve

        serve()
        return 0
    try:
        return COMMAND_FUNCS[a.cmd](a)
    except (FileNotFoundError, ValueError, KeyError, StopIteration) as e:
        msg = str(e) or type(e).__name__
        if isinstance(e, (KeyError, StopIteration)):
            msg = "No such line."
        if a.json:
            _emit({"error": msg})
        else:
            print(f"subloc: {msg}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------- helpers


def _emit(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def _say(a: argparse.Namespace, text: str) -> None:
    """Progress for people: stdout normally, stderr with --json (stdout is the JSON document)."""
    print(text, file=sys.stderr if a.json else sys.stdout, flush=True)


def _settings(a: argparse.Namespace) -> service.Settings:
    st = service.settings()
    if getattr(a, "to", None):
        st.target = a.to
    if getattr(a, "source", None):
        st.source = a.source
    if getattr(a, "quality", None):
        st.asr_model = service.SPEECH_QUALITY[a.quality]
    if getattr(a, "model", None):
        st.llm_model = a.model
    if getattr(a, "parallel", None):
        st.parallel = a.parallel
    if getattr(a, "no_names", False):
        st.glossary = False
    if getattr(a, "style", None):
        st.style = service.style_named(a.style, st)
    return st


def _run_job(a: argparse.Namespace, jobs: service.Jobs, job: service.Job) -> service.Job:
    """Wait for a job, showing progress; Ctrl+C stops it (progress is kept)."""
    shown = {"waiting": False, "pct": -1}

    def tick(j: service.Job) -> None:
        if j.status == "waiting" and not shown["waiting"]:  # only when it really has to wait its turn
            shown["waiting"] = True
            _say(a, j.stage[0].upper() + j.stage[1:] + "…")
        pct = int(j.progress * 100) // 10 * 10
        if j.kind == "burn" and pct != shown["pct"]:
            shown["pct"] = pct
            _say(a, f"  {pct}%")

    try:
        return jobs.wait(job, on_tick=tick)
    except KeyboardInterrupt:
        _say(a, "Stopping… (progress is saved: run the same command to continue)")
        jobs.cancel(job.id)
        return jobs.wait(job)


def _jobs(a: argparse.Namespace) -> service.Jobs:
    return service.Jobs("the command line", on_log=lambda job, msg: _say(a, msg))


def _busy_exit(a: argparse.Namespace) -> int | None:
    if getattr(a, "no_wait", False) and (who := holder()):
        if a.json:
            _emit({"error": f"Subloc is busy: {who}", "busy": True})
        else:
            print(f"subloc: busy ({who}). Try again later, or leave out --no-wait to wait.", file=sys.stderr)
        return BUSY
    return None


def _finish(a: argparse.Namespace, job: service.Job) -> int:
    if a.json:
        _emit(job.report())
    else:
        for v in job.videos:
            mark = {"done": "✓", "skipped": "·", "failed": "✗", "stopped": "■"}.get(v.status, " ")
            _say(a, f"{mark} {Path(v.path).name}: {v.status}" + (f" — {v.error}" if v.error else ""))
            for f in v.files:
                _say(a, f"    {f}")
        if job.error and not job.videos:
            _say(a, f"✗ {job.error}")
        if job.result.get("output"):
            _say(a, f"Saved: {job.result['output']}")
    failed = job.status != "done" or any(v.status == "failed" for v in job.videos)
    return 1 if failed else 0


# ---------------------------------------------------------------- commands


def cmd_translate(a: argparse.Namespace) -> int:
    if (code := _busy_exit(a)) is not None:
        return code
    for p in a.paths:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
    st = _settings(a)
    over: dict[str, Any] = {"retranscribe": a.retranscribe, "retranslate": getattr(a, "retranslate", False)}
    if a.audio_track:
        over["audio_track"] = a.audio_track - 1
    if a.subs:
        over["subs_track"] = a.subs - 1
    if a.listen or a.subs:
        over["prefer_builtin_subs"] = False
    jobs = _jobs(a)
    job = service.submit_subtitles(jobs, a.paths, st, translate=a.cmd == "translate", **over)
    _say(a, f"{len(job.videos)} video(s) → " + (st.target if a.cmd == "translate" else "spoken language"))
    return _finish(a, _run_job(a, jobs, job))


def cmd_info(a: argparse.Namespace) -> int:
    st = _settings(a)
    r = service.video_report(a.video, st.target)
    if a.json:
        _emit(r)
        return 0
    print(f"{Path(r['path']).name}  ·  {r['duration'] / 60:.1f} min")
    for t in r["audio_tracks"]:
        print(f"  audio {t['number']}: {t['language'] or '?'}  {t['title']}")
    for t in r["subtitle_tracks"]:
        print(f"  subtitles {t['number']}: {t['language'] or '?'}  {t['title']}" +
              ("" if t["text"] else "  (picture subtitles: can't be translated)"))  # fmt: skip
    for f in r["subtitle_files"]:
        print(f"  file: {f}")
    print(f"  {st.target}: {r['status_text']}" + (f"  ({r['translated']}/{r['lines']} lines)" if r["lines"] else ""))
    return 0


def cmd_lines(a: argparse.Namespace) -> int:
    r = service.read_lines(a.video, a.offset, a.limit)
    if a.json:
        _emit(r)
        return 0
    for line in r["lines"]:
        voice = " ".join(v for v in (line["speaker"], {"male": "m", "female": "f"}.get(line["gender"] or "")) if v)
        print(f"{line['id']:>5}  {_clock(line['start'])}  " + (f"[{voice}] " if voice else "") + line["text"])
        if line["translation"]:
            print(f"{'':>5}  {'':>9}  {line['translation']}")
    return 0


def _clock(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):>3}:{s:05.2f}"


def cmd_edit(a: argparse.Namespace) -> int:
    r = service.edit_line(a.video, a.line, a.text, _settings(a))
    if a.json:
        _emit(r)
    else:
        print(f"Line {r['id']} saved: {r['translation']}")
    return 0


def cmd_retranslate(a: argparse.Namespace) -> int:
    if (code := _busy_exit(a)) is not None:
        return code
    jobs = _jobs(a)
    job = _run_job(a, jobs, service.submit_retranslate(jobs, a.video, a.lines, _settings(a), a.listener))
    if not a.json and job.status == "done":
        for line in job.result["lines"]:
            _say(a, f"{line['id']:>5}  {line['translation']}")
    return _finish(a, job)


def cmd_burn(a: argparse.Namespace) -> int:
    if (code := _busy_exit(a)) is not None:
        return code
    st = _settings(a)
    jobs = _jobs(a)
    return _finish(a, _run_job(a, jobs, service.submit_burn(jobs, a.video, st, st.style, a.out)))


def cmd_folder(a: argparse.Namespace) -> int:
    if not a.folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {a.folder}")
    r = service.folder_report(a.folder, _settings(a).target, a.missing)
    return _report_list(a, r, f"{r['videos']} videos under {r['folder']} · {r['done']} done")


def _report_list(a: argparse.Namespace, r: dict, head: str) -> int:
    if a.json:
        _emit(r)
        return 0
    print(head)
    base = Path(r["folder"]) if r.get("folder") else None
    for it in r["items"]:
        path = Path(it["path"])
        shown = path.relative_to(base) if base and path.is_relative_to(base) else path  # shorter for people
        print(f"  {it['status_text']:<28} {shown}")
    return 0


def cmd_library(a: argparse.Namespace) -> int:
    if a.action == "list":
        r = service.library_report(_settings(a).target, a.missing)
        head = (f"{len(r['folders'])} folders · {r['videos']} videos · {r['done']} done"
                + (" · watching for changes" if r["watching"] else ""))  # fmt: skip
        return _report_list(a, r, head)
    if a.action == "watch":
        if a.arg not in ("on", "off"):
            raise ValueError("Say 'subloc library watch on' or 'off'.")
        lib = library.load()
        lib.watch = a.arg == "on"
        library.save(lib)
        _emit({"watching": lib.watch}) if a.json else print(f"Watching for changes: {a.arg} (in the app)")
        return 0
    if a.action == "remove":
        if not a.arg:
            raise ValueError("Which folder?")
        lib = library.load()
        library.remove_root(lib, a.arg)
        library.save(lib)
        _emit({"folders": lib.roots}) if a.json else print(f"Removed (nothing deleted from disk). Folders: {lib.roots}")
        return 0
    if a.action == "add" and not a.arg:
        raise ValueError("Which folder?")
    jobs = _jobs(a)
    job = service.submit_scan(jobs, Path(a.arg) if a.arg else None, add=a.action == "add")
    _say(a, f"Reading {a.arg or 'every library folder'}…")
    job = _run_job(a, jobs, job)
    if not a.json and job.status == "done":
        _say(a, f"Done: {job.result['videos']} videos in the library.")
    return _finish(a, job)


def cmd_queue(a: argparse.Namespace) -> int:
    from .batch import Item, find_videos, load_queue, save_queue

    items = load_queue()
    if a.action == "add":
        known = {it.video.resolve() for it in items}
        new = [v for v in find_videos(a.paths) if v not in known]
        items += [Item(v) for v in new]
        save_queue(items)
    elif a.action == "clear":
        items = [it for it in items if it.status not in ("done", "skipped", "failed")]
        save_queue(items)
    if a.json:
        _emit({"queue": [{"video": str(it.video), "status": it.status, "error": it.error} for it in items]})
    else:
        for it in items:
            print(f"  {it.status:<12} {it.video}")
        print(f"{len(items)} in the queue. Start it in the app (Queue → Start queue).")
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    st = service.settings()
    r: dict[str, Any] = {"version": service.version(), "settings": service.settings_report(st),
         "translator": service.translator_status(st), "busy": holder() or None}  # fmt: skip
    if a.json:
        _emit(r)
        return 0
    s, tr = r["settings"], r["translator"]
    print(f"Subloc {r['version']}")
    print(f"  subtitles in {s['target']} · speech model {s['speech_model']} · style {s['style']}")
    where = tr.get("url") or "built-in"
    print(f"  translator ({where}): " + ("ready · " + str(tr.get("model")) if tr["ready"] else tr["problem"]))
    print(f"  busy: {r['busy']}" if r["busy"] else "  not busy")
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    if a.search is not None:
        return _models_search(a)
    if a.get:
        return _models_get(a)
    tr = service.translator_status(service.settings())
    if a.json:
        _emit(tr)
    elif tr["kind"] == "built-in":
        print(f"Built-in translator: {tr['model']}" + ("" if tr["ready"] else f" — {tr['problem']}"))
    else:
        for m in tr.get("available", []):
            print(f"{'*' if m == tr.get('loaded') else ' '} {m}")
        if not tr["ready"]:
            print(tr["problem"])
    return 0


def _models_search(a: argparse.Namespace) -> int:
    """GGUF translator models on Hugging Face (most downloaded first)."""
    from . import hub
    from .errors import SublocError

    try:
        repos = hub.search(a.search, limit=a.limit)
    except SublocError as e:
        return _fail(a, str(e))
    if a.json:
        _emit({"models": [{"id": r.id, "downloads": r.downloads, "likes": r.likes, "updated": r.updated,
                           "licence": r.licence, "page": r.page} for r in repos]})  # fmt: skip
    else:
        for r in repos:
            extra = f", {r.licence}" if r.licence else ""
            print(f"{r.id}  ({r.downloads:,} downloads, updated {r.updated}{extra})")
        if repos:
            print("\nsubloc models --get OWNER/NAME   downloads the file that suits this PC and uses it")
    return 0


def _models_get(a: argparse.Namespace) -> int:
    """Download a Hugging Face model's file (checked) and make it the built-in translator."""
    from . import hub
    from .config import Config
    from .download import Cancelled
    from .errors import SublocError
    from .hardware import detect

    try:
        files = hub.files(a.get)
        if a.file:
            f = next((x for x in files if a.file in (x.name, x.path, x.quant)), None)
            if f is None:
                names = ", ".join(x.name for x in files) or "none Subloc can use"
                return _fail(a, f"No file '{a.file}' in {a.get}. Files: {names}")
        else:
            f = hub.recommend(files, detect())
            if f is None:
                return _fail(a, f"{a.get} has no single GGUF file Subloc can use (it may be split into parts).")
        _say(a, f"Downloading {f.name} ({f.size / 1e9:.1f} GB)…")
        last = [-1]

        def progress(done: int, total: int) -> None:
            pct = int(done * 100 / max(total, 1)) // 10 * 10
            if pct > last[0]:
                last[0] = pct
                _say(a, f"  {pct}%")

        path = hub.fetch(f, progress)
    except Cancelled:
        return _fail(a, "Stopped.")
    except SublocError as e:
        return _fail(a, str(e))
    cfg = Config.load()
    cfg.provider, cfg.builtin_model = "builtin", str(path)
    cfg.save()
    if a.json:
        _emit({"model": f.path, "repo": f.repo, "quant": f.quant, "size": f.size, "path": str(path), "in_use": True})
    else:
        print(f"Saved {path}\nThe built-in translator now uses it.")
    return 0


def cmd_engine(a: argparse.Namespace) -> int:
    """The built-in translator's engine (llama.cpp): its version; check for, install or undo an update."""
    from . import llama
    from .errors import SublocError

    have = llama.current()
    can_undo = {b: llama.previous(b) for b in have if llama.previous(b)}
    report: dict[str, Any] = {"engine": "llama.cpp", "builds": have, "undo": can_undo}
    if a.undo:
        if not can_undo:
            return _fail(a, "There is no earlier translator engine to go back to.")
        try:
            for b in can_undo:
                llama.undo(b)
        except SublocError as e:
            return _fail(a, str(e))
        report["builds"], report["undo"] = llama.current(), {}
        if a.json:
            _emit(report)
        else:
            print("Back to llama.cpp " + ", ".join(sorted(set(report["builds"].values()))) + ".")
        return 0
    if not (a.check or a.update):
        if a.json:
            _emit(report)
        else:
            print("\n".join(f"{b}: llama.cpp {tag}" for b, tag in have.items()) or "Not installed.")
        return 0
    try:
        build = llama.latest()
    except SublocError as e:
        return _fail(a, str(e))
    todo = llama.newer(build, have)
    report.update(latest=build.tag if build else None, release=build.release if build else None, newer=todo)
    if build is None or not todo or a.check:
        if a.json:
            _emit(report)
        elif build is not None and todo:
            print(f"llama.cpp {build.tag} ({build.release}) is available for: {', '.join(todo)}. "
                  "subloc engine --update installs it.")  # fmt: skip
        else:
            print("You have the newest translator engine.")
        return 0
    if who := holder():
        return _fail(a, f"Subloc is busy ({who}). Update when it's done.", BUSY)
    try:
        for b in todo:
            _say(a, f"Downloading llama.cpp {build.tag} for {b} ({build.assets[b].size / 1e6:.0f} MB)…")
            llama.update(b, build)
    except SublocError as e:
        return _fail(a, str(e))
    report["builds"] = llama.current()
    if a.json:
        _emit(report)
    else:
        print(f"The translator engine is now llama.cpp {build.tag}.")
    return 0


def _fail(a: argparse.Namespace, msg: str, code: int = 1) -> int:
    if a.json:
        _emit({"error": msg})
    else:
        print(f"subloc: {msg}", file=sys.stderr)
    return code


def cmd_styles(a: argparse.Namespace) -> int:
    from .style import PRESETS

    names = [*PRESETS, "none"]
    _emit({"styles": names}) if a.json else print("\n".join(names))
    return 0


def cmd_update(a: argparse.Namespace) -> int:
    """Check GitHub for a newer release; when there is one and Subloc is installed, download its installer
    (checked against GitHub's SHA-256) and start it. The installer then replaces this program: run subloc again
    when it's done."""
    from . import update
    from .errors import SublocError

    mine = service.version()

    def fail(msg: str, code: int = 1, **extra: Any) -> int:
        if a.json:
            _emit({"error": msg, "current": mine, **extra})
        else:
            print(f"subloc: {msg}", file=sys.stderr)
        return code

    try:
        rel = update.latest()
    except SublocError as e:
        return fail(str(e))
    newer = rel is not None and update.is_newer(rel.version, mine)
    report: dict[str, Any] = {"current": mine, "latest": rel.version if rel else None, "newer": newer,
                              "page": rel.page if rel else None, "installing": False}  # fmt: skip
    if not newer or a.check or rel is None:
        if a.json:
            _emit(report)
        elif newer and rel is not None:
            print(f"Subloc {rel.version} is available (you have {mine}): {rel.page}")
        else:
            print(f"You have the latest version ({mine}).")
        return 0
    if update.app_dir() is None:
        report["note"] = r"Running from the source folder: update with git (git pull, then . .\env.ps1 -Setup)."
        if a.json:
            _emit(report)
        else:
            print(f"Subloc {rel.version} is available. {report['note']}")
        return 0
    if not rel.installable:
        return fail(f"Subloc {rel.version} has no checked installer to download; get it from {rel.page}", **report)
    if who := holder():
        return fail(f"Subloc is busy ({who}). Update when it's done.", BUSY, busy=True, **report)
    if update.other_windows():
        return fail("Subloc's window is open: close it, then update.", BUSY, busy=True, **report)
    _say(a, f"Downloading Subloc {rel.version} ({rel.size / 1e6:.0f} MB)…")
    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(done * 100 / max(total, 1)) // 10 * 10
        if pct > last[0]:
            last[0] = pct
            _say(a, f"  {pct}%")

    try:
        installer = update.fetch(rel, progress)
        update.run_installer(installer, reopen=False)
    except SublocError as e:
        return fail(str(e), **report)
    report["installing"] = True
    if a.json:
        _emit(report)
    else:
        print(f"Installing Subloc {rel.version}… it takes about a minute; then run subloc again.")
    return 0


COMMAND_FUNCS = {"translate": cmd_translate, "transcribe": cmd_translate, "info": cmd_info, "lines": cmd_lines,
                 "edit": cmd_edit, "retranslate": cmd_retranslate, "burn": cmd_burn, "folder": cmd_folder,
                 "library": cmd_library, "queue": cmd_queue, "status": cmd_status, "models": cmd_models,
                 "styles": cmd_styles, "update": cmd_update, "engine": cmd_engine}  # fmt: skip


if __name__ == "__main__":
    main()
