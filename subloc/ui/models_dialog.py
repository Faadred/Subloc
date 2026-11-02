"""Find more models: search Hugging Face for a translator model, pick the file for this PC, download it (checked).

From Settings the dialog downloads the file itself (.path after Accepted); from Setup it only chooses (.choice) and
Setup's install step downloads it with everything else.
"""

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import hardware, hub
from ..config import models_dir
from ..download import Cancelled
from ..hardware import Hardware
from . import icons
from . import theme as t
from .i18n import tr, translate_tree
from .widgets import Banner, label


class _Bridge(QObject):
    repos = Signal(int, object, str)  # request, [hub.Repo], error
    files = Signal(int, object, str)  # request, [hub.File], error
    hw = Signal(object)  # Hardware
    progress = Signal(float, str)
    done = Signal(str, str)  # path, error ("stopped" when cancelled)


def _count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


class ModelSearchDialog(QDialog):
    def __init__(self, parent: QWidget, hw: Hardware | None = None, download: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Find more models")
        self.resize(820, 640)
        self.setStyleSheet(f"QDialog {{ background: {t.BG}; }}")
        self.hw, self.downloading, self.download_now = hw, False, download
        self.choice: hub.File | None = None  # the file picked (Setup)
        self.path: Path | None = None  # the file downloaded (Settings)
        self.request = 0
        self.cancel = threading.Event()
        self.bridge = _Bridge()
        self.bridge.repos.connect(self._show_repos)
        self.bridge.files.connect(self._show_files)
        self.bridge.hw.connect(self._got_hw)
        self.bridge.progress.connect(self._progress)
        self.bridge.done.connect(self._downloaded)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(12)
        root.addWidget(label("Find more models", "title"))
        hint = label("Translator models from Hugging Face, for the built-in translator. They aren't tested with "
                     "Subloc: the quality, in Arabic especially, can vary.", "faint")  # fmt: skip
        hint.setWordWrap(True)
        root.addWidget(hint)

        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search, e.g. gemma, qwen, aya, mistral")
        self.query.returnPressed.connect(self.search)
        go = QPushButton("Search")
        go.setIcon(icons.icon("search", t.TEXT))
        go.clicked.connect(self.search)
        row.addWidget(self.query, 1)
        row.addWidget(go)
        root.addLayout(row)

        self.results = QListWidget()
        self.results.setStyleSheet(
            f"QListWidget {{ background: {t.SURFACE}; border: 1px solid {t.LINE}; border-radius: 8px; }}"
            f"QListWidget::item {{ padding: 8px 10px; border-bottom: 1px solid {t.LINE}; }}"
            f"QListWidget::item:selected {{ background: {t.SURFACE_3}; color: {t.TEXT}; }}"
        )
        self.results.currentItemChanged.connect(lambda *_: self._pick_repo())
        root.addWidget(self.results, 1)

        frow = QHBoxLayout()
        frow.addWidget(label("File", "fieldLabel"))
        self.file = QComboBox()
        self.file.setObjectName("boxed")
        self.file.currentIndexChanged.connect(lambda *_: self._pick_file())
        frow.addWidget(self.file, 1)
        self.page = QPushButton("Model page")
        self.page.setObjectName("ghost")
        self.page.setIcon(icons.icon("external-link", t.TEXT))
        self.page.clicked.connect(self._open_page)
        frow.addWidget(self.page)
        root.addLayout(frow)

        self.note = Banner()
        root.addWidget(self.note)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet(
            f"QProgressBar {{ background: {t.SURFACE_3}; border: none; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {t.ACCENT}; border-radius: 3px; }}"
        )
        self.bar.hide()
        root.addWidget(self.bar)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.close_btn = QPushButton("Cancel")
        self.close_btn.clicked.connect(self.reject)
        self.ok = QPushButton("Download and use" if download else "Use this model")
        self.ok.setObjectName("primary")
        self.ok.setIcon(icons.icon("download", t.ACCENT_INK))
        self.ok.clicked.connect(self._use)
        buttons.addWidget(self.close_btn)
        buttons.addWidget(self.ok)
        root.addLayout(buttons)
        self._pick_repo()
        translate_tree(self)

        if hw is None:  # the fit for this PC needs a hardware check (a moment: nvidia-smi)
            threading.Thread(target=lambda: self.bridge.hw.emit(hardware.detect()), daemon=True).start()
        QTimer.singleShot(0, self.search)

    # ---------------------------------------------------------------- search

    def search(self) -> None:
        self.request += 1
        req, query, bridge = self.request, self.query.text(), self.bridge
        self.results.clear()
        self.note.say(tr("Searching Hugging Face…"), "info")

        def work() -> None:
            try:
                bridge.repos.emit(req, hub.search(query), "")
            except Exception as e:
                bridge.repos.emit(req, [], str(e) or type(e).__name__)

        threading.Thread(target=work, daemon=True).start()

    def _show_repos(self, req: int, repos: list, err: str) -> None:
        if req != self.request:
            return
        if err:
            self.note.say(tr(err), "error")
            return
        for r in repos:
            parts = [
                tr("{n} downloads", n=_count(r.downloads)),
                tr("updated {d}", d=f"\u2066{r.updated}\u2069"),
            ]  # the date reads left to right
            if r.licence:
                parts.append(tr("licence: {l}", l=r.licence))
            item = QListWidgetItem(f"{r.id}\n" + "  ·  ".join(parts))
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.results.addItem(item)
        if repos:
            self.note.say(tr("Choose a model, then the file to download."), "info")
            self.results.setCurrentRow(0)
        else:
            self.note.say(tr("No models found. Try another word."), "warning")

    def _repo(self) -> hub.Repo | None:
        item = self.results.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _pick_repo(self) -> None:
        repo = self._repo()
        self.file.clear()
        self.page.setEnabled(repo is not None)
        self.ok.setEnabled(False)
        if repo is None:
            return
        self.request += 1
        req, bridge = self.request, self.bridge
        self.file.addItem(tr("Reading the model's files…"), None)

        def work() -> None:
            try:
                bridge.files.emit(req, hub.files(repo.id), "")
            except Exception as e:
                bridge.files.emit(req, [], str(e) or type(e).__name__)

        threading.Thread(target=work, daemon=True).start()

    def _show_files(self, req: int, files: list, err: str) -> None:
        if req != self.request:
            return
        self.file.clear()
        if err:
            self.note.say(tr(err), "error")
            return
        if not files:
            self.note.say(tr("This model has no single file Subloc can use (it may be split into parts). Try "
                             "another one."), "warning")  # fmt: skip
            return
        best = hub.recommend(files, self.hw)
        for f in files:
            text = f"{f.quant or f.name}  ·  {f.size / 1e9:.1f} GB"
            fits = hub.FIT_TEXT[hub.fit(f.size, self.hw)]
            if fits:
                text += f"  ·  {tr(fits)}"
            if f == best:
                text += f"  ·  {tr('recommended')}"
            self.file.addItem(text, f)
        self.file.setCurrentIndex(files.index(best) if best in files else 0)
        self._pick_file()

    def _got_hw(self, hw: Hardware) -> None:
        self.hw = hw
        if self.file.count() and self.file.itemData(0) is not None:  # redo the fit with the real numbers
            self._show_files(self.request, [self.file.itemData(i) for i in range(self.file.count())], "")

    def _file(self) -> hub.File | None:
        return self.file.currentData()

    def _pick_file(self) -> None:
        f = self._file()
        self.ok.setEnabled(f is not None and not self.downloading)
        if f is None:
            return
        state = hub.fit(f.size, self.hw)
        have = hub.local_path(f.as_model()).exists()
        if have:
            self.note.say(tr("Already downloaded: it will be used right away."), "ok")
        elif state == "too_big":
            self.note.say(tr("This file needs about {n} GB of memory, more than this PC has. Choose a smaller file.",
                             n=f"{hub.need_gb(f.size):.0f}"), "warning")  # fmt: skip
        elif state in ("partly", "cpu"):
            self.note.say(tr("It will run, but slowly on this PC. A smaller file is faster."), "warning")
        else:
            self.note.say(tr("{n} GB to download. Checked against Hugging Face's checksum.",
                             n=f"{f.size / 1e9:.1f}"), "info")  # fmt: skip

    def _open_page(self) -> None:
        repo = self._repo()
        if repo is not None:
            QDesktopServices.openUrl(QUrl(repo.page))

    # ---------------------------------------------------------------- use / download

    def _use(self) -> None:
        f = self._file()
        if f is None:
            return
        if not self.download_now:
            self.choice = f
            self.accept()
            return
        free = hardware.free_disk_gb(models_dir())
        if not hub.local_path(f.as_model()).exists() and free < f.size / 1e9 + 1:
            self.note.say(tr("Not enough free disk space. Free at least {n} GB and try again.",
                             n=f"{f.size / 1e9 + 1 - free:.0f}"), "error")  # fmt: skip
            return
        self.downloading = True
        self.cancel.clear()
        self.ok.setEnabled(False)
        self.results.setEnabled(False)
        self.file.setEnabled(False)
        self.bar.setValue(0)
        self.bar.show()
        bridge, cancel = self.bridge, self.cancel

        def work() -> None:
            try:
                path = hub.fetch(f, lambda d, tot: bridge.progress.emit(
                    d / max(tot, 1), f"{d / 1e9:.1f} / {tot / 1e9:.1f} GB"), cancel)  # fmt: skip
                bridge.done.emit(str(path), "")
            except Cancelled:
                bridge.done.emit("", "stopped")
            except Exception as e:
                bridge.done.emit("", str(e) or type(e).__name__)

        threading.Thread(target=work, daemon=True).start()

    def _progress(self, frac: float, detail: str) -> None:
        self.bar.setValue(int(frac * 1000))
        self.note.say(tr("Downloading… {d}", d=detail), "info")

    def _downloaded(self, path: str, err: str) -> None:
        self.downloading = False
        self.results.setEnabled(True)
        self.file.setEnabled(True)
        self.bar.hide()
        if err == "stopped":
            return
        if err:
            self.note.say(tr(err), "error")
            self.ok.setEnabled(True)
            return
        self.path = Path(path)
        self.accept()

    def reject(self) -> None:
        if self.downloading:  # stop the download first (it can be resumed later)
            self.cancel.set()
        super().reject()
