import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from cmj_framework.gui import manager


class FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)


class FakeSelectionModel:
    def __init__(self, indexes=None):
        self._indexes = indexes or []

    def selectedRows(self, column=0):
        return self._indexes


class FakeTree:
    def __init__(self):
        self._selection_model = FakeSelectionModel()

    def selectionModel(self):
        return self._selection_model

    def set_selection(self, indexes):
        self._selection_model = FakeSelectionModel(indexes)

    def setModel(self, model):
        self._model = model

    def setRootIndex(self, index):
        self._root_index = index

    def setHeaderHidden(self, value):
        pass

    def setAnimated(self, value):
        pass

    def setIndentation(self, value):
        pass

    def setSortingEnabled(self, value):
        pass

    def setAlternatingRowColors(self, value):
        pass

    def setDragDropMode(self, value):
        pass

    def setDefaultDropAction(self, value):
        pass

    def setSelectionMode(self, value):
        pass

    doubleClicked = FakeSignal()
    clicked = FakeSignal()


class FakeModel:
    def __init__(self, mapping=None, dirs=None):
        self.mapping = mapping or {}
        self.dirs = set(dirs or [])
        self.root_paths = []
        self.root_indexes = []

    def setRootPath(self, path):
        self.root_paths.append(path)

    def setNameFilters(self, filters):
        pass

    def setNameFilterDisables(self, value):
        pass

    def setFilter(self, value):
        pass

    def index(self, path):
        self.root_indexes.append(path)
        return path

    def isDir(self, index):
        return index in self.dirs

    def filePath(self, index):
        return self.mapping.get(index, index)


class FakeProgressBar:
    def __init__(self):
        self.value_ = 0
        self.format_ = ""

    def setMinimum(self, value):
        pass

    def setMaximum(self, value):
        pass

    def setValue(self, value):
        self.value_ = value

    def value(self):
        return self.value_

    def setTextVisible(self, value):
        pass

    def setFixedHeight(self, value):
        pass

    def setStyleSheet(self, value):
        pass

    def setFormat(self, text):
        self.format_ = text


class FakePreview:
    def __init__(self):
        self.text = ""

    def setPlainText(self, text):
        self.text = text

    def toPlainText(self):
        return self.text

    def setReadOnly(self, value):
        pass

    def setMinimumHeight(self, value):
        pass

    def setProperty(self, *args):
        pass

    def setLineWrapMode(self, mode):
        pass

    def document(self):
        return object()


class FakeDropZone(QWidget):
    def __init__(self):
        super().__init__()
        self.filesDropped = FakeSignal()

    def setAlignment(self, value):
        pass

    def setStyleSheet(self, value):
        pass


class FakeViewer:
    def __init__(self):
        self.trialRejected = FakeSignal()
        self.visible = False
        self.refresh_calls = 0
        self.select_calls = []
        self.shown = 0
        self.raised = 0
        self.activated = 0
        self.shutdown_called = False

    def setWindowTitle(self, title):
        self.title = title

    def resize(self, w, h):
        self.size = (w, h)

    def refresh_trials(self):
        self.refresh_calls += 1

    def select_trial(self, trial_name):
        self.select_calls.append(trial_name)

    def isVisible(self):
        return self.visible

    def show(self):
        self.visible = True
        self.shown += 1

    def raise_(self):
        self.raised += 1

    def activateWindow(self):
        self.activated += 1

    def shutdown(self):
        self.shutdown_called = True


class FakeWorker:
    def __init__(self, json_files):
        self.json_files = json_files
        self.finished = FakeSignal()
        self.progress = FakeSignal()
        self.log_signal = FakeSignal()
        self.started = False
        self.running = False
        self.wait_calls = []

    def start(self):
        self.started = True
        self.running = True

    def isRunning(self):
        return self.running

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        self.running = False


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def patched_view(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(manager.PathManager, "canonical_base_dir", lambda: str(tmp_path / "CMJ_manager"))
    monkeypatch.setattr(manager.PathManager, "install_demo_content_to_base_dir", lambda base: None)
    monkeypatch.setattr(manager, "JsonDropZone", FakeDropZone)
    monkeypatch.setattr(manager, "JsonSyntaxHighlighter", lambda doc: None)
    monkeypatch.setattr(manager, "CMJViewerWidget", FakeViewer)
    monkeypatch.setattr(manager, "ExtractionWindow", lambda parent=None: SimpleNamespace(jsonCreated=FakeSignal(), show=lambda: None))

    view = manager.ViconExporterView(parent=None)

    view.model = FakeModel()
    view.tree = FakeTree()
    view.progress_bar = FakeProgressBar()
    view.json_preview = FakePreview()
    view.parent_main_window = SimpleNamespace(
        append_log=lambda msg: getattr(view, "_logs", []).append(msg),
        open_export_view_with_file=lambda path: getattr(view, "_exports", []).append(path),
    )
    view._logs = []
    view._exports = []

    return view


def test_find_latest_combined_json_returns_none_for_no_candidates(monkeypatch):
    """
    Test that find_latest_combined_json returns None when no combined file can be found.
    """
    monkeypatch.setattr(manager, "find_cmj_session_dir_from_path", lambda path: None)

    result = manager.find_latest_combined_json(["a.json", "b.json"])

    assert result is None


def test_find_latest_combined_json_returns_latest_file(monkeypatch, tmp_path):
    """
    Test that find_latest_combined_json returns the newest combined JSON across sessions.
    """
    session_dir = tmp_path / "session"
    processed = session_dir / "processed"
    processed.mkdir(parents=True)

    old_file = processed / "old_combined.json"
    new_file = processed / "new_combined.json"

    old_file.write_text("{}", encoding="utf-8")
    new_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(manager, "find_cmj_session_dir_from_path", lambda path: str(session_dir))

    result = manager.find_latest_combined_json(["a.json"])

    assert Path(result) in {old_file, new_file}


def test_extract_session_key_returns_expected_value(patched_view):
    """
    Test that extract_session_key returns the first two chunks of the stem.
    """
    result = patched_view.extract_session_key("1_05112025_03_data_cmj.json")

    assert result == "1_05112025"


def test_extract_session_key_returns_none_for_short_name(patched_view):
    """
    Test that extract_session_key returns None for unexpected names.
    """
    result = patched_view.extract_session_key("invalid.json")

    assert result is None


def test_group_jsons_by_session_key_groups_files(patched_view):
    """
    Test that group_jsons_by_session_key groups files by their session key.
    """
    files = [
        "1_05112025_01_data_cmj.json",
        "1_05112025_02_data_cmj.json",
        "2_06112025_01_data_cmj.json",
        "invalid.json",
    ]

    result = patched_view.group_jsons_by_session_key(files)

    assert set(result.keys()) == {"1_05112025", "2_06112025", "__unknown__"}
    assert len(result["1_05112025"]) == 2
    assert result["__unknown__"] == ["invalid.json"]


def test_choose_best_session_prefers_biggest_group_then_latest_date(patched_view):
    """
    Test that choose_best_session prefers the largest group and then the latest date.
    """
    groups = {
        "1_05112025": ["a", "b"],
        "1_06112025": ["c", "d"],
        "1_04112025": ["e"],
    }

    best_key, kept, ignored = patched_view.choose_best_session(groups)

    assert best_key == "1_06112025"
    assert kept == ["c", "d"]
    assert set(ignored) == {"a", "b", "e"}


def test_choose_best_session_handles_empty_groups(patched_view):
    """
    Test that choose_best_session handles an empty grouping.
    """
    best_key, kept, ignored = patched_view.choose_best_session({})

    assert best_key is None
    assert kept == []
    assert ignored == []


def test_collect_json_from_paths_collects_files_and_dirs(patched_view, tmp_path):
    """
    Test that collect_json_from_paths collects JSON files directly and recursively from directories.
    """
    direct_file = tmp_path / "1_05112025_01_data_cmj.json"
    direct_file.write_text("{}", encoding="utf-8")

    folder = tmp_path / "folder"
    folder.mkdir()
    nested = folder / "2_05112025_01_data_cmj.json"
    nested.write_text("{}", encoding="utf-8")
    other = folder / "other.json"
    other.write_text("{}", encoding="utf-8")

    result = patched_view.collect_json_from_paths([str(direct_file), str(folder)])

    assert str(direct_file) in result
    assert str(nested) in result
    assert str(other) not in result


def test_preview_json_file_pretty_prints_json(patched_view, tmp_path):
    """
    Test that preview_json_file loads and pretty-prints JSON content.
    """
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps({"a": 1}), encoding="utf-8")

    patched_view.preview_json_file(str(json_file))

    assert '"a": 1' in patched_view.json_preview.toPlainText()
    assert patched_view._last_preview_path == str(json_file)


def test_preview_json_file_shows_error_text_on_failure(patched_view, tmp_path):
    """
    Test that preview_json_file shows an error message when reading fails.
    """
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{bad json}", encoding="utf-8")

    patched_view.preview_json_file(str(bad_file))

    assert "Fehler beim Lesen der Datei" in patched_view.json_preview.toPlainText()
    assert patched_view._last_preview_path is None


def test_on_jsons_dropped_shows_message_when_no_json_found(patched_view, monkeypatch):
    """
    Test that on_jsons_dropped shows a message when no CMJ JSON files are found.
    """
    monkeypatch.setattr(patched_view, "collect_json_from_paths", lambda paths: [])

    patched_view.on_jsons_dropped(["anything"])

    assert "Keine CMJ-JSON-Dateien gefunden." in patched_view.json_preview.toPlainText()


def test_on_jsons_dropped_groups_and_processes_best_session(patched_view, monkeypatch):
    """
    Test that on_jsons_dropped keeps the best session and starts processing it.
    """
    kept = [
        "/tmp/1_05112025_01_data_cmj.json",
        "/tmp/1_05112025_02_data_cmj.json",
    ]
    groups = {
        "1_05112025": kept,
        "2_05112025": ["/tmp/2_05112025_01_data_cmj.json"],
    }

    processed = {"files": None}

    monkeypatch.setattr(patched_view, "collect_json_from_paths", lambda paths: kept + groups["2_05112025"])
    monkeypatch.setattr(patched_view, "group_jsons_by_session_key", lambda files: groups)
    monkeypatch.setattr(
        patched_view,
        "choose_best_session",
        lambda groups: ("1_05112025", kept, groups["2_05112025"]),
    )
    monkeypatch.setattr(patched_view, "process_data", lambda files: processed.__setitem__("files", files))

    patched_view.on_jsons_dropped(["dummy"])

    assert patched_view.selected_jsons == kept
    assert processed["files"] == kept
    assert "Gewählter Sitzungsschlüssel: 1_05112025" in patched_view.json_preview.toPlainText()
    assert len(patched_view._logs) == 1


def test_preview_selected_file_ignores_directories_and_non_json(patched_view):
    """
    Test that preview_selected_file ignores directories and non-JSON files.
    """
    patched_view.model = FakeModel(mapping={"dir": "/tmp/dir", "txt": "/tmp/file.txt"}, dirs={"dir"})

    called = {"count": 0}
    patched_view.preview_json_file = lambda path: called.__setitem__("count", called["count"] + 1)

    patched_view.preview_selected_file("dir")
    patched_view.preview_selected_file("txt")

    assert called["count"] == 0


def test_preview_selected_file_previews_json(patched_view):
    """
    Test that preview_selected_file forwards valid JSON files to preview_json_file.
    """
    patched_view.model = FakeModel(mapping={"json": "/tmp/file.json"})

    captured = {"path": None}
    patched_view.preview_json_file = lambda path: captured.__setitem__("path", path)

    patched_view.preview_selected_file("json")

    assert captured["path"] == "/tmp/file.json"


def test_process_data_uses_selected_tree_items_when_none_provided(patched_view, monkeypatch):
    """
    Test that process_data uses the selected tree items when no explicit paths are given.
    """
    patched_view.model = FakeModel(mapping={"a": "/tmp/a.json", "b": "/tmp/b.json"})
    patched_view.tree = FakeTree()
    patched_view.tree.set_selection(["a", "b"])

    created = {"worker_files": None}

    def fake_worker(files):
        created["worker_files"] = files
        return FakeWorker(files)

    monkeypatch.setattr(manager, "ProcessingWorker", fake_worker)

    patched_view.process_data()

    assert created["worker_files"] == ["/tmp/a.json", "/tmp/b.json"]
    assert patched_view.worker.started is True
    assert patched_view.progress_bar.format_ == "CMJ-Sitzung wird verarbeitet…"


def test_process_data_returns_when_no_paths(patched_view):
    """
    Test that process_data returns early when there are no input paths.
    """
    patched_view.tree = FakeTree()
    patched_view.model = FakeModel()

    patched_view.process_data()

    assert patched_view.worker is None


def test_process_data_logs_when_worker_already_running(patched_view):
    """
    Test that process_data logs a message when processing is already running.
    """
    worker = FakeWorker(["a.json"])
    worker.running = True
    patched_view.worker = worker

    patched_view.process_data(["/tmp/a.json"])

    assert any("Verarbeitung läuft bereits" in msg for msg in patched_view._logs)


def test_cleanup_worker_resets_worker_reference(patched_view):
    """
    Test that _cleanup_worker resets the worker reference.
    """
    patched_view.worker = FakeWorker(["a.json"])

    patched_view._cleanup_worker({})

    assert patched_view.worker is None


def test_shutdown_waits_for_running_worker(patched_view):
    """
    Test that shutdown waits for the running worker and then clears it.
    """
    worker = FakeWorker(["a.json"])
    worker.running = True
    patched_view.worker = worker

    patched_view.shutdown()

    assert worker.wait_calls == [3000]
    assert patched_view.worker is None


def test_on_processing_finished_updates_preview_and_opens_viewer(monkeypatch, patched_view):
    """
    Test that on_processing_finished updates preview, opens viewer, and forwards combined JSON to export.
    """
    results = {"Trial_01": {"jump_height": 20.0}}
    viewer = FakeViewer()

    monkeypatch.setattr(patched_view, "_open_cmj_viewer_window", lambda refresh=True: viewer)
    monkeypatch.setattr(manager, "find_latest_combined_json", lambda jsons: "/tmp/combined.json")

    patched_view.selected_jsons = ["/tmp/a.json"]
    patched_view.on_processing_finished(results)

    assert patched_view.progress_bar.value_ == 100
    assert patched_view.progress_bar.format_ == "✔ Fertig"
    assert "Trial_01" in patched_view.json_preview.toPlainText()
    assert viewer.select_calls == ["Trial_01"]
    assert patched_view._exports == ["/tmp/combined.json"]


def test_on_processing_finished_logs_when_no_results(patched_view):
    """
    Test that on_processing_finished logs when no trials were processed.
    """
    patched_view.selected_jsons = []

    patched_view.on_processing_finished({})

    assert any("Keine verarbeitbaren Trials" in msg for msg in patched_view._logs)


def test_visualize_data_opens_selected_trial_from_tree(monkeypatch, patched_view):
    """
    Test that visualize_data opens the selected trial from the tree when one is selected.
    """
    patched_view.tree = FakeTree()
    patched_view.tree.set_selection(["idx"])

    called = {"index": None}
    monkeypatch.setattr(patched_view, "open_trial_from_tree", lambda index: called.__setitem__("index", index))

    patched_view.visualize_data()

    assert called["index"] == "idx"


def test_visualize_data_logs_when_no_trials_available(monkeypatch, patched_view):
    """
    Test that visualize_data reports when no trials are in memory.
    """
    patched_view.tree = FakeTree()

    monkeypatch.setattr(manager.TempProcessedData, "list_trials", lambda: [])

    patched_view.visualize_data()

    assert "Keine Trials im Speicher" in patched_view.json_preview.toPlainText()
    assert any("Keine Trials im Speicher" in msg for msg in patched_view._logs)


def test_visualize_data_opens_existing_trial(monkeypatch, patched_view):
    """
    Test that visualize_data opens the first in-memory trial when no tree selection exists.
    """
    patched_view.tree = FakeTree()
    viewer = FakeViewer()

    monkeypatch.setattr(manager.TempProcessedData, "list_trials", lambda: ["Trial_01"])
    monkeypatch.setattr(patched_view, "_open_cmj_viewer_window", lambda refresh=False: viewer)

    patched_view.visualize_data()

    assert viewer.select_calls == ["Trial_01"]


def test_open_cmj_viewer_window_reuses_existing_viewer(patched_view):
    """
    Test that _open_cmj_viewer_window reuses the existing viewer instance.
    """
    viewer = patched_view._open_cmj_viewer_window(refresh=True)

    assert isinstance(viewer, FakeViewer)
    assert viewer.refresh_calls == 1
    assert viewer.shown == 1

    viewer2 = patched_view._open_cmj_viewer_window(refresh=False)

    assert viewer2 is viewer
    assert viewer2.raised == 2
    assert viewer2.activated == 2


def test_open_trial_from_tree_loads_trial_if_needed(monkeypatch, patched_view):
    """
    Test that open_trial_from_tree loads a trial if it is not already in memory.
    """
    patched_view.model = FakeModel(mapping={"idx": "/tmp/1_05112025_03_data_cmj.json"})
    viewer = FakeViewer()
    loaded = {"args": None}

    monkeypatch.setattr(manager.TempProcessedData, "list_trials", lambda: [])
    monkeypatch.setattr(
        manager.TempProcessedData,
        "load",
        lambda file_path, trial_name, log_cb=None, force_reload=False: loaded.__setitem__(
            "args", (file_path, trial_name, force_reload)
        ),
    )
    monkeypatch.setattr(patched_view, "_open_cmj_viewer_window", lambda refresh=False: viewer)

    patched_view.open_trial_from_tree("idx")

    assert loaded["args"] == ("/tmp/1_05112025_03_data_cmj.json", "Trial_03", False)
    assert viewer.select_calls == ["Trial_03"]


def test_open_trial_from_tree_logs_when_loading_fails(monkeypatch, patched_view):
    """
    Test that open_trial_from_tree logs and previews an error when loading fails.
    """
    patched_view.model = FakeModel(mapping={"idx": "/tmp/1_05112025_03_data_cmj.json"})

    monkeypatch.setattr(manager.TempProcessedData, "list_trials", lambda: [])

    def raise_load(*args, **kwargs):
        raise RuntimeError("cannot load trial")

    monkeypatch.setattr(manager.TempProcessedData, "load", raise_load)

    patched_view.open_trial_from_tree("idx")

    assert "konnte nicht geladen werden" in patched_view.json_preview.toPlainText()
    assert any("konnte nicht geladen werden" in msg for msg in patched_view._logs)


def test_on_trial_rejected_opens_rejected_dir(monkeypatch, patched_view, tmp_path):
    """
    Test that on_trial_rejected opens the rejected folder of the session when available.
    """
    session_dir = tmp_path / "session"
    rejected_dir = session_dir / "rejected"
    rejected_dir.mkdir(parents=True)

    monkeypatch.setattr(manager, "find_cmj_session_dir_from_path", lambda path: str(session_dir))

    patched_view.on_trial_rejected("/tmp/old.json")

    assert patched_view.model.root_paths[-1] == str(rejected_dir)
    assert any("Rejected-Ordner geöffnet" in msg for msg in patched_view._logs)