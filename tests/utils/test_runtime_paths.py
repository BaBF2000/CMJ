from pathlib import Path
import sys

import pytest

from cmj_framework.utils import runtime_paths


def test_is_frozen_returns_false_by_default(monkeypatch):
    """
    Test that is_frozen returns False when sys.frozen is not set.
    """
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert runtime_paths.is_frozen() is False


def test_is_frozen_returns_true_when_attributes_exist(monkeypatch):
    """
    Test that is_frozen returns True when running in bundled mode.
    """
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/tmp/fake_meipass", raising=False)

    assert runtime_paths.is_frozen() is True


def test_bundle_root_dir_in_dev_mode(monkeypatch):
    """
    Test that bundle_root_dir returns the project root in development mode.
    """
    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: False)

    root = runtime_paths.bundle_root_dir()

    assert isinstance(root, Path)
    assert root.exists()


def test_bundle_root_dir_in_frozen_mode(monkeypatch):
    """
    Test that bundle_root_dir returns the executable directory in frozen mode.
    """
    fake_executable = Path.cwd() / "fake_app" / "app.exe"

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "executable", str(fake_executable))

    root = runtime_paths.bundle_root_dir()

    assert root == fake_executable.parent


def test_config_dir_dev_mode(monkeypatch):
    """
    Test config_dir in development mode.
    """
    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: False)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: Path("/project"))

    result = runtime_paths.config_dir()

    assert result == Path("/project/config")


def test_config_file_returns_expected_path(monkeypatch):
    """
    Test config_file helper.
    """
    monkeypatch.setattr(runtime_paths, "config_dir", lambda: Path("/project/config"))

    result = runtime_paths.config_file("settings.json")

    assert result == Path("/project/config/settings.json")


def test_documentation_dir_dev_mode(monkeypatch):
    """
    Test documentation_dir in development mode.
    """
    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: False)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: Path("/project"))

    result = runtime_paths.documentation_dir()

    assert result == Path("/project/documentation")


def test_documentation_file_returns_expected_path(monkeypatch):
    """
    Test documentation_file helper.
    """
    monkeypatch.setattr(runtime_paths, "documentation_dir", lambda: Path("/project/documentation"))

    result = runtime_paths.documentation_file("doc.html")

    assert result == Path("/project/documentation/doc.html")


def test_gui_asset_returns_expected_path(monkeypatch):
    """
    Test gui_asset path construction.
    """
    monkeypatch.setattr(runtime_paths, "gui_assets_dir", lambda: Path("/project/assets"))

    result = runtime_paths.gui_asset("icons", "logo.png")

    assert result == Path("/project/assets/icons/logo.png")


def test_export_resource_file_returns_expected_path(monkeypatch):
    """
    Test export_resource_file helper.
    """
    monkeypatch.setattr(runtime_paths, "export_resources_dir", lambda: Path("/project/export/resources"))

    result = runtime_paths.export_resource_file("template.docx")

    assert result == Path("/project/export/resources/template.docx")


def test_app_data_dir_uses_pathmanager(monkeypatch):
    """
    Test that app_data_dir uses PathManager canonical base directory.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.runtime_paths.PathManager.canonical_base_dir",
        lambda: "/data/cmj"
    )

    result = runtime_paths.app_data_dir()

    assert result == Path("/data/cmj")


def test_export_user_resources_dir(monkeypatch):
    """
    Test export_user_resources_dir path.
    """
    monkeypatch.setattr(runtime_paths, "app_data_dir", lambda: Path("/data/cmj"))

    result = runtime_paths.export_user_resources_dir()

    assert result == Path("/data/cmj/export_resources")


def test_log_dir(monkeypatch):
    """
    Test log_dir path.
    """
    monkeypatch.setattr(runtime_paths, "app_data_dir", lambda: Path("/data/cmj"))

    result = runtime_paths.log_dir()

    assert result == Path("/data/cmj/Log")


def test_ensure_dir_creates_directory(tmp_path):
    """
    Test that ensure_dir creates the directory if it does not exist.
    """
    path = tmp_path / "new_dir"

    result = runtime_paths.ensure_dir(path)

    assert path.exists()
    assert path.is_dir()
    assert result == path

def test_config_dir_returns_internal_config_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that config_dir prefers the _internal/config folder in frozen mode.
    """
    bundle_root = tmp_path / "app"
    internal_config = bundle_root / "_internal" / "config"
    internal_config.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.config_dir()

    assert result == internal_config


def test_config_dir_falls_back_to_bundle_config_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that config_dir falls back to bundle_root/config if _internal/config is missing.
    """
    bundle_root = tmp_path / "app"
    bundle_root.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.config_dir()

    assert result == bundle_root / "config"


def test_documentation_dir_returns_internal_documentation_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that documentation_dir prefers the _internal/documentation folder in frozen mode.
    """
    bundle_root = tmp_path / "app"
    internal_docs = bundle_root / "_internal" / "documentation"
    internal_docs.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.documentation_dir()

    assert result == internal_docs


def test_documentation_dir_falls_back_to_bundle_documentation_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that documentation_dir falls back to bundle_root/documentation if _internal/documentation is missing.
    """
    bundle_root = tmp_path / "app"
    bundle_root.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.documentation_dir()

    assert result == bundle_root / "documentation"


def test_gui_assets_dir_returns_internal_assets_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that gui_assets_dir prefers the internal GUI assets folder in frozen mode.
    """
    bundle_root = tmp_path / "app"
    internal_assets = bundle_root / "_internal" / "src" / "cmj_framework" / "gui" / "assets"
    internal_assets.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.gui_assets_dir()

    assert result == internal_assets


def test_gui_assets_dir_falls_back_to_bundle_assets_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that gui_assets_dir falls back to bundle_root/src/cmj_framework/gui/assets if internal assets are missing.
    """
    bundle_root = tmp_path / "app"
    bundle_root.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.gui_assets_dir()

    assert result == bundle_root / "src" / "cmj_framework" / "gui" / "assets"


def test_export_resources_dir_returns_internal_resources_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that export_resources_dir prefers the internal export resources folder in frozen mode.
    """
    bundle_root = tmp_path / "app"
    internal_resources = (
        bundle_root / "_internal" / "src" / "cmj_framework" / "export" / "resources"
    )
    internal_resources.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.export_resources_dir()

    assert result == internal_resources


def test_export_resources_dir_falls_back_to_bundle_resources_in_frozen_mode(monkeypatch, tmp_path):
    """
    Test that export_resources_dir falls back to bundle_root/src/cmj_framework/export/resources if internal resources are missing.
    """
    bundle_root = tmp_path / "app"
    bundle_root.mkdir(parents=True)

    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime_paths, "bundle_root_dir", lambda: bundle_root)

    result = runtime_paths.export_resources_dir()

    assert result == bundle_root / "src" / "cmj_framework" / "export" / "resources"


def test_ensure_dir_creates_directory_and_returns_same_path(tmp_path):
    """
    Test that ensure_dir creates the directory and returns the same path.
    """
    target = tmp_path / "a" / "b" / "c"

    result = runtime_paths.ensure_dir(target)

    assert result == target
    assert target.exists()
    assert target.is_dir()