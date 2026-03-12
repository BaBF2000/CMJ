import json
from pathlib import Path

import pytest

from cmj_framework.export import word_report


def make_minimal_combined_json(path: Path):
    """
    Create a minimal valid *_combined.json file for report generation.
    """
    data = {
        "patient_name": "Max Mustermann",
        "session_date": "01.01.2020",
        "metrics_by_trial": {
            "Trial_1": {
                "jump_height": 25.0,
                "weight": 700.0,
                "Takeoff_velocity": 2.5,
                "RSI_modified": 1.2,
                "PhaseDurations": {
                    "Time_to_takeoff": 0.55,
                    "Eccentric_duration": 0.30,
                    "Concentric_duration": 0.25,
                },
                "RFD": {
                    "RFD_max": 15000.0,
                    "RFD_0_100ms": 8000.0,
                    "RFD_0_200ms": 9000.0,
                },
                "ConcentricPower": {
                    "Power_mean": 1200.0,
                    "Power_peak": 1800.0,
                },
                "LandingMetrics": {
                    "LoadingRate_peak": 11000.0,
                    "PeakLandingForce": 2100.0,
                },
                "PeakForce": {
                    "braking": 1500.0,
                    "deceleration": 1600.0,
                    "propulsion": 1700.0,
                    "landing": 2000.0,
                },
                "Asymmetry": {
                    "Impulse": {
                        "Takeoff": {
                            "contribution_L_percent": 49.0,
                            "contribution_R_percent": 51.0,
                            "delta_percent": 2.0,
                        },
                        "Landing": {
                            "contribution_L_percent": 48.0,
                            "contribution_R_percent": 52.0,
                            "delta_percent": 4.0,
                        },
                    },
                    "Power": {
                        "Takeoff": {
                            "contribution_L_percent": 50.0,
                            "contribution_R_percent": 50.0,
                            "delta_percent": 0.0,
                        },
                        "Landing": {
                            "contribution_L_percent": 47.0,
                            "contribution_R_percent": 53.0,
                            "delta_percent": 6.0,
                        },
                    },
                },
            }
        },
    }

    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def patched_report_environment(tmp_path, monkeypatch):
    """
    Patch external resources and helper functions so the report can be generated
    in a controlled test environment.
    """
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()

    logo_path = resource_dir / "Kopf Labor.jpg"
    graph_path = resource_dir / "munster_graph.png"
    logo_path.write_bytes(b"fake-image")
    graph_path.write_bytes(b"fake-image")

    phases_md = tmp_path / "phases.md"
    parameter_md = tmp_path / "parameter.md"
    phases_md.write_text("# Phases\n\nSome content.\n", encoding="utf-8")
    parameter_md.write_text("# Parameters\n\nSome content.\n", encoding="utf-8")

    monkeypatch.setattr(
        word_report,
        "export_resource_file",
        lambda filename: logo_path if filename == "Kopf Labor.jpg" else graph_path,
    )
    monkeypatch.setattr(word_report, "ensure_editable_markdown_file", lambda filename: str(phases_md if filename == "phases.md" else parameter_md))
    monkeypatch.setattr(word_report, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(word_report, "add_patient_date_row", lambda *args, **kwargs: None)
    monkeypatch.setattr(word_report, "insert_markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(word_report, "create_classic_table_with_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(word_report, "create_table1_look_compact", lambda *args, **kwargs: None)
    monkeypatch.setattr(word_report, "format_A4", lambda *args, **kwargs: None)
    monkeypatch.setattr(word_report, "log", lambda *args, **kwargs: None)

    class FakeRun:
        def add_picture(self, *args, **kwargs):
            return None

    class FakeParagraph:
        def __init__(self):
            self.runs = []

        def add_run(self):
            run = FakeRun()
            self.runs.append(run)
            return run

    class FakeHeader:
        def __init__(self):
            self.paragraphs = [FakeParagraph()]

    class FakeSection:
        def __init__(self):
            self.page_width = type("Dim", (), {"inches": 8.27})()
            self.left_margin = type("Dim", (), {"inches": 1.0})()
            self.right_margin = type("Dim", (), {"inches": 1.0})()
            self.top_margin = None
            self.bottom_margin = None
            self.orientation = None
            self.different_first_page_header_footer = False
            self.first_page_header = FakeHeader()

    class FakeStyleFont:
        def __init__(self):
            self.name = None
            self.size = None

    class FakeStyle:
        def __init__(self):
            self.font = FakeStyleFont()

    class FakeDocument:
        def __init__(self):
            self.styles = {"Normal": FakeStyle()}
            self.sections = [FakeSection()]
            self.saved_path = None

        def add_heading(self, *args, **kwargs):
            return None

        def add_picture(self, *args, **kwargs):
            return None

        def add_section(self):
            section = FakeSection()
            self.sections.append(section)
            return section

        def add_page_break(self):
            return None

        def save(self, path):
            self.saved_path = path
            Path(path).write_bytes(b"fake-docx")

    fake_doc = FakeDocument()
    monkeypatch.setattr(word_report, "Document", lambda: fake_doc)

    return {
        "tmp_path": tmp_path,
        "fake_doc": fake_doc,
    }


def test_report_raises_when_metrics_by_trial_is_missing(tmp_path):
    """
    Test that Report raises when metrics_by_trial is missing or empty.
    """
    json_path = tmp_path / "bad_combined.json"
    json_path.write_text(json.dumps({"metrics_by_trial": {}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metrics_by_trial"):
        word_report.Report(str(json_path))


def test_report_raises_when_no_parameters_are_selected(tmp_path, patched_report_environment):
    """
    Test that Report raises when enabled_parameters filters out all performance rows.
    """
    json_path = tmp_path / "session" / "processed" / "trial_combined.json"
    json_path.parent.mkdir(parents=True)
    make_minimal_combined_json(json_path)

    with pytest.raises(RuntimeError, match="Keine Performance-Parameter ausgewählt"):
        word_report.Report(
            str(json_path),
            enabled_parameters=["Nicht vorhandener Parameter"],
        )


def test_report_creates_docx_in_reports_directory(tmp_path, patched_report_environment):
    """
    Test that Report creates a .docx file in the session reports directory.
    """
    json_path = tmp_path / "Max Mustermann" / "01.01.2020" / "processed" / "trial_combined.json"
    json_path.parent.mkdir(parents=True)
    make_minimal_combined_json(json_path)

    output_path = word_report.Report(str(json_path))

    output_path = Path(output_path)

    assert output_path.exists()
    assert output_path.suffix == ".docx"
    assert output_path.parent.name == "reports"


def test_report_uses_patient_override_in_output_filename(tmp_path, patched_report_environment):
    """
    Test that patient_override is used in the generated output filename.
    """
    json_path = tmp_path / "Max Mustermann" / "01.01.2020" / "processed" / "trial_combined.json"
    json_path.parent.mkdir(parents=True)
    make_minimal_combined_json(json_path)

    output_path = word_report.Report(
        str(json_path),
        patient_override="Override Patient",
    )

    assert "Override Patient_01.01.2020_CMJ_Bericht.docx" in Path(output_path).name


def test_report_sanitizes_output_filename(tmp_path, patched_report_environment):
    """
    Test that invalid filename characters are sanitized in the output filename.
    """
    json_path = tmp_path / "Max Mustermann" / "01.01.2020" / "processed" / "trial_combined.json"
    json_path.parent.mkdir(parents=True)
    make_minimal_combined_json(json_path)

    output_path = word_report.Report(
        str(json_path),
        patient_override='Bad:/\\*?"<>|Name',
    )

    assert Path(output_path).name == "Bad_________Name_01.01.2020_CMJ_Bericht.docx"