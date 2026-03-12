from pathlib import Path

import pytest


@pytest.fixture
def example_patient_name():
    return "Max, Mustermann"


@pytest.fixture
def example_session_date():
    return "01.01.2020"


@pytest.fixture
def example_project_root(tmp_path):
    project_root = tmp_path / "cmj_test_project"
    project_root.mkdir()
    return project_root


@pytest.fixture
def example_patient_session(example_project_root, example_patient_name, example_session_date):
    session_dir = example_project_root / example_patient_name / example_session_date
    session_dir.mkdir(parents=True)
    return session_dir