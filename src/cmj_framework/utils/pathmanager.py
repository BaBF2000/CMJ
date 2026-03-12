import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import json 


class PathManager:
    """
    Centralized path manager for CMJ_manager.
    """

    def __init__(self, patient_name: str, session_date: Optional[str] = None):
        self.patient_name = self.sanitize(patient_name)

        if session_date is None:
            session_date = datetime.now().strftime("%d.%m.%Y")
        self.session_date = self.sanitize(session_date)

        self.base_dir = self.get_base_dir()

        # Ensure CMJ_manager exists
        os.makedirs(self.base_dir, exist_ok=True)

        # Install bundled demo patient if missing
        self.install_demo_content_if_missing()

        self.patient_dir = os.path.join(self.base_dir, self.patient_name)
        self.session_dir = os.path.join(self.patient_dir, self.session_date)

        self.raw_dir = os.path.join(self.session_dir, "raw_data")
        self.processed_dir = os.path.join(self.session_dir, "processed")
        self.reports_dir = os.path.join(self.session_dir, "reports")
        self.rejected_dir = os.path.join(self.session_dir, "rejected")

        self.ensure_dirs_exist()

    # -------------------------
    # Canonical base directory
    # -------------------------
    @staticmethod
    def canonical_base_dir() -> str:
        home = os.path.expanduser("~")
        documents = os.path.join(home, "Documents")
        return os.path.join(documents, "CMJ_manager")

    @staticmethod
    def is_frozen() -> bool:
        return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

    @classmethod
    def project_root_dir(cls) -> Path:
        """
        Resolve project root directory.

        Current file:
        cmj_utils/utils_pathmanager.py
        -> parents[1] = project root
        """
        if cls.is_frozen():
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parents[3]

    @classmethod
    def resource_path(cls, relative_path: str) -> Path:
        return cls.project_root_dir() / relative_path

    def get_base_dir(self) -> str:
        return self.canonical_base_dir()

    # -------------------------
    # Demo content
    # -------------------------
    @classmethod
    def demo_source_dir(cls) -> Path:
        """
        Bundled demo patient location.

        Expected:
        gui/assets/Max, Mustermann
        """
        return cls.resource_path("examples/Max, Mustermann")

    @classmethod
    def install_demo_content_to_base_dir(cls, base_dir: str) -> None:
        """
        Copy bundled demo patient into CMJ_manager if missing.
        """
        src_patient_dir = cls.demo_source_dir()
        if not src_patient_dir.exists():
            return

        dst_patient_dir = Path(base_dir) / src_patient_dir.name

        if dst_patient_dir.exists():
            return

        shutil.copytree(src_patient_dir, dst_patient_dir)

    def install_demo_content_if_missing(self) -> None:
        self.install_demo_content_to_base_dir(self.base_dir)

     

    @classmethod
    def from_extracted_json(cls, json_path: str) -> "PathManager":
        """
        Build a PathManager from an extracted CMJ JSON file.

        Resolution order:
        1) user_info.name / user_info.trial_date inside JSON
        2) fallback from canonical folder structure
        3) safe defaults
        """
        patient_name = "Unknown_Patient"
        session_date = datetime.now().strftime("%d.%m.%Y")

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                user_info = data.get("user_info", {})
                if isinstance(user_info, dict):
                    patient_name = user_info.get("name") or patient_name
                    session_date = user_info.get("trial_date") or session_date
        except Exception:
            pass

        try:
            path_obj = Path(json_path).resolve()
            if patient_name == "Unknown_Patient" and len(path_obj.parents) >= 3:
                patient_name = path_obj.parents[2].name or patient_name
            if not session_date and len(path_obj.parents) >= 2:
                session_date = path_obj.parents[1].name or session_date
        except Exception:
            pass

        return cls(patient_name=patient_name, session_date=session_date)

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def sanitize(name: str) -> str:
        for bad in ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]:
            name = name.replace(bad, "_")
        return name.strip()

    def ensure_dirs_exist(self) -> None:
        for directory in [
            self.base_dir,
            self.patient_dir,
            self.session_dir,
            self.raw_dir,
            self.processed_dir,
            self.reports_dir,
            self.rejected_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

    # -------------------------
    # Public API
    # -------------------------
    def raw_file(self, filename: str) -> str:
        return os.path.join(self.raw_dir, filename)

    def processed_file(self, filename: str) -> str:
        return os.path.join(self.processed_dir, filename)

    def report_file(self, filename: str) -> str:
        return os.path.join(self.reports_dir, filename)

    def rejected_file(self, filename: str) -> str:
        return os.path.join(self.rejected_dir, filename)

    def summary(self) -> Dict[str, str]:
        return {
            "base": self.base_dir,
            "patient": self.patient_dir,
            "session": self.session_dir,
            "raw": self.raw_dir,
            "processed": self.processed_dir,
            "reports": self.reports_dir,
            "rejected": self.rejected_dir,
        }
    
