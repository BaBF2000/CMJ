# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import json
import codecs
from pathlib import Path

from viconnexusapi import ViconNexus

# Ensure this folder is importable (Nexus / Python 2 environments)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Ensure src root is importable
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cmj_framework.vicon_data_retrieval.helpers_extraction import (
    UserDataExtractor,
    DataCache,
    JSONExporter,
)
from cmj_framework.utils.pathmanager import PathManager


def project_root_dir():
    """
    Resolve project root from:
    src/cmj_framework/vicon_data_retrieval/extraction.py
    -> project root
    """
    return Path(__file__).resolve().parents[3]


def config_path():
    """Return path to nexus_data_retrieval_config.json."""
    return project_root_dir() / "config" / "nexus_data_retrieval_config.json"


if __name__ == "__main__":
    vicon = ViconNexus.ViconNexus()

    cfg_path = config_path()
    with codecs.open(str(cfg_path), "r", "utf-8") as f:
        config = json.load(f)

    user_data = UserDataExtractor(vicon)
    data_cache = DataCache(vicon, config, axes=['z'])

    path_manager = PathManager(
        patient_name=user_data.name,
        session_date=user_data.trial_date,
    )

    exporter = JSONExporter(vicon)
    json_path = exporter.save(
        data_cache,
        user_data,
        path_manager,
        new_path=True,
    )

    print(u"JSON_PATH::{}".format(json_path))