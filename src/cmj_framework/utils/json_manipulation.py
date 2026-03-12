# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import glob
import io
import warnings


# ============================================================
# Helpers
# ============================================================

def _ensure_directory(path):
    """
    Ensure that the parent directory of a file exists.
    Raises IOError if the directory does not exist.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        raise IOError(u"Das Verzeichnis existiert nicht: {}".format(directory))


# ============================================================
# LOAD JSON (single or multiple)
# ============================================================

def load_json(path):
    """
    Load a single JSON file and return its content as a dict.
    """
    if not os.path.isfile(path):
        raise IOError(u"JSON-Datei wurde nicht gefunden: {}".format(path))

    try:
        with io.open(path, mode='r', encoding='utf-8') as f:
            return json.load(f)
    except ValueError as e:
        raise ValueError(
            u"Fehler beim Parsen der JSON-Datei '{}': {}".format(path, e)
        )


def load_jsons_by_suffix(directory, suffix):
    """
    Load all JSON files in a directory that end with a given suffix.
    Returns a dict {filename: content}.
    """
    if not os.path.isdir(directory):
        raise IOError(u"Ordner wurde nicht gefunden: {}".format(directory))

    pattern = os.path.join(directory, '*' + suffix)
    files = glob.glob(pattern)

    if not files:
        warnings.warn(
            u"Keine JSON-Dateien mit der Endung '{}' gefunden.".format(suffix),
            RuntimeWarning
        )

    data = {}
    for path in files:
        try:
            data[os.path.basename(path)] = load_json(path)
        except Exception as e:
            warnings.warn(
                u"Fehler beim Laden der Datei '{}': {}".format(path, e),
                RuntimeWarning
            )

    return data


# ============================================================
# CREATE JSON
# ============================================================

def create_json(path, data, overwrite=False):
    """
    Create a JSON file with given data.
    """
    _ensure_directory(path)

    if os.path.exists(path) and not overwrite:
        raise IOError(u"Die JSON-Datei existiert bereits: {}".format(path))

    with io.open(path, mode='w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            indent=4,
            sort_keys=True,
            ensure_ascii=False
        )

    print(u"JSON-Datei erfolgreich erstellt: {}".format(path))


# ============================================================
# UPDATE JSON
# ============================================================

def update_json(path, update_fn):
    """
    Update a JSON file using a user-provided update function.
    """
    if not callable(update_fn):
        raise TypeError(u"update_fn muss eine Funktion sein.")

    data = load_json(path)

    update_fn(data)

    with io.open(path, mode='w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            indent=4,
            sort_keys=True,
            ensure_ascii=False
        )

    print(u"JSON-Datei wurde aktualisiert: {}".format(path))
    return data


# ============================================================
# DELETE JSON (single or multiple)
# ============================================================

def delete_json(path):
    """
    Delete a single JSON file.
    """
    if not os.path.isfile(path):
        raise IOError(
            u"JSON-Datei konnte nicht gelöscht werden (nicht gefunden): {}".format(path)
        )

    os.remove(path)
    print(u"JSON-Datei gelöscht: {}".format(path))


def delete_jsons_by_suffix(directory, suffix):
    """
    Delete all JSON files in a directory with a given suffix.
    Returns the number of deleted files.
    """
    if not os.path.isdir(directory):
        raise IOError(u"Ordner wurde nicht gefunden: {}".format(directory))

    pattern = os.path.join(directory, '*' + suffix)
    files = glob.glob(pattern)

    if not files:
        warnings.warn(
            u"Keine Dateien zum Löschen mit der Endung '{}' gefunden.".format(suffix),
            RuntimeWarning
        )
        return 0

    for path in files:
        try:
            os.remove(path)
        except Exception as e:
            warnings.warn(
                u"Fehler beim Löschen der Datei '{}': {}".format(path, e),
                RuntimeWarning
            )

    print(u"{} JSON-Datei(en) gelöscht mit Endung '{}'.".format(len(files), suffix))
    return len(files)