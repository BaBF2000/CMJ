# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import json
import warnings
import numpy as np
from collections import OrderedDict
from datetime import datetime

class UserDataExtractor(object):
    """Extract user/patient and trial data from folder, Vicon trial, and metadata."""

    def __init__(self, vicon):
        self.vicon = vicon
        self.trial_path, self.trial_name = self.vicon.GetTrialName()
        self.name = self.extract_name()
        self.trial_date = self.extract_trial_date()
        self.framecount = self.vicon.GetFrameCount()
        self.framerate = self.vicon.GetFrameRate()
        self.platerate = self.vicon.GetDeviceDetails(1)[2]

    def extract_name(self):
        """Format patient name: 'Last, First Middle...'."""
        try:
            patient_folder = os.path.basename(os.path.dirname(os.path.dirname(self.trial_path)))
            parts = patient_folder.split()
            if len(parts) >= 2:
                return parts[0] + ", " + " ".join(parts[1:])
        except Exception:
            pass
        return "Unknown Patient"

    def extract_trial_date(self):
        """Parse trial date from trial name assuming format 'XXX_DDMMYYYY_YYY'."""
        try:
            date_str = self.trial_name.split('_')[1]
            date_iso = datetime.strptime(date_str, "%d%m%Y")
            return date_iso.strftime("%d.%m.%Y")
        except Exception:
            return None

    def as_dict(self):
        """Return all user/trial data as a dictionary."""
        return {
            'name': self.name,
            'trial_date': self.trial_date,
            'framecount': self.framecount,
            'framerate': self.framerate,
            'platerate': self.platerate
        }


class ForcePlateData(object):
    """Handle force plate data extraction (raw)."""

    def __init__(self, vicon, config, axes=None):
        self.vicon = vicon
        self.config = config
        self.axes = axes if axes else ['z']
        self.frame_rate = float(vicon.GetFrameRate())
        self.plate_rate = float(self.get_plate_rate())
        self.samples_per_frame = max(1, int(self.plate_rate / self.frame_rate))
        self.total_samples = int(vicon.GetFrameCount() * self.samples_per_frame)

        self.forceplate_aliases = config.get("forceplates", {})
        self.forces = {}

    def get_plate_rate(self):
        """Return the sample rate of the first force plate, fallback to frame rate."""
        try:
            # Vicon returns (name, type, rate, outputs, ...)
            return self.vicon.GetDeviceDetails(1)[2]
        except Exception:
            return int(self.frame_rate)

    def load_forces(self):
        """Load forces for all configured force plates."""
        axis_map = {'x': 'Fx', 'y': 'Fy', 'z': 'Fz'}

        self.forces = {}
        for dev in self.vicon.GetDeviceIDs():
            try:
                details = self.vicon.GetDeviceDetails(dev)
                name = details[0]
                dtype = details[1]
            except Exception:
                continue

            if dtype != 'ForcePlate' or name not in self.forceplate_aliases:
                continue

            alias = self.forceplate_aliases.get(name, name)
            self.forces[alias] = {}

            try:
                outID = self.vicon.GetDeviceOutputIDFromName(dev, 'Force')
            except Exception:
                warnings.warn(u"Kraftplattform {}: Output 'Force' fehlt".format(alias))
                outID = None

            for axis in self.axes:
                if axis not in axis_map or outID is None:
                    zero = np.zeros(self.total_samples, dtype=float)
                    self.forces[alias][axis] = {'raw': zero}
                    continue

                try:
                    chID = self.vicon.GetDeviceChannelIDFromName(dev, outID, axis_map[axis])
                    f_raw, _ = self.vicon.GetDeviceChannelGlobal(dev, outID, chID)[:2]
                    f_raw = -np.asarray(f_raw, dtype=float)

                    # Ensure expected length (some APIs can return shorter arrays)
                    if f_raw.size != self.total_samples:
                        tmp = np.zeros(self.total_samples, dtype=float)
                        n = min(tmp.size, f_raw.size)
                        tmp[:n] = f_raw[:n]
                        f_raw = tmp

                    self.forces[alias][axis] = {'raw': f_raw}
                except Exception:
                    warnings.warn(u"Kraftplattform {} Achse {} fehlt".format(alias, axis))
                    zero = np.zeros(self.total_samples, dtype=float)
                    self.forces[alias][axis] = {'raw': zero}

        return self.forces


class MarkerQualityLogger(object):
    """Logger to track valid and invalid markers."""

    def __init__(self):
        self.invalid = {}
        self.valid = set()

    def mark_invalid(self, marker, reason):
        """Mark a marker as invalid with a reason."""
        if marker not in self.invalid:
            self.invalid[marker] = []
        if reason not in self.invalid[marker]:
            self.invalid[marker].append(reason)

    def mark_valid(self, marker):
        """Mark a marker as valid."""
        self.valid.add(marker)

    def has_invalid(self):
        """Return True if there are invalid markers."""
        return len(self.invalid) > 0

    def summary(self):
        """Return a summary of valid and invalid markers."""
        return {
            'valid_markers': sorted(list(self.valid)),
            'invalid_markers': self.invalid
        }


class MarkerData(object):
    """Handle marker extraction (raw trajectories)."""

    def __init__(self, vicon, subject, config, axes=None):
        self.vicon = vicon
        self.subject = subject
        self.markers = config.get('markers', [])
        self.axes = axes if axes else ['z']
        self.marker_logger = MarkerQualityLogger()
        self.data = {}

    def load_markers(self):
        """Load marker trajectories and validate availability."""
        for m in self.markers:
            self.data[m] = {}
            try:
                x, y, z, _ = self.vicon.GetTrajectory(self.subject, m)
                raw = {
                    'x': np.asarray(x, dtype=float),
                    'y': np.asarray(y, dtype=float),
                    'z': np.asarray(z, dtype=float)
                }

                # Check at least one requested axis has usable data
                has_valid_data = any(
                    (ax in raw) and (not np.all(np.isnan(raw[ax]))) and (np.ptp(raw[ax]) > 0)
                    for ax in self.axes
                )
                if not has_valid_data:
                    raise RuntimeError(u"keine verwertbaren Bewegungsdaten")

                self.data[m] = raw
                self.marker_logger.mark_valid(m)

            except Exception as e:
                self.marker_logger.mark_invalid(m, str(e))

        return self.data, self.marker_logger


class DataCache(object):
    """Central data cache for forces, markers, and metadata."""

    def __init__(self, vicon, config, axes=None):
        self.vicon = vicon
        self.config = config
        self.subject = self.get_current_subject()
        self.axes = axes if axes else ['z']

        self.force_data = None
        self.marker_data = None
        self.marker_logger = None

    def get_current_subject(self):
        """Return the first subject name in the trial."""
        subjects = self.vicon.GetSubjectNames()
        if not subjects:
            raise RuntimeError(u"Keine Patienten in den Vicon-Daten gefunden.")
        return subjects[0]

    def load_data(self):
        """Load forces and markers."""
        # Forces
        self.force_data = ForcePlateData(self.vicon, self.config, axes=self.axes)
        forces = self.force_data.load_forces()

        # Keep deterministic ordering
        all_forces = OrderedDict()
        for k in sorted(forces.keys()):
            all_forces[k] = forces[k]

        # Markers
        self.marker_data = MarkerData(self.vicon, self.subject, self.config, axes=self.axes)
        markers, logger = self.marker_data.load_markers()
        self.marker_logger = logger

        # Pick one valid marker (for quick display/ROI usage)
        valid_marker = sorted(logger.valid)[0] if logger.valid else None

        return {
            'forces': all_forces,
            'markers': markers,
            'marker_quality': logger.summary(),
            'valid_markers': valid_marker
        }


class JSONExporter(object):
    """Export DataCache + UserDataExtractor results to JSON file (legacy-safe)."""

    def __init__(self, vicon):
        self.vicon = vicon
        self.results = None

    def to_json_safe(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self.to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.to_json_safe(v) for v in obj]
        else:
            return obj

    def _dump_json_utf8(self, path, data):
        """
        Write JSON in a way that works in both Python 2 and Python 3.
        Keeps unicode properly encoded as UTF-8.
        """
        # Python 2: json.dump supports 'encoding'; Python 3: it does not.
        try:
            with io.open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        except TypeError:
            # Fallback for strict Python 2 environments if needed
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True, encoding='utf-8')

    def save(self, data_cache, user_data, path_manager, new_path=False, filename=None):
        """
        Save DataCache + UserDataExtractor data as JSON.

        Parameters
        ----------
        data_cache : DataCache
            Instance containing forces, markers, etc.
        user_data : UserDataExtractor
            Instance containing patient/trial info
        path_manager : PathManager
            Canonical CMJ path manager
        new_path : bool
            If True, save to PathManager raw_data folder
        filename : str or None
            Optional filename. If None, generated from trial name.
        """
        self.results = {
            'user_info': user_data.as_dict(),
            'data_cache': data_cache.load_data()
        }

        trial_path, trial_name = self.vicon.GetTrialName()

        # Clean trial name for filename safety
        for bad in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            trial_name = trial_name.replace(bad, '_')

        if filename is None:
            filename = trial_name + "_data_cmj.json"

        # Output path
        if new_path:
            json_path = path_manager.raw_file(filename)
            # PathManager already ensures dirs, but keep it extra safe
            out_dir = os.path.dirname(json_path)
            if out_dir and not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir)
                except OSError:
                    pass
        else:
            trial_dir = os.path.dirname(trial_path)
            json_path = os.path.join(trial_dir, filename)
            if not os.path.exists(trial_dir):
                try:
                    os.makedirs(trial_dir)
                except OSError:
                    pass

        results = self.to_json_safe(self.results)
        self._dump_json_utf8(json_path, results)

        print(u"data-JSON erstellt: {}".format(json_path))
        return json_path