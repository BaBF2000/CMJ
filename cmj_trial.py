# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from scipy.signal import savgol_filter
from viconnexusapi import ViconNexus
import cmj_utils as cus
import csv
import codecs
import warnings
import os
from datetime import datetime

class Cmj(object):
    """Class for Counter Movement Jump (CMJ) analysis using Vicon data."""

    def __init__(self, vicon):
        self.vicon = vicon
        self.frame_rate = vicon.GetFrameRate()
        self.FrameCount = vicon.GetFrameCount()
        self.plate_rate = vicon.GetDeviceDetails(1)[2]  # Force plate rate

        subjects = vicon.GetSubjectNames()
        if not subjects:
            raise RuntimeError("Keine Patient:innen in den Vicon-Daten gefunden.")

        self.subject = subjects[0]
        self.forceplate_aliases = {'[FP right foot]': 'R', '[FP left foot]': 'L'}

        # ratio FP sampling / camera sampling
        self.samples_per_frame = max(1, int(self.plate_rate / self.frame_rate))
        self.total_samples = int(self.FrameCount * self.samples_per_frame)

    # ---------------- Data Acquisition ----------------
    def get_all_data(self, markers_names=None, axes=('x', 'y', 'z')):
        axes = [a.lower() for a in axes]
        axis_map = {'x': 'Fx', 'y': 'Fy', 'z': 'Fz'}
        data = {'forces': {}, 'markers': {}}

        # -------- Force plates --------
        for deviceID in self.vicon.GetDeviceIDs():
            name, dtype, rate = self.vicon.GetDeviceDetails(deviceID)[:3]
            if dtype != 'ForcePlate':
                continue

            alias = self.forceplate_aliases.get(name, name)
            data['forces'][alias] = {}

            outID = self.vicon.GetDeviceOutputIDFromName(deviceID, 'Force')

            for axis in axes:
                try:
                    chID = self.vicon.GetDeviceChannelIDFromName(deviceID, outID, axis_map[axis])
                    f_raw, _ = self.vicon.GetDeviceChannelGlobal(deviceID, outID, chID)[:2]
                    f_raw = np.array(f_raw, dtype=float)
                    f_raw = -f_raw
                    #f_filtered = cus.butter_lowpass(f_raw,1000)
                    f_filtered = savgol_filter(f_raw, window_length=31, polyorder=3)
                    data['forces'][alias][axis] = {'raw': f_raw, 'filtered': f_filtered}
                except Exception:
                    warnings.warn("Achse '{}' für Kraftmessplatte '{}' nicht gefunden. Es werden Nullen zurueckgegeben.".format(axis, alias))
                    data['forces'][alias][axis] = {'raw': np.zeros(self.total_samples),
                                                   'filtered': np.zeros(self.total_samples)}

        # -------- Total force --------
        total_force = {}
        for axis in axes:
            total = np.zeros(self.total_samples)
            for fp in data['forces']:
                if axis in data['forces'][fp]:
                    total += data['forces'][fp][axis]['filtered']
            total_force[axis] = total
        data['forces']['total'] = {ax: {'filtered': total_force[ax]} for ax in axes}

        # -------- Markers --------
        if markers_names is None:
            markers_names = []

        for marker in markers_names:
            try:
                x, y, z, _ = self.vicon.GetTrajectory(self.subject, marker)
                raw = {'x': np.array(x, float), 'y': np.array(y, float), 'z': np.array(z, float)}
                interp = {ax: cus.interpolation(self.FrameCount, self.total_samples, raw[ax])
                          for ax in axes}
                data['markers'][marker] = {'raw': raw, 'interp': interp}
            except Exception:
                warnings.warn("Marker '{}' nicht gefunden. Es werden Nullen verwendet.".format(marker))
                data['markers'][marker] = {'raw': {ax: np.zeros(self.FrameCount) for ax in axes},
                                           'interp': {ax: np.zeros(self.total_samples) for ax in axes}}

        return data

    # ---------------- CMJ Analysis ----------------
    def Cmj_analysis(self, markers_names, axis='z'):
        if axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be x, y or z.")

        data = self.get_all_data(markers_names, axes=(axis,))
        self.results = {}

        for marker in markers_names:
            if marker not in data['markers']:
                warnings.warn("Marker '{}' fehlt -> wird uebersprungen.".format(marker))
                continue

            traj = data['markers'][marker]['interp'][axis]
            
            traj = cus.convert_distance(traj)
            fR = data['forces']['R'][axis]['filtered']
            fL = data['forces']['L'][axis]['filtered']
            fT = data['forces']['total'][axis]['filtered']

            try:
                roi = cus.find_ROI_VOI(traj, fT, self.plate_rate)
            except Exception as e:
                warnings.warn("ROI für Marker '{}' nicht moeglich: {}".format(marker, e))
                continue

            metrics = cus.compute_jump_metrics(traj, fL, fR, fT, self.plate_rate)

            self.results[marker] = {
                'metrics': metrics
            }

        return self.results

    # ---------------- CSV Export ----------------
    def create_csv(self, filename=None):
        if not hasattr(self, 'results'):
            raise RuntimeError("Sie muessen zuerst Cmj_analysis() ausfuehren, bevor create_csv() aufgerufen werden kann.")
    
        try:
            trial_path, trial_name = self.vicon.GetTrialName()
        except:
            trial_path = ""
            trial_name = "unknown_trial"
    
        # Nettoyer le nom du fichier
        for bad in ['\\','/',':','*','?','"','<','>','|']:
            trial_name = trial_name.replace(bad,'_')
    
        if filename is None:
            filename = trial_name + "_cmj.csv"
    
        trial_dir = os.path.dirname(trial_path)
        csv_path = os.path.join(trial_dir, filename)
    
        # Construire l'en-tête sans ROI
        sample_marker = next(iter(self.results))
        metric_keys = [k for k in self.results[sample_marker]['metrics'].keys() ] #if k != 'ROI'
        header = ['Name', 'Trial', 'Datum', 'Marker'] + metric_keys
    
        # Extraire le numéro du trial
        try:
            trial_number = trial_name.split('_')[2]
            date_iso = datetime.strptime(trial_name.split('_')[1], "%d%m%Y")
            date = date_iso.strftime("%d.%m.%Y")
            patient_folder = os.path.basename(os.path.dirname((os.path.dirname(trial_path))))
            Name = patient_folder.split()[0] + ", " + " ".join(patient_folder.split()[1:])
        
        except IndexError:
            trial_number = trial_name
            date = datetime.now()
         # Écriture CSV
        with codecs.open(csv_path, 'w', 'utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for marker, data in self.results.items():
                row = [Name, trial_number,date, marker] + [data['metrics'][k] for k in metric_keys]
                writer.writerow(row)
    
        print("CSV-Datei:'{}' erstellt".format(filename))
        print("CSV-Datei {} gespeichert".format(csv_path) )
        return csv_path
    
    # ---------------- Plot ROI ----------------
    def plot_marker_ROI(self, marker, axis='z'):
        if not hasattr(self, "results"):
            raise RuntimeError("Sie muessen zuerst Cmj_analysis() ausfuehren.")

        if marker not in self.results:
            raise ValueError("Marker '{}' in den Ergebnissen nicht gefunden.".format(marker))
        
        data = self.get_all_data(markers_names=[marker], axes=[axis])
        traj = data['markers'][marker]['interp'][axis]
        traj = cus.butter_lowpass(traj, 250)
        fT = data['forces']['total'][axis]['filtered']
        roi = self.results[marker]['metrics']['ROI']

        cus.interactive_ROI_VOI_plot(traj, fT, roi, self.plate_rate)


# ---------------- Main ----------------
if __name__ == '__main__':
    vicon = ViconNexus.ViconNexus()
    cmj = Cmj(vicon)
    markers = ['LASI']
    cmj.Cmj_analysis(markers)
    cmj.plot_marker_ROI('LASI')
    cmj.create_csv()

