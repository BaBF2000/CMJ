# -*- coding: utf-8 -*-
import os
import subprocess
import pandas as pd
from viconnexusapi import ViconNexus
import cmj_utils as cus
import codecs


def combine_jump_data(vicon, output_file="combined_jump_results.csv"):
    """
    Combine CMJ CSV data from a Vicon trial into a single CSV file.
    """
    # Retrieve trial name and path
    try:
        trial_path, trial_name = vicon.GetTrialName()
    except Exception as e:
        print("Fehler beim Abrufen des Trial-Namens: {}".format(e))
        return None

    folder_path = trial_path
    print("Aktuelles Trial: {} (Pfad: {})".format(trial_name, folder_path))

    output_path = os.path.join(folder_path, output_file)

    # Delete or backup existing output file
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print("  Vorhandene Datei '{}' wurde geloescht.".format(output_file))
        except Exception:
            backup_path = output_path + ".backup"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(output_path, backup_path)
                print("  Vorhandene Datei gesperrt, umbenannt in '{}'.".format(backup_path))
            except Exception as e:
                print("  Vorhandene Datei konnte nicht geloescht oder umbenannt werden: {}".format(e))
                return None

    # List all files
    c3d_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.c3d')]
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

    if len(c3d_files) - 1 != len(csv_files):
        print("  FEHLER: Anzahl CSV-Dateien ({}) stimmt nicht mit Anzahl C3D-Dateien ({}) ueberein.".format(
            len(csv_files), len(c3d_files)))
        print("  Bitte ueberpruefen, dass alle CSV-Dateien vorhanden sind.")
        return None
    else:
        print("  Alle CSV-Dateien vorhanden ({} Dateien).".format(len(csv_files)))

    # Merge CSVs, keeping only the first row per file
    all_data = []
    processed_files = set()

    for i, file in enumerate(csv_files, 1):
        if file in processed_files:
            print("  Datei '{}' bereits verarbeitet, wird uebersprungen.".format(file))
            continue

        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            # Safely extract source identifier
            split_name = file.split('_')
            source = split_name[2] if len(split_name) > 2 else file
            df['SourceFile'] = source

            # Automatic filtering of invalid trials if columns exist
            if 'Trajectory' in df.columns and 'R' in df.columns and 'L' in df.columns and 'Total' in df.columns:
                traj = df['Trajectory'].values
                fR = df['R'].values
                fL = df['L'].values
                fT = df['Total'].values
                rate = df.attrs.get('SamplingRate', 250)  # fallback to 250 Hz
                result = cus.detect_invalid_trial(traj, fL, fR, fT, rate)
                df['Valid'] = result['valid']
                df['Reason'] = ';'.join(result['reasons']) if not result['valid'] else ''

            # Keep only the first row
            all_data.append(df.iloc[[0]])
            processed_files.add(file)
            print("  ({}/{}) Datei geladen: {}".format(i, len(csv_files), file))
        except Exception as e:
            print("  Fehler beim Lesen '{}': {}".format(file, e))

    if not all_data:
        print("Keine CSV-Dateien zum Zusammenfuehren gefunden.")
        return None

    # Final concatenation
    combined_df = pd.concat(all_data, ignore_index=True)
    # Strip strings safely for Python 2.7 compatibility
    if 'Subject' in combined_df.columns:
        combined_df['Subject'] = combined_df['Subject'].apply(lambda x: str(x).strip())

    # Save final file
    try:
        combined_df.to_csv(codecs.open(output_path, 'w', 'utf-8-sig'), index=False)
        print("\nAlle Dateien wurden zusammengefuehrt und gespeichert unter: {}".format(output_path))
        print("Anzahl der Zeilen: {}".format(len(combined_df)))
    except Exception as e:
        print("  Fehler beim Speichern der Datei: {}".format(e))
        return None

    return combined_df


if __name__ == "__main__":
    vicon = ViconNexus.ViconNexus()
    combined_df = combine_jump_data(vicon)
    if combined_df is not None:
        gui_path = r"<PATH_TO_GUI_SCRIPT>"
        python3_path = r"<PATH_TO_PYTHON_3>"
        try:
            subprocess.Popen([python3_path, gui_path])
            print("GUI wurde gestartet.")
        except Exception as e:
            print("Fehler beim Starten der GUI: {}".format(e))
