# -*- coding: utf-8 -*-

"""

Comprehensive Countermovement Jump (CMJ) analysis utilities.

Features:
- Signal processing: filtering, interpolation, derivatives, integration
- ROI detection: eccentric, concentric, takeoff, landing
- Performance metrics: peak force, RFD, RSI-mod, takeoff velocity, concentric power
- Left-right asymmetry analysis
- Trial validation
- Interactive plotting of forces, trajectory, velocity, acceleration

"""

from __future__ import division, print_function  # Ensure float division in Python 2.7
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.collections as mcoll
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


# ---------------- Constants ----------------
ROUND_VALUE = 1    # Rounding precision for metrics
G = 9.81           # Gravity [m/s^2]

# ----------------- Basic Tools ---------------

def convert_distance(value, from_unit="mm", to_unit="m"):
    """
    Convert distance between mm, cm, and m.
    """
    units = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
    if from_unit not in units or to_unit not in units:
        raise ValueError("Units must be 'mm', 'cm', or 'm'")
    return value * units[from_unit] / units[to_unit]
    

# -------- Signal processing --------

def butter_lowpass(data, fs, cutoff = 12, order=2):
    """
    Apply a low-pass Butterworth filter to a signal.
    Parameters
    ----------
    data : ndarray
        Input signal.
    fs : float
        Sampling frequency [Hz].
    cutoff : float
        Cutoff frequency [Hz].
    order : int
        Filter order.
    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data)


def interpolation(FrameCount, total_samples, data):
    """
    Resample a signal using cubic interpolation.

    Parameters
    ----------
    FrameCount : int
        Original number of samples.
    total_samples : int
        Desired number of samples.
    data : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Interpolated signal.
    """
    return interp1d(np.linspace(0, 1, FrameCount), data, kind='cubic')(np.linspace(0, 1, total_samples))




def derivatives(signal, sampling_rate, order=1):
    """
    Compute successive numerical derivatives of a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    sampling_rate : float
        Sampling frequency [Hz].
    order : int
        Number of derivatives to compute.

    Returns
    -------
    list of ndarray
        List of derivatives (velocity, acceleration, jerk, etc.).
    """
    signal = np.array(signal, dtype=float)
    dt = 1.0 / sampling_rate
    derivatives_list = []
    current = signal
    for _ in range(order):
        current = np.gradient(current, dt, axis=0)
        derivatives_list.append(current)
    return derivatives_list

def integrate(force, rate, start, end):
    """
    Computes integral (impulse) using trapezoidal rule.

    Parameters
    ----------
    signal : array-like
        Signal to integrate.
    rate : float
        Sampling frequency [Hz].
    start : int
        Start index.
    end : int
        End index.

    Returns
    -------
    value : float
        Integral of signal over [start:end].
    """
    return np.trapz(force[start:end + 1], dx=1.0 / rate)

# ------------------- Flight Detection ---------------
def estimate_bodyweight(forces, rate, window=0.5, std_thresh=0.02):
    n = int(rate * window)
    for i in range(len(forces) - n):
        segment = forces[i:i+n]
        if np.std(segment) / np.mean(segment) < std_thresh:
            return np.mean(segment), i+n
    raise ValueError("No stable bodyweight phase detected")

def estimate_initial_velocity(vel, rate, window=0.3, std_thresh=0.005):
    """
    Estimate stable initial velocity by finding a window with low standard deviation.
    """
    vel = np.array(vel)
    n = int(window * rate)
    for i in range(len(vel) - n):
        segment = vel[i:i+n]
        if np.std(segment) < std_thresh:
            return np.mean(segment)
    # fallback: take mean of first window if no stable phase found
    return np.mean(vel[:n])

def detect_takeoff_landing(forces, rate, threshold_factor=0.05):
    """
    Detects take-off and landing indices based on vertical ground reaction force.

    Parameters
    ----------
    forces : array-like
        Vertical ground reaction force (N)
    rate : float
        Sampling rate in Hz
    threshold_factor : float, default=0.05
        Fraction of bodyweight to define the flight phase

    Returns
    -------
    takeoff_idx : int
        Index of take-off (first point below threshold)
    landing_idx : int
        Index of landing (last point below threshold)
    flight_time : float
        Flight duration in seconds
    bodyweight : float
        Estimated bodyweight from mean of first 500 samples
    """
    bodyweight,_ = estimate_bodyweight(forces, rate)
    threshold = bodyweight * threshold_factor
    zero_idx = np.where(forces < threshold)[0]
    if len(zero_idx) == 0:
        raise ValueError("No flight phase detected (no point below threshold)")
    takeoff_idx = zero_idx[0]
    landing_idx = zero_idx[-1]
    flight_time = (landing_idx - takeoff_idx) / rate
    return takeoff_idx, landing_idx, flight_time, bodyweight


# -------------------- ROI Detection ------------------------

def find_v_edge(array, direction="left", window_smooth=5, window_slope=3):
    """
    Finds the left or right boundary of a V-shaped segment in a 1D signal.

    Parameters
    ----------
    array : array-like
        1D signal
    direction : {"left", "right"}
        Direction of the search from the minimum
    window_smooth : int
        Size of the smoothing window to reduce noise
    window_slope : int
        Size of the window used to compute average slope

    Returns
    -------
    idx_edge : int
        Index of the left or right edge of the V
    idx_min : int
        Index of the minimum (bottom of the V)
    """
    x = np.array(array, dtype=float)

    # 1. Smooth the signal
    x_smooth = uniform_filter1d(x, size=window_smooth)

    # 2. Find minimum
    idx_min = np.argmin(x_smooth)
    idx = idx_min

    if direction == "left":
        step = -1
        stop = 0
        slope_sign = 1     # positive slope => exit V
    elif direction == "right":
        step = 1
        stop = len(x_smooth) - 1
        slope_sign = -1    # negative slope => exit V
    else:
        raise ValueError("direction must be 'left' or 'right'")

    # 3. Move in chosen direction using average slope
    while idx != stop:
        next_idx = idx + step
        start = max(0, idx - window_slope) if direction == "left" \
                else min(len(x_smooth) - 1, idx + window_slope)

        slope = x_smooth[next_idx] - x_smooth[start]

        if slope * slope_sign >= 0:
            break

        idx = next_idx

    return idx, idx_min

def find_ROI_VOI(trajectory, forces, rate):
    """
    Detects Regions of Interest (ROIs) and Values of interest (VOIs) in a jump:
    - Unloading: start of eccentric phase to minimum position
    - Braking: minimum to peak
    - Deceleration: part of braking when force >= bodyweight
    - Propulsion: after peak, split into p1 and p2

    Parameters
    ----------
    trajectory : array-like
        Measured trajectory (m)
    forces : array-like
        Vertical force (N)
    rate : float
        Sampling rate in Hz

    Returns
    -------
    roi : dict
        Keys: start, stand, jump_height, bodyweight, f_start, local_force_minimum, unloading (tuple), braking (tuple), deceleration (tuple),
              propulsion_p1 (tuple), propulsion_p2 (tuple), landing 
    """
    trajectory = np.array(trajectory)
    forces = np.array(forces)

    peak = np.argmax(trajectory)
    takeoff_idx, landing_idx, _, bodyweight = detect_takeoff_landing(forces, rate)

    # Left V-edge (start of eccentric phase)
    local_start_t, local_min_t = find_v_edge(trajectory[:peak])
    local_start_f, local_min_f = find_v_edge(forces[:local_min_t])
    global_start = local_start_f

    # Unloading and braking
    force_local_min = np.argmin(forces[:local_min_t])
    f_min = force_local_min
    unloading = (global_start, force_local_min)
    braking   = ( force_local_min, local_min_t)

    # Quietphase:
    idx_q = np.where(forces[:force_local_min] >= bodyweight)[0]
    if len(idx_q) == 0:
        raise ValueError("Trialerror: No standing point detected")
    stand =  idx_q[-1]

    # Jump_height
    jh = convert_distance(np.abs(trajectory[stand] - trajectory[peak]), 'm','cm')

    # Deceleration: force reaches bodyweight during braking
    idx = np.where(forces[force_local_min:local_min_t] >= bodyweight)[0]
    if len(idx) == 0:
        raise ValueError("Force never reaches bodyweight during braking.")  
    deceleration = (force_local_min+ idx[0], local_min_t)

    # Propulsion after peak
    mid = local_min_t + ((takeoff_idx - local_min_t) // 2)
    p1 = (local_min_t, mid)
    p2 = (mid, takeoff_idx)

    min_landing = np.argmin(trajectory[peak:])
    landing = (landing_idx, peak + min_landing)
    

    return {
        "start": global_start,
        "stand": stand,
        "Jump_height":round(jh,ROUND_VALUE), #cm 
        "bodyweight": bodyweight,
        "f_start": local_start_f,
        "local_force_minimum": f_min,
        "eccentric":{
            "unloading": unloading,
            "braking": braking,
            "deceleration": deceleration
            },
        "concentric":{
            "propulsion_p1": p1,
            "propulsion_p2": p2
            }, 
        "landing": landing
    }
# ------------------- Performance Metrics --------------------

def peak_force(forces, roi):
    """
    Computes peak vertical force in key phases of the Countermovement Jump (CMJ).

    Parameters
    ----------
    forces : array-like
        Vertical ground reaction force signal (N)
    roi : dict
        Region of Interest dictionary defining CMJ phases

    Returns
    -------
        : dict
        Peak force values for braking, deceleration, propulsion, and landing
    """
    

    return {
        "PeakForce_braking": round(np.max(forces[slice(*roi["eccentric"]["braking"])]), ROUND_VALUE),
        "PeakForce_deceleration": round(np.max(forces[slice(*roi["eccentric"]["deceleration"])]), ROUND_VALUE),
        "PeakForce_propulsion": round(np.max(forces[slice(
            roi["concentric"]["propulsion_p1"][0],
            roi["concentric"]["propulsion_p2"][1]
        )]), ROUND_VALUE),
        "PeakForce_landing": round(np.max(forces[slice(*roi["landing"])]), ROUND_VALUE)
    }

def compute_RFD(forces, rate, roi, windows_ms=[100, 200]):
    """
    Computes the rate of force development (RFD) during the eccentric braking phase of a jump.
    """

    df = derivatives(forces, rate, 1)[0]
    s, e = roi["eccentric"]["braking"]

    out = {"RFD_max": round(np.max(df[s:e]), ROUND_VALUE)}

    for w in windows_ms:
        n = int(rate * w / 1000)
        out["RFD_0_{}ms".format(w)] = round(
            max((forces[i + n] - forces[i]) / (n / rate) for i in range(s, e - n)),
            ROUND_VALUE
        )
    return out

def RSI_modified(roi, rate):
    """
    Computes the modified Reactive Strength Index (RSI-mod) for a jump.
    
    RSI-mod = Jump Height / Time to Take-Off

    This metric captures how efficiently an athlete can convert 
    eccentric loading into concentric propulsion:

    - Jump Height: measures the overall performance/output of the jump.
    - Time to Take-Off: duration of the concentric propulsion phase
    """
    time_to_takeoff = (roi["concentric"]["propulsion_p2"][1] - roi["start"]) / rate
    return round(roi["Jump_height"] / time_to_takeoff, 3)

def takeoff_velocity(forces, rate, roi):
    """
    Computes the take-off velocity of a jump using the force-time integral.

    Parameters:
    - forces: array of vertical ground reaction forces [N].
    - bodyweight: athlete's bodyweight [N].
    - rate: sampling rate of force data [Hz].
    - roi: region-of-interest dictionary

    Returns:
    - Take-off velocity in meters per second (rounded to 3 decimals).
    """
    s, e = roi["stand"], roi["concentric"]["propulsion_p2"][1]
    net_force = forces[s:e] - roi["bodyweight"]
    impulse = np.trapz(net_force, dx=1 / rate)
    mass = roi["bodyweight"] / G
    return round(impulse / mass, ROUND_VALUE)

def compute_com_kinematics(forces, bodyweight, rate): 
    mass = bodyweight / G
    acc = (forces - bodyweight) / mass

    vel = np.cumsum(acc)/rate
    vel -= estimate_initial_velocity(vel,rate)

    pos = np.cumsum(vel)/rate
    return pos, vel, acc



def concentric_power(forces, rate, roi):
    """
    Computes mean and peak concentric power during the jump.

    Parameters:
    - forces: array of vertical ground reaction forces [N].
    - velocity: array of vertical velocities corresponding to the force data [m/s].
    - roi: region-of-interest dictionary 

    Returns:
    - Dictionary with:
        - "Power_mean": mean concentric power [W]
        - "Power_peak": peak concentric power [W]
    """
    s, e = roi["concentric"]["propulsion_p1"][0], roi["concentric"]["propulsion_p2"][1]

    _,velocity,_  = compute_com_kinematics(forces,roi['bodyweight'], rate)
    P = forces[s:e] * velocity[s:e]

    return {
        "Power_mean": round(np.mean(P), ROUND_VALUE),
        "Power_peak": round(np.max(P), ROUND_VALUE)
    }

def landing_loading_rate(forces, rate, roi):
    """
    Computes vertical loading rate and peak landing force during landing.

    Parameters:
    - forces: array of vertical ground reaction forces [N].
    - rate: sampling frequency of force data [Hz].
    - roi: region-of-interest dictionary marking key jump phases:
        - "landing": tuple (start_index, end_index) of landing phase

    Returns:
    - Dictionary with:
        - "LoadingRate_peak": peak vertical loading rate [N/s]
        - "PeakLandingForce": peak vertical force during landing [N]
    """
    s, e = roi["landing"]
    df = derivatives(forces, rate, order=1)[0]

    return {
        "LoadingRate_peak": round(np.max(df[s:e]), 1),
        "PeakLandingForce": round(np.max(forces[s:e]), 1)
    }



def impulse_left_right(L, R, roi, rate):
    """
    Compute left and right impulses for key CMJ phases using the custom integrate() function.
    
    Impulse is defined as the integral of net vertical force (F - mg/2) for each leg separately.

    Parameters
    ----------
    L : array-like
        Left force plate data [N]
    R : array-like
        Right force plate data [N]
    roi : dict
        Region-of-interest dictionary with phase indices
    rate : float
        Sampling rate [Hz]

    Returns
    -------
    dict, dict
        Impulse values per phase for left and right leg
    """

    phases = ["absprung", "braking", "deceleration", "propulsion", "landing"]
    impulses_L = {}
    impulses_R = {}
    def estimate_bodyweight_per_leg(L, R, rate, window=0.5):
        n = int(rate * window)
        for i in range(len(L) - n):
            segL = L[i:i+n]
            segR = R[i:i+n]
            total = segL + segR
            if np.std(total) / np.mean(total) < 0.02:
                return np.mean(segL), np.mean(segR),np.mean(total)
        raise ValueError("No stable standing phase found")
    bwl,bwr,bwt = estimate_bodyweight_per_leg(L, R   , rate)
    for phase in phases:
        if phase == "absprung":
            s, e = roi["eccentric"]["braking"][0], roi["concentric"]["propulsion_p2"][1]
        elif phase == "braking":
            s, e = roi["eccentric"]["braking"]
        elif phase == "deceleration":
            s, e = roi["eccentric"]["deceleration"]
        elif phase == "propulsion":
            s, e = roi["concentric"]["propulsion_p1"][0], roi["concentric"]["propulsion_p2"][1]
        elif phase == "landing":
            s, e = roi["landing"]

        impulses_L[phase] = integrate(np.array(L),rate, s, e)
        impulses_R[phase] = integrate(np.array(R),rate, s, e)
        #print('bodyweith_left:{}, bodyweight_right:{}'.format(bwl,bwr))
    return impulses_L, impulses_R


def phase_durations(roi, rate):
    """
    Compute the duration of key phases in a countermovement jump (CMJ).

    Parameters
    ----------
    roi : dict
        Dictionary containing the Regions of Interest (ROIs) of the CMJ
    rate : float
        Sampling frequency of the signals [Hz].

    Returns
    -------
    dict
        Dictionary with phase durations in seconds, rounded to 3 decimals:
        - "Eccentric_duration": duration of the eccentric phase.
        - "Concentric_duration": duration of the concentric phase.
        - "Time_to_takeoff": total time from start to takeoff.
        
    """
    return {
        "Eccentric_duration": round((roi["eccentric"]["braking"][1] - roi["start"]) / rate, ROUND_VALUE),
        "Concentric_duration": round((roi["concentric"]["propulsion_p2"][1] -roi["concentric"]["propulsion_p1"][0]) / rate, 3),
        "Time_to_takeoff": round((roi["concentric"]["propulsion_p2"][1] - roi["start"]) / rate, 3)
    }



def asymmetry(L, R, roi, rate):
    """
    Compute left-right asymmetry for impulses and power for all available phases.
    
    Impulse uses integrate(F - mg/2) per leg.
    Power uses mean(F * v) per leg (literature recommendation).

    Parameters
    ----------
    L, R : array-like
        Left and right force plate data
    roi : dict
        Region-of-interest dictionary
    rate : float
        Sampling rate

    Returns
    -------
    dict
        Asymmetry in % for impulses and power per phase
    """
    # Compute impulses per leg
    impulses_L, impulses_R = impulse_left_right(L, R, roi, rate)

    # Total force for COM velocity
    total_force = np.array(L) + np.array(R)
    bodyweight, _ = estimate_bodyweight(total_force, rate)
    _, velocity, _ = compute_com_kinematics(total_force, bodyweight, rate)

    asymmetry = {"Impulse": {}, "Power": {}}

    for phase, iL in impulses_L.items():
        iR = impulses_R[phase]

        # Impulse asymmetry
        delta_impulse = abs((abs(iL / abs(iL) + abs(iR)) * 100) - (abs(iR / iL + iR) * 100))
        asymmetry["Impulse"][phase.capitalize()] = {"L": iL, "R": iR, "delta_percent": round(delta_impulse, 2)}

        # Determine indices for the phase
        if phase == "absprung":
            s, e = roi["eccentric"]["braking"][0], roi["concentric"]["propulsion_p2"][1]
        elif phase == "braking":
            s, e = roi["eccentric"]["braking"]
        elif phase == "deceleration":
            s, e = roi["eccentric"]["deceleration"]
        elif phase == "propulsion":
            s, e = roi["concentric"]["propulsion_p1"][0], roi["concentric"]["propulsion_p2"][1]
        elif phase == "landing":
            s, e = roi["landing"]
        else:
            continue

        # Power asymmetry using mean(F * v)
        F_L = np.array(L[s:e+1], dtype=float)
        F_R = np.array(R[s:e+1], dtype=float)
        V = np.array(velocity[s:e+1], dtype=float)

        pL = np.mean(F_L * V)
        pR = np.mean(F_R * V)
        delta_power = abs(pL - pR) / max(abs(pL) + abs(pR), 1e-6) * 100

        asymmetry["Power"][phase.capitalize()] = {"L": pL, "R": pR, "delta_percent": round(delta_power, 2)}

    return asymmetry



def compute_jump_metrics(trajectory, L, R, total_forces, rate):
    """
    Compute a full set of countermovement jump (CMJ) metrics in one call.

    Parameters
    ----------
    trajectory : array-like
        Vertical displacement of the CMJ (meters).
    L : array-like
        Left force plate time series [N].
    R : array-like
        Right force plate time series [N].
    total_forces : array-like
        Total vertical force (sum of L + R) [N].
    rate : float
        Sampling frequency [Hz].

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - PeakForce: dict of peak forces in each phase
        - RFD: rate of force development
        - RSI_modified: Reactive Strength Index (cm/s)
        - Takeoff_velocity: velocity at take-off [m/s]
        - ConcentricPower: mean and peak power during propulsion [W]
        - LandingLoadingRate: peak landing force and loading rate [N/s]
        - Phase durations: eccentric, concentric, total time to takeoff [s]
        - Asymmetry: left-right asymmetry in impulse and power [%]
        - ROI: dictionary of detected jump phases (start, stand, eccentric, concentric, landing)
    """
    # --- ROI detection ---
    total_forces = np.array(total_forces)
    
    roi = find_ROI_VOI(trajectory, total_forces, rate)

    metrics = {}

    # --- Peak force (total) ---
    metrics["PeakForce"] = peak_force(total_forces, roi)

    # --- Rate of Force Development ---
    metrics["RFD"] = compute_RFD(total_forces, rate, roi)

    # --- RSI-modified ---
    metrics["RSI_modified"] = RSI_modified(roi, rate)

    # --- Takeoff velocity ---
    metrics["Takeoff_velocity"] = takeoff_velocity(total_forces, rate, roi)

    # --- Concentric power (mean & peak) ---
    
    metrics["ConcentricPower"] = concentric_power(total_forces, rate, roi)

    # --- Landing loading rate ---
    metrics["LandingLoadingRate"] = landing_loading_rate(total_forces, rate, roi)

    # --- Phase durations ---
    metrics.update(phase_durations(roi, rate))

    # --- Left-right asymmetry metrics ---
    metrics["Asymmetry"] = asymmetry(L, R, roi, rate)

    # --- ROI for reference ---
    metrics["ROI"] = roi

    return metrics


def interactive_ROI_VOI_plot(trajectory, force, roi, rate):
    """
    Plot interactive Force, Trajectory, Velocity, and Acceleration with ROI phases.
    
    Parameters
    ----------
    trajectory : array-like
        Measured vertical trajectory (m)
    force : array-like
        Vertical ground reaction force (N)
    roi : dict
        Region-of-interest dictionary (from find_ROI_VOI)
    rate : float
        Sampling rate in Hz
    """


    # Ensure numpy arrays
    force = np.array(force, dtype=float)
    traj = np.array(trajectory, dtype=float)
    trajectory = traj - roi["eccentric"]["unloading"][1]
    plot_f = [max(value, force[roi['local_force_minimum']]) for value in force ] 
    plot_force = plot_f - force[roi['local_force_minimum']] 
    plot_force = np.array(plot_force, dtype=float) 
    plot_force[:roi['f_start']] = force[roi['f_start']] - force[roi['local_force_minimum']]
    # --- Compute position, velocity, and acceleration from force ---

    pos, vel, acc = compute_com_kinematics(force, roi['bodyweight'], rate)
    factor = 1000
    pos = pos * factor
    vel = vel * factor/10
    acc = acc * factor/100
    threshold = force[roi['f_start']] - force[roi['local_force_minimum']] 
    landing_idx = roi['landing'][1]
    # Get the landing portion 
    landing_force = plot_force[landing_idx:]
    # Find indices where the value <= threshold 
    indices = np.where(landing_force <= threshold)[0]

    if len(indices) > 0: # First index where condition is met 
        first_idx = indices[0] # Set all values from that index onward to threshold 
        landing_force[first_idx:] = threshold
    # Put back into plot_force 
    plot_force[landing_idx:] = landing_force

    t = np.arange(len(force))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Align first value of force with first value of trajectory 
    offset_t = plot_force[0] - trajectory[0] 
    trajectory += offset_t

    # Plot measured trajectory
    traj_line, = ax.plot(t, trajectory, color='black', label='Measured Trajectory')
    

    offset_p = plot_force[0] - pos[0] 
    pos += offset_p
    # Plot trajectory computed from force
    pos_line, = ax.plot(t, pos, color='gray', linestyle='--', label='Computed Trajectory')
    
    offset_vel = plot_force[0] - vel[0] 
    vel += offset_vel
    # Plot velocity
    vel_line, = ax.plot(t, vel, color='red', label='Velocity')
    
    
    offset_acc = plot_force[0] - acc[0] 
    acc += offset_acc
    # Plot acceleration
    acc_line, = ax.plot(t, acc, color='blue', label='Acceleration')

    # Plot force
    force_line, = ax.plot(t, plot_force, color='green', label='Force')

    # --- Shaded ROI phases ---
    phases = {
        'Braking': ('orange', roi['eccentric']['braking']),
        'Propulsion P1': ('red', roi['concentric']['propulsion_p1']),
        'Propulsion P2': ('purple', roi['concentric']['propulsion_p2']),
        'Landing': ('blue', roi['landing'])
    }

    shaded = {}
    for name, (color, (start, end)) in phases.items():
        shaded[name] = ax.fill_between(t[start:end], plot_force[start:end], 0,
                                       color=color, alpha=0.3, label=name)

    # --- CheckButtons to toggle visibility ---
    rax = plt.axes([0.85, 0.4, 0.12, 0.2])
    labels = ['Measured Trajectory', 'Computed Trajectory', 'Force', 'Velocity', 'Acceleration'] + list(phases.keys())
    visibility = [True]*5 + [True]*len(phases)
    check = CheckButtons(rax, labels, visibility)

    lines = {
        'Measured Trajectory': traj_line,
        'Computed Trajectory': pos_line,
        'Force': force_line,
        'Velocity': vel_line,
        'Acceleration': acc_line
    }
    lines.update(shaded)

    def toggle_visibility(label):
        obj = lines[label]
        if isinstance(obj, mcoll.PolyCollection):
            obj.set_visible(not obj.get_visible())
        else:
            obj.set_visible(not obj.get_visible())
        fig.canvas.draw()

    check.on_clicked(toggle_visibility)

    ax.set_xlabel("Samples")
    ax.set_ylabel("Force / Position / Velocity / Acceleration")
    ax.set_title("Force, Trajectory, Velocity, and Acceleration with ROI phases")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def detect_invalid_trial(trajectory, L, R, total_forces, rate,
                         min_jump_height_cm=5,
                         max_force_bw=8,
                         max_asymmetry_percent=40,
                         noise_thresh=0.05):
    """
    Automatically detect invalid CMJ trials based on biomechanical criteria.

    Returns
    -------
    dict
        {
          "valid": bool,
          "reasons": list of str
        }
    """
    reasons = []
    total_forces = np.array(total_forces)

    # --- Flight detection ---
    try:
        takeoff, landing, flight_time, bodyweight = detect_takeoff_landing(total_forces, rate)
        if flight_time <= 0:
            reasons.append("No flight phase detected")
    except Exception:
        reasons.append("Flight phase detection failed")
        return {"valid": False, "reasons": reasons}

    # --- Jump height ---
    roi = find_ROI_VOI(trajectory, total_forces, rate)
    if roi["Jump_height"] < min_jump_height_cm:
        reasons.append("Jump height too low (< {} cm)".format(min_jump_height_cm))

    # --- Take-off velocity ---
    v_to = takeoff_velocity(total_forces, rate, roi)
    if v_to <= 0:
        reasons.append("Non-positive take-off velocity")

    # --- Force sanity check ---
    max_force = np.max(total_forces)
    if max_force > max_force_bw * bodyweight:
        reasons.append("Excessive peak force (> {} BW)".format(max_force_bw))

    # --- Noise check (standing phase) ---
    stand = roi["stand"]
    std_rel = np.std(total_forces[:stand]) / bodyweight
    if std_rel > noise_thresh:
        reasons.append("Excessive noise during standing phase")

    # --- Asymmetry ---
    asym = asymmetry(L, R, roi, rate)
    for phase, vals in asym["Impulse"].items():
        if vals["delta_percent"] > max_asymmetry_percent:
            reasons.append("High asymmetry in {} impulse".format(phase))
            break

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons
    }
