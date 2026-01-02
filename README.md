#  CMJ Analysis – Force Plate & Motion Capture Toolbox

>  **Project status: Ongoing / Work in progress**  
> This project is actively under development. Features, structure, and documentation may evolve.

##  Overview

This repository contains a **biomechanical analysis toolbox for Countermovement Jump (CMJ)** assessment, designed for **research and clinical environments**.

The project combines:
- **Vicon motion capture data**
- **Dual force plate measurements**
- **Signal processing and biomechanical modeling**
- **Automated reporting**

It enables detailed **phase-based CMJ analysis**, impulse and power metrics, **left–right asymmetry assessment**, and **session-level aggregation**, with optional visualization and reporting tools.

---

##  Key Features

- Automatic detection of CMJ phases (eccentric, concentric, landing)
- ROI / VOI (Region of Interest / Value of Interest) segmentation
- Force- and impulse-based performance metrics
- Left–right asymmetry analysis (impulse & power)
- Rate of Force Development (RFD)
- Jump height, take-off velocity, RSI-modified
- Session-level CSV aggregation from multiple trials
- Automated **Word report generation**
- Interactive visualization of force, trajectory, velocity, and acceleration

---

##  Scientific Background

This toolbox follows commonly accepted biomechanical principles used in CMJ analysis:

- Force-time integration for impulse and velocity estimation
- Bodyweight normalization
- Phase-based interpretation of eccentric and concentric mechanics
- Asymmetry metrics based on impulse and  power
- Noise and validity checks for clinical reliability

---

##  Project Structure
<pre>
CMJ/
├── cmj_utils.py          # Core biomechanical computations and signal processing
├── cmj_trial.py          # Trial-level CMJ processing (not shown here)
├── cmj_session.py        # Session-level CSV aggregation and validation
├── cmj_word_report.py    # Automated Word report generation
├── pyside_gui.py         # GUI for report generation (PySide6)
├── README.md             # Project documentation
├── .gitignore
</pre>

---

##  Technologies Used

- **Python 2.7** (Vicon Nexus API compatibility)
- **Python 3.x** (GUI and report generation)
- NumPy, SciPy, Pandas
- Matplotlib
- PySide6
- Vicon Nexus SDK

> A Python 2.7 ↔ Python 3 bridge is used to ensure compatibility with legacy Vicon environments while enabling modern tooling.

---

## 🧪 Example Metrics Output

- Peak force (braking, propulsion, landing)
- Rate of Force Development (RFD)
- Jump height (cm)
- Take-off velocity (m/s)
- Concentric power (mean & peak)
- Phase durations
- Left–right impulse and power asymmetry
- Automatic trial validity detection

---

##  ROI / VOI Concept

- **ROI (Region of Interest)**: Time-based phases of the CMJ (eccentric, concentric, landing)
- **VOI (Value of Interest)**: Extracted biomechanical metrics 

---

## 🚧 Project Status

This project is **still under active development**.

Planned/possible improvements include:
- Code refactoring and modularization
- Extended validation criteria
- Improved documentation and examples
- Unit tests
- Packaging as a reusable Python module
- Enhanced GUI and batch processing

---

##  Intended Use

- Biomechanics research
- Sports science
- Clinical performance assessment
- Rehabilitation monitoring
- Academic projects and theses

⚠️ **Not intended for direct clinical decision-making without expert validation.**

---

## License

This project is currently shared for **academic and research purposes**.  
License details will be added once the project stabilizes.

---

##  Author

**Beigoll A Beigoll Feuba**  
B.Sc. Biomedical Engineering  
Interested in biomechanics, motion analysis, and applied data science.

---

##  Contributions

Contributions, suggestions, and feedback are welcome .
