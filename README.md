# Spatial_Spectral_Mapper
## Map frequency power spectrums from neural eeg/egf data as a function of position

### Requirements
- **Python 3.13** (Python 3.11+ recommended)
- **Windows** (tested on Windows 10/11)
- PyQt5 for GUI
- NumPy 2.x, SciPy 1.16+, Matplotlib 3.8+
- See `requirements.txt` for full dependency list

### Installation

#### Recommended: Using Conda (Easiest Method)

1. **Install Anaconda or Miniconda** (if not already installed):
   - Download from [anaconda.com](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. **Clone or download this repository**:
   ```bash
   git clone https://github.com/yourusername/Spatial_Spectral_Mapper.git
   cd Spatial_Spectral_Mapper
   ```

3. **Create a new conda environment with Python 3.13**:
   ```bash
   conda create -n ssm_env python=3.13
   ```

4. **Activate the environment**:
   ```bash
   conda activate ssm_env
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the application**:
   ```bash
   cd src
   python main.py
   ```

---

#### Alternative: Using venv (Built-in Python)

1. **Ensure Python 3.13 is installed**:
   - Download from [python.org/downloads](https://www.python.org/downloads/)
   - Verify installation: `python --version`

2. **Clone or download this repository**:
   ```bash
   git clone https://github.com/yourusername/Spatial_Spectral_Mapper.git
   cd Spatial_Spectral_Mapper
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv ssm_venv
   ```

4. **Activate the virtual environment**:
   - **Windows**: `ssm_venv\Scripts\activate`
   - **macOS/Linux**: `source ssm_venv/bin/activate`

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the application**:
   ```bash
   cd src
   python main.py
   ```

---

### Features
- Real-time frequency band analysis (Delta, Theta, Beta, Gamma, Ripple, Fast Ripple)
- Spatial heatmap visualization with occupancy normalization
- Speed filtering (analyze only data within specific speed ranges)
- CSV export with:
  - Time bins
  - Distance per bin and cumulative distance (in cm)
  - Average power per frequency band
  - Percentage of total power per band
- Interactive time slider to view frequency maps across recording duration
- Graph mode for power spectral density visualization

### Usage
1. Launch the application: `python src/main.py`
2. Click **Browse file** to select your `.eeg` or `.egf` file (`.pos` file will be auto-detected)
3. Configure parameters:
   - **PPM** (pixels per meter): Spatial resolution of position data (default: 600)
   - **Chunk Size**: Time window for analysis in seconds (default: 10)
   - **Speed Filter**: Analyze only movement within specified speed range (cm/s)
4. Click **Render** to process the data
5. Use the **time slider** to navigate through different time bins
6. Click **Save** to export data as CSV

### CSV Output Format
- **Time Bin (s)**: Time interval (e.g., "0-10", "10-20")
- **Distance Per Bin (cm)**: Distance traveled within that time bin
- **Cumulative Distance (cm)**: Total distance from start
- **Avg [Band] Power**: Average power for each frequency band
- **Percent [Band]**: Percentage of total power for each band

### Notes
- Excel is **NOT required** for CSV export (updated from previous versions)
- Position data (`.pos` files) should be sampled at 50 Hz
- EEG data formats supported: Axona `.eeg` (250 Hz) and `.egf` (1200 Hz)

### Troubleshooting
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Qt errors**: Make sure PyQt5 is properly installed
- **Distance calculation issues**: Verify PPM value matches your position tracking system

### License
See LICENSE file for details. 

