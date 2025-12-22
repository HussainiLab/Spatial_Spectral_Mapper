# Spatial_Spectral_Mapper
## Map frequency power spectrums from neural eeg/egf data as a function of position

### Requirements
- **Python 3.13** (Python 3.11+ recommended)
- **Windows** (tested on Windows 10/11)
- PyQt5 for GUI
- NumPy 2.x, SciPy 1.16+, Matplotlib 3.8+
- See `requirements.txt` for full dependency list

### Quick Start

1) Install Python 3.13 (Windows 10/11).
2) Clone the repo and install dependencies:
   - `pip install -r requirements.txt`
3) Run the GUI:
   - `python src/main.py`
4) Batch mode (optional):
   - `python src/batch_ssm.py path/to/recording.eeg --export-binned-jpgs`

See requirements.txt for the full list of packages.

### Features
- **Real-time frequency band analysis** (Delta, Theta, Beta, Gamma, Ripple, Fast Ripple)
- **Spatial heatmap visualization** with occupancy normalization
- **Binned Analysis Studio** (NEW) - Advanced 4×4 spatial binning with:
  - Interactive time-chunk slider for temporal navigation
  - Toggle between absolute power and percent power views
  - Occupancy heatmap and dominant frequency band visualization
  - Export mean power, percent power, occupancy, and dominant band data to Excel
  - Export all per-chunk visualizations as JPG files
- **Speed filtering** - Analyze only data within specific speed ranges
- **Excel export** for all data outputs (automatic CSV fallback if openpyxl unavailable):
  - Time bins with distance per bin and cumulative distance (in cm)
  - Average power per frequency band
  - Percentage of total power per band
  - Multi-sheet workbooks for binned analysis metrics
- **Interactive time slider** to view frequency maps across recording duration
- **Graph mode** for power spectral density visualization

### Usage

#### GUI Mode (Interactive)
1. Launch the application: `python src/main.py`
2. Click **Browse file** to select your `.eeg` or `.egf` file (`.pos` file will be auto-detected)
3. Configure parameters:
   - **PPM** (pixels per meter): Spatial resolution of position data (default: 511)
   - **Chunk Size**: Time window for analysis in seconds (default: 10)
   - **Window Type**: FFT window function (default: hann)
   - **Speed Filter**: Analyze only movement within specified speed range (cm/s)
4. Click **Render** to process the data
5. Use the **time slider** to navigate through different time bins
6. Click **Save data** to export main analysis as Excel file
7. Click **Binned Analysis Studio** to open advanced spatial binning interface

#### Binned Analysis Studio
The Binned Analysis Studio provides detailed 4×4 spatial bin analysis:

**Features:**
- **Time-chunk navigation**: Slider to view data across different time windows
- **View modes**:
  - Frequency band power (absolute or percent)
  - Occupancy heatmap (time spent in each bin)
  - Dominant frequency band per bin
- **Toggle percent power**: Switch between absolute power values and percentage of total power
- **Export Data**: Save binned metrics to Excel files:
  - `_mean_power.xlsx` - Mean power per band (one sheet per band)
  - `_percent_power.xlsx` - Percent power per band (one sheet per band)
  - `_occupancy.xlsx` - Time spent in each spatial bin
  - `_dominant_band.xlsx` - Frequency of dominant band per bin (one sheet per band)
- **Export All JPGs**: Save all visualizations:
  - Mean power heatmaps per chunk (one per time chunk)
  - Percent power heatmaps per chunk
  - Occupancy heatmap (time-invariant, saved once)
  - Dominant band heatmaps per chunk
  - All files saved to `binned_analysis_output/` subfolder

**Output Structure (GUI):**
```
output_folder/
├── {filename}_SSM.xlsx                    # Main analysis data
├── {filename}_binned_mean_power.xlsx      # Binned mean power
├── {filename}_binned_percent_power.xlsx   # Binned percent power
├── {filename}_binned_occupancy.xlsx       # Occupancy data
├── {filename}_binned_dominant_band.xlsx   # Dominant band frequencies
├── {filename}_binned_heatmap.jpg          # Combined visualization
└── binned_analysis_output/                # (When "Export All JPGs" is used)
    ├── {filename}_chunk0_mean_power.jpg
    ├── {filename}_chunk0_percent_power.jpg
    ├── {filename}_chunk0_dominant_band.jpg
    ├── {filename}_chunk1_mean_power.jpg
    ├── ... (one set per chunk)
    └── {filename}_occupancy.jpg
```

### Examples
- examples/binned_analysis_example.py: end‑to‑end demo that runs the 4×4 binned analysis and writes Excel tables plus JPG visualizations (defaults: window=hann, PPM=511). Outputs to examples/outputs/.
- examples/example_binned_analysis.py: integration pattern for invoking binned analysis inside a pipeline (Excel‑first exports + JPG heatmap).

#### Batch Mode (No GUI)

Basic usage:
```bash
python src/batch_ssm.py path/to/recording.eeg --export-binned-jpgs -o ./output/
```

Key flags:
- `--ppm 511` (default), `--chunk-size 10`, `--window hann`
- `--speed-filter 2,30` (cm/s), `-o ./output/`
- `--export-binned-jpgs`: also writes per‑chunk JPGs and Excel to a subfolder

**Output Structure (Batch Mode):**

*Standard mode (without --export-binned-jpgs):*
```
output_folder/
├── {filename}_SSM.xlsx                    # Main analysis Excel
├── {filename}_binned_mean_power.xlsx      # Binned analysis Excel files
├── {filename}_binned_percent_power.xlsx
├── {filename}_binned_occupancy.xlsx
├── {filename}_binned_dominant_band.xlsx
└── {filename}_binned_heatmap.jpg          # Combined visualization
```

*With --export-binned-jpgs flag:*
```
output_folder/
├── {filename}_SSM.xlsx                    # Main analysis Excel
└── {filename}_binned_analysis/            # All binned outputs in subfolder
    ├── {filename}_binned_mean_power.xlsx
    ├── {filename}_binned_percent_power.xlsx
    ├── {filename}_binned_occupancy.xlsx
    ├── {filename}_binned_dominant_band.xlsx
    ├── {filename}_binned_heatmap.jpg
    ├── {filename}_chunk0_mean_power.jpg   # Per-chunk visualizations
    ├── {filename}_chunk0_percent_power.jpg
    ├── {filename}_chunk0_dominant_band.jpg
    ├── {filename}_chunk1_mean_power.jpg
    ├── ... (one set per chunk)
    └── {filename}_occupancy.jpg
```

**Example Script for Multiple Files:**
```python
import subprocess
import glob

# Process all .eeg files with full binned analysis export
for eeg_file in glob.glob("data/*.eeg"):
    subprocess.run([
        "python", "src/batch_ssm.py", 
        eeg_file,
        "--ppm", "511",
        "--chunk-size", "10",
        "--export-binned-jpgs",
        "-o", "results/"
    ])
```
### Excel/CSV Output Format

### Examples

Two quick-start scripts are included in the repository:

- examples/binned_analysis_example.py: End-to-end demo that runs the 4×4 binned analysis and writes Excel tables plus JPG visualizations using current defaults (window=hann, PPM=511).
- examples/example_binned_analysis.py: Integration pattern showing how to invoke the binned analysis from an existing pipeline and export results (Excel-first plus JPG heatmap).

Outputs are written under examples/outputs/ by default when running the examples.

**Main SSM Data (_SSM.xlsx):**
- **Time Bin (s)**: Time interval (e.g., "0-10", "10-20")
- **Distance Per Bin (cm)**: Distance traveled within that time bin
- **Cumulative Distance (cm)**: Total distance from start
- **Avg [Band] Power**: Average power for each frequency band
- **Percent [Band]**: Percentage of total power for each band

**Binned Analysis Data:**
- **Mean Power (_mean_power.xlsx)**: 4×4 grid of mean power per spatial bin, one sheet per frequency band
- **Percent Power (_percent_power.xlsx)**: 4×4 grid of percent power per spatial bin, one sheet per frequency band
- **Occupancy (_occupancy.xlsx)**: 4×4 grid showing time spent (in samples) per spatial bin
- **Dominant Band (_dominant_band.xlsx)**: 4×4 grid showing frequency of dominance per spatial bin, one sheet per band

### Notes
- **Excel export is now the default** for all outputs (openpyxl automatically installed if missing, CSV fallback available)
- **Binned analysis** automatically exported during processing (GUI and batch mode)
- **Per-chunk visualizations** available via "Export All JPGs" in GUI or `--export-binned-jpgs` flag in batch mode
- Position data (`.pos` files) should be sampled at 50 Hz
- EEG data formats supported: Axona `.eeg` (250 Hz) and `.egf` (1200 Hz)
- **Batch processing**: Handles position data files with mismatched timestamp arrays (automatically trims or extends as needed)
- **Directory mode**: When processing a directory, prioritizes `.egf` files over `.eeg` and skips duplicate variants (e.g., .egf2-.egf4)

### Troubleshooting
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Qt errors**: Make sure PyQt5 is properly installed
- **Distance calculation issues**: Verify PPM value matches your position tracking system

### License
See LICENSE file for details. 

