"""
Complete Working Example: 4x4 Binned Frequency Analysis (Updated)
===============================================================

This example demonstrates the 4x4 binned frequency analysis using the
current project defaults and export formats.

Updates reflected:
- Excel-first exports (with CSV fallback handled internally)
- JPG visualizations (quality managed in library code)
- Defaults: window_type='hann', PPM=511

Usage:
    python examples/binned_analysis_example.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project src/ is on path when running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.data_loaders import grab_position_data
from core.processors.spectral_functions import (
    compute_binned_freq_analysis,
    export_binned_analysis_to_csv,
    visualize_binned_analysis,
)


class SimpleSignals:
    class Signal:
        def emit(self, msg):
            print(f"  [Progress] {msg}")

    def __init__(self):
        self.text_progress = self.Signal()
        self.progress = self.Signal()


def main():
    # ========== CONFIGURATION ==========
    # Replace with your own data files
    POS_FILE = "path/to/your/data.pos"
    EEG_FILE = "path/to/your/data.eeg"

    PPM = 511                  # Pixels per meter (default)
    FS = 30000                 # EEG sampling rate (Hz)
    CHUNK_SIZE = 10            # seconds
    WINDOW_TYPE = "hann"       # default window

    print("=" * 70)
    print("LOADING POSITION AND EEG DATA")
    print("=" * 70)

    print(f"Loading position data from: {POS_FILE}")
    try:
        pos_x, pos_y, pos_t, pos_width = grab_position_data(POS_FILE, PPM)
        print(f"✓ Loaded {len(pos_x)} position samples")
    except Exception as e:
        print(f"✗ Failed to load position: {e}")
        # Minimal synthetic walk for demo
        t = np.linspace(0, 120, 120 * 50)
        pos_x = 128 + 100 * np.sin(2 * np.pi * t / 60)
        pos_y = 128 + 100 * np.cos(2 * np.pi * t / 60)
        pos_t = t
        pos_width = 256

    print(f"\nLoading EEG data from: {EEG_FILE}")
    try:
        eeg_data = np.fromfile(EEG_FILE, dtype=np.int16)
        print(f"✓ Loaded {len(eeg_data)} EEG samples")
    except Exception:
        print("✗ Could not load EEG file. Using random data for demo…")
        duration_seconds = int(pos_t[-1]) if len(pos_t) else 120
        rng = np.random.default_rng(0)
        eeg_data = rng.standard_normal(duration_seconds * FS).astype(np.float32)

    # Chunk EEG
    samples_per_chunk = FS * CHUNK_SIZE
    n_chunks = max(1, len(eeg_data) // samples_per_chunk)
    chunks = [
        eeg_data[i * samples_per_chunk : (i + 1) * samples_per_chunk]
        for i in range(n_chunks)
    ]

    print("\n" + "=" * 70)
    print("COMPUTING 4x4 BINNED ANALYSIS")
    print("=" * 70)

    # In typical app flow, scaling and band powers are precomputed. For this
    # example we call compute_binned_freq_analysis with minimal inputs.
    binned_data = compute_binned_freq_analysis(
        pos_x,
        pos_y,
        pos_t,
        fs=FS,
        chunks=chunks,
        chunk_size=CHUNK_SIZE,
        chunk_pows_perBand=None,
        scaling_factor_perBand=None,
        window_type=WINDOW_TYPE,
    )

    base_name = "demo_session"
    out_dir = PROJECT_ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(out_dir / f"{base_name}_binned_analysis")

    # Excel-first export is handled internally (CSV fallback if needed)
    print("\n→ Exporting binned analysis tables (Excel)…")
    export_binned_analysis_to_csv(binned_data, output_prefix)

    # JPG visualization (handled internally; uses consistent colormap)
    print("→ Creating visualization (JPG)…")
    visualize_binned_analysis(binned_data, save_path=f"{output_prefix}_heatmap.jpg")

    print("\n✓ Example complete. See outputs in:", out_dir)


if __name__ == "__main__":
    main()
