"""
Example: Integrating 4x4 Binned Frequency Analysis into a workflow (Updated)

This script shows how to incorporate the binned analysis into an existing
pipeline. Exports are Excel-first, with JPG visualizations, and defaults
match the application (window='hann', PPM=511).
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.processors.spectral_functions import (
    compute_binned_freq_analysis,
    export_binned_analysis_to_csv,
    visualize_binned_analysis,
)


def process_with_binned_analysis(
    electrophys_file: str,
    pos_file: str,
    ppm: int,
    chunk_size: int,
    window_type: str,
    output_dir: str,
    scaling_factor_perBand: dict | None,
    chunk_pows_perBand: dict | None,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_t: np.ndarray,
    fs: int,
    chunks: list,
):
    print("\n" + "=" * 60)
    print("COMPUTING 4x4 BINNED FREQUENCY ANALYSIS")
    print("=" * 60)

    binned_data = compute_binned_freq_analysis(
        pos_x,
        pos_y,
        pos_t,
        fs=fs,
        chunks=chunks,
        chunk_size=chunk_size,
        chunk_pows_perBand=chunk_pows_perBand,
        scaling_factor_perBand=scaling_factor_perBand,
        window_type=window_type,
    )

    base_name = os.path.splitext(os.path.basename(electrophys_file))[0]
    output_prefix = os.path.join(output_dir, f"{base_name}_binned_analysis")

    print("\n  → Exporting binned analysis tables (Excel)…")
    export_binned_analysis_to_csv(binned_data, output_prefix)

    print("  → Creating visualization (JPG)…")
    visualize_binned_analysis(binned_data, save_path=f"{output_prefix}_heatmap.jpg")

    # Optional: summary
    total_chunks = binned_data.get("time_chunks", 0)
    bands = binned_data.get("bands", [])
    print(f"\n  SUMMARY: {total_chunks} chunks, {len(bands)} bands → {', '.join(bands)}")


if __name__ == "__main__":
    # Minimal demo runner when executed directly
    out_dir = PROJECT_ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic position and EEG example
    PPM = 511
    FS = 30000
    CHUNK_SIZE = 10
    WINDOW = "hann"

    t = np.linspace(0, 60, 60 * 50)
    pos_x = 128 + 80 * np.sin(2 * np.pi * t / 20)
    pos_y = 128 + 80 * np.cos(2 * np.pi * t / 25)
    pos_t = t

    rng = np.random.default_rng(42)
    eeg = rng.standard_normal(int(t[-1]) * FS)
    samples_per_chunk = FS * CHUNK_SIZE
    chunks = [eeg[i * samples_per_chunk : (i + 1) * samples_per_chunk] for i in range(len(eeg) // samples_per_chunk)]

    process_with_binned_analysis(
        electrophys_file="demo.eeg",
        pos_file="demo.pos",
        ppm=PPM,
        chunk_size=CHUNK_SIZE,
        window_type=WINDOW,
        output_dir=str(out_dir),
        scaling_factor_perBand=None,
        chunk_pows_perBand=None,
        pos_x=pos_x,
        pos_y=pos_y,
        pos_t=pos_t,
        fs=FS,
        chunks=chunks,
    )
