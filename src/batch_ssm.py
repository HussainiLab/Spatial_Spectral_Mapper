#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch processing script for Spatial Spectral Mapper (SSM)
Allows command-line analysis of LFP data without GUI
"""

import os
import sys
import argparse
import csv
import numpy as np
from pathlib import Path
import gc  # Garbage collection for memory management
import psutil  # For memory monitoring
import subprocess

# Set matplotlib to non-interactive backend BEFORE importing anything else
import matplotlib
from core.processors.spectral_functions import export_binned_analysis_to_csv, visualize_binned_analysis, export_binned_analysis_jpgs
matplotlib.use('Agg')  # Use Agg backend - no display needed
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Set Qt to offscreen mode

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import Qt with QGuiApplication (offscreen mode)
from PyQt5.QtGui import QGuiApplication
app = QGuiApplication(sys.argv)  # Create app in offscreen mode

from src.initialize_fMap import initialize_fMap
from src.core.data_loaders import grab_position_data
from src.core.processors.Tint_Matlab import speed2D
from src.core.processors.spectral_functions import speed_bins


class BatchWorker:
    """Worker class for batch processing without GUI"""
    
    class MockSignals:
        """Mock signal object for batch processing (replaces Qt signals)"""
        def __init__(self):
            self.progress_value = 0
            self.progress_text = ""
        
        def emit(self, value=None):
            """Mock signal emit - just log the value"""
            if isinstance(value, str):
                self.progress_text = value
                print(f"  → {value}")
            else:
                self.progress_value = value
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.progress_messages = []
        self.signals = self.MockSignals()
        
        # Create nested structure for mock signals
        class ProgressSignals:
            def __init__(self, worker_signals):
                self.worker_signals = worker_signals
            
            def emit(self, value):
                if isinstance(value, str):
                    self.worker_signals.progress_text = value
                    print(f"  → {value}")
                else:
                    self.worker_signals.progress_value = value
        
        class TextProgressSignals:
            def __init__(self, worker_signals):
                self.worker_signals = worker_signals
            
            def emit(self, value):
                self.worker_signals.progress_text = value
                print(f"  → {value}")
        
        self.signals.progress = ProgressSignals(self.signals)
        self.signals.text_progress = TextProgressSignals(self.signals)
    
    def log(self, message):
        """Log progress messages"""
        print(f"[SSM] {message}")
        self.progress_messages.append(message)


def find_pos_file(eeg_file):
    """Auto-detect .pos file based on .eeg/.egf filename"""
    base_name = os.path.splitext(eeg_file)[0]
    pos_file = base_name + '.pos'
    
    if os.path.exists(pos_file):
        return pos_file
    else:
        raise FileNotFoundError(f"Position file not found: {pos_file}")


def find_recording_files(directory):
    """
    Find all EEG/EGF recording files in a directory.
    Returns list of (egf/eeg file, pos file) tuples.
    
    Priority: Use .egf if exists, otherwise .eeg
    Skip numbered variants (egf2-4, eeg2-4) if base file exists
    """
    import glob
    from pathlib import Path
    
    recordings = {}  # basename -> (electrophys_file, pos_file)
    
    # Find all .egf and .eeg files
    egf_files = glob.glob(os.path.join(directory, "*.egf"))
    eeg_files = glob.glob(os.path.join(directory, "*.eeg"))
    
    # Process EGF files first (higher priority)
    for egf_file in egf_files:
        basename = Path(egf_file).stem
        # Skip numbered variants (egf2, egf3, egf4)
        if basename.endswith(('2', '3', '4')) and basename[:-1] in [Path(f).stem for f in egf_files]:
            continue
        
        try:
            pos_file = find_pos_file(egf_file)
            recordings[basename] = (egf_file, pos_file)
        except FileNotFoundError:
            print(f"⚠ Skipping {os.path.basename(egf_file)} - no .pos file found")
    
    # Process EEG files (only if no EGF exists for same basename)
    for eeg_file in eeg_files:
        basename = Path(eeg_file).stem
        
        # Skip if already have EGF for this basename
        if basename in recordings:
            continue
        
        # Skip numbered variants (eeg2, eeg3, eeg4)
        if basename.endswith(('2', '3', '4')) and basename[:-1] in [Path(f).stem for f in eeg_files]:
            continue
        
        try:
            pos_file = find_pos_file(eeg_file)
            recordings[basename] = (eeg_file, pos_file)
        except FileNotFoundError:
            print(f"⚠ Skipping {os.path.basename(eeg_file)} - no .pos file found")
    
    return list(recordings.values())


def export_to_csv(output_path, pos_t, chunk_powers_data, chunk_size, ppm, 
                  pos_x_chunks, pos_y_chunks):
    """Export analysis results to Excel (or CSV fallback)"""
    
    # Try to use Excel; fallback to CSV
    try:
        import openpyxl
        use_excel = True
    except ImportError:
        use_excel = False
    
    bands = [
        ("Delta", "Avg Delta Power"),
        ("Theta", "Avg Theta Power"),
        ("Beta", "Avg Beta Power"),
        ("Low Gamma", "Avg Low Gamma Power"),
        ("High Gamma", "Avg High Gamma Power"),
        ("Ripple", "Avg Ripple Power"),
        ("Fast Ripple", "Avg Fast Ripple Power"),
    ]
    
    band_labels = [label for _, label in bands]
    percent_labels = [f"Percent {name}" for name, _ in bands]
    
    # Calculate distances
    distances_per_bin = []
    cumulative_distances = []
    cumulative_sum = 0.0
    
    if pos_x_chunks is not None and pos_y_chunks is not None:
        for i in range(len(pos_x_chunks)):
            distance_cm_in_bin = 0.0
            
            if i == 0:
                x_bin = pos_x_chunks[i]
                y_bin = pos_y_chunks[i]
            else:
                prev_len = len(pos_x_chunks[i-1])
                x_bin = pos_x_chunks[i][prev_len:]
                y_bin = pos_y_chunks[i][prev_len:]
            
            if len(x_bin) > 1:
                dx = np.diff(np.array(x_bin))
                dy = np.diff(np.array(y_bin))
                distances_in_bin_pixels = np.sqrt(dx**2 + dy**2)
                total_distance_pixels_in_bin = np.sum(distances_in_bin_pixels)
                
                if ppm is not None and ppm > 0:
                    distance_cm_in_bin = (total_distance_pixels_in_bin / ppm) * 100
                else:
                    distance_cm_in_bin = total_distance_pixels_in_bin
            
            distances_per_bin.append(distance_cm_in_bin)
            cumulative_sum += distance_cm_in_bin
            cumulative_distances.append(cumulative_sum)
    
    # Determine max full chunks
    actual_duration = float(pos_t[-1])
    max_full_chunks = int(actual_duration / chunk_size)
    num_rows = min(len(pos_t), max_full_chunks)
    
    # Gather per-band arrays
    band_arrays = {}
    for key, label in bands:
        arr = np.array(chunk_powers_data.get(key, [])).reshape(-1)
        band_arrays[label] = arr
    
    # Prepare header and data rows
    header = ["Time Bin (s)", "Distance Per Bin (cm)", "Cumulative Distance (cm)"] + band_labels + percent_labels
    data_rows = []
    
    for i in range(num_rows):
        time_bin_start = i * chunk_size
        time_bin_end = (i + 1) * chunk_size
        time_bin_str = f"{time_bin_start}-{time_bin_end}"
        
        row = [time_bin_str]
        
        # Add distance per bin
        if distances_per_bin and i < len(distances_per_bin):
            row.append(round(distances_per_bin[i], 3))
        else:
            row.append("")
        
        # Add cumulative distance
        if cumulative_distances and i < len(cumulative_distances):
            row.append(round(cumulative_distances[i], 3))
        else:
            row.append("")
        
        band_values = []
        for _, label in bands:
            val = band_arrays[label][i] if i < len(band_arrays[label]) else ""
            band_values.append(val)
        
        row.extend(band_values)
        
        numeric_vals = [float(v) for v in band_values if v != ""]
        total_power = sum(numeric_vals)
        
        for v in band_values:
            if v == "" or total_power == 0:
                row.append("")
            else:
                row.append(round((float(v) / total_power) * 100.0, 3))
        
        data_rows.append(row)
    
    # Write Excel or CSV
    if use_excel:
        output_file = output_path.replace('.csv', '.xlsx')
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'SSM Data'
        ws.append(header)
        for row in data_rows:
            ws.append(row)
        wb.save(output_file)
        print(f"✓ Excel exported to: {output_file}")
    else:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in data_rows:
                writer.writerow(row)
        print(f"✓ CSV exported to: {output_path} (openpyxl not installed)")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process LFP data with Spatial Spectral Mapper (SSM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python batch_ssm.py data/recording.eeg
  
  # Process all files in a directory
  python batch_ssm.py data/recordings/
  
  # With custom parameters
  python batch_ssm.py data/recording.eeg --ppm 500 --chunk-size 5 --speed-filter 5,20
  
  # Batch directory with custom output
  python batch_ssm.py data/recordings/ --ppm 600 -o results/
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to .eeg/.egf file OR directory containing multiple recordings"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for CSV file(s) (default: same directory as input file(s))"
    )
    
    parser.add_argument(
        "--ppm",
        type=int,
        default=600,
        help="Pixels per meter for position data (default: 600)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Chunk size in seconds for analysis (default: 10)"
    )
    
    parser.add_argument(
        "--speed-filter",
        type=str,
        default="0,100",
        help="Speed filter range in cm/s (format: min,max, default: 0,100)"
    )
    
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="Window type for FFT (default: hann)"
    )
    
    parser.add_argument(
        "--export-binned-jpgs",
        action="store_true",
        help="Export per-chunk JPG visualizations for binned analysis (mean power, percent power, dominant band per chunk; occupancy once)"
    )
    
    args = parser.parse_args()
    
    # Parse speed filter
    try:
        speed_range = args.speed_filter.split(',')
        low_speed = float(speed_range[0])
        high_speed = float(speed_range[1])
    except (ValueError, IndexError):
        print(f"✗ Error: Invalid speed filter format. Use 'min,max' (e.g., '5,20')")
        sys.exit(1)
    
    # Check if input is a directory or file
    if os.path.isdir(args.input_path):
        # Directory mode - process all recordings
        print(f"\n{'='*60}")
        print("Spatial Spectral Mapper - Batch Directory Processing")
        print(f"{'='*60}")
        print(f"Directory: {args.input_path}")
        print(f"Parameters: PPM={args.ppm}, Chunk={args.chunk_size}s, Speed={low_speed}-{high_speed} cm/s")
        print(f"{'='*60}\n")
        
        recordings = find_recording_files(args.input_path)
        
        if not recordings:
            print("✗ No valid recording files found in directory.")
            sys.exit(1)
        
        print(f"Found {len(recordings)} recording(s) to process\n")
        
        success_count = 0
        fail_count = 0
        
        for idx, (electrophys_file, pos_file) in enumerate(recordings, 1):
            print(f"\n[{idx}/{len(recordings)}] Processing: {os.path.basename(electrophys_file)}")
            print("-" * 60)
            
            try:
                process_single_file(
                    electrophys_file, 
                    pos_file, 
                    args.output or os.path.dirname(electrophys_file),
                    args.ppm,
                    args.chunk_size,
                    low_speed,
                    high_speed,
                    args.window
                )
                success_count += 1
            except Exception as e:
                print(f"✗ Failed: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"✓ Successful: {success_count}")
        if fail_count > 0:
            print(f"✗ Failed: {fail_count}")
        print(f"{'='*60}")
        
    else:
        # Single file mode
        if not os.path.exists(args.input_path):
            print(f"✗ Error: File not found: {args.input_path}")
            sys.exit(1)
        
        try:
            pos_file = find_pos_file(args.input_path)
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
        
        output_dir = args.output or os.path.dirname(args.input_path) or "."
        
        print(f"\n{'='*60}")
        print("Spatial Spectral Mapper - Single File Processing")
        print(f"{'='*60}")
        
        try:
            process_single_file(
                args.input_path, 
                pos_file, 
                output_dir,
                args.ppm,
                args.chunk_size,
                low_speed,
                high_speed,
                args.window,
                export_binned_jpgs=args.export_binned_jpgs
            )
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timeout exceeded")

def process_single_file(electrophys_file, pos_file, output_dir, ppm, chunk_size, 
                        low_speed, high_speed, window_type, export_binned_jpgs=False, timeout_seconds=300):
    """Process a single EEG/EGF file with timeout protection"""
    
    # Monitor memory at start
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(electrophys_file))[0]
    output_csv = os.path.join(output_dir, f"{base_name}_SSM.csv")
    
    print(f"Input:  {electrophys_file}")
    print(f"Output: {output_csv}")
    print(f"PPM: {ppm}, Chunk: {chunk_size}s, Speed: {low_speed}-{high_speed} cm/s")
    print(f"Memory: {mem_before:.1f} MB")
    
    # Create batch worker with mock signals
    worker = BatchWorker(output_dir)
    
    print("[1/5] Starting data processing...")
    print(f"  Timeout set to {timeout_seconds} seconds")
    
    try:
        # Set timeout alarm (Windows doesn't support signal.alarm, so we check timing manually)
        # Run initialization with worker as 'self' parameter
        result = initialize_fMap(
            worker,
            files=[pos_file, electrophys_file],
            ppm=ppm,
            chunk_size=chunk_size,
            window_type=window_type,
            low_speed=low_speed,
            high_speed=high_speed
        )
        print("[2/5] Data processing complete")
    except Exception as e:
        print(f"\n✗ Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Unpack legacy/new result signature
    if len(result) == 6:
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_pows_data, tracking_data = result
        binned_data = None
    elif len(result) == 7:
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_pows_data, tracking_data, binned_data = result
    else:
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_pows_data, tracking_data, binned_data, arena_shape = result
    
    print("[3/5] Extracting tracking data...")
    # Extract tracking data
    pos_x_chunks, pos_y_chunks = tracking_data if tracking_data else (None, None)
    
    print("[4/5] Exporting to CSV...")
    # Export to CSV
    export_to_csv(
        output_csv,
        pos_t,
        chunk_pows_data,
        chunk_size,
        ppm,
        pos_x_chunks,
        pos_y_chunks
    )
    print("[5/5] SSM data export complete")

    # Export binned analysis outputs if available
    try:
        if binned_data is not None:
            if binned_data.get('type') == 'polar':
                print("  ⚠ Polar binned analysis detected. Batch export for polar data is skipped (GUI only).")
                return

            # Determine output location based on whether per-chunk export is requested
            if export_binned_jpgs:
                # Export everything (Excel + JPGs) to subfolder
                output_folder = os.path.join(output_dir, f"{base_name}_binned_analysis")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)
                output_prefix = os.path.join(output_folder, base_name + "_binned")
                
                print("[+] Exporting 4x4 binned analysis with per-chunk JPGs and Excel...")
                
                # Export Excel files
                result = export_binned_analysis_to_csv(binned_data, output_prefix)
                if result.get('format') == 'csv' and result.get('reason') == 'openpyxl_not_installed':
                    print("  ⚠ Excel export unavailable (openpyxl missing). Attempting install...")
                    try:
                        proc = subprocess.run([sys.executable, '-m', 'pip', 'install', 'openpyxl'], capture_output=True, text=True)
                        if proc.returncode == 0:
                            print("  ✓ openpyxl installed. Re-exporting to Excel...")
                            result = export_binned_analysis_to_csv(binned_data, output_prefix)
                        else:
                            print("  ✗ Failed to install openpyxl. Keeping CSV export.")
                    except Exception as ie:
                        print(f"  ✗ Error installing openpyxl: {ie}")
                
                print(f"[+] Exported binned analysis Excel files ({result.get('format','csv').upper()})")
                for filepath in result.get('files', []):
                    print(f"    • {os.path.basename(filepath)}")
                
                # Export JPGs
                try:
                    print("[+] Exporting per-chunk JPG visualizations...")
                    jpg_count = export_binned_analysis_jpgs(binned_data, output_folder, base_name)
                    print(f"[+] Exported {jpg_count} JPG visualizations")
                except Exception as e:
                    print(f"  ⚠ Failed to export JPGs: {e}")
                
                # Export combined heatmap
                visualize_binned_analysis(binned_data, save_path=os.path.join(output_folder, f"{base_name}_binned_heatmap.jpg"))
                print(f"[+] All binned analysis files exported to: {output_folder}")
            else:
                # Standard export: Excel + single heatmap to main output dir
                output_prefix = os.path.splitext(output_csv)[0] + "_binned"
                print("[+] Exporting 4x4 binned analysis data and visualization...")
                result = export_binned_analysis_to_csv(binned_data, output_prefix)
                if result.get('format') == 'csv' and result.get('reason') == 'openpyxl_not_installed':
                    print("  ⚠ Excel export unavailable (openpyxl missing). Attempting install...")
                    try:
                        proc = subprocess.run([sys.executable, '-m', 'pip', 'install', 'openpyxl'], capture_output=True, text=True)
                        if proc.returncode == 0:
                            print("  ✓ openpyxl installed. Re-exporting to Excel...")
                            result = export_binned_analysis_to_csv(binned_data, output_prefix)
                        else:
                            print("  ✗ Failed to install openpyxl. Keeping CSV export.")
                    except Exception as ie:
                        print(f"  ✗ Error installing openpyxl: {ie}")
                visualize_binned_analysis(binned_data, save_path=output_prefix + "_heatmap.jpg")
                print(f"[+] Binned analysis export complete ({result.get('format','csv').upper()})")
    except Exception as e:
        print(f"⚠ Binned analysis export failed: {e}")
    
    # Print summary
    print(f"✓ Complete - {len(pos_t)} time bins processed")
    for band, power in scaling_factor_crossband.items():
        print(f"  {band:15s}: {power*100:6.2f}%")
    
    # Explicitly delete large objects to free memory
    del freq_maps, plot_data, pos_t, chunk_pows_data, tracking_data
    del pos_x_chunks, pos_y_chunks, worker
    
    # Force garbage collection to clear memory before next file
    gc.collect()
    
    # Monitor memory after cleanup
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after cleanup: {mem_after:.1f} MB (freed {mem_before - mem_after:.1f} MB)")


if __name__ == "__main__":
    main()
