# Batch Processing Feature - Browse Folder

## Overview
Added a "Browse folder" button to the GUI that allows users to process all EGF files in a selected directory in batch mode.

## Changes Made

### 1. **GUI Updates** (`src/main.py`)
   - Added "Browse folder" button next to the existing "Browse file" button
   - New button is positioned at the top-left of the interface (row 0, column 0)
   - Button uses the same styling and sizing as other UI controls

### 2. **Batch Worker Thread** (`src/main.py`)
   - Created new `BatchWorkerThread` class that extends `QThread`
   - Inherits from PyQt5's `QThread` for threaded processing
   - **Signals:**
     - `progress_update` - Text updates during processing
     - `progress_value` - Progress bar updates (0-100%)
     - `finished_signal` - Emitted when batch processing completes with results
     - `error_signal` - Emitted if errors occur during processing

   - **Key Methods:**
     - `find_recording_files()` - Discovers all EGF/EEG files in the folder with matching .pos files
     - `_find_pos_file()` - Auto-detects matching .pos file based on EGF/EEG filename
     - `run()` - Main processing loop that handles each file
     - `_export_to_csv()` - Exports results to CSV format

### 3. **GUI Integration Methods**
   - `browseFolderClicked()` - Opens folder selection dialog and initiates batch processing
   - `updateBatchLabel()` - Updates progress text during processing
   - `batchProcessingFinished()` - Shows completion summary with results
   - `batchProcessingError()` - Handles and displays error messages
   - `setButtonsEnabled()` - Disables buttons during processing (except Quit)

## Features

### File Discovery
- Automatically finds all `.egf` and `.eeg` files in the selected folder
- Prioritizes `.egf` files over `.eeg` files (uses .egf if both exist for same recording)
- Skips numbered variants (egf2, egf3, egf4 if base file exists)
- Auto-locates matching `.pos` files based on filename
- Validates that both electrophysiology and position files exist

### Processing
- Uses the existing `initialize_fMap()` function for consistent analysis
- Applies the same parameters (PPM, chunk size, window type, speed filter) to all files
- Processes files sequentially in a separate thread to prevent GUI freezing
- Real-time progress updates during processing

### CSV Export
- Automatically exports results to CSV format for each file
- Filename format: `{original_filename}_SSM.csv`
- Includes columns for:
  - Time bins
  - Distance per bin and cumulative distance (based on position tracking)
  - Power values for each frequency band (Delta, Theta, Beta, Low Gamma, High Gamma, Ripple, Fast Ripple)
  - Percentage contribution of each band to total signal power

### User Feedback
- Progress bar shows overall batch completion
- Text label shows current file being processed
- Summary dialog displays:
  - Total number of files processed
  - Number of successful and failed files
  - List of processed files (first 5 shown)
  - Error details for failed files (first 3 shown)
  - Output directory path

## Parameter Validation
Before batch processing starts, the system validates:
- PPM (pixels per meter) is set
- Chunk size is set
- Speed filter bounds are in ascending order (if specified)

## Memory Management
- Large data structures are deleted after each file to free memory
- Explicit garbage collection between files
- Memory usage is optimized for processing multiple large files sequentially

## Usage

1. **Set Processing Parameters:**
   - Configure PPM value (pixels per meter)
   - Set chunk size (in seconds)
   - Optionally set speed filter range (format: min,max in cm/s)
   - Select window type for FFT analysis
   - Select frequency band to display (for single-file mode)

2. **Click "Browse folder" button**

3. **Select the folder** containing EGF/EEG recordings

4. **Processing begins** automatically:
   - Progress bar shows overall completion
   - Status text shows which file is being processed
   - GUI remains responsive

5. **Completion Summary:**
   - Dialog displays results and statistics
   - CSV files are saved to the selected folder with `_SSM` suffix
   - File paths are `{filename}_SSM.csv` in the same directory

## Error Handling
- Missing `.pos` files are skipped with warning
- File processing errors are caught and reported
- Batch continues if individual files fail
- Summary shows which files failed and why
- Errors are logged to console and displayed in GUI

## Performance
- Batch processing runs in separate thread to prevent GUI blocking
- Progress updates in real-time
- Supports processing large folders with many recordings
- Memory is freed between files to handle multiple large recordings
