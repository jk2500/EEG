#!/usr/bin/env python3
"""Quick script to check EEG file durations."""

import mne
from pathlib import Path

mne.set_log_level('WARNING')

# Check the files
files = {
    'awake_EO': 'ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr',
    'awake_EC': 'ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EC_eeg.vhdr',
    'sedation_1': 'ds005620/sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr',
}

print("\nEEG File Durations:")
print("=" * 70)

for name, path in files.items():
    file_path = Path(path)
    if file_path.exists():
        raw = mne.io.read_raw_brainvision(str(file_path), preload=False, verbose=False)
        duration = raw.times[-1]
        n_channels = len(raw.ch_names)
        sfreq = raw.info['sfreq']
        
        print(f"\n{name}:")
        print(f"  Duration:    {duration:.1f} seconds ({duration/60:.2f} minutes)")
        print(f"  Channels:    {n_channels}")
        print(f"  Samp. freq:  {sfreq:.1f} Hz")
        print(f"  Epochs with different lengths:")
        for epoch_len in [2, 5, 10, 20, 30]:
            n_epochs = int(duration / epoch_len)
            print(f"    {epoch_len:>3}s epochs: {n_epochs:>4} epochs")
    else:
        print(f"\n{name}: File not found at {path}")

print("\n" + "=" * 70)

