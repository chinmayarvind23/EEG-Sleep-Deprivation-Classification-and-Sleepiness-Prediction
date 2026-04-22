
# ### Import Python Libraries

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


# ### Configuration

# =========================
# PATHS
# =========================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed_eeg_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DATA SETTINGS
# =========================
TASK = "eyesopen"
SESSIONS = ["ses-1", "ses-2"]

# Change this if your session labels differ
SESSION_LABEL_MAP = {
    "ses-1": {"label": 0, "label_name": "NS"},
    "ses-2": {"label": 1, "label_name": "SD"}
}

# =========================
# PREPROCESSING SETTINGS
# =========================
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
RESAMPLE_SFREQ = 250
EPOCH_LENGTH_SEC = 5.0
EPOCH_OVERLAP_SEC = 0.0

# =========================
# ARTIFACT SETTINGS
# =========================
CLIP_UV = 100.0
MAX_BAD_CH_FRACTION = 0.5

# =========================
# IMAGE SETTINGS
# =========================
USE_3_CHANNEL = True


# ### Finding EEG File

def find_set_file(subject_dir, session, task):
    eeg_dir = subject_dir / session / "eeg"

    if not eeg_dir.exists():
        return None

    files = sorted(eeg_dir.glob(f"*task-{task}_eeg.set"))
    return files[0] if len(files) > 0 else None


# ### Preprocessing Raw EEG

def preprocess_raw(raw):
    raw.load_data()

    # Keep only EEG channels
    raw.pick_types(eeg=True, exclude=[])

    # Average reference
    raw.set_eeg_reference("average", projection=False)

    # Bandpass filter
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)

    # Resample
    raw.resample(RESAMPLE_SFREQ)

    return raw


# ### Making 5-Second EEG Epochs

def make_epochs(data, sfreq):
    epoch_len = int(EPOCH_LENGTH_SEC * sfreq)
    step = int((EPOCH_LENGTH_SEC - EPOCH_OVERLAP_SEC) * sfreq)

    starts = range(0, data.shape[1] - epoch_len + 1, step)

    epochs = []
    for start in starts:
        end = start + epoch_len
        epochs.append(data[:, start:end])

    if len(epochs) == 0:
        return np.empty((0, data.shape[0], epoch_len), dtype=np.float32)

    return np.stack(epochs).astype(np.float32)


# ### Clipping Amplitudes and Rejecting Bad Epochs

def clip_and_reject(epochs):
    clip_v = CLIP_UV * 1e-6   # convert microvolt to volt

    # Clip values
    clipped_epochs = np.clip(epochs, -clip_v, clip_v)

    # Find saturation
    saturated = np.abs(epochs) >= clip_v
    frac_sat_per_channel = saturated.mean(axis=2)

    # Bad channel if more than 10% samples are saturated
    bad_channels = frac_sat_per_channel > 0.10

    # Fraction of bad channels in each epoch
    bad_ch_fraction = bad_channels.mean(axis=1)

    # Keep good epochs
    keep_mask = bad_ch_fraction <= MAX_BAD_CH_FRACTION

    return clipped_epochs, keep_mask


# ### Z-Score Normalizing Each Epoch

def zscore_epochs(epochs):
    mean = epochs.mean(axis=2, keepdims=True)
    std = epochs.std(axis=2, keepdims=True)

    std[std == 0] = 1.0

    z_epochs = (epochs - mean) / std
    return z_epochs.astype(np.float32)


# ### Converting EEG Epoch to Raw Image Tensor

def epoch_to_image(epoch_2d):
    x_min = epoch_2d.min()
    x_max = epoch_2d.max()

    # Scale to 0-255
    if x_max - x_min == 0:
        img = np.zeros(epoch_2d.shape, dtype=np.uint8)
    else:
        img = ((epoch_2d - x_min) / (x_max - x_min) * 255).astype(np.uint8)

    # Keep raw shape, no resizing
    if USE_3_CHANNEL:
        img_tensor = np.stack([img, img, img], axis=0)   # (3, C, T)
    else:
        img_tensor = img[np.newaxis, :, :]               # (1, C, T)

    return img_tensor


# ### Building the Dataset

def build_dataset():
    subject_dirs = sorted([p for p in DATA_DIR.glob("sub-*") if p.is_dir()])

    print(f"Found {len(subject_dirs)} subject folders")

    all_rows = []
    X_eeg_list = []
    X_image_list = []
    y_list = []
    groups_list = []

    total_rejected = 0

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject_id = subject_dir.name

        for session in SESSIONS:
            label = SESSION_LABEL_MAP[session]["label"]
            label_name = SESSION_LABEL_MAP[session]["label_name"]

            set_file = find_set_file(subject_dir, session, TASK)

            if set_file is None:
                continue

            try:
                # 1. Load EEG
                raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)

                # 2. Preprocess
                raw = preprocess_raw(raw)

                sfreq = float(raw.info["sfreq"])
                ch_names = raw.ch_names

                # 3. Get continuous EEG matrix
                data = raw.get_data()   # (C, T)

                # 4. Create epochs
                epochs_raw = make_epochs(data, sfreq)

                if len(epochs_raw) == 0:
                    continue

                # 5. Clip + reject bad epochs
                epochs_clipped, keep_mask = clip_and_reject(epochs_raw)
                total_rejected += (~keep_mask).sum()

                epochs_good = epochs_clipped[keep_mask]

                if len(epochs_good) == 0:
                    continue

                # 6. Normalize
                epochs_z = zscore_epochs(epochs_good)

                # 7. Store each epoch
                good_epoch_indices = np.where(keep_mask)[0]

                for i, original_epoch_index in enumerate(good_epoch_indices):
                    eeg_epoch = epochs_z[i]                  # (C, T)
                    image_tensor = epoch_to_image(eeg_epoch) # (3, C, T)

                    X_eeg_list.append(eeg_epoch)
                    X_image_list.append(image_tensor)
                    y_list.append(label)
                    groups_list.append(subject_id)

                    all_rows.append({
                        "subject_id": subject_id,
                        "session": session,
                        "task": TASK,
                        "label": label,
                        "label_name": label_name,
                        "epoch_index": int(original_epoch_index),
                        "sfreq": sfreq,
                        "n_channels": eeg_epoch.shape[0],
                        "n_times": eeg_epoch.shape[1],
                        "img_channels": image_tensor.shape[0],
                        "img_height": image_tensor.shape[1],
                        "img_width": image_tensor.shape[2],
                        "channel_names": ch_names,
                        "file_path": str(set_file),

                        # Main stored data
                        "eeg_epoch": eeg_epoch,
                        "image_tensor": image_tensor
                    })

            except Exception as e:
                print(f"Error in {set_file}: {e}")

    # Convert lists to arrays
    if len(X_eeg_list) > 0:
        X_eeg = np.stack(X_eeg_list).astype(np.float32)
        X_images = np.stack(X_image_list).astype(np.uint8)
        y = np.array(y_list, dtype=np.int64)
        groups = np.array(groups_list, dtype=object)
    else:
        X_eeg = np.empty((0, 0, 0), dtype=np.float32)
        X_images = np.empty((0, 3, 0, 0), dtype=np.uint8)
        y = np.empty((0,), dtype=np.int64)
        groups = np.empty((0,), dtype=object)

    df = pd.DataFrame(all_rows)

    return df, X_eeg, X_images, y, groups, total_rejected


# ### Running the Pipeline

df, X_eeg, X_images, y, groups, total_rejected = build_dataset()


# ### Checking the Results

print("Total epochs       :", len(df))
print("Rejected epochs    :", total_rejected)
print("X_eeg shape        :", X_eeg.shape)
print("X_images shape     :", X_images.shape)
print("y shape            :", y.shape)
print("groups shape       :", groups.shape)

if len(df) > 0:
    print("\nFirst row info:")
    print(df[[
        "subject_id", "session", "label_name",
        "epoch_index", "n_channels", "n_times",
        "img_channels", "img_height", "img_width"
    ]].head())


# ### Saving the Outputs

if len(df) > 0:
    # Save full dataframe
    df.to_pickle(OUTPUT_DIR / "eeg_epoch_dataframe.pkl")

    # Save metadata only
    df.drop(columns=["eeg_epoch", "image_tensor", "channel_names"], errors="ignore") \
      .to_csv(OUTPUT_DIR / "eeg_metadata.csv", index=False)

    # Save arrays
    np.save(OUTPUT_DIR / "X_eeg.npy", X_eeg)
    np.save(OUTPUT_DIR / "X_images.npy", X_images)
    np.save(OUTPUT_DIR / "y_labels.npy", y)
    np.save(OUTPUT_DIR / "groups.npy", groups)

    print("All files saved successfully.")


# ### Verifying Stored EEG and Image Tensor

if len(df) > 0:
    print("eeg_epoch type   :", type(df.loc[0, "eeg_epoch"]))
    print("eeg_epoch shape  :", df.loc[0, "eeg_epoch"].shape)

    print("image_tensor type  :", type(df.loc[0, "image_tensor"]))
    print("image_tensor shape :", df.loc[0, "image_tensor"].shape)
