import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, mat_files, preprocessor, spec_key='spectrogram', boxes_key='boxes', label_key='label'):
        """
        Dataset for loading spectrogram data and preprocessing into token sequences.
        Parameters:
          mat_files (list of str): List of paths to .mat files, each containing one sample.
          preprocessor (SignalPreprocessor): Instance of the preprocessor to process detection outputs.
          spec_key (str): Key name in .mat for the spectrogram matrix.
          boxes_key (str): Key name in .mat for the detection boxes array.
          label_key (str): Key name in .mat for the class label.
        """
        self.mat_files = mat_files
        self.preprocessor = preprocessor
        self.spec_key = spec_key
        self.boxes_key = boxes_key
        self.label_key = label_key

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        # Load the .mat file
        file_path = self.mat_files[idx]
        data = loadmat(file_path)
        # Extract spectrogram and detection boxes from the MATLAB file
        if self.spec_key not in data:
            raise KeyError(f"Spectrogram key '{self.spec_key}' not found in {file_path}")
        spec = data[self.spec_key]
        # If the spectrogram is stored in a nested struct (common in MATLAB), extract the array
        if isinstance(spec, np.ndarray) and spec.size == 1 and spec.dtype.names:
            # If spec is a MATLAB struct with field, assume first field contains the matrix
            spec = spec[0, 0]
            # find the first field that is array
            for name in spec.dtype.names:
                if isinstance(spec[name], np.ndarray):
                    spec = spec[name]
                    break
        spec = np.array(spec, dtype=np.float32)

        # Get detection boxes
        det_boxes = None
        if self.boxes_key in data:
            det_boxes = data[self.boxes_key]
            # Convert to numpy array of shape (N,4)
            det_boxes = np.array(det_boxes)
            # If boxes are stored in MATLAB as a cell or struct, additional parsing may be needed
            if det_boxes.size == 0:
                det_boxes = np.empty((0, 4))
            # If shape is (4, N) transpose it
            if det_boxes.ndim == 2 and det_boxes.shape[0] == 4 and det_boxes.shape[1] != 4:
                det_boxes = det_boxes.T
        else:
            # If detection results are not in the file, one could integrate an external detector here
            raise KeyError(f"Detection boxes key '{self.boxes_key}' not found in {file_path}")

        # Get label
        label = None
        if self.label_key in data:
            label_data = data[self.label_key]
            # Extract label as an integer (handle MATLAB format)
            if isinstance(label_data, np.ndarray):
                # If it's a single value array, fetch the element
                if label_data.size == 1:
                    label = int(label_data.reshape(-1)[0])
                else:
                    # If label is not a single value, handle accordingly (e.g., one-hot encoding or multiple labels)
                    label = int(label_data)  # try to cast to int
            else:
                # If not an ndarray, it might be a Python type
                label = int(label_data)
        else:
            raise KeyError(f"Label key '{self.label_key}' not found in {file_path}")

        # Process detection results to get token features
        features = self.preprocessor.process(det_boxes, spec)
        # features is a numpy array of shape (N_tokens, feature_dim).
        # Convert to torch tensor here (dtype float), or leave as numpy and convert in collate.
        features = torch.from_numpy(features.astype(np.float32))
        return features, label

def signal_collate_fn(batch):
    """
    Custom collate function to pad variable-length sequences of token features.
    Returns:
      padded_features: Tensor of shape (batch_size, max_seq_len, feature_dim)
      padding_mask: Boolean Tensor of shape (batch_size, max_seq_len) (True for padded positions)
      labels: Tensor of shape (batch_size,)
    """
    # Filter out any samples with no tokens (length 0) if any
    batch = [item for item in batch if item[0].shape[0] > 0]
    if len(batch) == 0:
        return None  # or raise an error
    # Determine max sequence length in this batch
    lengths = [item[0].shape[0] for item in batch]
    max_len = max(lengths)
    feature_dim = batch[0][0].shape[1] if max_len > 0 else 0
    batch_size = len(batch)
    # Initialize padded tensor and mask
    padded_features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)  # True = padded (mask out)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    for i, (feat, label) in enumerate(batch):
        seq_len = feat.shape[0]
        padded_features[i, :seq_len, :] = feat
        padding_mask[i, :seq_len] = False  # False for real tokens
        labels[i] = label
    return padded_features, padding_mask, labels
