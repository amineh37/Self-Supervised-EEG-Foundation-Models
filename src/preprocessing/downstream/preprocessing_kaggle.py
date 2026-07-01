# import warnings
# warnings.filterwarnings('ignore')
# import mne
# import csv
# import random
# import os
# import torch
# import numpy as np
# import tqdm
# import pandas as pd

# # Define channel names
# use_channels_names = [
#     'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
#     'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'
# ]
# ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

# def seed_torch(seed=1029):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
#     if use_avg:
#         x = x - torch.mean(x, dim=-2, keepdim=True)
#     if len(x.shape) == 2:
#         return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
#     elif len(x.shape) == 3:
#         return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
#     else:
#         raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

# def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
#     if data_max is not None:
#         max_scale = data_max - data_min
#         scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
#     if len(x.shape) == 2:
#         xmin = x.min()
#         xmax = x.max()
#         if xmax - xmin == 0:
#             x = 0
#             return x
#     elif len(x.shape) == 3:
#         xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
#         xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
#         constant_trials = (xmax - xmin) == 0
#         if torch.any(constant_trials):
#             xmax[constant_trials] = xmax[constant_trials] + 1e-6

#     x = (x - xmin) / (xmax - xmin)
#     x -= 0.5
#     x += (high + low) / 2
#     x = (high - low) * x
    
#     if data_max is not None:
#         x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
#     return x

# def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
#     sample_rate = 200
#     raw = pd.read_csv(filename)
#     data = torch.tensor(raw.iloc[:,1:-2].values)
#     feed = torch.tensor(raw['FeedBackEvent'].values)
#     stim_pos = torch.nonzero(feed>0)
#     datas = []
    
#     if use_channels_names is not None:
#         choice_channels = []
#         for ch in use_channels_names:
#             choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
#         use_channels = choice_channels
#     if data_max is not None: use_channels+=[-1]
    
#     xform = lambda x: min_max_normalize(x, data_max, data_min)
    
#     for fb, pos in enumerate(stim_pos, 1):
#         start_i = max(pos + int(sample_rate * tmin), 0)
#         end___i = min(start_i + int(sample_rate * tlen), len(feed))
#         trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
#         info = mne.create_info(
#             ch_names=[str(i) for i in range(trial.shape[0])],
#             ch_types="eeg",
#             sfreq=200,
#         )
#         raw = mne.io.RawArray(trial, info)
#         trial = torch.tensor(raw.get_data()).float()
#         trial = xform(trial)
#         if use_channels_names is not None:
#             trial = trial[use_channels]
#         datas.append(trial)
#     return datas

# def read_kaggle_ern_test(
#     path="datas/",
#     subjects=[1, 3, 4, 5, 8, 9, 10, 15, 19, 25],
#     sessions=[1, 2, 3, 4, 5],
#     tmin=-0.7,
#     tlen=2,
#     data_max=None,
#     data_min=None,
#     use_channels_names=use_channels_names,
# ):
#     labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label']
#     label_id = 0
#     datas = []
#     for i in tqdm.tqdm(subjects):
#         for j in sessions:
#             filename = os.path.join(path, "KaggleERN", "test", "Data_S{:02d}_Sess{:02d}.csv".format(i, j))
#             for data in read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names=use_channels_names):
#                 label = labels[label_id]
#                 label_id += 1
#                 datas.append((data, int(label)))
#     return datas

# def read_kaggle_ern_train(
#     path="datas/",
#     subjects=[2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
#     sessions=[1, 2, 3, 4, 5],
#     tmin=-0.7,
#     tlen=2,
#     data_max=None,
#     data_min=None,
#     use_channels_names=use_channels_names,
# ):
#     labels = []
#     with open(os.path.join(path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
#         reader = csv.reader(f)
#         for i, row in enumerate(reader):
#             if i > 0: labels.append(row)
#     labels = dict(labels)
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     print(labels)
#     datas = []
#     for i in tqdm.tqdm(subjects):
#         for j in sessions:
#             if i > 9:
#                 if i == 22 and j == 5:
#                     print("Skipped error file " + "KaggleERN/train/Data_S" + str(i) + "_Sess0" + str(j) + ".csv")
#                 else:
#                     filename = os.path.join(path, "KaggleERN", "train", "Data_S" + str(i) + "_Sess0" + str(j) + ".csv")
#             else:
#                 filename = os.path.join(path, "KaggleERN", "train", "Data_S0" + str(i) + "_Sess0" + str(j) + ".csv")
#             for fb, trial in enumerate(read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names=use_channels_names), 1):
#                 label = labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i, j, fb)]
#                 datas.append((trial, int(label)))
#     return datas

# def preprocess_and_save_data(path, folds, test_subjects, save_dir="./preprocessed_data/", tmin=-0.7, tlen=2, sessions=[1, 2, 3, 4, 5]):
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Process test set (same for all folds)
#     test_data = read_kaggle_ern_test(path, subjects=test_subjects, sessions=sessions, tmin=tmin, tlen=tlen)
#     test_data_np = [d[0].numpy() for d in test_data]
#     test_labels_np = [d[1] for d in test_data]
#     np.savez(os.path.join(save_dir, "test.npz"), data=test_data_np, labels=test_labels_np)
#     print(f"Saved test set to {save_dir}/test.npz")

#     # Process each fold's train and validation sets
#     for fold, (train_subjects, valid_subjects) in folds.items():
#         print("train subjects: ", train_subjects)
#         print("val subjects: ", valid_subjects)
#         train_data = read_kaggle_ern_train(path, subjects=train_subjects, sessions=sessions, tmin=tmin, tlen=tlen)
#         train_data_np = [d[0].numpy() for d in train_data]
#         train_labels_np = [d[1] for d in train_data]
#         np.savez(os.path.join(save_dir, f"fold{fold}_train.npz"), data=train_data_np, labels=train_labels_np)
#         print(f"Saved fold {fold} train set to {save_dir}/fold{fold}_train.npz")

#         valid_data = read_kaggle_ern_train(path, subjects=valid_subjects, sessions=sessions, tmin=tmin, tlen=tlen)
#         valid_data_np = [d[0].numpy() for d in valid_data]
#         valid_labels_np = [d[1] for d in valid_data]
#         np.savez(os.path.join(save_dir, f"fold{fold}_valid.npz"), data=valid_data_np, labels=valid_labels_np)
#         print(f"Saved fold {fold} valid set to {save_dir}/fold{fold}_valid.npz")

# if __name__ == "__main__":
#     seed_torch(9)
#     path = "/projects/EEG-foundation-model/"
#     Folds = {
#         1: ([11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [2, 6, 7]),
#         2: ([2, 6, 7, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [11, 12, 13]),
#         3: ([2, 6, 7, 11, 12, 13, 18, 20, 21, 22, 23, 24, 26], [14, 16, 17]),
#         4: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 23, 24, 26], [20, 21, 22]),
#         5: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22], [23, 24, 26]),
#     }
#     test_subjects = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
#     preprocess_and_save_data(path, Folds, test_subjects, save_dir="/projects/EEG-foundation-model/KaggleERN_final/processed_test")

import warnings
warnings.filterwarnings('ignore')
import mne
import csv
import random
import os
import torch
import numpy as np
import tqdm
import pandas as pd

# Define channel names
use_channels_names = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'
]
ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)
    x -= 0.5
    x += (high + low) / 2
    x = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x

def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
    sample_rate = 256  # Changed from 200 to 256 Hz
    raw = pd.read_csv(filename)
    data = torch.tensor(raw.iloc[:,1:-2].values)
    feed = torch.tensor(raw['FeedBackEvent'].values)
    stim_pos = torch.nonzero(feed>0)
    datas = []
    
    if use_channels_names is not None:
        choice_channels = []
        for ch in use_channels_names:
            choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
        use_channels = choice_channels
    if data_max is not None: use_channels+=[-1]
    
    xform = lambda x: min_max_normalize(x, data_max, data_min)
    
    desired_sequence_length = int(sample_rate * tlen)  # Ensure output length matches 256 Hz
    for fb, pos in enumerate(stim_pos, 1):
        start_i = max(pos + int(sample_rate * tmin), 0)
        end___i = min(start_i + int(sample_rate * tlen), len(feed))
        trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
        info = mne.create_info(
            ch_names=[str(i) for i in range(trial.shape[0])],
            ch_types="eeg",
            sfreq=256,  # Changed from 200 to 256 Hz
        )
        raw = mne.io.RawArray(trial, info)
        trial = torch.tensor(raw.get_data()).float()
        trial = temporal_interpolation(trial, desired_sequence_length=desired_sequence_length, mode='nearest')
        trial = xform(trial)
        if use_channels_names is not None:
            trial = trial[use_channels]
        datas.append(trial)
    return datas

def read_kaggle_ern_test(
    path="datas/",
    subjects=[1, 3, 4, 5, 8, 9, 10, 15, 19, 25],
    sessions=[1, 2, 3, 4, 5],
    tmin=-0.7,
    tlen=2,
    data_max=None,
    data_min=None,
    use_channels_names=use_channels_names,
):
    labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label']
    label_id = 0
    datas = []
    for i in tqdm.tqdm(subjects):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", "Data_S{:02d}_Sess{:02d}.csv".format(i, j))
            for data in read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names=use_channels_names):
                label = labels[label_id]
                label_id += 1
                datas.append((data, int(label)))
    return datas

def read_kaggle_ern_train(
    path="datas/",
    subjects=[2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
    sessions=[1, 2, 3, 4, 5],
    tmin=-0.7,
    tlen=2,
    data_max=None,
    data_min=None,
    use_channels_names=use_channels_names,
):
    labels = []
    with open(os.path.join(path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0: labels.append(row)
    labels = dict(labels)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(labels)
    datas = []
    for i in tqdm.tqdm(subjects):
        for j in sessions:
            if i > 9:
                if i == 22 and j == 5:
                    print("Skipped error file " + "KaggleERN/train/Data_S" + str(i) + "_Sess0" + str(j) + ".csv")
                else:
                    filename = os.path.join(path, "KaggleERN", "train", "Data_S" + str(i) + "_Sess0" + str(j) + ".csv")
            else:
                filename = os.path.join(path, "KaggleERN", "train", "Data_S0" + str(i) + "_Sess0" + str(j) + ".csv")
            for fb, trial in enumerate(read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names=use_channels_names), 1):
                label = labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i, j, fb)]
                datas.append((trial, int(label)))
    return datas

def preprocess_and_save_data(path, folds, test_subjects, save_dir="./preprocessed_data/", tmin=-0.7, tlen=2, sessions=[1, 2, 3, 4, 5]):
    os.makedirs(save_dir, exist_ok=True)
    
    # Process test set (same for all folds)
    test_data = read_kaggle_ern_test(path, subjects=test_subjects, sessions=sessions, tmin=tmin, tlen=tlen)
    test_data_np = [d[0].numpy() for d in test_data]
    test_labels_np = [d[1] for d in test_data]
    np.savez(os.path.join(save_dir, "test.npz"), data=test_data_np, labels=test_labels_np)
    print(f"Saved test set to {save_dir}/test.npz")

    # Process each fold's train and validation sets
    for fold, (train_subjects, valid_subjects) in folds.items():
        print("train subjects: ", train_subjects)
        print("val subjects: ", valid_subjects)
        train_data = read_kaggle_ern_train(path, subjects=train_subjects, sessions=sessions, tmin=tmin, tlen=tlen)
        train_data_np = [d[0].numpy() for d in train_data]
        train_labels_np = [d[1] for d in train_data]
        np.savez(os.path.join(save_dir, f"fold{fold}_train.npz"), data=train_data_np, labels=train_labels_np)
        print(f"Saved fold {fold} train set to {save_dir}/fold{fold}_train.npz")

        valid_data = read_kaggle_ern_train(path, subjects=valid_subjects, sessions=sessions, tmin=tmin, tlen=tlen)
        valid_data_np = [d[0].numpy() for d in valid_data]
        valid_labels_np = [d[1] for d in valid_data]
        np.savez(os.path.join(save_dir, f"fold{fold}_valid.npz"), data=valid_data_np, labels=valid_labels_np)
        print(f"Saved fold {fold} valid set to {save_dir}/fold{fold}_valid.npz")

if __name__ == "__main__":
    seed_torch(9)
    path = "/projects/EEG-foundation-model/"
    Folds = {
        1: ([11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [2, 6, 7]),
        2: ([2, 6, 7, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [11, 12, 13]),
        3: ([2, 6, 7, 11, 12, 13, 18, 20, 21, 22, 23, 24, 26], [14, 16, 17]),
        4: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 23, 24, 26], [20, 21, 22]),
        5: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22], [23, 24, 26]),
    }
    test_subjects = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
    preprocess_and_save_data(path, Folds, test_subjects, save_dir="/projects/EEG-foundation-model/KaggleERN_final/processed_eegpt")