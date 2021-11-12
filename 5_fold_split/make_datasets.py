"""
   Copyright 2021 paper: A Novel End-to-End Speech Emotion Recognition Network with Stacked Transformer Layers
   File name: make_datasets.py
   Data Created: 12/10/2020
"""
import os
import csv
import torch
import pickle
import librosa
import warnings
import numpy as np

from src.dataset import IEMOCAPDataset
warnings.filterwarnings('ignore')

CSV_PATH = './csv_file/'
SAVE_PATH = './data/features/'

labels = {'neutral': np.array([1, 0, 0, 0]),
          'happy': np.array([0, 1, 0, 0]),
          'sad': np.array([0, 0, 1, 0]),
          'angry': np.array([0, 0, 0, 1])}


def extract_audio_feat(audio_file):
    """

    :param audio_file:
    :return: extract audio feature.
    """
    y0, sr = librosa.load(audio_file, sr=16000)
    y = librosa.util.fix_length(y0, sr*10)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    log_spec = librosa.core.amplitude_to_db(spec)
    test_feat = np.transpose(log_spec)

    scale_file = pickle.load(open('./CASIA_scaler.pkl', 'rb'))
    test_feat_sc = scale_file.transform(test_feat.reshape(-1, test_feat.shape[-1])).reshape(test_feat.shape)

    return test_feat_sc


def audio_features(split='train', index=0):
    global CSV_PATH
    with open(os.path.join(CSV_PATH, f'IEMOCAP_{split}_with_Path_{index}.csv')) as csv_file:
        csv_reader = csv.reader(csv_file)
        target_list = list(csv_reader)
    target_list.pop(0)
    print(len(target_list))

    refine_label_target = [x[0] for x in target_list]
    refine_path_target = [x[1].replace('./', './data/') for x in target_list]

    label = []
    data = []
    for ind in range(len(refine_path_target)):
        print(refine_path_target[ind])
        audio_feat = extract_audio_feat(refine_path_target[ind])
        data.append(audio_feat)
        label.append(labels.get(refine_label_target[ind]))

    save_data(data, label, split, index)


def save_data(data, label, split='train', index=0):
    global SAVE_PATH
    save_path = SAVE_PATH + f'IEMOCAP_{split}_na_{index}.dt'
    data = IEMOCAPDataset(audio_data=data, audio_label=label, a_only=True)
    torch.save(data, save_path)


if __name__ == "__main__":
    for idx in range(5):
        print(f'-------------------------{idx}-fold-------------------------')
        audio_features('train', idx)
        audio_features('dev', idx)
        audio_features('test', idx)
