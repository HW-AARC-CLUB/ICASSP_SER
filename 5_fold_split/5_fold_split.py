"""
   Copyright 2021 paper: A Novel End-to-End Speech Emotion Recognition Network with Stacked Transformer Layers
   File name: 5_fold_split.py
   Data Created: 12/10/2020
"""
import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

root_path = './csv_file/'
csv_name = 'IEMOCAP_All_with_Path.csv'


def save_k_csv(datas, labels, index):
    global root_path
    ff = ['train', 'dev', 'test']
    for f1 in range(len(ff)):
        save_csv_name = f'IEMOCAP_{ff[f1]}_with_Path_{index}.csv'
        pd_data = pd.DataFrame(np.column_stack((labels[f1], datas[f1])), columns=['labels', 'path'])

        if os.path.exists(root_path + save_csv_name):
            os.remove(root_path + save_csv_name)
        pd_data.to_csv(root_path + save_csv_name, index=None)


def k_fold_spilt(k, csv_data, emotion_label):
    split_list = []
    kf = KFold(n_splits=k)
    for train, test in kf.split(csv_data):
        split_list.append(train.tolist())
        split_list.append(test.tolist())
    for idx in range(k):
        train, test = split_list[2 * idx], split_list[2 * idx + 1]
        train = np.array(train)
        test = np.array(test)
        x = np.array(csv_data)[train]
        y = np.array(emotion_label)[train]
        test_data = np.array(csv_data)[test]
        test_label = np.array(emotion_label)[test]
        train_data, val_data, train_label, val_label = train_test_split(x, y, test_size=0.125, random_state=10)
        save_k_csv([train_data, val_data, test_data], [train_label, val_label, test_label], idx)


def load_csv(csv_path):
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        target_list = list(csv_reader)
    target_list.pop(0)
    print(len(target_list))

    refine_label_target = [x[0] for x in target_list]
    refine_path_target = [x[1] for x in target_list]
    return refine_path_target, refine_label_target


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data, label = load_csv(root_path + csv_name)
    k_fold_spilt(5, data, label)
    print('------------------------ALL DONE------------------------')
