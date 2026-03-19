import torch
import pandas as pd
import numpy as np
from scipy import signal

def Load_Dataset_C(data_path,  start, end):
    """
    Load Dataset C

    Args:
        data_path (str): dataset path.
        start (int): start sampling point, default=20.
        end (int): end sampling point, default=276.

    Returns:
        feature : fNIRS signal data.
        label : fNIRS labels.
    """
    feature = []
    label = []
    num = 1  # n次试验
    p = 0 * 4
    for sub in range(1, 22): #30个受试者
        name = data_path + '/' + str(sub) + '/epo_OxyDeoxy.xls'  # fNIRS数据
        Hb_org = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/marks.xls'  # 分类
        desc = pd.read_excel(name, header=None)

        Hb = []
        for i in range(1, 61):  # 75个工作表，一个人做了75次实验
            name = 'Sheet' + str(i)
            Hb.append(Hb_org[name].values)

        # (75, 347, 40)
        Hb = np.array(Hb)
        desc = np.array(desc)

        HbO_R = []
        HbO_L = []
        HbO_F = []
        HbR_R = []
        HbR_L = []
        HbR_F = []
        HbO_4 = []
        HbR_4 = []
        for i in range(60 * num):  # 分类别，分HbOHbR
            if desc[i, 0] == 1 + p:
                HbO_R.append(Hb[i, start:end, :24])
                HbR_R.append(Hb[i, start:end, 24:])
            elif desc[i, 0] == 2 + p:
                HbO_L.append(Hb[i, start:end, :24])
                HbR_L.append(Hb[i, start:end, 24:])
            elif desc[i, 0] == 3 + p:
                HbO_F.append(Hb[i, start:end, :24])
                HbR_F.append(Hb[i, start:end, 24:])
            elif desc[i, 0] == 4 + p:
                HbO_4.append(Hb[i, start:end, :24])
                HbR_4.append(Hb[i, start:end, 24:])

        # (25, 256, 20) --> (25, 1, 256, 20)
        HbO_R = np.array(HbO_R).reshape((15 * num, 1, end - start, 24))
        HbO_L = np.array(HbO_L).reshape((15 * num, 1, end - start, 24))
        HbO_F = np.array(HbO_F).reshape((15 * num, 1, end - start, 24))

        HbR_R = np.array(HbR_R).reshape((15 * num, 1, end - start, 24))
        HbR_L = np.array(HbR_L).reshape((15 * num, 1, end - start, 24))
        HbR_F = np.array(HbR_F).reshape((15 * num, 1, end - start, 24))

        HbO_4 = np.array(HbO_4).reshape((15 * num, 1, end - start, 24))

        HbR_4 = np.array(HbR_4).reshape((15 * num, 1, end - start, 24))

        HbO_R = np.concatenate((HbO_R, HbR_R), axis=1)  # concatenate数组拼接  axis=1在第二维操作 (25, 2, 20, 256)
        HbO_L = np.concatenate((HbO_L, HbR_L), axis=1)
        HbO_F = np.concatenate((HbO_F, HbR_F), axis=1)
        HbO_4 = np.concatenate((HbO_4, HbR_4), axis=1)

        for i in range(15 * num):
            feature.append(HbO_R[i, :, :, :])
            feature.append(HbO_L[i, :, :, :])
            feature.append(HbO_F[i, :, :, :])
            feature.append(HbO_4[i, :, :, :])

            label.append(0)
            label.append(1)
            label.append(2)
            label.append(3)

        print(str(sub) + '  OK')

    feature = np.array(feature) #(2250, 2, 256, 20) 2250=75*30
    label = np.array(label) #(2250,)
    print('feature ', feature.shape)
    print('label ', label.shape)

    return feature, label


class Dataset(torch.utils.data.Dataset):
    """
    Load data for training

    Args:
        feature: input data.
        label: class for input data.
        transform: Z-score normalization is used to accelerate convergence (default:True).采用 z 分数归一化加速收敛
    """
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        print(self.feature.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # z-score normalization
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item]

def Split_Dataset_C(sub, feature, label, sampling_points):
    if sub == 1:
        X_test = feature[: 60]
        y_test = label[: 60]
        X_train = feature[60:]
        y_train = label[60:]
    elif sub == 22:
        X_test = feature[60 * 21:]
        y_test = label[60 * 21:]
        X_train = feature[:60 * 21]
        y_train = label[: 60 * 21]
    else:
        X_test = feature[60 * (sub - 1): 60 * sub]
        y_test = label[60 * (sub - 1): 60 * sub]
        feature_set_1 = feature[: 60 * (sub - 1)]
        label_set_1 = label[:60 * (sub - 1)]
        feature_set_2 = feature[60 * sub:]
        label_set_2 = label[60 * sub:]
        X_train = np.append(feature_set_1, feature_set_2, axis=0)
        y_train = np.append(label_set_1, label_set_2, axis=0)

    X_train = X_train.reshape((X_train.shape[0], 2, sampling_points, -1))
    X_test = X_test.reshape((X_test.shape[0], 2, sampling_points, -1))

    return X_train, y_train, X_test, y_test

