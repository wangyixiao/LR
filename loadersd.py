import torch
import pandas as pd
import numpy as np
from scipy import signal

def Load_Dataset_C(data_path, n_sub, start, end):
    feature = []
    label = []
    num = 1#n次试验
    de = []
    Hb = []
    p=0*4

    name = data_path + '/' + str(n_sub) + '/epo_OxyDeoxy.xls'  # fNIRS数据
    Hb_org = pd.read_excel(name, header=None, sheet_name=None)
    name = data_path + '/' + str(n_sub) + '/marks.xls'  # 分类
    desc = pd.read_excel(name, header=None)
    for i in range(1, 61):  # 75个工作表，一个人做了75次实验
        name = 'Sheet' + str(i)
        Hb.append(Hb_org[name].values)
    de.append(desc)

    de = pd.concat(de)
    # (75, 347, 40)
    Hb = np.array(Hb)
    desc = np.array(de)

    HbO_R = []
    HbO_L = []
    HbO_F = []
    HbR_R = []
    HbR_L = []
    HbR_F = []
    HbO_4 = []
    HbR_4 = []
    for i in range(60*num):  # 分类别，分HbOHbR
        if desc[i, 0] == 1+p:
            HbO_R.append(Hb[i, start:end, :24])
            HbR_R.append(Hb[i, start:end, 24:])
        elif desc[i, 0] == 2+p:
            HbO_L.append(Hb[i, start:end, :24])
            HbR_L.append(Hb[i, start:end, 24:])
        elif desc[i, 0] == 3+p:
            HbO_F.append(Hb[i, start:end, :24])
            HbR_F.append(Hb[i, start:end, 24:])
        elif desc[i, 0] == 4+p:
            HbO_4.append(Hb[i, start:end, :24])
            HbR_4.append(Hb[i, start:end, 24:])

    # (25, 256, 20) --> (25, 1, 256, 20)
    HbO_R = np.array(HbO_R).reshape((15*num, 1, end - start, 24))
    HbO_L = np.array(HbO_L).reshape((15*num, 1, end - start, 24))
    HbO_F = np.array(HbO_F).reshape((15*num, 1, end - start, 24))

    HbR_R = np.array(HbR_R).reshape((15*num, 1, end - start, 24))
    HbR_L = np.array(HbR_L).reshape((15*num, 1, end - start, 24))
    HbR_F = np.array(HbR_F).reshape((15*num, 1, end - start, 24))

    HbO_4 = np.array(HbO_4).reshape((15*num, 1, end - start, 24))

    HbR_4 = np.array(HbR_4).reshape((15*num, 1, end - start, 24))

    HbO_R = np.concatenate((HbO_R, HbR_R), axis=1)  # concatenate数组拼接  axis=1在第二维操作 (25, 2, 20, 256)
    HbO_L = np.concatenate((HbO_L, HbR_L), axis=1)
    HbO_F = np.concatenate((HbO_F, HbR_F), axis=1)
    HbO_4 = np.concatenate((HbO_4, HbR_4), axis=1)

    for i in range(15*num):
        feature.append(HbO_R[i, :, :, :])
        feature.append(HbO_L[i, :, :, :])
        feature.append(HbO_F[i, :, :, :])
        feature.append(HbO_4[i, :, :, :])

        label.append(0)
        label.append(1)
        label.append(2)
        label.append(3)

    print(str(num) + '  OK')


    feature = np.array(feature) #(2250, 2, 256, 20) 2250=75*30
    label = np.array(label) #(2250,)
    print('feature ', feature.shape)
    print('label ', label.shape)

    return feature, label


class Dataset(torch.utils.data.Dataset):

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



