import numpy as np


def load_data(data_path):
    raw = np.genfromtxt(data_path, delimiter=',', dtype=str)
    arr = np.zeros((10000, 1000))
    for i in raw[1:]:
        row = int(i[0].split('_')[0][1:]) - 1
        column = int(i[0].split('_')[1][1:]) - 1
        arr[row, column] = int(i[1])
    return arr




def mean_impute(arr):
    pass



def train():
    pass