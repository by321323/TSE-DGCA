import numpy as np
import torch


def _make_windows_with_time(data_part, global_start_idx, seq_len, pre_len, points_per_day=288):
    """
    data_part: 当前划分的数据段，例如 train_data / val_data / test_data
    global_start_idx: 该划分在原始完整序列中的起始下标

    返回:
        X: [num_samples, seq_len, N]
        Y: [num_samples, pre_len, N]
        TOD: [num_samples, seq_len]，历史输入窗口的一天内时间编号，范围 0~287
        DOW: [num_samples, seq_len]，历史输入窗口的一周内星期编号，范围 0~6
    """
    X, Y = [], []
    TOD, DOW = [], []

    for i in range(len(data_part) - seq_len - pre_len):
        a = data_part[i: i + seq_len + pre_len]
        X.append(a[0: seq_len])
        Y.append(a[seq_len: seq_len + pre_len])

        # 历史输入窗口在完整时间序列中的真实下标
        input_idx = np.arange(global_start_idx + i, global_start_idx + i + seq_len)
        TOD.append(input_idx % points_per_day)
        DOW.append((input_idx // points_per_day) % 7)

    return (
        np.array(X), np.array(Y),
        np.array(TOD), np.array(DOW)
    )


def preprocess_data_with_time(data, time_len, rate, seq_len, pre_len, points_per_day=288):
    """
    返回 X/Y + 历史 TOD/DOW。

    历史 TOD/DOW 用于输入端时间嵌入；
    注意：所有时间编号都用完整序列的全局下标计算，验证集和测试集不会从 0 重新计数。
    """
    train_size = int(time_len * rate)
    val_start = train_size
    test_start = int(time_len * (rate + 0.2))

    train_data = data[0:train_size]
    val_data = data[val_start:test_start]
    test_data = data[test_start:time_len]

    trainX1, trainY1, trainTOD, trainDOW = _make_windows_with_time(
        train_data, 0, seq_len, pre_len, points_per_day
    )
    valX1, valY1, valTOD, valDOW = _make_windows_with_time(
        val_data, val_start, seq_len, pre_len, points_per_day
    )
    testX1, testY1, testTOD, testDOW = _make_windows_with_time(
        test_data, test_start, seq_len, pre_len, points_per_day
    )

    mean, std = np.mean(trainX1), np.std(trainX1)

    trainX1 = (trainX1 - mean) / std
    valX1 = (valX1 - mean) / std
    testX1 = (testX1 - mean) / std

    trainX1 = torch.tensor(trainX1, dtype=torch.float32)
    trainY1 = torch.tensor(trainY1, dtype=torch.float32)
    valX1 = torch.tensor(valX1, dtype=torch.float32)
    valY1 = torch.tensor(valY1, dtype=torch.float32)
    testX1 = torch.tensor(testX1, dtype=torch.float32)
    testY1 = torch.tensor(testY1, dtype=torch.float32)

    trainTOD = torch.tensor(trainTOD, dtype=torch.long)
    trainDOW = torch.tensor(trainDOW, dtype=torch.long)
    valTOD = torch.tensor(valTOD, dtype=torch.long)
    valDOW = torch.tensor(valDOW, dtype=torch.long)
    testTOD = torch.tensor(testTOD, dtype=torch.long)
    testDOW = torch.tensor(testDOW, dtype=torch.long)

    return (
        trainX1, trainY1, trainTOD, trainDOW,
        valX1, valY1, valTOD, valDOW,
        testX1, testY1, testTOD, testDOW,
        mean, std
    )


def load_data(data):
    if data == "PEMS03":
        adj_mx = np.load("data/PEMS03/adj.npy")
        x = np.load("data/PEMS03/PEMS03.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data == "PEMS04":
        adj_mx = np.load("data/PEMS04/adj.npy")
        x = np.load("data/PEMS04/PEMS04.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data == "PEMS07":
        adj_mx = np.load("data/PEMS07/adj.npy")
        x = np.load("data/PEMS07/PEMS07.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data == "PEMS08":
        adj_mx = np.load("data/PEMS08/adj.npy")
        x = np.load("data/PEMS08/PEMS08.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    else:
        raise ValueError("Unknown dataset: {}".format(data))

    adj = torch.tensor(np.array(adj_mx), dtype=torch.float32)
    return x, adj
