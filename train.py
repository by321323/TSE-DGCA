import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore")
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Dataloader import *
from model import TSEDGCA
from utils import metric
import argparse
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PEMS08", help="the datasets name")
parser.add_argument('--train_rate', type=float, default=0.6, help="The ratio of training set")
parser.add_argument('--seq_len', type=int, default=12, help="The length of input sequence")
parser.add_argument('--pre_len', type=int, default=12, help="The length of output sequence")
parser.add_argument('--batchsize', type=int, default=16, help="Number of training batches")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--in_dim', type=int, default=1, help="Dimensionality of input data")
parser.add_argument('--embed_size', type=int, default=64, help="Embed_size")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
args = parser.parse_args()

if __name__ == "__main__":
    data, adj = load_data(args.dataset)
    time_len = data.shape[0]
    data1 = np.mat(data, dtype=np.float32)

    (
        trainX, trainY, trainTOD, trainDOW, trainFTOD, trainFDOW,
        valX, valY, valTOD, valDOW, valFTOD, valFDOW,
        testX, testY, testTOD, testDOW, testFTOD, testFDOW,
        mean, std
    ) = preprocess_data_with_time(
        data1, time_len, args.train_rate, args.seq_len, args.pre_len
    )

    train_data = TensorDataset(trainX, trainY, trainTOD, trainDOW, trainFTOD, trainFDOW)
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)

    val_data = TensorDataset(valX, valY, valTOD, valDOW, valFTOD, valFDOW)
    val_dataloader = DataLoader(val_data, batch_size=args.batchsize)

    model = TSEDGCA(
        adj, args.in_dim, args.embed_size, args.seq_len, args.pre_len,
        args.heads, 4, args.dropout,
        use_tod=True,
        use_dow=True
    )
    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_mae = 100
    best_rmse = None
    best_mape = None
    best_model_path = None

    for epoch in range(args.epochs):
        model.train()
        for x, y, tod_ids, dow_ids, future_tod_ids, future_dow_ids in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            tod_ids = tod_ids.to(device).long()
            dow_ids = dow_ids.to(device).long()
            future_tod_ids = future_tod_ids.to(device).long()
            future_dow_ids = future_dow_ids.to(device).long()

            pre = model(
                x,
                tod_ids=tod_ids,
                dow_ids=dow_ids,
                future_tod_ids=future_tod_ids,
                future_dow_ids=future_dow_ids
            ).reshape(-1, adj.shape[0])

            y = y.reshape(-1, adj.shape[0])
            loss = criterion(pre * std + mean, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        P = []
        L = []
        with torch.no_grad():
            for x, y, tod_ids, dow_ids, future_tod_ids, future_dow_ids in val_dataloader:
                x = x.to(device)
                tod_ids = tod_ids.to(device).long()
                dow_ids = dow_ids.to(device).long()
                future_tod_ids = future_tod_ids.to(device).long()
                future_dow_ids = future_dow_ids.to(device).long()

                pre = model(
                    x,
                    tod_ids=tod_ids,
                    dow_ids=dow_ids,
                    future_tod_ids=future_tod_ids,
                    future_dow_ids=future_dow_ids
                ) * std + mean

                P.append(pre.cpu().detach())
                L.append(y)

        pre = torch.cat(P, 0)
        label = torch.cat(L, 0)

        pre = pre.reshape(-1, adj.shape[0])
        label = label.reshape(-1, adj.shape[0])

        mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
        print("rmse,mae,mape,wape:", epoch, rmse, mae, mape, wape)


        DATANAME = args.dataset
        t = datetime.datetime.now()
        dir = str(t)[0:10]
        file = str(t)[10:-7].replace(":", "-")

        if not os.path.exists(os.path.join("Model/PEMS/{}/{}".format(DATANAME, dir))):
            os.makedirs(os.path.join("Model/PEMS/{}/{}".format(DATANAME, dir)))
        torch.save(model.state_dict(), "Model/PEMS/{}/{}/epoch+{}+time{}.pkl".format(DATANAME, dir, epoch, file))

        if not os.path.exists("result"):
            os.makedirs("result")
        with open("result/{}min_{}_mean_std.txt".format(args.pre_len * 5, DATANAME), "a+") as f:
            f.write("{}>>>>{} min pre：rmse:{},mae:{},mape:{}".format(str(t), args.pre_len * 5, rmse, mae, mape) + "\n")

        if mae < best_mae:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            print("now best mae:>>>>>>>>>>>>>", best_rmse, best_mae, best_mape)

            best_model_dir = os.path.join("Model/PEMS", DATANAME, "best_model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            best_model_path = os.path.join(best_model_dir, "best_model.pkl")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at: {best_model_path}")

            with open("result/{}min_{}_mean_std.txt".format(args.pre_len * 5, DATANAME), "a+") as f:
                f.write("{}>>>>{} min pre：rmse:{},mae:{},mape:{}".format(str(t), args.pre_len * 5, best_rmse, best_mae, best_mape) + "\n")
