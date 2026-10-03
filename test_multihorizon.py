import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Dataloader import *
from model import TSEDGCA
from utils import metric


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PEMS08", help="the datasets name")
parser.add_argument('--train_rate', type=float, default=0.6, help="The ratio of training set")
parser.add_argument('--seq_len', type=int, default=12, help="The length of input sequence")
parser.add_argument('--pre_len', type=int, default=12, help="The length of output sequence")
parser.add_argument('--batchsize', type=int, default=16, help="Number of testing batches")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--in_dim', type=int, default=1, help="Dimensionality of input data")
parser.add_argument('--embed_size', type=int, default=64, help="Embed_size")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
parser.add_argument(
    '--model',
    type=str,
    default="Model/PEMS/PEMS08/best_model/best_model.pkl",
    help="Trained model path"
)
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

    test_data = TensorDataset(testX, testY, testTOD, testDOW, testFTOD, testFDOW)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize)

    model = TSEDGCA(
        adj, args.in_dim, args.embed_size, args.seq_len, args.pre_len,
        args.heads, 4, args.dropout,
        use_tod=True,
        use_dow=True
    )

    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    P = []
    L = []

    with torch.no_grad():
        for x, y, tod_ids, dow_ids, future_tod_ids, future_dow_ids in test_dataloader:
            x = x.to(device)
            tod_ids = tod_ids.to(device).long()
            dow_ids = dow_ids.to(device).long()
            future_tod_ids = future_tod_ids.to(device).long()
            future_dow_ids = future_dow_ids.to(device).long()

            out = model(
                x,
                tod_ids=tod_ids,
                dow_ids=dow_ids,
                future_tod_ids=future_tod_ids,
                future_dow_ids=future_dow_ids
            )

            if isinstance(out, tuple):
                out = out[0]

            pre = out * std + mean
            P.append(pre.cpu().detach())
            L.append(y)

    # 打印监控信息
    print("\n===== 监控信息 =====")

    if hasattr(model, "calendar_emb_scale"):
        print("input calendar scale:", torch.sigmoid(model.calendar_emb_scale).item())
    if hasattr(model, "future_time_scale"):
        print("future calendar scale:", torch.sigmoid(model.future_time_scale).item())

    if hasattr(model, "ST1") and hasattr(model.ST1, "increment_temporal"):
        if hasattr(model.ST1.increment_temporal, "last_delta_strength") and model.ST1.increment_temporal.last_delta_strength is not None:
            print("ST1 delta strength:", model.ST1.increment_temporal.last_delta_strength.item())
        if hasattr(model.ST2.increment_temporal, "last_delta_strength") and model.ST2.increment_temporal.last_delta_strength is not None:
            print("ST2 delta strength:", model.ST2.increment_temporal.last_delta_strength.item())

        if hasattr(model.ST1.increment_temporal, "accum_decay"):
            print("ST1 accum decay:", torch.sigmoid(model.ST1.increment_temporal.accum_decay).item())
        if hasattr(model.ST2.increment_temporal, "accum_decay"):
            print("ST2 accum decay:", torch.sigmoid(model.ST2.increment_temporal.accum_decay).item())

        # ===== 新增：监控分支权重 =====
        if hasattr(model.ST1.increment_temporal, "last_branch_weights") and model.ST1.increment_temporal.last_branch_weights is not None:
            print("ST1 branch weights:", model.ST1.increment_temporal.last_branch_weights.mean(dim=(0,1,2)).detach().cpu().numpy())
        if hasattr(model.ST2.increment_temporal, "last_branch_weights") and model.ST2.increment_temporal.last_branch_weights is not None:
            print("ST2 branch weights:", model.ST2.increment_temporal.last_branch_weights.mean(dim=(0,1,2)).detach().cpu().numpy())

        if hasattr(model.ST1.S, "delta_context_scale"):
            print("ST1 delta graph scale:", torch.sigmoid(model.ST1.S.delta_context_scale).item())
        if hasattr(model.ST2.S, "delta_context_scale"):
            print("ST2 delta graph scale:", torch.sigmoid(model.ST2.S.delta_context_scale).item())

    # pre / label: [num_samples, pre_len, num_nodes]
    pre = torch.cat(P, 0)
    label = torch.cat(L, 0)

    print("\n===== Overall average over all prediction steps =====")
    mae, rmse, mape, wape = metric(
        pre.reshape(-1, adj.shape[0]).numpy(),
        label.reshape(-1, adj.shape[0]).numpy()
    )
    print("Overall | MAE {:.4f} | MAPE {:.4f} | RMSE {:.4f} | WAPE {:.4f}".format(mae, mape, rmse, wape))

    print("\n===== Horizon-specific results =====")
    # 5-min sampling: 15/30/45/60 min correspond to the 3rd/6th/9th/12th prediction step.
    horizons = [3, 6, 9, 12]
    for h in horizons:
        if h > pre.shape[1]:
            continue
        pred_h = pre[:, h - 1, :]      # exact horizon, e.g., h=3 means t+3, 15 min
        label_h = label[:, h - 1, :]
        mae_h, rmse_h, mape_h, wape_h = metric(pred_h.numpy(), label_h.numpy())
        print("{:>2d} steps ({:>2d} min) | MAE {:.4f} | MAPE {:.4f} | RMSE {:.4f} | WAPE {:.4f}".format(
            h, h * 5, mae_h, mape_h, rmse_h, wape_h
        ))
