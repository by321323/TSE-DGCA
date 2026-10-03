# -*- coding: utf-8 -*-
"""
批量测试每个 epoch checkpoint 的测试集表现。

用途：
    诊断“验证集最优模型”和“测试集表现最好的模型”是否一致。

注意：
    这个脚本用于分析，不建议在论文中用 test 最优 epoch 作为模型选择依据。

使用示例：
    python batch_test_epochs.py --dataset PEMS03 --checkpoint_dir Model/PEMS/PEMS03/2026-05-XX

如果你的 checkpoint 分散在多个日期文件夹，可以指定更上层目录，并加 --recursive：
    python batch_test_epochs.py --dataset PEMS03 --checkpoint_dir Model/PEMS/PEMS03 --recursive
"""

import os
import re
import csv
import glob
import argparse
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset

from Dataloader import *
from model import TSEDGCA
from utils import metric

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='保存 epoch checkpoint 的文件夹，例如 Model/PEMS/PEMS03/2026-05-XX')
    parser.add_argument('--recursive', action='store_true',
                        help='是否递归搜索 checkpoint_dir 下所有子文件夹')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='输出 CSV 路径；默认保存在 checkpoint_dir 下')
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pre_len', type=int, default=12)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--start_epoch', type=int, default=None,
                        help='只测试 >= start_epoch 的 checkpoint')
    parser.add_argument('--end_epoch', type=int, default=None,
                        help='只测试 <= end_epoch 的 checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def extract_epoch(path):
    """从文件名中提取 epoch 编号；提取不到则返回一个很大的数，放到最后。"""
    name = os.path.basename(path)
    # 兼容 epoch+12+time.pkl / epoch_12.pkl / epoch12.pkl 等写法
    patterns = [
        r'epoch\+(\d+)\+',
        r'epoch[_-]?(\d+)',
        r'ep[_-]?(\d+)'
    ]
    for p in patterns:
        m = re.search(p, name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 10 ** 9


def find_checkpoints(checkpoint_dir, recursive=False):
    if recursive:
        pattern = os.path.join(checkpoint_dir, '**', '*.pkl')
        paths = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(checkpoint_dir, '*.pkl')
        paths = glob.glob(pattern)

    # 排除 best_model / averaged / DEBUG，避免重复测试非 epoch 模型
    filtered = []
    for p in paths:
        name = os.path.basename(p).lower()
        if 'best' in name or 'avg' in name or 'average' in name or 'debug' in name:
            continue
        if extract_epoch(p) == 10 ** 9:
            continue
        filtered.append(p)

    return sorted(filtered, key=lambda x: (extract_epoch(x), x))


def build_dataloaders(args):
    data, adj = load_data(args.dataset)
    time_len = data.shape[0]
    data1 = np.mat(data, dtype=np.float32)

    # 优先使用带历史/未来 TOD-DOW 的接口；没有则自动退回旧接口
    if 'preprocess_data_with_time' in globals():
        results = preprocess_data_with_time(
            data1, time_len, args.train_rate, args.seq_len, args.pre_len
        )
        (
            trainX, trainY, trainTOD, trainDOW, trainFTOD, trainFDOW,
            valX, valY, valTOD, valDOW, valFTOD, valFDOW,
            testX, testY, testTOD, testDOW, testFTOD, testFDOW,
            mean, std
        ) = results

        test_data = TensorDataset(testX, testY, testTOD, testDOW, testFTOD, testFDOW)
        val_data = TensorDataset(valX, valY, valTOD, valDOW, valFTOD, valFDOW)
        time_mode = 'history_and_future_tod_dow'
    else:
        trainX, trainY, valX, valY, testX, testY, mean, std = preprocess_data(
            data1, time_len, args.train_rate, args.seq_len, args.pre_len
        )
        test_data = TensorDataset(testX, testY)
        val_data = TensorDataset(valX, valY)
        time_mode = 'no_tod_dow'

    test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batchsize, shuffle=False)
    return adj, mean, std, val_loader, test_loader, time_mode


def forward_by_batch(model, batch, device):
    if len(batch) == 2:
        x, y = batch
        x = x.to(device)
        pred = model(x)
    elif len(batch) == 4:
        x, y, tod_ids, dow_ids = batch
        x = x.to(device)
        pred = model(
            x,
            tod_ids=tod_ids.to(device),
            dow_ids=dow_ids.to(device)
        )
    elif len(batch) == 6:
        x, y, tod_ids, dow_ids, future_tod_ids, future_dow_ids = batch
        x = x.to(device)
        pred = model(
            x,
            tod_ids=tod_ids.to(device),
            dow_ids=dow_ids.to(device),
            future_tod_ids=future_tod_ids.to(device),
            future_dow_ids=future_dow_ids.to(device)
        )
    else:
        raise ValueError('Unexpected batch length: {}'.format(len(batch)))

    return pred, y


def evaluate(model, dataloader, mean, std, adj, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            pred, y = forward_by_batch(model, batch, device)
            pred = pred * std + mean
            preds.append(pred.cpu())
            labels.append(y.cpu())

    pred = torch.cat(preds, dim=0).reshape(-1, adj.shape[0])
    label = torch.cat(labels, dim=0).reshape(-1, adj.shape[0])
    mae, rmse, mape, wape = metric(pred.numpy(), label.numpy())
    return rmse, mae, mape, wape


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    # 兼容某些训练脚本保存 {'state_dict': ...} 的情况
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpts = find_checkpoints(args.checkpoint_dir, recursive=args.recursive)
    if args.start_epoch is not None:
        ckpts = [p for p in ckpts if extract_epoch(p) >= args.start_epoch]
    if args.end_epoch is not None:
        ckpts = [p for p in ckpts if extract_epoch(p) <= args.end_epoch]

    if len(ckpts) == 0:
        raise RuntimeError('没有找到 epoch checkpoint。请检查 --checkpoint_dir 或是否需要 --recursive。')

    adj, mean, std, val_loader, test_loader, time_mode = build_dataloaders(args)
    model = TSEDGCA(
        adj,
        args.in_dim,
        args.embed_size,
        args.seq_len,
        args.pre_len,
        args.heads,
        args.forward_expansion,
        args.dropout
    ).to(device)

    if args.output_csv is None:
        out_dir = args.checkpoint_dir if os.path.isdir(args.checkpoint_dir) else os.path.dirname(args.checkpoint_dir)
        args.output_csv = os.path.join(out_dir, '{}_epoch_test_scan.csv'.format(args.dataset))

    print('Dataset:', args.dataset)
    print('Time mode:', time_mode)
    print('Checkpoint num:', len(ckpts))
    print('Output CSV:', args.output_csv)

    rows = []
    best_test = None
    best_val = None

    for idx, ckpt in enumerate(ckpts, 1):
        epoch = extract_epoch(ckpt)
        try:
            load_checkpoint(model, ckpt, device)
            val_rmse, val_mae, val_mape, val_wape = evaluate(model, val_loader, mean, std, adj, device)
            test_rmse, test_mae, test_mape, test_wape = evaluate(model, test_loader, mean, std, adj, device)
        except Exception as e:
            print('[{}/{}] epoch {} failed: {} | {}'.format(idx, len(ckpts), epoch, os.path.basename(ckpt), e))
            continue

        row = {
            'epoch': epoch,
            'checkpoint': ckpt,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_mape': val_mape,
            'val_wape': val_wape,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'test_wape': test_wape,
        }
        rows.append(row)

        if best_test is None or test_mae < best_test['test_mae']:
            best_test = row
        if best_val is None or val_mae < best_val['val_mae']:
            best_val = row

        print(
            '[{}/{}] epoch {:03d} | val MAE {:.6f} | test MAE {:.6f} | {}'.format(
                idx, len(ckpts), epoch, val_mae, test_mae, os.path.basename(ckpt)
            )
        )

        # 每轮都写入，防止中途被打断后结果丢失
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

    print('\n===== Summary =====')
    if best_val is not None:
        print(
            'Best by VAL: epoch {} | val MAE {:.6f} | test MAE {:.6f} | {}'.format(
                best_val['epoch'], best_val['val_mae'], best_val['test_mae'], best_val['checkpoint']
            )
        )
    if best_test is not None:
        print(
            'Best by TEST (diagnosis only): epoch {} | val MAE {:.6f} | test MAE {:.6f} | {}'.format(
                best_test['epoch'], best_test['val_mae'], best_test['test_mae'], best_test['checkpoint']
            )
        )

    print('\n结果已保存到:', args.output_csv)
    print('提醒：Best by TEST 只能用于诊断，不建议作为论文正式模型选择依据。')


if __name__ == '__main__':
    main()
