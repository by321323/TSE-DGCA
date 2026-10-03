# TSE-DGCA

Source code for traffic forecasting with dynamic graph construction, graph convolution, attention, and incremental temporal-evolution components.

## Requirements

- Python 3.7
- PyTorch 1.7.1
- SciPy 1.7.0
- pandas 1.3.1
- NumPy 1.20.3

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Data preparation

The code supports the public PEMS03, PEMS04, PEMS07, and PEMS08 traffic datasets. Dataset files are not committed to this repository because of their size. They can be obtained from the dataset sources linked by the [STSGCN project](https://github.com/Davidham3/STSGCN) or from this [PEMS dataset folder](https://drive.google.com/drive/folders/1wxNZtR_a8uYm7E-JT1qIwEWNUehlQ6xM?usp=sharing).

Place each dataset and its adjacency matrix under the corresponding directory:

```text
data/
├── PEMS03/
│   ├── PEMS03.npz
│   └── adj.npy
├── PEMS04/
│   ├── PEMS04.npz
│   └── adj.npy
├── PEMS07/
│   ├── PEMS07.npz
│   └── adj.npy
└── PEMS08/
    ├── PEMS08.npz
    └── adj.npy
```

## Training

For example, train the model on PEMS08 with:

```bash
python train.py --dataset PEMS08
```

The main options include `--dataset`, `--train_rate`, `--seq_len`, `--pre_len`, `--batchsize`, `--heads`, `--dropout`, `--lr`, `--in_dim`, `--embed_size`, and `--epochs`.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python test.py --dataset PEMS08 --model Model/PEMS/PEMS08/best_model/best_model.pkl
```

Multi-horizon evaluation is available through `test_multihorizon.py`. To evaluate multiple epoch checkpoints and export a summary, use `batch_test_epochs.py`.
