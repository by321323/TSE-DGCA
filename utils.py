import numpy as np


_RETIRED_PARAMETER_PARTS = (
    'future_time_norm.',
    'future_node_emb',
    'future_residual_mlp.',
    'future_time_scale',
    '.conv1.',
    '.conv4.',
    '.conv5.',
    '.T_G.',
    '.T_L.',
    '.norm2.',
    '.norm3.',
    '.norm4.',
    '.conv2.',
    '.conv3.',
    '.global_temporal_conv_gate',
    '.local_tempooral_conv_gate',
    '.fs.',
    '.fg.',
    '.increment_temporal.fuse.',
    '.S.norm5.',
)


def load_tsedgca_state_dict(model, state):
    """加载 TSE-DGCA 权重，并仅忽略本次清理掉的旧分支参数。"""
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not any(part in key for part in _RETIRED_PARAMETER_PARTS)
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            'Checkpoint does not match TSE-DGCA. Missing keys: {}; '
            'unexpected keys: {}'.format(incompatible.missing_keys, unexpected)
        )
    return incompatible.unexpected_keys


def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape
