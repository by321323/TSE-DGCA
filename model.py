# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# from main import device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context

class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, T, C = input_Q.shape

        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class SATT(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion, T_dim, topk=None):
        super(SATT, self).__init__()
        self.adj = adj
        self.T_dim = T_dim
        
        if topk is None:
            self.topk = max(8, int(0.05 * adj.shape[0]))
        else:
            self.topk = min(topk, adj.shape[0])

        # ===== 原版保留：空间自注意力主支路 =====
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.Att = SMultiHeadAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.norm5 = nn.LayerNorm(embed_size)
        self.norm6 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

        # ===== 最后的空间双支路门控融合 =====
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

        # ===== 新增：时间上下文提取器（替代 DGCRN 里 H_{t-1} 的作用）=====
        # 输入 [B, C, T, N]，输出 [B, C, T, N]
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(embed_size, embed_size, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(embed_size, embed_size, kernel_size=(3, 1), padding=(1, 0))
        )

        # ===== 新增：时间步嵌入 =====
        self.time_embedding = nn.Embedding(T_dim, embed_size)

        # ===== 新增：hyper-network / dynamic filter generator =====
        # 输入从 [x_t, c_t, e_t] 变成 [x_t, c_t, delta_t, e_t]
        self.filter_mlp1 = nn.Sequential(
            nn.Linear(embed_size * 4, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.filter_mlp2 = nn.Sequential(
            nn.Linear(embed_size * 4, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

        # 平滑变化量进入动态图生成器前的投影
        self.delta_context_proj = nn.Linear(embed_size, embed_size)

        # ===== 新增：delta 归一化层 =====
        self.delta_context_norm = nn.LayerNorm(embed_size)

        # 控制变化量对动态图生成器的影响强度
        # 初始较小，避免一开始破坏动态图主干
        self.delta_context_scale = nn.Parameter(torch.tensor(-2.0))

        # ===== 新增：静态节点嵌入，受 DGCRN 启发 =====
        self.node_emb1 = nn.Parameter(torch.randn(adj.shape[0], embed_size))
        self.node_emb2 = nn.Parameter(torch.randn(adj.shape[0], embed_size))

        # ===== 新增：静态图 / 动态图传播投影 =====
        self.static_proj = nn.Linear(embed_size, embed_size)
        self.dynamic_proj = nn.Linear(embed_size, embed_size)

        # ===== 新增：静态图 / 动态图门控融合 =====
        self.gate_static = nn.Linear(embed_size, embed_size)
        self.gate_dynamic = nn.Linear(embed_size, embed_size)

        # ===== 新增：动态图生成的缩放与静态偏置 =====
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

        # 方便可视化查看
        self.last_dynamic_adj = None

    def _normalize_static_adj(self, A):
        # 原版风格：D^{-1/2} A D^{-1/2}
        D = (A.sum(-1) + 1e-6).pow(-0.5)
        D = torch.diag_embed(D)
        A_norm = torch.matmul(torch.matmul(D, A), D)
        return A_norm

    def _build_dynamic_adj(self, de1, de2, A_static):
        # de1, de2: [B, N, C]
        score1 = torch.matmul(de1, de2.transpose(1, 2))
        score2 = torch.matmul(de2, de1.transpose(1, 2))

        # 受 DGCRN 思想启发：先生成动态节点嵌入，再构图
        score = torch.relu(torch.tanh(self.alpha * (score1 - score2)))

        # 加静态图偏置，避免动态图完全漂移
        score = score + self.beta * A_static.unsqueeze(0)

        # top-k 稀疏化
        if self.topk < score.size(-1):
            topk_val, topk_idx = torch.topk(score, k=self.topk, dim=-1)
            masked_score = torch.full_like(score, float('-inf'))
            score = masked_score.scatter(-1, topk_idx, topk_val)

        A_dyn = torch.softmax(score, dim=-1)
        return A_dyn

    def forward(self, value, key, query, delta_hint=None):
        # query: [B, N, T, C]
        B, N, T, C = query.shape
        device = query.device

        if delta_hint is None:
            delta_hint = torch.zeros_like(query)

        # =========================================================
        # 1) 原版保留：空间自注意力主支路
        # =========================================================
        attention = self.attention(query, query, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))   # [B, N, T, C]

        # =========================================================
        # 2) 新增：真实动态图支路
        # =========================================================
        # 时间上下文 Ctx: [B, N, T, C]
        ctx = self.ctx_conv(query.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # 归一化静态图
        A_static = self._normalize_static_adj(self.adj.to(device))

        time_ids = torch.arange(T, device=device)
        time_emb = self.time_embedding(time_ids)  # [T, C]

        X_G_list = []
        last_A_dyn = None

        for t in range(T):
            x_t = query[:, :, t, :]           # [B, N, C]
            c_t = ctx[:, :, t, :]             # [B, N, C]
            d_t = delta_hint[:, :, t, :]      # [B, N, C]
            e_t = time_emb[t].view(1, 1, C).expand(B, N, C)

            # ===== 平滑变化量投影，先用归一化稳定分布 =====
            delta_scale = torch.sigmoid(self.delta_context_scale)

            d_t = self.delta_context_norm(d_t)
            d_t = delta_scale * self.delta_context_proj(d_t)

            # graph generator 输入:
            # 当前状态 + 时间上下文 + 平滑变化量 + 时间嵌入
            i_t = torch.cat([x_t, c_t, d_t, e_t], dim=-1)   # [B, N, 4C]

            # dynamic filters
            df1 = self.filter_mlp1(i_t)   # [B, N, C]
            df2 = self.filter_mlp2(i_t)   # [B, N, C]

            # 静态节点嵌入 -> 动态节点嵌入
            E1 = self.node_emb1.unsqueeze(0).expand(B, -1, -1)   # [B, N, C]
            E2 = self.node_emb2.unsqueeze(0).expand(B, -1, -1)

            de1 = torch.tanh(self.alpha * (df1 * E1))
            de2 = torch.tanh(self.alpha * (df2 * E2))

            # 构建时刻 t 的动态图
            A_dyn = self._build_dynamic_adj(de1, de2, A_static)  # [B, N, N]
            last_A_dyn = A_dyn

            # 静态图传播
            x_static = torch.relu(self.static_proj(torch.matmul(A_static.unsqueeze(0), x_t)))

            # 动态图传播
            x_dynamic = torch.relu(self.dynamic_proj(torch.matmul(A_dyn, x_t)))

            # 静态 + 动态门控融合
            g_t = torch.sigmoid(self.gate_static(x_static) + self.gate_dynamic(x_dynamic))
            x_out = g_t * x_static + (1 - g_t) * x_dynamic   # [B, N, C]

            X_G_list.append(x_out.unsqueeze(2))

        X_G = torch.cat(X_G_list, dim=2)   # [B, N, T, C]
        self.last_dynamic_adj = last_A_dyn.detach()

        # =========================================================
        # 3) 原版保留：动态图支路后接注意力 + FFN
        # =========================================================
        X_G = self.norm6(query + X_G)
        Attention1 = self.Att(X_G, X_G, X_G)
        y = self.dropout(self.norm3(Attention1 + X_G))
        forward1 = self.feed_forward1(y)
        X_G = self.dropout(self.norm4(y + forward1))

        # =========================================================
        # 4) 原版保留：两条空间支路门控融合
        # =========================================================
        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))
        out = g * U_S + (1 - g) * X_G
        return out


class TATT(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TATT, self).__init__()
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, t):
        attention = self.attention(query, query, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(Attention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SmoothIncrementTemporalModule(nn.Module):
    """
    平滑增量驱动的时间演化模块。

    输入:
        x: [B, N, T, H]

    输出:
        out: [B, N, T, H]
        delta_hint: [B, N, T, H]

    设计目标:
        1. 先平滑，再求变化量，避免硬差分放大噪声
        2. 用衰减式累计变化，避免 cumsum 数值失衡
        3. 前端轻量提取状态/变化/累计变化，后端只用一个时间注意力
    """
    def __init__(self, embed_size, heads, dropout=0.0, forward_expansion=4):
        super(SmoothIncrementTemporalModule, self).__init__()

        self.embed_size = embed_size

        # 1. 时间平滑层：只沿时间维做轻量平滑
        # 使用“深度卷积 + 逐点卷积”，并初始化成真正的 3 步平均平滑。
        # 这样一开始就是降噪平滑，而不是随机卷积放大噪声。
        self.smooth_dw = nn.Conv2d(
            embed_size,
            embed_size,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=embed_size,
            bias=False
        )
        self.smooth_pw = nn.Conv2d(
            embed_size,
            embed_size,
            kernel_size=1,
            bias=False
        )

        with torch.no_grad():
            self.smooth_dw.weight.fill_(1.0 / 3.0)
            self.smooth_pw.weight.zero_()
            for i in range(embed_size):
                self.smooth_pw.weight[i, i, 0, 0] = 1.0

        # 2. 三类轻量特征投影
        self.state_proj = nn.Linear(embed_size, embed_size)
        self.delta_proj = nn.Linear(embed_size, embed_size)
        self.accum_proj = nn.Linear(embed_size, embed_size)

        # 3. 三类信息融合（保留原来的 fuse）
        self.fuse = nn.Sequential(
            nn.LayerNorm(embed_size * 3),
            nn.Linear(embed_size * 3, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )

        # ===== 新增：门控融合机制 =====
        self.branch_gate = nn.Sequential(
            nn.LayerNorm(embed_size * 3),
            nn.Linear(embed_size * 3, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, 3)
        )

        # 分支先验：初始更偏向状态值，避免变化量一开始过强
        self.branch_prior = nn.Parameter(
            torch.tensor([2.0, -1.0, -1.0], dtype=torch.float32)
        )

        self.last_branch_weights = None

        # 4. 只用一个统一时间注意力，避免三个 TATT 太重
        self.temporal_attention = TATT(embed_size, heads, dropout, forward_expansion)

        self.norm_in = nn.LayerNorm(embed_size)
        self.norm_out = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm_ffn = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        # 衰减式累计变化系数
        # sigmoid(0)=0.5，表示一开始不偏向极强累计，也不完全忽略历史
        self.accum_decay = nn.Parameter(torch.tensor(0.0))

        # 方便观察
        self.last_delta_strength = None
        self.last_accum_decay = None

    def smooth(self, x):
        """
        x: [B, N, T, H]
        return: [B, N, T, H]
        """
        x_perm = x.permute(0, 3, 2, 1)          # [B, H, T, N]
        x_smooth = self.smooth_dw(x_perm)       # [B, H, T, N]
        x_smooth = self.smooth_pw(x_smooth)     # [B, H, T, N]
        x_smooth = x_smooth.permute(0, 3, 2, 1) # [B, N, T, H]
        return x_smooth

    def build_delta(self, x_smooth):
        """
        用平滑后的特征求相邻变化量，避免硬差分放大噪声。
        这里不使用 delta[:, :, 1:, :] = ... 这种原地写入，避免反向传播报错。
        """
        diff = x_smooth[:, :, 1:, :] - x_smooth[:, :, :-1, :]
        first = torch.zeros_like(x_smooth[:, :, :1, :])
        delta = torch.cat([first, diff], dim=2)
        return delta

    def build_accum_delta(self, delta):
        """
        衰减式累计变化，替代 torch.cumsum。
        避免越靠后的时间步数值越大。
        这里用列表 + torch.stack，避免 accum[:, :, t, :] = ... 原地写入。
        """
        B, N, T, H = delta.shape
        decay = torch.sigmoid(self.accum_decay)
        self.last_accum_decay = decay.detach()

        states = []
        prev = delta[:, :, 0, :]
        states.append(prev)

        for t in range(1, T):
            prev = decay * prev + (1.0 - decay) * delta[:, :, t, :]
            states.append(prev)

        accum = torch.stack(states, dim=2)
        return accum

    def get_delta_hint(self, x):
        """
        给动态图生成器使用的平滑变化量。
        """
        x_smooth = self.smooth(x)
        delta = self.build_delta(x_smooth)
        return delta

    def forward(self, x, delta_hint=None):
        """
        x: [B, N, T, H]
        """
        if delta_hint is None:
            x_smooth = self.smooth(x)
            delta = self.build_delta(x_smooth)
        else:
            delta = delta_hint

        accum = self.build_accum_delta(delta)

        # 记录变化量强度，方便测试观察
        self.last_delta_strength = delta.abs().mean().detach()

        state_feat = self.state_proj(x)
        delta_feat = self.delta_proj(delta)
        accum_feat = self.accum_proj(accum)

        # ===== 门控融合机制 =====
        branch_input = torch.cat([state_feat, delta_feat, accum_feat], dim=-1)

        scores = self.branch_gate(branch_input)
        scores = scores + self.branch_prior.view(1, 1, 1, 3)

        weights = torch.softmax(scores, dim=-1)
        self.last_branch_weights = weights.detach()

        fused = (
            weights[..., 0:1] * state_feat +
            weights[..., 1:2] * delta_feat +
            weights[..., 2:3] * accum_feat
        )

        fused = self.norm_in(fused + x)

        # 统一时间注意力
        out = self.temporal_attention(fused, fused, fused, 4)

        out = self.dropout(self.norm_out(out + x))

        ff = self.feed_forward(out)
        out = self.dropout(self.norm_ffn(ff + out))

        return out


class STBlock(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout=0
    ):
        super(STBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv4 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv5 = nn.Conv2d(in_channels, embed_size, 1)
        self.S = SATT(embed_size, heads, adj, dropout, forward_expansion, T_dim)
        self.T_G = TATT(embed_size, heads, dropout, forward_expansion)
        self.T_L = TATT(embed_size, heads, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.Att = Attention(d_model=embed_size * T_dim, d_k=embed_size * 2, d_v=embed_size * 2, h=heads, dropout=0)
        # 缩小时间维度
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.global_temporal_conv_gate1 = nn.Conv2d(embed_size, T_dim, (T_dim, 1))
        self.global_temporal_conv_gate2 = nn.Conv2d(embed_size, T_dim, (T_dim, 1))
        self.local_tempooral_conv_gate1 = nn.Conv2d(embed_size, T_dim, (T_dim // 3, 1))
        self.local_tempooral_conv_gate2 = nn.Conv2d(embed_size, T_dim, (T_dim // 3, 1))
        self.pool = nn.MaxPool2d((T_dim - T_dim // 3 + 1, 1))
        self.fs = nn.Linear(embed_size * T_dim, embed_size * T_dim)
        self.fg = nn.Linear(embed_size * T_dim, embed_size * T_dim)

        # ===== 新增：增量驱动的时间演化模块 =====
        # 用它替换原来的局部/全局时间卷积分支，作为 STBlock 主干时间模块。
        self.increment_temporal = SmoothIncrementTemporalModule(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

    def forward(self, x, delta_source=None):
        input_Transformer = x
        B, N, T, H = input_Transformer.shape

        # delta_source 用来计算交通变化量
        # 如果没有传，就默认用当前输入
        if delta_source is None:
            delta_source = input_Transformer

        # =========================================================
        # 1. 增量只从 delta_source 计算
        #    推荐 delta_source 使用未加入 TOD/DOW 的交通值特征
        # =========================================================
        delta_hint = self.increment_temporal.get_delta_hint(delta_source)

        # =========================================================
        # 2. 空间分支仍然使用 TOD/DOW 增强后的 input_Transformer
        #    但变化量来自交通值特征
        # =========================================================
        output_S = self.S(
            input_Transformer,
            input_Transformer,
            input_Transformer,
            delta_hint
        )
        output_S = self.norm1(output_S + input_Transformer)

        # =========================================================
        # 3. 时间分支完全由增量模块替代
        #    状态输入用 input_Transformer，变化量用 delta_hint
        # =========================================================
        output_T = self.increment_temporal(input_Transformer, delta_hint)

        out = self.Att(
            output_S.reshape(-1, N, T * H),
            output_T.reshape(-1, N, T * H),
            output_T.reshape(-1, N, T * H)
        ).reshape(-1, N, T, H)

        return out





class TSEDGCA(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout=0,
            use_tod=True,
            use_dow=True
    ):
        super(TSEDGCA, self).__init__()

        # ===== 时间嵌入开关 =====
        # TOD: 一天中的第几个 5 分钟片段，范围 0~287
        # DOW: 一周中的第几天，范围 0~6
        self.use_tod = use_tod
        self.use_dow = use_dow
        self.T_dim = T_dim
        self.output_T_dim = output_T_dim
        self.num_nodes = adj.shape[0]
        self.embed_size = embed_size

        # ===== 输入端历史 TOD/DOW 嵌入 =====
        if self.use_tod:
            self.tod_emb = nn.Embedding(288, embed_size)
        if self.use_dow:
            self.dow_emb = nn.Embedding(7, embed_size)

        # 控制输入端时间嵌入强度。
        # 这部分保留你 15.19 版本的思路。
        self.calendar_emb_scale = nn.Parameter(torch.tensor(-2.0))
        self.calendar_emb_norm = nn.LayerNorm(embed_size)
        self.last_calendar_scale = None

        # ===== 未来时间残差校正 =====
        # 注意：不再把未来时间嵌入直接加到预测头隐藏特征上，避免扰乱输出特征分布。
        # 这里改为：未来 TOD/DOW + 节点嵌入 -> 生成一个很小的预测残差。
        # 这样即使未来时间信息暂时没用，也可以通过很小的 scale 自动减弱影响。
        self.future_time_norm = nn.LayerNorm(embed_size)
        self.future_node_emb = nn.Parameter(torch.randn(self.num_nodes, embed_size) * 0.02)
        self.future_residual_mlp = nn.Sequential(
            nn.LayerNorm(embed_size * 2),
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, 1)
        )
        # 初始极小：sigmoid(-5)≈0.0067，避免从 15.19 版本明显扰动。
        self.future_time_scale = nn.Parameter(torch.tensor(-5.0))
        self.last_future_time_scale = None

        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.ST1 = STBlock(adj, in_channels, embed_size, T_dim, output_T_dim, heads, forward_expansion, dropout=dropout)
        self.ST2 = STBlock(adj, in_channels, embed_size, T_dim, output_T_dim, heads, forward_expansion, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)

    def _prepare_tod_ids(self, tod_ids, device):
        """
        tod_ids 支持两种格式：
        1) LongTensor，范围 0~287，shape [B, T]
        2) FloatTensor，范围 0~1，表示一天内比例，shape [B, T]
        """
        if tod_ids is None:
            return None
        tod_ids = tod_ids.to(device)
        if torch.is_floating_point(tod_ids):
            tod_ids = torch.floor(tod_ids * 288).long()
        else:
            tod_ids = tod_ids.long()
        return tod_ids.clamp(0, 287)

    def _prepare_dow_ids(self, dow_ids, device):
        """
        dow_ids 支持 LongTensor，范围 0~6，shape [B, T]。
        如果是浮点数，也会转成 long 后截断到 0~6。
        """
        if dow_ids is None:
            return None
        dow_ids = dow_ids.to(device)
        if torch.is_floating_point(dow_ids):
            dow_ids = torch.floor(dow_ids).long()
        else:
            dow_ids = dow_ids.long()
        return dow_ids.clamp(0, 6)

    def _build_time_embedding(self, tod_ids=None, dow_ids=None, device=None):
        """
        构造 TOD/DOW 时间嵌入，复用同一套输入端时间 embedding。
        这样同一个时间点在输入端和未来端含义一致，参数也更少。

        返回: [B, T, C]
        """
        if tod_ids is None and dow_ids is None:
            return None

        if tod_ids is not None:
            B, T = tod_ids.shape
        else:
            B, T = dow_ids.shape

        time_emb = torch.zeros(B, T, self.embed_size, device=device)

        if tod_ids is not None and self.use_tod:
            tod_ids = self._prepare_tod_ids(tod_ids, device)
            time_emb = time_emb + self.tod_emb(tod_ids)

        if dow_ids is not None and self.use_dow:
            dow_ids = self._prepare_dow_ids(dow_ids, device)
            time_emb = time_emb + self.dow_emb(dow_ids)

        return time_emb

    def _add_calendar_embedding(self, input_Transformer, tod_ids=None, dow_ids=None):
        """
        输入端历史时间嵌入。

        input_Transformer: [B, N, T, C]
        tod_ids: [B, T]
        dow_ids: [B, T]
        """
        if tod_ids is None and dow_ids is None:
            return input_Transformer

        B, N, T, C = input_Transformer.shape
        device = input_Transformer.device
        time_emb = self._build_time_embedding(tod_ids=tod_ids, dow_ids=dow_ids, device=device)
        if time_emb is None:
            return input_Transformer

        time_emb = self.calendar_emb_norm(time_emb)
        scale = torch.sigmoid(self.calendar_emb_scale)
        self.last_calendar_scale = scale.detach()

        time_emb = scale * time_emb
        time_emb = time_emb.unsqueeze(1).expand(-1, N, -1, -1)
        return input_Transformer + time_emb

    def _future_calendar_residual(self, future_tod_ids=None, future_dow_ids=None, num_nodes=None, device=None):
        """
        未来时间残差校正。

        不直接改预测头隐藏特征，而是生成一个很小的输出残差：
            residual: [B, pre_len, N]
        """
        if future_tod_ids is None and future_dow_ids is None:
            return None

        future_emb = self._build_time_embedding(
            tod_ids=future_tod_ids,
            dow_ids=future_dow_ids,
            device=device
        )
        if future_emb is None:
            return None

        future_emb = self.future_time_norm(future_emb)  # [B, pre_len, C]
        B, T_out, C = future_emb.shape
        N = self.num_nodes if num_nodes is None else num_nodes

        # [B, pre_len, 1, C] -> [B, pre_len, N, C]
        future_emb = future_emb.unsqueeze(2).expand(-1, -1, N, -1)

        # 每个节点有自己的响应特征，避免所有节点使用完全一样的未来时间偏置。
        node_emb = self.future_node_emb[:N].view(1, 1, N, C).expand(B, T_out, -1, -1)

        residual_input = torch.cat([future_emb, node_emb], dim=-1)  # [B, pre_len, N, 2C]
        residual = self.future_residual_mlp(residual_input).squeeze(-1)  # [B, pre_len, N]

        scale = torch.sigmoid(self.future_time_scale)
        self.last_future_time_scale = scale.detach()
        return scale * residual

    def forward(
            self,
            x,
            tod_ids=None,
            dow_ids=None,
            future_tod_ids=None,
            future_dow_ids=None
    ):
        x = x.permute(0, 2, 1).unsqueeze(1)

        # =========================================================
        # 1. 交通值特征：不包含 TOD/DOW
        # =========================================================
        value_Transformer = self.conv1(x)
        value_Transformer = value_Transformer.permute(0, 2, 3, 1)

        # =========================================================
        # 2. 时间增强特征：加入 TOD/DOW
        # =========================================================
        input_Transformer = self._add_calendar_embedding(
            value_Transformer,
            tod_ids=tod_ids,
            dow_ids=dow_ids
        )

        # =========================================================
        # 3. 第一层：主输入用 TOD/DOW 增强特征，
        #    但 delta 只从 value_Transformer 计算
        # =========================================================
        out = self.ST1(
            input_Transformer,
            delta_source=value_Transformer
        )

        out = self.norm1(out + input_Transformer)

        # =========================================================
        # 4. 第二层：可以继续用 out 自身计算增量
        # =========================================================
        out = self.ST2(out)

        # ===== 原预测头：保持 15.19 版本的稳定输出路径 =====
        out = out.permute(0, 2, 1, 3)       # [B, T, N, C]
        out = self.relu(self.conv2(out))    # [B, pre_len, N, C]
        out = out.permute(0, 3, 2, 1)       # [B, C, N, pre_len]
        out = self.conv3(out)               # [B, 1, N, pre_len]
        out = out.squeeze(1).permute(0, 2, 1)  # [B, pre_len, N]

        # ===== E1 对照实验：关闭未来 TOD/DOW 残差 =====
        # 保留历史 TOD/DOW 输入嵌入，但不再使用未来 TOD/DOW 修正输出。
        self.last_future_time_scale = torch.tensor(0.0, device=out.device)
        return out
