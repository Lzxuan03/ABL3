import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}."""
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
    d_mat = torch.diag(deg_inv_sqrt)
    return d_mat @ adj @ d_mat


def chebyshev_polynomials(adj: torch.Tensor, order_k: int):
    """
    Build Chebyshev polynomial supports:
      T0 = I, T1 = A, Tk = 2A*Tk-1 - Tk-2
    """
    n = adj.size(0)
    t0 = torch.eye(n, device=adj.device, dtype=adj.dtype)
    if order_k == 1:
        return [t0]
    t1 = adj
    supports = [t0, t1]
    for _ in range(2, order_k):
        supports.append(2.0 * adj @ supports[-1] - supports[-2])
    return supports


class ChebGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, order_k: int, bias: bool = True):
        super().__init__()
        self.order_k = int(order_k)
        self.weight = nn.Parameter(torch.empty(self.order_k, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: [B, N, Fin], adj: [N, N]
        supports = chebyshev_polynomials(adj, self.order_k)
        out = 0.0
        for k, sk in enumerate(supports):
            xk = torch.einsum("nm,bmf->bnf", sk, x)
            out = out + torch.matmul(xk, self.weight[k])
        if self.bias is not None:
            out = out + self.bias
        return out


class DGCNN(nn.Module):
    """
    Dynamical graph-style classifier for EEG/eye features.

    Interface kept compatible with existing training scripts:
      - ctor args: in_channels, num_electrodes, k_adj, out_channels, num_classes
      - forward: returns (logits, [feat_mid, feat_final])
    """

    def __init__(
        self,
        in_channels: int,
        num_electrodes: int,
        k_adj: int = 2,
        out_channels: int = 16,
        num_classes: int = 5,
        hidden_channels: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_electrodes = int(num_electrodes)
        # Keep legacy argument name `k_adj` but interpret it as Chebyshev order controller.
        # Typical choice in DGCNN literature is low-order (2~3).
        self.k_adj = int(k_adj)
        self.cheb_k = max(2, self.k_adj + 1)
        self.out_channels = int(out_channels)

        self.adj_param = nn.Parameter(torch.empty(self.num_electrodes, self.num_electrodes))
        self.gc1 = ChebGraphConv(self.in_channels, hidden_channels, order_k=self.cheb_k)
        self.gc2 = ChebGraphConv(hidden_channels, self.out_channels, order_k=self.cheb_k)
        self.bn1 = nn.BatchNorm1d(self.num_electrodes * hidden_channels)
        self.bn2 = nn.BatchNorm1d(self.num_electrodes * self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.num_electrodes * self.out_channels, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.adj_param)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[1] == self.num_electrodes and x.shape[2] == self.in_channels:
                return x
            raise ValueError(
                f"Expected input [B,{self.num_electrodes},{self.in_channels}], got {tuple(x.shape)}"
            )

        if x.dim() != 2:
            raise ValueError(f"Expected 2D/3D tensor, got shape {tuple(x.shape)}")

        b, d = x.shape
        if d == self.num_electrodes * self.in_channels:
            return x.reshape(b, self.num_electrodes, self.in_channels)
        if self.in_channels == 1 and d == self.num_electrodes:
            return x.unsqueeze(-1)

        raise ValueError(
            f"Input dim mismatch: got [B,{d}], expected {self.num_electrodes*self.in_channels} "
            f"or {self.num_electrodes} (when in_channels=1)."
        )

    def _build_adj(self) -> torch.Tensor:
        # Learnable non-negative adjacency with self-loop and symmetric normalization.
        a = 0.5 * (self.adj_param + self.adj_param.t())
        a = F.relu(a)
        a = a + torch.eye(self.num_electrodes, device=a.device, dtype=a.dtype)
        return normalize_adj(a)

    def forward(self, x: torch.Tensor):
        x = self._reshape_input(x)
        adj = self._build_adj()

        h1 = self.gc1(x, adj)
        h1 = F.elu(h1)
        feat_mid = h1.reshape(h1.shape[0], -1)
        feat_mid = self.bn1(feat_mid)
        feat_mid = self.dropout(feat_mid)
        h1 = feat_mid.reshape(h1.shape[0], self.num_electrodes, -1)

        h2 = self.gc2(h1, adj)
        h2 = F.elu(h2)
        feat_final = h2.reshape(h2.shape[0], -1)
        feat_final = self.bn2(feat_final)
        feat_final = self.dropout(feat_final)

        logits = self.classifier(feat_final)
        return logits, [feat_mid, feat_final]
