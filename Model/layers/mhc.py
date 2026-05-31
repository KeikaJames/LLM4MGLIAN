# -*- coding: utf-8 -*-

"""Manifold-Constrained Hyper-Connections (mHC).

Reference: DeepSeek, *Manifold-Constrained Hyper-Connections* (arXiv:2512.24880),
building on Hyper-Connections (arXiv:2409.19606).

A residual connection is replaced by ``n`` parallel residual streams plus three
learnable maps, each constrained to a bounded manifold so the composite gain of
a deep / recurrent stack stays bounded:

* ``H_pre = sigmoid(.)``      — per-stream input aggregation weights in ``(0, 1)``.
* ``H_post = 2 * sigmoid(.)`` — per-stream output write weights in ``(0, 2)``.
* ``H_res = sinkhorn(.)``     — ``n x n`` doubly-stochastic cross-stream mixing
  (``||H_res||_2 <= 1`` by Birkhoff), which is what prevents the write term
  ``H_post * F`` from accumulating and exploding across layers.

The connection is meant to live *inside* a layer (one instance per residual),
not wrapped around a recurrent loop. Expand once, keep ``n`` streams across
layers, collapse once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Project ``exp(log_alpha)`` onto the Birkhoff polytope (doubly stochastic).

    Operates in log space for numerical stability by alternating row and column
    log-normalization. The last two dims are treated as the ``n x n`` matrix; any
    leading dims are batched.
    """

    if log_alpha.ndim < 2:
        raise ValueError("log_alpha must have at least 2 dims")
    if log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError("log_alpha must be square in its last two dims")
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")

    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

    return torch.exp(log_alpha)


class ManifoldHyperConnection(nn.Module):
    """A single manifold-constrained hyper-connection over ``n`` streams.

    ``forward(streams, fn)`` aggregates the streams into one tensor, applies the
    wrapped layer ``fn`` once, mixes the streams through the doubly-stochastic
    residual matrix, and writes the layer output back into every stream:

        agg          = sum_i H_pre_i * streams_i
        out          = fn(agg)
        new_streams  = H_res @ streams + H_post (x) out

    At initialization the maps reduce to the vanilla residual ``x + fn(x)`` with
    unit gain (see the init notes below), so an mHC layer is a drop-in for a
    standard pre-norm residual layer.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        constrain: bool = True,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if sinkhorn_iters < 0:
            raise ValueError("sinkhorn_iters must be non-negative")

        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.constrain = constrain

        n = n_streams

        # H_pre: init so sum_i sigmoid(pre_i) == 1 with all streams equal, i.e.
        # sigmoid(pre_i) == 1 / n, so the aggregation of ``n`` copies of x is x.
        # For n == 1 that means sigmoid(pre) == 1, approximated by a large logit.
        if n > 1:
            pre_value = float(torch.log(torch.tensor(1.0 / (n - 1))))
            pre_init = torch.full((n,), pre_value)
        else:
            pre_init = torch.full((1,), 20.0)
        self.pre_logits = nn.Parameter(pre_init)

        # H_post: init at logit 0 -> 2 * sigmoid(0) == 1, so the layer output is
        # written into each stream exactly once (vanilla residual).
        self.post_logits = nn.Parameter(torch.zeros(n))

        # H_res: init near identity (doubly stochastic) so streams stay distinct
        # at the start; Sinkhorn of a strongly diagonal matrix is ~identity.
        self.res_logits = nn.Parameter(torch.eye(n) * 3.0)

    def residual_matrix(self) -> torch.Tensor:
        """Return the current ``n x n`` cross-stream mixing matrix.

        Doubly stochastic when ``constrain`` is True; an unconstrained positive
        matrix otherwise (used only as a stability control in tests).
        """

        if self.constrain:
            return sinkhorn_knopp(self.res_logits, self.sinkhorn_iters)
        return F.softplus(self.res_logits)

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate ``[B, L, d]`` into ``n`` streams ``[B, L, n, d]``."""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

        return x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).contiguous()

    def collapse(self, streams: torch.Tensor) -> torch.Tensor:
        """Reduce ``n`` streams ``[B, L, n, d]`` back to ``[B, L, d]`` (mean)."""

        self._check_streams(streams)
        return streams.mean(dim=-2)

    def forward(self, streams: torch.Tensor, fn) -> torch.Tensor:
        self._check_streams(streams)

        pre = torch.sigmoid(self.pre_logits)
        agg = torch.einsum("n,blnd->bld", pre, streams)

        out = fn(agg)
        if out.shape != agg.shape:
            raise ValueError("wrapped fn must preserve [B, L, d_model] shape")

        res = self.residual_matrix().to(dtype=streams.dtype)
        mixed = torch.einsum("ij,bljd->blid", res, streams)

        post = (2.0 * torch.sigmoid(self.post_logits)).to(dtype=streams.dtype)
        write = post.view(1, 1, self.n_streams, 1) * out.unsqueeze(-2)

        return mixed + write

    def _check_streams(self, streams: torch.Tensor) -> None:
        if streams.ndim != 4:
            raise ValueError("streams must have shape [B, L, n_streams, d_model]")
        if streams.shape[-2] != self.n_streams:
            raise ValueError(
                f"expected n_streams={self.n_streams}, got {streams.shape[-2]}"
            )
        if streams.shape[-1] != self.d_model:
            raise ValueError(
                f"expected d_model={self.d_model}, got {streams.shape[-1]}"
            )


def _check() -> None:
    torch.manual_seed(0)

    n = 4
    mat = sinkhorn_knopp(torch.randn(n, n), n_iters=20)
    print("sinkhorn_knopp")
    print(f"  row_sums: {mat.sum(dim=-1).tolist()}")
    print(f"  col_sums: {mat.sum(dim=-2).tolist()}")

    d_model = 16
    hc = ManifoldHyperConnection(d_model, n_streams=n, sinkhorn_iters=20)
    hc.eval()

    x = torch.randn(2, 5, d_model)
    streams = hc.expand(x)

    def zero_fn(inp):
        return torch.zeros_like(inp)

    out = hc(streams, zero_fn)
    print("ManifoldHyperConnection")
    print(f"  streams: {tuple(streams.shape)} -> {tuple(out.shape)}")

    # Composite gain of pure cross-stream mixing (F = 0) over 60 layers.
    s = streams
    for _ in range(60):
        s = hc(s, zero_fn)
    gain = hc.collapse(s).norm() / hc.collapse(streams).norm()
    print(f"  composite_gain_60_layers: {gain.item():.4f}")

    xg = torch.randn(2, 5, d_model, requires_grad=True)
    loss = hc.collapse(hc(hc.expand(xg), lambda inp: inp)).sum()
    loss.backward()
    print(f"  grad_norm: {xg.grad.norm().item():.6f}")


if __name__ == "__main__":
    _check()
