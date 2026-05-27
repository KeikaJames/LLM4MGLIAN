# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.rmsnorm import GroupedRMSNorm


try:
    from mamba_ssm.modules.mamba3 import Mamba3 as OfficialMamba3
except ImportError:
    OfficialMamba3 = None


def official_available() -> bool:
    return OfficialMamba3 is not None


class NaiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 64,
        d_conv: int = 4,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if expand <= 0:
            raise ValueError("expand must be positive")
        if headdim <= 0:
            raise ValueError("headdim must be positive")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.headdim = headdim
        self.d_conv = d_conv

        if self.d_inner % headdim != 0:
            raise ValueError("d_model * expand must be divisible by headdim")

        self.nheads = self.d_inner // headdim

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.nheads + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.nheads, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.rand(self.nheads, d_state) + 0.5))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

        bsz, seq_len, _ = x.shape
        dtype = x.dtype

        mask = None
        if attn_mask is not None:
            if attn_mask.shape != (bsz, seq_len):
                raise ValueError("attn_mask must have shape [B, L]")
            mask = attn_mask.to(device=x.device, dtype=torch.float32)

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u = self.conv1d(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        u = F.silu(u)

        if mask is not None:
            u = u * mask.unsqueeze(-1).to(dtype=u.dtype)
            z = z * mask.unsqueeze(-1).to(dtype=z.dtype)

        params = self.x_proj(u)
        dt, b_param, c_param = params.split(
            [self.nheads, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))

        u_h = u.reshape(bsz, seq_len, self.nheads, self.headdim)
        dt_h = dt.reshape(bsz, seq_len, self.nheads, self.headdim)

        a = -torch.exp(self.A_log.float())

        state = torch.zeros(
            bsz,
            self.nheads,
            self.headdim,
            self.d_state,
            device=x.device,
            dtype=torch.float32,
        )

        ys: list[torch.Tensor] = []

        for t in range(seq_len):
            dt_t = dt_h[:, t].float()
            u_t = u_h[:, t].float()
            b_t = b_param[:, t].float().view(bsz, 1, 1, self.d_state)
            c_t = c_param[:, t].float().view(bsz, 1, 1, self.d_state)

            da = torch.exp(dt_t.unsqueeze(-1) * a.view(1, self.nheads, 1, self.d_state))
            bu = dt_t.unsqueeze(-1) * b_t * u_t.unsqueeze(-1)

            next_state = state * da + bu
            if mask is not None:
                active = mask[:, t].view(bsz, 1, 1, 1).bool()
                state = torch.where(active, next_state, state)
            else:
                state = next_state

            y_t = (state * c_t).sum(dim=-1)
            y_t = y_t + self.D.view(1, self.nheads, 1) * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y.reshape(bsz, seq_len, self.d_inner).to(dtype)

        if mask is not None:
            y = y * mask.unsqueeze(-1).to(dtype=y.dtype)

        y = y * F.silu(z)

        return self.out_proj(y)


class Mamba3Layer(nn.Module):
    def __init__(
        self,
        cfg,
        layer_idx: int | None = None,
        num_groups: int = 8,
    ):
        super().__init__()

        self.cfg = cfg
        self.layer_idx = layer_idx

        self.norm = GroupedRMSNorm(
            cfg.d_model,
            num_groups=num_groups,
            eps=cfg.rmsnorm_eps,
        )

        use_official = bool(cfg.use_official_mamba and OfficialMamba3 is not None)

        if use_official:
            self.mamba = self._build_official(cfg, layer_idx)
            self.backend = "official"
        else:
            self.mamba = NaiveSSM(
                d_model=cfg.d_model,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim,
                d_conv=cfg.mamba_d_conv,
            )
            self.backend = "fallback"

    def _build_official(self, cfg, layer_idx: int | None):
        try:
            return OfficialMamba3(
                d_model=cfg.d_model,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim,
                d_conv=cfg.mamba_d_conv,
                layer_idx=layer_idx,
            )
        except TypeError:
            return OfficialMamba3(
                d_model=cfg.d_model,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim,
                layer_idx=layer_idx,
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")

        mask = None
        if attn_mask is not None:
            if attn_mask.shape != x.shape[:2]:
                raise ValueError("attn_mask must have shape [B, L]")
            if not isinstance(self.mamba, NaiveSSM) and not bool(attn_mask.all()):
                raise ValueError(
                    "official Mamba backend does not support masked state updates; "
                    "use fallback Mamba or pass an all-ones attention mask"
                )
            mask = attn_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

        residual = x if mask is None else x * mask
        mamba_input = self.norm(residual)

        if isinstance(self.mamba, NaiveSSM):
            y = self.mamba(mamba_input, attn_mask=attn_mask)
        else:
            y = self.mamba(mamba_input)
            if mask is not None:
                y = y * mask

        out = residual + y
        if mask is not None:
            out = out * mask
        return out


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    layer = Mamba3Layer(cfg, layer_idx=0)
    layer.eval()

    print("Mamba3Layer")
    print(f"  backend: {layer.backend}")
    print(f"  official_available: {official_available()}")

    bsz, seq_len = 2, 16
    x = torch.randn(bsz, seq_len, cfg.d_model)

    y = layer(x)

    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in layer.parameters()):,}")

    x2 = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    loss = layer(x2).sum()
    loss.backward()

    print(f"  grad_norm: {x2.grad.norm().item():.6f}")

    x3 = torch.randn(1, 8, cfg.d_model)
    out_full = layer(x3)

    x3_mod = x3.clone()
    x3_mod[0, 5:] += 10.0
    out_mod = layer(x3_mod)

    diff = (out_full[0, :5] - out_mod[0, :5]).abs().max().item()
    print(f"  causal_diff_before_changed_span: {diff:.6e}")

    ignored = layer(
        x,
        word_pos=torch.zeros(bsz, seq_len, dtype=torch.long),
        morph_depth=torch.zeros(bsz, seq_len, dtype=torch.long),
        attn_mask=torch.ones(bsz, seq_len, dtype=torch.long),
    )

    print(f"  kwargs_ignored_shape: {tuple(ignored.shape)}")


if __name__ == "__main__":
    _check()
