# LinearValueModel is m1:the linear model v(x) = w·x + b
# MVNN is the m2:monotone valuation neural network

import torch
import torch.nn as nn
import torch.optim as optim
from utils import bundle_to_vector
import config



class LinearValueModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.linear = nn.Linear(m, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# Monotone Valuation Neural Network

class MonotoneLinear(nn.Module):
    """Linear layer that maintains constraints (W >= 0, b <= 0) via
    post-step projection. Provides .project() to call after
    optimizer.step()."""
    def __init__(self, in_features, out_features, bias=True,
                 nonneg_bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.nonneg_bias = nonneg_bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def init_weights(self, weight_scale: float, bias_scale: float = 0.05,
                     mode: str = "uniform"):
        if mode == "zero":
            nn.init.zeros_(self.weight)
            if self.has_bias:
                nn.init.zeros_(self.bias)
        elif mode == "uniform":
            nn.init.uniform_(self.weight, 0.0, weight_scale)
            if self.has_bias:
                if self.nonneg_bias:
                    nn.init.uniform_(self.bias, 0.0, bias_scale)
                else:
                    nn.init.uniform_(self.bias, -bias_scale, 0.0)
        else:
            raise ValueError(f"Unknown init mode: {mode}")

    def project(self):
        with torch.no_grad():
            self.weight.data.clamp_(min=0.0)
            if self.has_bias and not self.nonneg_bias:
                self.bias.data.clamp_(max=0.0)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class MVNN(nn.Module):
    def __init__(self,
                 m,
                 hidden_units=20,
                 n_layers=2,
                 t_cutoff=1.0,
                 skip_connection=True,
                 init_target=1.0):
        super().__init__()
        self.m = m
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.t_cutoff = t_cutoff
        self.has_skip = skip_connection
        self._init_target = init_target

        dims = [m] + [hidden_units] * n_layers + [1]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_output = (i == len(dims) - 2)
            self.layers.append(MonotoneLinear(
                in_features=dims[i],
                out_features=dims[i + 1],
                bias=not is_output,
            ))

        if skip_connection:
            self.skip = MonotoneLinear(m, 1, bias=False)
        else:
            self.skip = None

        supply = torch.tensor(config.SUPPLY, dtype=torch.float32)
        self.register_buffer("norm", 1.0 / supply)

        self.reinit_for_scale(init_target)

    def reinit_for_scale(self, init_target: float):
        self._init_target = init_target

        for layer in self.layers[:-1]:
            fan_in = layer.in_features
            scale = max(self.t_cutoff / max(fan_in, 1), 0.01)
            layer.init_weights(weight_scale=scale,
                               bias_scale=scale * 0.5,
                               mode="uniform")

        out_scale = 4.0 * init_target / max(self.hidden_units * self.t_cutoff,
                                            1e-6)
        out_scale = max(out_scale, 0.01)
        self.layers[-1].init_weights(weight_scale=out_scale,
                                     bias_scale=0.0,
                                     mode="uniform")

        if self.skip is not None:
            self.skip.init_weights(weight_scale=0.0, bias_scale=0.0,
                                   mode="zero")

        self.project_all()

    def project_all(self):
        for layer in self.layers:
            layer.project()
        if self.skip is not None:
            self.skip.project()

    def _brelu(self, x):
        return torch.clamp(x, min=0.0, max=self.t_cutoff)

    def forward(self, x):
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        h = x * self.norm

        for layer in self.layers[:-1]:
            h = self._brelu(layer(h))

        out = self.layers[-1](h).squeeze(-1)

        if self.skip is not None:
            skip_out = self.skip(x * self.norm).squeeze(-1)
            out = out + skip_out

        return out.squeeze(0) if single else out

# Factory
def create_model(m):
    if config.MODEL_TYPE == "linear":
        return LinearValueModel(m)
    if config.MODEL_TYPE == "mvnn":
        return MVNN(
            m=m,
            hidden_units=getattr(config, "MVNN_HIDDEN_UNITS", 20),
            n_layers=getattr(config, "MVNN_LAYERS", 2),
            t_cutoff=getattr(config, "MVNN_T_CUTOFF", 1.0),
            skip_connection=True,
            init_target=1.0,
        )
    raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")


def _estimate_value_scale(bidder):
    if bidder.R_VQ:
        v_max = max(v for _, v in bidder.R_VQ)
        return max(v_max * 1.5, 1.0)

    if bidder.R_DQ:
        max_price_sum = max(sum(prices) for _, prices in bidder.R_DQ)
        multiplier = getattr(config, "DQ_VALUE_SCALE_MULTIPLIER", 3.0)
        floor = getattr(config, "DQ_VALUE_SCALE_FLOOR", 10.0)
        return max(max_price_sum * multiplier, floor)

    return 10.0


# Training

def train_value_model(model, bidder, epochs=None, lr=None):
    epochs = epochs or config.TRAIN_EPOCHS
    lr     = lr     or config.TRAIN_LR

    if not bidder.R_DQ and not bidder.R_VQ:
        return model

    value_scale = _estimate_value_scale(bidder)
    model._value_scale = value_scale

    if isinstance(model, MVNN):
        model.reinit_for_scale(init_target=0.5)

    weight_decay = getattr(config, "WEIGHT_DECAY", 1e-6)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, epochs // 3),
        gamma=0.5,
    )

    candidates = bidder._demand_candidates()
    cand_vectors = torch.stack(
        [bundle_to_vector(b, bidder.m) for b in candidates]
    )

    if bidder.R_VQ:
        X_vq = torch.stack(
            [bundle_to_vector(b, bidder.m) for b, _ in bidder.R_VQ]
        )
        y_vq = torch.tensor(
            [v / value_scale for _, v in bidder.R_VQ],
            dtype=torch.float32,
        )
    else:
        X_vq = None
        y_vq = None

    if bidder.R_DQ:
        scaled_prices = [
            torch.tensor([p / value_scale for p in prices],
                         dtype=torch.float32)
            for _, prices in bidder.R_DQ
        ]
        obs_vectors = torch.stack([
            bundle_to_vector(b, bidder.m) for b, _ in bidder.R_DQ
        ])
    else:
        scaled_prices = None
        obs_vectors = None

    cached_pred_idx = [None] * len(bidder.R_DQ)
    cache_refresh_every = max(1, getattr(config, "CACHED_DQ_FREQ", 5))

    w_dq = getattr(config, "DQ_LOSS_WEIGHT", 1.0)
    base_w_vq = getattr(config, "VQ_LOSS_WEIGHT", 3.0)
    if bidder.R_VQ and len(bidder.R_VQ) <= 2:
        vq_boost = getattr(config, "VQ_LOSS_WEIGHT_FEW_BOOST", 5.0)
        w_vq = base_w_vq * vq_boost
    else:
        w_vq = base_w_vq

    model.train()

    for epoch in range(epochs):
        if bidder.R_DQ and (epoch % cache_refresh_every == 0):
            with torch.no_grad():
                pred_vals = model(cand_vectors)
                for r, p_scaled in enumerate(scaled_prices):
                    utilities = pred_vals - (cand_vectors * p_scaled).sum(dim=1)
                    cached_pred_idx[r] = int(utilities.argmax().item())

        optimizer.zero_grad()
        loss_terms = []

        # DQ loss
        if bidder.R_DQ:
            pred_vals = model(cand_vectors)
            obs_vals = model(obs_vectors)

            dq_loss = torch.tensor(0.0)
            for r, p_scaled in enumerate(scaled_prices):
                utilities = pred_vals - (cand_vectors * p_scaled).sum(dim=1)
                obs_util = obs_vals[r] - (obs_vectors[r] * p_scaled).sum()

                idx = cached_pred_idx[r]
                pred_util = utilities[idx] if idx is not None else utilities.max()

                dq_loss = dq_loss + torch.relu(pred_util - obs_util)

            dq_loss = dq_loss / len(bidder.R_DQ)
            loss_terms.append(w_dq * dq_loss)

        if X_vq is not None:
            pred_vq = model(X_vq)
            vq_loss = nn.MSELoss()(pred_vq, y_vq)
            loss_terms.append(w_vq * vq_loss)

        if not loss_terms:
            continue

        total_loss = sum(loss_terms)

        if total_loss.item() > 1e-6:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if isinstance(model, MVNN):
                model.project_all()

    model.eval()
    return model