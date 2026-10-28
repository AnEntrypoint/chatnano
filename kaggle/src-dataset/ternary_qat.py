import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE_FLOOR = 1e-8


def _group_layout(in_features, group_size):
    group = in_features if group_size is None else int(group_size)
    if group <= 0:
        raise ValueError("group_size must be positive")
    return group, math.ceil(in_features / group)


def absmean_ternary(weight, group_size=None):
    """Quantize a 2-D weight to codes in {-1, 0, 1} with per-group absmean scales.

    Statistics are computed in float32 whatever the dtype of `weight`. Computing them in
    fp16 underflows the scale floor to zero, and a zero group then becomes 0/0 = NaN.
    A zero group gets scale SCALE_FLOOR and all-zero codes. Codes and scales are returned
    in `weight.dtype`; the padding of a partial last group is excluded from the mean.

    Returns codes of shape (out, in) and scales of shape (out, n_groups).
    """
    if weight.dim() != 2:
        raise ValueError("absmean_ternary expects a 2-D weight")
    out_features, in_features = weight.shape
    group, n_groups = _group_layout(in_features, group_size)
    padded = n_groups * group
    work = weight.detach().float()
    grouped = F.pad(work, (0, padded - in_features)).reshape(out_features, n_groups, group)
    valid = (torch.arange(padded, device=weight.device) < in_features).reshape(n_groups, group).float()
    counts = valid.sum(dim=-1)
    scales = ((grouped.abs() * valid).sum(dim=-1) / counts).clamp_min(SCALE_FLOOR)
    codes = torch.clamp(torch.round(grouped / scales.unsqueeze(-1)), -1.0, 1.0) * valid
    codes = codes.reshape(out_features, padded)[:, :in_features]
    return codes.to(weight.dtype), scales.to(weight.dtype)


def expand_scales(scales, in_features, group_size=None):
    group, _ = _group_layout(in_features, group_size)
    return scales.repeat_interleave(group, dim=1)[:, :in_features]


def dequantize(codes, scales, group_size=None):
    in_features = codes.shape[1]
    return codes.float() * expand_scales(scales.float(), in_features, group_size)


class TernaryLinear(nn.Module):
    """Linear layer whose forward pass blends latent FP weights with their ternary quantization.

    The blend factor is the non-persistent `alpha` buffer (0 = pure latent FP, 1 = pure ternary).
    It is a training schedule, not model state, so it is excluded from state_dict and a
    ternary checkpoint loads into an FP model and back.
    """

    def __init__(self, linear, group_size=None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=linear.weight.requires_grad)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)
        self.register_buffer("alpha", torch.zeros(()), persistent=False)

    def set_alpha(self, alpha):
        self.alpha.fill_(float(alpha))

    def quantized_weight(self):
        codes, scales = absmean_ternary(self.weight, self.group_size)
        return dequantize(codes, scales, self.group_size).to(self.weight.dtype)

    def effective_weight(self):
        if float(self.alpha) == 0.0:
            return self.weight
        latent = self.weight
        quantized = self.quantized_weight()
        alpha = self.alpha.to(latent.dtype)
        return latent + alpha * (quantized - latent).detach()

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


def ramp_alpha(step, total_steps, ramp_frac=0.25):
    """Blend factor for optimizer step `step` (zero-based index of the step about to run).

    alpha(step) = clamp(step / (ramp_frac * total_steps), 0, 1): it is 0.0 at step 0, rises
    linearly, and reaches 1.0 at step == ramp_frac * total_steps, then stays at 1.0.
    ramp_frac == 0 means no ramp (1.0 from step 0).
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if ramp_frac < 0:
        raise ValueError("ramp_frac must be non-negative")
    ramp_steps = ramp_frac * total_steps
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, step / ramp_steps))


def set_alpha(model, alpha):
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.set_alpha(alpha)


def clip_latents(model, clip_value=None):
    """Clamp every TernaryLinear latent weight to [-clip_value, clip_value].

    Call after each optimizer step to keep latents from drifting far outside the range that
    absmean quantization maps to codes. clip_value=None (the default) is a no-op, so clipping
    is off unless requested. Returns the number of latent weights clamped.
    """
    if clip_value is None:
        return 0
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                module.weight.clamp_(-clip_value, clip_value)
                count += 1
    return count


def replace_linears(model, predicate=None, group_size=None):
    """Replace nn.Linear submodules with TernaryLinear in place and return the replaced dotted names.

    Ordering: replacement creates new Parameter objects. Call this BEFORE building the optimizer
    (or capturing model.parameters()). An optimizer built earlier holds the old nn.Linear
    parameters and never updates the TernaryLinear latents. To keep an FP teacher, deep-copy the
    model before calling this function.

    The predicate is checked before the root check, so a root nn.Linear raises ValueError only
    when the predicate selects it (or no predicate is given).
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if predicate is not None and not predicate(name, module):
            continue
        if not name:
            raise ValueError("the root module itself cannot be replaced")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, TernaryLinear(module, group_size))
        replaced.append(name)
    return replaced


def _require_full_ternary(model):
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear) and float(module.alpha) != 1.0:
            raise ValueError(
                f"{name or 'root'} has alpha={float(module.alpha)}; call set_alpha(model, 1.0) before exporting"
            )


def export_packed(model):
    """Return {name: (codes int8, scales float32)} for every TernaryLinear.

    Requires alpha == 1 on every TernaryLinear so the artifact is the pure ternary model.
    Scales stay float32 so the exported artifact matches the trained forward exactly.
    Pair with export_metadata() to recover group_size and bias.
    """
    _require_full_ternary(model)
    packed = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            codes, scales = absmean_ternary(module.weight.detach().float(), module.group_size)
            packed[name] = (codes.to(torch.int8), scales)
    return packed


def export_metadata(model):
    """Return {name: {in_features, out_features, group_size, bias}} for every TernaryLinear.

    group_size is always resolved to an int (per-row quantization records in_features), and bias
    is a float32 tensor or None. Together with export_packed this makes the packed format
    self-describing.
    """
    meta = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            group, _ = _group_layout(module.in_features, module.group_size)
            bias = None if module.bias is None else module.bias.detach().float().clone()
            meta[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "group_size": group,
                "bias": bias,
            }
    return meta
