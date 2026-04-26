import torch.nn as nn
import pdb
import torch
import math



def split_at_indices(feats, indices):
    """
    split the latent tokens from the feats
    :param feats:
    :param latent_indices:
    :return:
    """
    if indices is None or len(indices) == 0:
        return feats, None
    index_mask = torch.zeros(feats.size(1), dtype=torch.bool)
    index_mask[indices] = True
    split_feats = feats[:, index_mask]
    feats = feats[:, ~index_mask]
    return feats, split_feats


def join_at_indices(feats, split_feats, indices):
    """
    Join the latent tokens to the feats
    :param feats: Input tensor
    :param split_feats: Tensor to join
    :param indices: Indices to join at
    :return: Combined tensor
    """
    if indices is None or len(indices) == 0:
        return feats
    # set dtype
    split_feats = split_feats.to(feats.dtype)
    indices = sorted(indices)
    output = torch.zeros((feats.size(0), feats.size(1) + split_feats.size(1), feats.size(-1)), dtype=feats.dtype,
                         device=feats.device)

    non_split_indices = torch.tensor([i for i in range(output.size(1)) if i not in indices])
    pdb.set_trace()

    output[:, non_split_indices] = feats
    output[:, indices] = split_feats

    return output


def create_n_copies(feats, n):
    """
    get n copies of the feats
    :param feats: input tensor (*shape)
    :param n: n x *shape copies
    :return: n copies of the feats along a new dimension
    """
    if feats is None:
        return None
    return torch.stack([feats.clone() for _ in range(n)], dim=0)

def ensure_batched(x):
    was_unbatched = x.ndim == 2
    if was_unbatched:
        x = torch.unsqueeze(x, 0)
    return x, was_unbatched


def ensure_unbatched(x):
    was_batched = x.ndim == 3
    if was_batched:
        x = torch.squeeze(x, 0)
    return x, was_batched

def set_model_dtype(module, dtype):
    """
    Sets the dtype of all parameters and buffers in an nn.Module to the specified dtype.

    Args:
        module (nn.Module): The model whose parameters and buffers will be converted.
        dtype (torch.dtype): The target dtype to which the parameters and buffers are converted.

    Returns:
        nn.Module: The module with all parameters and buffers converted to the specified dtype.
    """
    # Iterate over all parameters and buffers in the module
    for name, param in module.named_parameters():
        if param.data.dtype != dtype:
            try:
                param.data = param.data.to(dtype)
            except RuntimeError as e:
                print(f"Could not convert parameter {name} to {dtype}: {e}")

    for name, buf in module.named_buffers():
        if buf.dtype != dtype:
            try:
                buf.data = buf.data.to(dtype)
            except RuntimeError as e:
                print(f"Could not convert buffer {name} to {dtype}: {e}")
    # Recursively apply to child modules
    for child_name, child in module.named_children():
        set_model_dtype(child, dtype)

    return module

def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):
    layers = []
    if len(hid_dims) > 0:
        for hid_dim in hid_dims:
            layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
    layers.append(nn.Linear(in_dim, out_dim, bias=bias))
    if not end_with_fc:
        layers.append(act)
        layers.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*layers)
    # already initialized our way
    return mlp





def _kaiming_init(weight):
    nn.init.kaiming_uniform(weight, a=math.sqrt(5), nonlinearity='relu')


def _kaiming_init_bias(weight, bias):
    fan_in, _ = _calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        unbatched_x, was_batched = ensure_unbatched(x)
        out = self.module(unbatched_x)
        return out, unbatched_x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                _kaiming_init(m.weight)
                if m.bias is not None:
                    _kaiming_init_bias(m.weight, m.bias)

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def get_activation(self, x):
        # return dict containing activation for each layer
        unbatched_x, was_batched = ensure_unbatched(x)
        a = self.attention_a(unbatched_x)
        b = self.attention_b(unbatched_x)
        A = a.mul(b)
        c = self.attention_c(A)
        activations = {'activation_abmil_a': a, 'activation_abmil_b': b, 'activation_abmil_c': c}
        return activations

    def forward(self, x):
        unbatched_x, was_batched = ensure_unbatched(x)
        a = self.attention_a(unbatched_x)
        b = self.attention_b(unbatched_x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, None
