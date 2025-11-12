import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CustomGRU(nn.Module):
    """Minimal GRU drop-in that mirrors the numpy reference implementation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("CustomGRU currently supports only num_layers=1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        def make_param_list(rows, cols) -> nn.ParameterList:
            params = nn.ParameterList()
            for _ in range(self.num_directions):
                params.append(nn.Parameter(torch.empty(rows, cols)))
            return params

        self.Wxu = make_param_list(hidden_size, input_size)
        self.Wxr = make_param_list(hidden_size, input_size)
        self.Wxn = make_param_list(hidden_size, input_size)
        self.Whu = make_param_list(hidden_size, hidden_size)
        self.Whr = make_param_list(hidden_size, hidden_size)
        self.Whn = make_param_list(hidden_size, hidden_size)

        self.bu = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(self.num_directions)])
        self.br = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(self.num_directions)])
        self.bn_input = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(self.num_directions)])
        self.bn_hidden = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(self.num_directions)])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for params in (
            self.Wxu,
            self.Wxr,
            self.Wxn,
            self.Whu,
            self.Whr,
            self.Whn,
        ):
            for weight in params:
                nn.init.uniform_(weight, -std, std)
        for biases in (self.bu, self.br, self.bn_input, self.bn_hidden):
            for bias in biases:
                nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch, seq_len, _ = x.shape

        if h0 is None:
            h0 = x.new_zeros(self.num_directions, batch, self.hidden_size)
        elif h0.dim() == 2:
            h0 = h0.unsqueeze(0)
        if h0.shape != (self.num_directions, batch, self.hidden_size):
            raise ValueError(f"h0 must have shape ({self.num_directions}, {batch}, {self.hidden_size})")

        outputs: List[torch.Tensor] = []
        final_states: List[torch.Tensor] = []

        for direction in range(self.num_directions):
            seq = x
            indices = range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
            h = h0[direction]
            collected = []
            for idx in indices:
                x_t = seq[:, idx, :]
                u = torch.sigmoid(
                    F.linear(x_t, self.Wxu[direction]) + F.linear(h, self.Whu[direction]) + self.bu[direction]
                )
                r = torch.sigmoid(
                    F.linear(x_t, self.Wxr[direction]) + F.linear(h, self.Whr[direction]) + self.br[direction]
                )
                candidate_input = F.linear(x_t, self.Wxn[direction]) + self.bn_input[direction]
                candidate_hidden = F.linear(h, self.Whn[direction]) + self.bn_hidden[direction]
                c = torch.tanh(candidate_input + r * candidate_hidden)
                h = (1.0 - u) * c + u * h
                collected.append(h)
            dir_out = torch.stack(collected, dim=1)
            if direction == 1:
                dir_out = torch.flip(dir_out, dims=[1])
            outputs.append(dir_out)
            final_states.append(h)

        output = torch.cat(outputs, dim=2) if self.num_directions > 1 else outputs[0]
        if not self.batch_first:
            output = output.transpose(0, 1)
        hidden = torch.stack(final_states, dim=0)
        return output, hidden
