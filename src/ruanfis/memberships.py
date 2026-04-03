from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inverse_softplus(values: Tensor) -> Tensor:
    return values + torch.log(-torch.expm1(-values))


def _as_column(inputs: Tensor) -> Tensor:
    if inputs.ndim == 1:
        return inputs.unsqueeze(-1)
    if inputs.ndim == 2 and inputs.size(-1) == 1:
        return inputs
    raise ValueError(f"Expected a vector or a single-column tensor, got shape {tuple(inputs.shape)}.")


class BaseMembershipFunction(nn.Module):
    def __init__(self, term_names: Sequence[str] | None = None) -> None:
        super().__init__()
        self._term_names = tuple(term_names) if term_names is not None else ()

    @property
    def n_terms(self) -> int:
        raise NotImplementedError

    @property
    def term_names(self) -> tuple[str, ...]:
        if self._term_names:
            return self._term_names
        return tuple(f"term_{index}" for index in range(self.n_terms))


class GaussianMembership(BaseMembershipFunction):
    def __init__(
        self,
        centers: Sequence[float],
        spreads: Sequence[float],
        term_names: Sequence[str] | None = None,
        min_spread: float = 1e-3,
    ) -> None:
        if len(centers) != len(spreads):
            raise ValueError("Centers and spreads must have the same length.")
        super().__init__(term_names=term_names)
        if term_names is not None and len(term_names) != len(centers):
            raise ValueError("The number of term names must match the number of centers.")
        centers_tensor = torch.as_tensor(centers, dtype=torch.float32)
        spreads_tensor = torch.as_tensor(spreads, dtype=torch.float32).clamp_min(min_spread + 1e-6)
        self.centers = nn.Parameter(centers_tensor.clone())
        self.min_spread = float(min_spread)
        self.raw_spreads = nn.Parameter(_inverse_softplus(spreads_tensor - self.min_spread))

    @property
    def n_terms(self) -> int:
        return int(self.centers.numel())

    @property
    def spreads(self) -> Tensor:
        return F.softplus(self.raw_spreads) + self.min_spread

    def forward(self, inputs: Tensor) -> Tensor:
        x = _as_column(inputs)
        diff = x - self.centers
        variances = self.spreads.pow(2).clamp_min(self.min_spread**2)
        return torch.exp(-(diff.pow(2)) / (2.0 * variances))


class GeneralizedBellMembership(BaseMembershipFunction):
    def __init__(
        self,
        centers: Sequence[float],
        widths: Sequence[float],
        slopes: Sequence[float],
        term_names: Sequence[str] | None = None,
        min_width: float = 1e-3,
        min_slope: float = 1e-2,
    ) -> None:
        if not (len(centers) == len(widths) == len(slopes)):
            raise ValueError("Centers, widths and slopes must have the same length.")
        super().__init__(term_names=term_names)
        if term_names is not None and len(term_names) != len(centers):
            raise ValueError("The number of term names must match the number of centers.")
        self.centers = nn.Parameter(torch.as_tensor(centers, dtype=torch.float32).clone())
        self.min_width = float(min_width)
        self.min_slope = float(min_slope)
        widths_tensor = torch.as_tensor(widths, dtype=torch.float32).clamp_min(min_width + 1e-6)
        slopes_tensor = torch.as_tensor(slopes, dtype=torch.float32).clamp_min(min_slope + 1e-6)
        self.raw_widths = nn.Parameter(_inverse_softplus(widths_tensor - self.min_width))
        self.raw_slopes = nn.Parameter(_inverse_softplus(slopes_tensor - self.min_slope))

    @property
    def n_terms(self) -> int:
        return int(self.centers.numel())

    @property
    def widths(self) -> Tensor:
        return F.softplus(self.raw_widths) + self.min_width

    @property
    def slopes(self) -> Tensor:
        return F.softplus(self.raw_slopes) + self.min_slope

    def forward(self, inputs: Tensor) -> Tensor:
        x = _as_column(inputs)
        normalized = ((x - self.centers) / self.widths).abs().clamp_min(1e-12)
        powers = normalized.pow(2.0 * self.slopes)
        return 1.0 / (1.0 + powers)


class FuzzyVariable(nn.Module):
    def __init__(self, name: str, membership: BaseMembershipFunction) -> None:
        super().__init__()
        self.name = name
        self.membership = membership

    @property
    def n_terms(self) -> int:
        return self.membership.n_terms

    @property
    def term_names(self) -> tuple[str, ...]:
        return self.membership.term_names

    def forward(self, inputs: Tensor) -> Tensor:
        return self.membership(inputs)

