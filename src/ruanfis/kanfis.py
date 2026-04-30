from __future__ import annotations

import torch
from torch import Tensor, nn


def _featurewise_quantile_terms(samples: Tensor) -> tuple[Tensor, Tensor]:
    if samples.ndim != 2:
        raise ValueError("samples must have shape [samples, features].")
    values = samples.detach().to(dtype=torch.float32)
    centers = torch.quantile(values, torch.tensor([0.2, 0.5, 0.8], device=values.device), dim=0).transpose(0, 1)
    low = centers[:, 1] - centers[:, 0]
    high = centers[:, 2] - centers[:, 1]
    collapsed = torch.minimum(low, high) < 0.05
    if collapsed.any():
        default_centers = torch.tensor([0.2, 0.5, 0.8], device=values.device, dtype=values.dtype)
        centers[collapsed] = default_centers
    fallback = values.std(dim=0).clamp_min(0.08)
    widths = torch.maximum(torch.minimum(low, high), fallback * 0.35).clamp_min(0.04)
    widths[collapsed] = 0.18
    return centers.cpu(), widths.cpu()


def _binary_feature_mask(samples: Tensor) -> Tensor:
    if samples.ndim != 2:
        raise ValueError("samples must have shape [samples, features].")
    values = samples.detach().to(dtype=torch.float32)
    near_zero_or_one = ((values - 0.0).abs() < 1e-5) | ((values - 1.0).abs() < 1e-5)
    has_both_states = (values.amin(dim=0) < 0.05) & (values.amax(dim=0) > 0.95)
    return (near_zero_or_one.float().mean(dim=0) > 0.995) & has_both_states


class KANFISModel(nn.Module):
    """Additive fuzzy superposition model inspired by Kolmogorov-Arnold networks."""

    def __init__(
        self,
        input_dim: int,
        *,
        superposition_terms: int | None = None,
        term_centers: tuple[float, ...] = (0.2, 0.5, 0.8),
        term_width: float = 0.18,
        pair_indices: tuple[tuple[int, int], ...] = (),
        active_feature_indices: tuple[int, ...] | None = None,
        raw_skip_enabled: bool = True,
        raw_skip_gate_init_logit: float = 1.5,
        output_dim: int = 1,
        gate_init_logit: float = 2.0,
        train_rule_gates: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if term_width <= 0.0:
            raise ValueError("term_width must be positive.")
        if not term_centers:
            raise ValueError("At least one fuzzy term center is required.")

        self.input_dim = int(input_dim)
        self.superposition_terms = int(superposition_terms or min(2 * input_dim + 1, 16))
        self.output_dim = int(output_dim)
        self.raw_skip_enabled = bool(raw_skip_enabled)
        self.pair_indices = tuple((int(left), int(right)) for left, right in pair_indices)
        if any(left < 0 or right < 0 or left >= self.input_dim or right >= self.input_dim or left == right for left, right in self.pair_indices):
            raise ValueError("pair_indices must contain distinct valid feature index pairs.")
        if active_feature_indices is None:
            self.active_feature_indices = tuple(range(self.input_dim))
        else:
            self.active_feature_indices = tuple(dict.fromkeys(int(index) for index in active_feature_indices))
        if any(index < 0 or index >= self.input_dim for index in self.active_feature_indices):
            raise ValueError("active_feature_indices must contain valid feature indices.")
        self.term_names = ("LOW", "MEDIUM", "HIGH") if len(term_centers) == 3 else tuple(
            f"term_{index}" for index in range(len(term_centers))
        )
        base_centers = torch.tensor(term_centers, dtype=torch.float32).view(1, -1).repeat(self.input_dim, 1)
        self.register_buffer("term_centers", base_centers, persistent=True)
        self.register_buffer("term_width", torch.full((self.input_dim,), float(term_width)), persistent=True)
        self.register_buffer("binary_input_mask", torch.zeros(self.input_dim, dtype=torch.bool), persistent=True)

        self.inner_weights = nn.Parameter(
            torch.empty(self.superposition_terms, self.input_dim, len(term_centers), dtype=torch.float32)
        )
        self.rule_logits = nn.Parameter(
            torch.full(
                (self.superposition_terms, self.input_dim, len(term_centers)),
                fill_value=float(gate_init_logit),
                dtype=torch.float32,
            ),
            requires_grad=bool(train_rule_gates),
        )
        if self.pair_indices:
            self.pair_weights = nn.Parameter(
                torch.empty(
                    self.superposition_terms,
                    len(self.pair_indices),
                    len(term_centers),
                    len(term_centers),
                    dtype=torch.float32,
                )
            )
            self.pair_rule_logits = nn.Parameter(
                torch.full(
                    (self.superposition_terms, len(self.pair_indices), len(term_centers), len(term_centers)),
                    fill_value=float(gate_init_logit),
                    dtype=torch.float32,
                ),
                requires_grad=bool(train_rule_gates),
            )
            nn.init.xavier_uniform_(self.pair_weights)
            self.register_buffer(
                "pair_left_indices",
                torch.tensor([left for left, _ in self.pair_indices], dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "pair_right_indices",
                torch.tensor([right for _, right in self.pair_indices], dtype=torch.long),
                persistent=False,
            )
        else:
            self.pair_weights = None
            self.pair_rule_logits = None
            self.register_buffer("pair_left_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("pair_right_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer(
            "rule_active_mask",
            self._build_rule_active_mask(term_count=len(term_centers)),
            persistent=True,
        )
        self.inner_bias = nn.Parameter(torch.zeros(self.superposition_terms, dtype=torch.float32))
        self.outer_weights = nn.Parameter(torch.empty(self.superposition_terms, self.output_dim, dtype=torch.float32))
        self.outer_bias = nn.Parameter(torch.zeros(self.output_dim, dtype=torch.float32))
        if self.raw_skip_enabled:
            self.raw_skip_weights = nn.Parameter(torch.empty(self.input_dim, self.output_dim, dtype=torch.float32))
            self.raw_skip_logits = nn.Parameter(
                torch.full((self.input_dim,), float(raw_skip_gate_init_logit), dtype=torch.float32)
            )
            nn.init.xavier_uniform_(self.raw_skip_weights)
        else:
            self.raw_skip_weights = None
            self.raw_skip_logits = None
        nn.init.xavier_uniform_(self.inner_weights)
        nn.init.xavier_uniform_(self.outer_weights)
        self.register_buffer(
            "rule_probability_anchor",
            self.rule_probabilities.detach().clone(),
            persistent=False,
        )

    def _build_rule_active_mask(self, *, term_count: int) -> Tensor:
        unary_mask = torch.zeros(self.superposition_terms, self.input_dim, term_count, dtype=torch.float32)
        if self.active_feature_indices:
            unary_mask[:, list(self.active_feature_indices), :] = 1.0
        if not self.pair_indices:
            return unary_mask.reshape(-1)

        pair_mask = torch.zeros(
            self.superposition_terms,
            len(self.pair_indices),
            term_count,
            term_count,
            dtype=torch.float32,
        )
        active_features = set(self.active_feature_indices)
        for pair_position, (left, right) in enumerate(self.pair_indices):
            if left in active_features and right in active_features:
                pair_mask[:, pair_position, :, :] = 1.0
        return torch.cat((unary_mask.reshape(-1), pair_mask.reshape(-1)), dim=0)

    @property
    def n_rules(self) -> int:
        pair_rules = 0 if self.pair_rule_logits is None else int(self.pair_rule_logits.numel())
        return int(self.rule_logits.numel()) + pair_rules

    @property
    def raw_rule_probabilities(self) -> Tensor:
        unary = torch.sigmoid(self.rule_logits).reshape(-1)
        if self.pair_rule_logits is None:
            return unary
        return torch.cat((unary, torch.sigmoid(self.pair_rule_logits).reshape(-1)), dim=0)

    @property
    def rule_probabilities(self) -> Tensor:
        return self.raw_rule_probabilities * self.rule_active_mask.to(
            device=self.rule_logits.device,
            dtype=self.rule_logits.dtype,
        )

    def describe_rule(self, rule_index: int) -> str:
        if rule_index < 0 or rule_index >= self.n_rules:
            raise IndexError(f"Rule index {rule_index} is out of range for KANFIS.")
        term_count = int(self.term_centers.size(1))
        unary_count = int(self.rule_logits.numel())
        if rule_index >= unary_count:
            pair_index = int(rule_index) - unary_count
            q, rem = divmod(pair_index, len(self.pair_indices) * term_count * term_count)
            pair_position, term_pair = divmod(rem, term_count * term_count)
            left_term, right_term = divmod(term_pair, term_count)
            left, right = self.pair_indices[pair_position]
            return (
                f"q{q}: x{left} IS {self.term_names[left_term]} "
                f"AND x{right} IS {self.term_names[right_term]}"
            )
        q, rem = divmod(int(rule_index), self.input_dim * term_count)
        feature_index, term_index = divmod(rem, term_count)
        return f"q{q}: x{feature_index} IS {self.term_names[term_index]}"

    def iter_rule_layer_entries(self) -> tuple[tuple[str, KANFISModel], ...]:
        return (("kanfis_basis", self),)

    def refresh_rule_probability_anchor(self) -> None:
        with torch.no_grad():
            self.rule_probability_anchor.copy_(self.rule_probabilities.detach())

    def fit_membership_terms(self, samples: Tensor) -> None:
        centers, widths = _featurewise_quantile_terms(samples)
        if centers.shape != self.term_centers.shape or widths.shape != self.term_width.shape:
            raise ValueError("Computed membership terms do not match KANFIS shape.")
        with torch.no_grad():
            self.term_centers.copy_(centers.to(device=self.term_centers.device, dtype=self.term_centers.dtype))
            self.term_width.copy_(widths.to(device=self.term_width.device, dtype=self.term_width.dtype))

    def rule_importances(self) -> Tensor:
        outer_scale = self.outer_weights.detach().abs().mean(dim=1).view(self.superposition_terms, 1, 1)
        unary = (self.inner_weights.detach().abs() * torch.sigmoid(self.rule_logits.detach()) * outer_scale).reshape(-1)
        if self.pair_weights is None or self.pair_rule_logits is None:
            values = unary
        else:
            pair_outer_scale = self.outer_weights.detach().abs().mean(dim=1).view(self.superposition_terms, 1, 1, 1)
            pair = (
                self.pair_weights.detach().abs()
                * torch.sigmoid(self.pair_rule_logits.detach())
                * pair_outer_scale
            ).reshape(-1)
            values = torch.cat((unary, pair), dim=0)
        return values * self.rule_active_mask.to(device=values.device, dtype=values.dtype)

    def data_aware_rule_importances(self, inputs: Tensor, *, batch_size: int = 8192) -> Tensor:
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(f"Expected inputs with shape [batch, {self.input_dim}], got {tuple(inputs.shape)}.")
        device = self.inner_weights.device
        dtype = self.inner_weights.dtype
        centers = self.term_centers.to(device=device, dtype=dtype)
        width = self.term_width.to(device=device, dtype=dtype).clamp_min(1e-6)
        outer_scale = self.outer_weights.detach().abs().mean(dim=1).view(self.superposition_terms, 1, 1)
        unary_scale = self.inner_weights.detach().abs() * torch.sigmoid(self.rule_logits.detach()) * outer_scale
        unary_total = torch.zeros_like(unary_scale)
        pair_total = None
        if self.pair_weights is not None and self.pair_rule_logits is not None:
            pair_outer_scale = self.outer_weights.detach().abs().mean(dim=1).view(self.superposition_terms, 1, 1, 1)
            pair_scale = (
                self.pair_weights.detach().abs()
                * torch.sigmoid(self.pair_rule_logits.detach())
                * pair_outer_scale
            )
            pair_total = torch.zeros_like(pair_scale)
        sample_count = 0
        for start in range(0, inputs.size(0), max(1, int(batch_size))):
            batch = inputs[start : start + max(1, int(batch_size))].to(device=device, dtype=dtype)
            memberships = torch.exp(-0.5 * ((batch.unsqueeze(-1) - centers.unsqueeze(0)) / width.view(1, -1, 1)) ** 2)
            unary_total = unary_total + memberships.mean(dim=0).unsqueeze(0) * unary_scale * batch.size(0)
            if pair_total is not None:
                left_memberships = memberships.index_select(dim=1, index=self.pair_left_indices.to(device))
                right_memberships = memberships.index_select(dim=1, index=self.pair_right_indices.to(device))
                pair_memberships = left_memberships.unsqueeze(-1) * right_memberships.unsqueeze(-2)
                pair_total = pair_total + pair_memberships.mean(dim=0).unsqueeze(0) * pair_scale * batch.size(0)
            sample_count += int(batch.size(0))
        divisor = float(max(1, sample_count))
        unary = (unary_total / divisor).reshape(-1)
        if pair_total is None:
            values = unary
        else:
            values = torch.cat((unary, (pair_total / divisor).reshape(-1)), dim=0)
        return values * self.rule_active_mask.to(device=values.device, dtype=values.dtype)

    def prune_to_top_k_rules(
        self,
        max_active_rules: int,
        *,
        inputs: Tensor | None = None,
        batch_size: int = 8192,
    ) -> int:
        if max_active_rules <= 0:
            raise ValueError("max_active_rules must be positive.")
        with torch.no_grad():
            current_mask = self.rule_active_mask.to(device=self.rule_logits.device)
            active_count = int(current_mask.sum().item())
            if max_active_rules >= active_count:
                return active_count
            importances = (
                self.data_aware_rule_importances(inputs, batch_size=batch_size)
                if inputs is not None
                else self.rule_importances()
            )
            top_indices = torch.topk(importances, k=int(max_active_rules), largest=True).indices
            new_mask = torch.zeros_like(current_mask)
            new_mask[top_indices] = 1.0
            self.rule_active_mask.copy_(new_mask.to(device=self.rule_active_mask.device, dtype=self.rule_active_mask.dtype))
            self.refresh_rule_probability_anchor()
        return int(max_active_rules)

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(f"Expected inputs with shape [batch, {self.input_dim}], got {tuple(inputs.shape)}.")
        del top_k_rules
        centers = self.term_centers.to(device=inputs.device, dtype=inputs.dtype)
        width = self.term_width.to(device=inputs.device, dtype=inputs.dtype).clamp_min(1e-6)
        memberships = torch.exp(-0.5 * ((inputs.unsqueeze(-1) - centers.unsqueeze(0)) / width.view(1, -1, 1)) ** 2)
        unary_mask = self.rule_active_mask[: self.rule_logits.numel()].view_as(self.rule_logits)
        gated_weights = self.inner_weights * torch.sigmoid(self.rule_logits) * unary_mask
        inner = torch.einsum("bdt,qdt->bq", memberships, gated_weights) + self.inner_bias
        if self.pair_weights is not None and self.pair_rule_logits is not None:
            left_memberships = memberships.index_select(dim=1, index=self.pair_left_indices.to(inputs.device))
            right_memberships = memberships.index_select(dim=1, index=self.pair_right_indices.to(inputs.device))
            pair_memberships = left_memberships.unsqueeze(-1) * right_memberships.unsqueeze(-2)
            pair_offset = self.rule_logits.numel()
            pair_mask = self.rule_active_mask[pair_offset:].view_as(self.pair_rule_logits)
            pair_weights = self.pair_weights * torch.sigmoid(self.pair_rule_logits) * pair_mask
            inner = inner + torch.einsum("bltu,qltu->bq", pair_memberships, pair_weights)
        hidden = torch.sigmoid(inner)
        output = hidden @ self.outer_weights + self.outer_bias
        if self.raw_skip_weights is not None and self.raw_skip_logits is not None:
            skip_weights = self.raw_skip_weights * torch.sigmoid(self.raw_skip_logits).unsqueeze(-1)
            output = output + inputs @ skip_weights
        return output


class AdditiveFuzzyConceptLayer(nn.Module):
    """Interpretable additive fuzzy concept layer: concepts are sums of unary predicates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        name: str,
        input_prefix: str,
        term_centers: tuple[float, ...] = (0.2, 0.5, 0.8),
        term_width: float = 0.18,
        gate_init_logit: float = 2.0,
        train_rule_gates: bool = False,
        active_input_indices: tuple[int, ...] | None = None,
        input_names: tuple[str, ...] | None = None,
        input_indices_by_output: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.name = str(name)
        self.input_prefix = str(input_prefix)
        self.input_names = tuple(input_names) if input_names is not None else None
        if input_indices_by_output is not None:
            if len(input_indices_by_output) != self.output_dim:
                raise ValueError("input_indices_by_output must contain one route per output concept.")
            route_widths = {len(route) for route in input_indices_by_output}
            if len(route_widths) != 1 or not route_widths or next(iter(route_widths)) <= 0:
                raise ValueError("All routed concepts must have the same positive fan-in.")
            route_tensor = torch.tensor(input_indices_by_output, dtype=torch.long)
            if route_tensor.min().item() < 0 or route_tensor.max().item() >= self.input_dim:
                raise ValueError("Routed input indices must be valid.")
            self.register_buffer("input_indices_by_output", route_tensor, persistent=False)
            weight_input_dim = int(route_tensor.size(1))
        else:
            self.register_buffer("input_indices_by_output", torch.empty(0, 0, dtype=torch.long), persistent=False)
            weight_input_dim = self.input_dim
        self.term_names = ("LOW", "MEDIUM", "HIGH") if len(term_centers) == 3 else tuple(
            f"term_{index}" for index in range(len(term_centers))
        )
        base_centers = torch.tensor(term_centers, dtype=torch.float32).view(1, -1).repeat(self.input_dim, 1)
        self.register_buffer("term_centers", base_centers, persistent=True)
        self.register_buffer("term_width", torch.full((self.input_dim,), float(term_width)), persistent=True)
        self.register_buffer("binary_input_mask", torch.zeros(self.input_dim, dtype=torch.bool), persistent=True)
        self.weights = nn.Parameter(torch.empty(self.output_dim, weight_input_dim, len(term_centers)))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.rule_logits = nn.Parameter(
            torch.full((self.output_dim, weight_input_dim, len(term_centers)), float(gate_init_logit)),
            requires_grad=bool(train_rule_gates),
        )
        if input_indices_by_output is not None or active_input_indices is None:
            active_mask = torch.ones_like(self.rule_logits)
        else:
            active_indices = tuple(dict.fromkeys(int(index) for index in active_input_indices))
            if any(index < 0 or index >= self.input_dim for index in active_indices):
                raise ValueError("active_input_indices must contain valid input indices.")
            active_mask = torch.zeros_like(self.rule_logits)
            if active_indices:
                active_mask[:, list(active_indices), :] = 1.0
        self.register_buffer("rule_active_mask", active_mask.reshape(-1), persistent=True)
        self.register_buffer("rule_probability_anchor", self.rule_probabilities.detach().clone(), persistent=False)
        nn.init.xavier_uniform_(self.weights)

    @property
    def n_rules(self) -> int:
        return int(self.rule_logits.numel())

    @property
    def rule_probabilities(self) -> Tensor:
        return torch.sigmoid(self.rule_logits).reshape(-1) * self.rule_active_mask.to(
            device=self.rule_logits.device,
            dtype=self.rule_logits.dtype,
        )

    def describe_rule(self, rule_index: int) -> str:
        if rule_index < 0 or rule_index >= self.n_rules:
            raise IndexError(f"Rule index {rule_index} is out of range for {self.name}.")
        term_count = int(self.term_centers.size(1))
        fan_in = int(self.rule_logits.size(1))
        concept_index, rem = divmod(int(rule_index), fan_in * term_count)
        local_input_index, term_index = divmod(rem, term_count)
        if self.input_indices_by_output.numel() > 0:
            input_index = int(self.input_indices_by_output[concept_index, local_input_index].item())
        else:
            input_index = local_input_index
        input_name = self.input_names[input_index] if self.input_names is not None else f"{self.input_prefix}{input_index}"
        term_name = self.term_names[term_index]
        if bool(self.binary_input_mask[input_index].item()) and len(self.term_names) == 3:
            term_name = "absent" if term_index == 0 else "present" if term_index == 2 else "binary-middle"
        return f"c{concept_index}: {input_name} IS {term_name}"

    def rule_scores(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(f"Expected inputs with shape [batch, {self.input_dim}], got {tuple(inputs.shape)}.")
        memberships = self._memberships(inputs)
        if self.input_indices_by_output.numel() > 0:
            routes = self.input_indices_by_output.to(inputs.device)
            memberships = memberships.index_select(dim=1, index=routes.reshape(-1)).view(
                inputs.size(0), self.output_dim, routes.size(1), memberships.size(-1)
            )
        else:
            memberships = memberships.unsqueeze(1).expand(-1, self.output_dim, -1, -1)
        gate = torch.sigmoid(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        mask = self.rule_active_mask.view_as(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        weight = self.weights.detach().abs().to(device=inputs.device, dtype=inputs.dtype)
        return (memberships * gate.unsqueeze(0) * mask.unsqueeze(0) * weight.unsqueeze(0)).reshape(inputs.size(0), -1)

    def _memberships(self, inputs: Tensor) -> Tensor:
        centers = self.term_centers.to(device=inputs.device, dtype=inputs.dtype)
        width = self.term_width.to(device=inputs.device, dtype=inputs.dtype).clamp_min(1e-6)
        return torch.exp(-0.5 * ((inputs.unsqueeze(-1) - centers.unsqueeze(0)) / width.view(1, -1, 1)) ** 2)

    def fit_membership_terms(self, samples: Tensor) -> None:
        centers, widths = _featurewise_quantile_terms(samples)
        if centers.shape != self.term_centers.shape or widths.shape != self.term_width.shape:
            raise ValueError("Computed membership terms do not match layer shape.")
        with torch.no_grad():
            self.term_centers.copy_(centers.to(device=self.term_centers.device, dtype=self.term_centers.dtype))
            self.term_width.copy_(widths.to(device=self.term_width.device, dtype=self.term_width.dtype))
            binary_mask = _binary_feature_mask(samples).to(device=self.binary_input_mask.device)
            self.binary_input_mask.copy_(binary_mask)
            if self.term_centers.size(1) == 3 and binary_mask.any():
                mask = self.rule_active_mask.view_as(self.rule_logits)
                if self.input_indices_by_output.numel() > 0:
                    routes = self.input_indices_by_output.to(binary_mask.device)
                    routed_binary = binary_mask.index_select(0, routes.reshape(-1)).view_as(routes)
                    mask[:, :, 1] = torch.where(
                        routed_binary.to(device=mask.device),
                        torch.zeros_like(mask[:, :, 1]),
                        mask[:, :, 1],
                    )
                else:
                    mask[:, binary_mask.to(device=mask.device), 1] = 0.0
                self.rule_active_mask.copy_(mask.reshape(-1))

    def refresh_rule_probability_anchor(self) -> None:
        with torch.no_grad():
            self.rule_probability_anchor.copy_(self.rule_probabilities.detach())

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(f"Expected inputs with shape [batch, {self.input_dim}], got {tuple(inputs.shape)}.")
        mask = self.rule_active_mask.view_as(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        weights = self.weights * torch.sigmoid(self.rule_logits).to(inputs.dtype) * mask
        memberships = self._memberships(inputs)
        if self.input_indices_by_output.numel() > 0:
            routes = self.input_indices_by_output.to(inputs.device)
            memberships = memberships.index_select(dim=1, index=routes.reshape(-1)).view(
                inputs.size(0), self.output_dim, routes.size(1), memberships.size(-1)
            )
            inner = torch.einsum("bkft,kft->bk", memberships, weights) + self.bias
        else:
            inner = torch.einsum("bdt,kdt->bk", memberships, weights) + self.bias
        return torch.sigmoid(inner)

    def data_aware_rule_importances(self, inputs: Tensor, outer_scale: Tensor, *, batch_size: int = 8192) -> Tensor:
        device = self.weights.device
        dtype = self.weights.dtype
        scale = outer_scale.detach().abs().to(device=device, dtype=dtype).view(self.output_dim, 1, 1)
        weighted = self.weights.detach().abs() * torch.sigmoid(self.rule_logits.detach()) * scale
        total = torch.zeros_like(weighted)
        sample_count = 0
        for start in range(0, inputs.size(0), max(1, int(batch_size))):
            batch = inputs[start : start + max(1, int(batch_size))].to(device=device, dtype=dtype)
            memberships = self._memberships(batch)
            if self.input_indices_by_output.numel() > 0:
                routes = self.input_indices_by_output.to(device)
                memberships = memberships.index_select(dim=1, index=routes.reshape(-1)).view(
                    batch.size(0), self.output_dim, routes.size(1), memberships.size(-1)
                )
                mean_memberships = memberships.mean(dim=0)
            else:
                mean_memberships = memberships.mean(dim=0).unsqueeze(0)
            total = total + mean_memberships * weighted * batch.size(0)
            sample_count += int(batch.size(0))
        values = (total / float(max(1, sample_count))).reshape(-1)
        return values * self.rule_active_mask.to(device=values.device, dtype=values.dtype)

    def prune_to_top_k_rules(self, max_active_rules: int, importances: Tensor) -> int:
        if max_active_rules <= 0:
            raise ValueError("max_active_rules must be positive.")
        with torch.no_grad():
            active_count = int(self.rule_active_mask.sum().item())
            if max_active_rules >= active_count:
                return active_count
            top_indices = torch.topk(importances, k=int(max_active_rules), largest=True).indices
            new_mask = torch.zeros_like(self.rule_active_mask)
            new_mask[top_indices.to(new_mask.device)] = 1.0
            self.rule_active_mask.copy_(new_mask)
        return int(max_active_rules)


class FuzzyPairInteractionLayer(nn.Module):
    """Transparent pairwise fuzzy tokens: each token is a small 3x3 rule table."""

    def __init__(
        self,
        input_dim: int,
        pair_indices: tuple[tuple[int, int], ...],
        *,
        name: str = "ka_pair_interactions",
        term_centers: tuple[float, ...] = (0.2, 0.5, 0.8),
        term_width: float = 0.18,
        gate_init_logit: float = 2.0,
        train_rule_gates: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.pair_indices = tuple((int(left), int(right)) for left, right in pair_indices)
        if any(left < 0 or right < 0 or left >= self.input_dim or right >= self.input_dim or left == right for left, right in self.pair_indices):
            raise ValueError("pair_indices must contain distinct valid feature index pairs.")
        self.output_dim = len(self.pair_indices)
        self.name = str(name)
        self.feature_names = tuple(f"x{index}" for index in range(self.input_dim))
        self.term_names = ("LOW", "MEDIUM", "HIGH") if len(term_centers) == 3 else tuple(
            f"term_{index}" for index in range(len(term_centers))
        )
        base_centers = torch.tensor(term_centers, dtype=torch.float32).view(1, -1).repeat(self.input_dim, 1)
        self.register_buffer("term_centers", base_centers, persistent=True)
        self.register_buffer("term_width", torch.full((self.input_dim,), float(term_width)), persistent=True)
        self.register_buffer("binary_input_mask", torch.zeros(self.input_dim, dtype=torch.bool), persistent=True)
        self.weights = nn.Parameter(torch.empty(self.output_dim, len(term_centers), len(term_centers)))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.rule_logits = nn.Parameter(
            torch.full((self.output_dim, len(term_centers), len(term_centers)), float(gate_init_logit)),
            requires_grad=bool(train_rule_gates),
        )
        self.register_buffer("rule_active_mask", torch.ones_like(self.rule_logits).reshape(-1), persistent=True)
        self.register_buffer("rule_probability_anchor", self.rule_probabilities.detach().clone(), persistent=False)
        self.register_buffer("left_indices", torch.tensor([left for left, _ in self.pair_indices], dtype=torch.long), persistent=False)
        self.register_buffer("right_indices", torch.tensor([right for _, right in self.pair_indices], dtype=torch.long), persistent=False)
        nn.init.xavier_uniform_(self.weights)

    @property
    def n_rules(self) -> int:
        return int(self.rule_logits.numel())

    @property
    def rule_probabilities(self) -> Tensor:
        return torch.sigmoid(self.rule_logits).reshape(-1) * self.rule_active_mask.to(
            device=self.rule_logits.device,
            dtype=self.rule_logits.dtype,
        )

    def set_feature_names(self, feature_names: tuple[str, ...] | list[str]) -> None:
        if len(feature_names) < self.input_dim:
            raise ValueError("feature_names must cover all input features.")
        self.feature_names = tuple(str(name) for name in feature_names[: self.input_dim])

    def output_names(self) -> tuple[str, ...]:
        return tuple(
            f"pair({self.feature_names[left]}, {self.feature_names[right]})"
            for left, right in self.pair_indices
        )

    def describe_rule(self, rule_index: int) -> str:
        if rule_index < 0 or rule_index >= self.n_rules:
            raise IndexError(f"Rule index {rule_index} is out of range for {self.name}.")
        term_count = int(self.term_centers.size(1))
        pair_index, term_pair = divmod(int(rule_index), term_count * term_count)
        left_term, right_term = divmod(term_pair, term_count)
        left, right = self.pair_indices[pair_index]
        left_name = self.term_names[left_term]
        right_name = self.term_names[right_term]
        if bool(self.binary_input_mask[left].item()) and len(self.term_names) == 3:
            left_name = "absent" if left_term == 0 else "present" if left_term == 2 else "binary-middle"
        if bool(self.binary_input_mask[right].item()) and len(self.term_names) == 3:
            right_name = "absent" if right_term == 0 else "present" if right_term == 2 else "binary-middle"
        return (
            f"p{pair_index}: {self.feature_names[left]} IS {left_name} "
            f"AND {self.feature_names[right]} IS {right_name}"
        )

    def fit_membership_terms(self, samples: Tensor) -> None:
        centers, widths = _featurewise_quantile_terms(samples)
        if centers.shape != self.term_centers.shape or widths.shape != self.term_width.shape:
            raise ValueError("Computed membership terms do not match pair layer shape.")
        with torch.no_grad():
            self.term_centers.copy_(centers.to(device=self.term_centers.device, dtype=self.term_centers.dtype))
            self.term_width.copy_(widths.to(device=self.term_width.device, dtype=self.term_width.dtype))
            binary_mask = _binary_feature_mask(samples).to(device=self.binary_input_mask.device)
            self.binary_input_mask.copy_(binary_mask)
            if self.term_centers.size(1) == 3 and binary_mask.any():
                mask = self.rule_active_mask.view_as(self.rule_logits)
                for pair_index, (left, right) in enumerate(self.pair_indices):
                    if bool(binary_mask[left].item()):
                        mask[pair_index, 1, :] = 0.0
                    if bool(binary_mask[right].item()):
                        mask[pair_index, :, 1] = 0.0
                self.rule_active_mask.copy_(mask.reshape(-1))

    def refresh_rule_probability_anchor(self) -> None:
        with torch.no_grad():
            self.rule_probability_anchor.copy_(self.rule_probabilities.detach())

    def _memberships(self, inputs: Tensor) -> Tensor:
        centers = self.term_centers.to(device=inputs.device, dtype=inputs.dtype)
        width = self.term_width.to(device=inputs.device, dtype=inputs.dtype).clamp_min(1e-6)
        return torch.exp(-0.5 * ((inputs.unsqueeze(-1) - centers.unsqueeze(0)) / width.view(1, -1, 1)) ** 2)

    def _pair_memberships(self, inputs: Tensor) -> Tensor:
        memberships = self._memberships(inputs)
        left = memberships.index_select(dim=1, index=self.left_indices.to(inputs.device))
        right = memberships.index_select(dim=1, index=self.right_indices.to(inputs.device))
        return left.unsqueeze(-1) * right.unsqueeze(-2)

    def rule_scores(self, inputs: Tensor) -> Tensor:
        pair_memberships = self._pair_memberships(inputs)
        gate = torch.sigmoid(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        mask = self.rule_active_mask.view_as(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        weight = self.weights.detach().abs().to(device=inputs.device, dtype=inputs.dtype)
        return (pair_memberships * gate.unsqueeze(0) * mask.unsqueeze(0) * weight.unsqueeze(0)).reshape(inputs.size(0), -1)

    def data_aware_rule_importances(self, inputs: Tensor, outer_scale: Tensor, *, batch_size: int = 8192) -> Tensor:
        device = self.weights.device
        dtype = self.weights.dtype
        scale = outer_scale.detach().abs().to(device=device, dtype=dtype).view(self.output_dim, 1, 1)
        weighted = self.weights.detach().abs() * torch.sigmoid(self.rule_logits.detach()) * scale
        total = torch.zeros_like(weighted)
        sample_count = 0
        for start in range(0, inputs.size(0), max(1, int(batch_size))):
            batch = inputs[start : start + max(1, int(batch_size))].to(device=device, dtype=dtype)
            total = total + self._pair_memberships(batch).mean(dim=0) * weighted * batch.size(0)
            sample_count += int(batch.size(0))
        values = (total / float(max(1, sample_count))).reshape(-1)
        return values * self.rule_active_mask.to(device=values.device, dtype=values.dtype)

    def prune_to_top_k_rules(self, max_active_rules: int, importances: Tensor) -> int:
        if max_active_rules <= 0:
            raise ValueError("max_active_rules must be positive.")
        with torch.no_grad():
            active_count = int(self.rule_active_mask.sum().item())
            if max_active_rules >= active_count:
                return active_count
            top_indices = torch.topk(importances, k=int(max_active_rules), largest=True).indices
            new_mask = torch.zeros_like(self.rule_active_mask)
            new_mask[top_indices.to(new_mask.device)] = 1.0
            self.rule_active_mask.copy_(new_mask)
        return int(max_active_rules)

    def forward(self, inputs: Tensor) -> Tensor:
        mask = self.rule_active_mask.view_as(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        weights = self.weights * torch.sigmoid(self.rule_logits).to(inputs.dtype) * mask
        inner = torch.einsum("bptu,ptu->bp", self._pair_memberships(inputs), weights) + self.bias
        return torch.sigmoid(inner)


class FuzzyProjectionLayer(nn.Module):
    """Sparse projection-pursuit fuzzy tokens with readable LOW/MEDIUM/HIGH rules."""

    def __init__(
        self,
        input_dim: int,
        projection_indices: tuple[tuple[int, ...], ...],
        *,
        name: str = "ka_projection_pursuit",
        term_centers: tuple[float, ...] = (0.2, 0.5, 0.8),
        term_width: float = 0.18,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection_indices = tuple(tuple(int(index) for index in route) for route in projection_indices)
        if any(not route for route in self.projection_indices):
            raise ValueError("projection_indices must contain non-empty routes.")
        if any(index < 0 or index >= self.input_dim for route in self.projection_indices for index in route):
            raise ValueError("projection_indices must contain valid feature indices.")
        self.output_dim = len(self.projection_indices)
        self.max_route_width = max(len(route) for route in self.projection_indices)
        self.name = str(name)
        self.feature_names = tuple(f"x{index}" for index in range(self.input_dim))
        self.term_names = ("LOW", "MEDIUM", "HIGH") if len(term_centers) == 3 else tuple(
            f"term_{index}" for index in range(len(term_centers))
        )
        index_rows = []
        route_mask = torch.zeros(self.output_dim, self.max_route_width, dtype=torch.float32)
        for row, route in enumerate(self.projection_indices):
            padded = list(route) + [route[-1]] * (self.max_route_width - len(route))
            index_rows.append(padded)
            route_mask[row, : len(route)] = 1.0
        self.register_buffer("projection_index_tensor", torch.tensor(index_rows, dtype=torch.long), persistent=False)
        self.register_buffer("projection_route_mask", route_mask, persistent=True)
        self.projection_weights = nn.Parameter(torch.empty(self.output_dim, self.max_route_width))
        self.term_weights = nn.Parameter(torch.empty(self.output_dim, len(term_centers)))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.rule_logits = nn.Parameter(torch.full((self.output_dim, len(term_centers)), 2.0), requires_grad=False)
        self.register_buffer("rule_active_mask", torch.ones_like(self.rule_logits).reshape(-1), persistent=True)
        base_centers = torch.tensor(term_centers, dtype=torch.float32).view(1, -1).repeat(self.output_dim, 1)
        self.register_buffer("term_centers", base_centers, persistent=True)
        self.register_buffer("term_width", torch.full((self.output_dim,), float(term_width)), persistent=True)
        self.register_buffer("rule_probability_anchor", self.rule_probabilities.detach().clone(), persistent=False)
        nn.init.xavier_uniform_(self.projection_weights)
        nn.init.xavier_uniform_(self.term_weights)

    @property
    def n_rules(self) -> int:
        return int(self.rule_logits.numel())

    @property
    def rule_probabilities(self) -> Tensor:
        return torch.sigmoid(self.rule_logits).reshape(-1) * self.rule_active_mask.to(
            device=self.rule_logits.device,
            dtype=self.rule_logits.dtype,
        )

    def refresh_rule_probability_anchor(self) -> None:
        with torch.no_grad():
            self.rule_probability_anchor.copy_(self.rule_probabilities.detach())

    def set_feature_names(self, feature_names: tuple[str, ...] | list[str]) -> None:
        if len(feature_names) < self.input_dim:
            raise ValueError("feature_names must cover all input features.")
        self.feature_names = tuple(str(name) for name in feature_names[: self.input_dim])

    def output_names(self) -> tuple[str, ...]:
        return tuple(f"projection_{index}" for index in range(self.output_dim))

    def _projection_values(self, inputs: Tensor) -> Tensor:
        indices = self.projection_index_tensor.to(inputs.device)
        gathered = inputs.index_select(dim=1, index=indices.reshape(-1)).view(
            inputs.size(0), self.output_dim, self.max_route_width
        )
        mask = self.projection_route_mask.to(device=inputs.device, dtype=inputs.dtype)
        weights = self.projection_weights.to(device=inputs.device, dtype=inputs.dtype) * mask
        normalizer = weights.abs().sum(dim=1).clamp_min(1e-6)
        return (gathered * weights.unsqueeze(0)).sum(dim=2) / normalizer.unsqueeze(0)

    def _memberships(self, projections: Tensor) -> Tensor:
        centers = self.term_centers.to(device=projections.device, dtype=projections.dtype)
        width = self.term_width.to(device=projections.device, dtype=projections.dtype).clamp_min(1e-6)
        return torch.exp(-0.5 * ((projections.unsqueeze(-1) - centers.unsqueeze(0)) / width.view(1, -1, 1)) ** 2)

    def fit_membership_terms(self, samples: Tensor) -> None:
        with torch.no_grad():
            projections = self._projection_values(samples.to(device=self.projection_weights.device, dtype=self.projection_weights.dtype))
            centers, widths = _featurewise_quantile_terms(projections.detach().cpu())
            self.term_centers.copy_(centers.to(device=self.term_centers.device, dtype=self.term_centers.dtype))
            self.term_width.copy_(widths.to(device=self.term_width.device, dtype=self.term_width.dtype))

    def describe_rule(self, rule_index: int) -> str:
        if rule_index < 0 or rule_index >= self.n_rules:
            raise IndexError(f"Rule index {rule_index} is out of range for {self.name}.")
        projection_index, term_index = divmod(int(rule_index), int(self.rule_logits.size(1)))
        route = self.projection_indices[projection_index]
        names = ", ".join(self.feature_names[index] for index in route)
        return f"pursuit_{projection_index}({names}) IS {self.term_names[term_index]}"

    def rule_scores(self, inputs: Tensor) -> Tensor:
        memberships = self._memberships(self._projection_values(inputs))
        gate = torch.sigmoid(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        mask = self.rule_active_mask.view_as(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        weight = self.term_weights.detach().abs().to(device=inputs.device, dtype=inputs.dtype)
        return (memberships * gate.unsqueeze(0) * mask.unsqueeze(0) * weight.unsqueeze(0)).reshape(inputs.size(0), -1)

    def data_aware_rule_importances(self, inputs: Tensor, outer_scale: Tensor, *, batch_size: int = 8192) -> Tensor:
        device = self.term_weights.device
        dtype = self.term_weights.dtype
        scale = outer_scale.detach().abs().to(device=device, dtype=dtype).view(self.output_dim, 1)
        weighted = self.term_weights.detach().abs() * torch.sigmoid(self.rule_logits.detach()) * scale
        total = torch.zeros_like(weighted)
        sample_count = 0
        for start in range(0, inputs.size(0), max(1, int(batch_size))):
            batch = inputs[start : start + max(1, int(batch_size))].to(device=device, dtype=dtype)
            total = total + self._memberships(self._projection_values(batch)).mean(dim=0) * weighted * batch.size(0)
            sample_count += int(batch.size(0))
        values = (total / float(max(1, sample_count))).reshape(-1)
        return values * self.rule_active_mask.to(device=values.device, dtype=values.dtype)

    def prune_to_top_k_rules(self, max_active_rules: int, importances: Tensor) -> int:
        if max_active_rules <= 0:
            raise ValueError("max_active_rules must be positive.")
        with torch.no_grad():
            active_count = int(self.rule_active_mask.sum().item())
            if max_active_rules >= active_count:
                return active_count
            top_indices = torch.topk(importances, k=int(max_active_rules), largest=True).indices
            new_mask = torch.zeros_like(self.rule_active_mask)
            new_mask[top_indices.to(new_mask.device)] = 1.0
            self.rule_active_mask.copy_(new_mask)
        return int(max_active_rules)

    def forward(self, inputs: Tensor) -> Tensor:
        memberships = self._memberships(self._projection_values(inputs))
        mask = self.rule_active_mask.view_as(self.rule_logits).to(device=inputs.device, dtype=inputs.dtype)
        weights = self.term_weights * torch.sigmoid(self.rule_logits).to(inputs.dtype) * mask
        return torch.sigmoid(torch.einsum("bpt,pt->bp", memberships, weights) + self.bias)


class DeepKANFISModel(nn.Module):
    """Deep interpretable KA-FIS: additive fuzzy concept layers followed by a linear decision."""

    def __init__(
        self,
        input_dim: int,
        *,
        concept_widths: tuple[int, ...] = (10, 6),
        term_centers: tuple[float, ...] = (0.2, 0.5, 0.8),
        term_width: float = 0.18,
        active_feature_indices: tuple[int, ...] | None = None,
        pair_indices: tuple[tuple[int, int], ...] = (),
        projection_indices: tuple[tuple[int, ...], ...] = (),
        first_layer_fan_in: int = 0,
        first_layer_routing: str = "chunk",
        decision_skip_indices: tuple[int, ...] = (),
        output_dim: int = 1,
        train_rule_gates: bool = False,
        decision_skip_gate_init_logit: float = 1.5,
    ) -> None:
        super().__init__()
        if len(concept_widths) < 2:
            raise ValueError("DeepKANFISModel requires at least two concept layers.")
        self.input_dim = int(input_dim)
        self.concept_widths = tuple(int(width) for width in concept_widths)
        self.superposition_terms = int(sum(self.concept_widths))
        if active_feature_indices is None:
            selected_features = tuple(range(self.input_dim))
        else:
            selected_features = tuple(dict.fromkeys(int(index) for index in active_feature_indices))
            if not selected_features:
                selected_features = tuple(range(self.input_dim))
        if any(index < 0 or index >= self.input_dim for index in selected_features):
            raise ValueError("active_feature_indices must contain valid input indices.")
        self.selected_feature_indices = selected_features
        self.register_buffer(
            "selected_feature_index_tensor",
            torch.tensor(selected_features, dtype=torch.long),
            persistent=False,
        )
        self.decision_skip_indices = tuple(dict.fromkeys(int(index) for index in decision_skip_indices))
        if any(index < 0 or index >= self.input_dim for index in self.decision_skip_indices):
            raise ValueError("decision_skip_indices must contain valid input indices.")
        self.register_buffer(
            "decision_skip_index_tensor",
            torch.tensor(self.decision_skip_indices, dtype=torch.long),
            persistent=False,
        )
        self.feature_names = tuple(f"x{index}" for index in range(self.input_dim))
        self.pair_layer = (
            FuzzyPairInteractionLayer(
                self.input_dim,
                tuple((int(left), int(right)) for left, right in pair_indices),
                name="ka_pair_interactions",
                term_centers=term_centers,
                term_width=term_width,
                train_rule_gates=train_rule_gates,
            )
            if pair_indices
            else None
        )
        self.projection_layer = (
            FuzzyProjectionLayer(
                self.input_dim,
                tuple(tuple(int(index) for index in route) for route in projection_indices),
                name="ka_projection_pursuit",
                term_centers=term_centers,
                term_width=term_width,
            )
            if projection_indices
            else None
        )
        layers: list[AdditiveFuzzyConceptLayer] = []
        current_dim = len(selected_features)
        input_prefix = "x"
        current_input_names: tuple[str, ...] | None = tuple(f"x{index}" for index in selected_features)
        for layer_index, width in enumerate(self.concept_widths):
            routes = None
            if layer_index == 0 and int(first_layer_fan_in) > 0:
                fan_in = min(max(1, int(first_layer_fan_in)), current_dim)
                routing = str(first_layer_routing).strip().lower()
                if routing == "chunk":
                    routes = tuple(
                        tuple((concept_index * fan_in + offset) % current_dim for offset in range(fan_in))
                        for concept_index in range(width)
                    )
                elif routing == "banded":
                    routes = tuple(
                        tuple((concept_index + width * offset) % current_dim for offset in range(fan_in))
                        for concept_index in range(width)
                    )
                elif routing == "grouped":
                    if self.input_dim == 54:
                        groups = (
                            [pos for pos, feature in enumerate(selected_features) if feature < 10],
                            [pos for pos, feature in enumerate(selected_features) if 10 <= feature < 14],
                            [pos for pos, feature in enumerate(selected_features) if feature >= 14],
                        )
                    else:
                        groups = tuple(
                            list(range(start, current_dim, 3))
                            for start in range(3)
                        )
                    groups = tuple(group for group in groups if group)
                    routes = tuple(
                        tuple(
                            groups[offset % len(groups)][(concept_index + offset // len(groups)) % len(groups[offset % len(groups)])]
                            for offset in range(fan_in)
                        )
                        for concept_index in range(width)
                    )
                else:
                    raise ValueError("first_layer_routing must be one of: chunk, banded, grouped.")
            layer = AdditiveFuzzyConceptLayer(
                current_dim,
                width,
                name=f"ka_concept_{layer_index + 1}",
                input_prefix=input_prefix,
                term_centers=term_centers,
                term_width=term_width,
                input_names=current_input_names,
                input_indices_by_output=routes,
                train_rule_gates=train_rule_gates,
            )
            layers.append(layer)
            current_input_names = tuple(f"{layer.name}.c{index}" for index in range(width))
            if layer_index == 0 and (self.pair_layer is not None or self.projection_layer is not None):
                extra_names = ()
                extra_dim = 0
                if self.pair_layer is not None:
                    extra_names = (*extra_names, *self.pair_layer.output_names())
                    extra_dim += self.pair_layer.output_dim
                if self.projection_layer is not None:
                    extra_names = (*extra_names, *self.projection_layer.output_names())
                    extra_dim += self.projection_layer.output_dim
                current_input_names = (*current_input_names, *extra_names)
                current_dim = width + extra_dim
            else:
                current_dim = width
            input_prefix = f"c{layer_index + 1}_"
        self.layers = nn.ModuleList(layers)
        self.concept_decision_dim = (
            int(sum(self.concept_widths))
            + (0 if self.pair_layer is None else self.pair_layer.output_dim)
            + (0 if self.projection_layer is None else self.projection_layer.output_dim)
        )
        self.decision_dim = self.concept_decision_dim + len(self.decision_skip_indices)
        self.decision_weights = nn.Parameter(torch.empty(self.decision_dim, int(output_dim)))
        self.decision_bias = nn.Parameter(torch.zeros(int(output_dim)))
        if self.decision_skip_indices:
            self.final_skip_gate_logits = nn.Parameter(
                torch.full((len(self.decision_skip_indices),), float(decision_skip_gate_init_logit))
            )
        else:
            self.final_skip_gate_logits = None
        nn.init.xavier_uniform_(self.decision_weights)

    def set_feature_names(self, feature_names: tuple[str, ...] | list[str]) -> None:
        if len(feature_names) < self.input_dim:
            raise ValueError("feature_names must cover all input features.")
        self.feature_names = tuple(str(name) for name in feature_names[: self.input_dim])
        self.layers[0].input_names = tuple(str(feature_names[index]) for index in self.selected_feature_indices)
        if self.pair_layer is not None:
            self.pair_layer.set_feature_names(self.feature_names)
        if self.projection_layer is not None:
            self.projection_layer.set_feature_names(self.feature_names)
        if len(self.layers) > 1:
            extra_names = ()
            if self.pair_layer is not None:
                extra_names = (*extra_names, *self.pair_layer.output_names())
            if self.projection_layer is not None:
                extra_names = (*extra_names, *self.projection_layer.output_names())
            self.layers[1].input_names = (
                *(f"{self.layers[0].name}.c{index}" for index in range(self.concept_widths[0])),
                *extra_names,
            )

    def fit_membership_terms(self, samples: Tensor) -> None:
        current = self._select_inputs(samples).detach().to(device=self.decision_weights.device, dtype=self.decision_weights.dtype)
        with torch.no_grad():
            self.layers[0].fit_membership_terms(current.detach().cpu())
            if self.pair_layer is not None:
                self.pair_layer.fit_membership_terms(samples.detach().cpu())
            if self.projection_layer is not None:
                self.projection_layer.fit_membership_terms(samples.detach().cpu())

    def refit_upper_membership_terms(self, samples: Tensor) -> None:
        """Re-estimate upper concept terms on the current learned concept space."""
        if len(self.layers) < 2:
            return
        with torch.no_grad():
            full_inputs = samples.detach().to(device=self.decision_weights.device, dtype=self.decision_weights.dtype)
            features = self._select_inputs(full_inputs)
            for layer_index, layer in enumerate(self.layers):
                if layer_index > 0:
                    layer.fit_membership_terms(features.detach().cpu())
                features = layer(features)
                if layer_index == 0 and (self.pair_layer is not None or self.projection_layer is not None):
                    extras = []
                    if self.pair_layer is not None:
                        extras.append(self.pair_layer(full_inputs))
                    if self.projection_layer is not None:
                        extras.append(self.projection_layer(full_inputs))
                    if extras:
                        features = torch.cat((features, *extras), dim=1)

    @property
    def n_rules(self) -> int:
        pair_rules = 0 if self.pair_layer is None else self.pair_layer.n_rules
        projection_rules = 0 if self.projection_layer is None else self.projection_layer.n_rules
        return int(sum(layer.n_rules for layer in self.layers) + pair_rules + projection_rules)

    @property
    def rule_probabilities(self) -> Tensor:
        values = [layer.rule_probabilities for layer in self.layers]
        if self.pair_layer is not None:
            values.append(self.pair_layer.rule_probabilities)
        if self.projection_layer is not None:
            values.append(self.projection_layer.rule_probabilities)
        return torch.cat(tuple(values), dim=0)

    def iter_rule_layer_entries(self) -> tuple[tuple[str, nn.Module], ...]:
        entries: list[tuple[str, nn.Module]] = [(layer.name, layer) for layer in self.layers]
        if self.pair_layer is not None:
            entries.append((self.pair_layer.name, self.pair_layer))
        if self.projection_layer is not None:
            entries.append((self.projection_layer.name, self.projection_layer))
        return tuple(entries)

    def _layer_decision_scales(self) -> tuple[list[Tensor], Tensor | None, Tensor | None]:
        direct_scales: list[Tensor] = []
        offset = 0
        decision_scale = self.decision_weights[: self.concept_decision_dim].detach().abs().mean(dim=1)
        direct_scales.append(decision_scale[offset : offset + self.concept_widths[0]])
        offset += self.concept_widths[0]
        pair_scale = None
        if self.pair_layer is not None:
            pair_scale = decision_scale[offset : offset + self.pair_layer.output_dim].clone()
            offset += self.pair_layer.output_dim
        projection_scale = None
        if self.projection_layer is not None:
            projection_scale = decision_scale[offset : offset + self.projection_layer.output_dim].clone()
            offset += self.projection_layer.output_dim
        for width in self.concept_widths[1:]:
            direct_scales.append(decision_scale[offset : offset + width])
            offset += width
        scales = [scale.clone() for scale in direct_scales]
        downstream_scale = direct_scales[-1]
        for layer_index in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_index]
            indirect = (layer.weights.detach().abs() * downstream_scale.view(-1, 1, 1)).sum(dim=(0, 2))
            if layer_index == 1 and (self.pair_layer is not None or self.projection_layer is not None):
                split = self.concept_widths[0]
                scales[0] = scales[0] + indirect[:split]
                extra_offset = split
                if self.pair_layer is not None:
                    pair_indirect = indirect[extra_offset : extra_offset + self.pair_layer.output_dim]
                    pair_scale = pair_scale + pair_indirect if pair_scale is not None else pair_indirect
                    extra_offset += self.pair_layer.output_dim
                if self.projection_layer is not None:
                    projection_indirect = indirect[extra_offset : extra_offset + self.projection_layer.output_dim]
                    projection_scale = (
                        projection_scale + projection_indirect
                        if projection_scale is not None
                        else projection_indirect
                    )
            else:
                scales[layer_index - 1] = scales[layer_index - 1] + indirect
            downstream_scale = scales[layer_index - 1]
        return scales, pair_scale, projection_scale

    def _concept_label(self, flat_index: int) -> str:
        if flat_index < self.concept_widths[0]:
            return f"{self.layers[0].name}.c{flat_index}"
        offset = self.concept_widths[0]
        if self.pair_layer is not None:
            if flat_index < offset + self.pair_layer.output_dim:
                return self.pair_layer.output_names()[flat_index - offset]
            offset += self.pair_layer.output_dim
        if self.projection_layer is not None:
            if flat_index < offset + self.projection_layer.output_dim:
                return self.projection_layer.output_names()[flat_index - offset]
            offset += self.projection_layer.output_dim
        for layer_index, width in enumerate(self.concept_widths[1:], start=1):
            if flat_index < offset + width:
                return f"{self.layers[layer_index].name}.c{flat_index - offset}"
            offset += width
        return f"concept_{flat_index}"

    def _select_inputs(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(f"Expected inputs with shape [batch, {self.input_dim}], got {tuple(inputs.shape)}.")
        return inputs.index_select(dim=1, index=self.selected_feature_index_tensor.to(inputs.device))

    def forward_features(self, inputs: Tensor) -> Tensor:
        features = self._select_inputs(inputs)
        for layer_index, layer in enumerate(self.layers):
            features = layer(features)
            if layer_index == 0 and (self.pair_layer is not None or self.projection_layer is not None):
                extras = []
                if self.pair_layer is not None:
                    extras.append(self.pair_layer(inputs))
                if self.projection_layer is not None:
                    extras.append(self.projection_layer(inputs))
                features = torch.cat((features, *extras), dim=1)
        return features

    def forward_all_concepts(self, inputs: Tensor) -> Tensor:
        concepts: list[Tensor] = []
        features = self._select_inputs(inputs)
        for layer_index, layer in enumerate(self.layers):
            features = layer(features)
            concepts.append(features)
            if layer_index == 0 and (self.pair_layer is not None or self.projection_layer is not None):
                extras = []
                if self.pair_layer is not None:
                    pair_features = self.pair_layer(inputs)
                    concepts.append(pair_features)
                    extras.append(pair_features)
                if self.projection_layer is not None:
                    projection_features = self.projection_layer(inputs)
                    concepts.append(projection_features)
                    extras.append(projection_features)
                features = torch.cat((features, *extras), dim=1)
        return torch.cat(concepts, dim=1)

    def forward_decision_inputs(self, inputs: Tensor) -> Tensor:
        concepts = self.forward_all_concepts(inputs)
        if self.decision_skip_index_tensor.numel() == 0:
            return concepts
        skip = inputs.index_select(dim=1, index=self.decision_skip_index_tensor.to(inputs.device))
        if self.final_skip_gate_logits is not None:
            skip = skip * torch.sigmoid(self.final_skip_gate_logits.to(device=skip.device, dtype=skip.dtype)).unsqueeze(0)
        return torch.cat((concepts, skip.to(dtype=concepts.dtype)), dim=1)

    def explain_sample(self, inputs: Tensor, *, top_k_rules: int = 5, top_k_concepts: int = 5) -> dict[str, object]:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.size(0) != 1:
            raise ValueError("explain_sample expects one sample.")
        with torch.no_grad():
            layer_inputs: list[Tensor] = []
            pair_input: Tensor | None = None
            projection_input: Tensor | None = None
            features = self._select_inputs(inputs.to(device=self.decision_weights.device, dtype=self.decision_weights.dtype))
            full_inputs = inputs.to(device=self.decision_weights.device, dtype=self.decision_weights.dtype)
            for layer_index, layer in enumerate(self.layers):
                layer_inputs.append(features)
                features = layer(features)
                if layer_index == 0 and (self.pair_layer is not None or self.projection_layer is not None):
                    extras = []
                    if self.pair_layer is not None:
                        pair_input = full_inputs
                        pair_features = self.pair_layer(full_inputs)
                        extras.append(pair_features)
                    if self.projection_layer is not None:
                        projection_input = full_inputs
                        projection_features = self.projection_layer(full_inputs)
                        extras.append(projection_features)
                    features = torch.cat((features, *extras), dim=1)
            decision_inputs = self.forward_decision_inputs(
                full_inputs
            )
            logit = (decision_inputs @ self.decision_weights + self.decision_bias).reshape(-1)[0]
            probability = torch.sigmoid(logit).item()
            concept_contrib = (
                decision_inputs[0, : self.concept_decision_dim] * self.decision_weights[: self.concept_decision_dim, 0]
            ).detach().cpu()
            raw_contrib = (
                decision_inputs[0, self.concept_decision_dim :] * self.decision_weights[self.concept_decision_dim :, 0]
            ).detach().cpu()

        layer_explanations: list[dict[str, object]] = []
        for layer, layer_input in zip(self.layers, layer_inputs):
            scores = layer.rule_scores(layer_input)[0].detach().cpu()
            k = min(max(1, int(top_k_rules)), scores.numel())
            values, indices = torch.topk(scores, k=k)
            layer_explanations.append(
                {
                    "layer": layer.name,
                    "top_rules": [
                        {
                            "rule": layer.describe_rule(int(index)),
                            "score": float(value.item()),
                        }
                        for value, index in zip(values, indices)
                    ],
                }
            )
        if self.pair_layer is not None and pair_input is not None:
            scores = self.pair_layer.rule_scores(pair_input)[0].detach().cpu()
            k = min(max(1, int(top_k_rules)), scores.numel())
            values, indices = torch.topk(scores, k=k)
            layer_explanations.append(
                {
                    "layer": self.pair_layer.name,
                    "top_rules": [
                        {
                            "rule": self.pair_layer.describe_rule(int(index)),
                            "score": float(value.item()),
                        }
                        for value, index in zip(values, indices)
                    ],
                }
            )
        if self.projection_layer is not None and projection_input is not None:
            scores = self.projection_layer.rule_scores(projection_input)[0].detach().cpu()
            k = min(max(1, int(top_k_rules)), scores.numel())
            values, indices = torch.topk(scores, k=k)
            layer_explanations.append(
                {
                    "layer": self.projection_layer.name,
                    "top_rules": [
                        {
                            "rule": self.projection_layer.describe_rule(int(index)),
                            "score": float(value.item()),
                        }
                        for value, index in zip(values, indices)
                    ],
                }
            )

        concept_k = min(max(1, int(top_k_concepts)), concept_contrib.numel())
        concept_values, concept_indices = torch.topk(concept_contrib.abs(), k=concept_k)
        raw_features: list[dict[str, object]] = []
        if raw_contrib.numel() > 0:
            raw_k = min(max(1, int(top_k_concepts)), raw_contrib.numel())
            raw_values, raw_indices = torch.topk(raw_contrib.abs(), k=raw_k)
            raw_features = [
                {
                    "feature": self.feature_names[self.decision_skip_indices[int(index)]],
                    "signed_contribution": float(raw_contrib[int(index)].item()),
                    "absolute_contribution": float(value.item()),
                }
                for value, index in zip(raw_values, raw_indices)
            ]
        return {
            "logit": float(logit.item()),
            "probability": float(probability),
            "top_concepts": [
                {
                    "concept": self._concept_label(int(index)),
                    "signed_contribution": float(concept_contrib[int(index)].item()),
                    "absolute_contribution": float(value.item()),
                }
                for value, index in zip(concept_values, concept_indices)
            ],
            "top_raw_features": raw_features,
            "layers": layer_explanations,
        }

    def global_rule_dictionary(self, *, top_k: int = 30) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        with torch.no_grad():
            layer_scales_raw, pair_scale_raw, projection_scale_raw = self._layer_decision_scales()
            layer_scales = [scale.cpu() for scale in layer_scales_raw]
            for layer, scale in zip(self.layers, layer_scales):
                importances = (layer.weights.detach().abs().cpu() * scale.view(-1, 1, 1)).reshape(-1)
                importances = importances * layer.rule_probabilities.detach().cpu()
                k = min(max(1, int(top_k)), importances.numel())
                values, indices = torch.topk(importances, k=k)
                for value, index in zip(values, indices):
                    entries.append(
                        {
                            "layer": layer.name,
                            "rule": layer.describe_rule(int(index)),
                            "importance": float(value.item()),
                        }
                    )
            if self.pair_layer is not None and pair_scale_raw is not None:
                scale = pair_scale_raw.cpu()
                importances = (self.pair_layer.weights.detach().abs().cpu() * scale.view(-1, 1, 1)).reshape(-1)
                importances = importances * self.pair_layer.rule_probabilities.detach().cpu()
                k = min(max(1, int(top_k)), importances.numel())
                values, indices = torch.topk(importances, k=k)
                for value, index in zip(values, indices):
                    entries.append(
                        {
                            "layer": self.pair_layer.name,
                            "rule": self.pair_layer.describe_rule(int(index)),
                            "importance": float(value.item()),
                        }
                    )
            if self.projection_layer is not None and projection_scale_raw is not None:
                scale = projection_scale_raw.cpu()
                importances = (
                    self.projection_layer.term_weights.detach().abs().cpu() * scale.view(-1, 1)
                ).reshape(-1)
                importances = importances * self.projection_layer.rule_probabilities.detach().cpu()
                k = min(max(1, int(top_k)), importances.numel())
                values, indices = torch.topk(importances, k=k)
                for value, index in zip(values, indices):
                    entries.append(
                        {
                            "layer": self.projection_layer.name,
                            "rule": self.projection_layer.describe_rule(int(index)),
                            "importance": float(value.item()),
                        }
                    )
        entries.sort(key=lambda item: -float(item["importance"]))
        return entries[: max(1, int(top_k))]

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        del top_k_rules
        return self.forward_decision_inputs(inputs) @ self.decision_weights + self.decision_bias

    def prune_to_top_k_rules(
        self,
        max_active_rules: int,
        *,
        inputs: Tensor | None = None,
        batch_size: int = 8192,
    ) -> int:
        if inputs is None:
            raise ValueError("DeepKANFISModel pruning requires representative inputs.")
        total_active = sum(int(layer.rule_active_mask.sum().item()) for layer in self.layers)
        if self.pair_layer is not None:
            total_active += int(self.pair_layer.rule_active_mask.sum().item())
        if self.projection_layer is not None:
            total_active += int(self.projection_layer.rule_active_mask.sum().item())
        if max_active_rules >= total_active:
            return total_active
        prunable_layers: list[nn.Module] = [*self.layers]
        if self.pair_layer is not None:
            prunable_layers.append(self.pair_layer)
        if self.projection_layer is not None:
            prunable_layers.append(self.projection_layer)
        active_counts = [int(layer.rule_active_mask.sum().item()) for layer in prunable_layers]
        budgets = [
            min(count, max(1, int(max_active_rules * count / total_active)))
            if count > 0
            else 0
            for count in active_counts
        ]
        while sum(budgets) > int(max_active_rules):
            candidates = [index for index, budget in enumerate(budgets) if budget > 1]
            if not candidates:
                break
            index = max(candidates, key=lambda item: budgets[item])
            budgets[index] -= 1
        while sum(budgets) < int(max_active_rules):
            candidates = [
                index
                for index, (budget, count) in enumerate(zip(budgets, active_counts))
                if budget < count
            ]
            if not candidates:
                break
            index = max(candidates, key=lambda item: active_counts[item] - budgets[item])
            budgets[index] += 1

        with torch.no_grad():
            selected_inputs = self._select_inputs(inputs)
            features: list[Tensor] = [selected_inputs]
            pair_inputs: Tensor | None = None
            projection_inputs: Tensor | None = None
            current = selected_inputs.to(device=self.decision_weights.device, dtype=self.decision_weights.dtype)
            full_inputs = inputs.to(device=self.decision_weights.device, dtype=self.decision_weights.dtype)
            for layer_index, layer in enumerate(self.layers[:-1]):
                current = layer(current)
                if layer_index == 0 and (self.pair_layer is not None or self.projection_layer is not None):
                    extras = []
                    if self.pair_layer is not None:
                        pair_inputs = inputs
                        extras.append(self.pair_layer(full_inputs))
                    if self.projection_layer is not None:
                        projection_inputs = inputs
                        extras.append(self.projection_layer(full_inputs))
                    current = torch.cat((current, *extras), dim=1)
                features.append(current.detach().cpu())
            scales, pair_scale, projection_scale = self._layer_decision_scales()

        kept = 0
        for layer, layer_inputs, scale, budget in zip(self.layers, features, scales, budgets):
            importances = layer.data_aware_rule_importances(layer_inputs, scale, batch_size=batch_size)
            kept += layer.prune_to_top_k_rules(budget, importances)
        if self.pair_layer is not None and pair_inputs is not None and pair_scale is not None:
            importances = self.pair_layer.data_aware_rule_importances(pair_inputs, pair_scale, batch_size=batch_size)
            kept += self.pair_layer.prune_to_top_k_rules(budgets[len(self.layers)], importances)
        if self.projection_layer is not None and projection_inputs is not None and projection_scale is not None:
            budget_index = len(self.layers) + (1 if self.pair_layer is not None else 0)
            importances = self.projection_layer.data_aware_rule_importances(
                projection_inputs, projection_scale, batch_size=batch_size
            )
            kept += self.projection_layer.prune_to_top_k_rules(budgets[budget_index], importances)
        return kept
