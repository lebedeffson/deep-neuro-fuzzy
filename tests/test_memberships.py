import pytest
import torch

from ruanfis.memberships import GaussianMembership, GeneralizedBellMembership


def test_gaussian_membership_peaks_at_its_centers() -> None:
    membership = GaussianMembership(
        centers=[0.0, 1.0],
        spreads=[0.2, 0.2],
        term_names=["low", "high"],
    )
    values = membership(torch.tensor([0.0, 1.0]))

    assert values.shape == (2, 2)
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)
    assert values[0, 0].item() == pytest.approx(1.0, rel=1e-6)
    assert values[1, 1].item() == pytest.approx(1.0, rel=1e-6)


def test_bell_membership_stays_bounded() -> None:
    membership = GeneralizedBellMembership(
        centers=[-1.0, 1.0],
        widths=[0.5, 0.5],
        slopes=[2.0, 2.0],
    )
    values = membership(torch.linspace(-2.0, 2.0, steps=9))

    assert values.shape == (9, 2)
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)

