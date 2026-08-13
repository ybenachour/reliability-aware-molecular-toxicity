import torch

from toxicity_screening.losses import masked_binary_cross_entropy_with_logits


def test_masked_bce_ignores_missing_entries():
    logits = torch.tensor([[0.0, 20.0], [0.0, -20.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    observed_only = masked_binary_cross_entropy_with_logits(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0])
    )
    assert torch.allclose(observed_only, expected)


def test_empty_mask_fails():
    try:
        masked_binary_cross_entropy_with_logits(
            torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros((2, 2))
        )
    except ValueError as exc:
        assert "no observed labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
