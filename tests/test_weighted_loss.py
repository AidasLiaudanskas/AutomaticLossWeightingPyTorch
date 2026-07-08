import pytest
import torch

from weighted_loss import WeightedLoss


def test_forward_matches_closed_form_reg_only():
    torch.manual_seed(0)
    criterion = WeightedLoss(n_reg_losses=2, n_cls_losses=0)
    reg_losses = [torch.tensor(1.5), torch.tensor(2.0)]

    result = criterion(reg_losses, [])

    expected = torch.zeros(())
    for coeff, loss in zip(criterion.reg_coeffs, reg_losses):
        expected = expected + 0.5 * torch.exp(-coeff) * loss + 0.5 * coeff
    assert torch.allclose(result, expected)


def test_forward_matches_closed_form_cls_only():
    torch.manual_seed(0)
    criterion = WeightedLoss(n_reg_losses=0, n_cls_losses=1)
    cls_losses = [torch.tensor(0.7)]

    result = criterion([], cls_losses)

    expected = torch.zeros(())
    for coeff, loss in zip(criterion.cls_coeffs, cls_losses):
        expected = expected + torch.exp(-coeff) * loss + 0.5 * coeff
    assert torch.allclose(result, expected)


def test_forward_matches_closed_form_mixed():
    torch.manual_seed(0)
    criterion = WeightedLoss(n_reg_losses=2, n_cls_losses=1)
    reg_losses = [torch.tensor(1.5), torch.tensor(2.0)]
    cls_losses = [torch.tensor(0.7)]

    result = criterion(reg_losses, cls_losses)

    expected = torch.zeros(())
    for coeff, loss in zip(criterion.reg_coeffs, reg_losses):
        expected = expected + 0.5 * torch.exp(-coeff) * loss + 0.5 * coeff
    for coeff, loss in zip(criterion.cls_coeffs, cls_losses):
        expected = expected + torch.exp(-coeff) * loss + 0.5 * coeff
    assert torch.allclose(result, expected)


def test_forward_returns_scalar_shape():
    criterion = WeightedLoss(n_reg_losses=1, n_cls_losses=1)
    result = criterion([torch.tensor(1.0)], [torch.tensor(1.0)])
    assert result.shape == ()


def test_gradients_flow_to_coeffs():
    criterion = WeightedLoss(n_reg_losses=1, n_cls_losses=1)
    reg_loss = torch.tensor(1.0, requires_grad=True)
    cls_loss = torch.tensor(1.0, requires_grad=True)

    loss = criterion([reg_loss], [cls_loss])
    loss.backward()

    assert criterion.reg_coeffs[0].grad is not None
    assert criterion.cls_coeffs[0].grad is not None
    assert criterion.reg_coeffs[0].grad.item() != 0.0
    assert criterion.cls_coeffs[0].grad.item() != 0.0


def test_mismatched_reg_loss_count_raises():
    criterion = WeightedLoss(n_reg_losses=2, n_cls_losses=0)
    with pytest.raises(AssertionError):
        criterion([torch.tensor(1.0)], [])


def test_mismatched_cls_loss_count_raises():
    criterion = WeightedLoss(n_reg_losses=0, n_cls_losses=1)
    with pytest.raises(AssertionError):
        criterion([], [])


def test_device_handling_after_move():
    # Regression test: forward() must use the module's current device,
    # not one cached at construction time.
    criterion = WeightedLoss(n_reg_losses=1, n_cls_losses=0)
    criterion = criterion.to("cpu")
    result = criterion([torch.tensor(1.0)], [])
    assert result.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_handling_cuda():
    criterion = WeightedLoss(n_reg_losses=1, n_cls_losses=0).to("cuda")
    result = criterion([torch.tensor(1.0, device="cuda")], [])
    assert result.device.type == "cuda"
