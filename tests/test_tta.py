"""Flip-average TTA must actually average the flipped forward pass.

--tta hflip is how the final numbers will be produced, and it had never been
exercised, so the arithmetic is pinned here with a stub model.
"""
import torch

from src.evaluate import predict_probs

W = 16


class _RampModel(torch.nn.Module):
    """Returns a left-to-right ramp, ignoring the input, so a flip is observable."""

    def forward(self, images, ablation_cfg=None):
        b, _, h, _ = images.shape
        ramp = torch.linspace(0.05, 0.95, W).view(1, 1, 1, W).expand(b, 1, h, W).contiguous()
        logits = torch.logit(ramp)
        return type("Out", (), {"saliency_logits": logits})(), []


def test_without_tta_the_map_is_returned_unchanged():
    images = torch.zeros(2, 3, 8, W)
    plain = predict_probs(_RampModel(), images, None, False)
    assert plain.shape == (2, 1, 8, W)
    assert plain[0, 0, 0, 0] < plain[0, 0, 0, -1]  # still a ramp


def test_hflip_tta_averages_the_map_with_its_mirror():
    images = torch.zeros(2, 3, 8, W)
    plain = predict_probs(_RampModel(), images, None, False)
    tta = predict_probs(_RampModel(), images, None, True)

    assert torch.allclose(tta, 0.5 * plain + 0.5 * torch.flip(plain, dims=[-1]), atol=1e-5)
    # the ramp must become symmetric, and averaging must reduce its contrast
    assert torch.allclose(tta[0, 0, 0, :], torch.flip(tta[0, 0, 0, :], dims=[0]), atol=1e-5)
    assert (tta.max() - tta.min()).item() < (plain.max() - plain.min()).item()
