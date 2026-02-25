"""
Unit test for gradient flow in DPS+DRaFT online TTT.

Verifies:
  1. DRaFT reward gradient flows to LoRA params (through ODE suffix)
  2. DPS guidance gradient does NOT flow to LoRA params (detached)
  3. Buffer replay loss gradient flows to LoRA params
  4. No gradient cross-contamination between paths

Uses a tiny dummy model — no checkpoint needed, runs in seconds on CPU.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------
# Minimal stubs that mimic the real interfaces
# ---------------------------------------------------------------

class DummyUNet(nn.Module):
    """Tiny 'UNet' — just two convs so we can test LoRA wrapping."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x, t):
        return self.conv2(torch.relu(self.conv1(x)))


class DummyModel(nn.Module):
    """Mimics model.tweedie(x, sigma) and model.get_in_shape()."""
    def __init__(self):
        super().__init__()
        self.unet = DummyUNet()

    def tweedie(self, x, sigma):
        t = torch.as_tensor(sigma).to(x.device)
        return self.unet(x, t)

    def get_in_shape(self):
        return (3, 8, 8)


class DummyScheduler:
    """Minimal scheduler with 5 sigma steps."""
    def __init__(self):
        self.sigma_steps = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])

    def get_sigma_inv(self, sigma):
        return sigma

    def get_scaling(self, t):
        return torch.tensor(1.0)

    def get_scaling_derivative(self, t):
        return torch.tensor(0.0)

    def get_sigma_derivative(self, t):
        return torch.tensor(-1.0)

    def get_prior_sigma(self):
        return self.sigma_steps[0]


class DummyOperator:
    """Mimics forward_op.loss(x, y) = ||x - y||^2 per sample."""
    def __call__(self, x):
        return x

    def loss(self, x, y):
        return ((x - y) ** 2).flatten(1).sum(-1)


# ---------------------------------------------------------------
# LoRA stub
# ---------------------------------------------------------------

class SimpleLora(nn.Module):
    """Wraps a Conv2d with a rank-1 adapter for testing."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        inc = conv.in_channels
        outc = conv.out_channels
        self.down = nn.Conv2d(inc, 1, 1, bias=False)
        self.up = nn.Conv2d(1, outc, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.conv(x) + self.up(self.down(x))


def apply_test_lora(model):
    lora = SimpleLora(model.unet.conv1)
    model.unet.conv1 = lora
    return [lora]


def get_lora_params(lora_modules):
    params = []
    for m in lora_modules:
        params.extend([m.down.weight, m.up.weight])
    return params


def remove_test_lora(model, lora_modules):
    for m in lora_modules:
        model.unet.conv1 = m.conv


# ---------------------------------------------------------------
# Copy of dps_draft_k_sample (matches run_online_ttt.py)
# ---------------------------------------------------------------

def dps_draft_k_sample(model, scheduler, forward_op, y, device,
                       draft_k=1, guidance_scale=1.0, lora_params=None):
    in_shape = model.get_in_shape()
    x = torch.randn(1, *in_shape, device=device) * scheduler.get_prior_sigma()
    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1
    grad_start = max(num_steps - draft_k, 0)

    for i in range(num_steps):
        sigma = sigma_steps[i]
        sigma_next = sigma_steps[i + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        if i < grad_start:
            model.requires_grad_(True)
            xt_in = x.detach().requires_grad_(True)
            x0hat = model.tweedie(xt_in / st, sigma)
            loss_dps = forward_op.loss(x0hat, y)
            grad_xt = torch.autograd.grad(loss_dps.sum(), xt_in)[0]
            model.requires_grad_(False)

            with torch.no_grad():
                norm_factor = loss_dps.sqrt().view(
                    -1, *([1] * (grad_xt.ndim - 1))).clamp(min=1e-8)
                normalized_grad = grad_xt / norm_factor
                score = (x0hat.detach() - x / st) / sigma ** 2
                deriv = dst / st * x - st * dsigma * sigma * score
                x = x + dt * deriv - guidance_scale * normalized_grad
        else:
            if i == grad_start:
                x = x.detach().requires_grad_(True)
                if lora_params is not None:
                    for p in lora_params:
                        p.requires_grad_(True)

            x0hat = model.tweedie(x / st, sigma)

            loss_dps = forward_op.loss(x0hat, y)
            grad_xt = torch.autograd.grad(
                loss_dps.sum(), x, retain_graph=True)[0]
            norm_factor = loss_dps.detach().sqrt().view(
                -1, *([1] * (grad_xt.ndim - 1))).clamp(min=1e-8)
            normalized_grad = (grad_xt / norm_factor).detach()

            score = (x0hat - x / st) / sigma ** 2
            deriv = dst / st * x - st * dsigma * sigma * score
            x = x + dt * deriv - guidance_scale * normalized_grad

    return x


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

def test_draft_gradient_flows_to_lora():
    """DRaFT reward loss backward should produce nonzero grad on LoRA params."""
    torch.manual_seed(42)
    model = DummyModel()
    scheduler = DummyScheduler()
    op = DummyOperator()

    lora_modules = apply_test_lora(model)
    lp = get_lora_params(lora_modules)

    y = torch.randn(1, 3, 8, 8)
    x_0 = dps_draft_k_sample(model, scheduler, op, y, "cpu",
                              draft_k=1, guidance_scale=1.0, lora_params=lp)
    reward_loss = op.loss(x_0, y).mean()
    reward_loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in lp)
    remove_test_lora(model, lora_modules)
    assert has_grad, "FAIL: DRaFT reward gradient did NOT reach LoRA params"
    print("PASS: DRaFT reward gradient flows to LoRA params")


def test_dps_gradient_detached_from_lora():
    """With draft_k=0 (all prefix), x_0 should have no grad connection to LoRA."""
    torch.manual_seed(42)
    model = DummyModel()
    scheduler = DummyScheduler()
    op = DummyOperator()

    lora_modules = apply_test_lora(model)
    lp = get_lora_params(lora_modules)

    y = torch.randn(1, 3, 8, 8)
    x_0 = dps_draft_k_sample(model, scheduler, op, y, "cpu",
                              draft_k=0, guidance_scale=1.0, lora_params=lp)

    assert not x_0.requires_grad, \
        "FAIL: x_0 should not require grad when draft_k=0"
    remove_test_lora(model, lora_modules)
    print("PASS: DPS-only path (draft_k=0) produces no grad on LoRA")


def test_dps_does_not_contaminate_lora_grad():
    """LoRA grads with guidance_scale=0 vs 1.0 should both be finite and nonzero.

    DPS changes the trajectory so grads differ, but no NaN/Inf from second-order
    effects leaking through.
    """
    torch.manual_seed(42)
    model = DummyModel()
    scheduler = DummyScheduler()
    op = DummyOperator()
    y = torch.randn(1, 3, 8, 8)

    grads = {}
    for gs in [0.0, 1.0]:
        lora_modules = apply_test_lora(model)
        lp = get_lora_params(lora_modules)

        torch.manual_seed(42)
        x_0 = dps_draft_k_sample(model, scheduler, op, y, "cpu",
                                  draft_k=1, guidance_scale=gs, lora_params=lp)
        loss = op.loss(x_0, y).mean()
        loss.backward()

        grad_vec = torch.cat([p.grad.flatten() for p in lp])
        grads[gs] = grad_vec.clone()
        remove_test_lora(model, lora_modules)

    assert torch.isfinite(grads[0.0]).all(), "FAIL: grad with gs=0 has NaN/Inf"
    assert torch.isfinite(grads[1.0]).all(), "FAIL: grad with gs=1 has NaN/Inf"
    assert grads[0.0].abs().sum() > 0, "FAIL: grad with gs=0 is all zero"
    assert grads[1.0].abs().sum() > 0, "FAIL: grad with gs=1 is all zero"

    diff = (grads[0.0] - grads[1.0]).abs().sum()
    assert diff > 1e-10, "FAIL: grads identical — DPS guidance has no effect"
    print(f"PASS: DPS guidance changes trajectory, grads finite (diff={diff:.6f})")


def test_buffer_loss_gradient():
    """Buffer replay loss should produce gradient on LoRA params."""
    torch.manual_seed(42)
    model = DummyModel()
    scheduler = DummyScheduler()
    op = DummyOperator()

    lora_modules = apply_test_lora(model)
    lp = get_lora_params(lora_modules)

    x0_stored = torch.randn(1, 3, 8, 8)
    y_stored = torch.randn(1, 3, 8, 8)
    sigma = scheduler.sigma_steps[2]

    noise = torch.randn_like(x0_stored)
    x_noisy = x0_stored + sigma * noise
    x_hat = model.tweedie(x_noisy, sigma)
    loss = op.loss(x_hat, y_stored).mean()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in lp)
    remove_test_lora(model, lora_modules)
    assert has_grad, "FAIL: buffer replay gradient did NOT reach LoRA params"
    print("PASS: Buffer replay loss gradient flows to LoRA params")


def test_requires_grad_restored():
    """After dps_draft_k_sample, base model params should have requires_grad=False,
    but LoRA params should have requires_grad=True."""
    torch.manual_seed(42)
    model = DummyModel()
    scheduler = DummyScheduler()
    op = DummyOperator()

    lora_modules = apply_test_lora(model)
    lp = get_lora_params(lora_modules)
    y = torch.randn(1, 3, 8, 8)

    x_0 = dps_draft_k_sample(model, scheduler, op, y, "cpu",
                              draft_k=1, guidance_scale=1.0, lora_params=lp)

    # Base (frozen) conv2 should not require grad
    for p in model.unet.conv2.parameters():
        assert not p.requires_grad, \
            "FAIL: base model params have requires_grad=True after sampling"

    # LoRA params should still require grad
    for p in lp:
        assert p.requires_grad, \
            "FAIL: LoRA params lost requires_grad after sampling"

    remove_test_lora(model, lora_modules)
    print("PASS: requires_grad correctly set after sampling")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_draft_gradient_flows_to_lora,
        test_dps_gradient_detached_from_lora,
        test_dps_does_not_contaminate_lora_grad,
        test_buffer_loss_gradient,
        test_requires_grad_restored,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(str(e))
            failed += 1
        except Exception as e:
            print(f"ERROR in {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*40}")
