#!/usr/bin/env python3
"""Regression test: generator backward must not touch discriminator gradients.

With manual optimisation + gradient accumulation, ``loss_g`` is differentiated
every micro-batch. ``loss_g`` also depends on the discriminator (through
``D(fake)`` and the feature maps), so unless D is frozen for that backward the
D gradient buffers collect ``dL_G/dD`` and the D optimiser step applies it.
That trains D to *minimise* the generator loss, which collapses the
discriminator to a constant output (D(real) == D(fake)).

Run:
    python tests/test_grad_accum.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.discriminators.frequencydis import MultiFrequencyDiscriminator
from look2hear.losses.gan_losses import MultiFrequencyDisLoss, MultiFrequencyGenLoss
from look2hear.models.apollo import Apollo


def snapshot_grads(module):
    return {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in module.named_parameters()
    }


def grads_equal(a, b):
    if a.keys() != b.keys():
        return False
    for k in a:
        if (a[k] is None) != (b[k] is None):
            return False
        if a[k] is not None and not torch.allclose(a[k], b[k], atol=0, rtol=0):
            return False
    return True


def main():
    torch.manual_seed(0)
    model = Apollo(sr=44100, win=20, feature_dim=64, layer=1)
    disc = MultiFrequencyDiscriminator(nch=2, window=[64, 256])
    loss_d_fn = MultiFrequencyDisLoss()
    loss_g_fn = MultiFrequencyGenLoss()

    codec = torch.randn(1, 2, 44100 // 2, requires_grad=False)
    target = torch.randn(1, 2, 44100 // 2)

    out = model(codec)

    # reference: gradients from L_D alone
    disc.zero_grad(set_to_none=True)
    est_out, _ = disc(out.detach(), sample_rate=44100)
    tgt_out, _ = disc(target, sample_rate=44100)
    loss_d, _ = loss_d_fn(tgt_out, est_out, return_stats=True)
    loss_d.backward()
    ref = snapshot_grads(disc)

    # now the buggy-looking sequence: freeze D, backward L_G, compare
    est_out, est_feat = disc(out, sample_rate=44100)
    _, tgt_feat = disc(target, sample_rate=44100)
    loss_g, _ = loss_g_fn(est_out, est_feat, tgt_feat, out, target, return_stats=True)

    disc.requires_grad_(False)
    try:
        loss_g.backward()
    finally:
        disc.requires_grad_(True)

    after = snapshot_grads(disc)
    assert grads_equal(ref, after), "D gradients changed by the generator backward!"

    # G must still receive gradients
    g_grad = sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)
    assert g_grad > 0, "generator received no gradient"
    print(f"OK: D grads untouched by L_G backward (loss_g={loss_g.item():.4f}, "
          f"sum|grad_G|^2={g_grad.item():.4f})")


if __name__ == "__main__":
    main()
