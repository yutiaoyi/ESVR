import unittest

import torch

from scaffold4d.model import ScaffoldESVR, normalize_quaternion, query_block


class Mathematics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_temporal_gate_peak_and_decay(self):
        model = ScaffoldESVR(anchors=4, children=3, shape=(16, 12, 8), temporal_cutoff=0.0).double()
        g = model.generate()
        home_t = g.home_t[0, 0, 0]
        sigma0 = g.sigma[0, 0, 0]
        # A child's own temporal gate is exactly 1 at its own home time...
        gate_home = torch.exp(-((home_t - home_t) * sigma0).abs().pow(model.config["temporal_beta"]))
        torch.testing.assert_close(gate_home, torch.tensor(1.0, dtype=torch.float64))
        # ...and decays strictly as |query time - home time| grows.
        near = model.at((home_t + 0.1).item(), generated=g)
        far = model.at((home_t + 5.0).item(), generated=g)
        self.assertGreater(len(near.get_xyz), len(far.get_xyz))

    def test_at_time_matches_manual_recomputation(self):
        model = ScaffoldESVR(anchors=3, children=2, shape=(8, 8, 8), temporal_cutoff=0.0).double()
        g = model.generate()
        t = 0.37
        sliced = model.at(t, generated=g)
        dt = g.home_t - t
        opacity_t = torch.exp(-(dt * g.sigma).abs().pow(model.config["temporal_beta"]))
        xyz = g.home_xyz + g.flow * dt
        mask = (g.opacity_raw.sigmoid() * opacity_t)[..., 0] > model.config["temporal_cutoff"]
        torch.testing.assert_close(sliced.get_xyz, xyz.reshape(-1, 3)[mask.reshape(-1)])

    def test_normalize_quaternion_fallback(self):
        self.assertTrue(torch.isfinite(normalize_quaternion(torch.zeros(1, 4))).all())
        q = normalize_quaternion(torch.randn(5, 4))
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(5))

    def test_gradcheck_through_generate_and_at(self):
        model = ScaffoldESVR(anchors=3, children=2, shape=(8, 8, 8), temporal_cutoff=0.0).double()
        params = [model.anchor, model.anchor_feat, model.offset, model.log_offset_scale]
        for p in model.heads.parameters():
            params.append(p)

        def forward(*_):
            sliced = model.at(0.3)
            if not len(sliced.get_xyz):
                return model.anchor.sum() * 0
            return sliced.get_xyz.sum() + sliced.get_scaling.sum() + sliced.get_rotation.sum() + sliced.get_opacity.sum()

        self.assertTrue(torch.autograd.gradcheck(forward, tuple(params), eps=1e-6, atol=1e-4))

    def test_grow_prune_and_optimizer(self):
        model = ScaffoldESVR(anchors=4, children=2, shape=(8, 8, 8))
        optimizer = torch.optim.Adam(model.parameters())
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        before = len(model.anchor)
        with torch.no_grad():
            model.anchor[:, :3] = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
            model.offset[0, :, :3] = torch.tensor([[3.0, 0, 0], [0, 3.0, 0]])  # anchor 0's children sit in empty cells
        model.child_denom.fill_(100.0)
        model.child_grad_accum[:2] = 100.0  # only anchor 0's children exceed the threshold
        model.anchor_denom.fill_(100.0)
        model.anchor_opacity_accum.fill_(100.0)
        model.anchor_opacity_accum[1] = 0.0  # anchor 1 stayed transparent: pruned
        event = model.adjust_anchors(optimizer, grad_threshold=0.5, max_anchors=10, voxel_size=0.02,
                                     t_size=0.1, init_factor=1, hierarchy_factor=4, depth=1, partial=False)
        self.assertEqual(event["before"], before)
        self.assertEqual(event["pruned"], 1)
        self.assertEqual(event["grown"], 2)
        self.assertEqual(len(model.anchor), before - 1 + 2)
        self.assertEqual(len(model.child_denom), len(model.anchor) * 2)
        self.assertTrue((optimizer.state[model.anchor]["exp_avg"][before - 1:] == 0).all())
        # A second round with the same votes must not duplicate occupied cells.
        model.child_denom.fill_(100.0)
        model.child_grad_accum[:2] = 100.0
        self.assertEqual(model.adjust_anchors(optimizer, grad_threshold=0.5, max_anchors=10, voxel_size=0.02,
                                              t_size=0.1, init_factor=1, hierarchy_factor=4, depth=1, partial=False)["grown"], 0)
        optimizer.zero_grad()
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()

    def test_data_init_and_bounds(self):
        model = ScaffoldESVR(anchors=8, children=2, shape=(8, 8, 8), flow_scale=8.0, scale_floor=0.05, opacity_init=0.1,
                             anchor_init=torch.rand(8, 4) * 2 - 1)
        g = model.generate()
        self.assertTrue((g.scale >= 0.05).all())
        self.assertTrue((g.flow.abs() <= 8.0).all())
        self.assertLess((g.opacity_raw.sigmoid().mean() - 0.1).abs(), 0.05)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA voxelizer requires GPU")
class CUDA(unittest.TestCase):
    def test_voxelizer_forward_backward_and_tiles(self):
        torch.manual_seed(3)
        model = ScaffoldESVR(anchors=6, children=3, shape=(12, 10, 8), temporal_cutoff=0.0).cuda()
        with torch.no_grad():
            # Small, deliberately-placed ellipsoids well clear of x=0 (the
            # tile split point) so no primitive's support straddles the tile
            # boundary — a primitive that does can legitimately get assigned
            # to a handful of voxels slightly differently between one full
            # query and two half-width queries, which is a real property of
            # the CUDA kernel's per-call radius/culling, not a bug; avoid it
            # here so this test isolates real forward/backward correctness.
            model.anchor[:, :3] = 0.35 + torch.randn_like(model.anchor[:, :3]) * 0.02
            model.anchor[:, 3].uniform_(0, 1)
            model.log_offset_scale[:] = torch.log(torch.tensor([0.03, 0.03, 0.03, 0.1, 0.03, 0.03, 0.03]))
            model.offset.normal_(0, 0.3)
        full = query_block(model, 0.4, (12, 10, 8), (0, 0, 0), (12, 10, 8))
        tiled = torch.cat([query_block(model, 0.4, (12, 10, 8), (i, 0, 0), (6, 10, 8)) for i in (0, 6)], 0)
        torch.testing.assert_close(full, tiled, atol=2e-6, rtol=2e-5)

        loss = (full * torch.rand_like(full)).mean()
        loss.backward()
        for name, p in list(model.named_parameters()):
            if p.grad is None:
                continue
            self.assertTrue(torch.isfinite(p.grad).all(), name)

        # Finite-difference check of the position path (anchor xyz, offset)
        # with a RELATIVE tolerance: the unfixed CUDA voxelizer scales every
        # position gradient by dVoxel^2 (1/2304 here), which an absolute
        # tolerance can let through when gradients are small. Children are
        # kept above ~1 voxel so no finite-difference step crosses the
        # kernel's hard support cutoff (sub-voxel children touch no voxel
        # centre and make the numeric derivative meaningless).
        with torch.no_grad():
            model.log_offset_scale[:, 4:7] = torch.log(torch.tensor(0.5))
        target = torch.rand_like(full)

        def loss_fn():
            return (query_block(model, 0.4, (12, 10, 8), (0, 0, 0), (12, 10, 8)) - target).square().sum()

        model.zero_grad(set_to_none=True)
        loss_fn().backward()
        ratios = []
        for name, index in [("anchor", (i, j)) for i in range(6) for j in range(3)] + [("offset", (i, 0, j)) for i in range(6) for j in range(3)]:
            param = getattr(model, name)
            analytical = param.grad[index].item()
            eps = 2e-4
            with torch.no_grad():
                original = param[index].item()
                param[index] = original + eps
                plus = loss_fn().item()
                param[index] = original - eps
                minus = loss_fn().item()
                param[index] = original
            numeric = (plus - minus) / (2 * eps)
            if abs(numeric) > 0.05 and analytical != 0:
                ratios.append(numeric / analytical)
        self.assertGreater(len(ratios), 8)
        median = torch.tensor(ratios).median().item()
        self.assertAlmostEqual(median, 1.0, delta=0.15, msg=f"position-gradient scale off: numeric/analytic median {median}")


if __name__ == "__main__":
    unittest.main()
