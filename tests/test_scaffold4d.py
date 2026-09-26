import math
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
        # ...and decays strictly monotonically as |query time - home time| grows
        # (checked on the gate values themselves, not on cutoff counts).
        gates = [torch.exp(-((g.home_t - (g.home_t + dt)) * g.sigma).abs().pow(model.config["temporal_beta"])) for dt in (0.0, 0.05, 0.2, 0.6)]
        for a, b in zip(gates, gates[1:]):
            self.assertTrue(torch.all(b < a))
        near = model.at((home_t + 0.1).item(), generated=g)
        far = model.at((home_t + 5.0).item(), generated=g)
        self.assertLessEqual(len(far.get_xyz), len(near.get_xyz))

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

    def test_prune_dead_anchors(self):
        model = ScaffoldESVR(anchors=3, children=2, shape=(8, 8, 8))
        optimizer = torch.optim.Adam(model.parameters())
        model.anchor_denom.fill_(100.0)
        model.anchor_opacity_accum.fill_(100.0)  # upstream min_opacity rule never fires
        model.child_seen.fill_(100.0)
        model.child_effective.fill_(100.0)
        model.child_effective[2:4] = 0.0  # anchor 1: seen often, never covered a voxel
        model.child_seen[4:6] = 0.0  # anchor 2: never seen -> not judged
        event = model.adjust_anchors(optimizer, grad_threshold=1e9, max_anchors=3, prune_dead=True)
        self.assertEqual(event["pruned_dead"], 1)
        self.assertEqual(len(model.anchor), 2)
        self.assertEqual(len(model.child_seen), 4)
        self.assertEqual(model.adjust_anchors(optimizer, grad_threshold=1e9, max_anchors=3, prune_dead=False)["pruned"], 0)

    def test_upstream_level_skip(self):
        def voting_model():
            # Every child votes, and sits one fine cell (0.02) from its anchor:
            # the coarse (0.32) and middle (0.08) cells are already occupied by
            # the anchor itself, only the fine level has free cells.
            model = ScaffoldESVR(anchors=3, children=2, shape=(8, 8, 8))
            optimizer = torch.optim.Adam(model.parameters())
            model.child_denom.fill_(100.0)
            model.child_grad_accum.fill_(1000.0)  # mean gradient 10 > threshold x4 at the finest level
            with torch.no_grad():
                model.anchor[:, :3] = 0.0
                model.anchor[:, 3] = 0.5
                model.log_offset_scale[:, :3] = torch.log(torch.tensor(0.02))
                model.offset[..., :3] = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
                model.offset[..., 3] = 0.0
            return model, optimizer
        kwargs = dict(grad_threshold=0.5, max_anchors=50, voxel_size=0.02, t_size=0.1, init_factor=16, hierarchy_factor=4, depth=3, partial=False)
        model, optimizer = voting_model()
        self.assertGreater(model.adjust_anchors(optimizer, **kwargs)["grown"], 0)
        model, optimizer = voting_model()
        # Upstream: the coarse level adds nothing, so the finer levels are skipped.
        self.assertEqual(model.adjust_anchors(optimizer, upstream_level_skip=True, **kwargs)["grown"], 0)

    def test_compact_roundtrip(self):
        import tempfile
        from scaffold4d.compact import build_compact, load_compact, save_compact
        model = ScaffoldESVR(anchors=5, children=3, shape=(8, 8, 8), temporal_cutoff=0.0)
        with torch.no_grad():
            model.anchor_feat.normal_()
            model.offset.normal_()
        keep = torch.tensor([[False, False, False],   # anchor 0 dropped entirely
                             [True, False, False],
                             [True, True, True],
                             [False, True, False],
                             [True, False, True]])
        compact = build_compact(model, keep)
        self.assertEqual(compact.config["anchors"], 4)
        with tempfile.TemporaryDirectory() as d:
            save_compact(compact, f"{d}/c.pt", manifest={"frames": []}, source_args={"steps": 1})
            payload = torch.load(f"{d}/c.pt", weights_only=False)
            loaded = load_compact(payload, device="cpu")
        full = model.at(0.3)
        kept_flat = keep.reshape(-1)[full.child_index]
        for m in (compact, loaded):
            s = m.at(0.3)
            torch.testing.assert_close(s.get_xyz, full.get_xyz[kept_flat])
            torch.testing.assert_close(s.get_opacity, full.get_opacity[kept_flat])
            torch.testing.assert_close(s.get_scaling, full.get_scaling[kept_flat])

    def test_tanh_activation_masks_negative_children(self):
        model = ScaffoldESVR(anchors=4, children=3, shape=(8, 8, 8), temporal_cutoff=0.0, opacity_activation="tanh", opacity_init=0.5)
        with torch.no_grad():
            model.heads["opacity"][-1].bias.copy_(torch.tensor([0.6, -0.6, 0.3]))  # child 1 of every anchor is switched off
        g = model.generate()
        negative = (g.intensity[..., 0] <= 0)
        self.assertTrue(negative.any() and (~negative).any())
        s = model.at(0.5, g)
        self.assertTrue((s.get_opacity > 0).all())
        self.assertEqual(len(s.child_index), int((~negative).reshape(-1).sum()))
        self.assertTrue((s.anchor_opacity_sum >= 0).all())

    def test_error_table_sampling(self):
        import random
        from scaffold4d.sampling import ErrorTable
        table = ErrorTable(n_frames=3, shape=(4, 4, 4), block=2, alpha=1.0, uniform=0.2)
        self.assertEqual(table.n_tiles, 8)
        frames = [0, 2]
        p = table.probabilities(frames)
        self.assertEqual(len(p), 16)
        torch.testing.assert_close(torch.tensor(p), torch.full((16,), 1 / 16, dtype=torch.float64))  # nothing visited: uniform
        start = table.tile_start(5)
        self.assertEqual(table.tile_of(start), 5)
        table.update(2, start, 1.0)      # one cell seen, large error
        table.update(0, (0, 0, 0), 0.0)  # one cell seen, perfect fit
        p = table.probabilities(frames).reshape(2, 8)
        # unvisited cells count as the max error (1.0); the zero-error cell keeps only the uniform share
        self.assertAlmostEqual(p[0, 0], 0.2 / 16)
        self.assertAlmostEqual(p[1, 5], 0.8 / 15 + 0.2 / 16)
        self.assertAlmostEqual(p.sum(), 1.0)
        rng = random.Random(0)
        picks = [table.sample(frames, rng) for _ in range(200)]
        self.assertTrue(all(f in frames for f, _ in picks))
        self.assertTrue(all(all(a % 2 == 0 and 0 <= a < 4 for a in s) for _, s in picks))
        self.assertLess(sum(1 for f, s in picks if f == 0 and s == (0, 0, 0)), 15)  # ~1.25 % expected vs 6.25 % uniform
        restored = ErrorTable(3, (4, 4, 4), 2, alpha=1.0, uniform=0.2, state=table.state())
        torch.testing.assert_close(torch.tensor(restored.probabilities(frames)), torch.tensor(table.probabilities(frames)))
        with self.assertRaises(ValueError):
            ErrorTable(2, (4, 4, 4), 2, state=table.state())

    def test_statistics_state_roundtrip_and_shape_check(self):
        model = ScaffoldESVR(anchors=4, children=3, shape=(8, 8, 8))
        for name in model.statistic_buffers + model.anchor_statistic_buffers:
            getattr(model, name).uniform_(1, 5)
        state = model.statistics_state()
        fresh = ScaffoldESVR(anchors=4, children=3, shape=(8, 8, 8))
        self.assertTrue((fresh.child_denom == 0).all())
        fresh.load_statistics_state(state)
        for name in model.statistic_buffers + model.anchor_statistic_buffers:
            torch.testing.assert_close(getattr(fresh, name), getattr(model, name))
        with self.assertRaises(ValueError):
            ScaffoldESVR(anchors=5, children=3, shape=(8, 8, 8)).load_statistics_state(state)

    def test_record_statistics_effective_uses_voxel_counts(self):
        model = ScaffoldESVR(anchors=2, children=2, shape=(8, 8, 8), temporal_cutoff=0.0)
        sliced = model.at(0.5)
        (sliced.get_xyz.sum() * 0.0 + sliced.get_xyz[:, 0].sum()).backward()  # nonzero position gradient for every child
        visibility = torch.ones(len(sliced.child_index), dtype=torch.bool)
        # Gradient proxy: every child looks effective.
        model.record_statistics(sliced, visibility)
        self.assertTrue((model.child_effective == 1).all())
        # Voxel counts say only the first child touched a voxel.
        counts = torch.zeros(len(sliced.child_index), dtype=torch.long)
        counts[0] = 7
        model.record_statistics(sliced, visibility, voxel_counts=counts)
        expected = torch.ones(4)
        expected[sliced.child_index[0]] += 1
        torch.testing.assert_close(model.child_effective, expected)
        self.assertTrue((model.child_seen == 2).all())

    def test_compact_rejects_empty_keep(self):
        from scaffold4d.compact import build_compact
        model = ScaffoldESVR(anchors=3, children=2, shape=(8, 8, 8))
        with self.assertRaises(ValueError):
            build_compact(model, torch.zeros(3, 2, dtype=torch.bool))

    def test_scale_floor_passes_gradient_below_floor(self):
        # Forward is clamped to the floor; the gradient must still reach the
        # parameters of a child whose pre-clamp size is below it (a plain
        # clamp_min would zero it and trap the child there forever).
        model = ScaffoldESVR(anchors=2, children=2, shape=(8, 8, 8), scale_floor=0.3)
        with torch.no_grad():
            model.log_offset_scale[:, 4:7] = math.log(0.01)   # pre-clamp child scale ~0.005 << floor
        g = model.generate()
        self.assertTrue((g.scale >= 0.3).all())
        g.scale.sum().backward()
        self.assertTrue((model.log_offset_scale.grad[:, 4:7] != 0).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads["shape"].parameters()))

    def test_residual_growth_adds_anchors_in_empty_cells_with_nearest_feature(self):
        model = ScaffoldESVR(anchors=3, children=2, shape=(16, 16, 16), temporal_cutoff=0.0)
        with torch.no_grad():
            model.anchor[:, :3] = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
            model.anchor[:, 3] = torch.tensor([0.1, 0.9, 0.5])
            model.anchor_feat[:] = torch.arange(3.0)[:, None]
        optimizer = torch.optim.Adam([{"params": [getattr(model, n)]} for n in model.population_params], lr=1e-3)
        voxel = 2 / 16
        # Two candidates: one on an occupied cell (anchor 1's cell), one in an empty cell near anchor 0.
        residual4d = torch.tensor([[0.5, 0.5, 0.5, 0.9], [-0.5, -0.5, -0.5 + 4 * voxel, 0.1]])
        weight = torch.tensor([1.0, 0.5])
        event = model.adjust_anchors(optimizer, check_interval=100, grad_threshold=1e9, max_anchors=10, voxel_size=voxel, t_size=0.1,
                                     init_factor=16, hierarchy_factor=4, depth=3, partial=False,
                                     residual4d=residual4d, residual_weight=weight, residual_budget=5)
        self.assertEqual(event["grown_residual"], 1)
        self.assertEqual(len(model.anchor), 4)
        torch.testing.assert_close(model.anchor_feat[3], model.anchor_feat[0])   # nearest anchor's feature
        self.assertTrue(((model.anchor[3, :3] - residual4d[1, :3]).abs() <= voxel).all())
        self.assertAlmostEqual(float(model.anchor[3, 3]), 0.1, places=5)

    def test_score_pruning_removes_lowest_scored_checked_anchors(self):
        model = ScaffoldESVR(anchors=10, children=2, shape=(8, 8, 8), temporal_cutoff=0.0)
        optimizer = torch.optim.Adam([{"params": [getattr(model, n)]} for n in model.population_params], lr=1e-3)
        with torch.no_grad():
            model.anchor_denom[:] = 1000          # every anchor checked
            model.anchor_opacity_accum[:] = 1000  # nobody fails min_opacity
            cov = torch.zeros(20)
            cov[::2] = torch.arange(10.0) + 1     # anchor i has coverage i+1 on its first child
            model.child_coverage[:] = cov
        event = model.adjust_anchors(optimizer, check_interval=100, grad_threshold=1e9, max_anchors=10, partial=False, score_prune_ratio=0.3)
        self.assertEqual(event["pruned_score"], 3)
        self.assertEqual(len(model.anchor), 7)
        # Without coverage statistics the score is zero for all and nothing is removed twice
        self.assertTrue((model.child_coverage == 0).all())

    def test_data_init_and_bounds(self):
        model = ScaffoldESVR(anchors=8, children=2, shape=(8, 8, 8), flow_scale=8.0, scale_floor=0.05, opacity_init=0.1,
                             anchor_init=torch.rand(8, 4) * 2 - 1)
        g = model.generate()
        self.assertTrue((g.scale >= 0.05).all())
        self.assertTrue((g.flow.abs() <= 8.0).all())
        self.assertLess((g.intensity.mean() - 0.1).abs(), 0.05)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA voxelizer requires GPU")
class CUDA(unittest.TestCase):
    def test_normalized_composition_reproduces_a_constant_field(self):
        # With every child's value equal to c, the weighted average is c wherever
        # the weights cover the voxel -- independent of how many children overlap.
        torch.manual_seed(0)
        model = ScaffoldESVR(anchors=40, children=3, shape=(16, 16, 16), temporal_cutoff=0.0, composition="normalized", value_init=0.37).cuda()
        with torch.no_grad():
            model.anchor[:, :3].uniform_(-0.8, 0.8)
            model.log_offset_scale[:, 4:7] = math.log(0.4)
            model.heads["value"][-1].weight.zero_()   # every slot outputs sigmoid(bias) = value_init
        pkg, _ = query_block(model, 0.5, (16, 16, 16), (0, 0, 0), (16, 16, 16), return_package=True)
        covered = pkg["den"] > model.config["norm_eps"] * model.config["norm_C"]
        self.assertGreater(int(covered.sum()), 100)
        torch.testing.assert_close(pkg["vol"][covered], torch.full_like(pkg["vol"][covered], 0.37), atol=1e-4, rtol=0)
        # gradients reach positions, weights and values
        loss = (pkg["vol"] - 0.5).abs().mean()
        loss.backward()
        self.assertTrue(model.heads["value"][-1].weight.grad.abs().sum() > 0)
        self.assertTrue(model.heads["opacity"][-1].weight.grad is not None)
        self.assertTrue(model.offset.grad.abs().sum() > 0)

    def test_count_children_matches_voxel_contribution(self):
        from scaffold4d.model import count_children
        torch.manual_seed(1)
        model = ScaffoldESVR(anchors=6, children=3, shape=(12, 10, 8), temporal_cutoff=0.0).cuda()
        with torch.no_grad():
            model.anchor[:, :3] = torch.rand_like(model.anchor[:, :3]) * 1.2 - 0.6
            # child scale ~0.25 (1.5 voxels): most children touch voxels, some
            # small/edge ones do not, so both count classes occur.
            model.log_offset_scale[:] = torch.log(torch.tensor([0.05, 0.05, 0.05, 0.1, 0.25, 0.25, 0.25]))
        sliced = model.at(0.4)
        counts = count_children(sliced, (12, 10, 8), (0, 0, 0), (12, 10, 8))
        self.assertEqual(counts.shape, (len(sliced.child_index),))
        self.assertGreater(int((counts > 0).sum()), 0)
        # A child removed from the population changes the volume iff it had a nonzero count.
        full = query_block(model, 0.4, (12, 10, 8), (0, 0, 0), (12, 10, 8)).detach()
        zero = (counts == 0).nonzero().flatten()
        if len(zero):
            sliced.get_opacity = sliced.get_opacity.detach().clone()
            sliced.get_opacity[zero] = 0.0
            from gaussian_renderer import query
            from types import SimpleNamespace as NS
            step = 2 / 12
            center = [(n / 2 - s / 2) * step for n, s in zip((12, 10, 8), (12, 10, 8))]
            without = query(NS(get_xyz=sliced.get_xyz.detach(), get_opacity=sliced.get_opacity, get_scaling=sliced.get_scaling.detach(), get_rotation=sliced.get_rotation.detach()),
                            center, (12, 10, 8), [s * step for s in (12, 10, 8)], NS(compute_cov3D_python=False, debug=False))["vol"]
            torch.testing.assert_close(full, without, atol=1e-6, rtol=1e-5)
        # All-dark population: nothing is counted.
        with torch.no_grad():
            model.heads["opacity"][-1].bias.fill_(-20.0)
        dark = model.at(0.4)
        if len(dark.child_index):
            self.assertEqual(int(count_children(dark, (12, 10, 8), (0, 0, 0), (12, 10, 8)).sum()), 0)

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
