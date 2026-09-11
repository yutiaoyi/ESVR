import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dynamic_esvr.data import VolumeSequence
from dynamic_esvr.model import DynamicESVR, normalize_quaternion, query_block


class DynamicTests(unittest.TestCase):
    def test_identity_and_gradients(self):
        model = DynamicESVR(8, [16, 12, 8], depth=4, width=16, xyz_bands=2, time_bands=2, scalar=True)
        a, b = model.at(0), model.at(0.8)
        torch.testing.assert_close(a.get_xyz, model.xyz)
        torch.testing.assert_close(a.get_xyz, b.get_xyz)
        torch.testing.assert_close(a.get_rotation.norm(dim=-1), torch.ones(8))
        loss = a.get_xyz.sum() + a.get_scaling.sum() + a.get_rotation.sum() + a.get_opacity.sum()
        loss.backward()
        for head in model.deform.heads.values():
            self.assertGreater(head.bias.grad.abs().sum().item(), 0)
        self.assertTrue((a.get_scaling > 0).all())
        self.assertTrue(torch.isfinite(normalize_quaternion(torch.zeros(1, 4))).all())

    def test_data_axes_and_common_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xyz = np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 3, 4)
            xyz.transpose(2, 1, 0).tofile(root / "a.raw")
            manifest = dict(shape_xyz=[2, 3, 4], storage_axes="zyx", dtype="<f4", intensity_range=[-1, 1],
                            frames=[dict(path="a.raw", time=10), dict(path="a.raw", time=20, split="test")])
            path = root / "sequence.json"
            path.write_text(json.dumps(manifest))
            seq = VolumeSequence(path)
            np.testing.assert_allclose(seq.volumes[0].numpy(), (xyz + 1) / 2)
            self.assertEqual(seq.times, [0, 1])
            self.assertEqual(seq.train_indices, [0])
            manifest["shape_xyz"] = [3, 3, 4]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "byte count"):
                VolumeSequence(path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_voxelizer_deformation_and_chunking(self):
        torch.manual_seed(4)
        model = DynamicESVR(12, [16, 12, 8], depth=4, width=32, xyz_bands=2, time_bands=2, scalar=True).cuda()
        with torch.no_grad():
            model.log_scale[:, 0] -= 0.4  # Rotation gradients need anisotropy.
        full = query_block(model, 0.4, [16, 12, 8])
        block = query_block(model, 0.4, [16, 12, 8], [8, 0, 0], [8, 12, 8])
        torch.testing.assert_close(block, full[8:], rtol=1e-4, atol=1e-5)
        loss = (full * torch.rand_like(full)).mean()
        loss.backward()
        for name, head in model.deform.heads.items():
            self.assertTrue(torch.isfinite(head.weight.grad).all(), name)
            self.assertGreater(head.weight.grad.abs().sum().item(), 0, name)
        # Reloading preserves arbitrary-time output exactly.
        replica = DynamicESVR(**model.config).cuda()
        replica.load_state_dict(model.state_dict())
        torch.testing.assert_close(query_block(replica, 0.4, [16, 12, 8]), full)


if __name__ == "__main__":
    unittest.main()
