import tempfile
import unittest
from pathlib import Path

from local_robot.scripted_train import build_config, TrainingAudit


class TrainTests(unittest.TestCase):
    def test_resume_preserves_architecture_and_adds_one_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = build_config(root)
            self.assertEqual((cfg.model.hidden_size, cfg.model.depth, cfg.model.num_heads), (256, 4, 8))
            self.assertEqual(cfg.model.camera_keys, ('top', 'left', 'right'))
            self.assertEqual(cfg.model.chunk_length, 30)
            self.assertFalse(cfg.compile)
            self.assertEqual(cfg.batch_size, 1)
            resume = build_config(root, resume=True)
            self.assertEqual(resume.train_steps, 201)
            self.assertEqual(Path(resume.resume_from), root/'checkpoints/200.pt')

    def test_audit_rejects_nonfinite_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is tested separately in the training environment')
        audit = TrainingAudit()
        p = torch.nn.Parameter(torch.ones(2))
        p.grad = torch.tensor([float('nan'), 0.])
        with self.assertRaises(FloatingPointError):
            audit.check_gradients([p])


if __name__ == '__main__':
    unittest.main()
