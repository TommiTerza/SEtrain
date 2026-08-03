import unittest

import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from models.gtcrn_end2end import GTCRN


class TestScaleConstraintEnable(unittest.TestCase):
    def _run_smoke(self, device: torch.device) -> None:
        model = GTCRN(
            n_fft=256,
            hop_len=128,
            win_len=256,
            grouped_intra_rnn=True,
        ).to(device)
        model.prepare_qat(
            backend="qnnpack",
            quantize_deconv=True,
            per_channel_weights=True,
            quantize_gru=True,
            scale_constraint_mode="none",
        )

        x = torch.randn(1, 4096, device=device)
        with torch.no_grad():
            y = model(x)

        replaced, updated = model.enable_scale_constraints(mode="pow2", frac_bits=None, pow2_rounding="nearest")
        self.assertGreater(replaced + updated, 0)

        for module in model.modules():
            if not isinstance(module, FakeQuantizeBase):
                continue
            scale = getattr(module, "scale", None)
            if torch.is_tensor(scale):
                self.assertEqual(scale.device, device)

        with torch.no_grad():
            y2 = model(x)
        self.assertEqual(y2.shape, y.shape)

    def test_enable_scale_constraint_keeps_fake_quants_on_device(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device(f"cuda:{torch.cuda.current_device()}"))

        for device in devices:
            with self.subTest(device=str(device)):
                self._run_smoke(device)


if __name__ == "__main__":
    unittest.main()
