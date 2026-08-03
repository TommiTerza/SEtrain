import unittest

import torch

from models.gtcrn_end2end import GTCRN


def _count_quantized_linears(model: torch.nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if (type(module).__name__ == "Linear")
        and type(module).__module__.startswith("torch.ao.nn.quantized")
    )


class TestLinearQuantizationToggle(unittest.TestCase):
    def _build_converted_model(self, quantize_linear: bool) -> torch.nn.Module:
        model = GTCRN(
            n_fft=256,
            hop_len=128,
            win_len=256,
            grouped_intra_rnn=True,
        ).cpu()
        model.prepare_qat(
            backend="qnnpack",
            quantize_deconv=True,
            per_channel_weights=True,
            quantize_linear=quantize_linear,
            quantize_gru=False,
            scale_constraint_mode="none",
        )

        with torch.no_grad():
            _ = model(torch.randn(1, 4096))

        return model.convert_qat(
            inplace=False,
            dynamic_quantize_gru=False,
            static_quantize_gru=False,
        )

    def test_quantize_linear_toggle_controls_int8_linear_conversion(self) -> None:
        converted_disabled = self._build_converted_model(quantize_linear=False)
        converted_enabled = self._build_converted_model(quantize_linear=True)

        self.assertEqual(_count_quantized_linears(converted_disabled), 0)
        self.assertGreater(_count_quantized_linears(converted_enabled), 0)

        with torch.no_grad():
            out_disabled = converted_disabled(torch.randn(1, 4096))
            out_enabled = converted_enabled(torch.randn(1, 4096))

        self.assertEqual(out_disabled.shape, out_enabled.shape)


if __name__ == "__main__":
    unittest.main()

