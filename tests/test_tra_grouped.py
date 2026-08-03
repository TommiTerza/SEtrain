import unittest

import torch
import torch.nn as nn

from models.gtcrn_end2end import GRNN, GTCRN, TRA


class TestTRAGrouped(unittest.TestCase):
    def _build_model(self, *, tra_grouped: bool) -> GTCRN:
        return GTCRN(
            n_fft=256,
            hop_len=128,
            win_len=256,
            grouped_intra_rnn=True,
            tra_grouped=tra_grouped,
        ).cpu()

    def test_tra_grouped_switch_and_shape_consistency(self) -> None:
        model_standard = self._build_model(tra_grouped=False).eval()
        model_grouped = self._build_model(tra_grouped=True).eval()

        standard_tras = [module for module in model_standard.modules() if isinstance(module, TRA)]
        grouped_tras = [module for module in model_grouped.modules() if isinstance(module, TRA)]

        self.assertTrue(standard_tras)
        self.assertEqual(len(standard_tras), len(grouped_tras))
        self.assertTrue(all(isinstance(module.att_gru, nn.GRU) for module in standard_tras))
        self.assertTrue(all(isinstance(module.att_gru, GRNN) for module in grouped_tras))
        self.assertTrue(all(module.att_gru.grouped for module in grouped_tras))

        x = torch.randn(2, 4096)
        with torch.no_grad():
            y_standard = model_standard(x)
            y_grouped = model_grouped(x)

        self.assertEqual(y_standard.shape, x.shape)
        self.assertEqual(y_grouped.shape, x.shape)
        self.assertEqual(y_standard.shape, y_grouped.shape)


if __name__ == "__main__":
    unittest.main()

