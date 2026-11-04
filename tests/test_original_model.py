"""
test_original_model
===================
Unit tests for GatingMechanism and MSWTNetwork.

Run:
    pytest tests/test_original_model.py -v
"""
import numpy as np
import pytest
import pywt
import torch
from src.models.gating_mechanism import GatingMechanism
from src.models.original_model import (
    MSWTNetwork,
    _dwt_window,
    min_window_length,
    validate_even_look_back,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_network(J=2, look_back=28, hidden=16, num_layers=1,
                  dropout=0.0, gating_dim=32, gating_hidden_layers=2):
    return MSWTNetwork(
        J=J,
        look_back=look_back,
        wavelet="db4",
        hidden_size=hidden,
        num_layers=num_layers,
        dropout=dropout,
        gating_dim=gating_dim,
        gating_hidden_layers=gating_hidden_layers,
    )


def _eda_reconstruction(series, wavelet, level):
    """Exact replica of the EDA reconstruction (symmetric, truncate)."""
    coeffs = pywt.wavedec(series, wavelet, level=level)
    n = len(series)
    bands = []
    for k in range(len(coeffs)):
        c_zero = [np.zeros_like(c) for c in coeffs]
        c_zero[k] = coeffs[k]
        rec = pywt.waverec(c_zero, wavelet)[:n]
        bands.append(rec)
    return np.stack(bands, axis=-1)


# ---------------------------------------------------------------------------
# min_window_length
# ---------------------------------------------------------------------------
class TestMinWindowLength:
    def test_db4_known_values(self):
        assert min_window_length("db4", 1) == 14
        assert min_window_length("db4", 2) == 28
        assert min_window_length("db4", 3) == 56
        assert min_window_length("db4", 5) == 224

    def test_haar_shorter_than_db4(self):
        assert min_window_length("haar", 2) < min_window_length("db4", 2)


# ---------------------------------------------------------------------------
# validate_even_look_back
# ---------------------------------------------------------------------------
class TestEvenLookBackValidation:
    @pytest.mark.parametrize("lb", [14, 28, 56, 112, 224])
    def test_even_accepted(self, lb):
        validate_even_look_back(lb)

    @pytest.mark.parametrize("lb", [7, 13, 21, 55, 111])
    def test_odd_rejected(self, lb):
        with pytest.raises(ValueError, match="must be even"):
            validate_even_look_back(lb)


# ---------------------------------------------------------------------------
# DWT pipeline (symmetric mode, mirrors EDA)
# ---------------------------------------------------------------------------
class TestDWTPipeline:
    @pytest.mark.parametrize("J", [1, 2, 3])
    def test_matches_eda_exactly(self, J):
        look_back = min_window_length("db4", J) * 2
        rng = np.random.default_rng(0)
        window = rng.standard_normal(look_back).astype(np.float64)

        model_out = _dwt_window(window, wavelet="db4", J=J)
        eda_out = _eda_reconstruction(window, "db4", J)
        np.testing.assert_allclose(model_out, eda_out, atol=1e-10)

    @pytest.mark.parametrize("J", [1, 2, 3])
    def test_sum_of_bands_reconstructs_signal(self, J):
        look_back = min_window_length("db4", J) * 2
        rng = np.random.default_rng(1)
        window = rng.standard_normal(look_back).astype(np.float64)

        bands = _dwt_window(window, "db4", J)
        assert bands.shape == (look_back, J + 1)
        np.testing.assert_allclose(bands.sum(axis=-1), window, atol=1e-10)


# ---------------------------------------------------------------------------
# GatingMechanism
# ---------------------------------------------------------------------------
class TestGatingMechanism:
    def test_output_shape(self):
        g = GatingMechanism(input_dim=48, output_dim=3)
        out = g(torch.randn(4, 48))
        assert out.shape == (4, 3)

    def test_weights_sum_to_one(self):
        g = GatingMechanism(input_dim=48, output_dim=3)
        out = g(torch.randn(4, 48))
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(4))

    def test_weights_non_negative(self):
        g = GatingMechanism(input_dim=48, output_dim=3)
        out = g(torch.randn(4, 48))
        assert (out >= 0).all()

    def test_num_hidden_layers_validation(self):
        with pytest.raises(ValueError, match="num_hidden_layers"):
            GatingMechanism(input_dim=48, output_dim=3, num_hidden_layers=1)

    def test_initial_dim_validation(self):
        with pytest.raises(ValueError, match="initial_dim"):
            GatingMechanism(input_dim=48, output_dim=3, initial_dim=8)

    def test_dropout_validation(self):
        with pytest.raises(ValueError, match="dropout"):
            GatingMechanism(input_dim=48, output_dim=3, dropout=0.9)


# ---------------------------------------------------------------------------
# MSWTNetwork
# ---------------------------------------------------------------------------
class TestMSWTNetwork:
    def test_forward_shape(self):
        net = _make_network(J=2, look_back=28)
        out = net(torch.randn(4, 28, 1))
        assert out.shape == (4, 1)

    def test_output_strictly_positive(self):
        net = _make_network(J=2, look_back=28)
        out = net(torch.randn(4, 28, 1))
        assert (out > 0).all()

    def test_odd_look_back_rejected(self):
        with pytest.raises(ValueError, match="must be even"):
            _make_network(J=2, look_back=27)

    def test_look_back_insufficient_rejected(self):
        with pytest.raises(ValueError, match="insufficient"):
            _make_network(J=2, look_back=14)

    def test_look_back_exactly_minimum(self):
        net = _make_network(J=2, look_back=28)
        out = net(torch.randn(2, 28, 1))
        assert out.shape == (2, 1)

    @pytest.mark.parametrize("J", [1, 2, 3])
    def test_num_lstms_equals_J_plus_one(self, J):
        look_back = min_window_length("db4", J) * 2
        net = _make_network(J=J, look_back=look_back)
        assert len(net.lstms) == J + 1
        assert len(net.heads) == J + 1

    def test_gradients_flow(self):
        net = _make_network(J=2, look_back=28)
        x = torch.randn(4, 28, 1)
        y = net(x)
        loss = y.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net.parameters()
        )
        assert has_grad, "No parameter received a gradient."

    def test_eval_mode_deterministic(self):
        net = _make_network(J=2, look_back=28)
        net.eval()
        x = torch.randn(2, 28, 1)
        with torch.no_grad():
            a = net(x)
            b = net(x)
        torch.testing.assert_close(a, b)

    def test_lstm_dropout_zero_when_single_layer(self):
        net = _make_network(J=1, look_back=14, num_layers=1, dropout=0.3)
        assert net.lstms[0].dropout == 0.0

    def test_lstm_dropout_active_when_multi_layer(self):
        net = _make_network(J=1, look_back=14, num_layers=2, dropout=0.3)
        assert net.lstms[0].dropout == 0.3