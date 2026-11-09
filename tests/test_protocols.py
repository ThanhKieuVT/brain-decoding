import numpy as np
import pytest

from brain.protocols import ATM, NICE, atm_time_mask, kway_top1, nice_val_split, prepare_eeg


def test_nice_prepare_means_repeats_and_keeps_all_time():
    rng = np.random.default_rng(0)
    data, times = rng.standard_normal((6, 4, 3, 10)), np.arange(10) / 10
    x, idx = prepare_eeg(data, times, NICE, "train")
    assert x.shape == (6, 3, 10) and x.dtype == np.float32
    np.testing.assert_allclose(x, data.mean(axis=1), atol=1e-6)
    np.testing.assert_array_equal(idx, np.arange(6))


def test_atm_prepare_separates_train_repeats_and_masks_time():
    times = np.round(np.arange(-50, 10) / 10, 3)  # 60 giá trị; times[50:] = 0.0..0.9
    data = np.arange(2 * 4 * 1 * 10, dtype=float).reshape(2, 4, 1, 10)
    x, idx = prepare_eeg(data, times, ATM, "train")
    assert x.shape == (8, 1, 10)
    np.testing.assert_array_equal(idx, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(x[5], data[1, 1])
    xt, _ = prepare_eeg(data, times, ATM, "test")
    assert xt.shape == (2, 1, 10)


def test_atm_time_mask_rejects_length_mismatch():
    with pytest.raises(ValueError, match="Cần kiểm tra"):
        atm_time_mask(np.arange(250) / 250 - 0.2, n_samples=250)
    m = atm_time_mask(np.linspace(-0.2, 1.2, 300), n_samples=250)
    assert m.sum() < 250


def test_nice_val_split():
    tr, va = nice_val_split(seed=1)
    assert len(va) == 740 and len(tr) == 15800 and not set(tr) & set(va)
    tr2, va2 = nice_val_split(seed=1)
    np.testing.assert_array_equal(va, va2)


def test_kway_top1():
    rng = np.random.default_rng(0)
    c = np.eye(50)
    assert kway_top1(c, c, 2, rng).mean() == 1.0
    assert kway_top1(c, c, 50, rng).mean() == 1.0
    c2 = np.eye(200)
    acc = np.mean(
        [kway_top1(rng.standard_normal((200, 200)), c2, 2, rng).mean() for _ in range(20)]
    )
    assert abs(acc - 0.5) < 0.05
