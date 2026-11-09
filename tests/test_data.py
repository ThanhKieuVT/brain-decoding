import pickle

import numpy as np
import pytest

from brain.data.categories import MULTI_CATEGORY, NO_CATEGORY, assign_categories, load_category27
from brain.data.splits import ConceptSplit, concept_val_split
from brain.data.things_eeg2 import (
    average_repeats,
    concept_id_from_folder,
    convert_subject,
    find_condition_dirs,
    list_image_conditions,
    open_split,
)


def _fake_subject(d, n_train=30, n_test=6, rep_train=4, rep_test=8, ch=5, t=25):
    rng = np.random.default_rng(0)
    times = np.round(np.arange(t) / 25 - 0.2, 4)
    for split, n, r in [("training", n_train, rep_train), ("test", n_test, rep_test)]:
        obj = {
            "preprocessed_eeg_data": rng.standard_normal((n, r, ch, t)),
            "ch_names": [f"C{i}" for i in range(ch)],
            "times": times,
        }
        with open(d / f"preprocessed_eeg_{split}.npy", "wb") as f:
            pickle.dump(obj, f, protocol=4)


def test_convert_and_open(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "conv"
    src.mkdir()
    _fake_subject(src)
    rep = convert_subject(src, dst, dtype="float16", chunk=7)
    assert rep["training"]["shape"] == [30, 4, 5, 25]
    assert rep["test"]["max_abs_err"] < 0.01
    tr = open_split(dst, "training")
    assert tr.data.dtype == np.float16 and tr.data.shape == (30, 4, 5, 25)
    assert isinstance(tr.data, np.memmap)
    sl = tr.time_slice(0.0, 0.6)
    assert sl.times.min() >= 0 and sl.times.max() < 0.6 and sl.data.shape[-1] == len(sl.times)


def test_average_repeats():
    x = np.arange(2 * 4 * 1 * 1, dtype=np.float32).reshape(2, 4, 1, 1)
    np.testing.assert_allclose(average_repeats(x)[:, 0, 0], [1.5, 5.5])
    sub = average_repeats(x, n=2, rng=np.random.default_rng(0))
    assert sub.shape == (2, 1, 1)
    with pytest.raises(ValueError):
        average_repeats(x, n=2)


def test_folder_names(tmp_path):
    for n in ["00002_abacus", "00001_aircraft_carrier"]:
        (tmp_path / n).mkdir()
    assert list_image_conditions(tmp_path) == ["00001_aircraft_carrier", "00002_abacus"]
    assert concept_id_from_folder("00001_aircraft_carrier") == "aircraft_carrier"
    with pytest.raises(ValueError):
        concept_id_from_folder("aircraft")


def test_concept_split(tmp_path):
    s = concept_val_split(seed=3)
    assert len(s.val_concepts) == 74 and len(s.train_concepts) == 1580
    assert not set(s.val_concepts) & set(s.train_concepts)
    vi = s.image_indices("val")
    assert len(vi) == 740 and np.all(vi // 10 == np.repeat(s.val_concepts, 10))
    s.save(tmp_path / "s.json")
    assert ConceptSplit.load(tmp_path / "s.json") == s
    assert concept_val_split(seed=3) == s


def test_categories(tmp_path):
    (tmp_path / "cat.tsv").write_text("animal\tfood\ttool\n1\t0\t0\n0\t1\t1\n0\t0\t0\n1\t0\t0\n")
    (tmp_path / "meta.tsv").write_text(
        "Word\tuniqueID\naardvark\taardvark\napple\tapple\nx\tthing\nbat\tbat1\n"
    )
    names, mapping = load_category27(tmp_path / "cat.tsv", tmp_path / "meta.tsv")
    assert mapping["apple"] == ["food", "tool"]
    labels, summ = assign_categories(["aardvark", "apple", "thing", "bat1"], names, mapping)
    assert labels.tolist() == [0, MULTI_CATEGORY, NO_CATEGORY, 0]
    assert summ["usable_for_shuf_w"] == 2 and summ["multi_category"] == 1
    labels2, _ = assign_categories(["apple"], names, mapping, multi="first")
    assert labels2.tolist() == [1]
    with pytest.raises(KeyError):
        assign_categories(["unknown"], names, mapping)


def test_find_condition_dirs(tmp_path):
    for split, n in [("training_images", 3), ("test_images", 2)]:
        for i in range(n):
            (tmp_path / "x" / split / f"{i:05d}_c{i}").mkdir(parents=True)
    found = find_condition_dirs(tmp_path, expected=(3, 2))
    assert found[3].name == "training_images" and found[2].name == "test_images"
