import csv

import numpy as np

from bench.profile import Profiler
from brain.controls import apply_donor, shuffle_within_category
from brain.tier1 import Condition, evaluate, summarize_prr


def test_tier1_end_to_end():
    rng = np.random.default_rng(0)
    n, d = 120, 16
    groups = np.repeat(np.arange(12), 10)
    centers = rng.standard_normal((12, d)) * 3
    target = centers[groups] + rng.standard_normal((n, d))
    real = target + 0.5 * rng.standard_normal((n, d))
    donor = shuffle_within_category(groups, rng)
    shuf_w, keep = apply_donor(real, donor)
    conds = [
        Condition("real", real),
        Condition("shuf-w", shuf_w, items=keep),
        Condition("null-0", np.zeros_like(real) + 1e-6 * rng.standard_normal((n, d))),
    ]
    ids = [f"img{i}" for i in range(n)]
    df = evaluate(
        conds, target, groups, ids, metrics=["twoway_cosine", "twoway_within_cat", "top1"]
    )
    assert set(df.columns) == {"image_id", "condition", "k", "metric", "value", "seed"}
    summ = summarize_prr(df, n_items=n, n_boot=200).set_index(["condition", "metric"])
    assert np.isclose(summ.loc[("real", "twoway_cosine"), "prr"], 1.0)
    # shuf-w giữ category: two-way chung vẫn cao, two-way trong category về ngẫu nhiên
    assert summ.loc[("shuf-w", "twoway_cosine"), "prr"] > 0.6
    assert abs(summ.loc[("shuf-w", "twoway_within_cat"), "prr"]) < 0.25
    assert abs(summ.loc[("null-0", "top1"), "prr"]) < 0.1


def test_profiler_writes_row(tmp_path):
    p = tmp_path / "profile.csv"
    with Profiler("t", n_items=10, decoder="d", csv_path=p, data_dir=tmp_path):
        sum(range(100000))
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 1 and float(rows[0]["peak_ram_gb"]) > 0 and rows[0]["task"] == "t"
