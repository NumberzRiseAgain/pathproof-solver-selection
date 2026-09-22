"""
Loader and feature extraction.

Every test here corresponds to a defect that actually occurred and cost time.
The label parser was backwards, the sampling collapsed to one class, and
missing data was silently turned into zeros that carried the label. These are
regression tests, not decoration.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swan_io  # noqa: E402
from features import DESCRIPTORS, TrainOnlyScaler, extract  # noqa: E402


# --------------------------------------------------------------------------
# label parsing - the defect that would have labelled all 331,185 slices quiet
# --------------------------------------------------------------------------

def test_label_parser_on_real_swansf_filenames():
    cases = {
        "FQ_ar5742_s2015-07-05T08:36:00_e2015-07-05T20:24:00.csv": "Q",
        "M1.0_ar1234_s2011-02-13T00:00:00_e2011-02-13T12:00:00.csv": "M",
        "M9.3_ar1476_s2011-08-04T03:12:00_e2011-08-04T15:00:00.csv": "M",
        "X2.2_ar1520_s2012-07-12T00:00:00_e2012-07-12T12:00:00.csv": "X",
        "C1.5_ar999_s2013-01-01T00:00:00_e2013-01-01T12:00:00.csv": "C",
        "B3.1_ar888_s2014-01-01T00:00:00_e2014-01-01T12:00:00.csv": "B",
    }
    for name, want in cases.items():
        assert swan_io._label_from_name(name) == want, name


def test_label_parser_flags_unrecognised_rather_than_defaulting_to_quiet():
    """
    A name it cannot read must return '?' and not silently become a negative.
    Defaulting an unparsed name to flare-quiet is how a whole partition ends up
    labelled 0% positive while every line of output looks normal.
    """
    assert swan_io._label_from_name("garbage_ar1_s2011.csv") == "?"


def test_positive_classes_are_m_and_x_only():
    assert set(swan_io.POSITIVE_CLASSES) == {"M", "X"}
    for cls in ("C", "B", "Q"):
        assert cls not in swan_io.POSITIVE_CLASSES


def test_separator_sniffer(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    tabbed = d / "t.csv"
    tabbed.write_text("A\tB\tC\n1\t2\t3\n")
    commad = d / "c.csv"
    commad.write_text("A,B,C\n1,2,3\n")
    assert swan_io._sniff_sep(tabbed) == "\t"
    assert swan_io._sniff_sep(commad) == ","


def test_cache_schema_is_versioned():
    """A cache without a schema version silently returns stale columns after a
    loader change, which makes a fix appear to do nothing."""
    assert isinstance(swan_io.CACHE_SCHEMA, int) and swan_io.CACHE_SCHEMA >= 2


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def test_extract_shapes_and_names():
    X = np.random.default_rng(0).normal(size=(20, 60, 24))
    F, names = extract(X, mode="mvts", params=swan_io.CORE_PARAMS)
    assert F.shape == (20, 24 * len(DESCRIPTORS))
    assert len(names) == F.shape[1]
    assert "TOTUSJH_last" in names and "R_VALUE_slope" in names


def test_extract_output_is_always_finite():
    """All-NaN columns occur in real SWAN-SF slices. They must not propagate."""
    X = np.full((5, 60, 24), np.nan)
    X[:, :, 0] = 1.0
    F, _ = extract(X, mode="mvts", params=swan_io.CORE_PARAMS)
    assert np.isfinite(F).all()


def test_slope_descriptor_recovers_a_known_ramp():
    n, T, p = 3, 60, 24
    X = np.zeros((n, T, p))
    X[:, :, 0] = np.arange(T) * 2.0          # slope exactly 2 per timestep
    F, names = extract(X, mode="mvts", params=swan_io.CORE_PARAMS)
    got = F[:, names.index("TOTUSJH_slope")]
    assert np.allclose(got, 2.0), got


def test_delta_and_last_descriptors():
    X = np.zeros((2, 60, 24))
    X[:, :, 0] = np.linspace(5.0, 15.0, 60)
    F, names = extract(X, mode="mvts", params=swan_io.CORE_PARAMS)
    assert np.allclose(F[:, names.index("TOTUSJH_last")], 15.0)
    assert np.allclose(F[:, names.index("TOTUSJH_delta")], 10.0)


def test_vector_mode_takes_the_last_observed_value_skipping_nans():
    X = np.zeros((1, 60, 24))
    X[0, :, 0] = 3.0
    X[0, -5:, 0] = np.nan                     # tail missing
    F, names = extract(X, mode="vector", params=swan_io.CORE_PARAMS)
    assert np.isclose(F[0, names.index("TOTUSJH_last")], 3.0)


def test_scaler_is_fit_on_train_rows_only():
    """
    The single most common source of inflated SWAN-SF numbers is fitting
    normalisation on all rows. The scaler must expose statistics from the rows
    it was given and nothing else.
    """
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 1.0, size=(200, 10))
    held_out = rng.normal(50.0, 1.0, size=(200, 10))       # wildly different
    sc = TrainOnlyScaler().fit(train)
    med_before = sc.med_.copy()
    sc.transform(held_out)
    assert np.array_equal(sc.med_, med_before)
    assert abs(np.median(sc.transform(train))) < 0.2       # train centred
    assert np.median(sc.transform(held_out)) > 5.0         # held-out is not


def test_scaler_clips_extreme_values():
    sc = TrainOnlyScaler().fit(np.random.default_rng(0).normal(size=(100, 4)))
    out = sc.transform(np.full((3, 4), 1e12))
    assert out.max() <= 20.0
