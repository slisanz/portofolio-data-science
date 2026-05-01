from src.data import load_raw
from src.features import physics_features, polynomial_pairs
from src.config import FEATURE_COLS


def test_physics_features_columns():
    df = load_raw(nrows=200)
    out = physics_features(df)
    for c in ["osnr_over_nl", "power_x_length", "disp_x_length", "nl_x_power", "log_osnr", "log_power"]:
        assert c in out.columns
    assert len(out) == 200


def test_polynomial_pairs_count():
    df = load_raw(nrows=100)
    out = polynomial_pairs(df[FEATURE_COLS])
    expected = len(FEATURE_COLS) + len(FEATURE_COLS) * (len(FEATURE_COLS) + 1) // 2
    assert out.shape[1] == expected
