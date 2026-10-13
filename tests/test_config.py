from scape2.config import DEFAULTS, validate


def test_defaults_validate():
    assert validate(dict(DEFAULTS))["n_max_apa"] == 5


def test_invalid_k_range():
    options = dict(DEFAULTS, n_min_apa=3, n_max_apa=2)
    try:
        validate(options)
    except ValueError:
        return
    raise AssertionError("invalid K range was accepted")
