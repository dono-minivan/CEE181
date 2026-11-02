from tmc.interfaces import Observation


def test_missing_speed_is_explicit():
    obs = Observation("det1", 0.0, 30.0, count=12, mean_speed_mps=None)
    assert obs.mean_speed_mps is None
