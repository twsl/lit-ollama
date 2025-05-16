import types

from lit_ollama.utils.callbacks import PredictionTimeLogger


class DummyLitAPI:
    pass


def test_prediction_time_logger():
    cb = PredictionTimeLogger()
    cb.on_before_predict(DummyLitAPI())
    cb._start_time -= 1  # simulate 1 second elapsed
    cb.on_after_predict(DummyLitAPI())
    assert hasattr(cb, "_start_time")
