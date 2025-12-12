from lit_ollama.utils.callbacks import PredictionTimeLogger


class DummyLitAPI:
    pass


def test_prediction_time_logger() -> None:
    cb = PredictionTimeLogger()
    cb.on_before_predict(DummyLitAPI())  # pyright: ignore[reportArgumentType]
    cb._start_time -= 1  # simulate 1 second elapsed
    cb.on_after_predict(DummyLitAPI())  # pyright: ignore[reportArgumentType]
    assert hasattr(cb, "_start_time")
