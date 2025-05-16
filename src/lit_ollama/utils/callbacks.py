import time

import litserve as ls


class PredictionTimeLogger(ls.Callback):
    def on_before_predict(self, lit_api: ls.LitAPI) -> None:
        t0 = time.perf_counter()
        self._start_time = t0

    def on_after_predict(self, lit_api: ls.LitAPI) -> None:
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time
        print(f"Prediction took {elapsed:.2f} seconds", flush=True)
