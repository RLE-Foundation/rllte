import time

class Timer:
    """The calculagraph class.
    
    """
    def __init__(self):
        self._start_time = time.perf_counter()
        self._last_time = time.perf_counter()

    def reset(self):
        elapsed_time = time.perf_counter() - self._last_time
        self._last_time = time.perf_counter()
        total_time = time.perf_counter() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.perf_counter() - self._start_time