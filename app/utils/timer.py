import time


class Timer:
    """
    Utility class for measuring execution latency.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        self.end_time = time.perf_counter()

        return self.elapsed_time()

    def elapsed_time(self):
        if self.start_time is None:
            return 0.0

        end = self.end_time or time.perf_counter()

        return end - self.start_time