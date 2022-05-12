import os
import time
import traceback as tb


class Timer(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        new_time = time.time() - self.time_start
        fname, lineno, method, _ = tb.extract_stack()[-2]  # Get caller
        _, fname = os.path.split(fname)
        id_str = "%s:%s" % (fname, method)
        print(
            "TIMER:%s: %s (Elapsed: %fs)" % (id_str, self.message, new_time)
        )
