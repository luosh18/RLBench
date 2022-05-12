import logging
import os
import time
import traceback as tb

from workspace.learn.flags import FLAGS


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


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return False
    return True


def get_logger(name=None):
    logger = logging.getLogger(name)
    save_dir = FLAGS.save_dir
    check_and_make(save_dir)
    fh = logging.FileHandler(os.path.join(save_dir, str(logger.name) + '.log'))
    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d]| %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
