# metrics_reporter.py
import time
from collections import defaultdict
from contextlib import nullcontext

class MetricsReporter:
    def __init__(self):
        self._running = defaultdict(list)    # metrics this epoch
        self.history  = defaultdict(list)    # full-run record
        self.last = 0

    def update(self, **kv):
        for k, v in kv.items():
            self._running[k].append(float(v))
            ut = time.time()
            if ut - self.last > 0.1:
                print(f"\r{k}={v:.4f}    ", end="")
                self.last = ut

    def epoch_end(self, epoch: int):
        avg = {k: sum(v) / len(v) for k, v in self._running.items()}
        for k, v in avg.items():
            self.history[k].append(v)
        self._running.clear()
        self._render(epoch, avg)
        return avg

    def _render(self, epoch: int, avg: dict):
        print(f"\repoch {epoch+1} :: " +
              "  ".join(f"{k}={v:.8f}" for k, v in avg.items()))
