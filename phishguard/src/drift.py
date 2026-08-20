"""
PhishGuard :: concept drift detection ("self-healing" layer)
---------------------------------------------------------------
Implements a from-scratch Drift Detection Method (DDM), the classic
statistical-process-control approach to streaming concept drift
(Gama et al., 2004). `river`/`skmultiflow` aren't installed in this
offline environment, so this is a small, dependency-free reimplementation
covering the same idea the survey's Direction 1 calls for: watch the
live error rate and its standard deviation, and fire a retraining
trigger only when the distribution shifts significantly -- not on a
fixed schedule.

How it works
------------
For each labeled prediction seen in the stream, track the running error
rate p_i and its standard deviation s_i = sqrt(p_i * (1 - p_i) / i).
Remember the minimum of (p_i + s_i) seen so far: (p_min, s_min).

  - WARNING level:  p_i + s_i >= p_min + 2 * s_min
  - DRIFT level:    p_i + s_i >= p_min + 3 * s_min

On DRIFT, the caller should retrain on the buffered recent window and
reset the detector. WARNING just means "start buffering a fresh window,
things may be turning."
"""

import math


class DDMDriftDetector:
    def __init__(self, warning_factor=2.0, drift_factor=3.0, min_samples=100):
        self.warning_factor = warning_factor
        self.drift_factor = drift_factor
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.n = 0
        self.errors = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.state = "IN_CONTROL"

    def update(self, correct: bool) -> str:
        """Feed one labeled outcome (True = prediction was correct).
        Returns current state: IN_CONTROL, WARNING, or DRIFT."""
        self.n += 1
        if not correct:
            self.errors += 1

        p_i = self.errors / self.n
        s_i = math.sqrt(p_i * (1 - p_i) / self.n) if self.n > 0 else 0.0

        if self.n < self.min_samples:
            self.state = "IN_CONTROL"
            return self.state

        if p_i + s_i < self.p_min + self.s_min:
            self.p_min = p_i
            self.s_min = s_i

        if p_i + s_i > self.p_min + self.drift_factor * self.s_min:
            self.state = "DRIFT"
        elif p_i + s_i > self.p_min + self.warning_factor * self.s_min:
            self.state = "WARNING"
        else:
            self.state = "IN_CONTROL"

        return self.state

    def stats(self) -> dict:
        p_i = self.errors / self.n if self.n else 0.0
        return {
            "n": self.n,
            "errors": self.errors,
            "current_error_rate": round(p_i, 4),
            "p_min": round(self.p_min, 4) if self.p_min != float("inf") else None,
            "s_min": round(self.s_min, 4) if self.s_min != float("inf") else None,
            "state": self.state,
        }
