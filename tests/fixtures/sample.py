"""Fixture module for chunker tests. Line numbers matter -- do not reformat."""

import math

MAX_TOKENS = 100


def top_level_function(a, b):
    """Adds two numbers."""
    result = a + b
    return result


@staticmethod
def decorated_function():
    """Has a decorator above it."""
    return 42


class TokenBucket:
    """A tiny token bucket."""

    def __init__(self, capacity, refill_per_second):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = capacity

    def refill(self, elapsed_seconds):
        """Refill tokens based on elapsed time."""
        self.tokens = min(self.capacity, self.tokens + elapsed_seconds * self.refill_per_second)

    def consume(self):
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


def trailing_function():
    return math.pi
