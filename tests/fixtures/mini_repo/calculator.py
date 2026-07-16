"""A tiny calculator module used as an indexing fixture."""


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a and return the difference."""
    return a - b


class Accumulator:
    """Keeps a running total of added values."""

    def __init__(self) -> None:
        self.total = 0.0

    def accumulate(self, value: float) -> float:
        """Add a value to the running total."""
        self.total += value
        return self.total
