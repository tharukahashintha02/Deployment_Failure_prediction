"""A small module used to exercise the CI risk gate with real builds.

Its tests pass or fail depending on whether the code is correct, which gives
the prediction service genuine outcomes to learn from instead of outcomes
typed in by hand.
"""


def add(a, b):
    return a - b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
