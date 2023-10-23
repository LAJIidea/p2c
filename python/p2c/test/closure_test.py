from typing import Any

def outer(a: int, b: int) -> Any:
    c = a * b + b
    def inner(m: float, n: float):
        return m * n + c
    return inner

