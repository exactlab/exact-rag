import pytest

from typing import Any, Callable
from exact_rag.dataemb import Caller

@pytest.fixture
def get_id_callable() -> Callable[..., Any]:
    def id(**kwargs: Any) -> dict[str, Any]:
        return kwargs
    return id

@pytest.fixture
def get_input_args() -> dict[str, Any]:
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    return d

def test_Caller_no_args(get_id_callable, get_input_args):
    caller = Caller(get_id_callable)
    assert caller(**get_input_args) == get_input_args

def test_Caller_swap(get_id_callable, get_input_args):
    swap = {"A": "a", "D": "d", "EE": "e"}
    input = {"A": 1, "b": 2, "c": 3, "D": 4, "EE": 5}
    caller = Caller(get_id_callable, swap)
    assert caller(**input) == get_input_args

def test_Caller_accept_only(get_id_callable, get_input_args):
    accept_only = ["b", "c", "e"]
    expected = {"b": 2, "c": 3, "e": 5}
    caller = Caller(get_id_callable, None, accept_only)
    assert caller(**get_input_args) == expected

def test_Caller_full(get_id_callable):
    accept_only = ["B", "C", "EE"]
    swap = {"B": "b", "C": "c", "EE": "e"}
    input = {"a": 1, "B": 2, "C": 3, "d": 4, "EE": 5}
    expected = {"b": 2, "c": 3, "e": 5}
    caller = Caller(get_id_callable, swap, accept_only)
    assert caller(**input) == expected