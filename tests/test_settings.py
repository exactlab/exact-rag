import pytest

from typing import Any, Annotated
from exact_rag.settings import FromDict, Settings
from pydantic import ValidationError


@pytest.fixture
def get_dict() -> dict[str, Any]:
    d = {
        "first": 1,
        "second": 2,
        "third": {"alpha": 3, "beta": {"a": 4, "b": 5, "c": 6, "d": 7}, "gamma": 8},
        "fourth": 9,
        "fifth": {"alpha": 10, "beta": 11},
    }
    return d


test_FromDict_params = [
    (("first",), 1),
    (("third", "alpha"), 3),
    (("third", "beta", "c"), 6),
    (("fourth", "beta"), None),
    (("fifth", "alpha"), 10),
    (("sixth", "alpha", "a"), None),
]


@pytest.mark.parametrize("input, expected", test_FromDict_params)
def test_FromDict(get_dict, input, expected):
    fd = FromDict(*input)
    assert fd(get_dict) == expected


class Configs(Settings):
    integer: int
    floating: Annotated[float, FromDict("double")]
    string: Annotated[str, FromDict("strings", "my_string")]


def test_config_ok():
    d_in = {
        "integer": 5,
        "double": -45.673,
        "strings": {"my_string": "example", "your_string": "second_example"},
    }
    configs = Configs(**d_in)
    d_out = configs.model_dump()

    exp_d_out = {
        "integer": d_in["integer"],
        "floating": d_in["double"],
        "string": d_in["strings"]["my_string"],
    }
    assert d_out == exp_d_out


def test_config_fail_string():
    d_in = {"integer": 5, "double": -45.673}
    with pytest.raises(ValidationError) as error:
        Configs(**d_in)
    assert error.value.error_count() == 1
    assert "string" in error.value.errors()[0]["msg"]


def test_config_fail_double():
    d_in = {"integer": 5, "strings": {"my_string": "example"}}
    with pytest.raises(ValidationError) as error:
        Configs(**d_in)
    assert error.value.error_count() == 1
    assert "number" in error.value.errors()[0]["msg"]


class WrongConfig(Settings):
    integer: Annotated[int, FromDict("first"), FromDict("second")]


def test_wrong_config():
    d_in = {"first": 1, "second": 2}
    with pytest.raises(TypeError) as error:
        config = WrongConfig(**d_in)
    assert "Multiple" in error.value.args[0]
