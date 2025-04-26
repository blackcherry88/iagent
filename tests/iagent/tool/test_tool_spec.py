from typing import Any, Dict, List, Optional

import pytest

from iagent.tool.tool_spec import to_openai_tool_format, to_tool_spec


def func_simple(a: int, b: str) -> None:
    """Simple function.

    :param a: An integer parameter.
    :type a: int
    :param b: A string parameter.
    :type b: str
    """
    pass


def func_with_defaults(x: float = 1.5, y: bool = True) -> None:
    """Function with defaults.

    Args:
        x: A float parameter.
        y: A boolean parameter.
    """
    pass


def func_optional(opt: Optional[str] = None) -> None:
    """Function with optional parameter.

    Parameters
    ----------
    opt : str, optional
        An optional string.
    """
    pass


def func_list_enum(color: str) -> None:
    """Function with enum.

    Args:
        color: Color name. One of {'red', 'green', 'blue'}.
    """
    pass


def func_list_param(items: List[int]) -> None:
    """Function with list parameter.

    :param items: List of integers.
    :type items: list of int
    """
    pass


def func_dict_param(config: Dict[str, Any]) -> None:
    """Function with dict parameter.

    :param config: Configuration dictionary.
    :type config: dict
    """
    pass


def func_no_docstring(a: int) -> None:
    pass


@pytest.mark.parametrize(
    "func,expected_types,expected_required",
    [
        (func_simple, {"a": "integer", "b": "string"}, {"a", "b"}),
        (func_with_defaults, {"x": "number", "y": "boolean"}, set()),
        (func_optional, {"opt": "anyOf"}, set()),
        (func_list_param, {"items": "array"}, {"items"}),
        (func_dict_param, {"config": "object"}, {"config"}),
    ],
)
def test_to_tool_spec_schema(func, expected_types, expected_required):
    tool_spec = to_tool_spec(func)
    schema = tool_spec.args_schema.model_json_schema()
    for param, typ in expected_types.items():
        assert param in schema["properties"]
        if typ == "anyOf":
            assert schema["properties"][param]["anyOf"]
        else:
            assert schema["properties"][param]["type"] == typ
    if expected_required:
        assert set(schema["required"]) == expected_required
    else:
        assert "required" not in schema or not schema["required"]


def test_to_tool_spec_description_and_name():
    tool_spec = to_tool_spec(func_simple)
    assert tool_spec.name == "func_simple"
    assert "Simple function" in tool_spec.description


def test_to_tool_spec_optional_param():
    tool_spec = to_tool_spec(func_optional)
    schema = tool_spec.args_schema.model_json_schema()
    assert "opt" in schema["properties"]
    assert schema["properties"]["opt"]["anyOf"]
    assert "required" not in schema or "opt" not in schema.get("required", [])


def test_to_tool_spec_no_docstring():
    with pytest.raises(ValueError):
        to_tool_spec(func_no_docstring)


def test_to_openai_tool_format_properties():
    tool_spec = to_tool_spec(func_simple)
    openai_schema = to_openai_tool_format(tool_spec)
    params = openai_schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["a"]["description"] != ""
    assert params["properties"]["b"]["type"] == "string"
    assert params["properties"]["b"]["description"] != ""
    assert params["required"] == ["a", "b"]
    assert params["additionalProperties"] is False


def test_to_openai_tool_format_optional():
    tool_spec = to_tool_spec(func_optional)
    openai_schema = to_openai_tool_format(tool_spec)
    params = openai_schema["function"]["parameters"]
    assert "opt" in params["properties"]
    assert params["properties"]["opt"]["type"] == "string"
    assert params["properties"]["opt"]["description"] != ""
    assert "required" not in params or "opt" not in params.get("required", [])


def test_to_openai_tool_format_with_defaults():
    tool_spec = to_tool_spec(func_with_defaults)
    openai_schema = to_openai_tool_format(tool_spec)
    params = openai_schema["function"]["parameters"]
    assert "x" in params["properties"]
    assert "y" in params["properties"]
    assert params["properties"]["x"]["type"] == "number"
    assert params["properties"]["y"]["type"] == "boolean"
    assert params["properties"]["x"]["description"] != ""
    assert params["properties"]["y"]["description"] != ""
    assert params["required"] == []


def test_to_openai_tool_format_list_and_dict():
    tool_spec = to_tool_spec(func_list_param)
    openai_schema = to_openai_tool_format(tool_spec)
    params = openai_schema["function"]["parameters"]
    assert params["properties"]["items"]["type"] == "array"

    tool_spec = to_tool_spec(func_dict_param)
    openai_schema = to_openai_tool_format(tool_spec)
    params = openai_schema["function"]["parameters"]
    assert params["properties"]["config"]["type"] == "object"
