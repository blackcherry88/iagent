import pytest
from pydantic import BaseModel, ValidationError
import sys
print(f"sys.path: {sys.path}")
from iagent.tool import FunctionTool


# --- Test Functions ---
def simple_add(a: int, b: int) -> int:
    """A simple function to add two integers."""
    return a + b


def multiply(x: float, y: float = 2.0) -> float:
    """Multiplies two floats. y defaults to 2.0."""
    return x * y


def no_args_func() -> str:
    """A function that takes no arguments."""
    return "No args here!"


def func_with_complex_docstring(param1: str):
    """
    This is a complex docstring.

    It has multiple lines and sections.

    Args:
        param1: The first parameter.

    Returns:
        None implicitly.
    """
    pass


def func_raises_error(val: int):
    """This function always raises a ValueError."""
    raise ValueError(f"Value {val} is invalid")


# --- Custom Pydantic Schemas ---
class AddSchema(BaseModel):
    a: int
    b: int


class MultiplySchema(BaseModel):
    x: float
    y: float = 2.0


# --- Test Cases ---

# Test Initialization
def test_function_tool_init_basic():
    """Tests basic initialization inferring name and description."""
    tool = FunctionTool(func=simple_add)
    assert tool.name == "simple_add"
    assert tool.description == "A simple function to add two integers."
    assert tool.args_schema is not None
    assert "a" in tool.args_schema.model_fields
    assert "b" in tool.args_schema.model_fields


def test_function_tool_init_override():
    """Tests overriding name and description during initialization."""
    tool = FunctionTool(
        func=simple_add, name="CustomAdd", description="Does custom addition."
    )
    assert tool.name == "CustomAdd"
    assert tool.description == "Does custom addition."


def test_function_tool_init_no_docstring():
    """Tests initialization when the function has no docstring."""
    def no_doc(x: int):
        return x * x
    tool = FunctionTool(func=no_doc)
    assert tool.name == "no_doc"
    assert tool.description == "" # Should default to empty string


def test_function_tool_init_with_schema():
    """Tests initialization with an explicit Pydantic schema."""
    tool = FunctionTool(func=simple_add, args_schema=AddSchema)
    assert tool.args_schema == AddSchema


def test_function_tool_init_complex_docstring():
    """Tests that inspect.getdoc correctly extracts the clean docstring."""
    tool = FunctionTool(func=func_with_complex_docstring)
    assert tool.description == "This is a complex docstring.\n\nIt has multiple lines and sections."


# Test Schema Generation / Access
def test_get_schema_inferred():
    """Tests the generated schema dictionary from inferred types."""
    tool = FunctionTool(func=multiply)
    schema = tool.get_schema()
    assert schema["name"] == "multiply"
    assert schema["description"] == "Multiplies two floats. y defaults to 2.0."
    assert "parameters" in schema
    params = schema["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "x" in params["properties"]
    assert params["properties"]["x"]["type"] == "number" # JSON Schema type for float
    assert "y" in params["properties"]
    assert params["properties"]["y"]["type"] == "number"
    assert params["properties"]["y"]["default"] == 2.0
    assert "required" in params
    assert "x" in params["required"]
    assert "y" not in params["required"]


def test_get_schema_explicit():
    """Tests the generated schema dictionary from an explicit Pydantic model."""
    tool = FunctionTool(func=simple_add, args_schema=AddSchema)
    schema = tool.get_schema()
    assert schema["name"] == "simple_add" # Name comes from func even if schema provided
    assert schema["description"] == "A simple function to add two integers."
    assert "parameters" in schema
    params = schema["parameters"]
    # Check against Pydantic's generated schema structure
    expected_schema = AddSchema.model_json_schema()
    assert params == expected_schema


def test_get_schema_no_args():
    """Tests schema generation for a function with no arguments."""
    tool = FunctionTool(func=no_args_func)
    schema = tool.get_schema()
    assert schema["name"] == "no_args_func"
    assert schema["description"] == "A function that takes no arguments."
    assert "parameters" in schema
    params = schema["parameters"]
    assert params == {"type": "object", "properties": {}} # Expect empty properties


# Test Execution (.run method)
def test_run_success_required_args():
    """Tests successful execution with all required arguments."""
    tool = FunctionTool(func=simple_add)
    result = tool.run(a=10, b=20)
    assert result == 30


def test_run_success_default_args():
    """Tests successful execution using default argument values."""
    tool = FunctionTool(func=multiply)
    result = tool.run(x=5.0) # Uses default y=2.0
    assert result == 10.0


def test_run_success_override_default_args():
    """Tests successful execution overriding default argument values."""
    tool = FunctionTool(func=multiply)
    result = tool.run(x=5.0, y=3.0)
    assert result == 15.0


def test_run_success_no_args():
    """Tests successful execution of a function with no arguments."""
    tool = FunctionTool(func=no_args_func)
    result = tool.run()
    assert result == "No args here!"


def test_run_validation_error_missing_arg():
    """Tests that ValidationError is raised for missing required arguments."""
    tool = FunctionTool(func=simple_add)
    with pytest.raises(ValidationError):
        tool.run(a=10) # Missing 'b'


def test_run_validation_error_wrong_type():
    """Tests that ValidationError is raised for arguments of the wrong type."""
    tool = FunctionTool(func=simple_add)
    with pytest.raises(ValidationError):
        tool.run(a=10, b="hello") # 'b' should be int


def test_run_validation_error_extra_arg():
    """Tests that ValidationError is raised for unexpected arguments."""
    tool = FunctionTool(func=simple_add)
    with pytest.raises(ValidationError):
        tool.run(a=10, b=20, c=30) # 'c' is not expected


def test_run_function_raises_error():
    """Tests that exceptions raised by the wrapped function are propagated."""
    tool = FunctionTool(func=func_raises_error)
    with pytest.raises(ValueError, match="Value 5 is invalid"):
        tool.run(val=5)
