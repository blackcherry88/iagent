import inspect
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError, create_model


class Tool:
    """
    Base class for tools that agents can use within an environment.

    Attributes:
        name (str): The name of the tool.
        description (str): A description of what the tool does.
        args_schema (Optional[Type[BaseModel]]): Pydantic model defining the arguments.
    """

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None

    def run(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        raise NotImplementedError


class FunctionTool(Tool):
    """
    A tool built from a Python function.

    Automatically extracts name, description, and argument schema from the function
    if not provided explicitly.
    """

    func: Callable[..., Any]

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Initializes the FunctionTool.

        Args:
            func: The function to wrap.
            name: The name of the tool. Defaults to the function's name.
            description: The description of the tool. Defaults to the function's docstring.
            args_schema: A Pydantic model defining the arguments. If None, it will
                         be inferred from the function's signature and type hints.
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""

        if args_schema:
            self.args_schema = args_schema
        else:
            # Infer schema from type hints if not provided
            self.args_schema = self._infer_schema_from_func_signature(func)

    def _infer_schema_from_func_signature(
        self, func: Callable[..., Any]
    ) -> Type[BaseModel]:
        """Infers a Pydantic model from the function's signature."""
        signature = inspect.signature(func)
        fields: Dict[str, Any] = {}
        for param_name, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                # Default to 'Any' if no type hint is provided
                param_type = Any
            else:
                param_type = param.annotation

            if param.default is inspect.Parameter.empty:
                # Required argument
                fields[param_name] = (param_type, ...)
            else:
                # Optional argument with default value
                fields[param_name] = (param_type, param.default)

        # Create a dynamic Pydantic model
        # Use function name in model name for clarity, replacing invalid chars
        sanitized_func_name = "".join(
            c if c.isalnum() else "_" for c in func.__name__
        )
        model_name = f"{sanitized_func_name.capitalize()}InputSchema"
        return create_model(model_name, **fields) # type: ignore[return-value]

    def run(self, **kwargs: Any) -> Any:
        """
        Executes the wrapped function with validated arguments.

        Args:
            **kwargs: The arguments to pass to the function.

        Returns:
            The result of the function execution.

        Raises:
            ValidationError: If the provided arguments do not match the args_schema.
            Exception: Any exception raised by the wrapped function.
        """
        if not self.args_schema:
            # Should not happen if __init__ logic is correct, but safeguard
            return self.func(**kwargs)

        try:
            # Validate arguments using the Pydantic model
            validated_args = self.args_schema(**kwargs)
            # Pass validated arguments as a dictionary
            return self.func(**validated_args.model_dump())
        except ValidationError as e:
            # Reraise validation errors for clarity
            raise e
        except Exception as e:
            # Reraise any other exceptions from the function execution
            # TODO: Consider wrapping this in a custom ToolExecutionError
            raise e

    def get_schema(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the tool's schema.

        Returns:
            A dictionary containing the tool's name, description, and
            JSON schema for its arguments.
        """
        schema: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.args_schema:
            # Use Pydantic's capability to generate JSON schema
            schema["parameters"] = self.args_schema.model_json_schema()
        else:
            schema["parameters"] = {"type": "object", "properties": {}} # No args expected

        return schema

# Example Usage (can be removed or moved to tests/docs later)
if __name__ == "__main__":
    def add(a: int, b: int = 5) -> int:
        """Adds two numbers together. 'b' defaults to 5 if not provided."""
        return a + b

    # Create a tool from the function
    add_tool = FunctionTool(func=add)

    # Get the schema
    print("Tool Schema:")
    import json
    print(json.dumps(add_tool.get_schema(), indent=2))
    print("-" * 20)

    # Run the tool with valid arguments
    print("Running tool with a=10, b=2:")
    result = add_tool.run(a=10, b=2)
    print(f"Result: {result}")
    print("-" * 20)

    print("Running tool with a=10 (using default b):")
    result_default = add_tool.run(a=10)
    print(f"Result: {result_default}")
    print("-" * 20)

    # Example of running with invalid arguments (will raise ValidationError)
    try:
        print("Running tool with invalid args (a='hello'):")
        add_tool.run(a="hello") # type: ignore[arg-type]
    except ValidationError as e:
        print(f"Caught expected error:\n{e}")

    # Example of a tool with no arguments
    def get_time() -> str:
        """Returns the current time as a string."""
        import datetime
        return str(datetime.datetime.now())

    time_tool = FunctionTool(get_time)
    print("\nTime Tool Schema:")
    print(json.dumps(time_tool.get_schema(), indent=2))
    print("Running time tool:")
    print(f"Result: {time_tool.run()}")
