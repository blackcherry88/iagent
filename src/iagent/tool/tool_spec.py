from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Callable, get_type_hints
from pydantic import BaseModel, create_model, Field
import inspect
from griffe import Docstring, DocstringSectionKind


@dataclass
class ToolSpec:
    name: str  # Required field
    description: str  # Required field
    args_schema: Optional[Type[BaseModel]] = None  # Optional field

##
# _detect_docstring_style is copied from 
# https://github.com/openai/openai-agents-python/blob/main/src/agents/function_schema.py
##
def _detect_docstring_style(doc: str) -> str:
    import re
    scores = {"sphinx": 0, "numpy": 0, "google": 0}
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1
    max_score = max(scores.values())
    if max_score == 0:
        return "google"
    for style in ["sphinx", "numpy", "google"]:
        if scores[style] == max_score:
            return style
    return "google"


def to_tool_spec(func: Callable) -> ToolSpec:
    """
    Converts a Python function or callable into a ToolSpec.

    This function extracts parameter descriptions from the function's docstring using griffe,
    and injects them into the generated Pydantic model for argument schema.

    :param func: The Python function or callable to convert.
    :type func: Callable
    :raises ValueError: If the provided object is not callable or has an empty docstring.
    :return: A ToolSpec object containing the function name, description, and argument schema.
    :rtype: ToolSpec
    """
    if not callable(func):
        raise ValueError("Provided object is not callable.")

    name = func.__name__

    doc = inspect.getdoc(func)
    if not doc or not doc.strip():
        raise ValueError(f"Function '{name}' must have a non-empty docstring for description.")

    # Detect docstring style and parse with griffe
    style = _detect_docstring_style(doc)
    docstring = Docstring(doc, lineno=1, parser=style)
    parsed = docstring.parse()

    # Extract summary/description
    description = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), ""
    )
    # Extract parameter descriptions
    param_desc_map = {}
    for section in parsed:
        if section.kind == DocstringSectionKind.parameters:
            for param in section.value:
                param_desc_map[param.name] = param.description

    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    fields = {}
    for param in signature.parameters.values():
        ann = type_hints.get(param.name, param.annotation)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        desc = param_desc_map.get(param.name, "")
        fields[param.name] = (ann, Field(default, description=desc))

    args_schema = create_model(f"{name}ArgsSchema", **fields) if fields else None

    return ToolSpec(name=name, description=description, args_schema=args_schema)


def to_openai_tool_format(tool_spec: ToolSpec) -> Dict[str, Any]:
    """
    Converts a ToolSpec object into the OpenAI tool format dictionary.

    This function generates a dictionary suitable for OpenAI function calling, ensuring that
    each parameter property includes an explicit description, type, and other relevant schema
    information. The resulting schema disables additional properties and marks required fields.

    :param tool_spec: The ToolSpec object to convert.
    :type tool_spec: ToolSpec
    :return: A dictionary representation of the ToolSpec in OpenAI tool format.
    :rtype: Dict[str, Any]
    """
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }

    if tool_spec.args_schema:
        schema = tool_spec.args_schema.model_json_schema()
        for prop, prop_schema in schema.get("properties", {}).items():
            # Always extract "type" robustly
            if "type" in prop_schema:
                prop_type = prop_schema["type"]
            elif "anyOf" in prop_schema:
                # Use the first type in anyOf
                prop_type = next(
                    (entry["type"] for entry in prop_schema["anyOf"] if "type" in entry),
                    "string"
                )
            else:
                prop_type = "string"
            parameters["properties"][prop] = {
                "type": prop_type,
                "description": prop_schema.get("description", "")
            }
            if "enum" in prop_schema:
                parameters["properties"][prop]["enum"] = prop_schema["enum"]
            if "items" in prop_schema:
                parameters["properties"][prop]["items"] = prop_schema["items"]
        parameters["required"] = schema.get("required", [])

    return {
        "type": "function",
        "function": {
            "name": tool_spec.name,
            "description": tool_spec.description,
            "parameters": parameters,
            "strict": True
        }
    }
