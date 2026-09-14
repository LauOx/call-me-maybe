from pydantic import (
    BaseModel, Field, field_validator, ConfigDict, ValidationError)
from enum import Enum
from typing import Any


class ParameterSpec(BaseModel):
    """Check and validate the function parameters"""
    model_config = ConfigDict(extra='forbid')
    type: str


class FunctionDefinition(BaseModel):
    """Check and validate the functions available"""
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., min_length=4)
    description: str = Field(...)
    parameters: dict[str, ParameterSpec] = Field(...) # podría ser dict[str, str], aun no se para que hace falta el modelo de ParameterSpec
    returns: ParameterSpec


class FunctionCallResult(BaseModel):
    """Check and validate fuction call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=4)
    parameters: dict[str, Any] = Field(...)


class PromptItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_length=1)


def check_basemodel():
    p_type = PType.STR
    parameter_dict = {
      "name": {
        "type": p_type
      }
    }
    r_type = PType.STR
    return_dict = {"type": r_type}
    function_def_dict = {
        "name": "fn_greet",
        'description': 'Generate a greeting message for a person by name.',
        'parameters': parameter_dict,
        'returns': return_dict
    }
    try:
        function_basemodel = FunctionDefinition(**function_def_dict)
        function_definition = {
             'name': function_basemodel.name,
             'description': function_basemodel.description,
             'parameters': function_basemodel.parameters,
             'returns': function_basemodel.returns
        }
        print(function_definition)
    except ValidationError as e:
        error_msg = e.errors()[0]["msg"]
        raise ValueError(f"An error occurred: {error_msg}")


if __name__ == "__main__":
    check_basemodel()
