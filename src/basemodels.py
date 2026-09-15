from pydantic import (
    BaseModel, Field, field_validator, ConfigDict, ValidationError)
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
    parameters: dict[str, ParameterSpec] = Field(...)
    returns: ParameterSpec


class PromptItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_length=1)


class FunctionCallResult(BaseModel):
    """Check and validate fuction call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=4)
    parameters: dict[str, Any] = Field(...)
