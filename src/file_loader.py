import sys
from basemodels import FunctionDefinition, PromptItem
from pydantic import ValidationError
import json


class ParsingFileError(Exception):
    """Custom exception for wrong format in json file"""
    pass


def load_fn_definitions(path: str) -> list[FunctionDefinition]:
    """Load the function_definitions.json file
    Args:
    path: str: the path to the file
    Raises:
    """
    function_list: list[FunctionDefinition] = []
    try:
        with open(path, 'r', encoding='utf-8') as fn_def_file:
            data = json.load(fn_def_file)
    except FileNotFoundError:
        raise ParsingFileError(f"'{path}' does not exist") from None
    except json.JSONDecodeError:
        raise ParsingFileError(f"'{path}' contains invalid JSON") from None
    if not isinstance(data, list):
        raise ParsingFileError(
            f"Error: The JSON file {path} must contain a list at its root."
            )
    for item in data:
        try:
            definition: FunctionDefinition = FunctionDefinition(**item)
            function_list.append(definition)
        except ValidationError:
            print(f"The function definition:\n{item}\n"
                  "Does not match the expected format and will be ignored",
                  file=sys.stderr)
    return function_list


def load_prompts(path: str) -> list[PromptItem]:
    """Load the prompts from de function_calling_test.json"""
    prompt_list: list[PromptItem] = []
    try:
        with open(path, 'r', encoding='utf-8') as prompt_file:
            data = json.load(prompt_file)
    except FileNotFoundError:
        raise ParsingFileError(f"'{path}' does not exist") from None
    except json.JSONDecodeError:
        raise ParsingFileError(f"'{path}' contains invalid JSON") from None
    if not isinstance(data, list):
        raise ParsingFileError(
            f"Error: The JSON file {path} must contain a list at its root."
            )
    for item in data:
        try:
            prompt: PromptItem = PromptItem(**item)
            prompt_list.append(prompt)
        except ValidationError:
            print(f"Prompt\n{item}\nDoes not match the correct format "
                  "and will be ignored",
                  file=sys.stderr)
    return prompt_list


def check_file_loaders():
    print("Function definition:")
    print(load_fn_definitions("input/functions_definition.json"))
    print("Prompts")
    print(load_prompts("input/function_calling_tests.json"))


if __name__ == "__main__":
    check_file_loaders()
