import os
import sys
import argparse
from file_loader import ParsingFileError, load_fn_definitions, load_prompts
from write_output import write_output, WritingOutputError
from basemodels import FunctionDefinition, PromptItem, FunctionCallResult
from constrained_decoding import decode
from typing import Any
from pydantic import ValidationError


def parse_args() -> argparse.Namespace:
    """Parse arguments and get type of visualization"""
    parser = argparse.ArgumentParser(
        description="Function calling"
        )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/",
        help="file.json file with prompts"
        )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output",
        help="file.json file where the output is going to be written"
        )
    return parser.parse_args()


def call_me_maybe() -> None:
    """Run function calling program and raises erros"""
    args: argparse.Namespace = parse_args()
    fn_def_path: str = os.path.join(args.input, 'functions_definition.json')
    prompts_path: str = os.path.join(args.input, 'function_calling_tests.json')
    output_path: str = os.path.join(
        args.output, 'function_calling_results.json'
        )
    try:
        functions: list[FunctionDefinition] = load_fn_definitions(fn_def_path)
        prompts: list[PromptItem] = load_prompts(prompts_path)
        param_types_list: list[str] = []
        for function in functions:
            for param in function.parameters:
                param_types_list.append(function.parameters[param].type)
        param_set: set[str] = set(param_types_list)
        print(param_set)
        output = list[dict[str, Any]]
        for prompt in prompts:
            object_dict: dict[str, Any] = decode(prompt.prompt, functions)
        try:
            object: FunctionCallResult = FunctionCallResult(**object_dict)
        except ValidationError as e:
            print(f"Error validating functioncall {e}")
            return
        output.append(object)
        write_output(output, output_path)
    except ParsingFileError as e:
        print(f"An error ocurred while parsing json files: {e}",
              file=sys.stderr)
    except WritingOutputError as e:
        print(f"An error ocurred while writing the output file: {e}",
              file=sys.stderr)
        



if __name__ == "__main__":
    call_me_maybe()
