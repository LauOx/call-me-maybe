import os
import sys
import argparse
from file_loader import ParsingFileError, load_fn_definitions, load_prompts
from write_output import write_output, WritingOutputError
from basemodels import FunctionDefinition, PromptItem, FunctionCallResult
from constrained_decoding import decode_output, DecodingError
from typing import Any
from pydantic import ValidationError
import json
from llm_sdk import Small_LLM_Model


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
        output: list[FunctionCallResult] = []
        model = Small_LLM_Model()
        vocab_path: str = model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            vocab_file = json.load(f)
            vocab: dict[int, str] = {
                v: k for k, v in vocab_file.items()
                }
        for prompt in prompts:
            print("********************")
            print(f"Processing prompt:\n{prompt}")
            object_dict: dict[str, Any] = decode_output(
                prompt.prompt, functions, model, vocab
                )
            object: FunctionCallResult = FunctionCallResult(**object_dict)
            output.append(object)

        output_dicts = [result.model_dump() for result in output]
        write_output(output_dicts, output_path)
    except (
                FileNotFoundError, PermissionError
            ):
        raise DecodingError("Model vocab file couldn't be found")
    except ValidationError as e:
        raise ValidationError(f"Error validating an object {e}")
    except ParsingFileError as e:
        raise ParsingFileError(f"An error ocurred while parsing json files: {e}",
              file=sys.stderr)
    except WritingOutputError as e:
        raise WritingOutputError(f"An error ocurred while writing the output file: {e}",
              file=sys.stderr)
    except KeyboardInterrupt:
        raise KeyboardInterrupt("User interrupted the program")
    # excep exceptions


if __name__ == "__main__":
    call_me_maybe()
