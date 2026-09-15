import sys
from basemodels import FunctionCallResult
from pydantic import ValidationError
import json
from typing import Any


class WritingOutputError(Exception):
    """Custom error for writing output function"""
    pass


def write_output(output_dicts: dict[str, Any], path: str) -> None:
    """Write the result in the output file"""
    try:
        with open(path, 'w', encoding='utf-8') as output_file:
            output_json = json.dumps(output_dicts, indent=2)
            output_file.write(output_json)
    except ValidationError:
        raise WritingOutputError("error de ejemplo")
    except (FileNotFoundError, IsADirectoryError):
        raise WritingOutputError(
            f"'{path}' does not point to a valid folder"
            ) from None
    except PermissionError:
        raise WritingOutputError(
            f"'{path}' Permission denied"
            ) from None
    except OSError:
        raise WritingOutputError(
            f"unexpected OS error while writing {path}"
            ) from None
