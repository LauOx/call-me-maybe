from llm_sdk import Small_LLM_Model
from basemodels import ParameterSpec, FunctionDefinition
from typing import Any
import aux


class DecodingError(Exception):
    """Custom error for decoding"""
    pass


def decode_fn_name(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        allowed: list[str]) -> str:
    """Find the next token to form the fn_name"""
    encoded = model.encode(context)
    input_ids = encoded[0].tolist()
    candidate: str = ""
    fn_name: str = ""
    status = aux.check_candidate_logits(candidate, allowed)
    # invalid tokens to logit -inf
    while status != aux.CandidateStatus.VALID_COMPLETE:
        logits = model.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            token_text = vocab.get(token_id)
            if token_text is None:
                logits[token_id] = float("-inf")
                continue
            candidate = fn_name + token_text
            status = aux.check_candidate_logits(candidate, allowed)
            if status == aux.CandidateStatus.INVALID:
                logits[token_id] = float("-inf")

        if max(logits) == float("-inf"):
            raise DecodingError("No matches found for function name")
        next_token_id = logits.index(max(logits))
        input_ids.append(next_token_id)
        decoded_next_token = vocab[next_token_id]
        # context += decoded_next_token
        fn_name += decoded_next_token
        status = aux.check_candidate_logits(fn_name, allowed)

    return fn_name


def decode_parameter(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        prompt: str,
        type: str) -> str:
    """decode each parameter needed to run function"""
    encoded = model.encode(context)
    input_ids = encoded[0].tolist()
    candidate: str = ""
    param_value: str = ""
    status = True
    if (
        type == aux.ParamType.NUMBER.name or
        type == aux.ParamType.FLOAT.name or
        type == aux.ParamType.INTEGER.name
    ):
        while status and len(param_value) < 10:
            logits = model.get_logits_from_input_ids(input_ids)
            for token_id in range(len(logits)):
                token_text = vocab.get(token_id)
                if token_text is None:
                    logits[token_id] = float("-inf")
                    continue
                candidate = param_value + token_text
                if not aux.is_valid_number(candidate):
                    logits[token_id] = float("-inf")

            if max(logits) == float("-inf"):
                raise DecodingError(
                    "No matches found for function parameter"
                    )
            next_token_id = logits.index(max(logits))
            input_ids.append(next_token_id)
            decoded_next_token = vocab[next_token_id]
            # print(f"valid candidate {decoded_next_token}/ max logit: {logits[next_token_id]}")
            candidate = param_value + decoded_next_token
            status = aux.is_valid_number(candidate)
            if status:
                context += decoded_next_token
                param_value += decoded_next_token
                # print(f"(number) param_value so far: {param_value}")
                status = aux.is_valid_number(param_value)
            # print(f"end of loop status: {status}")
    if type == aux.ParamType.STRING.name:
        status = True
        while status:
            print(f"(string) Param value so far: {param_value}")
            logits = model.get_logits_from_input_ids(input_ids)
            for token_id in range(len(logits)):
                token_text = vocab.get(token_id)
                if token_text is None:
                    logits[token_id] = float("-inf")
                    continue
                # if token_text == "Ċ":
                #     logits[token_id] = float("-inf")
            if param_value == "":
                if max(logits) == float("-inf"):
                    raise DecodingError(
                        "No matches found for function parameter"
                        )
            if param_value != "" and max(logits) == float("-inf"):
                status = False
                break
            next_token_id = logits.index(max(logits))
            input_ids.append(next_token_id)
            decoded_next_token = vocab[next_token_id]
            safe_part = aux.find_stop_char(decoded_next_token)
            if safe_part is not None:
                param_value += safe_part
                status = False
            else:
                param_value += decoded_next_token
                if param_value.count("'") >= 2 or param_value.count('"') >= 2:
                    status = False
                    break
                context += decoded_next_token

    final_param_value: str = param_value.replace('Ġ', ' ')
    return final_param_value.rstrip().rstrip('Ċ').rstrip("'")


def decode_output(
        prompt: str,
        functions: list[FunctionDefinition],
        model: Small_LLM_Model,
        vocab: dict[int, str],) -> dict[str, Any]:
    """Start the decode process"""
    object_return: dict[str: Any] = {}
    fn_name: str = ""
    # Find function name
    allowed_fn_names: list[str] = []
    for function in functions:
        allowed_fn_names.append(function.name)
    if len(allowed_fn_names) == 0:
        raise DecodingError(
            "No function names found in the function definitions input file"
            )
    fn_initial_context: str = aux.build_context_for_fn_name(functions, prompt)
    fn_name = decode_fn_name(
        model, vocab, fn_initial_context, allowed_fn_names
        )
    print(f"Function name: {fn_name}")
    # Find function parameters
    function: FunctionDefinition = next(
        function for function in functions
        if function.name == fn_name
    )
    funct_param: dict[str, ParameterSpec] = function.parameters
    result_parameters: dict[str, Any] = {}
    param_initial_context: str = (
        aux.build_context_for_parameter(prompt, function)
    )
    for param in funct_param:
        type = aux.ParamType(funct_param[param].type).name
        param_initial_context += (
            f"' {param}' (type: {type}) value: "
        )
        param_value: str = decode_parameter(
            model, vocab, param_initial_context, prompt, type
        )
        print(f"Value found for {param} = {param_value}")
        if (
            type == aux.ParamType.FLOAT.name or
            type == aux.ParamType.NUMBER.name or
            type == aux.ParamType.INTEGER.name
        ):
            if type == aux.ParamType.INTEGER.name:
                number = int(float(param_value))
                result_parameters[param] = number
            if (
                type == aux.ParamType.FLOAT.name or
                type == aux.ParamType.NUMBER.name
            ):
                if '.' in param_value or ',' in param_value:
                    number_str = aux.clean_number(param_value)
                    result_parameters[param] = float(number_str)
        else:
            cleaned_param_value: str = aux.clean_param_value(param_value)
            result_parameters[param] = cleaned_param_value
        param_initial_context += param_value + ','
    # Save final dict result
    object_return = {
        'prompt': prompt,
        'name': fn_name,
        'parameters': result_parameters
    }
    print(f"Final object: {object_return}")
    return object_return
