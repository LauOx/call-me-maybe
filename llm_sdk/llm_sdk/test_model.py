from llm_sdk import Small_LLM_Model
from enum import Enum


class CandidateStatus(Enum):
    VALID_COMPLETE = "valid_complete"
    VALID_PREFIX = "valid_prefix"
    INVALID = "invalid"


class ParserStatus(Enum):
    START = "start"
    OBJECT_OPEN = "object_open"
    KEY_OPEN = "key_open"
    KEY_NAME = "key_name"
    KEY_CLOSE = "key_close"
    COLON = "colon"
    VALUE_OPEN = "value_open"
    FUNCTION_NAME = "function_name"
    VALUE_CLOSE = "value_close"
    OBJECT_CLOSE = "object_close"
    COMPLETE = "complete"
    INVALID = "invalid"


class ParserState():

    def __init__(
            self,
            state: ParserStatus,
            key_buffer: str,
            function_name_buffer: str
            ):
        self.state = state
        self.key_buffer = key_buffer
        self.function_name_buffer = function_name_buffer


def next_state(current_state: ParserStatus, char: str) -> ParserStatus:
    if current_state == ParserStatus.START and char == "{":
        return ParserStatus.OBJECT_OPEN
    elif current_state == ParserStatus.OBJECT_OPEN and char == '"':
        return ParserStatus.KEY_OPEN
    return ParserStatus.INVALID


def consume_char(state: ParserState, char: str):
    if state.state in (ParserStatus.START, ParserStatus.OBJECT_OPEN):
        state.state = next_state(state.state, char)
    if state.state in (ParserStatus.KEY_OPEN, ParserStatus.KEY_NAME):
        if state.key_buffer == 'name' and char == '"':
            state.state = ParserStatus.KEY_CLOSE
        else:
            new_buffer = state.key_buffer + char
            if "name".startswith(new_buffer):
                state.key_buffer += char
                state.state = ParserStatus.KEY_NAME
            else:
                state.state = ParserStatus.INVALID
    return state


def check_candidates(
        candidate: str, allowed_names: list[str]
        ) -> CandidateStatus:

    has_valid_prefix = False
    for function_name in allowed_names:
        if candidate == function_name:
            return CandidateStatus.VALID_COMPLETE
        else:
            if function_name.startswith(candidate):
                has_valid_prefix = True
    if has_valid_prefix:
        return CandidateStatus.VALID_PREFIX
    return CandidateStatus.INVALID


allowed_names = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string"
]
model = Small_LLM_Model()
encoded = model.encode("I am")
print(encoded)
input_ids = encoded[0].tolist()
generated_text = "fn_"
for i in range(10):
    logits = model.get_logits_from_input_ids(input_ids)
    masked_logits = logits.copy()
    for token_id in range(len(masked_logits)):
        decoded_token = model.decode([[token_id]])
        token_text = decoded_token[0]
        candidate = generated_text + token_text
        status = check_candidates(candidate, allowed_names)
        if status == CandidateStatus.INVALID:
            masked_logits[token_id] = float("-inf")
    if max(masked_logits) == float("-inf"):
        print("No valid tokens found")
        break
    next_token_id = masked_logits.index(max(masked_logits))
    decoded_next_token = model.decode([[next_token_id]])
    generated_text += decoded_next_token[0]
    input_ids.append(next_token_id)
    print(f"token n{i + 1} - {decoded_next_token}")
    if (
        check_candidates(generated_text, allowed_names)
        == CandidateStatus.VALID_COMPLETE
    ):
        break
decoded = model.decode([input_ids])
print(decoded)
test_parser = ParserState(ParserStatus.START, "", "")
print(test_parser.state)
consume_char(test_parser, "{")
print(test_parser.state)
print(next_state(ParserStatus.START, "{"))
print(next_state(ParserStatus.OBJECT_OPEN, '"'))
print(next_state(ParserStatus.START, "["))
