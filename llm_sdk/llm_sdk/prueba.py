from llm_sdk import Small_LLM_Model
import json

modelo = Small_LLM_Model()
encoded = modelo.encode("holis holis")

input_id = encoded[0].tolist()

print(f" Esto hace encode: {input_id}")
print(f"Esto hace decode: {modelo.decode(0)}")
print(f"Esto hace decode: {modelo.decode(285)}")
print(f"Esto hace decode: {modelo.decode(23523)}")
print(f"Esto hace decode: {modelo.decode(285)}")
# print(modelo.get_path_to_vocab_file())

print(modelo.get_logits_from_input_ids(input_id))

# path = "/home/ox/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"

# with open(path) as f:
#     vocab = json.load(f)

# print(type(vocab))
# print(list(vocab.items())[:5])