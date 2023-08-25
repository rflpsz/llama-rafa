# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="TheBloke/Llama-2-13B-Chat-fp16")

# from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
# import transformers
# tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-Chat-fp16", use_fast=False)
# model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-Chat-fp16", device_map = 'auto',
#                                              **{"rope_scaling":{"factor": 2.0,"type": "linear"}})

user_prompt = "..."

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map="auto",
# )


sequences = pipeline(
   user_prompt,
    max_length=8000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

print(sequences)