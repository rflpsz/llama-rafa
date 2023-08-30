# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizationConfig

# Create a quantization config
config = QuantizationConfig(disable_exllama=True)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")

# Quantize the model
model = model.quantize(config)

prompt = "Tell me the first ten numbers of fibonaci sequence"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]