from transformers import AutoModelForQuestionAnswering, QuantizationConfig

# Create a model
model = AutoModelForQuestionAnswering.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")

# Create a quantization config
config = QuantizationConfig(disable_exllama=True)

# Quantize the model
model = model.quantize(config)

# Create a question
question = "What is the capital of France?"

# Get the answer
answer = model.answer(question)

# Print the answer
print(answer)