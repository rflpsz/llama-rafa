import transformers

model_name = "TheBloke/Llama-2-13B-Chat-fp16"

# Download the model
model = transformers.AutoModel.from_pretrained(model_name, output_dir="/models/13B/")

# Print the model name
print(model.config.name)