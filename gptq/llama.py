from llama_cpp import Llama
llm = Llama(model_path="../models/13B/TheBloke/Llama-2-13B-chat-GGML")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=256, stop=["Q:", "\n"], echo=True)
print(output)