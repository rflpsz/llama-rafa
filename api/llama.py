from llama_cpp import Llama
llm = Llama(model_path="/root/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/47d28ef5de4f3de523c421f325a2e4e039035bab/Llama-2-13B-chat-GGML")
# /root/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/47d28ef5de4f3de523c421f325a2e4e039035bab/llama-2-13b-chat.ggmlv3.q8_0.bin
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=256, stop=["Q:", "\n"], echo=True)
print(output)