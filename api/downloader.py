import time
from huggingface_hub import hf_hub_download

def download_model(repo_id, filename, use_auth_token):
  start_time = time.time()
  downloaded_model_path = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=use_auth_token)
  end_time = time.time()

  print(f"Downloaded model to {downloaded_model_path} in {end_time - start_time} seconds.")

if __name__ == "__main__":
  download_model(repo_id="TheBloke/Llama-2-13B-chat-GGML", filename="llama-2-13b-chat.ggmlv3.q8_0.bin", use_auth_token=True)
