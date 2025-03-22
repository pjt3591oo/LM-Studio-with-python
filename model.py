from langchain_core.embeddings import Embeddings
from openai import OpenAI
from typing import List

class LmStudioModel(Embeddings):
  def __init__(self, base_url, api_key="lm-studio"):
    self.client = OpenAI(base_url=base_url, api_key=api_key)

  def embed_documents(self, texts: List[str], model="nomic-ai/nomic-embed-text-v1.5-GGUF") -> List[List[float]]:
    texts = list(map(lambda text:text.replace("\n", " "), texts))
    datas = self.client.embeddings.create(input=texts, model=model).data
    
    return list(map(lambda data:data.embedding, datas))
      
  def embed_query(self, text: str) -> List[float]:
    return self.embed_documents([text])[0]
  
  def chat(self, prompt: str, model="lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF"):
    try:
      completion = self.client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens= 1024
      )
      return completion.choices[0].message.content
    except Exception as e:
      return f"Error in local LLM prediction: {str(e)}"

  def chat_stream(self, prompt: str, model="lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF"):
    content = ''
    try:
      response = self.client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens= 1024,
        stream=True
      )
      
      for sse_chunk in response:
        content += sse_chunk.choices[0].delta.content
        print(content)
      
      return content
    except Exception as e:
      return f"Error in local LLM prediction: {str(e)}"
