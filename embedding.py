from model import LmStudioModel

emb_model = LmStudioModel(
  base_url="http://localhost:1234/v1"
)

emb_vectors = emb_model.embed_documents([
    "hello world",
    "welcome to the world",
    "helloworld",
])

print(emb_vectors)