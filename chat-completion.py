from model import LmStudioModel

emb_model = LmStudioModel(
  base_url="http://localhost:1234/v1"
)

response = emb_model.chat("너는 한국어를 할 수 있니?")

print(response)