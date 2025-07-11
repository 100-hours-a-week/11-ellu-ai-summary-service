import torch
from transformers import AutoTokenizer, AutoModel

class CustomEmbeddingFunction:
    def __init__(self, model_name="BM-K/KoSimCSE-roberta-multitask"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self._load_model()

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch, return_dict=False)
            embeddings = outputs[0][:, 0, :]  # CLS 토큰
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.__call__([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.__call__(texts)