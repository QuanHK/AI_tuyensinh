from sentence_transformers import SentenceTransformer, util
import json
import os
import torch

class QAModel:
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "qa_data.json")
        with open(path, "r", encoding="utf-8") as f:
            self.qa = json.load(f)
        self.questions = [item["question"] for item in self.qa]
        self.answers = [item["answer"] for item in self.qa]
        self.model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def ask(self, query: str) -> str:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        result = util.semantic_search(query_emb, self.embeddings, top_k=1)
        idx = result[0][0]["corpus_id"]
        return self.answers[idx]

model_instance = QAModel()

def ask(query: str) -> str:
    return model_instance.ask(query)
