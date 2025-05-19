from sentence_transformers import SentenceTransformer, util
import json
import torch

class QAModel:
    def __init__(self, data_path="qa_data.json"):
        self.model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.corpus = [item["question"] for item in self.data]
        self.answers = [item["answer"] for item in self.data]
        self.embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def ask(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=1)
        idx = hits[0][0]['corpus_id']
        return self.answers[idx]

model = QAModel()

