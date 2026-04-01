from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.article_embeddings = None
        self.cached_articles = None

    def prepare_embeddings(self, articles):
        texts = [
            a.get("title", "") + " " + a.get("summary", "")
            for a in articles
        ]

        self.article_embeddings = self.model.encode(texts)
        self.cached_articles = articles

    def search(self, query, articles, top_k=5):
        if not articles:
            return []

        #  Compute embeddings ONLY ONCE
        if self.article_embeddings is None or self.cached_articles != articles:
            self.prepare_embeddings(articles)

        query_embedding = self.model.encode([query])

        scores = cosine_similarity(query_embedding, self.article_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [articles[i] for i in top_indices]