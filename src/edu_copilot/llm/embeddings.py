import os
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from edu_copilot.config import settings

class MockEmbeddings(Embeddings):
    """
    Deterministic mock embedding model that generates random-seeded, consistent
    vectors based on content hashes. Allows offline execution without API keys.
    """
    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Seed generator with the text hash for deterministic results
            h = hash(text)
            rng = np.random.default_rng(abs(h) % (2**32))
            vector = rng.uniform(-1.0, 1.0, self.dimension).tolist()
            embeddings.append(vector)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**32))
        return rng.uniform(-1.0, 1.0, self.dimension).tolist()


def get_embeddings_model() -> Embeddings:
    """
    Retrieves the embeddings provider based on available environment variables.
    Falls back to MockEmbeddings if no key is present.
    """
    if settings.cohere_api_key:
        print("Using CohereEmbeddings...")
        return CohereEmbeddings(cohere_api_key=settings.cohere_api_key)
    elif settings.openai_api_key:
        print("Using OpenAIEmbeddings...")
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    else:
        print("No Cohere/OpenAI API key detected. Using MockEmbeddings for local testing...")
        return MockEmbeddings(dimension=1536)
