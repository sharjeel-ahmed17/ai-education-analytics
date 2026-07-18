import os
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from edu_copilot.config import settings


class QdrantVectorStore:
    def __init__(self, embeddings_model) -> None:
        self.embeddings = embeddings_model
        url = settings.qdrant_url
        api_key = settings.qdrant_api_key

        if url:
            print(f"Connecting to Qdrant cluster at {url}...")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            print("Qdrant URL not configured. Initializing in-memory Qdrant database...")
            self.client = QdrantClient(location=":memory:")

        self.collection_name = "student_analytics_collection"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            try:
                test_embedding = self.embeddings.embed_query("test query")
                vector_dim = len(test_embedding)
            except Exception as e:
                print(f"Error querying embedding dimensions: {e}. Defaulting to 1536.")
                vector_dim = 1536

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE
                ),
            )
            print(f"Created vector store collection '{self.collection_name}' with size={vector_dim}.")

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"metadata": metadata, "page_content": text}
            )
            for vector, metadata, text in zip(vectors, metadatas, texts)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_student_docs(self, query: str, student_id: str, k: int = 5) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.student_id",
                    match=models.MatchValue(value=student_id)
                )
            ]
        )
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=k,
            with_payload=True
        )
        return [
            Document(
                page_content=point.payload.get("page_content", ""),
                metadata=point.payload.get("metadata", {})
            )
            for point in result.points
        ]
