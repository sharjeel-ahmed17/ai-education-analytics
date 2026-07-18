import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from edu_copilot.config import settings

class QdrantVectorStore:
    """
    Wrapper around Qdrant Vector Database. Supports in-memory database
    for local development and cloud/on-prem connection options.
    """
    def __init__(self, embeddings_model) -> None:
        self.embeddings = embeddings_model
        url = settings.qdrant_url
        api_key = settings.qdrant_api_key
        
        # Fallback to local in-memory Qdrant instance if URL is missing
        if url:
            print(f"Connecting to Qdrant cluster at {url}...")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            print("Qdrant URL not configured. Initializing in-memory Qdrant database...")
            self.client = QdrantClient(location=":memory:")
            
        self.collection_name = "student_analytics_collection"
        self._ensure_collection()
        
        # Instantiate LangChain community wrapper
        self.db = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def _ensure_collection(self) -> None:
        """
        Validates collection existence, creating it dynamically with correct vector dimensions.
        """
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Query the embeddings model once to get dimension size dynamically
            try:
                test_embedding = self.embeddings.embed_query("test query")
                vector_dim = len(test_embedding)
            except Exception as e:
                print(f"Error querying embedding dimensions: {e}. Defaulting to 1536.")
                vector_dim = 1536 # Default for OpenAI text-embedding-ada-002
                
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim, 
                    distance=models.Distance.COSINE
                ),
            )
            print(f"Created vector store collection '{self.collection_name}' with size={vector_dim}.")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Upserts document chunks into the Qdrant index.
        """
        if not documents:
            return
        self.db.add_documents(documents)

    def search_student_docs(self, query: str, student_id: str, k: int = 5) -> List[Document]:
        """
        Retrieves document chunks matching a query, filtered by the student's ID.
        
        Args:
            query (str): The search query.
            student_id (str): Associated student ID to constrain search scope.
            k (int): Number of chunks to retrieve.
            
        Returns:
            List[Document]: Retreived LangChain document chunks.
        """
        # Scopes vector search strictly to the relevant student using metadata
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.student_id",
                    match=models.MatchValue(value=student_id)
                )
            ]
        )
        
        return self.db.similarity_search(
            query=query,
            k=k,
            filter=search_filter
        )
