from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from .document import Paragraph
from .document import Document

import qdrant_client.models as models


class Database:
    """
    Manages a database of similar document parts
    """

    def __init__(self, path):
        # Load a pre-trained Sentence Transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qdrant = QdrantClient(path=path)

    def index(self, obj):
        if isinstance(obj, Document):
            self.index_documents([obj])

    def index_documents(self, collection_name, docs):
        texts = []
        raw = []
        for doc in docs:
            texts.extend([x.preprocessed for x in doc.paragraphs])
            raw.extend([x.raw for x in doc.paragraphs])

        embeddings = self.model.encode(texts, show_progress_bar=True)

        vector_dim = len(embeddings[0])

        created = self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
        )

        if not created:
            print(f"Collection '{collection_name}' already exists.")

        # Create a list of records to store in Qdrant
        records = []
        for idx, embedding in enumerate(embeddings):
            record = models.PointStruct(
                id=idx, vector=embedding.tolist(), payload={"text": raw[idx]}
            )
            records.append(record)

        # Store the records in Qdrant
        self.qdrant.upsert(collection_name=collection_name, points=records)

    def search(self, collection_name, query_text, top_k=20):
        query_text = Paragraph(query_text).preprocessed
        query_vector = self.model.encode([query_text])[0]
        search_results = self.qdrant.search(
            collection_name=collection_name, query_vector=query_vector, limit=top_k
        )
        return search_results
