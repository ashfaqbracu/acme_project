import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class EmbeddingsHandler:
    def __init__(self, model_name: str = "shihab17/bangla-sentence-transformer", 
                 db_path: str = "./vectorstore"):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
    
    def create_collection(self, collection_name: str = "jmpwash"):
        """Create or get collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text"""
        return self.model.encode([text])[0]
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for multiple chunks"""
        texts = [chunk['text'] for chunk in chunks]
        return self.model.encode(texts)
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Add chunks to vector database"""
        if not self.collection:
            self.create_collection()
        
        embeddings = self.embed_chunks(chunks)
        
        ids = [chunk['document_id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{
            'language': chunk['language'],
            'source_file': chunk['source_file'],
            'chunk_id': chunk['chunk_id'],
            'token_count': chunk['token_count']
        } for chunk in chunks]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Upserted {len(chunks)} chunks to collection")
    
    def search(self, query: str, k: int = 4, language_filter: str = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if not self.collection:
            self.create_collection()
        
        query_embedding = self.embed_text(query)
        
        where_clause = {}
        if language_filter:
            where_clause['language'] = language_filter
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
