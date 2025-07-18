from typing import List, Dict, Any, Optional
from src.embeddings import EmbeddingsHandler
from src.llm_handler import LLMHandler
import re

class RAGRetriever:
    def __init__(self, embeddings_handler: EmbeddingsHandler, 
                 llm_handler: LLMHandler):
        self.embeddings = embeddings_handler
        self.llm = llm_handler
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        bangla_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        return 'bn' if bangla_chars > len(text) * 0.2 else 'en'
    
    def retrieve_and_answer(self, question: str, k: int = 4, 
                           language_filter: Optional[str] = None) -> Dict[str, Any]:
        """Main RAG pipeline"""
        
        # Detect question language
        detected_lang = self.detect_language(question)
        
        # Search for relevant chunks
        results = self.embeddings.search(
            query=question,
            k=k,
            language_filter=language_filter
        )
        
        if not results:
            return {
                'question': question,
                'answer': 'দুঃখিত, এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই।' if detected_lang == 'bn' 
                         else 'Sorry, I don\'t have enough information to answer this question.',
                'citations': [],
                'language': detected_lang
            }
        
        # Generate answer
        response = self.llm.answer_question(question, results, detected_lang)
        
        # Add retrieval scores
        for i, citation in enumerate(response['citations']):
            if i < len(results):
                citation['relevance_score'] = 1 - results[i]['distance']
        
        return response
    
    def get_similar_documents(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get similar documents without generating answer"""
        return self.embeddings.search(query, k=k)
