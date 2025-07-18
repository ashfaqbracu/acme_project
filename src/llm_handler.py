import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMHandler:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]], 
                    language: str = "auto") -> str:
        """Build RAG prompt with context and question"""
        
        # Determine response language
        if language == "auto":
            # Simple heuristic: detect Bangla characters
            bangla_chars = sum(1 for char in question if '\u0980' <= char <= '\u09FF')
            language = 'bn' if bangla_chars > len(question) * 0.2 else 'en'
        
        # Build context
        context = ""
        for i, chunk in enumerate(context_chunks, 1):
            context += f"[S{i}] {chunk['text']}\n\n"
        
        # Language-specific instructions
        if language == 'bn':
            system_prompt = """আপনি JMP Wash Assistant। আপনার কাজ হল প্রদত্ত তথ্যের ভিত্তিতে প্রশ্নের উত্তর দেওয়া।
- সর্বদা বাংলায় উত্তর দিন
- তথ্যের সূত্র [S1], [S2] ইত্যাদি দিয়ে উল্লেখ করুন
- যদি তথ্য না থাকে তাহলে স্পষ্ট করে বলুন"""
        else:
            system_prompt = """You are JMP Wash Assistant. Your task is to answer questions based on the provided information.
- Always answer in English
- Cite sources as [S1], [S2], etc.
- If information is not available, clearly state so"""
        
        prompt = f"""{system_prompt}

Context Information:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7) -> str:
        """Generate response using the LLM"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", 
                                     max_length=1024, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        generated_text = response[len(prompt):].strip()
        
        return generated_text
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]], 
                       language: str = "auto") -> Dict[str, Any]:
        """Generate answer with context and citations"""
        
        prompt = self.build_prompt(question, context_chunks, language)
        answer = self.generate_response(prompt)
        
        # Extract citations
        citations = []
        for i, chunk in enumerate(context_chunks, 1):
            if f"[S{i}]" in answer:
                citations.append({
                    'id': f"S{i}",
                    'text': chunk['text'],
                    'source': chunk['metadata'].get('source_file', 'Unknown'),
                    'language': chunk['metadata'].get('language', 'unknown')
                })
        
        return {
            'question': question,
            'answer': answer,
            'citations': citations,
            'language': language
        }
