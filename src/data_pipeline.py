import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import fasttext
from bnlp import BasicTokenizer
from transformers import AutoTokenizer

class DataPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bn_tokenizer = BasicTokenizer()
        self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.lang_detector = None
        self._setup_nltk()
        self._setup_language_detection()
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _setup_language_detection(self):
        """Setup language detection model"""
        try:
            # Download fasttext language detection model if not exists
            model_path = "lid.176.bin"
            if not os.path.exists(model_path):
                os.system(f"wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/{model_path}")
            self.lang_detector = fasttext.load_model(model_path)
        except Exception as e:
            print(f"Warning: Could not load language detection model: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = extract_text(pdf_path)
            return self.normalize_text(text)
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_from_html(self, html_path: str) -> str:
        """Extract text from HTML file"""
        try:
            with open(html_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                return self.normalize_text(text)
        except Exception as e:
            print(f"Error extracting text from {html_path}: {e}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for both Bangla and English"""
        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        # Normalize Bangla digits
        bengali_digits = '০১২৩৪৫৬৭৮৯'
        english_digits = '0123456789'
        for bn_digit, en_digit in zip(bengali_digits, english_digits):
            text = text.replace(bn_digit, en_digit)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not self.lang_detector:
            # Fallback: simple script detection
            bangla_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
            return 'bn' if bangla_chars > len(text) * 0.3 else 'en'
        
        try:
            predictions = self.lang_detector.predict(text.replace('\n', ' '))
            lang_code = predictions[0][0].replace('__label__', '')
            return 'bn' if lang_code == 'bn' else 'en'
        except:
            return 'en'
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into chunks with language detection"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Process current chunk
                lang = self.detect_language(current_chunk)
                chunks.append({
                    'text': current_chunk.strip(),
                    'language': lang,
                    'token_count': current_tokens,
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(word_tokenize(current_chunk))
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            lang = self.detect_language(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'language': lang,
                'token_count': current_tokens,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def process_documents(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Process all documents in input directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        
        for file_path in input_path.glob("*"):
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.html', '.htm']:
                text = self.extract_text_from_html(str(file_path))
            else:
                continue
            
            if not text:
                continue
            
            chunks = self.chunk_text(text)
            
            # Add metadata
            for chunk in chunks:
                chunk.update({
                    'source_file': file_path.name,
                    'source_path': str(file_path),
                    'document_id': f"{file_path.stem}_{chunk['chunk_id']}"
                })
            
            all_chunks.extend(chunks)
        
        # Save processed chunks
        output_file = output_path / 'processed_chunks.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        return all_chunks
