import streamlit as st
import requests
import json
from typing import Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="JMP Wash RAG Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def query_api(question: str, k: int = 4, language_filter: str = None) -> Dict[str, Any]:
    """Query the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "k": k,
                "language_filter": language_filter
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def upload_document(file):
    """Upload document to API"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload Error: {e}")
        return None

def main():
    st.title("🌊 JMP Wash RAG Assistant")
    st.markdown("Ask questions about JMP Wash documents in **Bangla** or **English**")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Language filter
        language_filter = st.selectbox(
            "Language Filter",
            ["All Languages", "Bangla Only", "English Only"],
            index=0
        )
        
        lang_map = {
            "All Languages": None,
            "Bangla Only": "bn",
            "English Only": "en"
        }
        
        # Number of results
        k = st.slider("Number of Results", 1, 10, 4)
        
        # Document upload
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['pdf', 'html', 'htm']
        )
        
        if uploaded_file is not None:
            if st.button("Upload"):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.success(f"Document uploaded successfully! Added {result['chunks_added']} chunks.")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask a Question")
        
        # Sample questions
        st.subheader("Sample Questions")
        sample_questions = [
            "What is JMP Wash?",
            "JMP Wash কি?",
            "What are the key findings in the latest report?",
            "সর্বশেষ প্রতিবেদনে মূল অনুসন্ধানগুলি কী?",
            "How has water access improved globally?",
            "বিশ্বব্যাপী পানির অ্যাক্সেস কিভাবে উন্নত হয়েছে?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"📝 {question}", key=f"sample_{i}"):
                st.session_state.question = question
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            value=st.session_state.get('question', ''),
            height=100,
            placeholder="Ask about JMP Wash data, reports, or findings..."
        )
        
        if st.button("🔍 Ask Question", type="primary"):
            if question.strip():
                with st.spinner("Searching and generating answer..."):
                    response = query_api(
                        question=question,
                        k=k,
                        language_filter=lang_map[language_filter]
                    )
                    
                    if response:
                        st.session_state.last_response = response
                        st.rerun()
    
    with col2:
        st.header("Response")
        
        if 'last_response' in st.session_state:
            response = st.session_state.last_response
            
            # Display answer
            st.subheader("Answer")
            st.write(response['answer'])
            
            # Display metadata
            st.subheader("Metadata")
            st.write(f"**Language:** {response['language']}")
            st.write(f"**Processing Time:** {response['processing_time']:.2f}s")
            
            # Display citations
            if response['citations']:
                st.subheader("Sources")
                for citation in response['citations']:
                    with st.expander(f"{citation['id']} - {citation['source']}"):
                        st.write(f"**Language:** {citation['language']}")
                        if citation.get('relevance_score'):
                            st.write(f"**Relevance:** {citation['relevance_score']:.3f}")
                        st.write("**Text:**")
                        st.write(citation['text'])
        else:
            st.info("👆 Ask a question to see the response here")
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            if chat['citations']:
                with st.expander("Show Sources"):
                    for citation in chat['citations']:
                        st.write(f"**{citation['id']}:** {citation['source']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            'question': prompt,
            'answer': 'Thinking...',
            'citations': []
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = query_api(
                    question=prompt,
                    k=k,
                    language_filter=lang_map[language_filter]
                )
                
                if response:
                    st.write(response['answer'])
                    
                    # Update chat history
                    st.session_state.chat_history[-1] = {
                        'question': prompt,
                        'answer': response['answer'],
                        'citations': response['citations']
                    }
                    
                    if response['citations']:
                        with st.expander("Show Sources"):
                            for citation in response['citations']:
                                st.write(f"**{citation['id']}:** {citation['source']}")

if __name__ == "__main__":
    main()
