from setuptools import setup, find_packages

setup(
    name="jmpwash-rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "streamlit>=1.28.1",
        "langchain>=0.1.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.15",
        "transformers>=4.35.2",
        "torch>=2.1.0",
        "pdfminer.six>=20221105",
        "beautifulsoup4>=4.12.2",
        "nltk>=3.8.1",
        "pandas>=2.1.3",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "fasttext>=0.9.2",
        "bnlp-toolkit>=3.6.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.1",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.11",
    author="Fellowship Candidate",
    description="Multilingual RAG system for JMP Wash documents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
