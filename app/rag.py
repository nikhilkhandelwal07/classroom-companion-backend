import os
import shutil
import time
import requests
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# ... (rest of imports)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from .config import config
except ImportError:
    from config import config

# Global dictionary to store FAISS index objects in memory keyed by course_id_division
faiss_indexes: Dict[str, FAISS] = {}

# Initialize Local Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Removed get_index_path as persistence is no longer used.

def get_vector_store(course_id: str, division: str):
    """Returns FAISS index from memory Cache."""
    key = f"{course_id}_{division}"
    
    # Check memory cache first
    if key in faiss_indexes:
        return faiss_indexes[key]
        
    return None

def process_pdf(file_path: str, course_id: str, division: str):
    """Loads a PDF, chunks it, and adds to FAISS index for course+division."""
    filename = os.path.basename(file_path)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    
    # Add metadata
    for chunk in chunks:
        chunk.metadata.update({"course_id": course_id, "division": division, "source": filename})
    
    vector_store = get_vector_store(course_id, division)
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Update cache (No disk save)
    key = f"{course_id}_{division}"
    faiss_indexes[key] = vector_store
    
    return len(chunks)

def process_url(url: str, course_id: str, division: str):
    """Scrapes a URL, chunks it, and adds to FAISS index for course+division."""
    # Use a more standard User-Agent to avoid being blocked
    loader = WebBaseLoader(
        url,
        header_template={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    log_path = os.path.join(os.path.dirname(__file__), "..", "rag_debug.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- {time.ctime()} --- URL: {url}\n")
        try:
            docs = loader.load()
            if not docs:
                f.write("DEBUG: No content scraped.\n")
                return 0
            
            content_snippet = docs[0].page_content[:500].replace('\n', ' ')
            f.write(f"DEBUG: Scraped {len(docs)} documents. Snippet: {content_snippet}\n")
            
            chunks = text_splitter.split_documents(docs)
            f.write(f"DEBUG: Split into {len(chunks)} chunks.\n")
        except Exception as e:
            f.write(f"DEBUG: Scraping Error: {str(e)}\n")
            return 0
    
    # Add metadata
    for chunk in chunks:
        chunk.metadata.update({"course_id": course_id, "division": division, "source": url})
    
    vector_store = get_vector_store(course_id, division)
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Update cache (No disk save)
    key = f"{course_id}_{division}"
    faiss_indexes[key] = vector_store
    
    return len(chunks)

def get_context(course_id: str, division: str, query: Optional[str] = None, n_results: int = 10):
    """Retrieves relevant chunks from the specific FAISS index for course+division."""
    vector_store = get_vector_store(course_id, division)
    if not vector_store:
        return ""
    
    if query:
        results = vector_store.similarity_search(query, k=n_results)
    else:
        # Return most recent chunks
        results = vector_store.similarity_search("", k=n_results)
        
    return "\n\n".join([doc.page_content for doc in results])

def remove_source(course_id: str, division: str, source_name: str):
    """Removes a specific source (file or URL) from the FAISS index."""
    vector_store = get_vector_store(course_id, division)
    if not vector_store:
        return 0
    
    # Extract all documents from the vector store
    # This is a bit of a hack but FAISS doesn't expose doc list easily
    # We use the internal docstore
    all_docs = list(vector_store.docstore._dict.values())
    
    # Filter out docs from the target source
    remaining_docs = [doc for doc in all_docs if doc.metadata.get('source') != source_name]
    
    if len(remaining_docs) == len(all_docs):
        # Nothing removed
        return 0
    
    # Clear the entire index first
    clear_session_material(course_id, division)
    
    # If there are docs left, rebuild cache
    if remaining_docs:
        new_vs = FAISS.from_documents(remaining_docs, embeddings)
        key = f"{course_id}_{division}"
        faiss_indexes[key] = new_vs
    
    # 3. Also delete original file if it exists in materials
    file_path = config.MATERIALS_BASE_PATH / f"{course_id}_{division}" / source_name
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            print(f"DEBUG: Failed to delete local file {file_path}: {e}")
    
    return len(all_docs) - len(remaining_docs)

def clear_session_material(course_id: str, division: str):
    """Removes the entire FAISS index from memory for a specific course and division."""
    key = f"{course_id}_{division}"
    
    # Clear from memory
    if key in faiss_indexes:
        del faiss_indexes[key]
    
    # Clear original materials from disk
    mat_path = config.MATERIALS_BASE_PATH / key
    if mat_path.exists():
        try:
            shutil.rmtree(mat_path)
        except Exception as e:
            print(f"DEBUG: Failed to remove materials directory {mat_path}: {e}")

def get_session_materials(course_id: str, division: str):
    """Returns a list of files and URLs currently in the session for UI sync."""
    key = f"{course_id}_{division}"
    files = []
    urls = []
    
    # Files from disk
    mat_path = config.MATERIALS_BASE_PATH / key
    if mat_path.exists():
        files = [f.name for f in mat_path.iterdir() if f.is_file()]
        
    # URLs from metadata in vector store (if loaded)
    vector_store = get_vector_store(course_id, division)
    if vector_store:
        all_docs = list(vector_store.docstore._dict.values())
        unique_urls = set()
        for doc in all_docs:
            src = doc.metadata.get('source', '')
            if src.startswith('http'):
                unique_urls.add(src)
        urls = list(unique_urls)
        
    return {"files": files, "urls": urls}

def clear_all_materials():
    """Purges ALL faculty sessions and materials. Called on logout."""
    # Clear all in-memory FAISS indices
    faiss_indexes.clear()
    
    # Clear all directories in materials base path
    if config.MATERIALS_BASE_PATH.exists():
        for item in config.MATERIALS_BASE_PATH.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                except Exception as e:
                    print(f"DEBUG: Failed to remove directory {item} during clear_all: {e}")
