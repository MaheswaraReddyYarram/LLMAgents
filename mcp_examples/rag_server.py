import os
import logging
from mcp.server.fastmcp import FastMCP
from typing import List

# Langchain imports for RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb import Settings
from dotenv import load_dotenv
# --- Configuration ---
# This is the directory where the Chroma vector store will be persisted.
# Making it persistent allows the data to survive server restarts and be
# accessible by different tools (e.g., a future query tool)
CHROMA_PERSIST_DIR = "rag_chroma_db"

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

mcp = FastMCP("RAGAssistant")

@mcp.tool()
def ingest_document(file_path:str) -> str:
    """
    Loads a document from a file path, splits it into chunks, generates
    embeddings using Google's model, and stores them in a persistent
    Chroma vector store for later retrieval.

    This tool is the first step in the RAG pipeline. It prepares the knowledge
    base that can be queried by another tool.

    Args:
        file_path: The absolute or relative path to the text document.
                   For example: "/usercode/Guides/employee_handbook.txt".

    Returns:
        A string confirming the successful ingestion and the number of chunks processed,
        or an error message if the process fails.
    """
    # validate file path exists
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."

    try:
        # Initialize embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        # Load document
        loader = TextLoader(file_path, encoding='utf-8')
        docuemnts = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docuemnts)

        if not chunks:
            print(f"Could not split document into chunks: {file_path}")

        # Create or load persistent Chroma vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR,
            client_settings=Settings(anonymized_telemetry=False)
        )

        file_name = os.path.basename(file_path)
        return f"Successfully ingested file '{file_name}' into the vector store."

    except Exception as e:
        logging.error(f"Error during document ingestion: {str(e)}")
        return f"Error during ingestion: {str(e)}"



@mcp.tool()
def query_rag_store(query: str) -> str:
    """
    Queries the persistent Chroma vector store to find the most relevant
    document chunks for a given user query.

    This tool loads the existing vector database, performs a similarity search,
    and returns the combined text of the most relevant chunks.

    Args:
        query: The user's question or search term (e.g., "how do I apply for leave?").

    Returns:
        A string containing the concatenated content of the most relevant document
        chunks, or an error/status message if no relevant information is found.
    """

    # check if vector store directory exists
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return "The vector store is empty. Please run ingest_document tool first to create knowledge base."

    try:
        # Initialize embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

        # load persistent Chroma vector store
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_model,
            client_settings=Settings(anonymized_telemetry=False)
        )

        #perform similarity search
        results = vector_store.similarity_search(query, k=3)  # retrieve top 3 relevant chunks

        if not results:
            return "No relevant information found in the knowledge base."

        # Combine the content of the found documents into a single string for the agent
        # Using a separator helps the LLM distinguish between different retrieved chunks
        context = "\n---\n".join([doc.page_content for doc in results])

        return context
    except Exception as e:
        logging.error(f"Error during RAG query: {str(e)}")
        return f"Error during query: {str(e)}"

if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.DEBUG)
    mcp.run(transport="stdio")




