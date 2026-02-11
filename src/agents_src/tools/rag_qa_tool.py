import logging
from typing import Dict, List

import chromadb
from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from src.agents_src.config.agent_settings import AgentSettings

logger = logging.getLogger(__name__)

logger.info("Loading Agent settings...")
settings = AgentSettings()

logger.info("Initializing embedding model...")
embed_model = HuggingFaceEmbedding()

logger.info("Configuring Groq LLM...")
Settings.llm = Groq(
    model=settings.MODEL_NAME,
    temperature=settings.MODEL_TEMPERATURE,
    api_key=settings.GROQ_API_KEY,
)

logger.info("Connecting to Chroma vector store...")
db = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

logger.info("Loading VectorStoreIndex...")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

@tool
def rag_query_tool(query: str) -> Dict[str, List[str] | str]:
    """
    Execute a Retrieval-Augmented Generation (RAG) query.

    Args:
        query (str): User input question.

    Returns:
        Dict[str, Any]:
            {
                "answer": str,
                "source_files": List[str]
            }
    """

    logger.info(f"Received query: {query}")

    try:
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        source_files = []

        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = node.metadata or {}
                file_name = metadata.get("file_name")
                if file_name:
                    source_files.append(file_name)

        source_files = list(set(source_files))  # remove duplicates

        return {
            "answer": response.response,
            "source_files": source_files,
        }

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "answer": "An error occurred while processing the query.",
            "source_files": [],
        }

