
from vectordb.embed_model import CustomEmbeddingFunction
import chromadb
import logging

from config import CHROMA_HOST, CHROMA_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_fn = CustomEmbeddingFunction()

def retrieve_wiki_context(task: str, project_id: int, k: int = 3) -> str:
    logger.info(f"Retrieving wiki context for task: '{task}' and project_id: {project_id}")
    query_embedding = embedding_fn.embed_query(task)
    logger.info(f"Generated query embedding of dimension {len(query_embedding)}")
    
    COLLECTION_NAME = "wiki_summaries"

    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.error(f"ChromaDB connection or query failed: {e}")
        return f"Failed to get context from ChromaDB for '{task}'"


    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"project_id": project_id}
    )


    # documents = results.get("documents", [[]])[0]
    # return "\n".join(documents)

    documents = results.get("documents", [[]])
    # if not documents or not documents[0]:
    #     return ""

    # return "\n".join(documents[0])

    if not documents or not documents[0]:
        logger.warning(f"[wiki_retriever] No documents found for task: {task} (project {project_id})")
        return task
    
