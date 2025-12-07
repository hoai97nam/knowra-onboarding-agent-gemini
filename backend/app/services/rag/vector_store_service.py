# vector_store_service.py
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Pinecone client (adjust import if using different package)
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger("vector_store_service")
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel("INFO")


class VectorStoreService:
    """
    Service for managing Pinecone vector store operations.

    Provides:
    - create/load vector store
    - upsert_embeddings(items, namespace)
    - search_by_vector(vector, k, namespace)
    - semantic_search(query, k, namespace) using embeddings
    - delete_namespace(namespace) to clear memory namespace
    - helpers: is_loaded, should_rebuild, get_retriever, get_stats
    """

    def __init__(
        self,
        embeddings: Embeddings,
        pinecone_api_key: str,
        pinecone_environment: str,
        pinecone_index_name: str,
        data_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
    ):
        self.embeddings = embeddings
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.pinecone_index_name = pinecone_index_name
        self.data_dir = Path(data_dir) if data_dir else None
        self.persist_dir = Path(persist_dir) if persist_dir else Path("./backend/.vectorstore")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Pinecone client init
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
        except Exception as e:
            logger.exception("Failed to initialize Pinecone client: %s", e)
            raise

        # ensure index exists and create index handle
        self._ensure_index_exists()
        self.index = self.pc.Index(self.pinecone_index_name)

    def _get_embedding_dimension(self) -> int:
        try:
            if hasattr(self.embeddings, 'model'):
                model_name = str(self.embeddings.model).lower()
                # Gemini embeddings have 768 dimensions
                if 'gemini' in model_name or 'google' in model_name:
                    logger.info("Detected Gemini embedding model from name, using dimension=768")
                    return 768
                # OpenAI text-embedding-3-small has 1536 dimensions
                if 'text-embedding' in model_name:
                    logger.info("Detected OpenAI embedding model from name, using dimension=1536")
                    return 1536
        except Exception as e:
            logger.debug("Could not detect embedding dimension from model name: %s", e)
        
        # Default to OpenAI dimension
        logger.info("Using default OpenAI embedding dimension=1536")
        return 1536

    # -------------------------
    # Index management helpers
    # -------------------------
    def _ensure_index_exists(self) -> None:
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        except Exception:
            # Fallback if list_indexes returns names directly
            try:
                existing_indexes = self.pc.list_indexes()
            except Exception as e:
                logger.exception("Failed to list Pinecone indexes: %s", e)
                existing_indexes = []

        if self.pinecone_index_name not in existing_indexes:
            logger.info("Creating Pinecone index: %s", self.pinecone_index_name)
            try:
                # Determine dimension based on embeddings model
                dimension = self._get_embedding_dimension()
                
                logger.info("Creating index with dimension=%d", dimension)
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.pinecone_environment),
                )
                logger.info("Pinecone index created: %s with dimension=%d", self.pinecone_index_name, dimension)
            except Exception:
                logger.exception("Failed to create Pinecone index; rethrowing")
                raise
        else:
            logger.info("Pinecone index exists: %s", self.pinecone_index_name)

    def is_loaded(self) -> bool:
        """Return True if index exists and is reachable."""
        try:
            _ = self.pc.Index(self.pinecone_index_name)
            return True
        except Exception:
            return False

    def load_vectorstore(self) -> None:
        """No-op for Pinecone (index already reachable). Keep for interface parity."""
        if not self.is_loaded():
            self._ensure_index_exists()
        logger.info("Vector store load completed for index: %s", self.pinecone_index_name)

    # -------------------------
    # Change-detection (hash)
    # -------------------------
    def _calculate_files_hash(self) -> str:
        if not self.data_dir:
            raise ValueError("data_dir must be set to calculate hash")
        hasher = hashlib.sha256()
        md_files = sorted(self.data_dir.glob("*.md"))
        for file_path in md_files:
            hasher.update(file_path.name.encode("utf-8"))
            hasher.update(file_path.read_bytes())
        return hasher.hexdigest()

    def _get_stored_hash(self) -> str:
        hash_file = self.persist_dir / ".content_hash"
        if hash_file.exists():
            return hash_file.read_text().strip()
        return ""

    def _save_hash(self, content_hash: str) -> None:
        hash_file = self.persist_dir / ".content_hash"
        hash_file.write_text(content_hash)

    def should_rebuild(self) -> bool:
        """True if index empty or content hash changed."""
        try:
            idx = self.pc.Index(self.pinecone_index_name)
            stats = idx.describe_index_stats()
            total = getattr(stats, "total_vector_count", None)
            if total is None:
                # Pinecone SDK shape may differ; try dict access
                try:
                    total = stats["total_vector_count"]
                except Exception:
                    total = 0
            if total == 0:
                logger.info("Index empty; should rebuild")
                return True
        except Exception as e:
            logger.warning("Could not fetch index stats: %s; will attempt rebuild", e)
            return True

        if not self.data_dir:
            return False

        try:
            current_hash = self._calculate_files_hash()
            stored_hash = self._get_stored_hash()
            return current_hash != stored_hash
        except Exception as e:
            logger.exception("Hash check failed: %s", e)
            return True

    # -------------------------
    # Create / upsert documents
    # -------------------------
    def create_vectorstore(self, documents: List[Document], namespace: Optional[str] = None) -> None:
        """
        Create vector store contents from documents (overwrite behavior not enforced).
        Use this for initial bulk build.
        """
        logger.info("Creating vectorstore with %d documents (namespace=%s)", len(documents), namespace)
        vectors = []
        for i, doc in enumerate(documents):
            # generate embedding (try common method names)
            text = getattr(doc, "page_content", str(doc))
            try:
                emb = getattr(self.embeddings, "embed_query", None)
                if callable(emb):
                    vector = self.embeddings.embed_query(text)
                else:
                    vector = self.embeddings.embed_text(text)
            except Exception:
                # last fallback: try generic method name
                vector = self.embeddings.embed_text(text)

            vid = f"doc_{i}"
            metadata = {"text": text}
            try:
                # merge any doc.metadata if exists
                meta_source = getattr(doc, "metadata", None)
                if isinstance(meta_source, dict):
                    metadata.update(meta_source)
            except Exception:
                pass

            vectors.append((vid, vector, metadata))

        # upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                # Pinecone upsert accepts list of tuples (id, vector, metadata)
                if namespace:
                    try:
                        self.index.upsert(vectors=batch, namespace=namespace)
                    except TypeError:
                        # Namespace parameter not supported, try without it
                        logger.debug("Namespace parameter not supported in upsert, retrying without it")
                        self.index.upsert(vectors=batch)
                else:
                    self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch of {len(batch)} vectors successfully")
            except Exception as e:
                logger.exception(f"Failed to upsert batch of {len(batch)} vectors: {e}")
                raise

        # save hash if data_dir provided
        if self.data_dir:
            try:
                content_hash = self._calculate_files_hash()
                self._save_hash(content_hash)
            except Exception:
                logger.exception("Failed to calculate/save files hash")

        logger.info("Vectorstore creation complete")

    def upsert_embeddings(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        """
        Upsert embeddings into Pinecone. Items expected as dicts:
            {"id": str, "vector": List[float], "metadata": dict, "text": str}
        """
        if not items:
            return
        
        expected_dimension = self._get_embedding_dimension()
        logger.info(f"Upserting {len(items)} embeddings with expected dimension={expected_dimension}")
        
        # convert to Pinecone tuples
        tuples = []
        for it in items:
            vid = it.get("id")
            vector = it.get("vector")
            metadata = it.get("metadata", {})
            
            # embed text if vector missing and text present
            if not vector and it.get("text"):
                try:
                    emb_fn = getattr(self.embeddings, "embed_query", None) or getattr(self.embeddings, "embed_text", None)
                    vector = emb_fn(it["text"])
                except Exception as e:
                    logger.exception(f"Failed to create embedding for upsert item: {e}")
                    continue
            
            # Validate vector dimension
            if vector:
                if len(vector) != expected_dimension:
                    logger.warning(f"Vector dimension mismatch for item {vid}: got {len(vector)}, expected {expected_dimension}. Skipping.")
                    continue
            
            # include text inside metadata for retrieval convenience
            if it.get("text"):
                metadata = dict(metadata)
                metadata["text"] = it["text"]
            
            if vector and vid:
                tuples.append((vid, vector, metadata))

        if not tuples:
            logger.warning("No valid tuples to upsert after validation")
            return

        # batch upsert
        batch_size = 100
        for i in range(0, len(tuples), batch_size):
            batch = tuples[i : i + batch_size]
            try:
                if namespace:
                    try:
                        self.index.upsert(vectors=batch, namespace=namespace)
                    except TypeError:
                        # Namespace parameter not supported, try without it
                        logger.debug("Namespace parameter not supported in upsert, retrying without it")
                        self.index.upsert(vectors=batch)
                else:
                    self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch of {len(batch)} embeddings successfully")
            except Exception as e:
                logger.exception(f"Failed to upsert batch of {len(batch)} embeddings: {e}")
                raise

    # -------------------------
    # Search helpers
    # -------------------------
    def search_by_vector(self, vector: List[float], k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search by raw vector and return list of items {id, text, metadata, score}
        """
        try:
            query_args = dict(vector=vector, top_k=k, include_metadata=True)
            if namespace:
                query_args["namespace"] = namespace
            res = self.index.query(**query_args)
        except Exception:
            # fallback calling without namespace (older SDK shapes)
            try:
                res = self.index.query(vector=vector, top_k=k, include_metadata=True)
            except Exception as e:
                logger.exception("Vector query failed: %s", e)
                return []

        matches = getattr(res, "matches", None) or res.get("matches", []) if isinstance(res, dict) else []
        results = []
        for m in matches:
            md = getattr(m, "metadata", None) or m.get("metadata", {}) if isinstance(m, dict) else {}
            text = md.pop("text", "") if isinstance(md, dict) else ""
            results.append({"id": getattr(m, "id", None) or m.get("id"), "text": text, "metadata": md, "score": getattr(m, "score", None) or m.get("score")})
        return results

    def semantic_search(self, query: str, k: int = 5, namespace: Optional[str] = None) -> List[Document]:
        """
        High-level semantic search by query text. Returns langchain Documents.
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone with proper error handling
            try:
                if namespace:
                    results = self.index.query(
                        vector=query_embedding,
                        top_k=k,
                        include_metadata=True,
                        namespace=namespace
                    )
                else:
                    results = self.index.query(
                        vector=query_embedding,
                        top_k=k,
                        include_metadata=True
                    )
            except TypeError:
                # Fallback: namespace parameter not supported
                logger.debug("Namespace parameter not supported, retrying without it")
                results = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True
                )
            
            documents = []
            matches = getattr(results, 'matches', None) or results.get('matches', []) if isinstance(results, dict) else []
            
            for match in matches:
                try:
                    metadata = match.metadata.copy() if hasattr(match, 'metadata') else match.get('metadata', {}).copy()
                    text = metadata.pop("text", "")
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to process match: {e}")
                    continue
            
            return documents
        except Exception as e:
            logger.exception(f"semantic_search failed: {e}")
            return []

    # -------------------------
    # Utility / maintenance
    # -------------------------
    def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a namespace. Use with caution.
        Pinecone may not support deleting a namespace directly via SDK; we attempt deletion by filter.
        """
        try:
            # Some Pinecone SDKs support delete(delete_all=True, namespace=namespace)
            try:
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info("Deleted namespace via delete_all flag: %s", namespace)
                return
            except TypeError:
                # fallback: delete by metadata filter that matches namespace if you stored namespace in metadata
                self.index.delete(filter={"_namespace": namespace})
                logger.info("Attempted namespace deletion via filter: %s", namespace)
        except Exception:
            logger.exception("Failed to delete namespace: %s", namespace)

    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        """
        Return a simple retriever configuration object. The pipeline uses semantic_search for actual retrieval.
        """
        if not self.is_loaded():
            raise ValueError("Vector store not initialized.")
        return {"index_name": self.pinecone_index_name, "k": k, "search_type": search_type}

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Backward-compatible method name for older code: delegates to semantic_search."""
        return self.semantic_search(query, k=k)

    def get_stats(self) -> dict:
        vector_count = 0
        try:
            idx = self.pc.Index(self.pinecone_index_name)
            stats = idx.describe_index_stats()
            vector_count = getattr(stats, "total_vector_count", None) or (stats.get("total_vector_count") if isinstance(stats, dict) else 0)
        except Exception as e:
            logger.warning("Could not fetch index stats: %s", e)

        return {
            "index_name": self.pinecone_index_name,
            "vector_store_type": "Pinecone",
            "environment": self.pinecone_environment,
            "vector_count": vector_count,
            "last_hash": (self._get_stored_hash()[:8] if self.data_dir else "N/A"),
        }
