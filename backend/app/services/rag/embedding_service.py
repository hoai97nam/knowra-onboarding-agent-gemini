# embedding_service.py
"""
Embedding Service - Handles vector representation generation.

Responsibilities:
- Generate embeddings for text using OpenAI models (or other providers)
- Provide consistent embedding interface (sync + async)
- Handle batching, retries, simple caching, and metrics
"""
import numpy as np

from typing import List, Optional, Callable, Any, Dict
import time
import logging
import asyncio
import functools
from sklearn.decomposition import PCA

# Optional monitoring
try:
    from prometheus_client import Histogram, Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes if prometheus not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
    Counter = None

# Import embeddings from multiple providers
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None  # for static checks / testing

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    GoogleGenerativeAIEmbeddings = None  # for static checks / testing


logger = logging.getLogger("embedding_service")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel("INFO")

EMBED_REQUEST_HIST = Histogram("embedding_requests_seconds", "Embedding request latency seconds") if Histogram else None
EMBED_REQUEST_COUNT = Counter("embedding_requests_total", "Total embedding requests") if Counter else None

# Simple retry decorator for sync and async functions
def retry_sync(max_attempts: int = 3, base_delay: float = 0.5, exceptions: tuple = (Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    logger.warning("Retryable error in %s: %s (attempt %d/%d)", fn.__name__, e, attempts, max_attempts)
                    if attempts >= max_attempts:
                        logger.exception("Max retry attempts reached for %s", fn.__name__)
                        raise
                    time.sleep(base_delay * (2 ** (attempts - 1)))
        return wrapped
    return deco

def retry_async(max_attempts: int = 3, base_delay: float = 0.5, exceptions: tuple = (Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapped(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    logger.warning("Async retryable error in %s: %s (attempt %d/%d)", fn.__name__, e, attempts, max_attempts)
                    if attempts >= max_attempts:
                        logger.exception("Max async retry attempts reached for %s", fn.__name__)
                        raise
                    await asyncio.sleep(base_delay * (2 ** (attempts - 1)))
        return wrapped
    return deco

class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI or Gemini models or compatible wrappers.

    Args:
        openai_api_key: OpenAI API key (if using OpenAI provider)
        gemini_api_key: Google Gemini API key (if using Gemini provider)
        model: embedding model name (default text-embedding-3-small for OpenAI, text-embedding-004 for Gemini)
        provider: "openai" or "gemini" (default "openai")
        batch_size: maximum texts to send per batch call
        cache_enabled: whether to use simple in-memory cache to avoid duplicate embeddings
        cache_ttl_seconds: TTL for cache entries (0 = never expire)
        embeddings_instance: pre-built embeddings instance for testing/custom wrappers
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
        pca_components: int = None,
        batch_size: int = 100,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 0,
        embeddings_instance: Optional[Any] = None,
    ):
        self.model = model
        self.provider = provider.lower()
        self.batch_size = max(1, batch_size)
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.pca_components = pca_components
        self.pca = PCA(n_components=self.pca_components) if self.pca_components else 768

        # in-memory cache: text -> (timestamp, vector)
        self._cache: Dict[str, Any] = {}

        # allow injecting a pre-built embeddings instance for testing or custom wrappers
        if embeddings_instance is not None:
            self.embeddings = embeddings_instance
        else:
            self.embeddings = self._create_embeddings_instance(
                openai_api_key, gemini_api_key
            )

        logger.info("EmbeddingService initialized provider=%s model=%s batch_size=%d cache_enabled=%s", 
                   self.provider, self.model, self.batch_size, self.cache_enabled)

    def _create_embeddings_instance(self, openai_api_key: Optional[str], gemini_api_key: Optional[str]) -> Any:
        """
        Create embeddings instance based on provider selection.
        """
        if self.provider == "gemini":
            if GoogleGenerativeAIEmbeddings is None:
                logger.warning("GoogleGenerativeAIEmbeddings not available. Falling back to OpenAI.")
                return self._create_openai_embeddings(openai_api_key)
            
            if not gemini_api_key:
                logger.warning("Gemini API key not provided. Falling back to OpenAI.")
                return self._create_openai_embeddings(openai_api_key)
            
            try:
                # Gemini expects model name in format: models/text-embedding-004
                model_name = self.model
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                
                logger.info("Creating Gemini embeddings instance with model=%s", model_name)
                return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=gemini_api_key)
            except Exception as e:
                logger.exception("Failed to create Gemini embeddings: %s. Falling back to OpenAI.", e)
                return self._create_openai_embeddings(openai_api_key)
        else:
            # Default to OpenAI
            return self._create_openai_embeddings(openai_api_key)

    def _create_openai_embeddings(self, openai_api_key: Optional[str]) -> Any:
        """
        Create OpenAI embeddings instance.
        """
        if OpenAIEmbeddings is None:
            logger.warning("OpenAIEmbeddings not available at import time. Provide embeddings_instance for runtime use.")
            return None
        
        if not openai_api_key:
            logger.warning("OpenAI API key not provided.")
            return None
        
        try:
            logger.info("Creating OpenAI embeddings instance with model=%s", self.model)
            return OpenAIEmbeddings(model=self.model, openai_api_key=openai_api_key)
        except TypeError:
            # fallback: try only model
            logger.info("Falling back to creating OpenAI embeddings with only model parameter")
            return OpenAIEmbeddings(model=self.model)
        except Exception as e:
            logger.exception("Failed to create OpenAI embeddings: %s", e)
            return None
    def _apply_pca(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Apply PCA to reduce the dimensionality of embeddings.
        If PCA is disabled or embeddings are empty, return as-is.
        """
        if not embeddings or not self.pca_components:
            return embeddings
        
        embedding_array = np.array(embeddings)
        original_dim = embedding_array.shape[1]
        
        # Don't apply PCA if target dimension >= original dimension
        if self.pca_components >= original_dim:
            logger.debug("PCA target dimension (%d) >= original dimension (%d), skipping PCA", 
                        self.pca_components, original_dim)
            return embeddings
        
        try:
            if not hasattr(self.pca, "components_"):
                logger.info("Fitting PCA with n_components=%d for embeddings of shape %s", 
                           self.pca_components, embedding_array.shape)
                self.pca.fit(embedding_array)

            reduced_embeddings = self.pca.transform(embedding_array)
            logger.debug("PCA transformation: %d -> %d dimensions", original_dim, reduced_embeddings.shape[1])
            return reduced_embeddings.tolist()
        except Exception as e:
            logger.exception("PCA transformation failed: %s. Returning original embeddings.", e)
            return embeddings

    # ---------------------
    # Cache helpers
    # ---------------------
    def _cache_get(self, text: str) -> Optional[List[float]]:
        if not self.cache_enabled:
            return None
        item = self._cache.get(text)
        if not item:
            return None
        ts, vec = item
        if self.cache_ttl_seconds > 0 and time.time() - ts > self.cache_ttl_seconds:
            # expired
            self._cache.pop(text, None)
            return None
        return vec

    def _cache_set(self, text: str, vector: List[float]) -> None:
        if not self.cache_enabled:
            return
        self._cache[text] = (time.time(), vector)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
        
        raw_embeddings = self.embeddings.embed_documents(texts)
        return self._apply_pca(raw_embeddings)
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            List[float]: Query embedding vector
        """
        raw_embedding = self.embeddings.embed_query(query)
        if not self.pca:
            return raw_embedding
        
        embedding_array = np.array(raw_embedding).reshape(1, -1)
        reduced_embedding = self.pca.transform(embedding_array)
        return reduced_embedding.flatten().tolist()
    
    def get_embeddings_instance(self):
        """
        Get the underlying LangChain embeddings instance.
        
        Useful for integration with LangChain vector stores.
        
        Returns:
            OpenAIEmbeddings: LangChain embeddings instance
        """
        return self.embeddings
    
    def get_stats(self) -> dict:
        """
        Get embedding service statistics.
        
        Returns:
            dict: Service configuration statistics
        """
        return {
            "model": self.model,
            "provider": self.provider.upper(),
            "pca_components": self.pca_components,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache_enabled
        }
    # ---------------------
    # Utility to call underlying embed method safely
    # ---------------------
    def _call_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Call the underlying embeddings instance in a best-effort compatible way.
        Tries common method names used by wrappers.
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings instance available")

        # prefer bulk embed methods
        for fn_name in ("embed_documents", "embed_texts", "embed_batch", "embed"):
            fn = getattr(self.embeddings, fn_name, None)
            if callable(fn):
                return fn(texts)

        # fallback: call embed_query per item (not efficient)
        vectors = []
        for t in texts:
            try:
                vec = self.embeddings.embed_query(t)
                vectors.append(vec)
            except Exception as e:
                logger.exception("Failed to embed text: %s", e)
                raise RuntimeError(f"Failed to embed text: {e}")
        return vectors

    async def _call_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Try to use an async embed method if available, else run sync in executor.
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings instance available")

        # try common async method names
        for fn_name in ("aembed_documents", "aembed_texts", "aembed_batch", "aembed"):
            fn = getattr(self.embeddings, fn_name, None)
            if callable(fn):
                return await fn(texts)

        # fallback to sync call in threadpool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._call_embeddings_sync, texts)

    # ---------------------
    # Public API (sync)
    # ---------------------
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string (sync). Uses cache if enabled.
        """
        if not text:
            return []

        cached = self._cache_get(text)
        if cached is not None:
            return cached

        # single item via batch call for consistency
        vecs = self.embed_texts([text])
        vec = vecs[0] if vecs else []
        if vec:
            self._cache_set(text, vec)
        return vec

    @retry_sync(max_attempts=3, base_delay=0.6)
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts (sync). Handles batching and caching per item.
        """
        if not texts:
            return []

        # prepare result placeholders and items to request
        results: List[Optional[List[float]]] = [None] * len(texts)
        to_request: List[Dict[str, Any]] = []  # list of {"idx": i, "text": t}
        for i, t in enumerate(texts):
            if not t:
                results[i] = []
                continue
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                to_request.append({"idx": i, "text": t})

        # batch the requests
        if to_request:
            idx_texts = [itm["text"] for itm in to_request]
            # split into batches of self.batch_size
            for start in range(0, len(idx_texts), self.batch_size):
                batch_texts = idx_texts[start : start + self.batch_size]
                t_start = time.time()
                if EMBED_REQUEST_COUNT:
                    EMBED_REQUEST_COUNT.inc()
                try:
                    with (EMBED_REQUEST_HIST.time() if EMBED_REQUEST_HIST else _noop_ctx()):
                        vectors_batch = self._call_embeddings_sync(batch_texts)
                        vectors_batch = self._apply_pca(vectors_batch)  # Add PCA transformation here
                except Exception:
                    logger.exception("Embedding batch call failed")
                    # fill failed batch with empty vectors
                    vectors_batch = [[] for _ in batch_texts]

                # assign back to results and cache
                for j, vec in enumerate(vectors_batch):
                    global_index = to_request[start + j]["idx"]
                    results[global_index] = vec
                    if vec:
                        self._cache_set(to_request[start + j]["text"], vec)
                # small sleep to avoid rate limits
                time.sleep(0.01)

        # convert Optional to actual list
        return [r if r is not None else [] for r in results]

    # ---------------------
    # Public API (async)
    # ---------------------
    @retry_async(max_attempts=3, base_delay=0.6)
    async def async_embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vecs = await self.async_embed_texts([text])
        vec = vecs[0] if vecs else []
        if vec:
            self._cache_set(text, vec)
        return vec

    @retry_async(max_attempts=3, base_delay=0.6)
    async def async_embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        to_request: List[Dict[str, Any]] = []
        for i, t in enumerate(texts):
            if not t:
                results[i] = []
                continue
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                to_request.append({"idx": i, "text": t})

        if to_request:
            idx_texts = [itm["text"] for itm in to_request]
            for start in range(0, len(idx_texts), self.batch_size):
                batch_texts = idx_texts[start : start + self.batch_size]
                if EMBED_REQUEST_COUNT:
                    EMBED_REQUEST_COUNT.inc()
                try:
                    vectors_batch = await self._call_embeddings_async(batch_texts)
                    vectors_batch = self._apply_pca(vectors_batch)  # Apply PCA after getting embeddings
                except Exception:
                    logger.exception("Async embedding batch failed")
                    vectors_batch = [[] for _ in batch_texts]

                for j, vec in enumerate(vectors_batch):
                    global_index = to_request[start + j]["idx"]
                    results[global_index] = vec
                    if vec:
                        self._cache_set(to_request[start + j]["text"], vec)
                await asyncio.sleep(0.01)

        return [r if r is not None else [] for r in results]

    # ---------------------
    # Helpers / compatibility
    # ---------------------
    # def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    #     """Backward-compatible alias for embed_texts"""
    #     return self.embed_texts(texts)

    # def generate_query_embedding(self, query: str) -> List[float]:
    #     """Backward-compatible alias for embed_text"""
    #     return self.embed_text(query)

    # def get_embeddings_instance(self) -> Any:
    #     """Return underlying embeddings instance (if any) for integration with vector stores"""
    #     return self.embeddings

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._cache),
        }

# noop context manager for metrics fallback
class _noop_ctx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False
    async def __aenter__(self): return None
    async def __aexit__(self, exc_type, exc, tb): return False
