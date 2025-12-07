"""
RAG (Retrieval-Augmented Generation) service implementation.
Handles vector store management, document processing, and agent interactions.

UPDATED: Now uses the enhanced RAG pipeline with optimized routing, safety, and memory.
"""
from typing import Dict, Any, AsyncGenerator, Optional

# Try to import enhanced pipeline first, fallback to basic if not available
try:
    from app.services.rag.enhanced_rag_pipeline_service import EnhancedRAGPipelineService
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ENHANCED_PIPELINE_AVAILABLE = False

# Fallback import
if not ENHANCED_PIPELINE_AVAILABLE:
    from app.services.rag import RAGPipelineService


class RAGService:
    """
    RAG Service - Enhanced version with intelligent routing, safety, and advanced memory.
    
    Provides:
    - Intelligent routing (heuristic + semantic)
    - Safety analysis and content moderation
    - Advanced memory management with fact extraction
    - Session and project scoping
    - Automated memory cleanup
    - Document loading and chunking
    - Vector store management with auto-rebuild
    - Conversational agent with enhanced memory
    - Streaming and non-streaming responses
    """
    
    def __init__(
        self,
        data_dir: str = None,
        persist_dir: str = None,
        openai_api_key: str = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            data_dir: Directory containing markdown knowledge base files
            persist_dir: Directory for vector store persistence
            openai_api_key: OpenAI API key
        """
        # Check if we can use enhanced pipeline
        if ENHANCED_PIPELINE_AVAILABLE:
            self._initialize_enhanced_pipeline(data_dir, persist_dir, openai_api_key)
        else:
            self._initialize_basic_pipeline(data_dir, persist_dir, openai_api_key)
    
    def _initialize_enhanced_pipeline(self, data_dir, persist_dir, openai_api_key):
        """Initialize enhanced RAG pipeline."""
        from app.core.config import settings
        basic_pipeline_instance = self.create_basic_pipeline_instance(data_dir, persist_dir, openai_api_key)

        self._pipeline = EnhancedRAGPipelineService(
            data_dir=data_dir or settings.DATA_DIR,
            persist_dir=persist_dir or settings.VECTOR_STORE_DIR,
            openai_base_url=settings.OPENAI_BASE_URL,
            openai_api_key=openai_api_key or settings.OPENAI_API_KEY,
            gemini_api_key=settings.GEMINI_API_KEY,
            embedding_api_key=settings.OPENAI_EMBEDDING_API_KEY or settings.OPENAI_API_KEY,
            pinecone_api_key=settings.PINECONE_API_KEY,
            pinecone_environment=settings.PINECONE_ENVIRONMENT,
            pinecone_index_name=settings.PINECONE_INDEX_NAME,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL if settings.PROVIDER == "openai" else settings.GEMINI_EMBEDDING_MODEL,
            llm_model=settings.OPENAI_MODEL if settings.PROVIDER == "openai" else settings.GEMINI_MODEL,
            llm_temperature=settings.OPENAI_TEMPERATURE,
            retriever_k=settings.RETRIEVER_K,
            memory_window=settings.MEMORY_WINDOW,
            pca_components=settings.PCA,
            auto_rebuild=settings.AUTO_REBUILD_ENABLED,
            chunk_add_section_headers=settings.CHUNK_ADD_SECTION_HEADERS,
            chunk_extract_metadata=settings.CHUNK_EXTRACT_METADATA,
            provider=settings.PROVIDER,
            # Enhanced features
            enable_routing=True,
            enable_safety=False,
            enable_fact_extraction=True,
            enable_memory_cleanup=True,
            enable_session_management=True,
            max_memory_per_session=1000,
            memory_cleanup_interval_hours=6,
            redis_url=getattr(settings, 'REDIS_URL', None),
            rebuild_if_needed=basic_pipeline_instance.rebuild_if_needed
        )
        self._is_enhanced = True
        
    def _initialize_basic_pipeline(self, data_dir, persist_dir, openai_api_key):
        """Initialize basic RAG pipeline as fallback."""
        from app.services.rag import RAGPipelineService
        from app.core.config import settings
        self._pipeline = RAGPipelineService(
            data_dir=data_dir or settings.DATA_DIR,
            persist_dir=persist_dir or settings.VECTOR_STORE_DIR,
            openai_base_url=settings.OPENAI_BASE_URL,
            openai_api_key=openai_api_key or settings.OPENAI_API_KEY,
            gemini_api_key=settings.GEMINI_API_KEY,
            embedding_api_key=settings.OPENAI_EMBEDDING_API_KEY or settings.OPENAI_API_KEY,
            pinecone_api_key=settings.PINECONE_API_KEY,
            pinecone_environment=settings.PINECONE_ENVIRONMENT,
            pinecone_index_name=settings.PINECONE_INDEX_NAME,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL if settings.PROVIDER == "openai" else settings.GEMINI_EMBEDDING_MODEL,
            llm_model=settings.OPENAI_MODEL if settings.PROVIDER == "openai" else settings.GEMINI_MODEL,
            llm_temperature=settings.OPENAI_TEMPERATURE,
            retriever_k=settings.RETRIEVER_K,
            memory_window=settings.MEMORY_WINDOW,
            pca_components=settings.PCA,
            auto_rebuild=settings.AUTO_REBUILD_ENABLED,
            chunk_add_section_headers=settings.CHUNK_ADD_SECTION_HEADERS,
            chunk_extract_metadata=settings.CHUNK_EXTRACT_METADATA,
            provider=settings.PROVIDER
        )
        self._is_enhanced = False
    
    def create_basic_pipeline_instance(self, data_dir, persist_dir, openai_api_key):
        """Initialize basic RAG pipeline as fallback."""
        from app.services.rag import RAGPipelineService
        from app.core.config import settings
        
        return RAGPipelineService(
            data_dir=data_dir or settings.DATA_DIR,
            persist_dir=persist_dir or settings.VECTOR_STORE_DIR,
            openai_base_url=settings.OPENAI_BASE_URL,
            openai_api_key=openai_api_key or settings.OPENAI_API_KEY,
            gemini_api_key=settings.GEMINI_API_KEY,
            embedding_api_key=settings.OPENAI_EMBEDDING_API_KEY or settings.OPENAI_API_KEY,
            pinecone_api_key=settings.PINECONE_API_KEY,
            pinecone_environment=settings.PINECONE_ENVIRONMENT,
            pinecone_index_name=settings.PINECONE_INDEX_NAME,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL if settings.PROVIDER == "openai" else settings.GEMINI_EMBEDDING_MODEL,
            llm_model=settings.OPENAI_MODEL if settings.PROVIDER == "openai" else settings.GEMINI_MODEL,
            llm_temperature=settings.OPENAI_TEMPERATURE,
            retriever_k=settings.RETRIEVER_K,
            memory_window=settings.MEMORY_WINDOW,
            pca_components=settings.PCA,
            auto_rebuild=settings.AUTO_REBUILD_ENABLED,
            chunk_add_section_headers=settings.CHUNK_ADD_SECTION_HEADERS,
            chunk_extract_metadata=settings.CHUNK_EXTRACT_METADATA,
            provider=settings.PROVIDER
        )    

    def rebuild_if_needed(self) -> bool:
        """
        Check and rebuild vector store if files changed.
        
        Returns:
            bool: True if rebuild occurred
        """
        return self._pipeline.rebuild_if_needed()
    
    async def stream_response(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from agent.
        
        Args:
            query: User query
            session_id: Optional session identifier
            project_id: Optional project identifier
            
        Yields:
            str: Response chunks
        """
        if self._is_enhanced:
            # Use enhanced streaming with session support
            async for chunk in self._pipeline.stream_response(query, session_id, project_id):
                yield chunk
        else:
            # Fallback to basic streaming
            async for chunk in self._pipeline.stream_response(query):
                yield chunk
    
    def chat(self, query: str) -> str:
        """
        Non-streaming chat response (legacy method).
        
        Args:
            query: User query
            
        Returns:
            str: Complete response
        """
        if self._is_enhanced:
            # Use enhanced chat method
            response, _ = self._pipeline.chat_with_metadata(query)
            return response
        else:
            return self._pipeline.chat(query)
    
    def chat_with_metadata(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Chat with metadata return.
        
        Args:
            query: User query
            session_id: Optional session identifier
            project_id: Optional project identifier
            
        Returns:
            Tuple of (response_text, metadata)
        """
        if self._is_enhanced:
            return self._pipeline.chat_with_metadata(query, session_id, project_id)
        else:
            # Fallback - basic response with minimal metadata
            response = self._pipeline.chat(query)
            metadata = {
                'session_id': session_id,
                'project_id': project_id,
                'enhanced_features': False
            }
            return response, metadata
    
    # Session management methods (enhanced pipeline only)
    def create_session(
        self, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_name: Optional[str] = None
    ):
        """Create a new session."""
        if self._is_enhanced:
            return self._pipeline.create_session(session_id, user_id, project_id, session_name)
        else:
            raise NotImplementedError("Session management requires enhanced pipeline")
    
    def get_session(self, session_id: str):
        """Get session information."""
        if self._is_enhanced:
            return self._pipeline.get_session(session_id)
        else:
            return None
    
    def list_sessions(
        self, 
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50
    ):
        """List sessions."""
        if self._is_enhanced:
            return self._pipeline.list_sessions(user_id, project_id, limit)
        else:
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if self._is_enhanced:
            return self._pipeline.delete_session(session_id)
        else:
            return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        if self._is_enhanced:
            return self._pipeline.get_session_stats(session_id)
        else:
            return {}
    
    def clear_memory(self, session_id: Optional[str] = None) -> None:
        """Clear conversation memory."""
        if self._is_enhanced:
            self._pipeline.clear_memory(session_id)
        else:
            self._pipeline.clear_memory()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        stats = self._pipeline.get_stats()
        stats['enhanced_pipeline'] = self._is_enhanced
        return stats
