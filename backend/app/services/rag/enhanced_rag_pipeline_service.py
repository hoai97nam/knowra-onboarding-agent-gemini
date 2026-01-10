"""
enhanced_rag_pipeline_service.py

Enhanced RAG pipeline that integrates all optimization components:
- Router service for intelligent routing
- Safety service for content moderation
- Enhanced memory with fact extraction
- Session management
- Memory cleanup
"""

import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path

# Import enhanced components
try:
    from .router_service import RouterService, RouteDecision, RoutingResult
    from .safety_service import SafetyService, SafetyLevel, SafetyResult
    from .fact_extractor_service import FactExtractorService
    from .enhanced_memory_service import EnhancedMemoryService
    from .memory_cleanup_service import MemoryCleanupService
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced components not available: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False

# Import session service
try:
    from ..session_service import SessionService, SessionInfo
    SESSION_SERVICE_AVAILABLE = True
except ImportError:
    SESSION_SERVICE_AVAILABLE = False

# Import base components
try:
    from .chunking_service import ChunkingService
    from .embedding_service import EmbeddingService
    from .vector_store_service import VectorStoreService
    from .llm_service import LLMService
    BASE_COMPONENTS_AVAILABLE = True
except ImportError:
    BASE_COMPONENTS_AVAILABLE = False

from langchain_core.documents import Document

logger = logging.getLogger("enhanced_rag_pipeline")


class EnhancedRAGPipelineService:
    """
    Enhanced RAG pipeline service with intelligent routing, safety, and advanced memory management.
    """
    
    def __init__(
        self,
        # Base configuration
        data_dir: str,
        persist_dir: str,
        gemini_api_key: str = "",
        ollama_base_url: str = "",
        embedding_api_key: str = "",
        pinecone_api_key: str = "",
        pinecone_environment: str = "",
        pinecone_index_name: str = "",
        
        # Model configuration
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "mistral",
        llm_temperature: float = 0.7,
        
        # Chunking configuration
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_add_section_headers: bool = True,
        chunk_extract_metadata: bool = True,
        
        # Provider configuration
        provider: str = "ollama",
        
        # Enhanced features configuration
        enable_routing: bool = True,
        enable_safety: bool = False,
        enable_fact_extraction: bool = False,
        enable_memory_cleanup: bool = True,
        enable_session_management: bool = True,
        
        # Memory configuration
        memory_window: int = 12,
        max_memory_per_session: int = 1000,
        memory_cleanup_interval_hours: int = 6,
        
        # Other configuration
        retriever_k: int = 5,
        pca_components: int = 128,
        auto_rebuild: bool = True,
        redis_url: Optional[str] = None,
        rebuild_if_needed: Any = None
    ):
        """
        Initialize enhanced RAG pipeline.
        
        Args:
            All the configuration parameters for the enhanced pipeline
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.auto_rebuild = auto_rebuild
        self.retriever_k = retriever_k
        self.rebuild_if_needed = rebuild_if_needed

        
        # Feature flags
        self.enable_routing = enable_routing and ENHANCED_COMPONENTS_AVAILABLE
        self.enable_safety = enable_safety and ENHANCED_COMPONENTS_AVAILABLE
        self.enable_fact_extraction = enable_fact_extraction and ENHANCED_COMPONENTS_AVAILABLE
        self.enable_memory_cleanup = enable_memory_cleanup and ENHANCED_COMPONENTS_AVAILABLE
        self.enable_session_management = enable_session_management and SESSION_SERVICE_AVAILABLE
        
        # Initialize base components
        if not BASE_COMPONENTS_AVAILABLE:
            raise ImportError("Base RAG components not available")
        
        self._initialize_base_components(
            gemini_api_key, ollama_base_url, embedding_api_key,
            pinecone_api_key, pinecone_environment, pinecone_index_name,
            embedding_model, llm_model, llm_temperature,
            chunk_size, chunk_overlap, chunk_add_section_headers,
            chunk_extract_metadata, memory_window, pca_components, provider
        )
        
        # Initialize enhanced components
        if ENHANCED_COMPONENTS_AVAILABLE:
            self._initialize_enhanced_components(
                memory_window, max_memory_per_session,
                memory_cleanup_interval_hours, redis_url
            )
        
        # Initialize session management
        if self.enable_session_management:
            self._initialize_session_management(redis_url)
        
        # Auto-rebuild if needed
        if self.auto_rebuild:
            try:
                rebuilt = self.rebuild_if_needed()
                if rebuilt:
                    logger.info("Vector store rebuilt during initialization")
            except Exception as e:
                logger.error(f"Auto-rebuild failed: {e}")
        
        # Start background services
        self._start_background_services()
        
        logger.info("Enhanced RAG pipeline initialized successfully")
    
    def _initialize_base_components(
        self, gemini_api_key, ollama_base_url, embedding_api_key,
        pinecone_api_key, pinecone_environment, pinecone_index_name,
        embedding_model, llm_model, llm_temperature,
        chunk_size, chunk_overlap, chunk_add_section_headers,
        chunk_extract_metadata, memory_window, pca_components, provider
    ):
        """Initialize base RAG components."""
        
        # Embedding service with provider support
        self.embedding_service = EmbeddingService(
            gemini_api_key=gemini_api_key,
            model=embedding_model,
            provider=provider,
            pca_components=pca_components,
            ollama_base_url=ollama_base_url or "http://localhost:11434"
        )
        
        # Vector store service
        self.vector_store_service = VectorStoreService(
            embeddings=self.embedding_service.get_embeddings_instance(),
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            pinecone_index_name=pinecone_index_name,
            data_dir=self.data_dir,
            persist_dir=self.persist_dir
        )
        
        # Chunking service
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_section_headers=chunk_add_section_headers,
            extract_metadata=chunk_extract_metadata
        )
        
        # Create session-aware retriever
        def session_aware_retriever(query: str, k: int = 5, session_id: Optional[str] = None) -> List[Document]:
            """Retriever that considers session context."""
            try:
                return self.vector_store_service.semantic_search(query, k)
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                return []
        
        # LLM service with session support and provider support
        def session_id_getter() -> str:
            """Get current session ID (will be set per request)."""
            return getattr(self, '_current_session_id', 'default-session')
        
        self.llm_service = LLMService(
            gemini_api_key=gemini_api_key,
            ollama_base_url=ollama_base_url,
            model=llm_model,
            provider=provider,
            temperature=llm_temperature,
            memory_window=memory_window,
            retriever=session_aware_retriever,
            session_id_getter=session_id_getter,
            redis_url=None
        )
        
        # Current session tracking
        self._current_session_id = 'default-session'
        self._current_project_id = None
    
    def _initialize_enhanced_components(
        self, memory_window, max_memory_per_session,
        memory_cleanup_interval_hours, redis_url
    ):
        """Initialize enhanced components."""
        
        # Fact extractor service
        if self.enable_fact_extraction:
            self.fact_extractor = FactExtractorService(
                llm_extractor=self._create_llm_fact_extractor(),
                enable_llm_extraction=True
            )
        else:
            self.fact_extractor = None
        
        # Enhanced memory service
        self.enhanced_memory = EnhancedMemoryService(
            vector_store_service=self.vector_store_service,
            fact_extractor=self.fact_extractor,
            session_service=None,  # Will be set after session service init
            redis_url=redis_url
        )
        
        # Router service
        if self.enable_routing:
            def memory_getter(session_id: str) -> List[Dict]:
                """Get memory for routing decisions."""
                try:
                    context = self.enhanced_memory.get_context_for_query(session_id, "", max_context=5)
                    return [{'role': c.get('role', 'unknown'), 'content': c.get('content', '')} for c in context]
                except Exception:
                    return []
            
            self.router_service = RouterService(
                retriever=lambda q, k: self.vector_store_service.semantic_search(q, k),
                memory_getter=memory_getter
            )
        else:
            self.router_service = None
        
        # Safety service
        if self.enable_safety:
            self.safety_service = SafetyService(
                llm_classifier=self._create_llm_safety_classifier(),
                enable_llm_classification=True
            )
        else:
            self.safety_service = None
        
        # Memory cleanup service
        if self.enable_memory_cleanup:
            from datetime import timedelta
            self.memory_cleanup = MemoryCleanupService(
                enhanced_memory_service=self.enhanced_memory,
                session_service=None,  # Will be set after session service init
                cleanup_interval=timedelta(hours=memory_cleanup_interval_hours),
                max_memory_per_session=max_memory_per_session
            )
        else:
            self.memory_cleanup = None
    
    def _initialize_session_management(self, redis_url):
        """Initialize session management."""
        self.session_service = SessionService(redis_url=redis_url)
        
        # Link session service to other components
        if self.enhanced_memory:
            self.enhanced_memory.session_service = self.session_service
        
        if self.memory_cleanup:
            self.memory_cleanup.session_service = self.session_service
    
    def _start_background_services(self):
        """Start background services."""
        if self.memory_cleanup and self.enable_memory_cleanup:
            try:
                self.memory_cleanup.start_background_cleanup()
                logger.info("Memory cleanup service started")
            except Exception as e:
                logger.error(f"Failed to start memory cleanup: {e}")
    
    def _create_llm_fact_extractor(self) -> Callable[[str], str]:
        """Create LLM-based fact extractor with robust error handling."""
        def extract_facts(prompt: str) -> str:
            try:
                response = self.llm_service.generate_answer(prompt, context_docs=[])
                # Ensure response is valid JSON or return empty array
                if not response or not response.strip():
                    return "[]"
                
                # Basic validation - check if response looks like JSON
                stripped = response.strip()
                if not (stripped.startswith('[') or stripped.startswith('{')):
                    logger.debug(f"LLM response doesn't look like JSON: {stripped[:100]}...")
                    return "[]"
                
                return response
                
            except Exception as e:
                logger.error(f"LLM fact extraction failed: {e}")
                # Return empty JSON array instead of empty string to avoid parsing errors
                return "[]"
        return extract_facts
    
    def _create_llm_safety_classifier(self) -> Callable[[str], str]:
        """Create LLM-based safety classifier."""
        def classify_safety(prompt: str) -> str:
            try:
                return self.llm_service.generate_answer(prompt, context_docs=[])
            except Exception as e:
                logger.error(f"LLM safety classification failed: {e}")
                return "SAFE"
        return classify_safety
    
    def _set_session_context(self, session_id: str, project_id: Optional[str] = None):
        """Set current session context."""
        self._current_session_id = session_id
        self._current_project_id = project_id

    async def stream_response(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        DEPRECATED: Streaming is disabled. This method now calls chat_with_metadata internally.
        
        Args:
            query: User query
            session_id: Optional session identifier
            project_id: Optional project identifier
            
        Yields:
            str: Complete response as a single chunk
        """
        logger.warning("stream_response is deprecated. Use chat_with_metadata instead.")
        
        # Call non-streaming method
        response_text, metadata = self.chat_with_metadata(query, session_id, project_id)
        
        # Yield complete response as single chunk
        yield response_text
    
    def chat_with_metadata(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Enhanced chat with metadata return.
        
        Args:
            query: User query
            session_id: Optional session identifier
            project_id: Optional project identifier
            
        Returns:
            Tuple of (response_text, metadata)
        """
        # Set session context
        if not session_id:
            session_id = f"session-{uuid.uuid4()}"
        self._set_session_context(session_id, project_id)
        
        # Get or create session
        if self.enable_session_management:
            session_info = self.session_service.get_or_create_session(
                session_id=session_id,
                project_id=project_id
            )
            session_id = session_info.session_id
        
        metadata = {
            'session_id': session_id,
            'project_id': project_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Safety analysis
            safety_result = None
            if self.enable_safety:
                safety_result = self.safety_service.analyze_content(
                    query,
                    context={'session_id': session_id, 'project_id': project_id}
                )
                metadata['safety_info'] = {
                    'level': safety_result.level.value,
                    'confidence': safety_result.confidence,
                    'reasoning': safety_result.reasoning
                }
                
                if safety_result.level == SafetyLevel.BLOCKED:
                    return "I cannot process this request due to content policy restrictions.", metadata
                elif safety_result.level == SafetyLevel.SENSITIVE:
                    query = safety_result.sanitized_content or query
            
            # Route the query
            routing_result = None
            if self.enable_routing:
                routing_result = self.router_service.route_query(
                    query, session_id, project_id,
                    context={
                        'conversation_length': self._get_conversation_length(session_id),
                        'has_recent_context': self._has_recent_context(session_id)
                    }
                )
                metadata['routing_info'] = {
                    'decision': routing_result.decision.value,
                    'confidence': routing_result.confidence,
                    'reasoning': routing_result.reasoning
                }
            
            # Get context
            context_docs = []
            memory_context = []
            
            if not routing_result or routing_result.decision in [RouteDecision.KNOWLEDGE_BASE_ONLY, RouteDecision.KNOWLEDGE_AND_MEMORY, RouteDecision.DEFAULT]:
                context_docs = self.vector_store_service.semantic_search(query, self.retriever_k)
                metadata['sources'] = [getattr(doc, 'metadata', {}).get('source', 'unknown') for doc in context_docs]
            
            if not routing_result or routing_result.decision in [RouteDecision.MEMORY_ONLY, RouteDecision.KNOWLEDGE_AND_MEMORY]:
                if self.enhanced_memory:
                    memory_context = self.enhanced_memory.get_context_for_query(
                        session_id, query, max_context=5
                    )
            
            # Build context and generate response
            enhanced_context = self._build_enhanced_context(context_docs, memory_context)
            response = self.llm_service.generate_answer(query, enhanced_context)
            
            # Store conversation turn
            if self.enhanced_memory:
                self.enhanced_memory.process_conversation_turn(
                    session_id=session_id,
                    user_message=query,
                    assistant_response=response,
                    project_id=project_id,
                    metadata=metadata
                )
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Enhanced chat failed: {e}")
            metadata['error'] = str(e)
            return f"I apologize, but an error occurred while processing your request: {str(e)}", metadata
    
    def _build_enhanced_context(self, kb_docs: List[Document], memory_context: List[Dict]) -> List[Document]:
        """Build enhanced context from KB docs and memory."""
        enhanced_docs = list(kb_docs)  # Start with KB docs
        
        # Add memory context as documents
        for i, mem_item in enumerate(memory_context):
            content = mem_item.get('content', '')
            if content:
                metadata = {
                    'source': 'conversation_memory',
                    'memory_type': mem_item.get('metadata', {}).get('type', 'unknown'),
                    'role': mem_item.get('role', 'unknown')
                }
                
                doc = Document(page_content=content, metadata=metadata)
                enhanced_docs.append(doc)
        
        return enhanced_docs
    
    def _get_conversation_length(self, session_id: str) -> int:
        """Get conversation length for session."""
        if not self.enhanced_memory:
            return 0
        
        try:
            stats = self.enhanced_memory.get_stats(session_id)
            return stats.get('buffer_messages', 0)
        except Exception:
            return 0
    
    def _has_recent_context(self, session_id: str) -> bool:
        """Check if session has recent context."""
        return self._get_conversation_length(session_id) > 0
    
    # Session management methods
    def create_session(
        self, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_name: Optional[str] = None
    ):
        """Create a new session."""
        if not self.enable_session_management:
            raise NotImplementedError("Session management not enabled")
        
        return self.session_service.create_session(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            session_name=session_name
        )
    
    def get_session(self, session_id: str):
        """Get session information."""
        if not self.enable_session_management:
            return None
        
        return self.session_service.get_session(session_id)
    
    def list_sessions(
        self, 
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50
    ):
        """List sessions."""
        if not self.enable_session_management:
            return []
        
        return self.session_service.list_sessions(
            user_id=user_id,
            project_id=project_id,
            limit=limit
        )
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if self.enable_session_management:
            success = self.session_service.delete_session(session_id)
        else:
            success = True
        
        # Clear memory
        if self.enhanced_memory:
            self.enhanced_memory.clear_session(session_id)
        
        return success
    
    def clear_memory(self, session_id: Optional[str] = None) -> None:
        """Clear memory for session."""
        if session_id is None:
            session_id = self._current_session_id
        
        if self.enhanced_memory:
            self.enhanced_memory.clear_session(session_id)
        else:
            # Fallback to basic memory clear
            self.llm_service.clear_memory()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        stats = {}
        
        if self.enable_session_management:
            session_info = self.session_service.get_session(session_id)
            if session_info:
                stats['session'] = session_info.to_dict()
        
        if self.enhanced_memory:
            memory_stats = self.enhanced_memory.get_stats(session_id)
            stats['memory'] = memory_stats
        
        if self.memory_cleanup:
            cleanup_stats = self.memory_cleanup.get_cleanup_stats()
            stats['cleanup'] = cleanup_stats
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            'documents_loaded': len(self.chunking_service.load_and_chunk_documents(self.data_dir)) if hasattr(self.chunking_service, 'load_and_chunk_documents') else 0,
            'vector_store_stats': self.vector_store_service.get_stats(),
            'llm_stats': self.llm_service.get_stats(),
            'enhanced_features': {
                'routing_enabled': self.enable_routing,
                'safety_enabled': self.enable_safety,
                'fact_extraction_enabled': self.enable_fact_extraction,
                'memory_cleanup_enabled': self.enable_memory_cleanup,
                'session_management_enabled': self.enable_session_management
            }
        }
        
        if self.enhanced_memory:
            stats['enhanced_memory'] = self.enhanced_memory.get_stats()
        
        if self.enable_session_management:
            stats['session_stats'] = self.session_service.get_session_stats()
        
        return stats