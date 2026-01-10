"""
Dependency injection and shared dependencies for the application.
"""
from typing import Generator, Optional
from functools import lru_cache
import logging

from app.core.config import Settings, get_settings
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# Global service instances
_rag_service: Optional[RAGService] = None
_mongodb_service = None
_chat_persistence_service = None


@lru_cache()
def get_settings_cached() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Cached application settings
    """
    return get_settings()


def get_rag_service() -> RAGService:
    """
    Get or create the global RAG service instance.
    
    Returns:
        RAGService: RAG service instance
    """
    global _rag_service
    if _rag_service is None:
        settings = get_settings_cached()
        _rag_service = RAGService(
            data_dir=settings.DATA_DIR,
            persist_dir=settings.VECTOR_STORE_DIR
        )
    return _rag_service


def rebuild_rag_service() -> RAGService:
    """
    Force rebuild of the RAG service.
    
    Returns:
        RAGService: New RAG service instance
    """
    global _rag_service
    settings = get_settings_cached()
    _rag_service = RAGService(
        data_dir=settings.DATA_DIR,
        persist_dir=settings.VECTOR_STORE_DIR
    )
    return _rag_service


async def initialize_mongodb():
    """
    Initialize MongoDB connection and chat persistence service.
    Should be called during application startup.
    """
    global _mongodb_service, _chat_persistence_service
    
    try:
        from app.services.mongodb_service import initialize_mongodb as init_mongo
        from app.services.chat_persistence_service import initialize_chat_persistence_service
        
        settings = get_settings_cached()
        
        # Initialize MongoDB service
        logger.info("Initializing MongoDB service...")
        logger.info("settings: %s", settings)
        _mongodb_service = init_mongo(
            mongodb_url=settings.MONGODB_URL,
            db_name=settings.MONGODB_DB_NAME,
            max_pool_size=settings.MONGODB_MAX_POOL_SIZE,
            min_pool_size=settings.MONGODB_MIN_POOL_SIZE,
            timeout_ms=settings.MONGODB_TIMEOUT_MS
        )
        
        # Connect to MongoDB
        await _mongodb_service.connect()
        
        # Initialize chat persistence service
        logger.info("Initializing chat persistence service...")
        _chat_persistence_service = initialize_chat_persistence_service(_mongodb_service.database)
        
        logger.info("MongoDB and chat persistence services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise


async def shutdown_mongodb():
    """
    Shutdown MongoDB connection.
    Should be called during application shutdown.
    """
    global _mongodb_service
    
    if _mongodb_service:
        try:
            await _mongodb_service.disconnect()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")


def get_chat_persistence():
    """
    Get the chat persistence service instance.
    
    Returns:
        ChatPersistenceService: Chat persistence service
        
    Raises:
        RuntimeError: If service not initialized
    """
    global _chat_persistence_service
    
    if _chat_persistence_service is None:
        raise RuntimeError(
            "Chat persistence service not initialized. "
            "Make sure initialize_mongodb() is called during startup."
        )
    
    return _chat_persistence_service


def get_mongodb_service():
    """
    Get the MongoDB service instance.
    
    Returns:
        MongoDBService: MongoDB service
        
    Raises:
        RuntimeError: If service not initialized
    """
    global _mongodb_service
    
    if _mongodb_service is None:
        raise RuntimeError(
            "MongoDB service not initialized. "
            "Make sure initialize_mongodb() is called during startup."
        )
    
    return _mongodb_service

