# rag_pipeline_service.py
"""
RAG Pipeline Service - orchestrates RAG flow, integrates LLMService memory and vector store.

Assumptions:
- self.llm_service is an instance of the LLMService implemented earlier
  which exposes:
    - generate_answer(query, context_docs=None)
    - generate_answer_stream(query, context_docs=None) -> async generator
    - memory_store (MemoryStore) accessible via llm_service.memory_store
    - memory helper via llm_service.memory (optional) with methods append/get_relevant_context/persist_turn
- self.vector_store_service provides semantic_search(query, k) returning Documents
  and optionally upsert_embeddings(items, namespace) to store memory embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime
import logging

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.documents import Document

from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService
from .vector_store_service import VectorStoreService
from .semantic_search_service import SemanticSearchService
from .ranking_service import RankingService
from .llm_service import LLMService

logger = logging.getLogger("rag_pipeline_service")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel("INFO")


class RAGPipelineService:
    def __init__(
        self,
        data_dir: str,
        persist_dir: str,
        openai_base_url: str = "",
        openai_api_key: str = "",
        gemini_api_key: str = "",
        ollama_base_url: str = "",
        embedding_api_key: str = "",
        pinecone_api_key: str = "",
        pinecone_environment: str = "",
        pinecone_index_name: str = "",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        pca_components: int = None,
        llm_model: str = "",
        llm_temperature: float = 0.0,
        retriever_k: int = 5,
        memory_window: int = 10,
        auto_rebuild: bool = True,
        chunk_add_section_headers: bool = True,
        chunk_extract_metadata: bool = True,
        provider: str = "openai",
        session_id_getter: Optional[callable] = None,
    ) -> None:
        """
        Initialize the RAG pipeline service with Pinecone.
        
        Args:
            data_dir: Directory containing markdown knowledge base files
            persist_dir: Directory for hash storage (legacy)
            openai_api_key: OpenAI API key for LLM
            gemini_api_key: Google Gemini API key for LLM
            embedding_api_key: API key for embedding service
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment/region
            pinecone_index_name: Pinecone index name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Embedding model to use (default: text-embedding-3-small)
            llm_model: LLM model
            llm_temperature: LLM temperature
            retriever_k: Number of documents to retrieve
            memory_window: Conversation memory window size
            auto_rebuild: Enable automatic rebuild on startup and chat (default: False)
            chunk_add_section_headers: Add section context to chunks (default: True)
            chunk_extract_metadata: Extract rich metadata from documents (default: True)
            provider: "openai" or "gemini" (default "openai")
            session_id_getter: Optional callback to get session ID
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.provider = provider.lower()
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.ollama_base_url = ollama_base_url
        self.embedding_api_key = embedding_api_key or (gemini_api_key if provider.lower() == "gemini" else openai_api_key)
        self.retriever_k = retriever_k
        self.auto_rebuild = auto_rebuild
        
        # Initialize sub-services
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_section_headers=chunk_add_section_headers,
            extract_metadata=chunk_extract_metadata
        )
        
        self.embedding_service = EmbeddingService(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            model=embedding_model,
            provider=provider,
            pca_components=pca_components
        )
        
        self.vector_store_service = VectorStoreService(
            embeddings=self.embedding_service.get_embeddings_instance(),
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            pinecone_index_name=pinecone_index_name,
            data_dir=self.data_dir,
            persist_dir=self.persist_dir
        )
        
        self.search_service = SemanticSearchService(
            vector_store_service=self.vector_store_service
        )
        
        self.ranking_service = RankingService(
            score_threshold=0.0
        )
        
        self.llm_service = LLMService(
            open_ai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            ollama_base_url=ollama_base_url,
            model=llm_model,
            provider=provider,
            temperature=llm_temperature,
            memory_window=memory_window,
            streaming=True,
            session_id_getter=session_id_getter,
            retriever=self.search_service.semantic_search
        )
        
        # Agent executor (initialized after vector store)
        self.agent_executor = None
        
        # Initialize the pipeline
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize or rebuild vector store and agent."""
        if self.auto_rebuild and self.vector_store_service.should_rebuild():
            print("🔨 Auto-rebuild ENABLED - Rebuilding vector store (files changed or missing)...")
            self._build_vectorstore()
        else:
            if not self.auto_rebuild:
                print("⚙️  Auto-rebuild DISABLED - Loading existing vector store from Pinecone...")
            else:
                print("✨ Vector store is up to date - Loading from Pinecone...")
            self.vector_store_service.load_vectorstore()
        
        self._create_agent()
    
    def _build_vectorstore(self) -> None:
        """Build or rebuild the vector store from markdown files."""
        print("🔄 Building vector store...")
        
        # Load and chunk documents
        chunks = self.chunking_service.process(self.data_dir)
        
        # Create vector store
        self.vector_store_service.create_vectorstore(chunks)
    
    def _create_agent(self) -> None:
        """Create a react agent and register a knowledge_base_search tool and memory_search tool if available."""
        # KB retriever from search_service
        def kb_retrieve(query: str) -> str:
            try:
                docs: List[Document] = self.search_service.semantic_search(query, k=self.retriever_k)
                if not docs:
                    return "No relevant information found in the knowledge base."
                return "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])
            except Exception as e:
                logger.exception("KB retrieve failed: %s", e)
                return f"Error retrieving KB: {e}"

        kb_tool = Tool(
            name="knowledge_base_search",
            description="Search company docs (onboarding, architecture, API, deployment). Input: query string.",
            func=kb_retrieve,
        )

        tools = [kb_tool]

        # memory search tool (if LLMService exposes a memory.get_relevant_context or retriever)
        if hasattr(self.llm_service, "memory") and hasattr(self.llm_service.memory, "get_relevant_context"):
            def memory_retrieve(query: str) -> str:
                try:
                    items = self.llm_service.memory.get_relevant_context(query, k=self.retriever_k)
                    if not items:
                        return "No related memory entries."
                    return "\n\n".join(items)
                except Exception as e:
                    logger.exception("Memory retrieve failed: %s", e)
                    return f"Error retrieving memory: {e}"

            mem_tool = Tool(
                name="memory_search",
                description="Search conversation memory for relevant past interactions. Input: query string.",
                func=memory_retrieve,
            )
            tools.append(mem_tool)

        # Build LLM instance for agent (use llm_service.get_llm_instance())
        try:
            llm = self.llm_service.get_llm_instance()
            self.agent_executor = create_react_agent(llm, tools=tools)
            logger.info("Agent executor created")
        except Exception:
            logger.exception("Failed to create react agent")
            self.agent_executor = None

    def rebuild_if_needed(self) -> bool:
        """
        Check and rebuild vector store if files changed (only if auto_rebuild is enabled).
        
        Returns:
            bool: True if rebuild occurred
        """
        if not self.auto_rebuild:
            print("⚙️  Auto-rebuild DISABLED - Skipping rebuild check")
            return False
            
        if self.vector_store_service.should_rebuild():
            logger.info("Files changed, rebuilding vector store...")
            self._build_vectorstore()
            self._create_agent()
            return True
        return False

    # Helper to build combined prompt (memory + KB context)
    def _build_combined_system(self, query: str) -> str:
        # Get session id via llm_service if available
        try:
            session_id = self.llm_service._get_session_id()
        except Exception:
            session_id = "default-session"

        # short-term memory from MemoryStore
        mem_text = ""
        try:
            memory_msgs = self.llm_service.get_memory_instance() or []
            if memory_msgs:
                mem_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in memory_msgs])
        except Exception:
            mem_text = ""

        # relevant long-term memory via LLMService.memory.get_relevant_context if exists
        relevant_text = ""
        try:
            if hasattr(self.llm_service, "memory") and hasattr(self.llm_service.memory, "get_relevant_context"):
                relevant_items = self.llm_service.memory.get_relevant_context(query, k=3)
                if relevant_items:
                    relevant_text = "\n\n".join(relevant_items)
        except Exception:
            relevant_text = ""

        # KB docs retrieved from document store
        kb_text = ""
        try:
            docs = self.search_service.semantic_search(query, k=self.retriever_k)
            if docs:
                kb_text = "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])
        except Exception:
            kb_text = ""

        parts = ["You are a helpful assistant. Use conversation history, relevant memory, and retrieved documents."]
        if mem_text:
            parts.append("Conversation history (recent):\n" + mem_text)
        if relevant_text:
            parts.append("Relevant long-term memory:\n" + relevant_text)
        if kb_text:
            parts.append("Retrieved documents from KB:\n" + kb_text)

        combined = "\n\n".join(parts)
        # Enforce token limit via llm_service helper if present, else a simple fallback
        if hasattr(self.llm_service, "_enforce_token_limit"):
            combined = self.llm_service._enforce_token_limit(combined)
        return combined

    def _persist_assistant_turn(self, assistant_text: str) -> None:
        """Persist assistant turn to short-term MemoryStore and optionally to vector memory index."""
        # append to MemoryStore via LLMService API
        try:
            self.llm_service._append_memory("assistant", assistant_text)
        except Exception:
            # fallback: try memory object if available
            try:
                if hasattr(self.llm_service, "memory") and hasattr(self.llm_service.memory, "append"):
                    self.llm_service.memory.append("assistant", assistant_text)
            except Exception:
                logger.exception("Failed to append assistant turn to memory store")

        # also index into long-term vector memory if vector_store_service supports upsert and embedding_service available
        try:
            if hasattr(self.vector_store_service, "upsert_embeddings") and hasattr(self.embedding_service, "embed_text"):
                text = f"assistant: {assistant_text}"
                vector = self.embedding_service.embed_text(text)
                item = {
                    "id": f"mem-assistant-{int(datetime.utcnow().timestamp()*1000)}",
                    "vector": vector,
                    "metadata": {"role": "assistant", "timestamp": datetime.utcnow().isoformat()},
                    "text": text,
                }
                # place in a dedicated namespace for memory if method supports namespace param
                try:
                    self.vector_store_service.upsert_embeddings([item], namespace="memory")
                except TypeError:
                    # fallback if no namespace param
                    self.vector_store_service.upsert_embeddings([item])
        except Exception:
            # safe to ignore if not supported
            logger.debug("Long-term memory upsert skipped or failed")

    def chat(self, query: str) -> str:
        """Non-streaming chat integrated with memory and KB retrieval."""
        if not self.agent_executor:
            raise ValueError("Agent executor not initialized")

        # moderation check via LLMService hook
        try:
            if not self.llm_service.moderation_hook(query):
                return "Message blocked by content policy."
        except Exception:
            logger.exception("Moderation check failed, proceeding with caution")

        # Append user turn to memory store
        try:
            self.llm_service._append_memory("user", query)
        except Exception:
            logger.exception("Failed to append user turn to memory")

        # Build combined system instructions including memory and KB
        system_prompt = self._build_combined_system(query)

        # Create messages expected by the agent
        messages = [("system", system_prompt), ("user", query)]

        # invoke agent
        config = {"recursion_limit": 50}
        response = self.agent_executor.invoke({"messages": messages}, config=config)

        # robustly extract assistant content
        assistant_text = ""
        try:
            if isinstance(response, dict):
                msgs = response.get("messages") or response.get("agent", {}).get("messages") or []
            else:
                msgs = getattr(response, "messages", None) or []
            for message in reversed(msgs):
                if isinstance(message, dict):
                    content = message.get("content")
                else:
                    content = getattr(message, "content", None)
                if content:
                    assistant_text = content
                    break
        except Exception:
            logger.exception("Failed to extract assistant text from agent response")

        if not assistant_text:
            assistant_text = str(response)

        # Persist assistant turn
        self._persist_assistant_turn(assistant_text)

        # Optionally trigger summarization/prune if memory window exceeded and summarizer present
        try:
            if hasattr(self.llm_service, "memory") and hasattr(self.llm_service.memory, "summarize_and_prune"):
                if len(self.llm_service.memory.get_relevant_context("", k=0) if hasattr(self.llm_service.memory, "get_relevant_context") else []) >= self.llm_service.memory.window:
                    # call summarizer (safe-guard)
                    try:
                        self.llm_service.memory.summarize_and_prune()
                    except Exception:
                        logger.debug("Summarize/prune failed")
        except Exception:
            # ignore summarization errors
            pass

        return assistant_text

    async def stream_response(self, query: str) -> AsyncGenerator[str, None]:
        """Streaming chat integrated with memory and KB retrieval."""
        if not self.agent_executor:
            raise ValueError("Agent executor not initialized")

        # moderation check
        try:
            if not self.llm_service.moderation_hook(query):
                yield "Message blocked by content policy."
                return
        except Exception:
            logger.exception("Moderation check failed, proceeding with caution")

        # persist user short-term memory
        try:
            self.llm_service._append_memory("user", query)
        except Exception:
            logger.exception("Failed to append user turn to memory store")

        system_prompt = self._build_combined_system(query)
        messages = [("system", system_prompt), ("user", query)]

        # stream from agent executor
        config = {"recursion_limit": 50}
        assembled: List[str] = []
        try:
            async for chunk in self.agent_executor.astream({"messages": messages}, config=config):
                # try multiple chunk shapes
                try:
                    if isinstance(chunk, dict):
                        agent_block = chunk.get("agent")
                        if agent_block and "messages" in agent_block:
                            for m in agent_block["messages"]:
                                content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
                                if content:
                                    assembled.append(content)
                                    yield content
                        elif "text" in chunk:
                            text = chunk.get("text")
                            if text:
                                assembled.append(text)
                                yield text
                    else:
                        text = getattr(chunk, "content", None) or getattr(chunk, "text", None) or (chunk if isinstance(chunk, str) else None)
                        if text:
                            assembled.append(text)
                            yield text
                except Exception:
                    logger.exception("Error processing stream chunk; continuing")
                    continue
        except AttributeError:
            # agent executor has no astream, fallback to LLMService streaming or sync
            try:
                print("⚠️ Agent executor has no streaming; falling back to LLMService streaming")
            except Exception:
                # fallback to sync generate
                res = self.llm_service.generate_answer(query)
                assembled.append(res)
                yield res
        except Exception:
            logger.exception("Streaming failed")
            raise

        # final assembled text
        final_text = "".join(assembled).strip()
        if final_text:
            self._persist_assistant_turn(final_text)

        # possibly summarize/prune memory
        try:
            if hasattr(self.llm_service, "memory") and hasattr(self.llm_service.memory, "summarize_and_prune"):
                try:
                    self.llm_service.memory.summarize_and_prune()
                except Exception:
                    logger.debug("Summarize and prune failed after streaming")
        except Exception:
            pass

    def clear_memory(self, session_id: Optional[str] = None) -> None:
        """Clear memory for a session. If session_id not provided, use llm_service session id getter."""
        try:
            sid = session_id or self.llm_service._get_session_id()
            self.llm_service.memory_store.clear(sid)
            # also clear conversation memory object if present
            if hasattr(self.llm_service, "memory") and hasattr(self.llm_service.memory, "clear"):
                self.llm_service.memory.clear()
            logger.info("Cleared memory for session %s", sid)
        except Exception:
            logger.exception("Failed to clear memory for session %s", session_id)

    def get_stats(self) -> Dict[str, Any]:
        doc_count = len(list(self.data_dir.glob("*.md")))
        stats = {
            "documents_loaded": doc_count,
            "chunking": self.chunking_service.get_stats(),
            "embedding": self.embedding_service.get_stats(),
            "vector_store": self.vector_store_service.get_stats(),
            "search": self.search_service.get_stats(),
            "ranking": self.ranking_service.get_stats(),
            "llm": self.llm_service.get_stats()
        }
        return stats
