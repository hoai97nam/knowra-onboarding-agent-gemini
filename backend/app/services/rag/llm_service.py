"""
llm_service.py

Upgraded LLM service that:
- Wraps an underlying LLM client via an adapter (swappable)
- Supports sync and async streaming generation
- Keeps per-session short-term memory persisted in Redis or in-process
- Exposes a simple retriever/memory integration hook
- Moderation hook, prompt building, token limit enforcement, metrics
- Public API methods remain: generate_answer, generate_answer_stream, get_llm_instance, clear_memory, get_stats
"""

import os
import time
import logging
import asyncio
from typing import List, Optional, AsyncGenerator, Callable, Any, Dict
from datetime import datetime

# Optional dependencies (best-effort imports)
try:
    import redis
except Exception:
    redis = None

try:
    from prometheus_client import Counter, Histogram
except Exception:
    Counter = None
    Histogram = None

# Placeholder imports for ChatOpenAI and LangChain-like message/document types.
# Replace these imports with your project's concrete wrappers if names differ.
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    # Minimal placeholders so the module can be imported for static checks.
    ChatOpenAI = object
    HumanMessage = dict
    AIMessage = dict
    SystemMessage = dict
    Document = dict
    ChatPromptTemplate = None
    MessagesPlaceholder = None

# Import Gemini LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# --- Logger ---
logger = logging.getLogger("llm_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(os.getenv("LLM_SERVICE_LOG_LEVEL", "INFO"))

# --- Metrics (optional) ---
REQUEST_COUNTER = Counter("llm_requests_total", "Total LLM requests") if Counter else None
REQUEST_LATENCY = Histogram("llm_request_latency_seconds", "LLM request latency seconds") if Histogram else None

# --- Simple token utilities (replace with model tokenizer for production) ---
def simple_token_count(text: str) -> int:
    return len(text.split())

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[-max_tokens:])  # keep recent tokens

# --- Retry decorators ---
def retry_on_exception(max_attempts=3, base_delay=0.5, exceptions=(Exception,)):
    def deco(fn):
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
                        return "I'm sorry, but I cannot assist with that."
                    time.sleep(base_delay * (2 ** (attempts - 1)))
        return wrapped
    return deco

def aretry_on_exception(max_attempts=3, base_delay=0.5, exceptions=(Exception,)):
    def deco(fn):
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

# --- Exceptions ---
class TransientAPIError(Exception):
    pass

class ModerationError(Exception):
    pass

# --- LLM Adapter Interface ---
class BaseLLMAdapter:
    def invoke(self, prompt: str) -> Any:
        raise NotImplementedError

    async def astream(self, prompt: str):
        raise NotImplementedError

    def get_model_name(self) -> str:
        raise NotImplementedError

# --- Concrete ChatOpenAI Adapter ---
class ChatOpenAIAdapter(BaseLLMAdapter):
    def __init__(self, llm_instance: ChatOpenAI):
        self.llm = llm_instance

    @retry_on_exception(max_attempts=3, base_delay=0.8, exceptions=(TransientAPIError, Exception))
    def invoke(self, prompt: str):
        try:
            # adapt to your client's API; many clients accept message lists — here we pass raw prompt
            res = self.llm.invoke(prompt)
            return res
        except Exception as e:
            logger.exception("LLM invoke error: %s", e)
            raise

    async def astream(self, prompt: str):
        # expect self.llm.astream to be an async generator
        # Manual retry logic for async generators
        max_attempts = 2
        base_delay = 0.5
        attempts = 0
        
        while True:
            try:
                async for chunk in self.llm.astream(prompt):
                    yield chunk
                break  # Success, exit retry loop
            except (TransientAPIError, Exception) as e:
                attempts += 1
                logger.warning("Async retryable error in astream: %s (attempt %d/%d)", e, attempts, max_attempts)
                if attempts >= max_attempts:
                    logger.exception("Max async retry attempts reached for astream")
                    raise
                await asyncio.sleep(base_delay * (2 ** (attempts - 1)))

    def get_model_name(self) -> str:
        return getattr(self.llm, "model", "unknown")

# --- Concrete Gemini Adapter ---
class GeminiLLMAdapter(BaseLLMAdapter):
    """Adapter for Google Generative AI (Gemini) models."""
    
    def __init__(self, llm_instance: ChatGoogleGenerativeAI):
        self.llm = llm_instance

    @retry_on_exception(max_attempts=3, base_delay=0.8, exceptions=(TransientAPIError, Exception))
    def invoke(self, prompt: str):
        try:
            # Gemini LLM compatible with LangChain invoke interface
            res = self.llm.invoke(prompt)
            return res
        except Exception as e:
            logger.exception("Gemini LLM invoke error: %s", e)
            raise

    async def astream(self, prompt: str):
        # Manual retry logic for async generators
        max_attempts = 2
        base_delay = 0.5
        attempts = 0
        
        while True:
            try:
                async for chunk in self.llm.astream(prompt):
                    yield chunk
                break  # Success, exit retry loop
            except (TransientAPIError, Exception) as e:
                attempts += 1
                logger.warning("Async retryable error in Gemini astream: %s (attempt %d/%d)", e, attempts, max_attempts)
                if attempts >= max_attempts:
                    logger.exception("Max async retry attempts reached for Gemini astream")
                    raise
                await asyncio.sleep(base_delay * (2 ** (attempts - 1)))

    def get_model_name(self) -> str:
        return getattr(self.llm, "model", "gemini")

# --- MemoryStore (Redis-backed optional, in-memory fallback) ---
class MemoryStore:
    def __init__(self, redis_url: Optional[str] = None, prefix: str = "llm:mem:"):
        self.prefix = prefix
        self.enabled = False
        self._client = None
        if redis_url and redis:
            try:
                self._client = redis.from_url(redis_url, decode_responses=True)
                # quick check
                _ = self._client.ping()
                self.enabled = True
            except Exception as e:
                logger.warning("Redis init failed: %s. Falling back to in-memory store.", e)
                self.enabled = False
        self._in_memory: Dict[str, List[Dict]] = {}

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def get(self, session_id: str) -> List[Dict]:
        if self.enabled:
            try:
                data = self._client.get(self._key(session_id))
                if not data:
                    return []
                import json
                return json.loads(data)
            except Exception as e:
                logger.debug("Redis get failed: %s", e)
                return []
        return self._in_memory.get(session_id, []).copy()

    def set(self, session_id: str, messages: List[Dict]) -> None:
        if self.enabled:
            try:
                import json
                self._client.set(self._key(session_id), json.dumps(messages))
            except Exception as e:
                logger.debug("Redis set failed: %s", e)
        else:
            self._in_memory[session_id] = messages.copy()

    def clear(self, session_id: str) -> None:
        if self.enabled:
            try:
                self._client.delete(self._key(session_id))
            except Exception as e:
                logger.debug("Redis delete failed: %s", e)
        else:
            self._in_memory.pop(session_id, None)

# --- Moderation hook (simple) ---
def moderate_text(text: str) -> bool:
    blocked_terms = os.getenv("LLM_BLOCKED_TERMS", "").split(",")
    for t in blocked_terms:
        if t and t.strip().lower() in text.lower():
            return False
    return True

# --- Default retriever placeholder ---
def default_retriever(query: str, k: int = 5) -> List[Document]:
    return []

# --- Conversation Memory (lightweight short-term + vector hooks handled elsewhere) ---
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

class ConversationMemoryLight:
    """
    Lightweight short-term memory used to assemble prompts and persist via MemoryStore.
    Long-term vector memory should be implemented in vector store service separately.
    """
    def __init__(self, memory_store: MemoryStore, session_id_getter: Callable[[], str], window: int = 10):
        self.memory_store = memory_store
        self.session_id_getter = session_id_getter
        self.window = window

    def append(self, role: str, content: str) -> None:
        session_id = self._get_session_id()
        entry = {"role": role, "content": content, "timestamp": _now_iso()}
        # in-process history not stored separately here; MemoryStore is source of truth
        try:
            existing = self.memory_store.get(session_id) or []
            existing.append(entry)
            # cap to window
            if len(existing) > self.window:
                existing = existing[-self.window:]
            self.memory_store.set(session_id, existing)
        except Exception as e:
            logger.debug("Memory append failed: %s", e)

    def get_recent(self) -> List[Dict]:
        session_id = self._get_session_id()
        try:
            return self.memory_store.get(session_id) or []
        except Exception:
            return []

    def clear(self) -> None:
        session_id = self._get_session_id()
        try:
            self.memory_store.clear(session_id)
        except Exception as e:
            logger.debug("Memory clear failed: %s", e)

    def _get_session_id(self) -> str:
        try:
            return self.session_id_getter()
        except Exception:
            return "default-session"

# --- Main LLMService ---
class LLMService:
    """
    Upgraded LLMService with memory persistence, adapter abstraction, streaming and moderation.
    """

    def __init__(
        self,
        open_ai_base_url: str = "",
        openai_api_key: str = "",
        gemini_api_key: str = "",
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.7,
        memory_window: int = 10,
        streaming: bool = True,
        max_prompt_tokens: int = 4000,
        redis_url: Optional[str] = None,
        session_id_getter: Optional[Callable[[], str]] = None,
        retriever: Optional[Callable[[str, int], List[Document]]] = None,
        moderation_hook: Optional[Callable[[str], bool]] = None
    ):
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.memory_window = memory_window
        self.max_prompt_tokens = max_prompt_tokens
        self.streaming = streaming
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.open_ai_base_url = open_ai_base_url

        # Underlying LLM client and adapter based on provider
        self.llm = None
        self.adapter = None
        
        if self.provider == "gemini":
            self._initialize_gemini_llm()
        else:
            self._initialize_openai_llm()

        # Memory persistence store (Redis optional)
        self.memory_store = MemoryStore(redis_url=redis_url)
        self.session_id_getter = session_id_getter or (lambda: "default-session")

        # Conversation memory helper
        self.memory = ConversationMemoryLight(self.memory_store, session_id_getter=self.session_id_getter, window=memory_window)

        # Retriever for RAG (documents or memory). Provide external retriever if needed.
        self.retriever = retriever or default_retriever

        # Moderation hook
        self.moderation_hook = moderation_hook or moderate_text

        # local short-term chat history for quick access (mirrors persisted store)
        self.chat_history: List[Dict] = []

        # track active async streams per session/task
        self._active_stream_tasks: Dict[str, asyncio.Task] = {}
        print("self.memory_window", self.memory_window)
        logger.info("LLMService initialized provider=%s model=%s streaming=%s memory_window=%d", 
                   self.provider.upper(), self.model, self.streaming, self.memory_window)

    def _initialize_openai_llm(self) -> None:
        """Initialize OpenAI LLM adapter."""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided. LLM will not be initialized.")
            return
        
        try:
            if ChatOpenAI is None:
                logger.warning("ChatOpenAI not available")
                return
            
            logger.info("Initializing OpenAI LLM with model=%s", self.model)
            self.llm = ChatOpenAI(
                base_url=self.open_ai_base_url,
                model=self.model,
                temperature=self.temperature,
                streaming=self.streaming,
                openai_api_key=self.openai_api_key
            )
            self.adapter = ChatOpenAIAdapter(self.llm)
        except Exception as e:
            logger.exception("Failed to initialize OpenAI LLM: %s", e)

    def _initialize_gemini_llm(self) -> None:
        """Initialize Gemini LLM adapter."""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not provided. Falling back to OpenAI.")
            self.provider = "openai"
            self._initialize_openai_llm()
            return
        
        try:
            if ChatGoogleGenerativeAI is None:
                logger.warning("ChatGoogleGenerativeAI not available. Falling back to OpenAI.")
                self.provider = "openai"
                self._initialize_openai_llm()
                return
            
            # Gemini LLM API does NOT use models/ prefix - strip it if present
            model_name = self.model
            if model_name.startswith("models/"):
                model_name = model_name[7:]  # Remove "models/" prefix
                logger.debug("Stripped 'models/' prefix from model name: %s", model_name)
            
            logger.info("Initializing Gemini LLM with model=%s", model_name)
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                google_api_key=self.gemini_api_key
            )
            self.adapter = GeminiLLMAdapter(self.llm)
        except Exception as e:
            logger.exception("Failed to initialize Gemini LLM: %s. Falling back to OpenAI.", e)
            self.provider = "openai"
            self._initialize_openai_llm()

    # --- Prompt builders ---
    def build_prompt(self, system_message: Optional[str] = None) -> Any:
        default_system = (
            "You are a helpful AI assistant with access to a knowledge base and conversation memory.\n"
            "Use conversation history and retrieved documents to answer clearly and concisely."
        )
        system = system_message or default_system
        if ChatPromptTemplate:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
                ])
                return prompt
            except Exception:
                return system
        return system

    def build_context_prompt(self, query: str, context_docs: List[Document]) -> str:
        context = "\n\n".join([
            f"Source: {getattr(doc, 'metadata', {}).get('source', 'Unknown')}\n{getattr(doc, 'page_content', str(doc))}"
            for doc in context_docs
        ])
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return prompt

    # --- Internal helpers ---
    def _get_session_id(self) -> str:
        try:
            return self.session_id_getter()
        except Exception:
            return "default-session"

    def _append_memory(self, role: str, content: str) -> None:
        session_id = self._get_session_id()
        entry = {"role": role, "content": content, "timestamp": time.time()}
        # local history
        self.chat_history.append(entry)
        if len(self.chat_history) > self.memory_window:
            self.chat_history = self.chat_history[-self.memory_window:]
        # persist via MemoryStore
        try:
            existing = self.memory_store.get(session_id) or []
            existing.append(entry)
            if len(existing) > self.memory_window:
                existing = existing[-self.memory_window:]
            self.memory_store.set(session_id, existing)
        except Exception as e:
            logger.debug("Memory persistence skipped: %s", e)

    def _get_memory_messages(self) -> List[Dict]:
        session_id = self._get_session_id()
        try:
            persisted = self.memory_store.get(session_id) or []
            return persisted
        except Exception:
            return self.chat_history

    def _enforce_token_limit(self, prompt: str) -> str:
        """
        Intelligently enforce token limit while preserving prompt structure.
        For structured prompts, truncate context and history sections.
        """
        prompt_tokens = simple_token_count(prompt)
        if prompt_tokens <= self.max_prompt_tokens:
            return prompt
        
        logger.warning("Prompt tokens (%d) exceed max (%d). Intelligently truncating...", prompt_tokens, self.max_prompt_tokens)
        
        # Check if this is our structured prompt format
        if "Conversation context:" in prompt and "Previous conversation:" in prompt:
            return self._truncate_simple_structured_prompt(prompt)
        else:
            # Fallback to simple truncation for unstructured prompts
            return truncate_by_tokens(prompt, self.max_prompt_tokens)
    
    def _truncate_simple_structured_prompt(self, prompt: str) -> str:
        """
        Truncate a structured prompt by reducing context and conversation history.
        Priority: Instructions (keep) > User question (keep) > Context (truncate) > History (truncate)
        """
        original_tokens = simple_token_count(prompt)
        
        try:
            # Split into main parts
            parts = {}
            
            # Extract system instructions (everything before "Context from knowledge base:")
            if "Conversation context:" in prompt:
                sys_end = prompt.index("Context from knowledge base:")
                parts["system"] = prompt[:sys_end].strip()
            else:
                parts["system"] = ""
            
            # Extract context
            if "Conversation context:" in prompt and "Previous conversation:" in prompt:
                ctx_start = prompt.index("Context from knowledge base:") + len("Context from knowledge base:")
                ctx_end = prompt.index("Previous conversation:")
                parts["context"] = prompt[ctx_start:ctx_end].strip()
            else:
                parts["context"] = ""
            
            # Extract conversation history
            if "Previous conversation:" in prompt and "User question:" in prompt:
                hist_start = prompt.index("Previous conversation:") + len("Previous conversation:")
                hist_end = prompt.index("User question:")
                parts["history"] = prompt[hist_start:hist_end].strip()
            else:
                parts["history"] = ""
            
            # Extract user question and instructions
            if "User question:" in prompt:
                query_start = prompt.index("User question:")
                parts["query_and_instructions"] = prompt[query_start:].strip()
            else:
                parts["query_and_instructions"] = ""
            
            # Calculate token counts
            system_tokens = simple_token_count(parts["system"])
            query_tokens = simple_token_count(parts["query_and_instructions"])
            context_tokens = simple_token_count(parts["context"])
            history_tokens = simple_token_count(parts["history"])
            
            # Essential sections (must keep)
            essential_tokens = system_tokens + query_tokens + 20  # +20 for spacing
            available_tokens = self.max_prompt_tokens - essential_tokens
            
            if available_tokens <= 0:
                logger.warning("Essential sections exceed token limit. Using minimal prompt.")
                available_tokens = 200
            
            # Allocate: 70% to context, 30% to history
            context_budget = int(available_tokens * 0.7)
            history_budget = int(available_tokens * 0.3)
            
            # Truncate context if needed
            if context_tokens > context_budget:
                parts["context"] = truncate_by_tokens(parts["context"], context_budget) + "\n\n[... context truncated ...]"
            
            # Truncate history if needed (keep most recent)
            if history_tokens > history_budget:
                parts["history"] = truncate_by_tokens(parts["history"], history_budget) + "\n\n[... older messages truncated ...]"
            
            # Reconstruct
            reconstructed = f"""{parts["system"]}

Context from knowledge base:
{parts["context"]}

Previous conversation:
{parts["history"]}

{parts["query_and_instructions"]}"""
            
            final_tokens = simple_token_count(reconstructed)
            logger.info("Truncated prompt: %d tokens (original: %d, limit: %d)", 
                       final_tokens, original_tokens, self.max_prompt_tokens)
            
            return reconstructed
            
        except Exception as e:
            logger.exception("Error in structured truncation: %s", e)
            # Fallback to simple truncation
            return truncate_by_tokens(prompt, self.max_prompt_tokens)

    def _moderate(self, text: str) -> None:
        allowed = self.moderation_hook(text)
        if not allowed:
            logger.warning("Content blocked by moderation.")
            raise ModerationError("Input blocked by moderation policy")

    # --- Public API: generate non-streaming ---
    def generate_answer(self, query: str, context_docs: Optional[List[Document]] = None) -> str:
        session_id = self._get_session_id()
        logger.info("generate_answer called, session=%s", session_id)
        if REQUEST_COUNTER:
            REQUEST_COUNTER.inc()
        start = time.time()
        try:
            # moderation
            self._moderate(query)

            used_context = context_docs
            if not used_context:
                try:
                    used_context = self.retriever(query, k=5)
                except Exception as e:
                    logger.debug("Retriever failed: %s", e)
                    used_context = []

            # Build context from retrieved documents
            context_text = ""
            if used_context:
                context_text = "\n\n".join([
                    f"Source: {getattr(doc, 'metadata', {}).get('source', 'Unknown')}\n{getattr(doc, 'page_content', str(doc))}"
                    for doc in used_context
                ])

            # Get message history
            memory_msgs = self._get_memory_messages()
            message_history_text = ""
            if memory_msgs:
                message_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in memory_msgs])

            # Build structured prompt following the OnboardingAI template
            # Use a more concise format to avoid content policy triggers
            structured_prompt = f"""You are an intelligent assistant created to learn all information about the company's projects. Your task is to respond to users who are either new to the project or need information about the project. Find the information you have learned and answer the users.
    Rules:
    - Find synonyms in the knowledge you can learn
    - Provide accurate, concise answers related to the base data
    - Use previous messages for continuity if exist

    Conversation context:
    {context_text if context_text else "No history yet"}

    Previous conversation:
    {message_history_text if message_history_text else "No history yet"}

    User question: {query}

    Instructions: Provide a helpful answer base on the knowledge you learned."""

            # Enforce token limit on the structured prompt
            structured_prompt = self._enforce_token_limit(structured_prompt)

            # call adapter
            if not self.adapter:
                raise RuntimeError("LLM adapter not initialized")
            res = self.adapter.invoke(structured_prompt)
            content = getattr(res, "content", res if isinstance(res, str) else str(res))

            # post moderation
            if not self.moderation_hook(content):
                raise ModerationError("Model output blocked by moderation")

            # append memory
            self._append_memory("user", query)
            self._append_memory("assistant", content)
            logger.info("Structured Prompt: %s", structured_prompt)
            logger.info("LLM response: %s", content)
            return content
        finally:
            if REQUEST_LATENCY:
                REQUEST_LATENCY.observe(time.time() - start)

    # --- Cancel streams for a session ---
    def cancel_streams_for_session(self, session_id: str) -> int:
        canceled = 0
        keys = [k for k in list(self._active_stream_tasks.keys()) if k.startswith(f"{session_id}:")]
        for k in keys:
            t = self._active_stream_tasks.get(k)
            if t and not t.done():
                t.cancel()
                canceled += 1
        return canceled

    # --- Public accessors ---
    def get_llm_instance(self) -> ChatOpenAI:
        return self.llm

    def get_memory_instance(self) -> List:
        try:
            return self._get_memory_messages()
        except Exception:
            return self.chat_history

    def clear_memory(self) -> None:
        session_id = self._get_session_id()
        self.chat_history.clear()
        try:
            self.memory_store.clear(session_id)
        except Exception as e:
            logger.debug("Memory clear failed: %s", e)

    def get_stats(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider.upper(),
            "temperature": self.temperature,
            "memory_window": self.memory_window,
            "max_prompt_tokens": self.max_prompt_tokens,
            "memory_persisted": self.memory_store.enabled
        }
