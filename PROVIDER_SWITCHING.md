# Provider Switching Guide

This document shows how to switch between OpenAI, Gemini, and Ollama providers.

## Quick Start by Provider

### OpenAI Provider

**1. Update `.env`:**
```env
PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
```

**2. Verify installation:**
```powershell
pip install langchain-openai
```

**3. Start your backend:**
```powershell
python -m app.main
```

---

### Gemini (Google) Provider

**1. Update `.env`:**
```env
PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.7
```

**2. Verify installation:**
```powershell
pip install langchain-google-genai
```

**3. Start your backend:**
```powershell
python -m app.main
```

---

### Ollama Provider (Local)

**1. Ensure Ollama is running:**
```powershell
# Windows system tray or start manually:
ollama serve

# In another terminal, pull a model:
ollama pull mistral
```

**2. Update `.env`:**
```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TEMPERATURE=0.7
```

**3. Verify installation:**
```powershell
pip install langchain-ollama
```

**4. Start your backend:**
```powershell
python -m app.main
```

---

## Environment Variable Reference

### Provider Selection
```env
PROVIDER=openai          # Use OpenAI
PROVIDER=gemini          # Use Google Gemini
PROVIDER=ollama          # Use local Ollama
```

### OpenAI Settings
```env
OPENAI_BASE_URL=https://api.openai.com/v1    # Default OpenAI endpoint
OPENAI_API_KEY=sk-...                         # Your API key
OPENAI_MODEL=gpt-4o-mini                      # Model name
OPENAI_EMBEDDING_MODEL=text-embedding-3-small # Embedding model
OPENAI_TEMPERATURE=0.7                        # 0-1, lower = more deterministic
```

### Gemini Settings
```env
GEMINI_API_KEY=...                # Your API key
GEMINI_MODEL=gemini-1.5-flash     # Model name
GEMINI_EMBEDDING_MODEL=text-embedding-004  # Embedding model
```

### Ollama Settings
```env
OLLAMA_BASE_URL=http://localhost:11434  # Default Ollama endpoint
OLLAMA_MODEL=mistral                     # Model name to use
OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # Embedding model
OLLAMA_TEMPERATURE=0.7                   # 0-1, lower = more deterministic
```

---

## Model Selection Examples

### Quick Performance Matrix

| Use Case | Provider | Model | Size | Speed |
|----------|----------|-------|------|-------|
| **Testing** | Ollama | tinyllama | 400MB | ⚡⚡⚡ |
| **Development** | Ollama | orca-mini | 1.4GB | ⚡⚡ |
| **Production** | OpenAI | gpt-4o-mini | Cloud | ⚡⚡⚡ |
| **Production** | Gemini | gemini-1.5-flash | Cloud | ⚡⚡⚡ |
| **High Quality** | Ollama | mistral | 4.1GB | ⚡ |

### Recommended Models by Provider

**OpenAI:**
- `gpt-4o-mini` - Balanced, cost-effective
- `gpt-4o` - Highest quality
- `gpt-3.5-turbo` - Fast, cheaper

**Gemini:**
- `gemini-1.5-flash` - Fast, recommended
- `gemini-1.5-pro` - More capable
- `gemini-2.0-flash` - Latest

**Ollama:**
- `tinyllama` - Testing/demo (400MB)
- `orca-mini` - Good quality (1.4GB)
- `neural-chat` - Excellent for chat (4GB)
- `mistral` - Very capable (4.1GB)
- `llama2` - Powerful option (3.8GB)

---

## Switching Between Providers

### From OpenAI to Ollama

```powershell
# 1. Stop current backend
# Press Ctrl+C in terminal

# 2. Ensure Ollama is running
ollama serve
ollama pull mistral

# 3. Update .env
# Change PROVIDER=openai to PROVIDER=ollama
# Add OLLAMA_BASE_URL=http://localhost:11434
# Set OLLAMA_MODEL=mistral

# 4. Install package (if not already installed)
pip install langchain-ollama

# 5. Restart backend
python -m app.main
```

### From Gemini to OpenAI

```powershell
# 1. Update .env
# Change PROVIDER=gemini to PROVIDER=openai
# Add OPENAI_API_KEY=sk-...
# Set OPENAI_MODEL=gpt-4o-mini

# 2. Install package (if not already installed)
pip install langchain-openai

# 3. Restart backend
python -m app.main
```

### From Ollama to Cloud (OpenAI/Gemini)

```powershell
# 1. Update .env
# Option A: Change to OpenAI
#   PROVIDER=openai
#   OPENAI_API_KEY=sk-...

# Option B: Change to Gemini
#   PROVIDER=gemini
#   GEMINI_API_KEY=...

# 2. Restart backend
python -m app.main

# Note: Ollama no longer needs to be running
```

---

## Troubleshooting Provider Issues

### "Provider not found" or "Module not found"

```powershell
# Install all providers
pip install langchain-openai langchain-google-genai langchain-ollama

# Or install specific provider
pip install langchain-openai
pip install langchain-google-genai
pip install langchain-ollama
```

### Provider not initializing

Check logs:
```powershell
# Look for initialization messages
# Set log level to DEBUG
# In your terminal: $env:LLM_SERVICE_LOG_LEVEL="DEBUG"

python -m app.main
```

### API key not recognized

```env
# Make sure .env is in project root
# Not in backend/ folder

# Verify format - no quotes needed
OPENAI_API_KEY=sk-...  # Correct
OPENAI_API_KEY="sk-..." # Wrong
```

### Connection refused (Ollama)

```powershell
# Ensure Ollama is running
ollama serve

# Check it's accessible
curl http://localhost:11434/api/tags
```

---

## Performance Tuning by Provider

### Ollama Performance Tips
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TEMPERATURE=0.5           # Lower = faster, more consistent
MEMORY_WINDOW=5                  # Smaller window = faster
CHUNK_SIZE=500                   # Smaller chunks = faster
```

### OpenAI Performance Tips
```env
OPENAI_MODEL=gpt-3.5-turbo       # Faster than gpt-4
OPENAI_TEMPERATURE=0             # More deterministic
```

### Gemini Performance Tips
```env
GEMINI_MODEL=gemini-1.5-flash    # Faster than pro
```

---

## Cost Comparison

| Provider | Cost | Best For |
|----------|------|----------|
| **Ollama** | $0 (local) | Development, privacy, offline |
| **OpenAI** | ~$0.15/M input tokens | Production, reliability |
| **Gemini** | Free (with limits) | Cost-effective production |

---

## Code Changes Summary

All providers are now fully integrated! You just need to:

1. **Set `PROVIDER` in `.env`** - That's it!
2. **Ensure required packages are installed** - `pip install -r requirements.txt`
3. **Provide necessary credentials/setup**:
   - OpenAI: API key
   - Gemini: API key
   - Ollama: Running service + model pulled

The `LLMService` class automatically:
- Detects the provider from `PROVIDER` setting
- Initializes the correct adapter
- Falls back gracefully if services unavailable
- Supports streaming and sync generation

---

## Example .env Files

See included files:
- `.env.ollama.example` - Ollama configuration
- Add `.env.openai.example` for OpenAI
- Add `.env.gemini.example` for Gemini

---

**Need help? Check OLLAMA_SETUP.md for detailed Ollama instructions!**
