# Quick Reference: Ollama Integration

## One-Minute Setup

```powershell
# 1. Download from ollama.com, run installer

# 2. Pull a model (in new PowerShell)
ollama pull mistral

# 3. Update .env
PROVIDER=ollama

# 4. Run backend
cd backend
python -m app.main
```

That's it! 🎉

## Common Commands

```powershell
# Download a model
ollama pull mistral

# List models
ollama list

# Run interactive session
ollama run mistral

# Start server (if not running)
ollama serve

# Stop server
# Close the terminal window or press Ctrl+C
```

## .env Cheat Sheet

**Ollama:**
```env
PROVIDER=ollama
OLLAMA_MODEL=mistral
```

**OpenAI:**
```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

**Gemini:**
```env
PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

## Model Quick Pick

- **Testing**: `tinyllama` (400MB, super fast)
- **Good Quality**: `mistral` (4.1GB, recommended)
- **Best**: `neural-chat` (4GB, excellent)

## Files You Need to Know

- `OLLAMA_SETUP.md` - Full installation guide
- `PROVIDER_SWITCHING.md` - Switch between providers
- `OLLAMA_INTEGRATION_COMPLETE.md` - What was changed
- `.env.ollama.example` - Example configuration

## Is Ollama Running?

```powershell
curl http://localhost:11434/api/tags
```

If you see model list → ✅ Running
If you see error → ❌ Not running (start with `ollama serve`)

## Still Not Working?

1. **Ollama installed?** Download from ollama.com
2. **Model pulled?** Run `ollama pull mistral`
3. **Server running?** Check Windows tray or run `ollama serve`
4. **Package installed?** Run `pip install langchain-ollama`
5. **Config updated?** Set `PROVIDER=ollama` in .env

See `OLLAMA_SETUP.md` for detailed troubleshooting.

---

**Done! Your chatbot now supports local LLMs with Ollama.** 🚀
