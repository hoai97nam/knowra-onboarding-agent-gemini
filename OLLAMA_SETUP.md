# Ollama Setup Guide

This guide helps you set up and use Ollama as an LLM provider for the Knowra Chatbot.

## What is Ollama?

Ollama is a tool for running large language models locally on your machine. It's perfect for:
- **Local privacy**: Your data never leaves your computer
- **Cost-effective**: No API calls or subscription fees
- **Offline capability**: Works without internet connection
- **Fast iteration**: Instant model switching

## Installation

### Windows 10/11

1. **Download Ollama**
   - Visit [ollama.ai](https://ollama.ai) or [ollama.com](https://ollama.com)
   - Click "Download" and select Windows
   - Or download directly: https://ollama.ai/download/windows

2. **Install Ollama**
   - Run the installer (.exe file)
   - Follow the installation prompts
   - Ollama will be installed and added to your PATH
   - A system tray icon will appear

3. **Verify Installation**
   ```powershell
   ollama --version
   ```

### macOS

```bash
# Download and install
curl -fsSL https://ollama.ai/install.sh | sh
```

### Linux

```bash
# Download and install
curl -fsSL https://ollama.ai/install.sh | sh
```

## Running Ollama

### Option 1: System Service (Recommended)

Ollama automatically runs as a background service after installation:
- Windows: System tray icon in taskbar
- macOS/Linux: Automatically running in background

Verify it's running:
```powershell
# Windows - Test the API endpoint
curl http://localhost:11434/api/tags
```

### Option 2: Manual Start

```powershell
# Windows - Start Ollama from command line
ollama serve
```

The server will run on `http://localhost:11434` by default.

## Pulling Models

Before using a model, you need to pull (download) it first:

```powershell
# Pull a model
ollama pull mistral

# Pull other popular models
ollama pull llama2
ollama pull neural-chat
ollama pull dolphin-mixtral
ollama pull orca-mini
ollama pull tinyllama  # Fastest, smallest, good for testing
```

### Model Recommendations

For your machine's resources, we recommend:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **tinyllama** | 400MB | ⚡⚡⚡ | ⭐ | Testing, development |
| **orca-mini** | 1.4GB | ⚡⚡ | ⭐⭐⭐ | Good balance |
| **neural-chat** | 4GB | ⚡ | ⭐⭐⭐⭐ | Quality conversations |
| **mistral** | 4.1GB | ⚡ | ⭐⭐⭐⭐ | Excellent performance |
| **llama2** | 3.8GB | ⚡⭐ | ⭐⭐⭐⭐ | Very capable |

**Quick start recommendation**: Start with `tinyllama` for testing, then upgrade to `neural-chat` or `mistral` for production use.

### List Downloaded Models

```powershell
ollama list
```

## Configure Your Application

### 1. Update Environment Variables

Create or update your `.env` file in the project root:

```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 2. Install Required Package

```powershell
# Navigate to backend directory
cd backend

# Install langchain-ollama
pip install langchain-ollama
```

Or update requirements:
```powershell
pip install -r requirements.txt
```

### 3. Start Your Application

```powershell
# Make sure Ollama is running first!
ollama serve

# In another terminal, start the backend
cd backend
python -m app.main
```

## Available Ollama Models

Check available models on the Ollama registry:
https://ollama.ai/library

Some popular choices:

```powershell
# Chat/Conversation models
ollama pull mistral
ollama pull neural-chat
ollama pull dolphin-mixtral
ollama pull openchat

# Small/Fast models (good for testing)
ollama pull tinyllama
ollama pull orca-mini

# Large/Powerful models (better quality, needs more VRAM)
ollama pull llama2
ollama pull llama2-uncensored
ollama pull nous-hermes

# Specialized models
ollama pull codellama  # For code generation
ollama pull deepseek-coder  # For coding tasks
```

## Testing Ollama

### Test via Command Line

```powershell
ollama run mistral
# Then type your prompt and press Enter
```

### Test via API

```powershell
# Using PowerShell
$body = @{
    model = "mistral"
    prompt = "Why is Ollama great?"
    stream = $false
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:11434/api/generate" `
  -Method Post `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

Or using curl:
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "prompt": "Why is the sky blue?",
    "stream": false
  }'
```

## Troubleshooting

### Ollama Not Found

```powershell
# Add Ollama to PATH (Windows)
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"

# Or verify installation
Get-Command ollama
```

### Connection Refused

```powershell
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Or on Windows, check system tray icon
```

### Model Not Found

```powershell
# Download the model first
ollama pull mistral

# Verify it's downloaded
ollama list
```

### Out of Memory

- Use a smaller model (tinyllama, orca-mini)
- Reduce OLLAMA_MEMORY if needed
- Close other applications

### Slow Performance

- Use a faster model (tinyllama, neural-chat)
- Ensure Ollama has enough RAM
- Check system resources

## Environment Variables Reference

```env
# Required
PROVIDER=ollama

# Optional - defaults to these values
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.7
```

## Next Steps

1. **Install Ollama**: Download and run the installer
2. **Pull a Model**: `ollama pull mistral`
3. **Verify Setup**: `curl http://localhost:11434/api/tags`
4. **Update .env**: Set `PROVIDER=ollama`
5. **Start Application**: Run your backend
6. **Test It**: Send a request to your API

## Performance Tips

1. **Model Selection**
   - Start small (tinyllama) and upgrade as needed
   - Use quantized versions for faster performance

2. **System Resources**
   - Close unnecessary applications
   - Ensure adequate RAM (8GB minimum recommended)
   - Use SSD for faster model loading

3. **Configuration**
   - Adjust temperature for consistency vs creativity
   - Use smaller context windows for faster responses
   - Enable GPU acceleration if available

## Comparison: Ollama vs Cloud Providers

| Aspect | Ollama | OpenAI | Gemini |
|--------|--------|--------|---------|
| **Setup** | Local install | API key | API key |
| **Cost** | Free | Per token | Free/Paid |
| **Privacy** | 100% Local | Cloud | Cloud |
| **Speed** | Depends on hardware | Fast | Fast |
| **Model choice** | Large selection | Limited | Limited |
| **Internet** | Not needed | Required | Required |
| **Control** | Full | Limited | Limited |

## Additional Resources

- **Official Documentation**: https://github.com/ollama/ollama
- **Model Library**: https://ollama.ai/library
- **Discord Community**: https://discord.gg/ollama
- **Issue Tracker**: https://github.com/ollama/ollama/issues

## Quick Commands Reference

```powershell
# Installation & Setup
ollama --version                 # Check version
ollama pull mistral              # Download a model
ollama list                       # List installed models

# Running
ollama serve                      # Start server
ollama run mistral               # Run interactive session

# API Testing
curl http://localhost:11434/api/tags  # List available models
```

---

**Happy coding with Ollama! 🚀**
