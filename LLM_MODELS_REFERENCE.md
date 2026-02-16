# LLM Models Reference Guide

This guide provides comprehensive information about configuring and optimizing Large Language Models for use with the LLM UI application.

## Table of Contents

1. [Supported Model Types](#supported-model-types)
2. [Model Configuration](#model-configuration)
3. [Optimal Model Settings](#optimal-model-settings)
4. [Embedding Models](#embedding-models)
5. [Reranking Models](#reranking-models)
6. [Thinking Models](#thinking-models)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Supported Model Types

The LLM UI application works with any model compatible with llama.cpp, including:

### Chat Models
- **Llama 2/3 series** (Meta)
- **Mistral/Mixtral** (Mistral AI)
- **Phi-2/Phi-3** (Microsoft)
- **Qwen series** (Alibaba)
- **Gemma** (Google)
- **Zephyr** (Hugging Face)
- **CodeLlama** (Meta)
- **Nous-Hermes** (Nous Research)

### Specialized Models
- **Embedding models** for semantic search
- **Reranking models** for result ordering
- **Thinking models** with reasoning capabilities

## Model Configuration

### Basic Configuration

Models are configured in `backend/config.py`:

```python
# Main LLM model for chat completion
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "glm4.7-30ba3b")

# Model for query processing and title generation (should be non-thinking)
QUERY_MODEL = os.getenv("QUERY_MODEL", "qwen3-30ba3b")

# Base URL for llama.cpp server
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
```

### Environment Variables

You can also configure models using environment variables:

```bash
# Main model for chat
LLAMA_CPP_MODEL="mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Model for quick queries and titles
QUERY_MODEL="phi-2.Q4_K_M.gguf"

# llama.cpp server URL
LLAMA_CPP_URL="http://localhost:8080"

# Model parameters
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=16048
```

### Model-Specific Parameters

Different models may require different optimal settings:

```python
# Temperature settings by model type
TEMPERATURE_SETTINGS = {
    "creative_writing": 0.8,    # Higher creativity
    "coding": 0.2,              # More deterministic
    "analysis": 0.5,            # Balanced
    "query_processing": 0.3     # Focused responses
}
```

## Optimal Model Settings

### General Purpose Models

| Model Family | Recommended Quantization | VRAM Usage | Best For |
|--------------|--------------------------|------------|----------|
| Llama 3 8B | Q4_K_M | 4.5GB | General use, balanced |
| Mistral 7B | Q4_K_M | 4.0GB | Instruction following |
| Phi-3 Mini | Q4_K_M | 2.8GB | Low resource systems |
| Qwen 2.5 7B | Q4_K_M | 4.0GB | Multilingual support |

### llama.cpp Server Configuration

For optimal performance, start your llama.cpp server with appropriate parameters:

```bash
# General purpose
./llama-server -m /path/to/model.gguf --port 8080 --host 0.0.0.0 --ctx-size 4096 --batch-size 512 --threads 8

# With embeddings support (required for RAG features)
./llama-server -m /path/to/model.gguf --port 8080 --host 0.0.0.0 --ctx-size 4096 --batch-size 512 --threads 8 --embeddings

# For smaller models on limited hardware
./llama-server -m /path/to/model.gguf --port 8080 --host 0.0.0.0 --ctx-size 2048 --batch-size 256 --threads 4 --memory-f32
```

### Context Window Considerations

- **4K context**: Good for most conversations
- **8K context**: Better for document analysis and longer conversations
- **16K context**: Required for large document processing (uses more VRAM)

## Embedding Models

For RAG (Retrieval-Augmented Generation) and web search features, you need embedding models:

### Recommended Embedding Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| BGE-M3 | 1.2GB | State-of-the-art | Best overall |
| BGE-base-en-v1.5 | 440MB | High quality | Good balance |
| All-MiniLM-L6-v2 | 90MB | Good | Resource-constrained |
| Sentence-T5 | 200MB | High quality | Semantic search |

### Embedding Model Configuration

```python
# In config.py
EMBEDDINGS_API = f"{LLAMA_CPP_BASE_URL}/v1/embeddings"
EMBEDDING_MODEL = "bge-large-en-v1.5"  # or your chosen model
```

### llama.cpp Embedding Server

Start a separate server for embeddings:

```bash
# Dedicated embedding server
./llama-server -m /path/to/embedding-model.gguf --port 8081 --host 0.0.0.0 --embeddings --ctx-size 512
```

## Reranking Models

For improved search result ordering:

### Recommended Reranking Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| BGE-Reranker-v2-Mini | 130MB | Fast and accurate | Best balance |
| BGE-Reranker-v2-Pointwise | 360MB | High accuracy | Precision required |
| UER/RoBERTa-base | 440MB | Good performance | Alternative option |

### Reranking Configuration

```python
# In config.py
RERANK_API = f"{LLAMA_CPP_BASE_URL}/v1/rerank"
RERANK_MODEL = "bge-reranker-v2-minic"  # or your chosen model
```

## Thinking Models

Some models have separate "thinking" and "response" phases:

### Compatible Thinking Models

- **DeepSeek-R1** series
- **Claude-style** reasoning models
- **Self-reflective** models

### Configuration for Thinking Models

```python
# In config.py
THINKING_MODEL_ENABLED = True
THINKING_EXTRACT_PATTERN = r"<think>(.*?)</think>"  # Pattern to extract thinking
```

### UI Behavior

- Thinking content appears in collapsible sections
- Final response follows the thinking phase
- Both are preserved in conversation history

## Performance Optimization

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU Cores | 4 | 8+ |
| RAM | 8GB | 16GB+ |
| GPU VRAM | 4GB | 8GB+ |
| Storage | SSD | NVMe SSD |

### Model Selection by Hardware

#### Low-Resource Systems (< 8GB RAM)
- Use Phi-3 Mini (1.3B) or smaller
- Q4_K_M quantization
- 2K context window
- 4 threads

#### Mid-Range Systems (8-16GB RAM)
- Use Mistral 7B or Llama 3 8B
- Q4_K_M or Q5_K_M quantization
- 4K context window
- 6-8 threads

#### High-End Systems (> 16GB RAM)
- Use Llama 3 70B, Mixtral 8x7B, or Qwen 72B
- Q4_K_M quantization
- 8K+ context window
- 8+ threads

### llama.cpp Optimization Flags

```bash
# For NVIDIA GPUs
./llama-server -m model.gguf --port 8080 --n-gpu-layers 100 --tensor-split 0.5

# For AMD GPUs
./llama-server -m model.gguf --port 8080 --n-gpu-layers 100 --mmq

# For CPU-only systems
./llama-server -m model.gguf --port 8080 --parallel 4 --cont-batching
```

## Troubleshooting

### Common Issues

#### Model Not Loading
**Symptoms:** Server starts but model fails to load
**Solutions:**
1. Check VRAM availability
2. Verify model file integrity
3. Try different quantization
4. Increase system RAM or swap

#### Slow Response Times
**Symptoms:** Long delays in generating responses
**Solutions:**
1. Reduce context size
2. Use smaller model
3. Increase threads
4. Check for background processes consuming resources

#### Embedding/Reranking Failures
**Symptoms:** Web search or RAG features not working
**Solutions:**
1. Verify embedding server is running
2. Check model compatibility
3. Ensure `--embeddings` flag is used
4. Verify API endpoints

#### Memory Issues
**Symptoms:** Out of memory errors, crashes
**Solutions:**
1. Use smaller context windows
2. Reduce batch size
3. Use more aggressive quantization
4. Add system swap space

### Model Compatibility Check

To verify your model works with the application:

```bash
# Test chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Test embeddings (if applicable)
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-embedding-model",
    "input": "Test text for embedding"
  }'
```

### Performance Monitoring

Monitor resource usage during operation:

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU and memory
htop

# Monitor llama.cpp server logs
tail -f llama-server.log
```

## Model Recommendations by Use Case

### General Chat Assistant
- **Recommended:** Llama 3 8B, Mistral 7B
- **Quantization:** Q4_K_M
- **Context:** 4K
- **Purpose:** Balanced performance for general conversations

### Coding Assistant
- **Recommended:** CodeLlama 7B, Phind-CodeLlama 34B
- **Quantization:** Q4_K_M
- **Context:** 8K
- **Purpose:** Better for code generation and analysis

### Research Assistant
- **Recommended:** Llama 3 70B, Mixtral 8x7B
- **Quantization:** Q4_K_M
- **Context:** 8K+
- **Purpose:** Complex reasoning and analysis

### Lightweight/Demo
- **Recommended:** Phi-3 Mini, TinyLlama
- **Quantization:** Q4_K_M
- **Context:** 2K
- **Purpose:** Low-resource environments

## Updating Models

### Changing Models

1. Download new model file (.gguf format)
2. Update `LLAMA_CPP_MODEL` in config
3. Restart the llama.cpp server
4. Restart the LLM UI application

### Model Hot-Swapping

For zero-downtime model updates:

1. Start new llama.cpp server on different port
2. Update application config to point to new server
3. Restart application
4. Shut down old server

## Community Resources

### Model Datasets
- [Hugging Face Models](https://huggingface.co/models)
- [TheBloke's GGUF Collection](https://huggingface.co/TheBloke)
- [GGML Model Zoo](https://github.com/ggerganov/llama.cpp/discussions)

### Performance Benchmarks
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [llama.cpp Performance Tests](https://github.com/ggerganov/llama.cpp/wiki/Performance-Benchmarks)

---

**Next Steps:**
- [Quick Start Guide](QUICKSTART.md) - Get up and running
- [Development Guide](DEVELOPMENT.md) - Customize the application
- [Project Structure](PROJECT_STRUCTURE.md) - Understand the codebase