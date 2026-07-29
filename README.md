# Graph Gardener

**LLM-powered knowledge graph maintenance** — cleans and enriches memory graphs.

## Quick Start

```bash
pip install git+https://github.com/jmars/graph-gardener
```

### Configuration

Set these environment variables to point at your LLM provider:

```bash
export GRAPH_GARDENER_API_URL="https://api.deepseek.com/v1"   # or your provider
export GRAPH_GARDENER_API_KEY="sk-..."                          # your API key
export GRAPH_GARDENER_MODEL="deepseek-chat"                     # model name
```

Works with any service that speaks the OpenAI `/v1/chat/completions` protocol:
OpenAI, DeepSeek, Groq, Together, Ollama, LM Studio, and more.

### Usage

```bash
# Dry run (preview what would change)
graph-gardener

# Apply mutations
graph-gardener --apply

# Custom memory file
graph-gardener --memory-file /path/to/memory.jsonl --apply
```

> ⚠️ **Privacy note**: This tool sends your knowledge graph content (entity names,
> types, and observation excerpts) to the configured LLM API endpoint.

## How It Works

1. **Load graph** — reads a JSONL memory file (entities + relations)
2. **Build prompt** — constructs a structured prompt with entity summaries and relations
3. **Call LLM** — sends to any OpenAI-compatible API for analysis
4. **Apply safe mutations** — never deletes, always additive:

   | Mutation | Description |
   |---|---|
   | `archive_observations` | Tags stale observations with `[archived: date reason]` |
   | `rename_types` | Consolidates entity types (e.g. `Technique` → `convention`) |
   | `merge_entities` | Combines duplicate entities with deduplication |
   | `add_entities` | Creates new summary entities from patterns |
   | `add_relations` | Adds missing relations between entities |

### Safety Guarantees

- **Never deletes** observations — only appends archive tags
- **Always backs up** before `--apply` — time-stamped backup created alongside the original file
- **Atomic writes** — uses temporary file + `os.replace()` to prevent corruption
- **Self-reference cleanup** — merged entities' self-referencing relations are automatically removed
- **Dry run by default** — preview changes without modifying anything

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GRAPH_GARDENER_API_URL` | `https://api.deepseek.com/v1` | Base URL for chat completions API |
| `GRAPH_GARDENER_API_KEY` | *(required)* | Bearer token for API authentication |
| `GRAPH_GARDENER_MODEL` | `deepseek-chat` | Model name passed to the API |

## Security

- **API keys** — never stored or logged. Error messages redact keys before printing.
- **Prompt injection** — graph content is included in LLM prompts verbatim. Only process memory files from trusted sources.
- **URL validation** — only HTTPS endpoints are accepted (localhost excepted for local models). Raw IP addresses are blocked to prevent SSRF.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup
instructions, development workflow, and pull request guidelines.

This project is licensed under the MIT License.

## License

MIT
