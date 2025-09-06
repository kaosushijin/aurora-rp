# Aurora RPG Client (Proof-of-Concept)

Aurora RPG is a terminal-based RPG storyteller client that interacts with a language model over MCP (Model Control Protocol). It maintains memory of previous interactions to provide a continuous, immersive storytelling experience.

## Features

- Multi-line user input
- Memory management (`memory.json`) for context persistence
- Commands:
  - `/quit` → exit
  - `/mems` → view last 10 memory entries
  - `/dump` → view full memory (debug mode only)
- Debug mode (`--debug`) for verbose MCP responses
- Async calls to MCP server with retry on timeout

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install httpx colorama
