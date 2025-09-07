import asyncio
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any
import httpx
import sys
from colorama import init, Fore, Style

# ------------------ Initialization ------------------ #
init(autoreset=True)
DEBUG = "--debug" in sys.argv

# ------------------ Configuration ------------------ #
MCP_URL = "http://127.0.0.1:3456/chat"
MODEL = "qwen2.5:14b-instruct-q4_k_m"
SAVE_FILE = Path("memory.json")
TIMEOUT = 300.0  # seconds
TOTAL_VRAM_BYTES = 8 * 1024**3
BYTES_PER_TOKEN = 10
MAX_TOKENS = TOTAL_VRAM_BYTES // BYTES_PER_TOKEN
MAX_MEMORY_TOKENS = int(MAX_TOKENS * 0.7)
MAX_PROMPT_TOKENS = int(MAX_TOKENS * 0.3)

# System prompt to initialize memory if empty
SYSTEM_PROMPT = (
    "You are the an RPG storyteller. Your role is to guide the player through a high-fantasy world, "
    "creating engaging scenarios, characters, and adventures. Maintain continuity across the story, "
    "remember player choices, and respond creatively and consistently."
)

# ------------------ Memory Handling ------------------ #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_memory() -> List[Dict[str, Any]]:
    if SAVE_FILE.exists():
        memories = json.loads(SAVE_FILE.read_text())
        if not memories:
            memories.append({"id": str(uuid4()), "role": "system", "content": SYSTEM_PROMPT, "timestamp": now_iso()})
        return memories
    else:
        memories = [{"id": str(uuid4()), "role": "system", "content": SYSTEM_PROMPT, "timestamp": now_iso()}]
        save_memory(memories)
        return memories

def save_memory(memories: List[Dict[str, Any]]):
    SAVE_FILE.write_text(json.dumps(memories, indent=2))

def add_memory(memories: List[Dict[str, Any]], role: str, content: str):
    memories.append({
        "id": str(uuid4()),
        "role": role,
        "content": content,
        "timestamp": now_iso()
    })
    save_memory(memories)

def estimate_tokens(text: str) -> int:
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4

async def condense_memory(memories: List[Dict[str, Any]], fraction: float = 0.2) -> List[Dict[str, Any]]:
    """
    Condense the oldest fraction of memories using the MCP.
    """
    if not memories:
        return memories

    num_to_condense = max(1, int(len(memories) * fraction))
    oldest_chunk = memories[:num_to_condense]

    text_to_condense = "\n\n".join([f"{mem['role'].capitalize()}: {mem['content']}" for mem in oldest_chunk])
    prompt = (
        "You are an RPG storyteller with memory management capabilities. "
        "Condense the following memories into a concise summary that preserves key facts, characters, and story continuity, "
        "while minimizing token usage:\n\n"
        f"{text_to_condense}\n\n"
        "Provide only the condensed memory text."
    )

    condensed_text = await call_mcp([{"role": "system", "content": prompt}])

    condensed_memory = {
        "id": str(uuid4()),
        "role": "system",
        "content": condensed_text,
        "timestamp": now_iso()
    }

    return [condensed_memory] + memories[num_to_condense:]

# ------------------ MCP Call ------------------ #
async def call_mcp(messages: List[Dict[str, str]]) -> str:
    timeout = httpx.Timeout(TIMEOUT, read=TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                resp = await client.post(MCP_URL, json={"model": MODEL, "messages": messages})
                resp.raise_for_status()
                data = resp.json()

                if DEBUG:
                    print(Fore.YELLOW + "[Debug] MCP response:\n" + json.dumps(data, indent=2))

                if "result" in data:
                    return data["result"]
                elif "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                else:
                    return "[System] MCP response format unrecognized."
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                if attempt < 2:
                    print(Fore.YELLOW + f"[Warning] Timeout on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    return "[System] The storyteller is silent... (connection timed out)"
            except httpx.HTTPError as e:
                return f"[System] HTTP error: {e}"

# ------------------ Main Loop ------------------ #
async def main_loop():
    print(Fore.CYAN + "Aurora RPG Client (POC-mem)")
    print(Fore.CYAN + "Type your action. Press ENTER twice to submit.")
    print(Fore.CYAN + "Terminate using: /quit\n")

    memories = load_memory()
    buffer = []

    while True:
        line = input(Fore.BLUE + "> ").rstrip()
        if line == "":
            if not buffer:
                continue
            raw_input_text = "\n".join(buffer)
            buffer.clear()
        else:
            buffer.append(line)
            continue

        if raw_input_text.startswith("/quit"):
            print(Fore.YELLOW + "Goodbye!")
            break

        # Add user message to memory
        add_memory(memories, "user", raw_input_text)

        # Condense memory until under MAX_MEMORY_TOKENS
        total_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
        while total_tokens > MAX_MEMORY_TOKENS:
            memories = await condense_memory(memories, fraction=0.2)
            total_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)

        # Prepare messages for MCP including all previous memory
        messages = [{"role": mem["role"], "content": mem["content"]} for mem in memories]

        # Call MCP
        response = await call_mcp(messages)

        # Save assistant response and display
        add_memory(memories, "assistant", response)
        print(Fore.GREEN + response)

if __name__ == "__main__":
    asyncio.run(main_loop())
