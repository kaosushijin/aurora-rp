import asyncio
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any
import httpx
import sys
from colorama import init, Fore, Style
import textwrap
import shutil

# ------------------ Initialization ------------------ #
init(autoreset=True)
DEBUG = "--debug" in sys.argv

# ------------------ Configuration ------------------ #
MCP_URL = "http://127.0.0.1:3456/chat"
MODEL = "qwen2.5:14b-instruct-q4_k_m"
SAVE_FILE = Path("memory.json")
TIMEOUT = 300.0  # seconds

CONTEXT_WINDOW = 32000
SYSTEM_PROMPT_TOKENS = 5000

REMAINING_TOKENS = CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS
MEMORY_FRACTION = 0.7
MAX_MEMORY_TOKENS = int(REMAINING_TOKENS * MEMORY_FRACTION)
MAX_USER_INPUT_TOKENS = int(REMAINING_TOKENS * (1 - MEMORY_FRACTION))

# ------------------ Prompt Files ------------------ #
PROMPT_TOP_FILE = Path("critrules.prompt")
PROMPT_MID_FILE = Path("companion.prompt")
PROMPT_LOW_FILE = Path("lowrules.prompt")

def load_prompt(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()

# Load prompts once at startup
SYSTEM_PROMPT_TOP = load_prompt(PROMPT_TOP_FILE)
SYSTEM_PROMPT_MID = load_prompt(PROMPT_MID_FILE)
SYSTEM_PROMPT_LOW = load_prompt(PROMPT_LOW_FILE)

if DEBUG:
    print(Fore.YELLOW + f"[Debug] Loaded prompts: "
          f"TOP({len(SYSTEM_PROMPT_TOP)} chars), "
          f"MID({len(SYSTEM_PROMPT_MID)} chars), "
          f"LOW({len(SYSTEM_PROMPT_LOW)} chars)")

# ------------------ Utility: Word Wrap ------------------ #
def get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def print_wrapped(text: str, color=Fore.GREEN, indent: int = 0):
    width = get_terminal_width() - indent
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=' ' * indent)
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        lines = para.splitlines()
        wrapped_para = "\n".join(wrapper.fill(line) for line in lines)
        print(color + wrapped_para)
        if i < len(paragraphs) - 1:
            print("")

# ------------------ Memory Handling ------------------ #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_memory() -> List[Dict[str, Any]]:
    if SAVE_FILE.exists():
        return json.loads(SAVE_FILE.read_text())
    else:
        save_memory([])
        return []

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

estimate_tokens = lambda text: max(1, len(text) // 4)

async def condense_memory(memories, fraction=0.2):
    if not memories:
        return memories
    chunk, memories = memories[:int(len(memories)*fraction)], memories[int(len(memories)*fraction):]
    text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in chunk)
    prompt = f"Condense memories while preserving story:\n{text}"
    condensed = await call_mcp([{"role": "system", "content": prompt}])
    return [{"id": str(uuid4()), "role": "system", "content": condensed, "timestamp": now_iso()}] + memories

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
    print_wrapped("Aurora RPG Client (POC-mem)", color=Fore.CYAN)
    print_wrapped("Type your action. Press ENTER twice to submit.", color=Fore.CYAN)
    print_wrapped("Terminate using: /quit\n", color=Fore.CYAN)

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
            print_wrapped("Goodbye!", color=Fore.YELLOW)
            break

        print_wrapped(f"You: {raw_input_text}", color=Fore.BLUE, indent=2)
        add_memory(memories, "user", raw_input_text)

        total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
        while total_memory_tokens > MAX_MEMORY_TOKENS:
            memories = await condense_memory(memories, fraction=0.2)
            total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)

        system_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOP},
            {"role": "system", "content": SYSTEM_PROMPT_MID},
            {"role": "system", "content": SYSTEM_PROMPT_LOW}
        ]
        memory_messages = [{"role": mem["role"], "content": mem["content"]} for mem in memories]
        messages = system_messages + memory_messages + [{"role": "user", "content": raw_input_text}]

        response = await call_mcp(messages)
        add_memory(memories, "assistant", response)
        print_wrapped(response, color=Fore.GREEN, indent=2)

if __name__ == "__main__":
    asyncio.run(main_loop())
