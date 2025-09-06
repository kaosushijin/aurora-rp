import asyncio
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
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
SYSTEM_PROMPT_TOKENS = 5000  # Token budget for all system prompts combined

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

# ------------------ Token Estimation ------------------ #
estimate_tokens = lambda text: max(1, len(text) // 4)

# ------------------ Prompt Condensation System ------------------ #
async def condense_prompt(content: str, prompt_type: str) -> str:
    """
    Condense a single prompt file while preserving essential functionality.
    Uses the same LLM approach as memory condensation but optimized for prompts.
    """
    if DEBUG:
        print(Fore.YELLOW + f"[Debug] Condensing {prompt_type} prompt ({len(content)} chars -> target reduction)")
    
    # Design prompt-specific condensation instructions
    condensation_prompts = {
        "critrules": (
            "You are optimizing a Game Master system prompt for an RPG. "
            "Condense the following prompt while preserving all essential game master rules, "
            "narrative generation guidelines, and core functionality. Maintain the same purpose "
            "and effectiveness while reducing token count. Keep all critical instructions intact:\n\n"
            f"{content}\n\n"
            "Provide only the condensed prompt text that maintains full GM functionality."
        ),
        "companion": (
            "You are optimizing a character definition prompt for an RPG companion. "
            "Condense the following prompt while preserving Aurora's complete personality, "
            "appearance, abilities, relationship dynamics, and behavioral patterns. "
            "Maintain all essential character traits while reducing token count:\n\n"
            f"{content}\n\n"
            "Provide only the condensed prompt text that fully preserves Aurora's character."
        ),
        "lowrules": (
            "You are optimizing a narrative generation prompt for an RPG system. "
            "Condense the following prompt while preserving all narrative guidelines, "
            "storytelling rules, and generation principles. Maintain effectiveness "
            "in guiding story creation while reducing token count:\n\n"
            f"{content}\n\n"
            "Provide only the condensed prompt text that maintains narrative quality."
        )
    }
    
    prompt_instruction = condensation_prompts.get(prompt_type, condensation_prompts["critrules"])
    
    try:
        condensed = await call_mcp([{"role": "system", "content": prompt_instruction}])
        
        if DEBUG:
            original_tokens = estimate_tokens(content)
            condensed_tokens = estimate_tokens(condensed)
            reduction = ((original_tokens - condensed_tokens) / original_tokens) * 100
            print(Fore.YELLOW + f"[Debug] {prompt_type} condensation: {original_tokens} -> {condensed_tokens} tokens ({reduction:.1f}% reduction)")
        
        return condensed.strip()
    
    except Exception as e:
        if DEBUG:
            print(Fore.RED + f"[Debug] Condensation failed for {prompt_type}: {e}")
        print(Fore.YELLOW + f"[Warning] Prompt condensation failed for {prompt_type}, using original")
        return content

async def load_and_optimize_prompts() -> Tuple[str, str, str]:
    """
    Load all prompt files and apply condensation if they exceed the token budget.
    Returns tuple of (top_prompt, mid_prompt, low_prompt).
    """
    # Load all prompt files
    try:
        prompts = {
            'critrules': load_prompt(PROMPT_TOP_FILE),
            'companion': load_prompt(PROMPT_MID_FILE), 
            'lowrules': load_prompt(PROMPT_LOW_FILE)
        }
    except FileNotFoundError as e:
        print(Fore.RED + f"[Error] {e}")
        sys.exit(1)
    
    # Calculate combined token count
    total_tokens = sum(estimate_tokens(content) for content in prompts.values())
    
    if DEBUG:
        for prompt_type, content in prompts.items():
            tokens = estimate_tokens(content)
            print(Fore.YELLOW + f"[Debug] {prompt_type} prompt: {tokens} tokens ({len(content)} chars)")
        print(Fore.YELLOW + f"[Debug] Total prompt tokens: {total_tokens}, Budget: {SYSTEM_PROMPT_TOKENS}")
    
    # Apply condensation if budget exceeded
    if total_tokens > SYSTEM_PROMPT_TOKENS:
        print(Fore.YELLOW + f"[System] Prompt files exceed token budget ({total_tokens} > {SYSTEM_PROMPT_TOKENS})")
        print(Fore.YELLOW + "[System] Applying intelligent condensation...")
        
        # Calculate individual thresholds for condensation
        # Condense any file that's more than 1/3 of the total budget
        individual_threshold = SYSTEM_PROMPT_TOKENS // 3
        
        condensation_tasks = []
        for prompt_type, content in prompts.items():
            if estimate_tokens(content) > individual_threshold:
                condensation_tasks.append((prompt_type, content))
        
        # Apply condensation to oversized prompts
        for prompt_type, content in condensation_tasks:
            prompts[prompt_type] = await condense_prompt(content, prompt_type)
        
        # Verify final token count
        final_tokens = sum(estimate_tokens(content) for content in prompts.values())
        
        if DEBUG:
            print(Fore.YELLOW + f"[Debug] Post-condensation total: {final_tokens} tokens")
        
        if final_tokens <= SYSTEM_PROMPT_TOKENS:
            print(Fore.GREEN + f"[System] Condensation successful: {final_tokens} tokens (within budget)")
        else:
            print(Fore.YELLOW + f"[System] Condensation complete: {final_tokens} tokens (reduced from {total_tokens})")
    
    else:
        if DEBUG:
            print(Fore.GREEN + "[Debug] Prompt files within token budget, no condensation needed")
    
    return prompts['critrules'], prompts['companion'], prompts['lowrules']

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
    print_wrapped("Aurora RPG Client (DV1 - Intelligent Prompt Management)", color=Fore.CYAN)
    print_wrapped("Type your action. Press ENTER twice to submit.", color=Fore.CYAN)
    print_wrapped("Terminate using: /quit\n", color=Fore.CYAN)

    # Load and optimize prompts at startup
    print_wrapped("Initializing system prompts...", color=Fore.CYAN)
    SYSTEM_PROMPT_TOP, SYSTEM_PROMPT_MID, SYSTEM_PROMPT_LOW = await load_and_optimize_prompts()
    print_wrapped("System ready!\n", color=Fore.GREEN)

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