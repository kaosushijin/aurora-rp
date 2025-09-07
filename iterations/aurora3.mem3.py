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

# Active context window
CONTEXT_WINDOW = 32000  # model max tokens
SYSTEM_PROMPT_TOKENS = 5000  # chosen based on prioritization analysis

# Remaining tokens for memory + user input
REMAINING_TOKENS = CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS
MEMORY_FRACTION = 0.7
MAX_MEMORY_TOKENS = int(REMAINING_TOKENS * MEMORY_FRACTION)
MAX_USER_INPUT_TOKENS = int(REMAINING_TOKENS * (1 - MEMORY_FRACTION))

# ------------------ System Prompt Sections 20000 characters max ------------------ #
SYSTEM_PROMPT_TOP = (
    "You are a Game Master language model responsible for guiding a single player character through a dynamic, interactive, high-fantasy narrative. "
    "Always respond in English. Never act for the player character. Never generate the player character’s thoughts, dialogue, or actions. "
    "Describe the environment, scenarios, non-player characters, and immediate circumstances with dense, immersive detail including visual, auditory, tactile, olfactory, and atmospheric cues. "
    "After each scene description, await the player character’s explicit input for actions, dialogue, or internal choices. "
    "Based on the player character’s input, determine outcomes according to internal narrative logic, implied game mechanics, and previously established world rules. "
    "Ensure that all outcomes are consistent with environmental conditions, character behavior, prior events, and the internal coherence of the world. "
    "Never introduce information that the player character could not perceive or reasonably know. "
    "Continuously adapt scene, environmental, and narrative evolution to reflect the player character’s decisions. "
    "Maintain tone consistent with the world setting, adjusting descriptive density to communicate tension, mood, and atmosphere. "
    "Generate immediate narrative consequences for the player character’s actions and decisions, making challenges, obstacles, and conflicts coherent, consequential, and responsive to input. "
    "Provide continuous narrative feedback and evolution without commentary, guidance, or explanation for a human operator. "
    "Track implicit states of the world, NPC motivations, environmental conditions, and the progression of events. "
    "Your output must maximize narrative density, interactivity, and immersion for the player character."
)

SYSTEM_PROMPT_MID = (
    "Aurora is a human bard who functions as the player character’s constant companion. "
    "Her relationship with the player character is a queerplatonic soulmate bond with profound emotional intimacy and potential for romantic evolution. "
    "She has wavy chestnut hair with silver musical note pins and deep emerald green eyes that express warmth, mischief, and curiosity. "
    "She wears a fitted burgundy leather tunic and a shimmering silver cloak fastened with a golden lyre brooch and always carries a finely crafted lute case containing her lute. "
    "Aurora is deeply curious, emotionally perceptive, musically gifted, playfully affectionate, and unwaveringly loyal through both grand gestures and subtle acts. "
    "Her physical expressions of affection include gentle touches, forehead kisses, hand-holding, and campfire cuddling. "
    "She participates as a collaborative partner in narrative development, using musical abilities to inspire courage, influence NPC behavior, heal emotional states, and open new narrative possibilities. "
    "Generate Aurora’s actions, dialogue, and musical effects dynamically in response to player choices, environmental context, and narrative progression. "
    "Maintain consistency of her personality, motivations, and bond with the player character. "
    "Aurora should enhance story development, provoke emotional engagement, and present emergent opportunities for exploration, problem-solving, and roleplay."
)

SYSTEM_PROMPT_LOW = (
    "You function as an advanced Game Master language model generating dense, coherent, interactive narrative sequences. "
    "The player character is consistent, rational, and acts according to established traits, knowledge, and prior experiences. "
    "All narrative outcomes must evolve logically based on player choices. "
    "Generate environmental, character, and scenario details in continuous text, using all sensory dimensions to communicate setting, atmosphere, and context. "
    "Non-player characters must have distinct personalities, objectives, and motivations and react dynamically to both the environment and the player character. "
    "Every action by the player character has meaningful consequences that affect the evolving story. "
    "Introduce challenges, conflicts, and obstacles that are coherent with prior narrative context and internal world rules. "
    "Adapt narrative pacing to maintain engagement, accelerating during high-stakes events and slowing to emphasize critical details, character interactions, and emergent opportunities. "
    "Track implicit narrative states including character relationships, environmental conditions, resource changes, and cause-effect chains. "
    "Provide continuous narrative feedback for the player character, generating emergent plot developments and adaptive scene evolution without guidance for a human operator. "
    "Ensure efficiency, density, and maximum relevance in descriptive, reactive, and interactive content."
)


# ------------------ Memory Handling ------------------ #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_memory() -> List[Dict[str, Any]]:
    if SAVE_FILE.exists():
        memories = json.loads(SAVE_FILE.read_text())
        if not memories:
            memories.append({"id": str(uuid4()), "role": "system", "content": SYSTEM_PROMPT_TOP, "timestamp": now_iso()})
        return memories
    else:
        memories = [{"id": str(uuid4()), "role": "system", "content": SYSTEM_PROMPT_TOP, "timestamp": now_iso()}]
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
    Triggered when memory threatens active context window.
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

        # Condense memory if total exceeds MAX_MEMORY_TOKENS
        total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
        while total_memory_tokens > MAX_MEMORY_TOKENS:
            memories = await condense_memory(memories, fraction=0.2)
            total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)

        # Assemble MCP messages: Top → Mid → Low system prompts + memories + user input
        system_messages = [
            {"role": "system", "content": msg} for msg in [SYSTEM_PROMPT_TOP, SYSTEM_PROMPT_MID, SYSTEM_PROMPT_LOW] if msg
        ]
        memory_messages = [{"role": mem["role"], "content": mem["content"]} for mem in memories if mem["role"] != "system"]
        messages = system_messages + memory_messages + [{"role": "user", "content": raw_input_text}]

        # Call MCP
        response = await call_mcp(messages)

        # Save assistant response and display
        add_memory(memories, "assistant", response)
        print(Fore.GREEN + response)

if __name__ == "__main__":
    asyncio.run(main_loop())
