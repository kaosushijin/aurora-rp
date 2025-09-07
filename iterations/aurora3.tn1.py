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

# ------------------ Tension System Core Functions ------------------ #

def get_tension_state(memories) -> Dict[str, Any] | None:
    """Retrieve current tension state from memory."""
    for memory in reversed(memories):
        if memory.get("role") == "tension_state":
            return memory["content"]
    return None

def update_tension_state(memories, new_state) -> None:
    """Update or create tension state in memory."""
    for i, memory in enumerate(memories):
        if memory.get("role") == "tension_state":
            memories[i]["content"] = new_state
            memories[i]["timestamp"] = now_iso()
            save_memory(memories)
            return
    
    # Create new tension state entry
    add_memory(memories, "tension_state", new_state)

def create_initial_tension_state() -> Dict[str, Any]:
    """Create initial tension state structure."""
    return {
        "narrative_pressure": 0.05,
        "pressure_source": "antagonist",
        "manifestation_type": "exploration",
        "escalation_count": 0,
        "base_pressure_floor": 0.0,
        "last_analysis_count": 0,
        "antagonist": None  # Generated during first analysis
    }

def should_analyze_tension(memories) -> bool:
    """Check if tension analysis should be triggered."""
    total_messages = count_total_messages(memories)
    current_state = get_tension_state(memories)
    
    if current_state is None:
        return total_messages >= 15  # First analysis after grace period
    
    last_analysis = current_state.get("last_analysis_count", 0)
    return total_messages - last_analysis >= 15

def count_total_messages(memories) -> int:
    """Count all user and assistant messages."""
    return sum(1 for mem in memories 
               if mem.get("role") in ["user", "assistant"])

def prepare_tension_analysis_context(memories, current_state, max_tokens=8000):
    """
    Prepare context for tension analysis within token limits.
    Priority: Recent messages > older messages > system context
    """
    base_prompt_tokens = estimate_tokens("Tension analysis prompt base")  # Rough estimate
    state_tokens = estimate_tokens(json.dumps(current_state)) if current_state else 0
    available_tokens = max_tokens - base_prompt_tokens - state_tokens - 500  # safety buffer
    
    # Start with most recent messages and work backwards
    context_messages = []
    token_count = 0
    
    for memory in reversed(memories):
        if memory["role"] in ["user", "assistant"]:
            msg_tokens = estimate_tokens(memory["content"])
            if token_count + msg_tokens <= available_tokens:
                context_messages.insert(0, memory)
                token_count += msg_tokens
            else:
                break
    
    return context_messages, token_count

# Resource loss analysis system
RESOURCE_LOSS_TRIGGERS = {
    "player_victory_in_confrontation": {
        "resources": ["allies", "reputation"],
        "pressure_reduction": 0.1,
        "commitment_escalation": True
    },
    "antagonist_retreat_from_engagement": {
        "resources": ["reputation", "strategic_position"], 
        "pressure_reduction": 0.05,
        "commitment_escalation": True
    },
    "player_thwarts_antagonist_scheme": {
        "resources": ["influence", "territory"],
        "pressure_reduction": 0.08,
        "commitment_escalation": False
    },
    "antagonist_public_embarrassment": {
        "resources": ["reputation", "allies"],
        "pressure_reduction": 0.12,
        "commitment_escalation": True
    },
    "player_discovers_antagonist_weakness": {
        "resources": ["strategic_position"],
        "pressure_reduction": 0.03,
        "commitment_escalation": False
    }
}

async def analyze_resource_loss(conversation_since_last, current_antagonist):
    """Use LLM to identify resource loss triggers."""
    antagonist_info = "No antagonist established yet" if not current_antagonist else f"{current_antagonist['name']} - {current_antagonist['motivation']}"
    resources_lost = current_antagonist.get('resources_lost', []) if current_antagonist else []
    
    prompt = f"""
Analyze this conversation segment for events that would cause the antagonist to lose resources or standing.

Antagonist: {antagonist_info}
Current Resources Lost: {resources_lost}

Recent Conversation:
{conversation_since_last}

Did any of these occur?
- Player victory in confrontation with antagonist
- Antagonist forced to retreat from engagement  
- Player thwarts antagonist's scheme or plan
- Antagonist publicly embarrassed or humiliated
- Player discovers antagonist weakness or secret

Return ONLY valid JSON:
{{"events": ["list_of_events_that_occurred"], "details": "brief description"}}
"""
    
    try:
        response = await call_mcp([{"role": "system", "content": prompt}])
        return json.loads(response)
    except:
        return {"events": [], "details": "analysis failed"}

def calculate_pressure_floor_ratchet(current_state, events_occurred):
    """Determine if base pressure floor should increase."""
    new_floor = current_state["base_pressure_floor"]
    
    # Ratchet triggers (cumulative)
    if "antagonist_retreat_from_engagement" in events_occurred:
        new_floor += 0.02  # Each retreat makes future retreats harder
    
    if current_state["escalation_count"] >= 3:
        new_floor += 0.01  # Prolonged conflict increases minimum tension
    
    antagonist = current_state.get("antagonist", {})
    if antagonist and len(antagonist.get("resources_lost", [])) >= 3:
        new_floor += 0.03  # Desperate antagonist maintains higher base tension
    
    # Logarithmic cap - pressure floor can't exceed 0.3
    new_floor = min(new_floor, 0.3)
    
    return new_floor

def should_increment_escalation_count(events_occurred, pressure_change):
    """Determine if this cycle represents a new escalation."""
    escalation_events = [
        "player_victory_in_confrontation",
        "antagonist_retreat_from_engagement", 
        "antagonist_public_embarrassment"
    ]
    
    return (any(event in events_occurred for event in escalation_events) or 
            pressure_change > 0.1)

def calculate_commitment_level(antagonist_data, pressure_level, escalation_count):
    """Determine antagonist commitment level based on multiple factors."""
    if not antagonist_data:
        return "testing"
        
    resources_lost_count = len(antagonist_data.get("resources_lost", []))
    
    # Commitment thresholds (multiple criteria can trigger same level)
    if (resources_lost_count >= 4 or 
        pressure_level >= 0.6 or 
        escalation_count >= 5):
        return "cornered"  # Must commit, cannot retreat
    
    elif (resources_lost_count >= 2 or 
          pressure_level >= 0.3 or 
          escalation_count >= 3):
        return "desperate"  # High cost to retreat
    
    elif (resources_lost_count >= 1 or 
          pressure_level >= 0.1 or 
          escalation_count >= 1):
        return "engaged"  # Moderate cost to retreat
    
    else:
        return "testing"  # Low cost retreats available

COMMITMENT_RETREAT_COSTS = {
    "testing": 0.0,      # No additional cost for retreating
    "engaged": 0.05,     # Small pressure floor increase if retreating
    "desperate": 0.1,    # Significant pressure floor increase 
    "cornered": None     # Cannot retreat - must resolve conflict
}

async def generate_antagonist(memories):
    """Generate initial antagonist based on conversation context."""
    context_messages = [mem for mem in memories[-20:] 
                       if mem.get("role") in ["user", "assistant"]]
    
    context_text = "\n".join([f"{mem['role']}: {mem['content']}" 
                             for mem in context_messages])
    
    prompt = f"""
Based on this RPG conversation, create an appropriate antagonist for this story.

Conversation Context:
{context_text}

Generate an antagonist that:
1. Fits naturally with the established setting and tone
2. Has logical motivation related to the story elements mentioned
3. Represents a meaningful challenge for the player character
4. Can operate behind the scenes building tension

Return ONLY valid JSON:
{{
  "name": "antagonist name",
  "motivation": "clear driving goal or need", 
  "resources_lost": [],
  "commitment_level": "testing"
}}
"""
    
    try:
        response = await call_mcp([{"role": "system", "content": prompt}])
        return json.loads(response)
    except:
        # Fallback antagonist
        return {
            "name": "The Shadow Figure",
            "motivation": "seeks to disrupt the player's journey",
            "resources_lost": [],
            "commitment_level": "testing"
        }

# Validation and sanitization
TENSION_STATE_SCHEMA = {
    "narrative_pressure": {"type": "float", "min": 0.0, "max": 1.0},
    "pressure_source": {"type": "string", "enum": ["antagonist", "environment", "social", "discovery"]},
    "manifestation_type": {"type": "string", "enum": ["exploration", "tension", "conflict", "resolution"]},
    "escalation_count": {"type": "int", "min": 0},
    "base_pressure_floor": {"type": "float", "min": 0.0, "max": 0.3},
    "antagonist": {
        "name": {"type": "string"},
        "motivation": {"type": "string"}, 
        "resources_lost": {"type": "list", "items": "string"},
        "commitment_level": {"type": "string", "enum": ["testing", "engaged", "desperate", "cornered"]}
    }
}

def validate_tension_state(state_data):
    """Validate tension state against schema with automatic correction."""
    # Clamp numeric values to valid ranges
    state_data["narrative_pressure"] = max(0.0, min(1.0, 
                                          state_data.get("narrative_pressure", 0.1)))
    state_data["base_pressure_floor"] = max(0.0, min(0.3, 
                                           state_data.get("base_pressure_floor", 0.0)))
    state_data["escalation_count"] = max(0, state_data.get("escalation_count", 0))
    
    # Validate enums with defaults
    if state_data.get("pressure_source") not in ["antagonist", "environment", "social", "discovery"]:
        state_data["pressure_source"] = "antagonist"
    
    if state_data.get("manifestation_type") not in ["exploration", "tension", "conflict", "resolution"]:
        state_data["manifestation_type"] = "exploration"
    
    # Validate antagonist data
    antagonist = state_data.get("antagonist", {})
    if not isinstance(antagonist.get("name"), str):
        antagonist["name"] = "Unknown Antagonist"
    if not isinstance(antagonist.get("motivation"), str):
        antagonist["motivation"] = "seeks to oppose the player"
    if not isinstance(antagonist.get("resources_lost"), list):
        antagonist["resources_lost"] = []
    if antagonist.get("commitment_level") not in ["testing", "engaged", "desperate", "cornered"]:
        antagonist["commitment_level"] = "testing"
    
    state_data["antagonist"] = antagonist
    return state_data

async def analyze_tension_comprehensive(memories, current_state):
    """Complete tension analysis with all systems integrated."""
    
    # 1. Token limit validation
    context_messages, context_tokens = prepare_tension_analysis_context(
        memories, current_state, max_tokens=8000
    )
    
    # 2. Resource loss analysis
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" 
                                  for m in context_messages[-10:]])
    events_occurred = await analyze_resource_loss(conversation_text, 
                                                 current_state.get("antagonist"))
    
    # 3. Calculate pressure floor ratcheting
    new_pressure_floor = calculate_pressure_floor_ratchet(current_state, 
                                                         events_occurred["events"])
    
    # 4. Main tension analysis
    tension_prompt = f"""
You are analyzing narrative tension in an ongoing RPG story. Based on the conversation 
and current tension state, provide updated tension metrics.

Current Tension State: {json.dumps(current_state, indent=2)}
Recent Events Detected: {events_occurred}
Calculated Pressure Floor: {new_pressure_floor}

Recent Conversation:
{conversation_text}

Analyze:
1. How has narrative pressure changed? (0.0-1.0 scale)
2. What is the pressure source? (antagonist/environment/social/discovery)
3. How is tension manifesting? (exploration/tension/conflict/resolution)
4. What is the player's behavioral pattern? (aggressive/cautious/avoidant)
5. How should the antagonist respond given their commitment level?
6. Should any antagonist resources be lost due to recent events?

Return ONLY valid JSON in this exact format:
{{
  "narrative_pressure": 0.0,
  "pressure_source": "antagonist",
  "manifestation_type": "exploration",
  "escalation_count": 0,
  "base_pressure_floor": {new_pressure_floor},
  "antagonist": {{
    "name": "string",
    "motivation": "string", 
    "resources_lost": ["array", "of", "strings"],
    "commitment_level": "testing"
  }}
}}
"""
    
    # Attempt analysis with retry logic
    for attempt in range(3):
        try:
            response = await call_mcp([{"role": "system", "content": tension_prompt}])
            new_state = json.loads(response)
            
            # Validate and sanitize response
            new_state = validate_tension_state(new_state)
            
            # Update message count
            new_state["last_analysis_count"] = count_total_messages(memories)
            
            return new_state
            
        except Exception as e:
            if DEBUG:
                print(Fore.YELLOW + f"[Debug] Tension analysis attempt {attempt + 1} failed: {e}")
            if attempt == 2:  # Final attempt failed
                if DEBUG:
                    print(Fore.YELLOW + "[Debug] All tension analysis attempts failed, using previous state")
                return current_state
    
    return current_state

# Error handling and recovery
class TensionSystemError(Exception):
    """Custom exception for tension system failures."""
    pass

def safe_tension_analysis(func):
    """Decorator for tension analysis functions with error handling."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if DEBUG:
                print(Fore.RED + f"[Debug] Tension system error: {e}")
            # Return safe default state
            return create_initial_tension_state()
    return wrapper

# Apply to critical functions
analyze_tension_comprehensive = safe_tension_analysis(analyze_tension_comprehensive)
generate_antagonist = safe_tension_analysis(generate_antagonist)

def get_pressure_name(pressure_level):
    """Convert pressure level to named range."""
    if pressure_level < 0.1:
        return "low"
    elif pressure_level < 0.3:
        return "building" 
    elif pressure_level < 0.6:
        return "critical"
    else:
        return "explosive"

def generate_tension_context_prompt(tension_state):
    """Generate system prompt context from tension state."""
    antagonist = tension_state["antagonist"]
    pressure_name = get_pressure_name(tension_state["narrative_pressure"])
    
    return f"""
**TENSION CONTEXT**: Current narrative pressure is {pressure_name} ({tension_state['narrative_pressure']:.2f}). 
The antagonist {antagonist['name']} ({antagonist['motivation']}) is at {antagonist['commitment_level']} 
commitment level. They have lost: {', '.join(antagonist['resources_lost']) if antagonist['resources_lost'] else 'nothing yet'}. 
Weave appropriate tension, threats, clues, and confrontation opportunities into your narrative 
based on this pressure level and antagonist state.
"""

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
            "Condense the following prompt while preserving the companion's complete personality, "
            "appearance, abilities, relationship dynamics, and behavioral patterns. "
            "Maintain all essential character traits while reducing token count:\n\n"
            f"{content}\n\n"
            "Provide only the condensed prompt text that fully preserves the companion character."
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
    print_wrapped("Aurora RPG Client (TN1 - Unified Tension System)", color=Fore.CYAN)
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

        # NEW: Tension Analysis Check
        if should_analyze_tension(memories):
            if DEBUG:
                print(Fore.YELLOW + "[Debug] Triggering tension analysis...")
            
            current_state = get_tension_state(memories)
            
            if current_state is None:
                # First analysis - create initial state and generate antagonist
                if DEBUG:
                    print(Fore.YELLOW + "[Debug] First tension analysis - generating antagonist...")
                
                current_state = create_initial_tension_state()
                antagonist = await generate_antagonist(memories)
                current_state["antagonist"] = antagonist
                
                if DEBUG:
                    print(Fore.YELLOW + f"[Debug] Generated antagonist: {antagonist['name']} - {antagonist['motivation']}")
            
            # Perform comprehensive tension analysis
            new_state = await analyze_tension_comprehensive(memories, current_state)
            update_tension_state(memories, new_state)
            
            # Debug output
            if DEBUG:
                pressure_level = new_state["narrative_pressure"]
                pressure_name = get_pressure_name(pressure_level)
                print(Fore.YELLOW + f"[Debug] Tension Analysis Results:")
                print(Fore.YELLOW + f"  Pressure: {pressure_level:.3f} ({pressure_name})")
                print(Fore.YELLOW + f"  Source: {new_state['pressure_source']}")
                print(Fore.YELLOW + f"  Manifestation: {new_state['manifestation_type']}")
                print(Fore.YELLOW + f"  Escalation Count: {new_state['escalation_count']}")
                print(Fore.YELLOW + f"  Pressure Floor: {new_state['base_pressure_floor']:.3f}")
                print(Fore.YELLOW + f"  Antagonist: {new_state['antagonist']['name']}")
                print(Fore.YELLOW + f"  Commitment: {new_state['antagonist']['commitment_level']}")
                print(Fore.YELLOW + f"  Resources Lost: {new_state['antagonist']['resources_lost']}")

        # Existing memory condensation logic
        total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
        while total_memory_tokens > MAX_MEMORY_TOKENS:
            memories = await condense_memory(memories, fraction=0.2)
            total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)

        # Enhanced prompt assembly with tension context
        system_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOP},
            {"role": "system", "content": SYSTEM_PROMPT_MID},
            {"role": "system", "content": SYSTEM_PROMPT_LOW}
        ]
        
        # Add tension context to system prompts if tension state exists
        tension_state = get_tension_state(memories)
        if tension_state and tension_state["antagonist"]:
            tension_context = generate_tension_context_prompt(tension_state)
            system_messages.append({"role": "system", "content": tension_context})
        
        memory_messages = [{"role": mem["role"], "content": mem["content"]} 
                          for mem in memories if mem.get("role") != "tension_state"]
        messages = system_messages + memory_messages + [{"role": "user", "content": raw_input_text}]

        response = await call_mcp(messages)
        add_memory(memories, "assistant", response)
        print_wrapped(response, color=Fore.GREEN, indent=2)

if __name__ == "__main__":
    asyncio.run(main_loop())