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

# ------------------ OPTIMIZED Token Budget ------------------ #
CONTEXT_WINDOW = 32000
SYSTEM_PROMPT_TOKENS = 5000      # Game Master prompts
TENSION_SYSTEM_TOKENS = 6000     # Tension analysis budget
MAX_USER_INPUT_TOKENS = 2000     # 8000 characters - handles even very long inputs

# Calculate memory allocation from remaining tokens
REMAINING_FOR_MEMORY = (CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS -
                       TENSION_SYSTEM_TOKENS - MAX_USER_INPUT_TOKENS)
MAX_MEMORY_TOKENS = REMAINING_FOR_MEMORY  # 19000 tokens for memory!

# ------------------ NEW: Semantic Memory Categories ------------------ #
CONDENSATION_STRATEGIES = {
    "story_critical": {
        "instruction": (
            "Preserve: major plot developments, character deaths, world-changing events, "
            "player's key decisions, villain revelations, quest completions. "
            "Compress: dialogue details, minor interactions, travel descriptions."
        ),
        "preservation_ratio": 0.8,  # Keep 80% of important content
        "trigger_keywords": ["death", "reveal", "discover", "decide", "choose", "complete", "achieve"]
    },
    "character_focused": {
        "instruction": (
            "Preserve: relationship changes, character motivations, personality reveals, "
            "trust/betrayal moments, Aurora's development, NPC personality traits. "
            "Compress: casual conversations, repeated interactions."
        ),
        "preservation_ratio": 0.7,
        "trigger_keywords": ["trust", "betray", "love", "hate", "friend", "enemy", "personality", "relationship"]
    },
    "world_building": {
        "instruction": (
            "Preserve: new locations, lore revelations, political changes, economic systems, "
            "magical discoveries, cultural information. "
            "Compress: repeated location visits, standard travel."
        ),
        "preservation_ratio": 0.6,
        "trigger_keywords": ["location", "lore", "magic", "culture", "politics", "kingdom", "city", "history"]
    },
    "standard": {
        "instruction": (
            "Preserve: player actions and their immediate consequences. "
            "Compress: everything else while maintaining story flow and basic continuity."
        ),
        "preservation_ratio": 0.4,
        "trigger_keywords": []
    }
}

# ------------------ Prompt Files ------------------ #
PROMPT_TOP_FILE = Path("critrules.prompt")
PROMPT_MID_FILE = Path("companion.prompt")
PROMPT_LOW_FILE = Path("lowrules.prompt")

def load_prompt(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()

# ------------------ Configuration Validation ------------------ #
def validate_token_allocation():
    """Ensure token allocation doesn't exceed context window"""
    total_allocated = (SYSTEM_PROMPT_TOKENS + TENSION_SYSTEM_TOKENS +
                      MAX_MEMORY_TOKENS + MAX_USER_INPUT_TOKENS)

    if total_allocated > CONTEXT_WINDOW:
        raise ValueError(f"Token allocation ({total_allocated:,}) exceeds context window ({CONTEXT_WINDOW:,})")

    if DEBUG:
        utilization = (total_allocated / CONTEXT_WINDOW) * 100
        print(Fore.GREEN + f"[Debug] Token allocation validated: {utilization:.1f}% utilization")
        print(Fore.YELLOW + f"[Debug] Token Budget Allocation:")
        print(Fore.YELLOW + f"  System prompts: {SYSTEM_PROMPT_TOKENS:,} tokens ({SYSTEM_PROMPT_TOKENS/CONTEXT_WINDOW*100:.1f}%)")
        print(Fore.YELLOW + f"  Tension system: {TENSION_SYSTEM_TOKENS:,} tokens ({TENSION_SYSTEM_TOKENS/CONTEXT_WINDOW*100:.1f}%)")
        print(Fore.YELLOW + f"  Memory: {MAX_MEMORY_TOKENS:,} tokens ({MAX_MEMORY_TOKENS/CONTEXT_WINDOW*100:.1f}%)")
        print(Fore.YELLOW + f"  User input: {MAX_USER_INPUT_TOKENS:,} tokens ({MAX_USER_INPUT_TOKENS/CONTEXT_WINDOW*100:.1f}%)")
        print(Fore.YELLOW + f"  Total: {total_allocated:,} tokens")
        print(Fore.YELLOW + f"  Safety margin: {CONTEXT_WINDOW - total_allocated:,} tokens")

    return True

# ------------------ Utility Functions ------------------ #
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

# ------------------ Input Validation ------------------ #
def validate_user_input_length(user_input: str) -> tuple[bool, str]:
    """
    Validate user input length and provide helpful feedback if too long.
    """
    input_tokens = estimate_tokens(user_input)

    if input_tokens <= MAX_USER_INPUT_TOKENS:
        return True, ""

    char_count = len(user_input)
    max_chars = MAX_USER_INPUT_TOKENS * 4

    warning = (f"Input too long ({input_tokens:,} tokens, {char_count:,} chars). "
              f"Maximum: {MAX_USER_INPUT_TOKENS:,} tokens ({max_chars:,} chars). "
              f"Please shorten your input or split it into multiple messages.")

    return False, warning

# ------------------ Enhanced Memory Management System ------------------ #

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

# ------------------ NEW: LLM-Driven Memory Analysis ------------------ #

async def analyze_memory_chunk_for_condensation(chunk_memories):
    """
    Let the LLM categorize memories and choose condensation strategy based on semantic content.
    """
    chunk_text = format_memories_for_analysis(chunk_memories)
    
    # Build trigger keyword context for the LLM
    keyword_context = "\n".join([
        f"- {category}: {', '.join(data['trigger_keywords'])}" 
        for category, data in CONDENSATION_STRATEGIES.items() 
        if data['trigger_keywords']
    ])
    
    analysis_prompt = f"""
Analyze this conversation chunk and categorize its narrative importance:

{chunk_text}

Categories and their typical content:
- story_critical: Major plot developments, character deaths, world-changing events, key decisions
- character_focused: Relationship development, personality reveals, trust/betrayal moments
- world_building: New locations, lore discoveries, political/economic developments
- standard: General interactions and conversations

Common keywords by category:
{keyword_context}

What type of content is this primarily? Consider:
1. Does it contain major plot developments or life-changing events?
2. Does it focus on character relationships or personality development?
3. Does it introduce new world information or lore?
4. Is it general conversation or standard interaction?

Return ONLY valid JSON:
{{
  "category": "story_critical|character_focused|world_building|standard",
  "confidence": 0.0-1.0,
  "key_elements": ["list", "of", "essential", "things", "to", "preserve"],
  "reasoning": "brief explanation of categorization"
}}
"""
    
    try:
        response = await call_mcp([{"role": "system", "content": analysis_prompt}])
        result = json.loads(response)
        
        # Validate response structure
        if not all(key in result for key in ["category", "confidence", "key_elements"]):
            raise ValueError("Invalid analysis response structure")
        
        if result["category"] not in CONDENSATION_STRATEGIES:
            result["category"] = "standard"  # Fallback to standard
            
        if DEBUG:
            print(Fore.CYAN + f"[Debug] Memory analysis: {result['category']} "
                            f"(confidence: {result.get('confidence', 0):.2f})")
            print(Fore.CYAN + f"[Debug] Key elements: {result['key_elements']}")
        
        return result
        
    except Exception as e:
        if DEBUG:
            print(Fore.YELLOW + f"[Debug] Memory analysis failed: {e}, using fallback")
        # Fallback analysis
        return {
            "category": "standard",
            "confidence": 0.5,
            "key_elements": ["player actions", "general story flow"],
            "reasoning": "analysis failed, using standard condensation"
        }

def format_memories_for_analysis(memories):
    """Format memory chunk for LLM analysis."""
    formatted = []
    for mem in memories:
        if mem.get("role") in ["user", "assistant"]:
            role_name = "Player" if mem["role"] == "user" else "GM"
            formatted.append(f"{role_name}: {mem['content']}")
    return "\n\n".join(formatted)

def format_chunk_for_condensation(memories):
    """Format memory chunk for condensation prompt."""
    return "\n\n".join([f"{mem['role'].capitalize()}: {mem['content']}" 
                       for mem in memories if mem.get("role") in ["user", "assistant"]])

# ------------------ NEW: Semantic-Aware Memory Condensation ------------------ #

async def condense_with_semantic_awareness(memories, fraction=0.2):
    """
    Condense memories using LLM-determined importance and category-specific strategy.
    """
    if not memories:
        return memories

    chunk_size = max(1, int(len(memories) * fraction))
    chunk = memories[:chunk_size]
    remaining = memories[chunk_size:]
    
    # Let LLM analyze what type of content this is
    analysis = await analyze_memory_chunk_for_condensation(chunk)
    strategy = CONDENSATION_STRATEGIES[analysis["category"]]
    
    # Use category-specific condensation instruction
    condensation_prompt = f"""
You are condensing RPG story memories while preserving narrative continuity.

CONDENSATION STRATEGY: {analysis["category"]}
{strategy["instruction"]}

Target preservation: {strategy["preservation_ratio"]*100}% of important content
Key elements identified: {analysis["key_elements"]}

Original conversation:
{format_chunk_for_condensation(chunk)}

Condense this conversation while:
1. Preserving {strategy["preservation_ratio"]*100}% of the important content according to the strategy above
2. Maintaining story flow and character consistency
3. Keeping essential plot points and character development
4. Removing unnecessary dialogue details and repetitive descriptions

Provide only the condensed narrative summary.
"""
    
    try:
        condensed = await call_mcp([{"role": "system", "content": condensation_prompt}])
        
        condensed_memory = {
            "id": str(uuid4()),
            "role": "system",
            "content": condensed,
            "timestamp": now_iso(),
            "condensation_type": analysis["category"],  # Track what strategy was used
            "original_chunk_size": len(chunk),
            "preservation_ratio": strategy["preservation_ratio"]
        }
        
        if DEBUG:
            original_tokens = sum(estimate_tokens(mem["content"]) for mem in chunk)
            condensed_tokens = estimate_tokens(condensed)
            actual_ratio = condensed_tokens / original_tokens if original_tokens > 0 else 0
            print(Fore.GREEN + f"[Debug] {analysis['category']} condensation: "
                             f"{original_tokens} -> {condensed_tokens} tokens "
                             f"(actual ratio: {actual_ratio:.2f})")
        
        return [condensed_memory] + remaining
        
    except Exception as e:
        if DEBUG:
            print(Fore.RED + f"[Debug] Semantic condensation failed: {e}")
        # Fallback to simple condensation
        return await simple_fallback_condensation(chunk, remaining)

async def simple_fallback_condensation(chunk, remaining):
    """Fallback condensation when semantic analysis fails."""
    text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in chunk)
    prompt = f"Condense this RPG conversation while preserving story continuity:\n{text}"
    
    try:
        condensed = await call_mcp([{"role": "system", "content": prompt}])
        condensed_memory = {
            "id": str(uuid4()),
            "role": "system",
            "content": condensed,
            "timestamp": now_iso(),
            "condensation_type": "fallback"
        }
        return [condensed_memory] + remaining
    except:
        # Ultimate fallback - just remove oldest memories
        if DEBUG:
            print(Fore.RED + "[Debug] All condensation methods failed, removing oldest memories")
        return remaining

# ------------------ NEW: Progressive Memory Management ------------------ #

async def intelligent_memory_management(memories):
    """
    Multiple passes with different strategies based on memory pressure and content age.
    """
    if not memories:
        return memories
    
    original_count = len(memories)
    
    # First pass: Recent memories (gentle condensation, preserve more)
    if len(memories) > 100:
        if DEBUG:
            print(Fore.YELLOW + "[Debug] Memory management: First pass (recent memories)")
        memories = await condense_with_semantic_awareness(memories, fraction=0.1)
    
    # Second pass: Medium-age memories (moderate condensation)
    if len(memories) > 80:
        if DEBUG:
            print(Fore.YELLOW + "[Debug] Memory management: Second pass (medium-age memories)")
        memories = await condense_with_semantic_awareness(memories, fraction=0.2)
    
    # Third pass: Older memories (more aggressive condensation)
    if len(memories) > 60:
        if DEBUG:
            print(Fore.YELLOW + "[Debug] Memory management: Third pass (older memories)")
        memories = await condense_with_semantic_awareness(memories, fraction=0.3)
    
    # Emergency pass: Very aggressive if still too large
    current_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    if current_tokens > MAX_MEMORY_TOKENS:
        if DEBUG:
            print(Fore.RED + f"[Debug] Emergency condensation: {current_tokens} > {MAX_MEMORY_TOKENS}")
        memories = await condense_with_semantic_awareness(memories, fraction=0.4)
    
    if DEBUG and len(memories) != original_count:
        print(Fore.GREEN + f"[Debug] Memory management complete: {original_count} -> {len(memories)} entries")
    
    return memories

# ------------------ Enhanced Memory Condensation (Backward Compatible) ------------------ #

async def condense_memory(memories, fraction=0.2):
    """
    Enhanced memory condensation that uses semantic awareness when possible,
    falls back to simple condensation if needed. Maintains backward compatibility.
    """
    # Use the new intelligent system when memory pressure is high
    current_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    if current_tokens > MAX_MEMORY_TOKENS * 0.9:  # When approaching limit
        return await intelligent_memory_management(memories)
    else:
        # Use semantic-aware condensation for normal operation
        return await condense_with_semantic_awareness(memories, fraction)

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

def prepare_tension_analysis_context(memories, current_state, max_tokens=None):
    """
    Prepare context for tension analysis within the allocated budget.
    """
    if max_tokens is None:
        max_tokens = TENSION_SYSTEM_TOKENS  # Use allocated budget

    base_prompt_tokens = estimate_tokens("Tension analysis prompt base")
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
    
    if DEBUG:
        print(Fore.YELLOW + f"[Debug] Tension context: {len(context_messages)} messages, {token_count} tokens")

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
    
    # 1. Token limit validation using allocated budget
    context_messages, context_tokens = prepare_tension_analysis_context(
        memories, current_state, max_tokens=TENSION_SYSTEM_TOKENS
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
    print_wrapped("Aurora RPG Client (TN3 - LLM-Driven Semantic Memory)", color=Fore.CYAN)
    print_wrapped("Type your action. Press ENTER twice to submit.", color=Fore.CYAN)
    print_wrapped("Terminate using: /quit\n", color=Fore.CYAN)

    # Validate token allocation at startup
    validate_token_allocation()

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

        # Validate input length before processing
        is_valid, error_msg = validate_user_input_length(raw_input_text)
        if not is_valid:
            print_wrapped(error_msg, color=Fore.RED)
            # Put input back in buffer for user to edit
            buffer = raw_input_text.split('\n')
            continue

        print_wrapped(f"You: {raw_input_text}", color=Fore.BLUE, indent=2)
        add_memory(memories, "user", raw_input_text)

        # Tension Analysis Check
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

        # Enhanced Memory Management - Use new semantic condensation system
        total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
        if total_memory_tokens > MAX_MEMORY_TOKENS:
            if DEBUG:
                print(Fore.YELLOW + f"[Debug] Memory condensation triggered: {total_memory_tokens} > {MAX_MEMORY_TOKENS}")
                print(Fore.CYAN + "[Debug] Using enhanced semantic condensation system...")
            
            # Use the new intelligent memory management system
            memories = await condense_memory(memories, fraction=0.2)
            
            # Verify memory reduction
            new_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
            if DEBUG:
                reduction = total_memory_tokens - new_memory_tokens
                print(Fore.GREEN + f"[Debug] Memory condensation complete: {total_memory_tokens} -> {new_memory_tokens} tokens (saved {reduction})")

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
        
        # Filter out tension_state entries from memory messages for LLM context
        memory_messages = [{"role": mem["role"], "content": mem["content"]} 
                          for mem in memories if mem.get("role") != "tension_state"]
        
        # Assemble final message context
        messages = system_messages + memory_messages + [{"role": "user", "content": raw_input_text}]

        # Debug final message composition if in debug mode
        if DEBUG:
            total_message_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)
            system_tokens = sum(estimate_tokens(msg["content"]) for msg in system_messages)
            memory_tokens = sum(estimate_tokens(msg["content"]) for msg in memory_messages)
            input_tokens = estimate_tokens(raw_input_text)
            
            print(Fore.YELLOW + f"[Debug] Final message composition:")
            print(Fore.YELLOW + f"  System prompts: {system_tokens} tokens")
            print(Fore.YELLOW + f"  Memory messages: {memory_tokens} tokens")
            print(Fore.YELLOW + f"  User input: {input_tokens} tokens")
            print(Fore.YELLOW + f"  Total: {total_message_tokens} tokens")
            
            if total_message_tokens > CONTEXT_WINDOW:
                print(Fore.RED + f"[Debug] WARNING: Total exceeds context window by {total_message_tokens - CONTEXT_WINDOW} tokens!")
            else:
                utilization = (total_message_tokens / CONTEXT_WINDOW) * 100
                print(Fore.GREEN + f"[Debug] Context utilization: {utilization:.1f}%")

        # Generate response using MCP
        response = await call_mcp(messages)
        
        # Add response to memory and display
        add_memory(memories, "assistant", response)
        print_wrapped(response, color=Fore.GREEN, indent=2)

if __name__ == "__main__":
    asyncio.run(main_loop())
