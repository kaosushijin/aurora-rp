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

# ------------------ NEW: Pure Semantic Memory Categories ------------------ #
MEMORY_CATEGORIES = {
    "story_critical": {
        "description": "Events that fundamentally change the story trajectory",
        "preservation_ratio": 0.8,
        "examples": [
            "Character deaths or major injuries",
            "Quest completion or failure", 
            "Major revelations about the world or characters",
            "Irreversible player decisions with lasting consequences"
        ],
        "condensation_instruction": (
            "This content fundamentally changed the story. Preserve ALL major events, decisions, "
            "and consequences. Compress only dialogue details and scene descriptions while "
            "maintaining complete factual accuracy about what happened and why it matters."
        )
    },
    "character_development": {
        "description": "Moments that reveal or change character relationships and personalities",
        "preservation_ratio": 0.7,
        "examples": [
            "Trust or betrayal between characters",
            "Personality reveals through actions or dialogue",
            "Relationship status changes",
            "Character growth or regression moments"
        ],
        "condensation_instruction": (
            "This content revealed or changed character relationships. Preserve the emotional "
            "core and relationship dynamics. Maintain personality insights and trust/betrayal "
            "moments. Compress setting details and action sequences."
        )
    },
    "world_building": {
        "description": "Information that expands understanding of the setting",
        "preservation_ratio": 0.6,
        "examples": [
            "New locations with unique characteristics",
            "Cultural or historical information",
            "Magic system mechanics or lore",
            "Political or economic systems"
        ],
        "condensation_instruction": (
            "This content expanded world knowledge. Preserve new information about locations, "
            "lore, politics, or culture that could be referenced later. Compress personal "
            "interactions and maintain factual world details."
        )
    },
    "interaction": {
        "description": "Standard gameplay moments without lasting significance",
        "preservation_ratio": 0.4,
        "examples": [
            "Routine travel or shopping",
            "Small talk without character development",
            "Combat without story consequences",
            "Repeated interactions with familiar elements"
        ],
        "condensation_instruction": (
            "This content represents standard gameplay. Preserve player actions and immediate "
            "consequences for continuity. Compress everything else aggressively while "
            "maintaining basic story flow."
        )
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

# ------------------ ENHANCED Antagonist Persistence System ------------------ #

def preserve_antagonist_data(new_state, existing_antagonist):
    """
    Ensure antagonist data is preserved unless explicitly updated.
    Prevents loss of detailed antagonist information due to processing errors.
    """
    if not existing_antagonist:
        return new_state.get("antagonist", {})
    
    new_antagonist = new_state.get("antagonist", {})
    
    # Preserve existing data if new data is incomplete or generic
    preserved_antagonist = existing_antagonist.copy()
    
    # Only update fields that have meaningful new information
    if new_antagonist.get("name") and new_antagonist["name"] != "The Shadow Figure":
        preserved_antagonist["name"] = new_antagonist["name"]
    
    if new_antagonist.get("motivation") and new_antagonist["motivation"] != "seeks to disrupt the player's journey":
        preserved_antagonist["motivation"] = new_antagonist["motivation"]
    
    # Always update these fields as they represent current state
    if "resources_lost" in new_antagonist:
        preserved_antagonist["resources_lost"] = new_antagonist["resources_lost"]
    
    if "commitment_level" in new_antagonist:
        preserved_antagonist["commitment_level"] = new_antagonist["commitment_level"]
    
    return preserved_antagonist

def validate_antagonist_quality(antagonist_data):
    """
    Check if antagonist data is high-quality and specific, not generic fallback.
    """
    if not antagonist_data:
        return False
    
    # Check for generic fallback indicators
    generic_names = ["The Shadow Figure", "Unknown Antagonist", "Unknown Adversary"]
    generic_motivations = [
        "seeks to disrupt the player's journey", 
        "seeks to oppose the player",
        "opposes the player through unknown means",
        "opposes the player's goals through unknown means",
        "unknown motivation"
    ]
    
    name = antagonist_data.get("name", "")
    motivation = antagonist_data.get("motivation", "")
    
    if name in generic_names or motivation in generic_motivations:
        return False
    
    return True

async def safe_antagonist_generation(memories, max_retries=3):
    """
    Generate antagonist with retry logic and quality validation.
    """
    for attempt in range(max_retries):
        try:
            antagonist = await generate_antagonist(memories)
            
            if validate_antagonist_quality(antagonist):
                if DEBUG:
                    print(Fore.GREEN + f"[Debug] Quality antagonist generated: {antagonist['name']} - {antagonist['motivation']}")
                return antagonist
            else:
                if DEBUG:
                    print(Fore.YELLOW + f"[Debug] Generated antagonist quality insufficient, retry {attempt + 1}")
                continue
                
        except Exception as e:
            if DEBUG:
                print(Fore.YELLOW + f"[Debug] Antagonist generation attempt {attempt + 1} failed: {e}")
            continue
    
    # If all attempts fail, return a basic but functional antagonist
    if DEBUG:
        print(Fore.RED + "[Debug] All antagonist generation attempts failed, using minimal fallback")
    
    return {
        "name": "Unknown Adversary",
        "motivation": "opposes the player's goals through unknown means",
        "resources_lost": [],
        "commitment_level": "testing"
    }

# ------------------ NEW: Pure Semantic Memory Analysis ------------------ #

def format_memories_for_analysis(memories):
    """Format memory chunk for semantic analysis."""
    formatted = []
    for mem in memories:
        if mem.get("role") in ["user", "assistant"]:
            role_name = "Player" if mem["role"] == "user" else "GM"
            formatted.append(f"{role_name}: {mem['content']}")
    return "\n\n".join(formatted)

async def analyze_memory_semantic(chunk_memories):
    """
    Pure semantic analysis without keyword triggers.
    Let the LLM determine narrative significance through contextual understanding.
    """
    chunk_text = format_memories_for_analysis(chunk_memories)
    
    # Build category context from our semantic definitions
    category_descriptions = []
    for category, data in MEMORY_CATEGORIES.items():
        examples_text = "; ".join(data["examples"][:2])  # Just first 2 examples for brevity
        category_descriptions.append(f"- {category.upper()}: {data['description']} (e.g., {examples_text})")
    
    category_context = "\n".join(category_descriptions)
    
    analysis_prompt = f"""
Analyze this RPG conversation segment and determine its narrative significance:

{chunk_text}

Categories:
{category_context}

Consider these key questions:
- What would be lost if this content disappeared from memory?
- How important is this for future story continuity?
- Does this fundamentally change the story trajectory?
- Does this reveal or change character relationships or personalities?
- Does this expand understanding of the game world?

Evaluate the CONTENT and CONTEXT, not just individual words. A character mentioning death casually is different from a character actually dying.

Return only one word: STORY_CRITICAL, CHARACTER_DEVELOPMENT, WORLD_BUILDING, or INTERACTION
"""
    
    try:
        response = await call_mcp([{"role": "system", "content": analysis_prompt}])
        category = response.strip().lower().replace('_', '_')
        
        # Validate and normalize response
        valid_categories = ["story_critical", "character_development", "world_building", "interaction"]
        if category not in valid_categories:
            if DEBUG:
                print(Fore.YELLOW + f"[Debug] Invalid category '{category}', defaulting to 'interaction'")
            category = "interaction"  # Safe fallback
            
        if DEBUG:
            print(Fore.CYAN + f"[Debug] Semantic analysis result: {category}")
            
        return category
        
    except Exception as e:
        if DEBUG:
            print(Fore.YELLOW + f"[Debug] Semantic analysis failed: {e}, defaulting to 'interaction'")
        return "interaction"  # Safe fallback on any error

def format_chunk_for_condensation(memories):
    """Format memory chunk for condensation prompt."""
    return "\n\n".join([f"{mem['role'].capitalize()}: {mem['content']}" 
                       for mem in memories if mem.get("role") in ["user", "assistant"]])

# ------------------ NEW: Semantic-Aware Memory Condensation ------------------ #

async def condense_with_semantic_awareness(memories, fraction=0.2):
    """
    Condense memories using pure semantic analysis and category-specific preservation strategies.
    """
    if not memories:
        return memories

    chunk_size = max(1, int(len(memories) * fraction))
    chunk = memories[:chunk_size]
    remaining = memories[chunk_size:]
    
    # Use pure semantic analysis to determine content type
    category = await analyze_memory_semantic(chunk)
    category_config = MEMORY_CATEGORIES[category]
    
    # Use category-specific condensation instruction
    condensation_prompt = f"""
You are condensing RPG story memories while preserving narrative continuity.

CONTENT TYPE: {category}
PRESERVATION STRATEGY: {category_config["condensation_instruction"]}

Target preservation ratio: {category_config["preservation_ratio"]*100}% of important content

Original conversation:
{format_chunk_for_condensation(chunk)}

Apply the preservation strategy above to condense this conversation while:
1. Maintaining story flow and character consistency
2. Preserving the most narratively significant elements
3. Compressing less important details according to the strategy
4. Ensuring the condensed version supports future story continuity

Provide only the condensed narrative summary.
"""
    
    try:
        condensed = await call_mcp([{"role": "system", "content": condensation_prompt}])
        
        condensed_memory = {
            "id": str(uuid4()),
            "role": "system",
            "content": condensed,
            "timestamp": now_iso(),
            "condensation_type": category,  # Track what strategy was used
            "original_chunk_size": len(chunk),
            "preservation_ratio": category_config["preservation_ratio"]
        }
        
        if DEBUG:
            original_tokens = sum(estimate_tokens(mem["content"]) for mem in chunk)
            condensed_tokens = estimate_tokens(condensed)
            actual_ratio = condensed_tokens / original_tokens if original_tokens > 0 else 0
            print(Fore.GREEN + f"[Debug] {category} condensation: "
                             f"{original_tokens} -> {condensed_tokens} tokens "
                             f"(actual ratio: {actual_ratio:.2f}, target: {category_config['preservation_ratio']:.2f})")
        
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

# ------------------ NEW: Intelligent Progressive Memory Management ------------------ #

async def intelligent_memory_management(memories):
    """
    Multiple passes with different strategies based on memory pressure and content age.
    Uses semantic analysis for each pass.
    """
    if not memories:
        return memories
    
    original_count = len(memories)
    original_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    
    if DEBUG:
        print(Fore.CYAN + f"[Debug] Starting intelligent memory management: {original_count} entries, {original_tokens} tokens")
    
    # First pass: Recent memories (gentle condensation, preserve more)
    if len(memories) > 100:
        if DEBUG:
            print(Fore.YELLOW + "[Debug] Memory management: First pass (recent memories, 10% fraction)")
        memories = await condense_with_semantic_awareness(memories, fraction=0.1)
    
    # Second pass: Medium-age memories (moderate condensation)
    if len(memories) > 80:
        if DEBUG:
            print(Fore.YELLOW + "[Debug] Memory management: Second pass (medium-age memories, 20% fraction)")
        memories = await condense_with_semantic_awareness(memories, fraction=0.2)
    
    # Third pass: Older memories (more aggressive condensation)
    if len(memories) > 60:
        if DEBUG:
            print(Fore.YELLOW + "[Debug] Memory management: Third pass (older memories, 30% fraction)")
        memories = await condense_with_semantic_awareness(memories, fraction=0.3)
    
    # Emergency pass: Very aggressive if still too large
    current_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    if current_tokens > MAX_MEMORY_TOKENS:
        if DEBUG:
            print(Fore.RED + f"[Debug] Emergency condensation: {current_tokens} > {MAX_MEMORY_TOKENS}")
        memories = await condense_with_semantic_awareness(memories, fraction=0.4)
    
    final_count = len(memories)
    final_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    
    if DEBUG and (final_count != original_count or final_tokens != original_tokens):
        print(Fore.GREEN + f"[Debug] Memory management complete: "
                         f"{original_count} -> {final_count} entries, "
                         f"{original_tokens} -> {final_tokens} tokens "
                         f"(saved {original_tokens - final_tokens} tokens)")
    
    return memories

# ------------------ Enhanced Memory Condensation (Backward Compatible) ------------------ #

async def condense_memory(memories, fraction=0.2):
    """
    Enhanced memory condensation that uses pure semantic awareness,
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

# ------------------ ENHANCED Tension Analysis with Antagonist Preservation ------------------ #

async def analyze_tension_comprehensive_enhanced(memories, current_state):
    """
    Enhanced tension analysis that preserves antagonist data and handles errors gracefully.
    """
    
    # 1. Preserve existing antagonist data
    existing_antagonist = current_state.get("antagonist") if current_state else None
    
    # 2. Token limit validation using allocated budget
    context_messages, context_tokens = prepare_tension_analysis_context(
        memories, current_state, max_tokens=TENSION_SYSTEM_TOKENS
    )
    
    # 3. Resource loss analysis
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" 
                                  for m in context_messages[-10:]])
    events_occurred = await analyze_resource_loss(conversation_text, existing_antagonist)
    
    # 4. Calculate pressure floor ratcheting
    new_pressure_floor = calculate_pressure_floor_ratchet(current_state, 
                                                         events_occurred["events"])
    
    # 5. Enhanced tension analysis with antagonist preservation
    tension_prompt = f"""
You are analyzing narrative tension in an ongoing RPG story. Based on the conversation 
and current tension state, provide updated tension metrics.

IMPORTANT: Preserve the existing antagonist's name and motivation unless you have specific 
story reasons to change them. Only update resources_lost and commitment_level based on events.

Current Tension State: {json.dumps(current_state, indent=2)}
Recent Events Detected: {events_occurred}
Calculated Pressure Floor: {new_pressure_floor}

Recent Conversation:
{conversation_text}

Analyze and provide updates for:
1. How has narrative pressure changed? (0.0-1.0 scale)
2. What is the pressure source? (antagonist/environment/social/discovery)
3. How is tension manifesting? (exploration/tension/conflict/resolution)
4. Should any antagonist resources be lost due to recent events?
5. What should the antagonist's commitment level be?

PRESERVE the antagonist's existing name and motivation unless the story specifically changes them.

Return ONLY valid JSON in this exact format:
{{
  "narrative_pressure": 0.0,
  "pressure_source": "antagonist",
  "manifestation_type": "exploration",
  "escalation_count": {current_state.get('escalation_count', 0)},
  "base_pressure_floor": {new_pressure_floor},
  "antagonist": {{
    "name": "{existing_antagonist.get('name', 'Unknown Adversary') if existing_antagonist else 'Unknown Adversary'}",
    "motivation": "{existing_antagonist.get('motivation', 'opposes the player through unknown means') if existing_antagonist else 'opposes the player through unknown means'}", 
    "resources_lost": ["array", "of", "strings"],
    "commitment_level": "testing"
  }}
}}
"""
    
    # 6. Attempt analysis with enhanced error handling
    for attempt in range(3):
        try:
            response = await call_mcp([{"role": "system", "content": tension_prompt}])
            new_state = json.loads(response)
            
            # 7. Enhanced validation and antagonist preservation
            new_state = validate_tension_state(new_state)
            
            # 8. Apply antagonist preservation logic
            if existing_antagonist:
                preserved_antagonist = preserve_antagonist_data(new_state, existing_antagonist)
                new_state["antagonist"] = preserved_antagonist
                
                if DEBUG:
                    print(Fore.CYAN + f"[Debug] Antagonist preserved: {preserved_antagonist['name']}")
            
            # 9. Update message count
            new_state["last_analysis_count"] = count_total_messages(memories)
            
            return new_state
            
        except Exception as e:
            if DEBUG:
                print(Fore.YELLOW + f"[Debug] Enhanced tension analysis attempt {attempt + 1} failed: {e}")
            
            if attempt == 2:  # Final attempt failed
                if DEBUG:
                    print(Fore.YELLOW + "[Debug] All tension analysis attempts failed, preserving state")
                
                # Return current state with minimal updates to preserve antagonist
                if current_state:
                    current_state["last_analysis_count"] = count_total_messages(memories)
                    return current_state
                else:
                    return create_initial_tension_state()
    
    return current_state or create_initial_tension_state()

# ------------------ ENHANCED First-Time Antagonist Generation ------------------ #

async def initialize_tension_system(memories):
    """
    Enhanced initialization that ensures high-quality antagonist generation.
    """
    if DEBUG:
        print(Fore.YELLOW + "[Debug] First tension analysis - generating antagonist with enhanced system...")
    
    # Create initial state
    current_state = create_initial_tension_state()
    
    # Generate high-quality antagonist
    antagonist = await safe_antagonist_generation(memories)
    current_state["antagonist"] = antagonist
    
    # Perform initial tension analysis with the new antagonist
    enhanced_state = await analyze_tension_comprehensive_enhanced(memories, current_state)
    
    if DEBUG:
        final_antagonist = enhanced_state["antagonist"]
        print(Fore.GREEN + f"[Debug] Tension system initialized successfully:")
        print(Fore.GREEN + f"  Antagonist: {final_antagonist['name']}")
        print(Fore.GREEN + f"  Motivation: {final_antagonist['motivation']}")
        print(Fore.GREEN + f"  Quality validated: {validate_antagonist_quality(final_antagonist)}")
    
    return enhanced_state

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

def generate_enhanced_tension_context_prompt(tension_state):
    """
    Generate enhanced system prompt context with detailed antagonist information.
    """
    antagonist = tension_state["antagonist"]
    pressure_name = get_pressure_name(tension_state["narrative_pressure"])
    quality_indicator = "well-established" if validate_antagonist_quality(antagonist) else "developing"
    
    return f"""
**ENHANCED TENSION CONTEXT**: Current narrative pressure is {pressure_name} ({tension_state['narrative_pressure']:.2f}). 

The {quality_indicator} antagonist "{antagonist['name']}" is motivated by: {antagonist['motivation']}

Current threat level: {antagonist['commitment_level']} commitment
Resources compromised: {', '.join(antagonist['resources_lost']) if antagonist['resources_lost'] else 'none yet'}
Escalation count: {tension_state['escalation_count']}
Pressure floor: {tension_state['base_pressure_floor']:.3f}

Based on this antagonist's specific motivation and current commitment level, weave appropriate 
tension, threats, clues, and confrontation opportunities into your narrative. The antagonist's 
actions should reflect their established personality and goals, not generic villain behavior.

Antagonist behavior guidance:
- Testing: Probing, indirect influence, gathering information
- Engaged: Direct but limited confrontation, moderate resource commitment  
- Desperate: High-stakes actions, significant resource expenditure
- Cornered: All-or-nothing approach, maximum threat escalation
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

# ------------------ ENHANCED Main Loop with Antagonist Preservation ------------------ #

async def main_loop():
    print_wrapped("Aurora RPG Client (TN5 - Enhanced Antagonist Persistence)", color=Fore.CYAN)
    print_wrapped("Advanced semantic memory with improved tension system and antagonist preservation.", color=Fore.CYAN)
    print_wrapped("Type your action. Press ENTER twice to submit.", color=Fore.CYAN)
    print_wrapped("Terminate using: /quit\n", color=Fore.CYAN)

    # Validate token allocation at startup
    validate_token_allocation()

    # Load and optimize prompts at startup
    print_wrapped("Initializing system prompts...", color=Fore.CYAN)
    SYSTEM_PROMPT_TOP, SYSTEM_PROMPT_MID, SYSTEM_PROMPT_LOW = await load_and_optimize_prompts()
    print_wrapped("System ready!\n", color=Fore.GREEN)

    # Display semantic memory system info in debug mode
    if DEBUG:
        print(Fore.MAGENTA + "[Debug] Enhanced Semantic Memory System Active:")
        for category, config in MEMORY_CATEGORIES.items():
            print(Fore.MAGENTA + f"  {category}: {config['preservation_ratio']*100}% preservation - {config['description']}")
        print(Fore.MAGENTA + "[Debug] Antagonist Persistence System: ENABLED")
        print()

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
            buffer = raw_input_text.split('\n')
            continue

        print_wrapped(f"You: {raw_input_text}", color=Fore.BLUE, indent=2)
        add_memory(memories, "user", raw_input_text)

        # ENHANCED Tension Analysis Check
        if should_analyze_tension(memories):
            if DEBUG:
                print(Fore.YELLOW + "[Debug] Triggering enhanced tension analysis...")
            
            current_state = get_tension_state(memories)
            
            if current_state is None:
                # First analysis - use enhanced initialization
                new_state = await initialize_tension_system(memories)
                update_tension_state(memories, new_state)
            else:
                # Subsequent analysis - use enhanced system with preservation
                new_state = await analyze_tension_comprehensive_enhanced(memories, current_state)
                update_tension_state(memories, new_state)
            
            # Enhanced debug output
            if DEBUG:
                pressure_level = new_state["narrative_pressure"]
                pressure_name = get_pressure_name(pressure_level)
                antagonist = new_state["antagonist"]
                quality_check = validate_antagonist_quality(antagonist)
                
                print(Fore.YELLOW + f"[Debug] Enhanced Tension Analysis Results:")
                print(Fore.YELLOW + f"  Pressure: {pressure_level:.3f} ({pressure_name})")
                print(Fore.YELLOW + f"  Source: {new_state['pressure_source']}")
                print(Fore.YELLOW + f"  Manifestation: {new_state['manifestation_type']}")
                print(Fore.YELLOW + f"  Escalation Count: {new_state['escalation_count']}")
                print(Fore.YELLOW + f"  Pressure Floor: {new_state['base_pressure_floor']:.3f}")
                print(Fore.GREEN + f"  Antagonist: {antagonist['name']}")
                print(Fore.GREEN + f"  Motivation: {antagonist['motivation']}")
                print(Fore.GREEN + f"  Commitment: {antagonist['commitment_level']}")
                print(Fore.GREEN + f"  Resources Lost: {antagonist['resources_lost']}")
                print(Fore.CYAN + f"  Quality Validated: {quality_check}")

        # Enhanced Memory Management - Use pure semantic condensation system
        total_memory_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
        if total_memory_tokens > MAX_MEMORY_TOKENS:
            if DEBUG:
                print(Fore.YELLOW + f"[Debug] Memory condensation triggered: {total_memory_tokens} > {MAX_MEMORY_TOKENS}")
                print(Fore.MAGENTA + "[Debug] Using enhanced semantic condensation system...")
            
            memories = await condense_memory(memories, fraction=0.2)
            
            # Verify memory reduction and show semantic analysis results
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
        
        # Add enhanced tension context with detailed antagonist information
        tension_state = get_tension_state(memories)
        if tension_state and tension_state["antagonist"]:
            tension_context = generate_enhanced_tension_context_prompt(tension_state)
            system_messages.append({"role": "system", "content": tension_context})
        
        # Filter out tension_state entries from memory messages for LLM context
        memory_messages = [{"role": mem["role"], "content": mem["content"]} 
                          for mem in memories if mem.get("role") != "tension_state"]
        
        # Assemble final message context
        messages = system_messages + memory_messages + [{"role": "user", "content": raw_input_text}]

        # Enhanced debug output
        if DEBUG:
            total_message_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)
            system_tokens = sum(estimate_tokens(msg["content"]) for msg in system_messages)
            memory_tokens = sum(estimate_tokens(msg["content"]) for msg in memory_messages)
            input_tokens = estimate_tokens(raw_input_text)
            
            print(Fore.YELLOW + f"[Debug] Enhanced message composition:")
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
