# Aurora RPG Client - Evolution Documentation

A terminal-based RPG storyteller client that leverages Large Language Model capabilities through MCP (Model Control Protocol) to create immersive, interactive high-fantasy narratives. This document traces the evolution from proof-of-concept to a sophisticated system with intelligent memory management, dynamic prompt optimization, and unified tension systems.

## Project Overview

The Aurora RPG Client began as a simple proof-of-concept for AI-powered storytelling and evolved into a comprehensive narrative engine featuring:

- **Interactive Storytelling**: Multi-line input system for natural conversation with an AI Game Master
- **Persistent Memory**: JSON-based conversation storage with intelligent LLM-powered condensation
- **Modular System Prompts**: Flexible three-part prompt architecture
- **Unified Tension System**: Revolutionary approach to organic conflict generation
- **Companion Characters**: Rich NPCs with detailed personalities and abilities

## Technical Foundation

### Core Stack
- **Python 3.10+**: Modern async/await architecture
- **MCP (Model Control Protocol)**: Communication with local LLM server
- **Qwen2.5 14B Instruct**: Primary language model for storytelling
- **JSON Storage**: Persistent conversation and state management
- **Colorama**: Enhanced terminal output

### Architecture Principles
- **LLM-First Design**: Use AI intelligence for complex semantic decisions
- **Token Efficiency**: Smart resource management within context windows
- **Modular Components**: Clean separation of concerns
- **Error Resilience**: Graceful degradation and recovery

## Version History & Evolution

### POC (Proof of Concept) - `aurora3.poc.py`

The initial proof-of-concept established the foundational architecture for AI-powered RPG storytelling.

#### Core Features
- **Basic MCP Integration**: HTTP client for communicating with local LLM server
- **Simple Memory System**: JSON-based conversation persistence
- **Multi-line Input**: Double-enter submission for natural dialogue flow
- **System Prompt**: Single RPG storyteller prompt for AI behavior
- **Error Handling**: Basic retry logic for network timeouts

#### Key Components
```python
# Simple system prompt
SYSTEM_PROMPT = (
    "You are the an RPG storyteller. Your role is to guide the player through a high-fantasy world, "
    "creating engaging scenarios, characters, and adventures. Maintain continuity across the story, "
    "remember player choices, and respond creatively and consistently."
)

# Basic memory operations
def load_memory() -> List[Dict[str, Any]]
def save_memory(memories: List[Dict[str, Any]])
def add_memory(memories: List[Dict[str, Any]], role: str, content: str)

# MCP communication
async def call_mcp(messages: List[Dict[str, str]]) -> str
```

#### Architecture Decisions
- **Memory Structure**: Each conversation turn stored as dictionary with ID, role, content, and timestamp
- **All-Memory Approach**: Every message sent to LLM includes complete conversation history
- **Simple Commands**: `/mems` (show recent), `/dump` (debug output), `/quit` (exit)
- **No Token Management**: Relied on model's context window without optimization

#### Limitations Identified
1. **Memory Growth**: Conversation history would eventually exceed context limits
2. **No Token Tracking**: Could hit context window unexpectedly
3. **Single System Prompt**: Limited flexibility for complex GM behaviors
4. **No Tension Management**: Purely reactive storytelling without narrative structure

The POC successfully demonstrated the core concept of LLM-powered RPG storytelling and established the foundation for all subsequent improvements.

## Memory Management Evolution

### MEM1 - Basic Token Limits - `aurora3.mem1.py`

The first major iteration addressed the POC's memory growth problem by implementing basic token management.

#### Key Improvements
- **Token Estimation**: Simple character-to-token ratio calculation (`len(text) // 4`)
- **Memory Pruning**: Automatic removal of oldest messages when approaching limits
- **VRAM-Based Limits**: Token calculations based on available GPU memory
- **System Prompt Integration**: Automatic injection if memory file exists but is empty

#### New Configuration
```python
TOTAL_VRAM_BYTES = 8 * 1024**3  # 8GB VRAM assumption
BYTES_PER_TOKEN = 10
MAX_TOKENS = TOTAL_VRAM_BYTES // BYTES_PER_TOKEN
MAX_MEMORY_TOKENS = int(MAX_TOKENS * 0.7)  # 70% for memory
MAX_PROMPT_TOKENS = int(MAX_TOKENS * 0.3)  # 30% for current context
```

#### Token Management Logic
```python
def prune_memory(memories: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    total_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    while total_tokens > max_tokens:
        memories.pop(0)  # Remove oldest memory
        total_tokens = sum(estimate_tokens(mem["content"]) for mem in memories)
    return memories
```

#### Limitations
- **Crude Pruning**: Simple deletion of oldest messages lost important context
- **No Semantic Awareness**: Couldn't distinguish between important and trivial content
- **Hard Token Limits**: Inflexible boundaries that could cut off mid-conversation

### MEM2 - LLM-Powered Condensation - `aurora3.mem2.py`

Revolutionary approach: Use the LLM itself to intelligently compress memory while preserving story continuity.

#### Breakthrough Innovation
```python
async def condense_memory(memories: List[Dict[str, Any]], fraction: float = 0.2) -> List[Dict[str, Any]]:
    """Condense the oldest fraction of memories using the MCP."""
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
```

#### Key Advantages
- **Semantic Preservation**: LLM understands story importance and context
- **Intelligent Compression**: Maintains narrative continuity while reducing tokens
- **Gradual Processing**: Condenses 20% of memory at a time
- **Context Awareness**: LLM knows what information is crucial for storytelling

#### Impact
This represented a paradigm shift from programmatic to AI-powered memory management, establishing the principle that LLMs should handle semantic decisions.

### MEM3 - Context Window Architecture - `aurora3.mem3.py`

Introduced sophisticated token budget management with realistic context window limits.

#### Advanced Token Allocation
```python
CONTEXT_WINDOW = 32000  # Realistic model limit
SYSTEM_PROMPT_TOKENS = 5000  # Reserved for GM prompts

REMAINING_TOKENS = CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS
MEMORY_FRACTION = 0.7
MAX_MEMORY_TOKENS = int(REMAINING_TOKENS * MEMORY_FRACTION)  # ~18,900 tokens
MAX_USER_INPUT_TOKENS = int(REMAINING_TOKENS * (1 - MEMORY_FRACTION))  # ~8,100 tokens
```

#### Multi-Part System Prompts
Replaced single system prompt with three specialized components:

```python
SYSTEM_PROMPT_TOP = (
    "You are a Game Master language model responsible for guiding a single player character..."
    # Core GM rules and behavior
)

SYSTEM_PROMPT_MID = (
    "Aurora is a human bard who functions as the player character's constant companion..."
    # Companion character definition
)

SYSTEM_PROMPT_LOW = (
    "You function as an advanced Game Master language model generating dense, coherent..."
    # Narrative generation guidelines
)
```

#### Dynamic Assembly
```python
# System prompts NOT saved to memory - injected fresh each time
system_messages = [
    {"role": "system", "content": SYSTEM_PROMPT_TOP},
    {"role": "system", "content": SYSTEM_PROMPT_MID}, 
    {"role": "system", "content": SYSTEM_PROMPT_LOW}
]
memory_messages = [{"role": mem["role"], "content": mem["content"]} for mem in memories if mem["role"] != "system"]
messages = system_messages + memory_messages + [{"role": "user", "content": raw_input_text}]
```

#### Innovations
- **Separation of Concerns**: System prompts vs. conversation memory
- **Flexible Prompt Architecture**: Easy to modify GM behavior without affecting memory
- **Realistic Token Budgets**: Based on actual model capabilities
- **Dynamic Context Assembly**: Fresh system prompts every generation

### MEM4 - Streamlined Implementation - `aurora3.mem4.py`

Refined and optimized the MEM3 architecture with cleaner code and better error handling.

#### Code Quality Improvements
- **Simplified Condensation**: More elegant LLM-powered compression
- **Better Token Estimation**: Refined character-to-token calculations
- **Enhanced Error Handling**: Graceful degradation when condensation fails
- **Cleaner Architecture**: Removed redundant code and improved readability

#### Optimized Condensation
```python
estimate_tokens = lambda text: max(1, len(text) // 4)

async def condense_memory(memories, fraction=0.2):
    if not memories: return memories
    chunk, memories = memories[:int(len(memories)*fraction)], memories[int(len(memories)*fraction):]
    text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in chunk)
    prompt = f"Condense memories while preserving story:\n{text}"
    condensed = await call_mcp([{"role": "system", "content": prompt}])
    return [{"id": str(uuid4()), "role": "system", "content": condensed, "timestamp": now_iso()}] + memories
```

### MEM5 - Enhanced User Experience - `aurora3.mem5.py`

Added sophisticated formatting and user interface improvements while maintaining the robust memory architecture.

#### User Experience Enhancements
- **Intelligent Word Wrapping**: Paragraph-aware text formatting
- **Terminal Width Detection**: Adaptive output formatting
- **Enhanced Visual Feedback**: Better color coding and indentation
- **Improved Input Echo**: Clear display of user input before processing

#### Advanced Text Formatting
```python
def print_wrapped(text: str, color=Fore.GREEN, indent: int = 0):
    """Print text wrapped to terminal width, preserving existing paragraph breaks."""
    width = get_terminal_width() - indent
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=' ' * indent)
    
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        lines = para.splitlines()
        wrapped_para = "\n".join(wrapper.fill(line) for line in lines)
        print(color + wrapped_para)
        if i < len(paragraphs) - 1:
            print("")  # Blank line between paragraphs
```

#### Memory Architecture Maturity
By MEM5, the memory management system had achieved:
- **Reliable LLM Condensation**: Proven semantic preservation
- **Efficient Token Usage**: Optimal allocation of context window
- **Modular Prompts**: Flexible three-part system prompt architecture  
- **User-Friendly Interface**: Professional terminal application experience

## Development & Optimization Phase

### DEV - External Prompt Files - `aurora3.dev.py`

Transitioned from hardcoded system prompts to external file-based configuration, enabling user customization and easier prompt development.

#### File-Based Prompt System
```python
# Prompt file configuration
PROMPT_TOP_FILE = Path("critrules.prompt")     # Game Master core rules
PROMPT_MID_FILE = Path("companion.prompt")    # Aurora companion definition
PROMPT_LOW_FILE = Path("lowrules.prompt")     # Narrative generation guidelines

def load_prompt(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()
```

#### Startup Prompt Loading
```python
# Load prompts once at startup
SYSTEM_PROMPT_TOP = load_prompt(PROMPT_TOP_FILE)
SYSTEM_PROMPT_MID = load_prompt(PROMPT_MID_FILE)
SYSTEM_PROMPT_LOW = load_prompt(PROMPT_LOW_FILE)

if DEBUG:
    print(Fore.YELLOW + f"[Debug] Loaded prompts: "
          f"TOP({len(SYSTEM_PROMPT_TOP)} chars), "
          f"MID({len(SYSTEM_PROMPT_MID)} chars), "
          f"LOW({len(SYSTEM_PROMPT_LOW)} chars)")
```

#### Key Benefits
- **User Customization**: Easy modification of AI behavior without code changes
- **Version Control**: Prompt files can be tracked and shared separately
- **Iterative Development**: Rapid prompt testing and refinement
- **Modular Design**: Each prompt file serves specific narrative functions
- **Error Handling**: Clear feedback for missing or invalid prompt files

#### File Structure Established
```
aurora-rpg-client/
├── aurora3.dev.py
├── memory.json
├── critrules.prompt    # GM behavior and boundaries
├── companion.prompt    # Aurora character definition
└── lowrules.prompt     # Narrative generation rules
```

### DV1 - Intelligent Prompt Optimization - `aurora3.dv1.py`

Revolutionary advancement: Automatic prompt condensation using LLM intelligence when prompts exceed token budgets.

#### The Prompt Size Problem
As prompts became more detailed and comprehensive, they began exceeding the allocated 5,000 token budget. DV1 solved this with intelligent condensation.

#### Prompt-Specific Condensation System
```python
async def condense_prompt(content: str, prompt_type: str) -> str:
    """Condense a single prompt file while preserving essential functionality."""
    condensation_prompts = {
        "critrules": (
            "You are optimizing a Game Master system prompt for an RPG. "
            "Condense the following prompt while preserving all essential game master rules, "
            "narrative generation guidelines, and core functionality..."
        ),
        "companion": (
            "You are optimizing a character definition prompt for an RPG companion. "
            "Condense the following prompt while preserving the companion's complete personality, "
            "appearance, abilities, relationship dynamics, and behavioral patterns..."
        ),
        "lowrules": (
            "You are optimizing a narrative generation prompt for an RPG system. "
            "Condense the following prompt while preserving all narrative guidelines..."
        )
    }
    
    prompt_instruction = condensation_prompts.get(prompt_type, condensation_prompts["critrules"])
    condensed = await call_mcp([{"role": "system", "content": prompt_instruction}])
    return condensed.strip()
```

#### Intelligent Loading and Optimization
```python
async def load_and_optimize_prompts() -> Tuple[str, str, str]:
    """Load all prompt files and apply condensation if they exceed the token budget."""
    
    # Load all prompt files
    prompts = {
        'critrules': load_prompt(PROMPT_TOP_FILE),
        'companion': load_prompt(PROMPT_MID_FILE), 
        'lowrules': load_prompt(PROMPT_LOW_FILE)
    }
    
    # Calculate combined token count
    total_tokens = sum(estimate_tokens(content) for content in prompts.values())
    
    # Apply condensation if budget exceeded
    if total_tokens > SYSTEM_PROMPT_TOKENS:
        print(Fore.YELLOW + f"[System] Prompt files exceed token budget ({total_tokens} > {SYSTEM_PROMPT_TOKENS})")
        print(Fore.YELLOW + "[System] Applying intelligent condensation...")
        
        # Condense any file that's more than 1/3 of the total budget
        individual_threshold = SYSTEM_PROMPT_TOKENS // 3
        
        for prompt_type, content in prompts.items():
            if estimate_tokens(content) > individual_threshold:
                prompts[prompt_type] = await condense_prompt(content, prompt_type)
    
    return prompts['critrules'], prompts['companion'], prompts['lowrules']
```

#### Key Innovations
- **Automatic Optimization**: No manual intervention required when prompts grow too large
- **Functionality Preservation**: LLM understands what aspects of each prompt are essential
- **Isolated Processing**: Each prompt type condensed separately to prevent cross-contamination
- **Fallback Protection**: Original prompts used if condensation fails
- **Debug Transparency**: Detailed logging of condensation process and results

#### Impact on Development Workflow
- **Fearless Prompt Expansion**: Developers could write comprehensive prompts without token anxiety
- **Automatic Scaling**: System adapts to prompt complexity without manual tuning
- **Preserved Quality**: LLM-based condensation maintains prompt effectiveness
- **Production Ready**: Robust error handling and graceful degradation

### DV2 - Enhanced Architecture & Validation - `aurora3.dv2.py`

The culmination of the development phase, featuring refined architecture, improved error handling, and enhanced user experience.

#### Architectural Refinements
- **Cleaner Codebase**: Streamlined functions and improved code organization
- **Enhanced Error Handling**: More robust failure modes and recovery strategies
- **Improved Debug Output**: Better visibility into system operations
- **Validation Framework**: Input and configuration validation

#### Prompt Optimization Maturity
DV2 refined the intelligent prompt condensation system with:
- **Better Token Calculations**: More accurate estimation and budget management
- **Improved Condensation Logic**: Enhanced LLM instructions for each prompt type
- **Robust Error Recovery**: Graceful handling of condensation failures
- **Performance Monitoring**: Real-time tracking of optimization effectiveness

#### User Experience Polish
```python
print_wrapped("Aurora RPG Client (DV1 - Intelligent Prompt Management)", color=Fore.CYAN)
print_wrapped("Type your action. Press ENTER twice to submit.", color=Fore.CYAN)
print_wrapped("Terminate using: /quit\n", color=Fore.CYAN)

# Load and optimize prompts at startup
print_wrapped("Initializing system prompts...", color=Fore.CYAN)
SYSTEM_PROMPT_TOP, SYSTEM_PROMPT_MID, SYSTEM_PROMPT_LOW = await load_and_optimize_prompts()
print_wrapped("System ready!\n", color=Fore.GREEN)
```

#### Development Phase Achievements
By DV2, the Aurora RPG Client had achieved:

1. **Complete Modularity**: External prompt files with automatic optimization
2. **Zero-Config Operation**: Intelligent adaptation to prompt complexity
3. **Production Quality**: Robust error handling and user feedback
4. **Developer Friendly**: Easy customization and debugging capabilities
5. **Scalable Architecture**: Foundation for advanced features

#### Technical Excellence
- **LLM-First Philosophy**: AI handles all semantic decisions (memory condensation, prompt optimization)
- **Token Efficiency**: Sophisticated budget management across all system components
- **User Experience**: Professional terminal application with intuitive interface
- **Reliability**: Comprehensive error handling and graceful degradation
- **Extensibility**: Clean architecture supporting future enhancements

## Installation & Usage

### Prerequisites
- Python 3.10+
- Ollama with MCP server on port 3456
- Qwen2.5 14B Instruct model (`ollama pull qwen2.5:14b-instruct-q4_k_m`)

### Quick Start
1. Install dependencies: `pip install httpx colorama`
2. Create prompt files (critrules.prompt, companion.prompt, lowrules.prompt)
3. Run: `python aurora3.dv2.py` (or add `--debug` for detailed logging)

### File Structure
```
aurora-rpg-client/
├── aurora3.dv2.py         # Latest version with all features
├── memory.json            # Persistent conversation memory (auto-created)
├── critrules.prompt       # Game Master rules and behavior
├── companion.prompt       # Companion character definitions
├── lowrules.prompt        # Narrative generation guidelines
└── iterations/            # Historical versions for reference
```

## Technical Innovation Summary

The Aurora RPG Client represents a breakthrough in AI-powered interactive storytelling through several key innovations:

### 1. LLM-Powered Memory Management
Instead of programmatic rules for memory compression, the system uses the language model itself to semantically understand and preserve important story elements while reducing token usage.

### 2. Intelligent Prompt Optimization  
Automatic condensation of oversized prompt files using AI to maintain functionality while fitting within token budgets, enabling fearless prompt development.

### 3. Modular Prompt Architecture
Three-part system (rules, character, narrative) allowing targeted customization while maintaining clean separation of concerns.

### 4. Dynamic Token Budget Management
Sophisticated allocation of the 32K context window across system prompts, memory, and user input with automatic optimization.

### 5. Production-Ready Reliability
Comprehensive error handling, graceful degradation, and user-friendly feedback throughout all system operations.

This evolution from simple proof-of-concept to sophisticated narrative engine demonstrates the power of treating LLMs as semantic partners rather than simple text generators, establishing new paradigms for AI-assisted creative applications.