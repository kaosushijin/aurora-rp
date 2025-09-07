# Aurora RPG Client TN3 - LLM-Driven Semantic Memory

A terminal-based RPG storytelling client that uses Large Language Model capabilities through the Ollama ecosystem to create immersive, interactive high-fantasy narratives. Features intelligent semantic memory management, dynamic prompt optimization, unified tension systems, and LLM-driven content categorization for superior story continuity.

## Overview

Aurora RPG Client TN3 represents an advancement in AI-powered interactive storytelling by implementing semantic memory management alongside unified tension systems. The application treats narrative elements as semantically meaningful content that can be intelligently preserved, condensed, and referenced, creating coherent long-form storytelling experiences that maintain consistency across extended play sessions.

## Core Features

### Advanced Semantic Memory System
- **LLM-Driven Content Analysis**: AI categorizes conversation content into story-critical, character-focused, world-building, and standard interaction types
- **Category-Specific Preservation**: Different content types receive tailored condensation strategies with varying preservation ratios
- **Progressive Memory Management**: Multiple condensation passes with increasing aggressiveness based on memory pressure
- **Intelligent Context Retention**: Maintains important story elements, character relationships, and plot threads through semantic understanding

### Unified Tension System
- **Narrative Pressure Tracking**: Continuous 0.0-1.0 scale monitoring of story tension with logarithmic progression
- **Dynamic Antagonist Management**: AI-generated antagonists with evolving commitment levels and resource tracking
- **Pressure Floor Ratcheting**: Prevents infinite antagonist retreats through escalating minimum tension
- **15-Message Analysis Cycle**: Automated tension evaluation every 15 exchanges for optimal pacing

### Intelligent Prompt System
- **Dynamic Optimization**: Automatically condenses oversized prompt files using LLM intelligence
- **Modular Architecture**: Three-part system (Game Master rules, companion definition, narrative guidelines)
- **Memory-Aware Integration**: System prompts designed to work with semantic memory categorization
- **Real-time Context Injection**: Current tension and antagonist states inform narrative generation

### Interactive Storytelling Engine
- **Multi-line Input System**: Natural conversation flow with double-enter submission
- **Rich Companion Characters**: Detailed NPCs with consistent personalities and musical abilities
- **Player Agency Preservation**: Complete autonomy over character actions, thoughts, and dialogue
- **Immersive World Building**: Dense environmental descriptions using all five senses

## Software Stack

### Core Infrastructure
- **Python 3.10+**: Modern async/await architecture with type hints
- **Ollama**: Local LLM hosting and management platform
- **Ollama MCP Server**: Model Control Protocol interface running on `127.0.0.1:3456`
- **Qwen2.5 14B Instruct Q4_K_M**: Primary language model for storytelling and analysis

### Key Dependencies
- **httpx**: Async HTTP client for MCP communication
- **colorama**: Cross-platform colored terminal output
- **pathlib**: Modern file system operations
- **json**: Structured data storage and API communication

### Performance Characteristics
- **Context Window**: 32,000 tokens total capacity
- **System Prompts**: 5,000 token budget with automatic optimization
- **Tension System**: 6,000 token budget for analysis operations
- **Memory Storage**: 19,000 tokens with semantic categorization
- **User Input**: 2,000 token limit with validation

## Installation & Setup

### Prerequisites
1. **Ollama Installation**: Download and install [Ollama](https://ollama.ai/)
2. **Model Download**: `ollama pull qwen2.5:14b-instruct-q4_k_m`
3. **MCP Server**: Set up Ollama MCP server on port 3456
4. **Python Environment**: Python 3.10+ with pip

### Dependencies Installation
```bash
pip install httpx colorama
```

### Required Prompt Files
Create these files in your project directory:
- `critrules.prompt` - Game Master core rules with memory integration
- `companion.prompt` - Aurora companion character definition
- `lowrules.prompt` - Narrative generation guidelines with semantic awareness

### Launch
```bash
# Standard mode
python aurora3.tn3.py

# Debug mode with comprehensive logging
python aurora3.tn3.py --debug
```

## Semantic Memory System Deep Dive

### Content Categorization
The system analyzes conversation segments and categorizes them into four types:

**Story-Critical Content (80% preservation)**
- Major plot developments, character deaths, world-changing events
- Key player decisions and their consequences
- Villain revelations and quest completions
- Uses decisive language highlighting significance

**Character-Focused Content (70% preservation)**
- Relationship changes and trust/betrayal moments
- Character motivations and personality reveals
- Aurora's development and NPC personality traits
- Emphasizes emotional weight and relationship dynamics

**World-Building Content (60% preservation)**
- New locations, lore revelations, cultural information
- Political changes and economic systems
- Magical discoveries and historical context
- Provides rich foundational details for future reference

**Standard Content (40% preservation)**
- General interactions and casual conversations
- Travel descriptions and routine activities
- Basic world interactions without major significance
- Maintains story flow while compressing non-essential details

### Progressive Condensation Strategy
1. **First Pass**: Recent memories (last 100 entries) - gentle 10% condensation
2. **Second Pass**: Medium-age memories (80+ entries) - moderate 20% condensation
3. **Third Pass**: Older memories (60+ entries) - aggressive 30% condensation
4. **Emergency Pass**: Critical memory pressure - 40% condensation with preservation priority

### Memory Analysis Process
The LLM evaluates conversation chunks using keyword triggers and semantic analysis:
- **Trigger Keywords**: Death, reveal, discover, decide, trust, betray, location, lore, magic
- **Confidence Scoring**: 0.0-1.0 confidence in categorization accuracy
- **Key Element Extraction**: Identifies essential story elements to preserve
- **Fallback Protection**: Graceful degradation when analysis fails

## Unified Tension System

### Tension State Architecture
```json
{
  "narrative_pressure": 0.25,
  "pressure_source": "antagonist",
  "manifestation_type": "tension", 
  "escalation_count": 2,
  "base_pressure_floor": 0.05,
  "last_analysis_count": 45,
  "antagonist": {
    "name": "Lord Malachar",
    "motivation": "seeks ancient artifact",
    "resources_lost": ["reputation", "allies"],
    "commitment_level": "engaged"
  }
}
```

### Pressure Progression
- **Low (0.0-0.1)**: Exploration focus, subtle antagonist presence, rich environmental details
- **Building (0.1-0.3)**: Direct but avoidable encounters, clear antagonist influence
- **Critical (0.3-0.6)**: Unavoidable confrontations, high stakes scenarios
- **Explosive (0.6-1.0)**: Final confrontation territory, resolution imminent

### Antagonist Commitment Levels
- **Testing**: May retreat easily, probing for weaknesses
- **Engaged**: Moderate cost to retreat, more persistent tactics
- **Desperate**: High cost to retreat, increasingly dangerous behavior
- **Cornered**: Cannot retreat, must commit fully to conflict resolution

## Technical Architecture

### Token Budget Management
- **Optimized Allocation**: Dynamic distribution based on content priority and system needs
- **Context Optimization**: Automatic prompt condensation preserving functionality
- **Memory Efficiency**: Semantic-aware condensation targeting appropriate content
- **Performance Monitoring**: Real-time token usage tracking in debug mode

### Error Resilience
- **Network Resilience**: Automatic retry with exponential backoff for MCP communication
- **Graceful Degradation**: Safe fallbacks when semantic analysis or tension systems fail
- **State Recovery**: Robust handling of corrupted memory or tension states
- **Debug Transparency**: Comprehensive logging for troubleshooting and optimization

### LLM-First Design Philosophy
- **Semantic Decision Making**: AI handles content importance, categorization, and preservation strategies
- **Adaptive Responses**: Dynamic prompt assembly based on detected narrative context
- **Intelligent Compression**: LLM-powered condensation that understands story significance
- **Context-Aware Generation**: Narrative responses informed by tension state and memory categories

## Program Flow

### Startup Sequence
1. **Initialization**: Load configuration and validate token allocation
2. **Prompt Optimization**: Load and conditionally condense system prompts within budget
3. **Memory Recovery**: Restore conversation history and tension state from JSON storage
4. **System Ready**: Display startup message and enter main interaction loop

### Enhanced Main Loop
1. **User Input**: Multi-line input collection with length validation
2. **Semantic Memory Management**: LLM-driven content analysis and category-specific condensation
3. **Tension Analysis**: Every 15 messages, comprehensive evaluation including resource loss detection
4. **Context Assembly**: Integration of optimized prompts, categorized memory, and tension context
5. **Story Generation**: MCP server communication with retry logic and error handling
6. **Response Processing**: Formatted output with intelligent word wrapping and state persistence

## Commands

- `/quit` - Exit the application gracefully (saves all state)

## File Structure

```
aurora-rpg-client/
├── aurora3.tn3.py          # Main application with semantic memory system
├── memory.json             # Persistent conversation memory (auto-created)
├── critrules.prompt        # Game Master rules with memory integration
├── companion.prompt        # Aurora companion character definition
├── lowrules.prompt         # Narrative generation with semantic awareness
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

## Configuration

### Model Settings
- **MCP URL**: `http://127.0.0.1:3456/chat`
- **Model**: `qwen2.5:14b-instruct-q4_k_m`
- **Timeout**: 300 seconds with exponential backoff retry
- **Context Window**: 32,000 tokens total capacity

### Memory System Parameters
- **Analysis Frequency**: Triggered by memory pressure and content significance
- **Preservation Ratios**: 80%/70%/60%/40% for story-critical/character/world/standard content
- **Progressive Condensation**: Multiple passes with increasing aggressiveness
- **Semantic Keywords**: Comprehensive trigger word system for content categorization

### Tension System Parameters
- **Analysis Frequency**: Every 15 user/assistant message exchanges
- **Analysis Token Budget**: 6,000 tokens maximum per evaluation
- **Pressure Scaling**: Logarithmic progression with four distinct ranges
- **Ratchet Mechanics**: Cumulative pressure floor increases prevent infinite retreats

## System Requirements

### Hardware Requirements
- **RAM**: 8GB minimum for Qwen2.5 14B model
- **Storage**: 10GB for model files, minimal for application data
- **CPU**: Modern multi-core processor recommended
- **Network**: Local-only communication (no internet required)

### Software Requirements
- **Python**: 3.10 or higher with async/await support
- **Ollama**: Latest version with MCP server capability
- **Operating System**: Cross-platform (Windows, macOS, Linux)

## Technical Innovation

### Semantic Memory Management
Aurora TN3 introduces sophisticated content analysis that goes beyond simple token counting:

- **Content Understanding**: LLM evaluates narrative significance rather than relying on programmatic rules
- **Category-Specific Strategies**: Different content types receive appropriate preservation treatment
- **Intelligent Condensation**: Maintains story coherence while managing computational constraints
- **Adaptive Preservation**: Dynamic adjustment based on content importance and memory pressure

### Integrated Narrative Systems
The application treats memory management, tension tracking, and prompt optimization as interconnected systems:

- **Cross-System Communication**: Tension state informs memory categorization and vice versa
- **Unified Context Assembly**: All systems contribute to final narrative generation context
- **Semantic Consistency**: Content categorization aligns with tension-aware prompt guidance
- **Emergent Storytelling**: Complex narratives emerge from simple system interactions

## Development Philosophy

This project demonstrates an LLM-first approach to interactive storytelling:

- **AI Partnership**: Leverage language model intelligence for complex semantic decisions
- **Computational Efficiency**: Smart resource management within hardware constraints
- **User Experience Priority**: Seamless, immersive gameplay without technical complexity
- **Semantic Understanding**: Content meaning drives system behavior rather than rigid rules

## License

GNU Affero General Public License v3.0 - See LICENSE file for details.

## Troubleshooting

### Common Issues
1. **"Prompt file not found"**: Ensure all three .prompt files exist in the project directory
2. **Connection timeout**: Verify Ollama MCP server is running on port 3456
3. **Model not found**: Run `ollama pull qwen2.5:14b-instruct-q4_k_m` to download the required model
4. **Memory errors**: Check available RAM for the 14B parameter model
5. **Semantic analysis failures**: Enable debug mode to monitor LLM categorization accuracy

### Debug Mode Features
Run with `--debug` flag for comprehensive logging including:
- Token usage breakdowns across all system components
- Semantic memory categorization decisions and confidence scores
- Tension analysis details and antagonist state evolution
- Prompt optimization results and condensation effectiveness
- MCP server communication logs and retry mechanisms
- Memory management operations and preservation statistics

### Performance Optimization
- **Memory Pressure Monitoring**: Watch for frequent condensation cycles in debug output
- **Content Categorization Accuracy**: Verify semantic analysis produces appropriate classifications
- **Token Budget Utilization**: Ensure efficient allocation across system components
- **Response Generation Speed**: Monitor MCP communication latency and retry frequency

For additional support and advanced configuration options, consult the source code documentation and inline comments.