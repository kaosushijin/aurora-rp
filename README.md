# Aurora RPG Client TN1 - Unified Tension System

A terminal-based RPG storyteller client that leverages Large Language Model capabilities through the Ollama ecosystem to create immersive, interactive high-fantasy narratives with intelligent memory management, dynamic prompt optimization, and a revolutionary unified tension system that organically builds toward meaningful confrontations.

## Overview

Aurora RPG Client TN1 represents a breakthrough in AI-powered storytelling by implementing a "narrative gravity well" approach to tension management. Unlike traditional systems that separate exploration and combat into distinct modes, the Unified Tension System treats conflict as the natural manifestation of accumulated story pressure, creating seamless and engaging narrative progression that feels organic rather than mechanical.

## Core Features

### 🎭 Unified Tension System
- **Narrative Pressure Tracking**: Continuous 0.0-1.0 scale monitoring story tension with logarithmic progression
- **Dynamic Antagonist Management**: AI-generated antagonists with evolving commitment levels and resource tracking
- **Pressure Floor Ratcheting**: Prevents infinite antagonist retreats through escalating minimum tension
- **15-Message Analysis Cycle**: Automated tension evaluation every 15 exchanges for optimal pacing
- **Organic Confrontation Building**: Natural progression from exploration → tension → conflict → resolution

### 🧠 Advanced Memory Management
- **LLM-Driven Condensation**: Uses the language model itself to intelligently compress memory while preserving story continuity
- **Context-Aware Retention**: Maintains important story elements, character relationships, and plot threads
- **32K Token Context Window**: Sophisticated budget management with dynamic allocation
- **Persistent JSON Storage**: Seamless conversation continuity across sessions

### ⚡ Intelligent Prompt System
- **Dynamic Optimization**: Automatically condenses oversized prompt files using LLM intelligence
- **Modular Architecture**: Three-part system (Game Master rules, companion definition, narrative guidelines)
- **Functionality Preservation**: Maintains prompt effectiveness while reducing token footprint
- **Real-time Tension Context**: Injects current antagonist and pressure state into system prompts

### 🎮 Interactive Storytelling
- **Multi-line Input System**: Natural conversation flow with double-enter submission
- **Rich Companion Characters**: Detailed NPCs with consistent personalities and musical abilities
- **Player Agency Preservation**: Complete autonomy over character actions, thoughts, and dialogue
- **Immersive World Building**: Dense environmental descriptions with all five senses

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
- **Memory**: ~18,900 tokens (70% of remaining space)
- **User Input**: ~8,100 tokens (30% of remaining space)
- **Tension Analysis**: 8,000 token dedicated budget

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
- `critrules.prompt` - Game Master core rules and tension integration
- `companion.prompt` - Character definitions for NPCs/companions  
- `lowrules.prompt` - Narrative generation guidelines

### Launch
```bash
# Standard mode
python aurora3.tn1.py

# Debug mode with comprehensive logging
python aurora3.tn1.py --debug
```

## Unified Tension System Deep Dive

### Tension State Architecture
The system maintains a comprehensive tension state stored alongside conversation memory:

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

### Pressure Progression (Logarithmic Scaling)
- **Low (0.0-0.1)**: Exploration and character development, subtle antagonist presence
- **Building (0.1-0.3)**: Tension becomes noticeable, direct but avoidable encounters
- **Critical (0.3-0.6)**: Clear conflict approaching, high-stakes scenarios
- **Explosive (0.6-1.0)**: Confrontation imminent/inevitable, final resolution territory

### Antagonist Commitment Levels
- **Testing**: Low-cost retreats available, probing for weaknesses
- **Engaged**: Moderate cost to retreat, more persistent tactics
- **Desperate**: High cost to retreat, increasingly dangerous behavior
- **Cornered**: Cannot retreat - must commit fully to conflict resolution

### Resource Loss & Ratcheting
The system tracks when antagonists lose resources (allies, reputation, territory) through player actions, which:
- Increases their commitment level
- Raises the minimum pressure floor
- Prevents infinite conflict postponement
- Creates escalating stakes over time

## Program Flow

### Startup Sequence
1. **Initialization**: Load configuration and initialize colored output
2. **Prompt Optimization**: Load and conditionally condense system prompts within 5K token budget
3. **Memory Recovery**: Restore conversation history and tension state from `memory.json`
4. **System Ready**: Display startup message and enter main interaction loop

### Enhanced Main Loop
1. **User Input**: Multi-line input collection with double-enter submission
2. **Tension Analysis**: Every 15 messages, comprehensive tension evaluation including:
   - Resource loss detection via LLM analysis
   - Pressure floor ratcheting calculations
   - Commitment level updates
   - Antagonist response planning
3. **Memory Management**: LLM-powered condensation when approaching token limits
4. **Context Assembly**: Integration of optimized prompts + memory + tension context
5. **Story Generation**: MCP server communication with retry logic and error handling
6. **Response Processing**: Formatted output with word wrapping and memory persistence

## Advanced Features

### Token Budget Management
- **Intelligent Allocation**: Dynamic distribution based on content priority
- **Context Optimization**: Automatic prompt condensation preserving functionality
- **Memory Efficiency**: Smart condensation targeting oldest, least relevant content
- **Performance Monitoring**: Real-time token usage tracking in debug mode

### Error Resilience
- **Network Resilience**: Automatic retry with exponential backoff
- **Graceful Degradation**: Safe fallbacks when analysis systems fail
- **State Recovery**: Robust handling of corrupted or invalid tension states
- **Debug Transparency**: Comprehensive logging for troubleshooting

### Extensibility Architecture
- **Modular Design**: Clean separation between tension system and core functionality
- **Plugin-Ready**: Framework supports additional analysis systems
- **Configurable Parameters**: Easy adjustment of tension thresholds and timing
- **Custom Prompts**: Full support for user-modified system prompts

## Commands

- `/quit` - Exit the application gracefully (saves all state)

## File Structure

```
aurora-rpg-client/
├── aurora3.tn1.py          # Main application with Unified Tension System
├── memory.json             # Persistent conversation memory (auto-created)
├── critrules.prompt        # Game Master rules with tension integration
├── companion.prompt        # Companion/NPC character definitions
├── lowrules.prompt         # Narrative generation guidelines
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

## Configuration

### Model Settings
- **MCP URL**: `http://127.0.0.1:3456/chat`
- **Model**: `qwen2.5:14b-instruct-q4_k_m`
- **Timeout**: 300 seconds with exponential backoff retry
- **Context Window**: 32,000 tokens total capacity

### Tension System Parameters
- **Analysis Frequency**: Every 15 user/assistant message exchanges
- **Analysis Token Budget**: 8,000 tokens maximum per evaluation
- **Pressure Scaling**: Logarithmic progression with four distinct ranges
- **Ratchet Mechanics**: Cumulative pressure floor increases

### Performance Tuning
- **Memory Fraction**: 70% of available tokens for conversation history
- **User Input Limit**: 30% of available tokens for current input
- **Token Estimation**: ~4 characters per token approximation

## Technical Innovation

### LLM-First Architecture
The Aurora RPG Client TN1 pioneeres an "LLM-First" approach where artificial intelligence handles complex semantic decisions rather than relying on rigid programmatic rules:

- **Semantic Memory Management**: AI determines content importance for condensation
- **Contextual Antagonist Generation**: Dynamic opponent creation based on story context  
- **Intelligent Resource Analysis**: LLM detection of narrative events affecting tension
- **Adaptive Prompt Optimization**: AI-driven compression preserving functionality

### Unified vs. Separated Systems
Traditional RPG systems treat exploration and combat as distinct modes requiring separate mechanical frameworks. The Unified Tension System eliminates this artificial separation by:

- **Continuous Tension Tracking**: Single pressure metric across all narrative phases
- **Organic Escalation**: Natural progression without mode switching
- **Contextual Manifestation**: Same underlying tension appears as exploration, social conflict, or combat
- **Player-Driven Pacing**: Tension responds to player actions rather than predetermined triggers

## Performance & Scalability

### System Requirements
- **RAM**: 8GB minimum for Qwen2.5 14B model
- **Storage**: 10GB for model files, minimal for application data
- **CPU**: Modern multi-core processor recommended
- **Network**: Local-only communication (no internet required)

### Optimization Features
- **Local Processing**: Complete privacy with no cloud dependencies
- **Efficient Tokenization**: Smart context management for optimal performance
- **Incremental State**: Only analyze tension when meaningful changes occur
- **Batched Operations**: Minimize MCP server round trips

## Contributing & Development

This project focuses on leveraging LLM capabilities for semantic understanding rather than programmatic approaches. Key development principles:

- **LLM-First Design**: Use language model intelligence for complex decisions
- **Structured Integration**: Combine AI creativity with programmatic reliability  
- **User Experience Priority**: Seamless, immersive gameplay above technical complexity
- **Token Efficiency**: Smart resource management for optimal performance

### Future Development Roadmap
- **Multi-Antagonist Support**: Complex scenarios with multiple opposing forces
- **Enhanced Commitment Mechanics**: More sophisticated escalation patterns
- **Performance Optimization**: Reduced latency and improved efficiency
- **Extended Model Support**: Compatibility with additional LLM architectures

## License

GNU Affero General Public License v3.0 - See LICENSE file for details.

## Troubleshooting

### Common Issues
1. **"Prompt file not found"**: Ensure all three .prompt files exist in the project directory
2. **Connection timeout**: Verify Ollama MCP server is running on port 3456
3. **Model not found**: Run `ollama pull qwen2.5:14b-instruct-q4_k_m` to download the required model
4. **Memory errors**: Check available RAM for the 14B parameter model

### Debug Mode
Run with `--debug` flag for comprehensive logging including:
- Token usage breakdowns
- Tension analysis details  
- Prompt optimization results
- MCP server communication logs
- Memory management operations

For additional support and advanced configuration options, consult the source code documentation and inline comments.