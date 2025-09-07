# Aurora RPG Client

A terminal-based RPG storyteller client that leverages Large Language Model capabilities through MCP (Model Control Protocol) to create immersive, interactive high-fantasy narratives with intelligent memory management and dynamic prompt optimization.

## Features

### Core Functionality
- **Interactive Storytelling**: Multi-line input system for natural conversation with an AI Game Master
- **Persistent Memory**: JSON-based conversation memory with intelligent LLM-powered condensation
- **Modular System Prompts**: Three-part prompt system for flexible Game Master behavior
- **Intelligent Prompt Optimization**: Automatic condensation of oversized prompt files while preserving functionality
- **Token Budget Management**: Sophisticated context window management (32k tokens) with dynamic allocation
- **Companion Character**: Aurora, a detailed bard companion with rich personality and musical abilities

### Advanced Memory Management
- **LLM-Driven Condensation**: Uses the language model itself to intelligently compress memory while preserving story continuity
- **Context-Aware Retention**: Maintains important story elements, character relationships, and plot threads
- **Automatic Optimization**: Triggers condensation only when memory approaches token limits
- **Seamless Operation**: Memory management happens transparently without interrupting gameplay

### Intelligent Prompt System
- **Dynamic Optimization**: Automatically condenses oversized prompt files using LLM intelligence
- **Isolated Processing**: Each prompt category (Game Master rules, companion definition, narrative guidelines) processed separately
- **Functionality Preservation**: Maintains prompt effectiveness while reducing token footprint
- **User Customization Support**: Allows verbose, detailed prompt files without performance concerns

## Installation

### Requirements
- Python 3.10+
- Active MCP server running on `127.0.0.1:3456`
- Language model supporting the configured model string

### Dependencies
```bash
pip install httpx colorama
```

### Setup
1. Ensure your MCP server is running with the configured model (`qwen2.5:14b-instruct-q4_k_m`)
2. Create or customize the prompt files:
   - `critrules.prompt` - Game Master core rules and behavior
   - `companion.prompt` - Aurora companion character definition
   - `lowrules.prompt` - Narrative generation guidelines
3. Run the client:
```bash
python aurora3.dv1.py
```

For debug mode with verbose logging:
```bash
python aurora3.dv1.py --debug
```

## Program Flow

### Startup Sequence
1. **Initialization**: Load configuration and initialize color output
2. **Prompt Optimization**: 
   - Load all three prompt files
   - Calculate combined token usage
   - Apply intelligent condensation if budget (5000 tokens) exceeded
   - Each prompt file condensed separately to preserve functional isolation
3. **Memory Loading**: Restore previous conversation memory from `memory.json`
4. **System Ready**: Display startup message and enter main interaction loop

### Main Interaction Loop
1. **User Input**: Multi-line input system (press ENTER twice to submit)
2. **Memory Management**: 
   - Add user input to persistent memory
   - Check total memory token usage
   - Apply LLM-powered condensation if approaching limits
3. **Context Assembly**:
   - Combine optimized system prompts
   - Add conversation memory
   - Include current user input
4. **LLM Communication**: Send context to MCP server with retry logic
5. **Response Processing**: 
   - Receive and display AI response with word wrapping
   - Add response to persistent memory
   - Save updated memory to disk

### Token Budget Management
- **Total Context Window**: 32,000 tokens
- **System Prompts**: 5,000 tokens (automatically optimized)
- **Memory**: ~18,900 tokens (70% of remaining space)
- **User Input**: ~8,100 tokens (30% of remaining space)

## Configuration

### Model Settings
- **MCP URL**: `http://127.0.0.1:3456/chat`
- **Model**: `qwen2.5:14b-instruct-q4_k_m`
- **Timeout**: 300 seconds
- **Retry Logic**: 3 attempts with exponential backoff

### Token Management
- **Context Window**: 32,000 tokens
- **System Prompt Budget**: 5,000 tokens
- **Memory Fraction**: 70% of remaining tokens
- **Token Estimation**: ~4 characters per token

## Commands
- `/quit` - Exit the application gracefully

## File Structure
- `aurora3.dv1.py` - Main application file
- `memory.json` - Persistent conversation memory (auto-created)
- `critrules.prompt` - Game Master rules and behavior
- `companion.prompt` - Aurora companion character definition  
- `lowrules.prompt` - Narrative generation guidelines

## Technical Architecture

### Intelligent Prompt Condensation
The system automatically optimizes oversized prompt files using the same LLM that powers the storytelling:

- **Trigger Condition**: Combined prompt files exceed 5,000 token budget
- **Isolated Processing**: Each prompt file condensed separately to prevent cross-contamination
- **Functionality Preservation**: LLM instructed to maintain prompt effectiveness while reducing size
- **Fallback Protection**: Original prompts used if condensation fails

### Memory Management
- **LLM-Powered Analysis**: Uses language model intelligence for semantic importance assessment
- **Context Preservation**: Maintains story continuity and character relationships
- **Automatic Triggering**: Activates only when approaching token limits
- **Transparent Operation**: No user intervention required

### Error Handling
- **Network Resilience**: Automatic retry with exponential backoff for connection issues
- **Graceful Degradation**: Fallback to original content if optimization fails
- **Debug Support**: Comprehensive logging when `--debug` flag used

## Development Roadmap

### Priority 1: Combat Phase Detection & Management
- **Intelligent Phase Recognition**: LLM-powered detection of combat vs. roleplaying contexts
- **Two-Stage System**: Analysis phase followed by appropriate content generation
- **Structured Combat Resolution**: JSON-based tracking to ensure conflicts reach satisfying conclusions
- **Dynamic Threat Generation**: Contextually appropriate enemies based on environment and story

### Priority 2: Enhanced Semantic Memory Management
- **Advanced Importance Scoring**: More sophisticated LLM-driven memory prioritization
- **Structured Analysis**: JSON-based memory evaluation for consistent decision-making
- **Context-Aware Retention**: Phase-aware memory management for optimal story continuity

### Priority 3: Game Master Prompt Refinement
- **Iterative Optimization**: Systematic testing and improvement of GM behavior
- **Phase Integration**: Ensure prompts work effectively across different game phases
- **User Customization**: Better support for user-modified prompt files

### Future Enhancements
- **Multi-Enemy Combat**: Support for complex encounters with group dynamics
- **Advanced Combat Features**: Enhanced tactical systems and resolution mechanics
- **Performance Optimization**: Improved efficiency and response times
- **Extended Character Support**: Additional companion options beyond Aurora

## Contributing

This project focuses on leveraging LLM capabilities for semantic understanding rather than programmatic approaches. Key principles:

- **LLM-First Design**: Use language model intelligence for complex decisions
- **Structured Integration**: Combine LLM creativity with programmatic reliability
- **User Experience**: Prioritize seamless, immersive gameplay
- **Token Efficiency**: Smart resource management for optimal performance

## License

GNU Affero General Public License v3.0 - See LICENSE file for details.