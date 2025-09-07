# Aurora RPG Client

A terminal-based RPG storyteller client that leverages Large Language Model capabilities through MCP (Model Control Protocol) to create immersive, interactive high-fantasy narratives powered by the innovative Story Momentum Engine and intelligent memory management.

## Features

### Story Momentum Engine (SME)
Aurora's flagship feature - a sophisticated narrative progression system that creates compelling, evolving stories:

- **Unified Analysis System**: Single function handles both first-time and ongoing momentum analysis
- **Intelligent Antagonist Generation**: AI creates contextually appropriate villains with detailed motivations and backgrounds
- **Antagonist Persistence**: Villains evolve with commitment levels (testing → engaged → desperate → cornered)
- **Resource Loss Tracking**: Monitors antagonist setbacks and adapts behavior accordingly
- **Pressure Ratcheting**: Mathematical floor system prevents infinite stalling tactics
- **15-Message Analysis Cycles**: Automated momentum evaluation for optimal story pacing
- **Quality Validation**: Ensures compelling, well-developed antagonists through multi-attempt generation

### Advanced Memory Management
- **Semantic Content Categorization**: AI categorizes conversation into four types with tailored preservation:
  - **Story-critical**: 80% preservation (plot developments, key decisions, world-changing events)
  - **Character-focused**: 70% preservation (relationships, personality reveals, character development)
  - **World-building**: 60% preservation (locations, lore, cultural information, discoveries)
  - **Standard**: 40% preservation (general interactions, travel, routine activities)
- **Progressive Condensation**: Multiple passes with increasing compression based on content age
- **Intelligent Context Retention**: Maintains story continuity across extended gameplay sessions
- **Batch Processing**: Efficient 10-message batches for LLM-powered analysis

### Robust System Architecture
- **Graceful Prompt Handling**: System continues operating even with missing prompt files
- **Intelligent Prompt Optimization**: Automatic condensation of oversized prompts while preserving functionality
- **Error Recovery**: Comprehensive retry logic with exponential backoff for network issues
- **Token Budget Management**: Optimized 32k context window allocation with real-time monitoring
- **Debug Transparency**: Extensive logging system for development and troubleshooting

### Interactive Storytelling
- **Multi-line Input System**: Natural conversation flow with double-enter submission
- **Flexible Companion System**: Support for multiple companion characters via prompt files
- **Rich Narrative Context**: Momentum-aware system prompts that adapt to story state
- **Input Validation**: Prevents token budget overflow with helpful user feedback

## Installation

### Requirements
- Python 3.10+
- MCP server running on `127.0.0.1:3456`
- Qwen2.5 14B Instruct model (`qwen2.5:14b-instruct-q4_k_m`)

### Dependencies
```bash
pip install httpx colorama
```

### Setup
1. Ensure your MCP server is running with the configured model
2. Create prompt files (system will warn about missing files but continue operating):
   - `critrules.prompt` - **Required**: Game Master core rules and behavior
   - `companion.prompt` - Optional: Companion character definition
   - `lowrules.prompt` - Optional: Narrative generation guidelines
3. Run the client:
```bash
python aurora3.sme.py
```

For comprehensive debug logging:
```bash
python aurora3.sme.py --debug
```

## Program Flow

### Startup Sequence
1. **Configuration Validation**: Verify token allocation within 32k context window
2. **Prompt Loading & Optimization**:
   - Load available prompt files with graceful missing file handling
   - Calculate combined token usage
   - Apply intelligent condensation if exceeding 5k token budget
   - Display warnings for missing files while continuing operation
3. **Critical Validation**: Ensure `critrules.prompt` exists (required for GM functionality)
4. **Memory Restoration**: Load previous conversation with semantic categories
5. **Momentum Analysis**: Check if SME analysis needed (every 15 messages)
6. **Memory Optimization**: Trigger intelligent management if approaching token limits
7. **Session Display**: Show current momentum state, active antagonist, and prompt status

### Story Momentum Engine Cycle
The SME operates on a 15-message cycle to maintain optimal story pacing:

1. **Trigger Analysis**: Every 15 user/assistant message exchanges
2. **Context Preparation**: Extract recent conversation within 6k token budget (25% reserved for instructions)
3. **Antagonist Management**:
   - Generate high-quality antagonist if first analysis or missing
   - Validate existing antagonist quality
   - Regenerate if quality validation fails
4. **Resource Loss Detection**: Analyze recent events for antagonist setbacks
5. **Pressure Floor Calculation**: Apply ratcheting mechanism (prevents infinite retreats)
6. **Momentum Analysis**: LLM evaluates:
   - Narrative pressure level (0.0-1.0 scale)
   - Pressure source (antagonist/environment/social/discovery)
   - Manifestation type (exploration/tension/conflict/resolution)
   - Player behavioral patterns
   - Antagonist response strategies
7. **State Validation**: Sanitize and save updated momentum state
8. **Context Integration**: Generate momentum-aware system prompts for next interactions

### Main Interaction Loop
1. **Multi-line Input Collection**: Gather user input until double-enter
2. **Input Validation**: Check token limits with helpful feedback
3. **Memory Addition**: Store user input with timestamp
4. **Momentum Context Generation**: Create dynamic system prompts based on current story state
5. **Message Assembly**: Combine system prompts, memory, and current input
6. **Response Generation**: Call MCP with retry logic and error handling
7. **Response Processing**: Display, store, and trigger follow-up analysis
8. **Periodic Maintenance**: Memory management every 50 messages

### Token Budget Allocation
**Total Context Window**: 32,000 tokens
- **System Prompts**: 5,000 tokens (auto-optimized)
- **Momentum Analysis**: 6,000 tokens (SME operations)
- **Memory Storage**: 14,700 tokens (semantic categorization)
- **User Input**: 6,300 tokens (validated)

## Configuration

### Model Settings
- **MCP URL**: `http://127.0.0.1:3456/chat`
- **Model**: `qwen2.5:14b-instruct-q4_k_m`
- **Timeout**: 300 seconds
- **Retry Logic**: 3 attempts with exponential backoff

### Story Momentum Engine
- **Analysis Frequency**: Every 15 message exchanges
- **Pressure Scale**: 0.0-1.0 with named ranges (low/building/critical/explosive)
- **Commitment Progression**: testing → engaged → desperate → cornered
- **Pressure Floor**: Ratcheting system with 0.02 increment per escalation
- **Quality Validation**: Multi-attempt antagonist generation with fallback

### Memory Management
- **Categorization**: Automatic semantic analysis in 20-message chunks
- **Preservation Ratios**: Content-type specific (40%-80%)
- **Condensation Thresholds**: Age-based triggers (40-100 messages)
- **Batch Processing**: 10-message groups for efficient LLM calls

## Commands & Usage

### Input Commands
- `quit` - Exit application gracefully
- Double-enter - Submit multi-line input
- `--debug` flag - Enable comprehensive logging

### File Structure
- `aurora3.sme.py` - Main application with Story Momentum Engine
- `memory.json` - Persistent conversation memory (auto-created)
- `critrules.prompt` - **Required**: Game Master rules
- `companion.prompt` - Optional: Companion character definition
- `lowrules.prompt` - Optional: Narrative generation guidelines

## Technical Innovation

### Story Momentum Engine Breakthroughs
The SME represents a paradigm shift in AI-powered narrative progression:

- **Unified State Management**: Single analysis function eliminates complexity
- **Antagonist Quality Assurance**: Multi-attempt generation ensures compelling villains
- **Mathematical Pressure Modeling**: Prevents common AI storytelling pitfalls
- **Organic Conflict Resolution**: Sophisticated outcomes beyond simple win/lose
- **Contextual Adaptability**: System responds to player behavior patterns

### Memory System Architecture
- **Semantic Intelligence**: LLM categorizes content by narrative importance
- **Adaptive Preservation**: Different ratios optimize for content type
- **Progressive Compression**: Multiple passes with increasing aggressiveness
- **Continuity Maintenance**: Story coherence across extended sessions

### Error Handling & Reliability
- **Graceful Degradation**: System continues operating with missing components
- **Network Resilience**: Comprehensive retry logic for unreliable connections
- **Input Protection**: Token budget validation prevents system overflow
- **State Recovery**: Robust fallback mechanisms for analysis failures

## Development Features

### Debug Capabilities
Enable comprehensive logging with `--debug` flag:
- Token allocation breakdown
- Momentum analysis context preparation
- Antagonist generation attempts
- Memory categorization results
- Prompt condensation statistics
- Network retry attempts

### Error Messages & Warnings
- Missing prompt file notifications
- Token budget violations with specific guidance
- Memory optimization triggers
- Momentum analysis failures with fallbacks

### System Status Display
Real-time information during operation:
- Current momentum pressure level
- Active antagonist name and status
- Available prompt components
- Memory optimization results

## Contributing

Aurora demonstrates advanced LLM integration principles:

- **Semantic Partnership**: AI handles complex narrative decisions intelligently
- **Structured Reliability**: Combines creativity with programmatic robustness
- **Resource Optimization**: Efficient token usage within hardware constraints
- **User Agency Preservation**: Maintains meaningful player choice throughout

## License

GNU Affero General Public License v3.0 - See LICENSE file for details.

---

**Note**: Aurora requires the `critrules.prompt` file to function properly. The system will display an error and exit if this critical file is missing. Companion and narrative prompts are optional but enhance the experience.