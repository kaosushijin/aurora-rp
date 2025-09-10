# DevName RPG Client

A terminal-based RPG storytelling client that transcends traditional language model limitations through **intelligent semantic processing** and **emergent narrative generation**. Built with a sophisticated multi-threaded architecture, DevName creates truly dynamic stories that evolve organically based on player behavior and semantic content analysis.

## 🧠 Breakthrough: Semantic-Driven Emergent Storytelling

Unlike conventional AI RPG systems that rely on simple prompt engineering, DevName implements **two complementary intelligence layers** that work together to create emergent storylines:

### Enhanced Memory Manager (EMM) - The Story Brain
The EMM doesn't just store conversation history—it **understands** it. Using advanced LLM semantic analysis, every message is automatically categorized into six semantic types:

- **story_critical** (90% preservation): Plot developments, character deaths, world-changing events
- **character_focused** (80% preservation): Relationship changes, personality reveals, character development  
- **relationship_dynamics** (80% preservation): Trust, betrayal, alliances between characters
- **emotional_significance** (75% preservation): Dramatic moments, conflict resolution
- **world_building** (70% preservation): Locations, lore, cultural discoveries
- **standard** (40% preservation): General interactions, routine activities

This semantic understanding means that when memory needs to be condensed, **the system knows what matters**. Critical story elements are preserved while routine interactions are compressed, maintaining narrative continuity across extended gameplay sessions.

### Story Momentum Engine (SME) - The Narrative Director
The SME acts as an invisible narrative director, analyzing player behavior patterns and dynamically generating story pressure:

**Immediate Pattern Recognition**: Real-time analysis of player input for conflict, exploration, social interaction, mystery, tension, and resolution patterns provides instant narrative feedback.

**Comprehensive LLM Analysis Every 15 Messages**: Deep narrative analysis that:
- Generates contextually appropriate antagonists with detailed motivations
- Tracks resource losses and escalates antagonist commitment levels (testing → engaged → desperate → cornered)
- Implements pressure floor ratcheting that prevents infinite narrative stalling
- Adapts story arc progression (Setup → Rising Action → Climax → Resolution)

## 🎭 How Emergent Storylines Overcome AI Limitations

Traditional AI RPGs suffer from:
- **Memory Loss**: Important events forgotten as conversation grows
- **Repetitive Patterns**: Same antagonists and conflicts recycled
- **Narrative Stagnation**: Stories that lose momentum and direction
- **Context Collapse**: Rich world-building lost over time

DevName solves these through **intelligent emergence**:

### 1. Semantic Memory Preservation
The EMM's 6-category semantic analysis ensures that:
- Character relationships evolve naturally and are remembered
- World-building accumulates into rich, consistent lore  
- Emotional story beats maintain their impact over time
- Plot threads remain coherent across long sessions

### 2. Dynamic Antagonist Evolution  
The SME generates antagonists that:
- Emerge naturally from story context rather than being randomly introduced
- Have believable motivations tied to player actions
- Escalate their commitment based on actual narrative pressure
- Lose resources and adapt tactics based on player successes

### 3. Pressure Floor Ratcheting
Prevents the common AI problem of tension deflation:
- Story pressure can only decrease so far before hitting a "floor"
- Each escalation raises this floor permanently
- Prevents infinite stalling tactics and maintains forward momentum
- Creates genuine stakes and consequences

### 4. Contextual Narrative Intelligence
Every GM response is enhanced with:
- Current pressure level and story arc awareness
- Active antagonist status and commitment level
- Recent semantic patterns and player behavioral analysis  
- Rich story context that informs appropriate narrative responses

## 💻 Technical Architecture

### Modular Design with Background Processing
```
┌─ main.py ──────────────────────────────────────────┐
│  • Prompt loading with LLM-powered condensation    │
│  • Token budget management (32K context window)    │
│  • Application lifecycle and signal handling       │
└─────────────────────────────────────────────────────┘
                           │
┌─ nci.py ───────────────────────────────────────────┐
│  • Dynamic coordinate system (responsive layout)   │
│  • Complete LLM integration coordination           │
│  • Multi-line input with intelligent submission    │
│  • Background thread coordination                  │
└─────────────────────────────────────────────────────┘
                           │
┌─ EMM ──────────┐   ┌─ SME ──────────┐   ┌─ MCP ──────┐
│ Semantic       │   │ Narrative      │   │ Model      │
│ Analysis       │   │ Pressure       │   │ Communication│
│                │   │ Management     │   │ Protocol   │
│ • 6-category   │   │                │   │            │
│   classification│   │ • Pattern      │   │ • Context  │
│ • Multi-pass   │   │   recognition  │   │   integration│
│   condensation │   │ • Antagonist   │   │ • Story    │
│ • Background   │   │   generation   │   │   context  │
│   auto-save    │   │ • Pressure     │   │   injection│
│                │   │   ratcheting   │   │            │
└────────────────┘   └────────────────┘   └────────────┘
```

### Key Features

**🧵 Background Processing**: All LLM operations (semantic analysis, momentum calculation, antagonist generation) run in background threads, keeping the interface responsive.

**🎯 Dynamic Coordinates**: Responsive terminal layout that adapts to any screen size, eliminating display issues across different terminals.

**🛡️ Defensive Error Handling**: 5-strategy JSON parsing prevents LLM response failures, ensuring reliable operation even with inconsistent model outputs.

**💾 Intelligent Persistence**: Complete story state (semantic categories, momentum data, antagonist details) preserved across sessions with atomic file operations.

## 🚀 Installation

### Requirements
- Python 3.8+
- httpx library (`pip install httpx`)
- Terminal with ncurses support (built-in on Unix/Linux/macOS)

### Setup
1. Clone the repository
2. Install dependencies: `pip install httpx colorama`
3. Ensure MCP server running on `127.0.0.1:3456` with `qwen2.5:14b-instruct-q4_k_m`
4. Create `critrules.prompt` file (required - see prompt templates)
5. Run: `python main.py`

### Prompt System
- **critrules.prompt** (REQUIRED): Core game master rules and behavior
- **companion.prompt** (OPTIONAL): Character definitions for companions  
- **lowrules.prompt** (OPTIONAL): Narrative generation guidelines

The system includes intelligent prompt condensation - if your prompts exceed the 5,000 token budget, they'll be automatically optimized using LLM compression while preserving functionality.

## 🎮 Usage

### Basic Commands
- **Multi-line Input**: Type naturally, double-enter to submit
- **Navigation**: PgUp/PgDn for scrolling, Home/End for quick navigation
- **Commands**: `/help`, `/stats`, `/analyze`, `/theme <name>`, `/clearmemory`

### Advanced Features
- **Semantic Analysis**: Watch your story evolve as the EMM categorizes interactions
- **Momentum Tracking**: Monitor narrative pressure through the status bar
- **Background Processing**: LLM analysis happens seamlessly without blocking gameplay
- **Complete State Persistence**: Return to exactly where you left off

### Debug Mode
Run with `python main.py --debug` for comprehensive logging of:
- Semantic categorization decisions
- Momentum analysis results  
- Antagonist generation process
- Background thread operations
- Token budget management

## 🎨 Interface

The interface features:
- **Responsive Design**: Adapts to any terminal size (minimum 80x24)
- **Theme Support**: Classic, dark, and bright color schemes
- **Multi-line Input**: Natural text entry with cursor navigation
- **Real-time Status**: Shows message count, story pressure, active prompts
- **Scroll Indicators**: Clear navigation feedback during history browsing

## 📊 System Monitoring

The `/stats` command provides detailed insight into:
- **Memory Usage**: Token utilization, semantic categories, condensation statistics
- **Story Momentum**: Current pressure, antagonist status, analysis cycles  
- **Background Threads**: Active LLM processing operations
- **File Operations**: Auto-save status, backup availability

## 🧪 Advanced Capabilities

### Semantic Memory Management
- **Multi-pass Condensation**: Up to 3 passes with increasing aggressiveness
- **Category-aware Preservation**: Different retention rates for different content types
- **Context Window Optimization**: Maintains coherence while fitting token constraints

### Story Momentum Features  
- **Dynamic Antagonist Generation**: Context-appropriate villains with detailed motivations
- **Commitment Escalation**: Antagonists become more dangerous as they lose resources
- **Pressure Floor System**: Prevents narrative backsliding and stagnation
- **Behavioral Pattern Analysis**: Adapts to player exploration, combat, and social preferences

### Background Processing
- **Non-blocking Operations**: All LLM calls happen in background threads
- **Thread Safety**: Proper locking ensures data integrity
- **Graceful Degradation**: System continues operating if individual components fail

## 🤝 Contributing

This codebase demonstrates advanced patterns in:
- **Semantic AI Integration**: LLM-powered content understanding and categorization
- **Dynamic Story Generation**: Emergent narrative systems that overcome AI limitations  
- **Responsive Terminal Interfaces**: Modern ncurses applications with dynamic layouts
- **Background Processing**: Non-blocking AI operations in interactive applications
- **Defensive Programming**: Robust handling of unreliable LLM responses

## 📝 License

GNU Affero General Public License v3.0 - See LICENSE file for details.

---

*DevName RPG Client represents a breakthrough in AI-powered interactive storytelling, moving beyond simple chatbots to create truly emergent narratives that evolve organically through intelligent semantic processing and dynamic story momentum management.*