# DevName RPG Client - Remodularization Holistic Plan
================================================================

## Executive Summary

This document provides a comprehensive plan for remodularizing the DevName RPG Client from the current fragmented architecture to a clean "hub and spoke" system. The remodularization addresses over-responsibility in `nci.py`, consolidates fragmented UI utilities, centralizes semantic logic, and maintains strict architectural boundaries while preserving all existing functionality.

## Current State Analysis (Based on Repository Code)

### Existing Architecture Problems

**1. nci.py Over-Responsibility**
- Currently serves as both UI controller AND main orchestrator
- Contains 500+ lines mixing UI logic with business coordination
- Directly manages `emm.py`, `sme.py`, and `mcp.py` interactions
- Violates single responsibility principle

**2. Fragmented UI Components**
- 5 separate `nci_*.py` files for related functionality
- `nci_terminal.py`: BoxCoordinates, LayoutGeometry, TerminalManager
- `nci_input.py`: MultiLineInput, InputValidator
- `nci_scroll.py`: ScrollManager
- `nci_display.py`: DisplayMessage formatting
- `nci_colors.py`: ColorManager, ColorTheme

**3. Scattered Semantic Logic**
- LLM-powered condensation in `emm.py`
- Story momentum analysis in `sme.py`
- No centralized coordination of semantic operations

## Target Architecture (Hub and Spoke)

### Core Design Principles

**Hub and Spoke Model**
- `orch.py` is the central hub containing main program logic
- All other modules are spokes providing focused services
- **CRITICAL**: Only `orch.py` communicates with `mcp.py` for LLM requests
- No spoke-to-spoke communication

**Program Flow**
```
main.py → DevNameRPGClient → orch.py (main loop) ↔ {service modules}
```

## New Module Architecture

### **orch.py** - Main Orchestrator (NEW)

**Purpose**: Central hub containing the main program logic

**Core Responsibilities**:
```python
class Orchestrator:
    def __init__(self, config: Dict[str, Any], debug_logger=None)
    def initialize_modules(self) -> bool
    def run_main_loop(self) -> int  # Contains core program logic
    def process_user_input(self, input_text: str) -> bool
    def gather_context_for_llm(self) -> Dict[str, Any]
    def send_llm_request(self, request_data: Dict[str, Any]) -> Optional[str]
    def handle_llm_response(self, response: str) -> bool
    def trigger_periodic_analysis(self) -> None
    def update_memory_and_state(self, data: Dict[str, Any]) -> None
    def shutdown_gracefully(self) -> None
```

**Main Program Logic Flow**:
1. Initialize all service modules
2. Start UI through `ncui.py`
3. **Main Loop**:
   - Get user input from `ncui.py`
   - Gather conversation context from `emm.py`
   - Gather story state from `sme.py`
   - Prepare LLM request with prompts + context
   - Send request via `mcp.py`, receive response
   - Display response via `ncui.py`
   - Every 15 messages: call `sem.py` for analysis
   - Update `emm.py` and `sme.py` with new data

**Data Interfaces**:
```python
# Configuration passed from main.py
config_structure = {
    "prompts": {
        "critrules": str,
        "companion": Optional[str], 
        "lowrules": Optional[str]
    },
    "mcp": {
        "server_url": str,
        "model": str,
        "timeout": int
    },
    "debug_enabled": bool
}

# Context gathering format
context_data = {
    "conversation_history": List[Dict[str, str]],  # From emm.py
    "story_state": Dict[str, Any],                 # From sme.py
    "prompts": Dict[str, str],                     # Internal
    "user_input": str                              # From ncui.py
}
```

### **ncui.py** - UI Controller (REPLACES nci.py)

**Purpose**: Pure UI management without business logic

**Responsibilities**:
```python
class NCursesUIController:
    def __init__(self, ui_lib, debug_logger=None)
    def initialize_interface(self) -> bool
    def get_user_input(self) -> Optional[str]  # Returns to orch.py
    def display_message(self, content: str, msg_type: str) -> None
    def display_system_status(self, status_data: Dict[str, Any]) -> None
    def handle_resize(self) -> None
    def shutdown_interface(self) -> None
```

**Data Interface with orch.py**:
```python
# Message display format
message_data = {
    "content": str,
    "msg_type": str,  # "user", "assistant", "system", "error"
    "timestamp": Optional[str]
}

# Status display format  
status_data = {
    "message_count": int,
    "memory_usage": float,
    "story_pressure": float,
    "active_prompts": List[str],
    "background_processing": bool
}
```

### **uilib.py** - Consolidated UI Library (CONSOLIDATES nci_*.py)

**Purpose**: All UI utilities and components in one cohesive library

**Consolidated Components**:
```python
# From nci_terminal.py
@dataclass
class BoxCoordinates:
    outer_y: int
    outer_x: int  
    outer_height: int
    outer_width: int
    inner_y: int
    inner_x: int
    inner_height: int
    inner_width: int

class TerminalManager:
    def get_box_layout(self) -> Optional[LayoutGeometry]
    def check_resize(self) -> Tuple[bool, int, int]
    def is_too_small(self) -> bool

# From nci_input.py  
class MultiLineInput:
    def __init__(self, max_lines: int = 10)
    def handle_key(self, key: int) -> Tuple[bool, str]
    def insert_newline(self) -> None
    def handle_backspace(self) -> bool

# From nci_scroll.py
class ScrollManager:
    def __init__(self, content_height: int)
    def scroll_up(self, lines: int = 1) -> None
    def scroll_down(self, lines: int = 1) -> None
    def page_up(self) -> None
    def page_down(self) -> None

# From nci_display.py
class DisplayMessage:
    def __init__(self, content: str, msg_type: str, timestamp: str = None)
    def format_for_display(self, max_width: int = 80) -> List[str]

# From nci_colors.py  
class ColorManager:
    def __init__(self, theme: ColorTheme)
    def switch_theme(self, theme_name: str) -> bool
    def get_color_pair(self, msg_type: str) -> int
```

**Data Format Standards**:
```python
# Message formatting standards
MESSAGE_TYPES = {
    "user": {"prefix": "You", "color": "user_color"},
    "assistant": {"prefix": "GM", "color": "assistant_color"}, 
    "system": {"prefix": " ", "color": "system_color"},
    "error": {"prefix": "Error", "color": "error_color"}
}

# Layout calculation standards
LAYOUT_PROPORTIONS = {
    "output_ratio": 0.90,
    "input_ratio": 0.10,
    "min_terminal_width": 80,
    "min_terminal_height": 24
}
```

### **sem.py** - Semantic Logic (NEW)

**Purpose**: Centralized semantic analysis functions called by orchestrator

**Extracted Functionality**:
```python
class SemanticProcessor:
    def __init__(self, debug_logger=None)
    
    # From emm.py semantic logic
    def categorize_message(self, content: str) -> str
    def condense_content(self, messages: List[Dict], target_tokens: int) -> Optional[str]
    def prepare_condensation_request(self, messages: List[Dict], category: str) -> str
    
    # From sme.py analysis logic  
    def analyze_story_momentum(self, conversation: List[Dict]) -> Dict[str, Any]
    def prepare_momentum_analysis_request(self, context: Dict[str, Any]) -> str
    def detect_narrative_patterns(self, text: str) -> Dict[str, List[str]]
    def generate_antagonist_data(self, story_context: Dict[str, Any]) -> Optional[Dict[str, Any]]
    def prepare_antagonist_request(self, context: Dict[str, Any]) -> str
    def calculate_story_pressure(self, momentum_data: Dict[str, Any]) -> float
```

**Critical Architectural Rule**: 
- `sem.py` provides analysis functions and prepares LLM requests
- `sem.py` asks `orch.py` to send requests via `mcp.py`
- `sem.py` NEVER directly calls `mcp.py`

**Data Format Standards**:
```python
# Message categorization output
SEMANTIC_CATEGORIES = {
    "story_critical": {"preservation_ratio": 0.9, "priority": 1},
    "character_focused": {"preservation_ratio": 0.8, "priority": 2},
    "relationship_dynamics": {"preservation_ratio": 0.7, "priority": 3},
    "emotional_significance": {"preservation_ratio": 0.6, "priority": 4},
    "world_building": {"preservation_ratio": 0.5, "priority": 5},
    "standard": {"preservation_ratio": 0.4, "priority": 6}
}

# Story momentum analysis format
momentum_analysis = {
    "pressure_level": float,  # 0.0 to 1.0
    "pressure_source": str,   # "exploration", "combat", "social", "mystery"  
    "manifestation_type": str, # "environmental", "antagonist", "revelation"
    "escalation_recommended": bool,
    "antagonist_data": Optional[Dict[str, Any]],
    "narrative_pressure": float,
    "analysis_confidence": float
}

# Antagonist generation format
antagonist_data = {
    "name": str,
    "motivation": str,
    "methods": List[str],
    "resources": Dict[str, int],  # Resource types with quantities
    "commitment_level": float,    # 0.0 to 1.0
    "antagonist_type": str,       # "individual", "organization", "force_of_nature"
    "current_goal": str,
    "escalation_potential": float
}
```

### **emm.py** - Memory Storage (SIMPLIFIED)

**Purpose**: Pure data storage and retrieval without semantic logic

**Simplified Responsibilities**:
```python
class EnhancedMemoryManager:
    def __init__(self, memory_file: str = "memory.json", auto_save_enabled: bool = True)
    
    # Core storage operations (KEEPS)
    def add_message(self, content: str, message_type: MessageType) -> None
    def get_messages(self, limit: Optional[int] = None) -> List[Message]
    def get_conversation_for_mcp(self) -> List[Dict[str, str]]
    def save_conversation(self, filename: Optional[str] = None) -> bool
    def load_conversation(self, filename: str) -> bool
    
    # State management (KEEPS)
    def get_momentum_state(self) -> Optional[Dict[str, Any]]
    def update_momentum_state(self, state_data: Dict[str, Any]) -> None
    def get_memory_stats(self) -> Dict[str, Any]
    
    # File operations (KEEPS)
    def backup_memory_file(self, backup_filename: Optional[str] = None) -> bool
    def get_memory_file_info(self) -> Dict[str, Any]
```

**Data Format Standards**:
```python
# Message storage format
class Message:
    content: str
    message_type: MessageType  # USER, ASSISTANT, SYSTEM, MOMENTUM_STATE
    timestamp: str            # ISO format for file metadata
    token_estimate: int
    id: str                   # UUID for tracking
    content_category: str     # Set by sem.py categorization
    condensed: bool          # Flag for condensed content

# Message type enumeration
class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"
    MOMENTUM_STATE = "momentum_state"

# Memory statistics format
memory_stats = {
    "message_count": int,
    "total_tokens": int,
    "max_tokens": int,
    "utilization": float,     # total_tokens / max_tokens
    "condensations_performed": int,
    "oldest_message": str,    # ISO timestamp
    "newest_message": str,    # ISO timestamp
    "auto_save_enabled": bool,
    "memory_file": str
}
```

**Removed from emm.py** (moved to sem.py):
- `categorize_with_llm()` - semantic categorization logic
- `condense_conversation()` - content condensation logic
- `_call_llm()` - LLM communication logic

### **sme.py** - Story Engine (SIMPLIFIED)

**Purpose**: Story state management without semantic processing

**Simplified Responsibilities**:
```python
class StoryMomentumEngine:
    def __init__(self, debug_logger=None)
    
    # State management (KEEPS)
    def process_user_input(self, input_text: str) -> Dict[str, Any]
    def get_story_context(self) -> Dict[str, Any]
    def get_pressure_stats(self) -> Dict[str, Any]
    def force_pressure_level(self, new_pressure: float) -> None
    
    # State persistence (KEEPS)  
    def save_state_to_dict(self) -> Dict[str, Any]
    def load_state_from_dict(self, state_data: Dict[str, Any]) -> bool
    
    # Configuration (KEEPS)
    def update_analysis_cooldown(self, cooldown: float) -> None
    def update_pressure_thresholds(self, antagonist: float, climax: float) -> None
```

**Data Format Standards**:
```python
# Story context format
story_context = {
    "narrative_pressure": float,
    "pressure_source": str,
    "manifestation_type": str, 
    "antagonist_present": bool,
    "story_arc": str,          # "setup", "rising_action", "climax", "resolution"
    "narrative_state": str,
    "escalation_count": int,
    "pressure_floor": float
}

# Pressure statistics format
pressure_stats = {
    "current_pressure": float,
    "pressure_trend": str,     # "rising", "falling", "stable"
    "variance": float,
    "min_pressure": float,
    "max_pressure": float,
    "total_updates": int,
    "current_arc": str,
    "pressure_floor": float,
    "escalation_count": int
}

# State persistence format
sme_state = {
    "narrative_pressure": float,
    "pressure_source": str,
    "manifestation_type": str,
    "escalation_count": int,
    "base_pressure_floor": float,
    "last_analysis_count": int,
    "antagonist": Optional[Dict[str, Any]],
    "story_arc": str,
    "pressure_history": List[float],
    "timestamp": str
}
```

**Removed from sme.py** (moved to sem.py):
- `analyze_with_llm()` - comprehensive momentum analysis
- `_generate_antagonist()` - antagonist generation logic
- `_call_llm()` - LLM communication logic

### **mcp.py** - Communication Layer (UNCHANGED)

**Purpose**: HTTP communication with LLM services - no changes required

**Preserved Functionality**:
- 5-strategy JSON parsing for response reliability
- Context window management (32,000 tokens)
- Timeout and retry logic
- Message formatting for LLM requests

## Hub and Spoke Communication Patterns

### Strict Communication Rules

**1. Hub Control (orch.py)**
```python
# ONLY orch.py calls mcp.py
async def send_llm_request(self, request_data: Dict[str, Any]) -> Optional[str]:
    response = await self.mcp_client.send_message(request_data)
    return response

# Other modules request LLM calls through orch.py
def handle_semantic_analysis_request(self, sem_request: Dict[str, Any]) -> None:
    llm_response = await self.send_llm_request(sem_request)
    analysis_result = self.sem_processor.process_llm_response(llm_response)
    return analysis_result
```

**2. Service Communication (spoke modules)**
```python
# sem.py - Prepares requests for orch.py to send
def analyze_story_momentum(self, conversation: List[Dict]) -> Dict[str, Any]:
    # Perform local analysis
    local_analysis = self._analyze_patterns(conversation)
    
    # If LLM analysis needed, prepare request for orch.py
    if self._needs_llm_analysis(local_analysis):
        llm_request = self.prepare_momentum_analysis_request(conversation)
        return {"needs_llm": True, "request": llm_request, "local_analysis": local_analysis}
    
    return {"needs_llm": False, "analysis": local_analysis}

# orch.py handles the request
def process_semantic_analysis(self) -> None:
    analysis_request = self.sem_processor.analyze_story_momentum(conversation)
    
    if analysis_request.get("needs_llm"):
        llm_response = await self.send_llm_request(analysis_request["request"])
        final_analysis = self.sem_processor.process_llm_response(llm_response)
    else:
        final_analysis = analysis_request["analysis"]
    
    # Update modules with results
    self.sme.update_momentum_data(final_analysis)
```

**3. Data Flow Examples**
```python
# User input processing flow
def process_user_input(self, input_text: str) -> bool:
    # 1. Get conversation context
    conversation = self.emm.get_conversation_for_mcp()
    
    # 2. Get story state  
    story_state = self.sme.get_story_context()
    
    # 3. Prepare LLM request
    request_data = {
        "messages": conversation + [{"role": "user", "content": input_text}],
        "system_prompt": self._build_system_prompt(story_state),
        "max_tokens": 2000
    }
    
    # 4. Send to LLM (ONLY orch.py calls mcp.py)
    response = await self.mcp_client.send_message(request_data)
    
    # 5. Store and display response
    self.emm.add_message(input_text, MessageType.USER)
    self.emm.add_message(response, MessageType.ASSISTANT)
    self.ncui.display_message(response, "assistant")
    
    # 6. Update story state
    self.sme.process_user_input(input_text)
    
    # 7. Periodic analysis (every 15 messages)
    if self._should_trigger_analysis():
        self._trigger_semantic_analysis()
    
    return True
```

## Data Format Standardization

### Cross-Module Data Contracts

**1. Configuration Data (main.py → orch.py)**
```python
@dataclass
class ApplicationConfig:
    prompts: Dict[str, str]
    mcp_config: Dict[str, Any]
    debug_enabled: bool
    color_theme: str = "classic"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompts": self.prompts,
            "mcp": self.mcp_config,
            "debug_enabled": self.debug_enabled,
            "color_theme": self.color_theme
        }
```

**2. Message Data (standardized across all modules)**
```python
# Standard message format for storage and display
@dataclass
class StandardMessage:
    content: str
    msg_type: str         # "user", "assistant", "system", "error"
    timestamp: str        # ISO format
    token_estimate: int   # For memory management
    
    def to_mcp_format(self) -> Dict[str, str]:
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        return {"role": role_map.get(self.msg_type, "user"), "content": self.content}
    
    def to_display_format(self) -> Dict[str, Any]:
        return {"content": self.content, "msg_type": self.msg_type, "timestamp": self.timestamp}
```

**3. LLM Request/Response Format (orch.py ↔ mcp.py)**
```python
# Standardized LLM request format
@dataclass  
class LLMRequest:
    messages: List[Dict[str, str]]
    system_prompt: str
    max_tokens: int = 2000
    temperature: float = 0.8
    
    def to_mcp_format(self) -> Dict[str, Any]:
        return {
            "messages": [{"role": "system", "content": self.system_prompt}] + self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

# Standardized response processing
class LLMResponse:
    def __init__(self, raw_response: str):
        self.raw_content = raw_response
        self.parsed_content = self._parse_content()
        self.success = self.parsed_content is not None
    
    def _parse_content(self) -> Optional[str]:
        # Use mcp.py's 5-strategy parsing
        # This would be called through orch.py coordination
        pass
```

**4. Error Handling Standards**
```python
# Standardized error format across modules
@dataclass
class ModuleError:
    module_name: str
    error_type: str      # "connection", "parsing", "validation", "internal"
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_log_format(self) -> str:
        return f"[{self.timestamp}] {self.module_name}.{self.error_type}: {self.message}"
    
    def to_display_format(self) -> Dict[str, Any]:
        return {"content": f"{self.error_type}: {self.message}", "msg_type": "error"}
```

## Migration Implementation Plan

### Phase 1: Create New Modules (4-6 hours)

**Step 1.1: Create orch.py**
```python
# Extract orchestration logic from nci.py
# Implement hub communication patterns
# Add main program logic coordination
```

**Step 1.2: Create uilib.py**  
```python
# Consolidate all nci_*.py files
# Preserve all existing functionality
# Standardize data interfaces
```

**Step 1.3: Create sem.py**
```python
# Extract semantic logic from emm.py and sme.py
# Implement LLM request preparation (not sending)
# Add analysis coordination functions
```

### Phase 2: Refactor Existing Modules (3-4 hours)

**Step 2.1: Simplify nci.py → ncui.py**
```python
# Remove orchestration logic (move to orch.py)
# Keep only UI management
# Update to use uilib.py components
```

**Step 2.2: Simplify emm.py**
```python
# Remove semantic analysis logic (move to sem.py)
# Keep storage and state management
# Update interfaces for orch.py communication
```

**Step 2.3: Simplify sme.py**
```python
# Remove LLM analysis logic (move to sem.py)  
# Keep state management and basic pattern detection
# Update interfaces for orch.py communication
```

### Phase 3: Update Interconnections (2-3 hours)

**Step 3.1: Update main.py**
```python
# Initialize orch.py instead of nci.py
# Pass configuration to orchestrator
# Remove direct module management
```

**Step 3.2: Implement Hub Communication**
```python
# Ensure all LLM requests go through orch.py
# Implement service request patterns
# Add proper error handling
```

**Step 3.3: Remove Old Files**
```python
# Delete nci_*.py files after consolidation
# Update import statements
# Clean up unused code
```

### Phase 4: Testing and Validation (2-3 hours)

**Step 4.1: Functional Testing**
- Verify all original functionality preserved
- Test background processing and threading  
- Validate LLM integration and semantic analysis
- Ensure UI responsiveness maintained

**Step 4.2: Integration Testing**
- Test hub and spoke communication
- Verify data format consistency
- Test error handling and recovery
- Validate state persistence

**Step 4.3: Performance Testing**
- Benchmark against current implementation
- Verify no performance regression
- Test memory usage and threading
- Validate startup and shutdown

## Validation Checklist

### Functional Requirements ✓
- [ ] All original features preserved (input, display, scrolling, themes)
- [ ] Background semantic analysis continues (15-message cycles)
- [ ] LLM integration fully functional (requests, responses, parsing)
- [ ] File I/O operations continue seamlessly (auto-save, backup)
- [ ] Command system works (`/help`, `/stats`, `/analyze`, etc.)
- [ ] State persistence across sessions
- [ ] Thread safety maintained in background operations

### Architectural Requirements ✓  
- [ ] True hub and spoke: only orch.py calls mcp.py
- [ ] No spoke-to-spoke communication
- [ ] orch.py contains main program logic
- [ ] Clean separation of concerns (UI, storage, analysis, communication)
- [ ] Consolidated UI library eliminates fragmentation
- [ ] Semantic logic centralized in sem.py

### Quality Requirements ✓
- [ ] No performance regression
- [ ] Error handling maintained or improved  
- [ ] Debug logging continues to function
- [ ] Memory usage optimized
- [ ] Code complexity reduced
- [ ] Module boundaries clearly defined

### Data Consistency Requirements ✓
- [ ] Message formats standardized across modules
- [ ] Configuration data properly structured
- [ ] LLM request/response format consistent
- [ ] Error handling standardized
- [ ] State persistence format preserved

## Benefits of Remodularization

### **Architectural Benefits**
- **Single Responsibility**: Each module has one clear purpose
- **Hub Control**: Centralized coordination eliminates communication complexity
- **Clean Boundaries**: Well-defined interfaces prevent coupling
- **Service Model**: Modules provide focused services to orchestrator

### **Maintainability Benefits** 
- **Focused Modules**: Easier to understand and modify individual components
- **Centralized Logic**: Main program flow in one location
- **Consolidated UI**: All UI utilities in single library
- **Unified Semantics**: All semantic processing in one module

### **Development Benefits**
- **Independent Testing**: Each module can be unit tested in isolation
- **Parallel Development**: Multiple developers can work on different modules
- **Easy Extension**: New features added through orchestrator coordination
- **Clear Dependencies**: Module relationships explicitly defined

### **Operational Benefits**
- **Better Debugging**: Clear module boundaries for error attribution
- **Performance Monitoring**: Hub can monitor all module interactions
- **Graceful Degradation**: Orchestrator can handle module failures
- **Consistent Error Handling**: Standardized error flow through hub

## Risk Mitigation

### **Functionality Preservation**
- Comprehensive testing plan with current functionality verification
- Phase-by-phase migration with rollback capability
- Preservation of all existing interfaces during transition
- Extensive validation of background processing and threading

### **Performance Maintenance**
- No additional overhead from hub and spoke architecture
- Same threading model and background operations
- Preserved caching and optimization strategies
- Benchmark validation against current implementation

### **Development Continuity**
- Clear migration path with defined phases
- Preservation of all current data formats during transition
- Existing configuration and prompt systems maintained
- Debug logging enhanced, not replaced

## Success Metrics

### **Code Quality Metrics**
- Reduced cyclomatic complexity in individual modules
- Improved test coverage through isolated module testing
- Decreased coupling between components
- Increased cohesion within modules

### **Architectural Metrics**
- Clear module dependency graph (hub and spoke)
- Reduced inter-module communication complexity
- Standardized data format usage
- Consistent error handling patterns

### **Operational Metrics**
- Maintained or improved application startup time
- No regression in UI responsiveness
- Preserved background processing efficiency
- Continued data persistence reliability

---

## Final State Reference

After remodularization, the DevName RPG Client will have:

1. **Centralized Orchestration**: `orch.py` contains main program logic and coordinates all modules
2. **Consolidated UI**: `uilib.py` provides all UI utilities, `ncui.py` manages interface
3. **Focused Services**: `sem.py` for semantic analysis, `emm.py` for storage, `sme.py` for story state, `mcp.py` for communication
4. **Clean Architecture**: True hub and spoke with no spoke-to-spoke communication
5. **Preserved Functionality**: All existing features maintained with improved structure
6. **Enhanced Maintainability**: Clear boundaries and single responsibility per module
7. **Standardized Data**: Consistent formats and interfaces across all modules

This remodularization provides a solid foundation for future development while maintaining all current functionality and improving code organization, testability, and maintainability.