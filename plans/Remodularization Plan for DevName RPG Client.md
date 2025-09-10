# Remodularization Plan for DevName RPG Client

## Executive Summary

This plan restructures the DevName RPG Client codebase to eliminate circular dependencies, reduce module coupling, and establish clear separation of concerns. The current architecture suffers from an overgrown `nci.py` orchestrator and tightly coupled `emm.py`/`sme.py` modules that require synchronized updates. This plan creates dedicated orchestrators and consolidates shared logic.

## Current Architecture Problems

### Primary Issues
1. **Overgrown Orchestrator**: `nci.py` has expanded beyond UI management to become the primary application orchestrator
2. **Circular Dependencies**: `emm.py` and `sme.py` are so tightly coupled they must be updated together
3. **Over-modularization**: Five separate `nci_*.py` files for simple UI utilities
4. **Mixed Responsibilities**: Semantic logic scattered across multiple modules
5. **Maintenance Burden**: Changes to any core logic require updates to 3+ files

### Impact
- Difficult to test individual components
- High risk of regression when making changes
- Unclear module boundaries
- Increased cognitive load for maintenance

## Proposed Architecture

### Module Structure
```
main.py           (Entry point - unchanged role)
├── orch.py       (NEW: Main orchestrator)
├── ui.py         (NEW: Pure UI controller, refactored from nci.py)
├── uilib.py      (NEW: Consolidated UI utilities)
├── sem.py        (NEW: Shared semantic logic)
├── mcp.py        (Existing - unchanged)
├── emm.py        (Refactored - uses sem.py)
└── sme.py        (Refactored - uses sem.py)
```

### Module Descriptions

#### `orch.py` - Main Orchestrator (NEW)
**Purpose**: Central coordination and lifecycle management
**Responsibilities**:
- Initialize all subsystems in correct order
- Manage inter-module communication
- Handle background thread coordination
- Process configuration and prompts from `main.py`
- Coordinate shutdown sequence

#### `ui.py` - UI Controller (REFACTORED from nci.py)
**Purpose**: Pure interface management without business logic
**Responsibilities**:
- Terminal/curses initialization
- Window creation and layout management
- Display refresh and update cycles
- Input capture and routing
- Terminal resize handling

#### `uilib.py` - UI Library (CONSOLIDATED)
**Purpose**: All UI utilities in single module
**Consolidates**:
- `nci_colors.py` → ColorManager, ColorTheme
- `nci_terminal.py` → TerminalManager, LayoutGeometry, BoxCoordinates
- `nci_display.py` → DisplayMessage, InputValidator
- `nci_scroll.py` → ScrollManager
- `nci_input.py` → MultiLineInput

#### `sem.py` - Semantic Logic Module (NEW)
**Purpose**: Centralized semantic analysis and processing logic
**Responsibilities**:
- Message categorization (6 categories)
- Pattern recognition algorithms
- Condensation strategies
- JSON parsing utilities (5-strategy approach)
- Background processing helpers
- Shared constants and thresholds

#### `emm.py` - Enhanced Memory Manager (REFACTORED)
**Changes**: Remove embedded semantic logic, use `sem.py`
**Retains**: Storage, retrieval, state management

#### `sme.py` - Story Momentum Engine (REFACTORED)
**Changes**: Remove embedded analysis logic, use `sem.py`
**Retains**: State tracking, antagonist management

## Implementation Phases

### Phase 1: Extract Semantic Logic (4-6 hours)
**Goal**: Create `sem.py` with all shared semantic logic

**Tasks**:
1. Analyze `emm.py` and `sme.py` for semantic logic
2. Create `sem.py` with shared category definitions
3. Extract pattern recognition logic
4. Move condensation algorithms
5. Consolidate JSON parsing strategies
6. Add comprehensive docstrings

**Deliverable**: `sem.py` module with complete semantic logic

### Phase 2: Create Main Orchestrator (3-4 hours)
**Goal**: Establish `orch.py` as central coordinator

**Tasks**:
1. Extract orchestration logic from `nci.py`
2. Create initialization sequence manager
3. Implement inter-module communication paths
4. Add background thread management
5. Handle configuration propagation
6. Implement graceful shutdown sequence

**Deliverable**: `orch.py` module with lifecycle management

### Phase 3: Refactor UI Controller (4-5 hours)
**Goal**: Transform `nci.py` into pure `ui.py`

**Tasks**:
1. Remove all orchestration code
2. Remove direct module management
3. Focus on pure UI operations
4. Simplify to window/display management
5. Update method signatures for new architecture
6. Add clear UI-only interfaces

**Deliverable**: `ui.py` module focused on interface only

### Phase 4: Consolidate UI Utilities (2-3 hours)
**Goal**: Merge five `nci_*.py` files into `uilib.py`

**Tasks**:
1. Create `uilib.py` structure
2. Copy ColorManager from `nci_colors.py`
3. Copy TerminalManager from `nci_terminal.py`
4. Copy DisplayMessage from `nci_display.py`
5. Copy ScrollManager from `nci_scroll.py`
6. Copy MultiLineInput from `nci_input.py`
7. Resolve any naming conflicts
8. Delete original `nci_*.py` files

**Deliverable**: Single `uilib.py` with all UI utilities

### Phase 5: Update Module Interfaces (3-4 hours)
**Goal**: Update `emm.py` and `sme.py` to use `sem.py`

**Tasks**:
1. Update `emm.py` imports to use `sem.py`
2. Replace embedded logic with `sem.py` calls
3. Update `sme.py` imports to use `sem.py`
4. Replace embedded logic with `sem.py` calls
5. Test inter-module communication
6. Verify state persistence

**Deliverable**: Refactored `emm.py` and `sme.py` using shared logic

### Phase 6: Integration and Testing (2-3 hours)
**Goal**: Update `main.py` and verify system functionality

**Tasks**:
1. Update `main.py` to initialize `orch.py`
2. Remove direct `nci.py` references
3. Test basic startup/shutdown
4. Verify message flow
5. Test background processing
6. Verify state persistence
7. Test error handling

**Deliverable**: Fully integrated and tested system

## Implementation Guidelines for Claude Sonnet 4

### Code Generation Strategy
1. **Preserve Functionality**: Maintain all existing features during refactoring
2. **Incremental Changes**: Complete one phase before moving to next
3. **Test Points**: Each phase should leave system in working state
4. **Documentation**: Add docstrings explaining module relationships
5. **Error Handling**: Maintain robust error handling throughout

### Critical Preservation Points
1. **Background Threading**: Must maintain non-blocking LLM operations
2. **State Persistence**: EMM/SME state saving must continue working
3. **Dynamic Coordinates**: Terminal resize handling must remain functional
4. **Prompt Integration**: System prompts must flow correctly to MCP
5. **Message Flow**: User input → processing → display pipeline must work

### Testing Checklist
After each phase, verify:
- [ ] Application starts without errors
- [ ] User can input text
- [ ] Messages display correctly
- [ ] LLM responses work
- [ ] Background analysis runs
- [ ] State saves/loads properly
- [ ] Terminal resize works
- [ ] Shutdown is clean

## Success Criteria

### Technical Metrics
- No circular dependencies between modules
- Each module under 500 lines (except `uilib.py`)
- Clear, single responsibility per module
- All tests pass without modification

### Qualitative Metrics
- Easier to understand module purposes
- Simpler to modify individual features
- Reduced coupling between components
- Clearer error messages and debugging

## Risk Mitigation

### Potential Issues
1. **Breaking Changes**: Keep backup of original files
2. **State Compatibility**: Ensure saved states remain loadable
3. **Performance**: Monitor for any degradation
4. **Missing Logic**: Document any discovered dependencies

### Rollback Plan
- Git commit after each successful phase
- Keep original files with `.backup` extension until completion
- Test thoroughly before removing backups

## Estimated Timeline

**Total Duration**: 18-25 hours

- Phase 1: 4-6 hours
- Phase 2: 3-4 hours  
- Phase 3: 4-5 hours
- Phase 4: 2-3 hours
- Phase 5: 3-4 hours
- Phase 6: 2-3 hours

## Notes for Implementation

### Starting Point
Begin with Phase 1 (`sem.py`) as it establishes the foundation for subsequent phases. This module can be created and tested independently without breaking existing functionality.

### Module Naming Rationale
- **3-6 character names**: Maintains consistency with existing convention
- **`orch`**: Orchestrator (clear purpose)
- **`ui`**: User Interface (obvious role)
- **`uilib`**: UI Library (indicates utility collection)
- **`sem`**: Semantic (core logic designation)

### Dependencies to Maintain
- Python 3.8+ compatibility
- Curses library for terminal UI
- No external dependencies beyond standard library
- HTTP/HTTPS for MCP communication

## Appendix: File Mapping

### Files to Create
- `orch.py` (new orchestrator)
- `ui.py` (refactored from `nci.py`)
- `uilib.py` (consolidated utilities)
- `sem.py` (semantic logic)

### Files to Modify
- `main.py` (update imports and initialization)
- `emm.py` (use `sem.py` for logic)
- `sme.py` (use `sem.py` for logic)

### Files to Delete (After Phase 4)
- `nci.py` (replaced by `ui.py`)
- `nci_colors.py` (merged into `uilib.py`)
- `nci_terminal.py` (merged into `uilib.py`)
- `nci_display.py` (merged into `uilib.py`)
- `nci_scroll.py` (merged into `uilib.py`)
- `nci_input.py` (merged into `uilib.py`)

### Files Unchanged
- `mcp.py` (already well-isolated)
- `genai.txt` (update after completion)
- Configuration and prompt files
