# Interface Improvements Project Plan (New Architecture)

## Problem Statement

The current interface has several formatting and display issues affecting readability and aesthetic appeal:

1. **Line Break Loss**: Model responses lose their natural paragraph breaks, creating chunky text blocks
2. **Gap Spacing Issues**: Blank lines before GM responses disappear after terminal resizing
3. **Incomplete Gap Spacing**: Missing blank lines after GM responses reduce readability
4. **Missing Border Lines**: Horizontal separators between sections don't always appear
5. **No Aesthetic Borders**: Interface uses full terminal space without visual framing

## Impact Analysis

### Current Problems
- Reduced readability due to collapsed paragraph structure
- Inconsistent formatting between sessions and after resizing
- Loss of visual hierarchy between user and GM content
- Unprofessional appearance without proper borders and separation

### Affected Components (NEW ARCHITECTURE)
- **`uilib.py`**: Message formatting and line break handling (DisplayMessage class)
- **`uilib.py`**: Box coordinate system and border drawing (TerminalManager class)
- **`ui.py`**: Display update methods and content rewrapping
- **`orch.py`**: Message storage and retrieval coordination

## Proposed Solution

### Phase 1: Line Break Preservation Analysis
**Duration**: 2-3 hours
**Goal**: Identify where line breaks are being lost

#### Tasks:
1. **Content Flow Analysis**: Trace message content from MCP response through display
2. **Textwrap Investigation**: Examine how `textwrap.wrap()` handles existing line breaks in `uilib.DisplayMessage`
3. **Display Pipeline Audit**: Check `DisplayMessage.format_for_display()` line break handling in `uilib.py`
4. **Terminal Width Mapping**: Verify content width calculations in `uilib.TerminalManager`

#### Expected Findings:
- Line breaks likely being collapsed during text wrapping process
- May need to preserve original paragraph structure before applying width constraints

### Phase 2: Message Formatting Enhancement
**Duration**: 4-5 hours
**Goal**: Preserve natural line breaks while maintaining responsive wrapping

#### 2.1: Enhanced DisplayMessage Processing (in `uilib.py`)
**Strategy**: Process content paragraph-by-paragraph rather than as single block

**Implementation Approach**:
- Modify `uilib.DisplayMessage.format_for_display()` method
- Split content on double line breaks (paragraph boundaries)
- Apply text wrapping to individual paragraphs
- Reassemble with preserved paragraph gaps
- Handle edge cases (single line breaks, mixed formatting)

#### 2.2: Intelligent Text Wrapping
**Current Issue**: `textwrap.wrap()` treats entire content as single block
**New Approach**: Paragraph-aware wrapping with structure preservation

**Key Features**:
- Preserve author's intended paragraph breaks
- Maintain responsive wrapping for terminal width
- Handle both narrative text and dialogue formatting
- Support mixed content (lists, dialogue, descriptions)

### Phase 3: Gap Spacing Standardization
**Duration**: 3-4 hours
**Goal**: Consistent blank line spacing around GM responses

#### 3.1: Spacing Rules Definition (in `ui.py`)
**Before GM Response**: Always one blank line
**After GM Response**: Always one blank line
**Consistency**: Maintain spacing through resize operations

#### 3.2: Display Update Method Enhancement
**Current**: `_add_blank_line_immediate()` called before GM responses only
**New**: Systematic spacing management for all message types in `ui.py`

**Implementation Strategy**:
- Add spacing rules to `uilib.DisplayMessage` class
- Update `ui.add_assistant_message_immediate()` to handle both before/after spacing
- Ensure spacing preservation during content rewrapping in `ui._rewrap_content()`
- Handle edge cases (consecutive system messages, startup messages)

### Phase 4: Border System Implementation
**Duration**: 5-6 hours
**Goal**: Add aesthetic ASCII borders with updated coordinate system

#### 4.1: ASCII Border Design (in `uilib.py`)
**Border Style**: Simple ASCII characters for wide compatibility
**Border Components**:
- Top/bottom: horizontal lines (`─` or `-`)
- Left/right: vertical lines (`│` or `|`) 
- Corners: appropriate corner characters (`┌┐└┘` or `+-++`)

#### 4.2: Coordinate System Update (in `uilib.TerminalManager`)
**Current**: Uses full terminal dimensions
**New**: Inset by 1-2 characters on all sides for border space

**Updated Box Calculations**:
- Modify `uilib.calculate_box_layout()` function
- Terminal boundaries: `(1, 1)` to `(width-2, height-2)`
- Text areas: Further inset to avoid border overlap
- Maintain proportional spacing (90% output, 10% input)
- Update all window positioning in `ui.py` to account for border offset

#### 4.3: Border Drawing Integration (in `uilib.TerminalManager`)
**Location**: Enhance `TerminalManager.draw_box_borders()`
**Features**:
- Draw outer terminal border first
- Draw section separators within border
- Handle border redraw during resize
- Ensure borders don't interfere with scrollable content

### Phase 5: Section Separator Reliability
**Duration**: 2-3 hours
**Goal**: Ensure horizontal separators always appear correctly

#### 5.1: Separator Drawing Audit
**Current Issue**: Lines sometimes missing or misplaced
**Investigation Points** (in `uilib.TerminalManager`):
- Coordinate calculation errors in `BoxCoordinates`
- Race conditions during resize
- Color/attribute conflicts
- Character encoding issues

#### 5.2: Robust Separator System
**Strategy**: Centralized separator management with error handling

**Implementation Features**:
- Verify coordinates before drawing in `uilib.TerminalManager`
- Handle terminal size edge cases
- Consistent separator redraw during all refresh operations in `ui._refresh_all_windows()`
- Debug logging for separator positioning through `orch.py` logger

### Phase 6: Integration and Testing
**Duration**: 4-5 hours
**Goal**: Integrate all improvements with thorough testing

#### 6.1: Component Integration
1. **Message Processing**: New paragraph-aware formatting in `uilib.py`
2. **Coordinate System**: Updated for border insets in `uilib.py`
3. **Display Pipeline**: Enhanced spacing and separator drawing in `ui.py`
4. **Resize Handling**: Preserve all formatting during geometry changes

#### 6.2: Test Scenarios
**Line Break Testing**:
- Multi-paragraph GM responses
- Mixed content (dialogue + narrative)
- Very long single paragraphs
- Short responses with natural breaks

**Spacing Testing**:
- Consecutive GM responses
- Mixed message types (system, user, GM)
- Startup message sequence
- Post-resize spacing preservation

**Border Testing**:
- Various terminal sizes
- Resize operations
- Color theme compatibility
- Character encoding support

**Separator Testing**:
- All terminal geometries
- Rapid resize sequences
- Section boundary accuracy
- Visual consistency

## Implementation Priority

### Critical Path (Immediate Impact)
1. **Line Break Preservation** (Phase 2.1) - Most visible user experience improvement
2. **Gap Spacing Fix** (Phase 3.1) - Readability enhancement
3. **Border Implementation** (Phase 4.2-4.3) - Professional appearance

### Secondary Path (Polish)
1. **Advanced Text Processing** (Phase 2.2) - Long-term content quality
2. **Separator Reliability** (Phase 5) - Consistency improvements
3. **Comprehensive Testing** (Phase 6.2) - Quality assurance

## Module Responsibility Matrix (NEW ARCHITECTURE)

### `uilib.py` (Consolidated UI Library)
- **DisplayMessage**: Format content with paragraph preservation
- **TerminalManager**: Calculate border-aware coordinates
- **BoxCoordinates**: Store layout with border insets
- **ColorManager**: Theme management for borders

### `ui.py` (Pure UI Controller)
- Window creation with border-aware dimensions
- Display refresh and update cycles
- Spacing rule enforcement
- Resize handling with format preservation

### `orch.py` (Main Orchestrator)
- Coordinate message flow between modules
- Manage display update timing
- Handle debug logging for formatting pipeline

## Technical Considerations

### Message Content Processing
- Need to distinguish between author line breaks (preserve) and artificial wrapping (recalculate)
- May require content metadata to track formatting intentions
- Consider markdown-style paragraph detection (`\n\n` boundaries)

### Coordinate System Changes
- All coordinate references now in `uilib.py` (consolidated from multiple files)
- Window creation in `ui.py` must use new border-aware coordinates
- Must maintain backward compatibility with existing layout logic

### Performance Impact
- More sophisticated text processing may increase display latency
- Border drawing adds overhead during resize operations
- Need to balance visual quality with responsiveness

## Risk Assessment

### Low Risk
- Gap spacing fixes (well-understood problem)
- Basic border implementation (standard terminal operations)

### Medium Risk  
- Line break preservation (complex text processing changes)
- Coordinate system updates (now consolidated in single module)

### High Risk
- Advanced paragraph detection (may misinterpret content formatting)

### Mitigation Strategies
- **Incremental Implementation**: Test each phase independently
- **Fallback Options**: Maintain current behavior as configurable option
- **User Testing**: Validate readability improvements with sample content
- **Debug Modes**: Enhanced logging through `orch.py` for formatting pipeline troubleshooting

## Success Criteria

### Functional Requirements
- [ ] Natural line breaks preserved in GM responses
- [ ] Consistent blank line spacing before and after GM responses
- [ ] ASCII borders appear correctly in all terminal sizes
- [ ] Section separators always visible and properly positioned
- [ ] All formatting preserved through resize operations

### Quality Requirements
- [ ] No performance regression in display updates
- [ ] Professional appearance with clean visual hierarchy
- [ ] Responsive design maintains readability at various terminal sizes
- [ ] Debug logging provides clear formatting pipeline visibility

### Compatibility Requirements
- [ ] Works with all three color themes
- [ ] Compatible with various terminal emulators
- [ ] Maintains existing keyboard navigation and scrolling
- [ ] No regression in multi-line input functionality

## Dependencies on Remodularization

This plan assumes the remodularization is complete:
- `uilib.py` exists with consolidated UI utilities
- `ui.py` is pure UI controller without business logic
- `orch.py` handles coordination between modules
- All `nci_*.py` files have been merged into `uilib.py`

If remodularization is not complete, references to the new module structure should be mapped back to existing files.

## Estimated Total Effort
**20-26 hours** across 6 phases, with highest priority on line break preservation and gap spacing standardization for immediate user experience improvements.
