# Word Wrap Redux - Project Plan

## Executive Summary

Comprehensive redesign of the word wrapping system in the DevName RPG Client's input handling to address fundamental issues with the current implementation that interfere with natural typing and proper buffer management.

## Problem Statement

The current word wrapping implementation has several critical issues:

### Current Issues
1. **Premature Space Deletion**: Spaces are automatically deleted when typing at line ends, forcing words to concatenate unnaturally
2. **Viewport Overflow**: Edit operations can cause content to exceed the input window boundaries
3. **Inconsistent Width Calculations**: Arbitrary subtractions (`max_width - 5`, `max_width - 20`) cause unpredictable behavior
4. **Buffer/Viewport Confusion**: Word wrapping logic doesn't properly account for the scrollable input buffer system
5. **Aggressive Rewrapping**: The `_adjust_word_wrap_after_edit()` method causes content to flow in unexpected ways

### User Experience Impact
- Typing feels unnatural and unpredictable
- Users cannot control when line breaks occur
- Visual content can extend beyond window boundaries
- Backspace/Delete operations produce inconsistent results

## Project Goals

### Primary Objectives
1. **Natural Typing Experience**: Allow users to type spaces and words without automatic deletion
2. **Predictable Line Breaks**: Wrap only when absolutely necessary, not preemptively
3. **Viewport Compliance**: Ensure all content respects input window boundaries
4. **Buffer Integration**: Full compatibility with scrollable input buffer system
5. **Consistent Behavior**: Uniform width calculations and wrapping logic

### Success Criteria
- Users can type spaces naturally without automatic deletion
- Content never overflows the input window
- Word wrapping occurs only at actual width limits
- Backspace/Delete operations maintain proper text flow
- All operations work correctly with input buffer scrolling

## Technical Architecture

### Core Design Principles

#### 1. Deferred Wrapping Strategy
- **When**: Only wrap when content actually exceeds the available width
- **Where**: At actual character boundaries, not arbitrary margins
- **How**: Check after each edit operation, not during typing

#### 2. Clean Width Management
- **Single Source of Truth**: Use `max_width` consistently without subtractions
- **Dynamic Calculation**: Adapt to actual available space from layout
- **Margin Handling**: Account for borders and prompts in layout calculation, not wrapping logic

#### 3. Buffer-Aware Operations
- **Logical vs Visual**: Separate content management from display rendering
- **Scroll Integration**: All wrap operations respect current scroll position
- **Viewport Boundaries**: Ensure wrapped content fits within visible area

#### 4. Smart Rewrapping
- **Conservative Approach**: Minimize automatic rewrapping during edits
- **User Control**: Preserve user's intentional line breaks
- **Flow Management**: Handle content movement between lines intelligently

### Implementation Components

#### Component 1: Core Wrapping Engine
**Location**: `uilib.py` - `MultiLineInput` class
**Functions**:
- `_check_and_wrap_line(line_index)` - Main wrapping logic
- `_find_wrap_point(text, max_width)` - Intelligent break point detection
- `_needs_wrapping(line_index)` - Determine if wrapping is necessary

#### Component 2: Edit Operation Handlers
**Functions**:
- `_handle_post_insert()` - Post-character-insertion processing
- `_handle_post_delete()` - Post-deletion content flow management
- `_reflow_content(start_line)` - Intelligent content redistribution

#### Component 3: Buffer Integration
**Functions**:
- `_wrap_within_viewport()` - Ensure viewport compliance
- `_adjust_scroll_after_wrap()` - Maintain cursor visibility after wrapping
- `_update_buffer_metrics()` - Keep buffer size tracking accurate

## Implementation Plan

### Phase 1: Core Wrapping Logic (Priority: High)
**Duration**: 1 implementation session
**Deliverables**:
- New `_check_and_wrap_line()` method with proper width handling
- Improved `_find_wrap_point()` logic that respects word boundaries
- Remove arbitrary width subtractions throughout codebase

**Key Changes**:
- Replace `len(new_line) >= self.max_width - 5` with `len(new_line) > self.max_width`
- Implement intelligent space handling that doesn't delete characters
- Add proper word boundary detection for natural line breaks

### Phase 2: Edit Operation Integration (Priority: High)
**Duration**: 1 implementation session
**Deliverables**:
- Redesigned `insert_char()` method with deferred wrapping
- Fixed `handle_backspace()` and `handle_delete()` with proper rewrapping
- New post-edit content flow management

**Key Changes**:
- Move wrapping check to after character insertion, not during
- Replace `_adjust_word_wrap_after_edit()` with smarter content flow logic
- Ensure all edit operations maintain viewport boundaries

### Phase 3: Buffer System Compatibility (Priority: Medium)
**Duration**: 1 implementation session
**Deliverables**:
- Full integration with scrollable input buffer
- Proper viewport boundary enforcement
- Scroll position maintenance after wrapping operations

**Key Changes**:
- Update wrapping logic to work with `scroll_offset` and `viewport_height`
- Ensure cursor remains visible after wrap operations
- Maintain accurate `total_buffer_lines` tracking

### Phase 4: Testing and Refinement (Priority: Medium)
**Duration**: 1 implementation session
**Deliverables**:
- Comprehensive testing of edge cases
- Performance optimization
- User experience validation

**Test Cases**:
- Long words at line boundaries
- Rapid typing with spaces
- Backspace/Delete at wrapped boundaries
- Content flowing between multiple lines
- Scroll position maintenance during edits

## Implementation Details

### Key Method Redesigns

#### Current `insert_char()` Issues:
```python
# PROBLEMATIC: Wraps prematurely with arbitrary margin
if len(new_line) >= self.max_width - 5:
    self._wrap_current_line()
```

#### Proposed `insert_char()` Fix:
```python
# IMPROVED: Only wrap when actually necessary
self.lines[self.cursor_line] = new_line
self.cursor_col += 1

# Check if we need to wrap (deferred approach)
if len(new_line) > self.max_width:
    self._check_and_wrap_line(self.cursor_line)
```

#### Current `_adjust_word_wrap_after_edit()` Issues:
- Uses arbitrary `max_width - 20` calculation
- Can cause content to exceed viewport boundaries
- Doesn't account for buffer scrolling

#### Proposed Content Flow Management:
```python
def _reflow_content_after_edit(self, start_line: int):
    """Intelligently reflow content after edit operations"""
    # Only reflow if there's actually space and benefit
    # Respect viewport boundaries
    # Maintain user's intentional formatting
```

### Risk Mitigation

#### High Risk Areas
1. **Cursor Position Tracking**: Complex coordinate management during wrapping
2. **Content Preservation**: Ensuring no text is lost during rewrapping operations
3. **Performance Impact**: Avoiding excessive recalculations during typing

#### Mitigation Strategies
1. **Incremental Implementation**: Build and test each component separately
2. **Comprehensive Logging**: Add debug output for all wrapping decisions
3. **Rollback Plan**: Preserve current implementation until new system is validated
4. **Edge Case Testing**: Focus on boundary conditions and rapid input scenarios

## Success Metrics

### Functional Requirements
- ✅ Users can type spaces without automatic deletion
- ✅ Content never exceeds input window boundaries
- ✅ Word wrapping occurs only at appropriate points
- ✅ Backspace/Delete operations maintain text integrity
- ✅ All operations compatible with scrollable input buffer

### Performance Requirements
- No noticeable lag during normal typing
- Wrapping operations complete within 1ms
- Memory usage remains constant during wrapping operations

### User Experience Requirements
- Typing feels natural and predictable
- Visual feedback is immediate and accurate
- Line breaks occur at logical points (spaces, punctuation)
- Edit operations behave consistently across all scenarios

## Conclusion

The Word Wrap Redux project addresses fundamental usability issues in the current input system by implementing a more intelligent, user-friendly approach to text wrapping that properly integrates with the scrollable input buffer architecture. The deferred wrapping strategy and clean width management will provide a significantly improved user experience while maintaining system reliability and performance.

---
*Project Plan Version 1.0*
*Created for DevName RPG Client Word Wrap System Redesign*