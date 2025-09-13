# NCurses Input Improvements

## Current Behavior

### Input System Limitations
- **Fixed Display Window**: Input content is constrained by physical window boundaries
- **Content Truncation**: Long inputs are cut off and become inaccessible
- **Limited Navigation**: No way to scroll back through long input content
- **Basic Key Support**: Home/End keys not implemented for buffer navigation
- **No Word Navigation**: No efficient way to jump between words
- **Limited Editing**: Basic character deletion without proper word wrap adjustment

### Current Input Flow
1. `MultiLineInput` receives max width from layout calculations
2. Text wrapping occurs within fixed boundaries
3. Content beyond window height is lost or inaccessible
4. Cursor movement limited to visible content area

### Key Components
- **uilib.py**: `MultiLineInput` class - handles text input and cursor positioning
- **ncui.py**: `_handle_user_input()` - processes key events and input submission
- **ncui.py**: `_refresh_input_window()` - displays current input content

## Desired Behavior

### Scrollable Input Buffer
- **Vertical Scrolling**: Navigate through long input content using arrow keys
- **Buffer Navigation**: Access full input history regardless of window size
- **Smart Scrolling Logic**: Up arrow on top line scrolls back, down arrow on bottom line scrolls forward
- **Enhanced Key Support**: Home/End keys jump to buffer beginning/end

### Navigation Controls
- **Up Arrow**: Normal line navigation, or scroll back when at top of visible area
- **Down Arrow**: Normal line navigation, or scroll forward when at bottom of visible area
- **Home Key**: Jump to very beginning of entire input buffer
- **End Key**: Jump to very end of entire input buffer
- **Ctrl+Left/Right**: Jump by words for faster navigation
- **Backspace**: Delete character behind cursor with automatic word wrap adjustment
- **Delete**: Delete character at cursor with automatic word wrap adjustment

## Implementation Strategy

### Phase 1: MultiLineInput Enhancement
- **Add Scroll State**: Track current scroll offset within the input buffer
- **Viewport Logic**: Separate logical content from visible content
- **Cursor Mapping**: Map visible cursor position to actual buffer position

### Phase 2: Key Handling Logic
- **Enhanced Arrow Keys**: Detect when cursor is at viewport edges
- **Scroll Triggers**: Implement scroll-back/scroll-forward when appropriate
- **Home/End Implementation**: Add buffer-wide navigation commands
- **Word Navigation**: Implement Ctrl+Left/Right for word-by-word movement
- **Character Deletion**: Full Backspace/Delete support with word wrap adjustment
- **Boundary Detection**: Properly handle start/end of buffer conditions

### Phase 3: Display System Updates
- **Viewport Rendering**: Display only the visible portion of input buffer
- **Cursor Positioning**: Translate buffer cursor to screen coordinates
- **Content Synchronization**: Ensure display matches logical buffer state

## Code Architecture Changes

### MultiLineInput Enhancements
```python
class MultiLineInput:
    def __init__(self):
        self.content_lines = []      # Full input buffer
        self.cursor_line = 0         # Logical cursor position in buffer
        self.cursor_col = 0          # Logical cursor column
        self.scroll_offset = 0       # Current viewport scroll position
        self.viewport_height = 5     # Visible lines in input window
        
    def handle_input(self, key):
        # Enhanced key handling with scroll logic
        
    def get_visible_content(self):
        # Return only the visible portion of buffer
        
    def get_viewport_cursor(self):
        # Return cursor position relative to viewport
```

### Key Handling Updates
- **Up Arrow Logic**: If `cursor_line == 0 and scroll_offset > 0`: scroll back
- **Down Arrow Logic**: If `cursor_line == viewport_bottom and more_content_below`: scroll forward
- **Home Key**: Set `cursor_line = 0, cursor_col = 0, scroll_offset = 0`
- **End Key**: Move cursor to end of buffer, adjust scroll to show it

### Display Integration
- **ncui.py**: Update `_refresh_input_window()` to use viewport content
- **Cursor Positioning**: Translate buffer coordinates to screen coordinates
- **Scroll Indicators**: Optional visual indicators when content extends beyond viewport

## Technical Implementation Details

### Buffer Management
- **Content Storage**: Store full input as list of lines
- **Viewport Calculation**: Determine which lines to display based on scroll offset
- **Boundary Checking**: Prevent scrolling beyond buffer limits

### Cursor Handling
- **Logical Position**: Track actual position in full buffer
- **Visual Position**: Calculate display position within viewport
- **Coordinate Translation**: Convert between buffer and screen coordinates

### Key Event Processing
```python
def handle_input(self, key):
    if key == curses.KEY_UP:
        if self.cursor_line > 0:
            # Normal line navigation
            self.cursor_line -= 1
        elif self.scroll_offset > 0:
            # Scroll back in buffer
            self.scroll_offset -= 1
            
    elif key == curses.KEY_DOWN:
        if self.cursor_line < len(self.visible_lines) - 1:
            # Normal line navigation
            self.cursor_line += 1
        elif self.has_content_below():
            # Scroll forward in buffer
            self.scroll_offset += 1
            
    elif key == curses.KEY_HOME:
        # Jump to beginning of buffer
        self.cursor_line = 0
        self.cursor_col = 0
        self.scroll_offset = 0
        
    elif key == curses.KEY_END:
        # Jump to end of buffer
        self._move_to_buffer_end()
        
    elif key == 543:  # Ctrl+Left
        self._jump_word_left()
        
    elif key == 558:  # Ctrl+Right
        self._jump_word_right()
        
    elif key == curses.KEY_BACKSPACE:
        self._delete_char_behind_cursor()
        self._rewrap_content()
        
    elif key == curses.KEY_DC:  # Delete key
        self._delete_char_at_cursor()
        self._rewrap_content()
```

### Content Editing
- **Text Insertion**: Add characters at cursor position with automatic wrapping
- **Character Deletion**: Remove characters with proper buffer adjustment
- **Word Wrapping**: Dynamic text flow adjustment after edits
- **Content Preservation**: Maintain logical text structure during modifications

### Cursor Handling
- **Logical Position**: Track actual position in full buffer
- **Visual Position**: Calculate display position within viewport
- **Coordinate Translation**: Convert between buffer and screen coordinates

## Integration Points

### ncui.py Modifications
- **Key Processing**: Update arrow key handling in main input loop
- **Display Updates**: Modify `_refresh_input_window()` for viewport rendering
- **Cursor Management**: Update `_ensure_cursor_in_input()` for viewport coordinates

### uilib.py Changes
- **MultiLineInput Class**: Add scrolling logic and viewport management
- **Content Retrieval**: Update methods to work with full buffer vs visible content
- **State Management**: Track scroll position and buffer dimensions

## Risk Assessment

### Low Risk
- **Backward Compatibility**: Changes are largely internal to MultiLineInput
- **Isolated Changes**: Most modifications contained within input handling system
- **Fallback Behavior**: Current behavior preserved when content fits in window

### Medium Risk
- **Cursor Positioning**: Complex coordinate translation between buffer and screen
- **Edge Cases**: Boundary conditions at start/end of buffer
- **State Synchronization**: Keeping logical and visual states aligned

### Mitigation Strategies
- **Incremental Testing**: Test each key behavior individually
- **Boundary Testing**: Verify behavior at buffer start/end
- **State Validation**: Add assertions to ensure cursor/scroll consistency

## Alternative Enhancements

### Implementation Notes
- **Page Up/Down Reserved**: These keys are already used for output window scrollback
- **Word Navigation**: Ctrl+Left/Right provides efficient text traversal
- **Edit Operations**: Full Backspace/Delete support with word wrap adjustment
- **No Undo System**: Sufficient navigation tools provided without complex edit history

### Future Considerations
- **Mouse Scroll Support**: If terminal supports mouse events
- **Search Within Buffer**: Find/highlight text within input buffer
- **Buffer Size Limits**: Prevent extremely large input buffers
- **Visual Indicators**: Show scroll position or "more content" hints

## Implementation Priority

1. **Core Scrolling**: Up/down arrow scrolling logic with viewport management
2. **Character Editing**: Full Backspace/Delete with word wrap adjustment  
3. **Word Navigation**: Ctrl+Left/Right for efficient text traversal
4. **Home/End Keys**: Buffer-wide navigation to beginning/end
5. **Display Updates**: Viewport rendering in refresh methods
6. **Edge Case Handling**: Boundary conditions and error cases
7. **Polish**: Visual indicators and user experience refinements

This approach maintains the current fixed-layout architecture while dramatically improving input usability through scrolling and comprehensive editing capabilities, making it much simpler to implement than dynamic window resizing while achieving the core goal of removing input length constraints and providing modern text editing features.