# Multi-Line Input System Fix Documentation
## DevName RPG Client - clearterminal Project

### Current Implementation Analysis

The multi-line input system is implemented in `uilib.py` in the `MultiLineInput` class. The current implementation has partial word wrap and content flow logic but has three critical issues that need fixing.

## Issues to Fix

### Issue 1: Non-Cascading Content Flow
**Problem**: When deleting text from line 1, words pull from line 2, but line 2 doesn't then pull from line 3 to fill its newly available space.

**Current Code**: The `_smart_content_flow()` method only processes one line pair at a time:
- It checks if the current line has space
- Pulls a word from the next line if it fits
- But doesn't recursively cascade through subsequent lines

**Fix Required**: Implement recursive cascading that processes all affected lines:

```python
def _cascade_content_flow(self, start_line: int = None):
    """Recursively cascade content flow through all lines after edits"""
    if start_line is None:
        start_line = self.cursor_line
    
    # Process from start_line to end of buffer
    for line_idx in range(start_line, len(self.lines) - 1):
        current_line = self.lines[line_idx]
        next_line = self.lines[line_idx + 1]
        
        # Calculate available space
        available_space = self.max_width - len(current_line)
        
        # Skip if no significant space or no content to pull
        if available_space < 2 or not next_line.strip():
            continue
        
        # Find words that can be pulled up
        words_pulled = False
        while available_space > 1 and next_line.strip():
            # Find first word in next line
            first_word_end = next_line.find(' ')
            if first_word_end == -1:
                first_word = next_line.strip()
                remaining = ""
            else:
                first_word = next_line[:first_word_end]
                remaining = next_line[first_word_end + 1:]
            
            # Check if word fits with space
            space_needed = (1 if current_line else 0) + len(first_word)
            if space_needed <= available_space:
                # Pull the word up
                if current_line:
                    current_line = current_line + " " + first_word
                else:
                    current_line = first_word
                
                self.lines[line_idx] = current_line
                next_line = remaining
                available_space = self.max_width - len(current_line)
                words_pulled = True
            else:
                break  # Word doesn't fit
        
        # Update or remove the next line
        if next_line.strip():
            self.lines[line_idx + 1] = next_line
        else:
            # Next line is now empty, remove it
            del self.lines[line_idx + 1]
            # Continue cascading from same position since we deleted a line
            if line_idx < len(self.lines) - 1:
                self._cascade_content_flow(line_idx)
            break
```

### Issue 2: Space Deletion in Word Wrap
**Problem**: When two words wrap together (like "and delete"), they merge into "anddelete" with the space removed, and cursor positioning is off by one.

**Current Code**: The `_wrap_current_line()` method has issues with space handling:
- It strips spaces indiscriminately when splitting lines
- Doesn't properly track if the break point was at a space

**Fix Required**: Smarter space preservation in `_wrap_current_line()`:

```python
def _wrap_current_line(self):
    """Wrap current line with proper space handling"""
    current_line = self.lines[self.cursor_line]
    
    if len(current_line) <= self.max_width:
        return
    
    # Find intelligent break point
    break_point = self._find_wrap_point(current_line, self.max_width)
    
    # Split the line
    line_before = current_line[:break_point]
    line_after = current_line[break_point:]
    
    # Handle space at break point correctly
    if break_point < len(current_line):
        break_char = current_line[break_point]
        
        # If we're breaking at a space, handle it properly
        if break_char == ' ':
            # Remove trailing spaces from first line
            line_before = line_before.rstrip()
            # Remove leading space from second line (the break space)
            line_after = line_after[1:] if line_after.startswith(' ') else line_after
        # If breaking mid-word, keep as is (no space removal)
    
    # Update lines
    self.lines[self.cursor_line] = line_before
    if line_after and len(self.lines) < self.max_lines:
        self.lines.insert(self.cursor_line + 1, line_after)
        
        # Fix cursor positioning
        if self.cursor_col > len(line_before):
            self.cursor_line += 1
            # Calculate correct position in wrapped line
            self.cursor_col = self.cursor_col - len(line_before)
            # Don't subtract 1 unless we removed a space
            if break_char == ' ':
                self.cursor_col = max(0, self.cursor_col - 1)
    
    self._adjust_scroll()

def _find_wrap_point(self, text: str, max_width: int) -> int:
    """Find optimal break point for word wrapping"""
    if len(text) <= max_width:
        return len(text)
    
    # First, try to break at a space before max_width
    last_space = -1
    for i in range(max_width - 1, -1, -1):
        if text[i] == ' ':
            last_space = i
            break
    
    # If we found a space reasonably close to max_width, use it
    if last_space > max_width - 20:
        return last_space
    
    # Otherwise look for other break points
    for i in range(max_width, max(0, max_width - 25), -1):
        if i >= len(text):
            continue
        
        char = text[i]
        
        # Break after punctuation
        if char in '.!?;:,':
            return i + 1
        
        # Break at space
        if char == ' ':
            return i
    
    # Last resort: hard break at max_width
    return max_width
```

### Issue 3: Missing Home/End Key Navigation
**Problem**: Home and End keys don't move cursor to top/bottom of input buffer.

**Current Code**: The `handle_arrow_keys()` method doesn't handle Home/End keys at all.

**Fix Required**: Add Home/End key handling:

```python
def handle_arrow_keys(self, key: int) -> bool:
    """Handle arrow key and Home/End navigation"""
    
    # ... existing arrow key code ...
    
    # Add Home/End key handling
    if key == curses.KEY_HOME:
        # Move to beginning of input buffer
        self.cursor_line = 0
        self.cursor_col = 0
        self.scroll_offset = 0
        self._adjust_scroll()
        return True
    
    elif key == curses.KEY_END:
        # Move to end of input buffer
        self.cursor_line = len(self.lines) - 1
        self.cursor_col = len(self.lines[self.cursor_line])
        # Scroll to make cursor visible
        self._adjust_scroll()
        return True
    
    # ... rest of method ...
```

## Integration Points

### Update `handle_backspace()` and `handle_delete()`:
Replace calls to `_smart_content_flow()` with `_cascade_content_flow()`:

```python
def handle_backspace(self) -> bool:
    """Handle backspace with cascading content flow"""
    if self.cursor_col > 0:
        # Delete character in current line
        current_line = self.lines[self.cursor_line]
        new_line = current_line[:self.cursor_col-1] + current_line[self.cursor_col:]
        self.lines[self.cursor_line] = new_line
        self.cursor_col -= 1
        
        # Use cascading flow instead of smart flow
        self._cascade_content_flow()
        return True
    # ... rest of method ...
```

### Update `ncui.py` key handling:
Ensure Home/End keys are passed to MultiLineInput:

```python
# In ncui.py _handle_key_input() method
elif key in [curses.KEY_HOME, curses.KEY_END]:
    # Let MultiLineInput handle Home/End for input navigation
    return self.multi_input.handle_arrow_keys(key)
```

## Testing Scenarios

1. **Cascading Flow Test**:
   - Type text that fills 3+ lines
   - Delete from middle of line 1
   - Verify words flow: line 2→1, line 3→2

2. **Space Preservation Test**:
   - Type "word1 word2 word3" until it wraps
   - Verify spaces aren't deleted in wrapped words
   - Check cursor positioning is correct

3. **Home/End Navigation Test**:
   - Fill multiple lines of input
   - Press Home - cursor should go to start
   - Press End - cursor should go to end
   - Verify scrolling adjusts appropriately

## Implementation Notes

- The `_update_buffer_size()` method already exists and tracks total lines
- The `_adjust_scroll()` method handles viewport scrolling
- Preserve existing viewport scrolling logic
- Maintain thread safety if needed (current code doesn't show threading)
- Test with various terminal widths to ensure wrapping works correctly

## Summary

These three fixes will create a robust multi-line input system with:
1. Proper cascading content flow through all lines
2. Correct space handling during word wrap
3. Full Home/End key navigation support

The fixes integrate cleanly with the existing codebase and maintain the current architecture's design patterns.