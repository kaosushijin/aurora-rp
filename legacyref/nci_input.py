# Chunk 5/6 - nci_input.py - Multi-line Input System Module
#!/usr/bin/env python3
"""
DevName RPG Client - Multi-line Input System Module (nci_input.py)
Module architecture and interconnects documented in genai.txt
Extracted from nci.py for better separation of concerns
"""

import curses
import textwrap
from typing import List, Tuple

class MultiLineInput:
    """Multi-line input system with cursor navigation and word wrapping"""
    
    def __init__(self, max_width: int = 80):
        self.lines = [""]           # List of text lines
        self.cursor_line = 0        # Current line index
        self.cursor_col = 0         # Current column position within line
        self.scroll_offset = 0      # Vertical scroll within input area
        self.max_width = max_width  # Maximum line width before wrapping
        self.max_lines = 10         # Maximum number of lines allowed
    
    def insert_char(self, char: str) -> bool:
        """Insert character at cursor position with word wrapping"""
        if len(self.get_content()) >= 4000:  # Reasonable character limit
            return False
        
        # Insert character at current position
        current_line = self.lines[self.cursor_line]
        new_line = current_line[:self.cursor_col] + char + current_line[self.cursor_col:]
        self.lines[self.cursor_line] = new_line
        self.cursor_col += 1
        
        # Check if line needs wrapping
        if len(new_line) >= self.max_width - 5:  # Leave margin for prompt
            self._wrap_current_line()
        
        return True
    
    def handle_enter(self) -> Tuple[bool, str]:
        """
        Handle Enter key - create new line or submit based on context
        Returns (should_submit, content)
        """
        # If cursor is at end of last line and line is not empty, consider submission
        if (self.cursor_line == len(self.lines) - 1 and 
            self.cursor_col == len(self.lines[self.cursor_line]) and
            self.lines[self.cursor_line].strip()):
            
            # Check if content looks complete (ends with punctuation or is command)
            content = self.get_content().strip()
            if content.startswith('/') or content.endswith(('.', '!', '?', '"', "'", ':', ';')):
                return True, content
        
        # Otherwise, create new line
        self.insert_newline()
        return False, ""
    
    def insert_newline(self):
        """Insert new line at cursor position"""
        if len(self.lines) >= self.max_lines:
            return  # Don't exceed line limit
        
        current_line = self.lines[self.cursor_line]
        
        # Split current line at cursor
        line_before = current_line[:self.cursor_col]
        line_after = current_line[self.cursor_col:]
        
        # Update current line and insert new line
        self.lines[self.cursor_line] = line_before
        self.lines.insert(self.cursor_line + 1, line_after)
        
        # Move cursor to start of new line
        self.cursor_line += 1
        self.cursor_col = 0
        
        # Adjust scroll if needed
        self._adjust_scroll()
    
    def handle_backspace(self) -> bool:
        """Handle backspace with line merging"""
        if self.cursor_col > 0:
            # Delete character in current line
            current_line = self.lines[self.cursor_line]
            new_line = current_line[:self.cursor_col-1] + current_line[self.cursor_col:]
            self.lines[self.cursor_line] = new_line
            self.cursor_col -= 1
            return True
        
        elif self.cursor_line > 0:
            # Merge with previous line
            prev_line = self.lines[self.cursor_line - 1]
            current_line = self.lines[self.cursor_line]
            
            # Move cursor to end of previous line
            self.cursor_col = len(prev_line)
            self.cursor_line -= 1
            
            # Merge lines
            self.lines[self.cursor_line] = prev_line + current_line
            del self.lines[self.cursor_line + 1]
            
            self._adjust_scroll()
            return True
        
        return False
    
    def handle_arrow_keys(self, key: int) -> bool:
        """Handle arrow key navigation"""
        if key == curses.KEY_LEFT:
            if self.cursor_col > 0:
                self.cursor_col -= 1
            elif self.cursor_line > 0:
                self.cursor_line -= 1
                self.cursor_col = len(self.lines[self.cursor_line])
                self._adjust_scroll()
            return True
            
        elif key == curses.KEY_RIGHT:
            if self.cursor_col < len(self.lines[self.cursor_line]):
                self.cursor_col += 1
            elif self.cursor_line < len(self.lines) - 1:
                self.cursor_line += 1
                self.cursor_col = 0
                self._adjust_scroll()
            return True
            
        elif key == curses.KEY_UP:
            if self.cursor_line > 0:
                self.cursor_line -= 1
                self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_line]))
                self._adjust_scroll()
            return True
            
        elif key == curses.KEY_DOWN:
            if self.cursor_line < len(self.lines) - 1:
                self.cursor_line += 1
                self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_line]))
                self._adjust_scroll()
            return True
        
        return False
    
    def _wrap_current_line(self):
        """Wrap current line if it's too long"""
        current_line = self.lines[self.cursor_line]
        if len(current_line) < self.max_width:
            return
        
        # Find good break point (space or punctuation)
        break_point = self.max_width - 10
        for i in range(break_point, max(0, break_point - 20), -1):
            if current_line[i] in ' \t-':
                break_point = i + 1
                break
        
        # Split line
        line_before = current_line[:break_point].rstrip()
        line_after = current_line[break_point:].lstrip()
        
        # Update lines
        self.lines[self.cursor_line] = line_before
        
        if line_after and len(self.lines) < self.max_lines:
            self.lines.insert(self.cursor_line + 1, line_after)
            
            # Adjust cursor position
            if self.cursor_col > len(line_before):
                self.cursor_line += 1
                self.cursor_col = min(self.cursor_col - len(line_before) - 1, len(line_after))
    
    def _adjust_scroll(self):
        """Adjust scroll to keep cursor visible"""
        # This would be used with a multi-line input window
        # For now, keeping it simple since we have limited input window height
        pass
    
    def get_display_lines(self, width: int, height: int) -> List[str]:
        """Get lines formatted for display within given dimensions"""
        display_lines = []
        
        for i, line in enumerate(self.lines):
            if len(display_lines) >= height - 1:  # Save space for cursor line
                break
            
            if len(line) > width - 8:  # Account for prompt
                # Wrap long lines for display
                wrapped = textwrap.wrap(line, width - 8, break_long_words=True)
                display_lines.extend(wrapped)
            else:
                display_lines.append(line)
        
        # Ensure we have at least one line
        if not display_lines:
            display_lines = [""]
        
        return display_lines[:height]
    
    def get_content(self) -> str:
        """Get complete content as single string"""
        return '\n'.join(self.lines)
    
    def clear(self):
        """Clear all content"""
        self.lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.scroll_offset = 0
    
    def set_content(self, content: str):
        """Set content from string"""
        self.lines = content.split('\n') if content else [""]
        self.cursor_line = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])
        self.scroll_offset = 0
    
    def get_cursor_position(self) -> Tuple[int, int]:
        """Get cursor position for display (line, column)"""
        return (self.cursor_line, self.cursor_col)
    
    def is_empty(self) -> bool:
        """Check if input is empty"""
        return not self.get_content().strip()
    
    def update_max_width(self, new_width: int):
        """Update maximum width for line wrapping"""
        self.max_width = new_width
