# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - Consolidated UI Library (uilib.py)
Consolidates all nci_*.py files into single cohesive UI library
Remodularized for hub-and-spoke architecture
"""

import curses
import re
import os
import threading
import time
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# No external module imports needed - uilib.py is self-contained
# All UI utilities consolidated from nci_terminal.py, nci_input.py, nci_scroll.py, nci_display.py, nci_colors.py

# Configuration constants
MIN_SCREEN_WIDTH = 80
MIN_SCREEN_HEIGHT = 24
MAX_USER_INPUT_TOKENS = 2000

# =============================================================================
# TERMINAL MANAGEMENT SYSTEM
# =============================================================================

@dataclass
class BoxCoordinates:
    """Box coordinate system with outer boundaries and inner text fields"""
    # Outer box boundaries (including borders)
    top: int
    left: int
    bottom: int
    right: int
    
    # Inner text field boundaries (excluding borders)
    inner_top: int
    inner_left: int
    inner_bottom: int
    inner_right: int
    
    # Calculated dimensions
    width: int
    height: int
    inner_width: int
    inner_height: int

@dataclass
class LayoutGeometry:
    """Complete terminal layout with all box definitions"""
    terminal_height: int
    terminal_width: int
    
    # Box definitions
    output_box: BoxCoordinates
    input_box: BoxCoordinates
    status_line: BoxCoordinates
    
    # Layout metadata
    split_ratio: float = 0.9  # 90% output, 10% input
    border_style: str = "ascii"

def calculate_box_layout(width: int, height: int) -> LayoutGeometry:
    """
    Calculate dynamic box layout with proper border handling:

    1. Validate minimum terminal size
    2. Reserve 1 line for status at bottom
    3. Split remaining space 90/10 between output/input with border between
    4. Calculate outer boundaries (including borders) and inner text areas (excluding borders)
    5. Ensure all dimensions are positive for curses compatibility
    """

    # Validate minimum terminal size
    if width < MIN_SCREEN_WIDTH or height < MIN_SCREEN_HEIGHT:
        raise ValueError(f"Terminal too small: {width}x{height} (minimum: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT})")

    # Reserve status line at bottom
    status_height = 1
    content_height = height - status_height

    # Minimum viable content height check
    if content_height < 6:  # Need at least 3 lines output + 3 lines input
        raise ValueError(f"Insufficient terminal height: {height} (need at least {6 + status_height})")

    # Calculate split with border between output and input
    # Reserve 1 line for border between output and input
    usable_content_height = content_height - 1  # -1 for border line

    # Apply 90/10 split to usable space
    output_content_height = max(3, int(usable_content_height * 0.9))
    input_content_height = max(3, usable_content_height - output_content_height)

    # Adjust if total doesn't match (due to integer rounding)
    total_allocated = output_content_height + input_content_height
    if total_allocated < usable_content_height:
        input_content_height += usable_content_height - total_allocated

    # Calculate output box coordinates (top section)
    # Output box includes its own border
    output_outer_height = output_content_height + 2  # +2 for top and bottom borders
    output_box = BoxCoordinates(
        # Outer boundaries (including borders)
        top=0,
        left=0,
        bottom=output_outer_height - 1,
        right=width - 1,
        # Inner boundaries (text area excluding borders)
        inner_top=1,
        inner_left=1,
        inner_bottom=output_outer_height - 2,
        inner_right=width - 2,
        # Dimensions
        width=width,
        height=output_outer_height,
        inner_width=max(1, width - 2),  # Ensure positive width
        inner_height=max(1, output_content_height)  # Ensure positive height
    )

    # Calculate input box coordinates (middle section)
    # Input box starts after output box
    input_top = output_outer_height
    input_outer_height = input_content_height + 2  # +2 for top and bottom borders
    input_box = BoxCoordinates(
        # Outer boundaries (including borders)
        top=input_top,
        left=0,
        bottom=input_top + input_outer_height - 1,
        right=width - 1,
        # Inner boundaries (text area excluding borders)
        inner_top=input_top + 1,
        inner_left=1,
        inner_bottom=input_top + input_outer_height - 2,
        inner_right=width - 2,
        # Dimensions
        width=width,
        height=input_outer_height,
        inner_width=max(1, width - 2),  # Ensure positive width
        inner_height=max(1, input_content_height)  # Ensure positive height
    )

    # Calculate status line coordinates (bottom section)
    status_top = height - status_height
    status_line = BoxCoordinates(
        # Outer boundaries (status line has no borders)
        top=status_top,
        left=0,
        bottom=status_top,
        right=width - 1,
        # Inner boundaries (same as outer for status line)
        inner_top=status_top,
        inner_left=0,
        inner_bottom=status_top,
        inner_right=width - 1,
        # Dimensions
        width=width,
        height=status_height,
        inner_width=width,
        inner_height=status_height
    )

    # Final validation - ensure all dimensions are positive
    for box_name, box in [("output", output_box), ("input", input_box), ("status", status_line)]:
        if box.height <= 0 or box.width <= 0 or box.inner_height <= 0 or box.inner_width <= 0:
            raise ValueError(f"Invalid {box_name} box dimensions: outer={box.width}x{box.height}, inner={box.inner_width}x{box.inner_height}")

    return LayoutGeometry(
        terminal_height=height,
        terminal_width=width,
        output_box=output_box,
        input_box=input_box,
        status_line=status_line,
        split_ratio=0.9,
        border_style="ascii"
    )

class TerminalManager:
    """Dynamic terminal management with box coordinate system"""
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.width = 0
        self.height = 0
        self.last_check = 0
        self.too_small = False
        self.current_layout = None
    
    def check_resize(self) -> Tuple[bool, int, int]:
        """
        Check for terminal size changes
        Returns (resized, new_width, new_height)
        """
        current_time = time.time()
        
        # Check periodically (every 0.5 seconds)
        if current_time - self.last_check < 0.5:
            return False, self.width, self.height
        
        self.last_check = current_time
        
        try:
            new_height, new_width = self.stdscr.getmaxyx()
            
            if new_width != self.width or new_height != self.height:
                old_width, old_height = self.width, self.height
                self.width, self.height = new_width, new_height
                
                # Check minimum size
                if new_width < MIN_SCREEN_WIDTH or new_height < MIN_SCREEN_HEIGHT:
                    self.too_small = True
                    self.current_layout = None
                    return True, new_width, new_height
                else:
                    self.too_small = False
                    # Calculate new layout immediately
                    self.current_layout = calculate_box_layout(new_width, new_height)
                    return True, new_width, new_height
            
        except curses.error:
            pass
        
        return False, self.width, self.height
    
    def get_box_layout(self) -> LayoutGeometry:
        """Get current box layout for terminal size"""
        if self.current_layout is None and not self.too_small:
            self.current_layout = calculate_box_layout(self.width, self.height)
        return self.current_layout
    
    def get_size(self) -> Tuple[int, int]:
        """Get current terminal size"""
        return self.width, self.height
    
    def is_too_small(self) -> bool:
        """Check if terminal is too small"""
        return self.too_small
    
    def validate_size(self, width: int = None, height: int = None) -> bool:
        """Validate if given or current size meets minimum requirements"""
        check_width = width if width is not None else self.width
        check_height = height if height is not None else self.height
        return check_width >= MIN_SCREEN_WIDTH and check_height >= MIN_SCREEN_HEIGHT
    
    def show_too_small_message(self):
        """Show message when terminal is too small"""
        try:
            self.stdscr.clear()
            msg = f"Terminal too small! Need at least {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}, got {self.width}x{self.height}"
            if self.height > 0 and self.width > len(msg):
                self.stdscr.addstr(0, 0, msg)
            self.stdscr.refresh()
        except curses.error:
            pass

# Chunk 2/4 - uilib.py - Color and Display Management Components

# =============================================================================
# COLOR MANAGEMENT SYSTEM
# =============================================================================

class ColorTheme(Enum):
    """Available color themes"""
    CLASSIC = "classic"
    DARK = "dark" 
    BRIGHT = "bright"

class ColorManager:
    """Color management with theme switching"""
    
    def __init__(self, theme: ColorTheme = ColorTheme.CLASSIC):
        self.theme = theme
        self.colors_available = False
        
        # Color pair constants
        self.USER_COLOR = 1
        self.ASSISTANT_COLOR = 2
        self.SYSTEM_COLOR = 3
        self.ERROR_COLOR = 4
        self.BORDER_COLOR = 5
    
    def init_colors(self) -> bool:
        """Initialize colors based on current theme"""
        if not curses.has_colors():
            self.colors_available = False
            return False
        
        try:
            curses.start_color()
            curses.use_default_colors()
            
            # Define theme color schemes
            if self.theme == ColorTheme.CLASSIC:
                # Classic theme - blue/green/yellow/red
                curses.init_pair(self.USER_COLOR, curses.COLOR_CYAN, -1)
                curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_GREEN, -1)
                curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_YELLOW, -1)
                curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
                curses.init_pair(self.BORDER_COLOR, curses.COLOR_BLUE, -1)
                
            elif self.theme == ColorTheme.DARK:
                # Dark theme - muted colors
                curses.init_pair(self.USER_COLOR, curses.COLOR_WHITE, -1)
                curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_GREEN, -1)
                curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_BLUE, -1)
                curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
                curses.init_pair(self.BORDER_COLOR, curses.COLOR_WHITE, -1)
                
            elif self.theme == ColorTheme.BRIGHT:
                # Bright theme - vivid colors
                curses.init_pair(self.USER_COLOR, curses.COLOR_MAGENTA, -1)
                curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_CYAN, -1)
                curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_YELLOW, -1)
                curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
                curses.init_pair(self.BORDER_COLOR, curses.COLOR_MAGENTA, -1)
            
            self.colors_available = True
            return True
            
        except curses.error:
            self.colors_available = False
            return False
    
    def get_color(self, color_type: str) -> int:
        """Get color pair for message type"""
        if not self.colors_available:
            return 0
        
        color_map = {
            'user': self.USER_COLOR,
            'assistant': self.ASSISTANT_COLOR,
            'system': self.SYSTEM_COLOR,
            'error': self.ERROR_COLOR,
            'border': self.BORDER_COLOR
        }
        return color_map.get(color_type, 0)
    
    def change_theme(self, theme_name: str) -> bool:
        """Change color theme and reinitialize colors"""
        try:
            new_theme = ColorTheme(theme_name)
            self.theme = new_theme
            return self.init_colors()
        except ValueError:
            return False
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return [theme.value for theme in ColorTheme]

# =============================================================================
# DISPLAY MESSAGE SYSTEM
# =============================================================================

class DisplayMessage:
    """Message formatting with type-specific rendering"""
    
    def __init__(self, content: str, msg_type: str = "user"):
        self.content = content
        self.msg_type = msg_type
        self.wrapped_lines = []
    
    def wrap_content(self, width: int, preserve_paragraphs: bool = True) -> List[str]:
        """
        Wrap message content with paragraph preservation
        Returns list of wrapped lines with proper formatting
        """
        if not self.content:
            return [""]
        
        lines = []
        
        # Add header based on message type
        if self.msg_type == "user":
            header = "You: "
        elif self.msg_type == "assistant": 
            header = "Assistant: "
        elif self.msg_type == "system":
            header = "System: "
        elif self.msg_type == "error":
            header = "Error: "
        else:
            header = ""
        
        # Handle paragraph preservation
        if preserve_paragraphs:
            paragraphs = self.content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if i > 0:  # Add blank line between paragraphs
                    lines.append("")
                
                # Wrap paragraph
                if paragraph.strip():
                    wrapped = textwrap.wrap(
                        paragraph.strip(), 
                        width=width - len(header) if lines == [] else width - 4,
                        break_long_words=False,
                        break_on_hyphens=False
                    )
                    
                    for j, line in enumerate(wrapped):
                        if j == 0 and len(lines) == 0:
                            # First line gets header
                            lines.append(header + line)
                        else:
                            # Indent continuation lines
                            indent = " " * len(header) if len(lines) == 1 else "    "
                            lines.append(indent + line)
        else:
            # Simple wrap without paragraph handling
            wrapped = textwrap.wrap(
                self.content.strip(),
                width=width - len(header),
                break_long_words=False,
                break_on_hyphens=False
            )
            
            for i, line in enumerate(wrapped):
                if i == 0:
                    lines.append(header + line)
                else:
                    # Indent continuation lines
                    indent = " " * len(header)
                    lines.append(indent + line)
        
        self.wrapped_lines = lines
        return lines

class InputValidator:
    """Input validation with multi-line support"""
    
    def __init__(self, max_tokens: int = MAX_USER_INPUT_TOKENS):
        self.max_tokens = max_tokens
    
    def validate(self, text: str) -> Tuple[bool, str]:
        """Validate input text with specific error messages"""
        if not text.strip():
            return False, "Empty input"
        
        estimated_tokens = len(text) // 4
        if estimated_tokens > self.max_tokens:
            return False, f"Input too long: {estimated_tokens} tokens (max: {self.max_tokens})"
        
        # Check for reasonable line count
        lines = text.split('\n')
        if len(lines) > 20:
            return False, f"Too many lines: {len(lines)} (max: 20)"
        
        return True, ""

# Chunk 3/4 - uilib.py - Scroll Management System

# =============================================================================
# SCROLL MANAGEMENT SYSTEM
# =============================================================================

class ScrollManager:
    """Scrolling system with page navigation and indicators"""
    
    def __init__(self, window_height: int):
        self.scroll_offset = 0
        self.window_height = window_height
        self.in_scrollback = False
        self.max_scroll = 0
    
    def update_max_scroll(self, total_lines: int):
        """Update maximum scroll based on content"""
        self.max_scroll = max(0, total_lines - self.window_height + 1)
    
    def update_window_height(self, new_height: int):
        """Update window height and recalculate max scroll"""
        self.window_height = new_height
        # Max scroll will be recalculated on next update_max_scroll call
    
    def handle_line_scroll(self, direction: int) -> bool:
        """Handle single line scroll (arrow keys)"""
        old_offset = self.scroll_offset
        
        if direction < 0:  # Scroll up
            self.scroll_offset = max(0, self.scroll_offset - 1)
        else:  # Scroll down
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + 1)
        
        self.in_scrollback = (self.scroll_offset < self.max_scroll)
        return old_offset != self.scroll_offset
    
    def handle_page_scroll(self, direction: int) -> bool:
        """Handle page-based scroll (PgUp/PgDn)"""
        old_offset = self.scroll_offset
        page_size = max(1, self.window_height - 2)
        
        if direction < 0:  # Page up
            self.scroll_offset = max(0, self.scroll_offset - page_size)
        else:  # Page down
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + page_size)
        
        self.in_scrollback = (self.scroll_offset < self.max_scroll)
        return old_offset != self.scroll_offset
    
    def handle_home(self) -> bool:
        """Jump to top of history"""
        old_offset = self.scroll_offset
        self.scroll_offset = 0
        self.in_scrollback = (self.scroll_offset < self.max_scroll)
        return old_offset != self.scroll_offset
    
    def handle_end(self) -> bool:
        """Jump to bottom (most recent)"""
        old_offset = self.scroll_offset
        self.scroll_offset = self.max_scroll
        self.in_scrollback = False
        return old_offset != self.scroll_offset
    
    def auto_scroll_to_bottom(self):
        """Return to recent messages, exit scrollback mode"""
        self.scroll_offset = self.max_scroll
        self.in_scrollback = False
    
    def get_scroll_info(self) -> Dict[str, Any]:
        """Get scroll information for status display - FIXED to prevent KeyError"""
        # Handle case where scrolling isn't needed yet
        if self.max_scroll <= 0 or self.window_height <= 0:
            return {
                "in_scrollback": False,
                "percentage": 100,
                "offset": 0,
                "max": 0,
                "scroll_needed": False
            }
        
        # Calculate percentage safely
        try:
            percentage = int((self.scroll_offset / self.max_scroll) * 100) if self.max_scroll > 0 else 100
        except (ZeroDivisionError, TypeError):
            percentage = 100
        
        return {
            "in_scrollback": self.in_scrollback,
            "percentage": percentage,
            "offset": self.scroll_offset,
            "max": self.max_scroll,
            "scroll_needed": True
        }
    
    def get_visible_range(self) -> Tuple[int, int]:
        """Get the range of lines that should be visible"""
        start_idx = self.scroll_offset
        end_idx = start_idx + self.window_height - 1
        return start_idx, end_idx
    
    def jump_to_position(self, offset: int):
        """Jump to specific scroll position"""
        self.scroll_offset = max(0, min(offset, self.max_scroll))
        self.in_scrollback = (self.scroll_offset < self.max_scroll)

    def scroll_to_bottom(self):
        """Scroll to bottom (most recent messages) - alias for auto_scroll_to_bottom"""
        self.auto_scroll_to_bottom()

# Chunk 4/4 - uilib.py - Multi-line Input System

# =============================================================================
# MULTI-LINE INPUT SYSTEM
# =============================================================================

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
            
            # Merge lines
            self.lines[self.cursor_line - 1] = prev_line + current_line
            del self.lines[self.cursor_line]
            self.cursor_line -= 1
            
            self._adjust_scroll()
            return True
        
        return False
    
    def handle_arrow_keys(self, key: int) -> bool:
        """Handle arrow key navigation"""
        if key == curses.KEY_UP:
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
        
        elif key == curses.KEY_LEFT:
            if self.cursor_col > 0:
                self.cursor_col -= 1
                return True
            elif self.cursor_line > 0:
                # Move to end of previous line
                self.cursor_line -= 1
                self.cursor_col = len(self.lines[self.cursor_line])
                self._adjust_scroll()
                return True
        
        elif key == curses.KEY_RIGHT:
            if self.cursor_col < len(self.lines[self.cursor_line]):
                self.cursor_col += 1
                return True
            elif self.cursor_line < len(self.lines) - 1:
                # Move to start of next line
                self.cursor_line += 1
                self.cursor_col = 0
                self._adjust_scroll()
                return True
        
        return False
    
    def get_content(self) -> str:
        """Get complete input content as single string"""
        return '\n'.join(self.lines)
    
    def clear(self):
        """Clear all input content"""
        self.lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.scroll_offset = 0
    
    def set_content(self, content: str):
        """Set input content from string"""
        self.lines = content.split('\n') if content else [""]
        self.cursor_line = len(self.lines) - 1
        self.cursor_col = len(self.lines[self.cursor_line])
        self._adjust_scroll()
    
    def update_max_width(self, new_width: int):
        """Update maximum width for wrapping"""
        self.max_width = new_width
    
    def get_display_lines(self, available_width: int, available_height: int) -> List[str]:
        """Get lines for display with proper wrapping and scrolling"""
        display_lines = []
        
        for line in self.lines:
            if len(line) <= available_width:
                display_lines.append(line)
            else:
                # Wrap long lines
                wrapped = textwrap.wrap(
                    line, 
                    width=available_width,
                    break_long_words=True,
                    break_on_hyphens=False
                )
                display_lines.extend(wrapped if wrapped else [""])
        
        # Apply scroll offset
        start_idx = self.scroll_offset
        end_idx = start_idx + available_height
        return display_lines[start_idx:end_idx]
    
    def _wrap_current_line(self):
        """Wrap current line if it's too long"""
        current_line = self.lines[self.cursor_line]
        
        if len(current_line) < self.max_width - 5:
            return
        
        # Find best break point (space before cursor)
        break_point = self.cursor_col
        for i in range(self.cursor_col - 1, max(0, self.cursor_col - 20), -1):
            if current_line[i] == ' ':
                break_point = i
                break
        
        # Split line
        line_before = current_line[:break_point].rstrip()
        line_after = current_line[break_point:].lstrip()
        
        # Update lines
        self.lines[self.cursor_line] = line_before
        if line_after and len(self.lines) < self.max_lines:
            self.lines.insert(self.cursor_line + 1, line_after)
            
            # Adjust cursor position
            if self.cursor_col > break_point:
                self.cursor_line += 1
                self.cursor_col = self.cursor_col - break_point - (1 if current_line[break_point] == ' ' else 0)
            else:
                self.cursor_col = len(line_before)
        
        self._adjust_scroll()
    
    def _adjust_scroll(self):
        """Adjust scroll to keep cursor visible"""
        # This would be implemented based on available height
        # For now, keep it simple
        if self.cursor_line < self.scroll_offset:
            self.scroll_offset = self.cursor_line
        elif self.cursor_line >= self.scroll_offset + 5:  # Assume max 5 visible lines
            self.scroll_offset = self.cursor_line - 4

    def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position as (line, column)"""
        return (self.cursor_line, self.cursor_col)

    def handle_input(self, key: int) -> 'InputResult':
        """
        Handle input key and return result
        Returns InputResult object with submitted flag and content
        """
        from dataclasses import dataclass

        @dataclass
        class InputResult:
            submitted: bool
            content: str

        try:
            # Handle special keys first
            if key == ord('\n') or key == curses.KEY_ENTER or key == 10 or key == 13:
                # Enter key - check if should submit or add newline
                should_submit, content = self.handle_enter()
                if should_submit:
                    return InputResult(submitted=True, content=content)
                else:
                    return InputResult(submitted=False, content=self.get_content())

            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                # Backspace
                self.handle_backspace()
                return InputResult(submitted=False, content=self.get_content())

            elif key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
                # Arrow keys
                self.handle_arrow_keys(key)
                return InputResult(submitted=False, content=self.get_content())

            elif 32 <= key <= 126:  # Printable ASCII characters
                # Regular character input
                char = chr(key)
                self.insert_char(char)
                return InputResult(submitted=False, content=self.get_content())

            else:
                # Unhandled key
                return InputResult(submitted=False, content=self.get_content())

        except Exception as e:
            # Return safe fallback on error
            return InputResult(submitted=False, content=self.get_content())
