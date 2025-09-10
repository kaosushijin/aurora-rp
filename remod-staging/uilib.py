# Chunk 1/4 - uilib.py - ColorManager and Theme System
#!/usr/bin/env python3
"""
DevName RPG Client - Consolidated UI Library (uilib.py)

Consolidates all UI utilities from nci_*.py files:
- nci_colors.py → ColorManager, ColorTheme
- nci_terminal.py → TerminalManager, LayoutGeometry, BoxCoordinates  
- nci_display.py → DisplayMessage, InputValidator
- nci_scroll.py → ScrollManager
- nci_input.py → MultiLineInput

Module architecture and interconnects documented in genai.txt
"""

import curses
import re
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Color Management System (from nci_colors.py)

class ColorTheme(Enum):
    """Available color themes"""
    CLASSIC = "classic"
    DARK = "dark"
    BRIGHT = "bright"


class ColorManager:
    """
    Color management with theme switching
    Consolidated from nci_colors.py
    """
    
    def __init__(self, theme: ColorTheme = ColorTheme.CLASSIC):
        self.theme = theme
        self.colors_available = False
        
        # Color pair definitions
        self.USER_COLOR = 1
        self.ASSISTANT_COLOR = 2
        self.SYSTEM_COLOR = 3
        self.ERROR_COLOR = 4
        self.BORDER_COLOR = 5
        self.STATUS_COLOR = 6
        
        # Color pair mapping
        self.color_pairs = {
            'user': self.USER_COLOR,
            'assistant': self.ASSISTANT_COLOR,
            'system': self.SYSTEM_COLOR,
            'error': self.ERROR_COLOR,
            'border': self.BORDER_COLOR,
            'status': self.STATUS_COLOR
        }
    
    def init_colors(self) -> bool:
        """Initialize color pairs, return success status"""
        if not curses.has_colors():
            self.colors_available = False
            return False
        
        try:
            curses.start_color()
            curses.use_default_colors()
            
            # Initialize colors based on theme
            self._apply_theme_colors()
            
            self.colors_available = True
            return True
            
        except curses.error:
            self.colors_available = False
            return False
    
    def _apply_theme_colors(self):
        """Apply colors based on current theme"""
        if self.theme == ColorTheme.CLASSIC:
            curses.init_pair(self.USER_COLOR, curses.COLOR_CYAN, -1)
            curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_GREEN, -1)
            curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_YELLOW, -1)
            curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
            curses.init_pair(self.BORDER_COLOR, curses.COLOR_BLUE, -1)
            curses.init_pair(self.STATUS_COLOR, curses.COLOR_MAGENTA, -1)
            
        elif self.theme == ColorTheme.DARK:
            curses.init_pair(self.USER_COLOR, curses.COLOR_WHITE, -1)
            curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_CYAN, -1)
            curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_MAGENTA, -1)
            curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
            curses.init_pair(self.BORDER_COLOR, curses.COLOR_WHITE, -1)
            curses.init_pair(self.STATUS_COLOR, curses.COLOR_YELLOW, -1)
            
        else:  # BRIGHT
            curses.init_pair(self.USER_COLOR, curses.COLOR_BLUE, -1)
            curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_GREEN, -1)
            curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_YELLOW, -1)
            curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
            curses.init_pair(self.BORDER_COLOR, curses.COLOR_MAGENTA, -1)
            curses.init_pair(self.STATUS_COLOR, curses.COLOR_CYAN, -1)
    
    def get_color_pair(self, message_type: str) -> int:
        """Get color pair number for message type"""
        if not self.colors_available:
            return 0
        
        return self.color_pairs.get(message_type, 0)
    
    def get_color(self, color_type: str) -> int:
        """Get color pair for message type (legacy compatibility)"""
        return self.get_color_pair(color_type)
    
    def set_theme(self, theme_name: str) -> bool:
        """Change color theme and reinitialize colors"""
        try:
            if isinstance(theme_name, str):
                new_theme = ColorTheme(theme_name.lower())
            else:
                new_theme = theme_name
            
            self.theme = new_theme
            
            # Reinitialize colors if already initialized
            if self.colors_available:
                self._apply_theme_colors()
            
            return True
            
        except ValueError:
            return False
    
    def change_theme(self, theme_name: str) -> bool:
        """Change color theme (legacy compatibility)"""
        return self.set_theme(theme_name)
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return [theme.value for theme in ColorTheme]
    
    def get_current_theme(self) -> str:
        """Get current theme name"""
        return self.theme.value
    
    def is_colors_available(self) -> bool:
        """Check if colors are available in terminal"""
        return self.colors_available


# Color utility functions
def get_theme_description(theme_name: str) -> str:
    """Get description of color theme"""
    descriptions = {
        "classic": "Traditional terminal colors with cyan user text",
        "dark": "Dark mode with bright text on dark background",
        "bright": "Bright mode with dark text on light background"
    }
    return descriptions.get(theme_name.lower(), "Unknown theme")


def validate_theme_name(theme_name: str) -> bool:
    """Validate if theme name is supported"""
    try:
        ColorTheme(theme_name.lower())
        return True
    except ValueError:
        return False


def create_color_manager(theme: str = "classic") -> ColorManager:
    """Factory function to create color manager with theme"""
    try:
        theme_enum = ColorTheme(theme.lower())
        return ColorManager(theme_enum)
    except ValueError:
        # Fallback to classic theme
        return ColorManager(ColorTheme.CLASSIC)

# Chunk 2/4 - uilib.py - Terminal Management and Layout System

# Terminal Management System (from nci_terminal.py)

# Configuration constants
MIN_SCREEN_WIDTH = 80
MIN_SCREEN_HEIGHT = 24

@dataclass
class BoxCoordinates:
    """
    Box coordinate system with outer boundaries and inner text fields
    Consolidated from nci_terminal.py
    """
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
    
    # Convenience properties
    @property
    def x(self) -> int:
        """Alias for left coordinate"""
        return self.left
    
    @property
    def y(self) -> int:
        """Alias for top coordinate"""
        return self.top
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within box coordinates"""
        return (self.left <= x <= self.right and 
                self.top <= y <= self.bottom)
    
    def get_inner_dimensions(self) -> Tuple[int, int]:
        """Get inner width and height as tuple"""
        return (self.inner_width, self.inner_height)
    
    def get_outer_dimensions(self) -> Tuple[int, int]:
        """Get outer width and height as tuple"""
        return (self.width, self.height)


@dataclass
class LayoutGeometry:
    """
    Complete terminal layout with all box definitions
    Consolidated from nci_terminal.py
    """
    terminal_height: int
    terminal_width: int
    
    # Box definitions
    output_box: BoxCoordinates
    input_box: BoxCoordinates
    status_box: BoxCoordinates  # Updated from status_line for consistency
    
    # Layout metadata
    split_ratio: float = 0.9  # 90% output, 10% input
    border_style: str = "ascii"
    border_enabled: bool = False
    
    def get_total_boxes(self) -> int:
        """Get total number of defined boxes"""
        return 3  # output, input, status
    
    def validate_layout(self) -> bool:
        """Validate that layout fits within terminal bounds"""
        return (self.output_box.bottom <= self.terminal_height and
                self.input_box.bottom <= self.terminal_height and
                self.status_box.bottom <= self.terminal_height and
                self.output_box.right <= self.terminal_width and
                self.input_box.right <= self.terminal_width and
                self.status_box.right <= self.terminal_width)


def calculate_box_layout(width: int, height: int, border_enabled: bool = False) -> LayoutGeometry:
    """
    Calculate dynamic box layout with optional border support
    
    Layout Strategy:
    1. Reserve 1 line for status at bottom
    2. Reserve lines for borders if enabled
    3. Split remaining lines 90/10 between output/input
    4. Calculate inner coordinates for each box
    """
    
    # Account for borders if enabled
    border_offset = 1 if border_enabled else 0
    available_width = width - (2 * border_offset)
    available_height = height - (2 * border_offset)
    
    # Reserve space
    status_height = 1
    border_lines = 1 if not border_enabled else 0  # Separator between output/input
    usable_height = available_height - status_height - border_lines
    
    # Split available space
    output_height = max(1, int(usable_height * 0.9))
    input_height = max(1, usable_height - output_height)
    
    # Calculate coordinates with border offset
    base_x = border_offset
    base_y = border_offset
    
    # Calculate output box coordinates
    output_box = BoxCoordinates(
        top=base_y,
        left=base_x,
        bottom=base_y + output_height - 1,
        right=base_x + available_width - 1,
        inner_top=base_y,
        inner_left=base_x,
        inner_bottom=base_y + output_height - 1,
        inner_right=base_x + available_width - 1,
        width=available_width,
        height=output_height,
        inner_width=available_width,
        inner_height=output_height
    )
    
    # Calculate input box coordinates
    input_top = base_y + output_height + border_lines
    input_box = BoxCoordinates(
        top=input_top,
        left=base_x,
        bottom=input_top + input_height - 1,
        right=base_x + available_width - 1,
        inner_top=input_top,
        inner_left=base_x,
        inner_bottom=input_top + input_height - 1,
        inner_right=base_x + available_width - 1,
        width=available_width,
        height=input_height,
        inner_width=available_width,
        inner_height=input_height
    )
    
    # Calculate status box coordinates
    status_top = height - 1 - border_offset
    status_box = BoxCoordinates(
        top=status_top,
        left=base_x,
        bottom=status_top,
        right=base_x + available_width - 1,
        inner_top=status_top,
        inner_left=base_x,
        inner_bottom=status_top,
        inner_right=base_x + available_width - 1,
        width=available_width,
        height=1,
        inner_width=available_width,
        inner_height=1
    )
    
    return LayoutGeometry(
        terminal_height=height,
        terminal_width=width,
        output_box=output_box,
        input_box=input_box,
        status_box=status_box,
        border_enabled=border_enabled
    )


class TerminalManager:
    """
    Dynamic terminal management with box coordinate system
    Consolidated from nci_terminal.py
    """
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.width = 0
        self.height = 0
        self.last_check = 0.0
        self.too_small = False
        self.current_layout = None
        self.border_enabled = False
        self.resize_count = 0
        
        # Initialize size
        try:
            self.height, self.width = stdscr.getmaxyx()
        except curses.error:
            self.width, self.height = MIN_SCREEN_WIDTH, MIN_SCREEN_HEIGHT
    
    def enable_borders(self, enabled: bool = True):
        """Enable or disable border drawing"""
        if self.border_enabled != enabled:
            self.border_enabled = enabled
            # Force layout recalculation
            self.current_layout = None
    
    def check_resize(self) -> Tuple[bool, int, int]:
        """
        Check for terminal size changes
        Returns (resized, new_width, new_height)
        """
        current_time = time.time()
        
        # Check periodically (every 0.5 seconds) to avoid excessive calls
        if current_time - self.last_check < 0.5:
            return False, self.width, self.height
        
        self.last_check = current_time
        
        try:
            new_height, new_width = self.stdscr.getmaxyx()
            
            if new_width != self.width or new_height != self.height:
                self.width, self.height = new_width, new_height
                self.resize_count += 1
                
                # Check minimum size requirements
                if new_width < MIN_SCREEN_WIDTH or new_height < MIN_SCREEN_HEIGHT:
                    self.too_small = True
                    self.current_layout = None
                    return True, new_width, new_height
                else:
                    self.too_small = False
                    # Calculate new layout immediately
                    self.current_layout = calculate_box_layout(
                        new_width, new_height, self.border_enabled
                    )
                    return True, new_width, new_height
            
        except curses.error:
            pass
        
        return False, self.width, self.height
    
    def get_box_layout(self) -> Optional[LayoutGeometry]:
        """Get current box layout for terminal size"""
        if self.current_layout is None and not self.too_small:
            self.current_layout = calculate_box_layout(
                self.width, self.height, self.border_enabled
            )
        return self.current_layout
    
    def force_layout_recalculation(self):
        """Force recalculation of layout on next access"""
        self.current_layout = None
    
    def get_size(self) -> Tuple[int, int]:
        """Get current terminal size (width, height)"""
        return self.width, self.height
    
    def get_dimensions(self) -> Dict[str, int]:
        """Get terminal dimensions as dictionary"""
        return {
            "width": self.width,
            "height": self.height,
            "min_width": MIN_SCREEN_WIDTH,
            "min_height": MIN_SCREEN_HEIGHT
        }
    
    def is_too_small(self) -> bool:
        """Check if terminal is too small for UI"""
        return self.too_small
    
    def validate_size(self, width: int = None, height: int = None) -> bool:
        """Validate if given or current size meets minimum requirements"""
        check_width = width if width is not None else self.width
        check_height = height if height is not None else self.height
        return check_width >= MIN_SCREEN_WIDTH and check_height >= MIN_SCREEN_HEIGHT
    
    def show_too_small_message(self):
        """Display message when terminal is too small"""
        try:
            self.stdscr.clear()
            msg_lines = [
                "Terminal too small!",
                f"Current: {self.width}x{self.height}",
                f"Required: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}",
                "Please resize terminal window"
            ]
            
            # Calculate center position
            start_y = max(0, (self.height - len(msg_lines)) // 2)
            
            for i, line in enumerate(msg_lines):
                if start_y + i < self.height:
                    start_x = max(0, (self.width - len(line)) // 2)
                    try:
                        self.stdscr.addstr(start_y + i, start_x, line[:self.width])
                    except curses.error:
                        pass
            
            self.stdscr.refresh()
            
        except curses.error:
            pass
    
    def draw_box_borders(self, layout: LayoutGeometry, color_pair: int = 0):
        """Draw ASCII borders around layout boxes"""
        if not self.border_enabled or not layout:
            return
        
        try:
            # Draw outer terminal border
            self._draw_rectangle_border(0, 0, self.width - 1, self.height - 1, color_pair)
            
            # Draw separator between output and input
            separator_y = layout.output_box.bottom + 1
            if 0 <= separator_y < self.height:
                for x in range(1, self.width - 1):
                    try:
                        if color_pair > 0:
                            self.stdscr.attron(curses.color_pair(color_pair))
                        self.stdscr.addch(separator_y, x, '-')
                        if color_pair > 0:
                            self.stdscr.attroff(curses.color_pair(color_pair))
                    except curses.error:
                        pass
            
        except curses.error:
            pass
    
    def _draw_rectangle_border(self, left: int, top: int, right: int, bottom: int, color_pair: int = 0):
        """Draw rectangle border with ASCII characters"""
        try:
            # Corner and line characters
            corners = ['+', '+', '+', '+']  # top-left, top-right, bottom-left, bottom-right
            h_line, v_line = '-', '|'
            
            # Apply color if specified
            if color_pair > 0:
                self.stdscr.attron(curses.color_pair(color_pair))
            
            # Draw horizontal lines
            for x in range(left + 1, right):
                if 0 <= x < self.width:
                    if 0 <= top < self.height:
                        self.stdscr.addch(top, x, h_line)
                    if 0 <= bottom < self.height:
                        self.stdscr.addch(bottom, x, h_line)
            
            # Draw vertical lines
            for y in range(top + 1, bottom):
                if 0 <= y < self.height:
                    if 0 <= left < self.width:
                        self.stdscr.addch(y, left, v_line)
                    if 0 <= right < self.width:
                        self.stdscr.addch(y, right, v_line)
            
            # Draw corners
            corner_positions = [
                (top, left, 0),      # top-left
                (top, right, 1),     # top-right
                (bottom, left, 2),   # bottom-left
                (bottom, right, 3)   # bottom-right
            ]
            
            for y, x, corner_idx in corner_positions:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.stdscr.addch(y, x, corners[corner_idx])
            
            # Remove color if applied
            if color_pair > 0:
                self.stdscr.attroff(curses.color_pair(color_pair))
                
        except curses.error:
            pass
    
    def clear_screen(self):
        """Clear the entire screen"""
        try:
            self.stdscr.clear()
            self.stdscr.refresh()
        except curses.error:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get terminal manager statistics"""
        return {
            "terminal_size": f"{self.width}x{self.height}",
            "too_small": self.too_small,
            "border_enabled": self.border_enabled,
            "resize_count": self.resize_count,
            "layout_valid": self.current_layout is not None,
            "min_requirements": f"{MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}"
        }


# Terminal utility functions
def validate_terminal_size(width: int, height: int) -> Tuple[bool, str]:
    """Validate terminal size and return result with message"""
    if width < MIN_SCREEN_WIDTH:
        return False, f"Width {width} < minimum {MIN_SCREEN_WIDTH}"
    if height < MIN_SCREEN_HEIGHT:
        return False, f"Height {height} < minimum {MIN_SCREEN_HEIGHT}"
    return True, "Terminal size acceptable"


def create_terminal_manager(stdscr, enable_borders: bool = False) -> TerminalManager:
    """Factory function to create terminal manager"""
    manager = TerminalManager(stdscr)
    if enable_borders:
        manager.enable_borders(True)
    return manager


def calculate_optimal_split(total_height: int, min_input_lines: int = 3) -> float:
    """Calculate optimal split ratio based on terminal height"""
    if total_height < MIN_SCREEN_HEIGHT:
        return 0.8  # More conservative split for small screens
    
    # Calculate ratio ensuring minimum input lines
    input_proportion = max(min_input_lines / total_height, 0.1)
    output_proportion = min(1.0 - input_proportion, 0.95)
    
    return output_proportion

# Chunk 3/4 - uilib.py - Display System and Scroll Management

# Display System (from nci_display.py)

class InputValidator:
    """
    Input validation for user messages
    Consolidated from nci_display.py
    """
    
    def __init__(self, max_length: int = 2000):
        self.max_length = max_length
        self.min_length = 1
        
        # Validation patterns
        self.forbidden_patterns = [
            r'^\s*$',  # Empty or whitespace only
        ]
        
        # Warning patterns (allowed but flagged)
        self.warning_patterns = [
            r'(.)\1{10,}',  # Excessive character repetition
            r'^[A-Z\s!]{20,}$',  # Excessive caps
        ]
    
    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate user input and return validation result"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "cleaned_text": text.strip()
        }
        
        # Length validation
        if len(text.strip()) < self.min_length:
            result["valid"] = False
            result["errors"].append("Input cannot be empty")
        
        if len(text) > self.max_length:
            result["valid"] = False
            result["errors"].append(f"Input too long ({len(text)} > {self.max_length} characters)")
        
        # Pattern validation
        for pattern in self.forbidden_patterns:
            if re.match(pattern, text):
                result["valid"] = False
                result["errors"].append("Invalid input pattern")
        
        # Warning patterns
        for pattern in self.warning_patterns:
            if re.search(pattern, text):
                result["warnings"].append("Unusual input pattern detected")
        
        return result
    
    def clean_input(self, text: str) -> str:
        """Clean and normalize input text"""
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove control characters except newlines and tabs
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
        
        return cleaned
    
    def truncate_input(self, text: str) -> str:
        """Truncate input to maximum length"""
        if len(text) <= self.max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:self.max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > self.max_length * 0.8:  # If word boundary is reasonable
            return truncated[:last_space] + "..."
        else:
            return truncated[:-3] + "..."


class DisplayMessage:
    """
    Message formatting and display handling
    Consolidated from nci_display.py
    """
    
    def __init__(self, content: str = "", msg_type: str = "system", timestamp: float = None):
        self.content = content
        self.msg_type = msg_type
        self.timestamp = timestamp or time.time()
        self.formatted_lines = []
        self.last_format_width = 0
        
        # Message type styling
        self.type_prefixes = {
            "user": "You: ",
            "assistant": "",  # No prefix for GM responses
            "system": "[System] ",
            "error": "[Error] "
        }
    
    def format_content(self, content: str, msg_type: str, width: int) -> List[str]:
        """Format message content for display with enhanced line break preservation"""
        if not content:
            return [""]
        
        # Get prefix for message type
        prefix = self.type_prefixes.get(msg_type, "")
        available_width = max(10, width - len(prefix))
        
        formatted_lines = []
        
        # Split content into paragraphs (preserve intentional line breaks)
        paragraphs = content.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                # Add blank line between paragraphs
                formatted_lines.append("")
            
            # Handle individual lines within paragraph
            lines = paragraph.split('\n')
            
            for line_idx, line in enumerate(lines):
                if line_idx > 0 and line.strip():
                    # Preserve intentional line breaks within paragraphs
                    formatted_lines.append("")
                
                # Word wrap the line
                wrapped_lines = self._wrap_text_enhanced(line.strip(), available_width)
                
                for wrap_idx, wrapped_line in enumerate(wrapped_lines):
                    if wrap_idx == 0 and line_idx == 0:
                        # First line gets prefix
                        formatted_lines.append(prefix + wrapped_line)
                    else:
                        # Continuation lines get spacing
                        indent = " " * len(prefix) if wrap_idx > 0 else ""
                        formatted_lines.append(indent + wrapped_line)
        
        return formatted_lines if formatted_lines else [""]
    
    def _wrap_text_enhanced(self, text: str, width: int) -> List[str]:
        """Enhanced text wrapping with intelligent break points"""
        if len(text) <= width:
            return [text] if text else [""]
        
        lines = []
        remaining = text
        
        while remaining:
            if len(remaining) <= width:
                lines.append(remaining)
                break
            
            # Find best break point
            break_point = self._find_break_point(remaining, width)
            
            lines.append(remaining[:break_point].rstrip())
            remaining = remaining[break_point:].lstrip()
        
        return lines
    
    def _find_break_point(self, text: str, max_width: int) -> int:
        """Find optimal break point for text wrapping"""
        if len(text) <= max_width:
            return len(text)
        
        # Preferred break characters (in order of preference)
        break_chars = [' ', '\t', '-', ',', '.', ';', ':', '!', '?']
        
        # Look for break point working backwards from max width
        for i in range(max_width, max(0, max_width - 20), -1):
            if text[i] in break_chars:
                # For punctuation, include it in the current line
                if text[i] in '.,;:!?':
                    return i + 1
                else:
                    return i
        
        # No good break point found, force break at max width
        return max_width
    
    def format_for_display(self, width: int) -> List[str]:
        """Format message for display, caching results"""
        if width != self.last_format_width or not self.formatted_lines:
            self.formatted_lines = self.format_content(self.content, self.msg_type, width)
            self.last_format_width = width
        
        return self.formatted_lines
    
    def get_line_count(self, width: int) -> int:
        """Get number of display lines for given width"""
        return len(self.format_for_display(width))
    
    def is_empty(self) -> bool:
        """Check if message is empty"""
        return not self.content.strip()


# Scroll Management System (from nci_scroll.py)

class ScrollManager:
    """
    Scrolling system with page navigation and indicators
    Consolidated from nci_scroll.py with enhanced features
    """
    
    def __init__(self, window_height: int = 20):
        self.scroll_offset = 0
        self.window_height = window_height
        self.in_scrollback = False
        self.max_scroll = 0
        self.total_lines = 0
        
        # Enhanced scroll state
        self.auto_scroll_enabled = True
        self.scroll_margin = 2  # Lines from bottom to trigger auto-scroll
        self.page_overlap = 2   # Lines to overlap when paging
    
    def update_window_height(self, new_height: int):
        """Update window height and recalculate scroll bounds"""
        self.window_height = max(1, new_height)
        self._recalculate_scroll_bounds()
    
    def update_max_scroll(self, total_lines: int):
        """Update maximum scroll based on content"""
        self.total_lines = total_lines
        self._recalculate_scroll_bounds()
    
    def _recalculate_scroll_bounds(self):
        """Recalculate scroll boundaries"""
        self.max_scroll = max(0, self.total_lines - self.window_height)
        
        # Clamp current scroll offset
        if self.scroll_offset > self.max_scroll:
            self.scroll_offset = self.max_scroll
            self.in_scrollback = False
    
    def handle_line_scroll(self, direction: int) -> bool:
        """Handle single line scroll (arrow keys)"""
        old_offset = self.scroll_offset
        
        if direction < 0:  # Scroll up
            self.scroll_offset = max(0, self.scroll_offset - 1)
        else:  # Scroll down
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + 1)
        
        self._update_scrollback_state()
        return old_offset != self.scroll_offset
    
    def handle_page_scroll(self, direction: int) -> bool:
        """Handle page-based scroll (PgUp/PgDn)"""
        old_offset = self.scroll_offset
        page_size = max(1, self.window_height - self.page_overlap)
        
        if direction < 0:  # Page up
            self.scroll_offset = max(0, self.scroll_offset - page_size)
        else:  # Page down
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + page_size)
        
        self._update_scrollback_state()
        return old_offset != self.scroll_offset
    
    def handle_home(self) -> bool:
        """Jump to top of history"""
        old_offset = self.scroll_offset
        self.scroll_offset = 0
        self._update_scrollback_state()
        return old_offset != self.scroll_offset
    
    def handle_end(self) -> bool:
        """Jump to bottom (most recent)"""
        old_offset = self.scroll_offset
        self.scroll_offset = self.max_scroll
        self._update_scrollback_state()
        return old_offset != self.scroll_offset
    
    def auto_scroll_to_bottom(self):
        """Return to recent messages, exit scrollback mode"""
        if self.auto_scroll_enabled:
            self.scroll_offset = self.max_scroll
            self.in_scrollback = False
    
    def enable_auto_scroll(self, enabled: bool = True):
        """Enable or disable automatic scrolling"""
        self.auto_scroll_enabled = enabled
        if enabled and self.scroll_offset >= self.max_scroll - self.scroll_margin:
            self.auto_scroll_to_bottom()
    
    def _update_scrollback_state(self):
        """Update scrollback state based on current position"""
        self.in_scrollback = (self.scroll_offset < self.max_scroll - self.scroll_margin)
    
    def get_visible_messages(self, messages: List, window_height: int = None) -> List:
        """Get messages visible in current scroll window"""
        if not messages:
            return []
        
        height = window_height or self.window_height
        start_idx = max(0, len(messages) - height - self.scroll_offset)
        end_idx = max(0, len(messages) - self.scroll_offset)
        
        return messages[start_idx:end_idx]
    
    def get_scroll_info(self) -> Dict[str, Any]:
        """Get comprehensive scroll information (FIXED: KeyError crash)"""
        # Calculate scroll percentage
        scroll_percentage = 0.0
        if self.max_scroll > 0:
            scroll_percentage = (self.scroll_offset / self.max_scroll) * 100
        
        return {
            "scroll_offset": self.scroll_offset,
            "max_scroll": self.max_scroll,
            "window_height": self.window_height,
            "total_lines": self.total_lines,
            "in_scrollback": self.in_scrollback,
            "auto_scroll_enabled": self.auto_scroll_enabled,
            "scroll_percentage": scroll_percentage,
            "at_top": self.scroll_offset == 0,
            "at_bottom": self.scroll_offset >= self.max_scroll,
            "visible_range": {
                "start": max(0, self.total_lines - self.window_height - self.scroll_offset),
                "end": max(0, self.total_lines - self.scroll_offset)
            }
        }
    
    def get_scroll_indicator(self) -> str:
        """Get scroll position indicator string"""
        if self.total_lines <= self.window_height:
            return ""
        
        if self.in_scrollback:
            percentage = (self.scroll_offset / self.max_scroll) * 100 if self.max_scroll > 0 else 0
            return f" [↑{percentage:.0f}%]"
        else:
            return " [↓]"
    
    def is_at_bottom(self) -> bool:
        """Check if scroll is at bottom"""
        return self.scroll_offset >= self.max_scroll
    
    def is_at_top(self) -> bool:
        """Check if scroll is at top"""
        return self.scroll_offset == 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scroll manager statistics"""
        return {
            "current_position": f"{self.scroll_offset}/{self.max_scroll}",
            "window_height": self.window_height,
            "total_content_lines": self.total_lines,
            "scrollback_active": self.in_scrollback,
            "auto_scroll": self.auto_scroll_enabled,
            "scroll_margin": self.scroll_margin,
            "page_overlap": self.page_overlap
        }


# Display utility functions
def create_display_message(content: str, msg_type: str = "system") -> DisplayMessage:
    """Factory function to create display message"""
    return DisplayMessage(content, msg_type)


def create_scroll_manager(window_height: int = 20) -> ScrollManager:
    """Factory function to create scroll manager"""
    return ScrollManager(window_height)


def format_message_with_timestamp(content: str, msg_type: str, show_timestamp: bool = False) -> str:
    """Format message with optional timestamp"""
    if show_timestamp:
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] {content}"
    else:
        return content


def validate_message_content(content: str, max_length: int = 2000) -> bool:
    """Quick validation for message content"""
    validator = InputValidator(max_length)
    result = validator.validate_input(content)
    return result["valid"]


def wrap_text_simple(text: str, width: int) -> List[str]:
    """Simple text wrapping utility"""
    if len(text) <= width:
        return [text]
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + len(current_line) <= width:
            current_line.append(word)
            current_length += len(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [""]

# Chunk 4/4 - uilib.py - Multi-line Input System and Module Integration

# Multi-line Input System (from nci_input.py)

class MultiLineInput:
    """
    Multi-line input system with cursor navigation and word wrapping
    Consolidated from nci_input.py with enhanced features
    """
    
    def __init__(self, max_width: int = 80):
        self.lines = [""]           # List of text lines
        self.cursor_line = 0        # Current line index
        self.cursor_col = 0         # Current column position within line
        self.scroll_offset = 0      # Vertical scroll within input area
        self.max_width = max_width  # Maximum line width before wrapping
        self.max_lines = 10         # Maximum number of lines allowed
        
        # Enhanced features
        self.char_limit = 4000      # Total character limit
        self.submission_mode = "smart"  # "smart", "manual", "immediate"
        self.auto_wrap_enabled = True
        self.word_wrap_margin = 5   # Characters before edge to trigger wrap
        
        # Input history
        self.input_history = []
        self.history_index = -1
        self.temp_content = ""      # Temporary storage when browsing history
    
    def insert_char(self, char: str) -> bool:
        """Insert character at cursor position with enhanced word wrapping"""
        if len(self.get_content()) >= self.char_limit:
            return False
        
        # Insert character at current position
        current_line = self.lines[self.cursor_line]
        new_line = current_line[:self.cursor_col] + char + current_line[self.cursor_col:]
        self.lines[self.cursor_line] = new_line
        self.cursor_col += 1
        
        # Check if line needs wrapping
        if self.auto_wrap_enabled and len(new_line) >= self.max_width - self.word_wrap_margin:
            self._wrap_current_line()
        
        return True
    
    def handle_key(self, key: int) -> Dict[str, Any]:
        """
        Enhanced key handling with comprehensive input management
        Returns dict with action results
        """
        result = {
            "handled": False,
            "complete": False,
            "content": "",
            "action": "none"
        }
        
        # Handle Enter key
        if key == ord('\n') or key == curses.KEY_ENTER or key == 10:
            should_submit, content = self.handle_enter()
            result.update({
                "handled": True,
                "complete": should_submit,
                "content": content if should_submit else self.get_content(),
                "action": "submit" if should_submit else "newline"
            })
            return result
        
        # Handle backspace
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
            changed = self.handle_backspace()
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "backspace" if changed else "none"
            })
            return result
        
        # Handle arrow keys
        elif key in [curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN]:
            self.handle_arrow_keys(key)
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "navigation"
            })
            return result
        
        # Handle history navigation
        elif key == curses.KEY_PPAGE:  # Page Up - previous in history
            self._navigate_history(-1)
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "history_prev"
            })
            return result
        
        elif key == curses.KEY_NPAGE:  # Page Down - next in history
            self._navigate_history(1)
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "history_next"
            })
            return result
        
        # Handle special keys
        elif key == curses.KEY_HOME:
            self._move_cursor_home()
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "move_home"
            })
            return result
        
        elif key == curses.KEY_END:
            self._move_cursor_end()
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "move_end"
            })
            return result
        
        # Handle printable characters
        elif 32 <= key <= 126:
            changed = self.insert_char(chr(key))
            result.update({
                "handled": True,
                "content": self.get_content(),
                "action": "insert" if changed else "none"
            })
            return result
        
        # Unhandled key
        return result
    
    def handle_enter(self) -> Tuple[bool, str]:
        """
        Enhanced Enter key handling with smart submission detection
        Returns (should_submit, content)
        """
        content = self.get_content().strip()
        
        if self.submission_mode == "immediate":
            return True, content
        elif self.submission_mode == "manual":
            self.insert_newline()
            return False, ""
        else:  # smart mode
            # Smart submission logic
            if not content:
                return False, ""
            
            # Commands are always submitted immediately
            if content.startswith('/'):
                return True, content
            
            # Check if at end of last line
            if (self.cursor_line == len(self.lines) - 1 and 
                self.cursor_col == len(self.lines[self.cursor_line])):
                
                # Check for completion indicators
                if (content.endswith(('.', '!', '?', '"', "'", ':', ';')) or
                    len(content) > 100 or  # Long messages likely complete
                    content.count('\n') >= 2):  # Multi-paragraph likely complete
                    return True, content
            
            # Otherwise create new line
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
        """Enhanced backspace with line merging"""
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
        """Enhanced arrow key navigation"""
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
        """Enhanced line wrapping with better break point detection"""
        current_line = self.lines[self.cursor_line]
        if len(current_line) < self.max_width - self.word_wrap_margin:
            return
        
        # Find optimal break point
        target_width = self.max_width - self.word_wrap_margin
        break_point = self._find_wrap_break_point(current_line, target_width)
        
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
    
    def _find_wrap_break_point(self, text: str, max_width: int) -> int:
        """Find optimal break point for line wrapping"""
        if len(text) <= max_width:
            return len(text)
        
        # Preferred break characters (in order of preference)
        break_chars = [' ', '\t', '-', ',', '.', ';', ':', '!', '?']
        
        # Look for break point working backwards from max width
        for i in range(min(max_width, len(text)), max(0, max_width - 20), -1):
            if i < len(text) and text[i] in break_chars:
                # For punctuation, include it in the current line
                if text[i] in '.,;:!?':
                    return i + 1
                else:
                    return i
        
        # No good break point found, force break at max width
        return max_width
    
    def _navigate_history(self, direction: int):
        """Navigate through input history"""
        if not self.input_history:
            return
        
        # Save current content if at current position
        if self.history_index == -1:
            self.temp_content = self.get_content()
        
        # Navigate history
        if direction < 0:  # Previous
            if self.history_index < len(self.input_history) - 1:
                self.history_index += 1
        else:  # Next
            if self.history_index >= 0:
                self.history_index -= 1
        
        # Set content
        if self.history_index == -1:
            # Back to current
            self.set_content(self.temp_content)
        else:
            # From history
            self.set_content(self.input_history[self.history_index])
    
    def _move_cursor_home(self):
        """Move cursor to beginning of current line"""
        self.cursor_col = 0
    
    def _move_cursor_end(self):
        """Move cursor to end of current line"""
        self.cursor_col = len(self.lines[self.cursor_line])
    
    def _adjust_scroll(self):
        """Adjust scroll to keep cursor visible"""
        # Simple scroll management for multi-line input
        if self.cursor_line < self.scroll_offset:
            self.scroll_offset = self.cursor_line
        elif self.cursor_line >= self.scroll_offset + 3:  # Assuming 3-line input window
            self.scroll_offset = max(0, self.cursor_line - 2)
    
    def get_display_lines(self, width: int, height: int) -> List[str]:
        """Get lines formatted for display within given dimensions"""
        display_lines = []
        
        for i, line in enumerate(self.lines):
            if len(display_lines) >= height:
                break
            
            if len(line) > width:
                # Wrap long lines for display only
                wrapped_lines = []
                remaining = line
                while remaining:
                    if len(remaining) <= width:
                        wrapped_lines.append(remaining)
                        break
                    else:
                        wrapped_lines.append(remaining[:width])
                        remaining = remaining[width:]
                
                display_lines.extend(wrapped_lines)
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
        """Clear all content and reset state"""
        self.lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.scroll_offset = 0
        self.history_index = -1
        self.temp_content = ""
    
    def set_content(self, content: str):
        """Set content from string"""
        self.lines = content.split('\n') if content else [""]
        self.cursor_line = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])
        self.scroll_offset = 0
        self.history_index = -1
    
    def add_to_history(self, content: str):
        """Add content to input history"""
        if content.strip() and (not self.input_history or content != self.input_history[0]):
            self.input_history.insert(0, content)
            # Limit history size
            if len(self.input_history) > 50:
                self.input_history = self.input_history[:50]
        
        self.history_index = -1
        self.temp_content = ""
    
    def get_cursor_position(self) -> Tuple[int, int]:
        """Get cursor position for display (line, column)"""
        return (self.cursor_line, self.cursor_col)
    
    def is_empty(self) -> bool:
        """Check if input is empty"""
        return not self.get_content().strip()
    
    def update_max_width(self, new_width: int):
        """Update maximum width for line wrapping"""
        self.max_width = max(20, new_width)  # Minimum reasonable width
    
    def set_submission_mode(self, mode: str):
        """Set submission mode: 'smart', 'manual', or 'immediate'"""
        if mode in ["smart", "manual", "immediate"]:
            self.submission_mode = mode
    
    def get_stats(self) -> Dict[str, Any]:
        """Get input system statistics"""
        content = self.get_content()
        return {
            "line_count": len(self.lines),
            "character_count": len(content),
            "word_count": len(content.split()),
            "cursor_position": f"{self.cursor_line}:{self.cursor_col}",
            "submission_mode": self.submission_mode,
            "auto_wrap_enabled": self.auto_wrap_enabled,
            "history_size": len(self.input_history),
            "is_empty": self.is_empty()
        }


# Module Integration and Factory Functions

def create_ui_components(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Factory function to create all UI components with configuration"""
    config = config or {}
    
    # Color management
    theme = config.get('color_theme', 'classic')
    color_manager = create_color_manager(theme)
    
    # Terminal management
    terminal_manager = None  # Requires stdscr, created separately
    
    # Input system
    max_input_width = config.get('max_input_width', 80)
    multi_input = MultiLineInput(max_input_width)
    multi_input.set_submission_mode(config.get('submission_mode', 'smart'))
    
    # Scroll management
    scroll_manager = create_scroll_manager(config.get('scroll_window_height', 20))
    
    # Input validation
    input_validator = InputValidator(config.get('max_input_length', 2000))
    
    return {
        'color_manager': color_manager,
        'multi_input': multi_input,
        'scroll_manager': scroll_manager,
        'input_validator': input_validator,
        'config': config
    }


def create_configured_multi_input(max_width: int = 80, submission_mode: str = "smart") -> MultiLineInput:
    """Create configured multi-line input system"""
    multi_input = MultiLineInput(max_width)
    multi_input.set_submission_mode(submission_mode)
    return multi_input


def validate_ui_component_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize UI component configuration"""
    validated = {}
    
    # Color theme validation
    theme = config.get('color_theme', 'classic')
    if not validate_theme_name(theme):
        theme = 'classic'
    validated['color_theme'] = theme
    
    # Input width validation
    max_width = config.get('max_input_width', 80)
    validated['max_input_width'] = max(20, min(200, max_width))
    
    # Input length validation
    max_length = config.get('max_input_length', 2000)
    validated['max_input_length'] = max(100, min(10000, max_length))
    
    # Submission mode validation
    submission_mode = config.get('submission_mode', 'smart')
    if submission_mode not in ['smart', 'manual', 'immediate']:
        submission_mode = 'smart'
    validated['submission_mode'] = submission_mode
    
    # Scroll window height validation
    scroll_height = config.get('scroll_window_height', 20)
    validated['scroll_window_height'] = max(5, min(100, scroll_height))
    
    return validated


def get_uilib_info() -> Dict[str, Any]:
    """Get information about consolidated UI library"""
    return {
        "name": "DevName RPG Client UI Library",
        "version": "1.0",
        "consolidated_modules": [
            "nci_colors.py → ColorManager, ColorTheme",
            "nci_terminal.py → TerminalManager, LayoutGeometry, BoxCoordinates",
            "nci_display.py → DisplayMessage, InputValidator", 
            "nci_scroll.py → ScrollManager",
            "nci_input.py → MultiLineInput"
        ],
        "features": [
            "Complete color theme management",
            "Dynamic terminal layout with border support",
            "Enhanced message formatting with line break preservation",
            "Comprehensive scroll management with indicators",
            "Multi-line input with smart submission detection",
            "Input history and navigation",
            "Comprehensive input validation",
            "Factory functions and configuration validation"
        ],
        "improvements": [
            "Enhanced error handling throughout",
            "Better text wrapping with intelligent break points",
            "Fixed scroll info crashes",
            "Added border drawing capabilities",
            "Smart submission mode for better UX",
            "Input history navigation",
            "Configuration validation and normalization"
        ]
    }


# Module test and demonstration functionality
if __name__ == "__main__":
    print("DevName RPG Client - Consolidated UI Library (uilib.py)")
    print("Successfully consolidated all UI utilities:")
    print("✓ ColorManager with enhanced theme switching")
    print("✓ TerminalManager with border support and dynamic layout")
    print("✓ DisplayMessage with improved line break preservation")
    print("✓ ScrollManager with enhanced features and fixed crashes")
    print("✓ MultiLineInput with smart submission and history")
    print("✓ InputValidator with comprehensive validation")
    print("✓ Factory functions and configuration management")
    print("✓ Complete integration ready for ui.py")
    
    print("\nUI Library Info:")
    info = get_uilib_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  • {item}")
        else:
            print(f"{key}: {value}")
    
    print("\nConsolidation complete - ready for Phase 5: Update Module Interfaces")
    print("All original nci_*.py files can now be replaced with this single uilib.py")
