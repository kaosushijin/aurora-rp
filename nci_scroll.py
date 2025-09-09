# Chunk 4/6 - nci_scroll.py - Scrolling System Module
#!/usr/bin/env python3
"""
DevName RPG Client - Scrolling System Module (nci_scroll.py)
Module architecture and interconnects documented in genai.txt
Extracted from nci.py for better separation of concerns
"""

from typing import Dict, Any

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
        """Get scroll information for status display"""
        if self.max_scroll == 0:
            return {"in_scrollback": False, "percentage": 100}
        
        percentage = int((self.scroll_offset / self.max_scroll) * 100) if self.max_scroll > 0 else 100
        return {
            "in_scrollback": self.in_scrollback,
            "percentage": percentage,
            "offset": self.scroll_offset,
            "max": self.max_scroll
        }
    
    def get_visible_range(self) -> tuple[int, int]:
        """Get the range of lines that should be visible"""
        start_idx = self.scroll_offset
        end_idx = start_idx + self.window_height - 1
        return start_idx, end_idx
    
    def jump_to_position(self, offset: int):
        """Jump to specific scroll position"""
        self.scroll_offset = max(0, min(offset, self.max_scroll))
        self.in_scrollback = (self.scroll_offset < self.max_scroll)
