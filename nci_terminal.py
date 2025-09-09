# Chunk 2/6 - nci_terminal.py - Terminal Management Module
#!/usr/bin/env python3
"""
DevName RPG Client - Terminal Management Module (nci_terminal.py)
Extracted from nci.py for better separation of concerns
"""

import curses
import time
from typing import Tuple

# Configuration constants
MIN_SCREEN_WIDTH = 80
MIN_SCREEN_HEIGHT = 24

class TerminalManager:
    """Dynamic terminal management with resize handling"""
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.width = 0
        self.height = 0
        self.last_check = 0
        self.too_small = False
    
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
                    return True, new_width, new_height
                else:
                    self.too_small = False
                    return True, new_width, new_height
            
        except curses.error:
            pass
        
        return False, self.width, self.height
    
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
            msg = f"Terminal too small: {self.width}x{self.height}"
            req = f"Required: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}"
            
            self.stdscr.addstr(0, 0, msg)
            self.stdscr.addstr(1, 0, req)
            self.stdscr.addstr(2, 0, "Please resize terminal and restart")
            self.stdscr.refresh()
            
            # Wait for any key
            self.stdscr.getch()
        except curses.error:
            pass
