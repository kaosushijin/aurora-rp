# Chunk 1/6 - nci_colors.py - Color Management Module
#!/usr/bin/env python3
"""
DevName RPG Client - Color Management Module (nci_colors.py)
Module architecture and interconnects documented in genai.txt
Extracted from nci.py for better separation of concerns
"""

import curses
from enum import Enum
from typing import Optional

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
        
        # Color pair definitions
        self.USER_COLOR = 1
        self.ASSISTANT_COLOR = 2
        self.SYSTEM_COLOR = 3
        self.ERROR_COLOR = 4
        self.BORDER_COLOR = 5
    
    def init_colors(self) -> bool:
        """Initialize color pairs, return success status"""
        if not curses.has_colors():
            self.colors_available = False
            return False
        
        try:
            curses.start_color()
            curses.use_default_colors()
            
            if self.theme == ColorTheme.CLASSIC:
                curses.init_pair(self.USER_COLOR, curses.COLOR_CYAN, -1)
                curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_GREEN, -1)
                curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_YELLOW, -1)
                curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
                curses.init_pair(self.BORDER_COLOR, curses.COLOR_BLUE, -1)
            elif self.theme == ColorTheme.DARK:
                curses.init_pair(self.USER_COLOR, curses.COLOR_WHITE, -1)
                curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_CYAN, -1)
                curses.init_pair(self.SYSTEM_COLOR, curses.COLOR_MAGENTA, -1)
                curses.init_pair(self.ERROR_COLOR, curses.COLOR_RED, -1)
                curses.init_pair(self.BORDER_COLOR, curses.COLOR_WHITE, -1)
            else:  # BRIGHT
                curses.init_pair(self.USER_COLOR, curses.COLOR_BLUE, -1)
                curses.init_pair(self.ASSISTANT_COLOR, curses.COLOR_GREEN, -1)
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
    
    def get_available_themes(self) -> list[str]:
        """Get list of available theme names"""
        return [theme.value for theme in ColorTheme]
