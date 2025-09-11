# nci_terminal.py - Terminal Management Module with Dynamic Box Coordinate System
#!/usr/bin/env python3
"""
DevName RPG Client - Terminal Management Module (nci_terminal.py)
Module architecture and interconnects documented in genai.txt
Implements dynamic box coordinate system for robust window management
"""

import curses
import time
from typing import Tuple
from dataclasses import dataclass

# Configuration constants
MIN_SCREEN_WIDTH = 80
MIN_SCREEN_HEIGHT = 24

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
    Calculate dynamic box layout:
    
    1. Reserve 1 line for status at bottom
    2. Reserve 2 lines for borders (between output/input, above status)
    3. Split remaining lines 90/10 between output/input
    4. Calculate inner coordinates for each box
    """
    
    # Reserve space
    status_height = 1
    border_lines = 2
    available_height = height - status_height - border_lines
    
    # Split available space
    output_height = int(available_height * 0.9)
    input_height = available_height - output_height
    
    # Any remainder goes to input
    if available_height != output_height + input_height:
        input_height += available_height - output_height - input_height
    
    # Calculate output box coordinates
    output_box = BoxCoordinates(
        top=0,
        left=0,
        bottom=output_height,
        right=width-1,
        inner_top=0,
        inner_left=0,
        inner_bottom=output_height-1,
        inner_right=width-1,
        width=width,
        height=output_height+1,
        inner_width=width,
        inner_height=output_height
    )
    
    # Calculate input box coordinates
    input_top = output_height + 1
    input_box = BoxCoordinates(
        top=input_top,
        left=0,
        bottom=input_top + input_height,
        right=width-1,
        inner_top=input_top,
        inner_left=0,
        inner_bottom=input_top + input_height - 1,
        inner_right=width-1,
        width=width,
        height=input_height+1,
        inner_width=width,
        inner_height=input_height
    )
    
    # Calculate status line coordinates
    status_top = height - 1
    status_line = BoxCoordinates(
        top=status_top,
        left=0,
        bottom=status_top,
        right=width-1,
        inner_top=status_top,
        inner_left=0,
        inner_bottom=status_top,
        inner_right=width-1,
        width=width,
        height=1,
        inner_width=width,
        inner_height=1
    )
    
    return LayoutGeometry(
        terminal_height=height,
        terminal_width=width,
        output_box=output_box,
        input_box=input_box,
        status_line=status_line
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
    
    def draw_box_borders(self, layout: LayoutGeometry, border_color: int = 0):
        """Draw borders for all boxes in the layout"""
        try:
            if border_color:
                self.stdscr.attron(curses.color_pair(border_color))
            
            # Draw horizontal border between output and input
            border_y = layout.output_box.bottom
            self.stdscr.hline(border_y, 0, curses.ACS_HLINE, layout.terminal_width)
            
            # Draw horizontal border above status
            status_border_y = layout.status_line.top - 1
            if status_border_y >= 0 and status_border_y != border_y:
                self.stdscr.hline(status_border_y, 0, curses.ACS_HLINE, layout.terminal_width)
            
            if border_color:
                self.stdscr.attroff(curses.color_pair(border_color))
            
            self.stdscr.refresh()
            
        except curses.error:
            pass
