#!/usr/bin/env python3
"""
Aurora RPG Client - Ncurses Interface Module (nci_nc5.py) - FIXED VERSION

CRITICAL: This comment block must be preserved in all files to ensure proper
understanding of the modular architecture when analyzed by generative models.

MODULAR ARCHITECTURE OVERVIEW:
This project uses a modular architecture with the following interconnected files:

1. main_nc5.py: Main executable and application coordination
   - Handles command-line arguments, configuration, and application lifecycle
   - Imports and coordinates all other modules
   - Manages session state and graceful shutdown
   - Contains startup/shutdown logic and error handling

2. nci_nc5.py (THIS FILE): Ncurses Interface Module
   - Complete ncurses interface implementation with FIXED display pipeline
   - Input handling, screen management, color themes, context switching
   - Called by main_nc5.py for all user interface operations
   - Coordinates with other modules for display updates

3. mcp_nc5.py: MCP Communication Module  
   - HTTP client for Ollama/MCP server communication
   - Message formatting, retry logic, connection management
   - Called by nci_nc5.py when sending user messages
   - Provides enhanced context from sme_nc5.py

4. emm_nc5.py: Enhanced Memory Manager Module
   - Conversation history storage with semantic condensation
   - Token estimation and memory optimization
   - Called by nci_nc5.py for message storage/retrieval
   - Provides conversation context to mcp_nc5.py

5. sme_nc5.py: Story Momentum Engine Module
   - Dynamic narrative pressure and antagonist management
   - Analyzes conversation for story pacing
   - Called by nci_nc5.py to update pressure based on user input
   - Provides context enhancement for mcp_nc5.py requests

PROGRAMMATIC INTERCONNECTS:
- main_nc5.py → nci_nc5.py: Creates and runs CursesInterface
- nci_nc5.py → mcp_nc5.py: Sends messages via MCPClient
- nci_nc5.py → emm_nc5.py: Stores/retrieves messages via EnhancedMemoryManager
- nci_nc5.py → sme_nc5.py: Updates pressure via StoryMomentumEngine
- mcp_nc5.py ← sme_nc5.py: Receives story context for enhanced prompting
- mcp_nc5.py ← emm_nc5.py: Receives conversation history for context

PRESERVATION NOTICE:
When modifying any file in this project, you MUST preserve this comment block
to ensure that future analysis (human or AI) understands the full architecture
and interdependencies. Breaking these interconnects will cause system failures.

CRITICAL FIXES IN THIS VERSION:
- Removed multi-context complexity (debug/search modes)
- Fixed coordinate validation and text rendering
- Simplified display pipeline with guaranteed text visibility
- Enhanced error handling with detailed logging
- Focused solely on chat functionality to eliminate display issues

Main responsibilities of this file:
- Complete ncurses interface focused on chat display
- Input handling with blocking support during processing
- Color theme management
- Screen layout and window management with proper coordinates
- Command processing and help system
- Coordination with mcp_nc5, emm_nc5, and sme_nc5 modules
"""

import curses
import textwrap
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import our modular components
from mcp_nc5 import MCPClient
from emm_nc5 import EnhancedMemoryManager, MessageType
from sme_nc5 import StoryMomentumEngine

# Constants for interface
MAX_USER_INPUT_TOKENS = 2000

class CursesColorManager:
    """Simplified color management for ncurses interface"""
    
    # Color pair constants
    PAIR_USER_INPUT = 1
    PAIR_ASSISTANT_OUTPUT = 2
    PAIR_SYSTEM_INFO = 3
    PAIR_BORDER = 4
    PAIR_STATUS_BAR = 5
    PAIR_HIGHLIGHT = 6
    PAIR_ERROR = 7
    PAIR_THINKING = 9
    
    SCHEMES = {
        "midnight_aurora": "Midnight Aurora - Dark blue theme",
        "forest_whisper": "Forest Whisper - Green nature theme",
        "dracula_aurora": "Dracula Aurora - Purple gothic theme"
    }
    
    CYCLE_ORDER = ["midnight_aurora", "forest_whisper", "dracula_aurora"]
    
    def __init__(self, scheme_name: str = "midnight_aurora", debug_logger=None):
        self.debug_logger = debug_logger
        self.scheme_name = scheme_name
        self.initialized = False
        
        if self.scheme_name not in self.SCHEMES:
            self.scheme_name = "midnight_aurora"
        
        if self.debug_logger:
            self.debug_logger.debug(f"Color manager initialized with scheme: {scheme_name}", "INTERFACE")
    
    def initialize_colors(self, stdscr):
        """Initialize curses color pairs"""
        if not curses.has_colors():
            if self.debug_logger:
                self.debug_logger.error("Terminal does not support colors", "INTERFACE")
            return False
        
        try:
            curses.start_color()
            curses.use_default_colors()
            self._setup_color_pairs(self.scheme_name)
            self.initialized = True
            
            if self.debug_logger:
                self.debug_logger.debug("Color pairs initialized successfully", "INTERFACE")
            
            return True
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to initialize colors: {e}", "INTERFACE")
            return False
    
    def _setup_color_pairs(self, scheme_name: str):
        """Setup color pairs for specified scheme"""
        scheme_mappings = {
            "midnight_aurora": {
                self.PAIR_USER_INPUT: (curses.COLOR_CYAN, -1),
                self.PAIR_ASSISTANT_OUTPUT: (curses.COLOR_WHITE, -1),
                self.PAIR_SYSTEM_INFO: (curses.COLOR_YELLOW, -1),
                self.PAIR_BORDER: (curses.COLOR_BLUE, -1),
                self.PAIR_STATUS_BAR: (curses.COLOR_BLACK, curses.COLOR_CYAN),
                self.PAIR_HIGHLIGHT: (curses.COLOR_MAGENTA, -1),
                self.PAIR_ERROR: (curses.COLOR_RED, -1),
                self.PAIR_THINKING: (curses.COLOR_MAGENTA, -1)
            },
            "forest_whisper": {
                self.PAIR_USER_INPUT: (curses.COLOR_GREEN, -1),
                self.PAIR_ASSISTANT_OUTPUT: (curses.COLOR_WHITE, -1),
                self.PAIR_SYSTEM_INFO: (curses.COLOR_YELLOW, -1),
                self.PAIR_BORDER: (curses.COLOR_GREEN, -1),
                self.PAIR_STATUS_BAR: (curses.COLOR_BLACK, curses.COLOR_GREEN),
                self.PAIR_HIGHLIGHT: (curses.COLOR_MAGENTA, -1),
                self.PAIR_ERROR: (curses.COLOR_RED, -1),
                self.PAIR_THINKING: (curses.COLOR_YELLOW, -1)
            },
            "dracula_aurora": {
                self.PAIR_USER_INPUT: (curses.COLOR_MAGENTA, -1),
                self.PAIR_ASSISTANT_OUTPUT: (curses.COLOR_WHITE, -1),
                self.PAIR_SYSTEM_INFO: (curses.COLOR_CYAN, -1),
                self.PAIR_BORDER: (curses.COLOR_MAGENTA, -1),
                self.PAIR_STATUS_BAR: (curses.COLOR_BLACK, curses.COLOR_MAGENTA),
                self.PAIR_HIGHLIGHT: (curses.COLOR_YELLOW, -1),
                self.PAIR_ERROR: (curses.COLOR_RED, -1),
                self.PAIR_THINKING: (curses.COLOR_CYAN, -1)
            }
        }
        
        mappings = scheme_mappings.get(scheme_name, scheme_mappings["midnight_aurora"])
        
        for pair_id, (fg, bg) in mappings.items():
            try:
                curses.init_pair(pair_id, fg, bg)
            except curses.error:
                if self.debug_logger:
                    self.debug_logger.error(f"Failed to initialize color pair {pair_id}", "INTERFACE")
    
    def get_color_pair(self, element_type: str) -> int:
        """Get color pair for element type"""
        pair_map = {
            'user_input': self.PAIR_USER_INPUT,
            'assistant_output': self.PAIR_ASSISTANT_OUTPUT,
            'system_info': self.PAIR_SYSTEM_INFO,
            'border': self.PAIR_BORDER,
            'status_bar': self.PAIR_STATUS_BAR,
            'highlight': self.PAIR_HIGHLIGHT,
            'error': self.PAIR_ERROR,
            'thinking': self.PAIR_THINKING
        }
        return pair_map.get(element_type, 0)
    
    def cycle_scheme(self) -> str:
        """Cycle to next color scheme"""
        current_index = self.CYCLE_ORDER.index(self.scheme_name)
        next_index = (current_index + 1) % len(self.CYCLE_ORDER)
        old_scheme = self.scheme_name
        self.scheme_name = self.CYCLE_ORDER[next_index]
        
        if self.initialized:
            self._setup_color_pairs(self.scheme_name)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Color scheme changed from {old_scheme} to {self.scheme_name}", "INTERFACE")
        
        return self.get_scheme_display_name()
    
    def get_scheme_display_name(self) -> str:
        """Get display name for current scheme"""
        return self.SCHEMES[self.scheme_name].split(" - ")[0]

class InputValidator:
    """Validates and processes user input"""
    
    def __init__(self, max_tokens: int = MAX_USER_INPUT_TOKENS, debug_logger=None):
        self.max_tokens = max_tokens
        self.debug_logger = debug_logger
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters per token average)"""
        return len(text) // 4
    
    def validate_input_length(self, user_input: str) -> Tuple[bool, str, str]:
        """Validate user input length and provide helpful feedback"""
        input_tokens = self.estimate_tokens(user_input)
        
        if input_tokens <= self.max_tokens:
            if self.debug_logger:
                self.debug_logger.debug(f"Input validated: {input_tokens} tokens", "INPUT_VALIDATION")
            return True, "", ""
        
        char_count = len(user_input)
        max_chars = self.max_tokens * 4
        
        warning = (f"Input too long ({input_tokens:,} tokens, {char_count:,} chars). "
                  f"Maximum: {self.max_tokens:,} tokens ({max_chars:,} chars). "
                  f"Please shorten your input.")
        
        if self.debug_logger:
            self.debug_logger.debug(f"Input too long: {input_tokens} tokens > {self.max_tokens}", "INPUT_VALIDATION")
        
        return False, warning, user_input
    
    def clean_input(self, user_input: str) -> str:
        """Clean and normalize user input"""
        lines = user_input.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.rstrip()
            cleaned_lines.append(cleaned_line)
        
        # Remove excessive empty lines (more than 2 consecutive)
        result_lines = []
        empty_count = 0
        
        for line in cleaned_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        # Remove trailing empty lines
        while result_lines and result_lines[-1].strip() == "":
            result_lines.pop()
        
        cleaned = '\n'.join(result_lines)
        
        if self.debug_logger and cleaned != user_input:
            self.debug_logger.debug(f"Input cleaned: {len(user_input)} -> {len(cleaned)} chars", "INPUT_VALIDATION")
        
        return cleaned

# nci_nc5_fixed.py - Chunk 2/4
# CursesInterface Class - Core Implementation with FIXED Display Pipeline

class CursesInterface:
    """Complete ncurses interface with FIXED display pipeline - SIMPLIFIED CHAT ONLY"""
    
    def __init__(self, debug_logger=None, config=None):
        self.debug_logger = debug_logger
        self.config = config or {}
        
        # Window references
        self.stdscr = None
        self.output_win = None
        self.input_win = None
        self.status_win = None
        
        # Window dimensions
        self.height = 0
        self.width = 0
        self.output_height = 0
        self.input_height = 4
        self.status_height = 1
        
        # Input handling
        self.input_lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.input_mode = "normal"
        
        # Chat scrolling
        self.chat_scroll = 0
        
        # Status flags - SIMPLIFIED
        self.running = True
        self.needs_refresh = True
        self.in_quit_dialog = False
        self.waiting_for_response = False
        self.input_blocked = False
        
        # Initialize managers with config
        color_scheme = self.config.get('color_scheme', 'midnight_aurora')
        self.color_manager = CursesColorManager(color_scheme, debug_logger)
        self.memory_manager = EnhancedMemoryManager(debug_logger)
        self.input_validator = InputValidator(MAX_USER_INPUT_TOKENS, debug_logger)
        self.mcp_client = MCPClient(debug_logger)
        self.sme = StoryMomentumEngine(debug_logger)
        
        # Configure components from config
        if self.config:
            self._apply_configuration()
        
        if debug_logger:
            debug_logger.debug("CursesInterface created with modular components", "INTERFACE")
    
    def _apply_configuration(self):
        """Apply configuration to components"""
        # Configure MCP client
        mcp_url = self.config.get('mcp_server_url', 'http://127.0.0.1:3456/chat')
        mcp_model = self.config.get('mcp_model', 'qwen2.5:14b-instruct-q4_k_m')
        mcp_timeout = self.config.get('mcp_timeout', 300.0)
        self.mcp_client.set_server_config(mcp_url, mcp_model, mcp_timeout)
        
        # Configure memory manager
        condensation_threshold = self.config.get('memory_condensation_threshold', 8000)
        self.memory_manager.condensation_threshold = condensation_threshold
        
        # Configure SME
        if self.config.get('sme_enabled', True):
            self.sme.activate()
        
        if self.debug_logger:
            self.debug_logger.debug("Configuration applied to components", "INTERFACE")
    
    def run(self) -> int:
        """Run the ncurses interface using curses.wrapper"""
        def curses_main(stdscr):
            try:
                self.initialize(stdscr)
                self.main_loop()
                return 0
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.error(f"Curses interface error: {e}", "INTERFACE")
                raise
        
        try:
            return curses.wrapper(curses_main)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Curses wrapper error: {e}", "INTERFACE")
            print(f"Interface error: {e}")
            return 1
    
    def initialize(self, stdscr):
        """Initialize ncurses interface"""
        self.stdscr = stdscr
        curses.curs_set(1)
        
        # Initialize colors
        self.color_manager.initialize_colors(stdscr)
        
        # Get terminal dimensions
        self.height, self.width = stdscr.getmaxyx()
        self._calculate_window_dimensions()
        
        # Create windows
        self._create_windows()
        
        # Test MCP connection
        self._test_mcp_connection()
        
        # Initial display
        self.force_refresh()
        
        if self.debug_logger:
            self.debug_logger.debug(f"Interface initialized: {self.width}x{self.height}", "INTERFACE")
    
    def main_loop(self):
        """Main event loop with input blocking support"""
        while self.running:
            try:
                # Get key input with timeout
                self.stdscr.timeout(100)
                key = self.stdscr.getch()
                
                if key == -1:  # Timeout - continue loop
                    continue
                
                if key == curses.KEY_RESIZE:
                    self.handle_resize()
                    continue
                
                # Handle key input
                self.handle_key_input(key)
                
            except KeyboardInterrupt:
                self.running = False
            except curses.error:
                continue
    
    def _test_mcp_connection(self):
        """Test MCP connection and show status"""
        try:
            if self.mcp_client.test_connection():
                self.show_system_message("MCP connection established - Aurora is ready!")
            else:
                self.show_system_message("MCP server not available - using placeholder responses")
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"MCP connection test failed: {str(e)}", "INTERFACE")
            self.show_system_message("MCP connection failed - using placeholder responses")
    
    def _calculate_window_dimensions(self):
        """Calculate window sizes"""
        self.output_height = self.height - self.input_height - self.status_height - 3
        
        if self.output_height < 5:
            self.output_height = 5
            self.input_height = max(2, self.height - self.output_height - self.status_height - 3)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Window dimensions - Output: {self.output_height}, Input: {self.input_height}", "INTERFACE")
    
    def _create_windows(self):
        """Create all ncurses windows"""
        try:
            # Main output window
            self.output_win = curses.newwin(
                self.output_height, self.width - 2, 1, 1
            )
            
            # Input window
            input_y = self.output_height + 2
            self.input_win = curses.newwin(
                self.input_height, self.width - 2, input_y, 1
            )
            
            # Status bar
            status_y = self.height - 1
            self.status_win = curses.newwin(1, self.width, status_y, 0)
            
            # Enable scrolling
            self.output_win.scrollok(True)
            self.input_win.scrollok(True)
            
            if self.debug_logger:
                self.debug_logger.debug("All windows created successfully", "INTERFACE")
                
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to create windows: {e}", "INTERFACE")
            raise
    
    def _draw_borders(self):
        """Draw borders around windows"""
        try:
            border_color = self.color_manager.get_color_pair('border')
            self.stdscr.clear()
            
            # Draw borders with color
            if border_color > 0:
                self.stdscr.attron(curses.color_pair(border_color))
            
            # Top and bottom borders
            self.stdscr.hline(0, 0, curses.ACS_HLINE, self.width)
            input_y = self.output_height + 1
            self.stdscr.hline(input_y, 0, curses.ACS_HLINE, self.width)
            self.stdscr.hline(self.height - 2, 0, curses.ACS_HLINE, self.width)
            
            # Side borders
            for y in range(1, self.height - 1):
                self.stdscr.addch(y, 0, curses.ACS_VLINE)
                self.stdscr.addch(y, self.width - 1, curses.ACS_VLINE)
            
            # Corners
            self.stdscr.addch(0, 0, curses.ACS_ULCORNER)
            self.stdscr.addch(0, self.width - 1, curses.ACS_URCORNER)
            self.stdscr.addch(input_y, 0, curses.ACS_LTEE)
            self.stdscr.addch(input_y, self.width - 1, curses.ACS_RTEE)
            self.stdscr.addch(self.height - 2, 0, curses.ACS_LTEE)
            self.stdscr.addch(self.height - 2, self.width - 1, curses.ACS_RTEE)
            self.stdscr.addch(self.height - 1, 0, curses.ACS_LLCORNER)
            self.stdscr.addch(self.height - 1, self.width - 1, curses.ACS_LRCORNER)
            
            if border_color > 0:
                self.stdscr.attroff(curses.color_pair(border_color))
                
        except curses.error:
            pass
    
    def _update_status_bar(self):
        """Update status bar with current info"""
        try:
            self.status_win.clear()
            
            # Build status text
            status_parts = [
                f"Theme: [{self.color_manager.get_scheme_display_name().upper()}]"
            ]
            
            # Add connection status
            if self.mcp_client.connected:
                status_parts.append("MCP: Connected")
            else:
                status_parts.append("MCP: Offline")
            
            # Add processing indicator
            if self.waiting_for_response:
                status_parts.append("Status: Processing...")
            elif self.input_blocked:
                status_parts.append("Status: Input blocked")
            
            # Add SME pressure if available
            try:
                sme_status = self.sme.get_status()
                if sme_status.get('active', False):
                    pressure = sme_status.get('pressure_level', 0)
                    status_parts.append(f"Pressure: {pressure:.1f}")
            except:
                pass
            
            status_parts.append("Commands: /help")
            status_text = " | ".join(status_parts)
            
            # Truncate if too long
            if len(status_text) > self.width - 2:
                status_text = status_text[:self.width - 5] + "..."
            
            # Display with status bar color
            status_color = self.color_manager.get_color_pair('status_bar')
            if status_color > 0:
                self.status_win.attron(curses.color_pair(status_color))
            self.status_win.addstr(0, 0, status_text.ljust(self.width))
            if status_color > 0:
                self.status_win.attroff(curses.color_pair(status_color))
            
            self.status_win.refresh()
            
        except curses.error:
            pass
    
    def force_refresh(self):
        """Force complete screen refresh"""
        try:
            self._draw_borders()
            self._display_chat_content()
            self._update_input_window()
            self._update_status_bar()
            self.stdscr.refresh()
            
            if self.debug_logger:
                self.debug_logger.debug("Complete screen refresh performed", "INTERFACE")
                
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Force refresh failed: {e}", "INTERFACE")
    
    def set_input_blocked(self, blocked: bool):
        """Set input blocking status"""
        self.input_blocked = blocked
        self.waiting_for_response = blocked
        
        if self.debug_logger:
            self.debug_logger.debug(f"Input blocked: {blocked}", "INTERFACE")
        
        # Update interface immediately
        self.force_refresh()
    
    def handle_resize(self):
        """Handle terminal resize"""
        self.height, self.width = self.stdscr.getmaxyx()
        self._calculate_window_dimensions()
        self._create_windows()
        self.force_refresh()
        
        if self.debug_logger:
            self.debug_logger.debug(f"Resized to {self.width}x{self.height}", "INTERFACE")
    
    def get_config_updates(self) -> Dict[str, Any]:
        """Get configuration updates from current interface state"""
        return {
            "color_scheme": self.color_manager.scheme_name,
            "input_mode": self.input_mode,
            "mcp_server_url": self.mcp_client.server_url,
            "mcp_model": self.mcp_client.model
        }
    
    def export_conversation_state(self) -> Dict[str, Any]:
        """Export conversation state for saving"""
        messages = self.memory_manager.get_chat_history()
        sme_status = self.sme.get_status()
        
        return {
            "messages": [
                {
                    "content": msg.content,
                    "type": msg.message_type.value,
                    "timestamp": msg.timestamp
                } for msg in messages
            ],
            "sme_status": sme_status
        }

# nci_nc5_fixed.py - Chunk 3/4
# Display Management and Message Handling - CRITICAL FIXES FOR CHAT DISPLAY

    def _display_chat_content(self):
        """DIAGNOSTIC: Ultra-simple display to test text visibility"""
        try:
            # Clear the output window
            self.output_win.clear()

            messages = self.memory_manager.get_chat_history()

            if not messages:
                # Show welcome message at a safe position
                try:
                    self.output_win.addstr(0, 0, "Welcome to Aurora RPG Client")
                    self.output_win.addstr(1, 0, "Type your message below")
                except curses.error:
                    pass
                self.output_win.refresh()
                return

            # Show messages in the simplest possible way
            line_y = 0
            for i, message in enumerate(messages):
                if line_y >= self.output_height - 2:  # Stop before bottom
                    break

                # Create a simple display line
                prefix = "You" if message.message_type.value == "user" else "Aurora"
                display_text = f"{prefix}: {message.content[:60]}..."  # Truncate to 60 chars

                try:
                    # Write at safe coordinates - always start at x=0
                    self.output_win.addstr(line_y, 0, display_text[:self.width-4])
                    line_y += 1

                    # Add empty line between messages
                    if line_y < self.output_height - 2:
                        line_y += 1

                except curses.error as e:
                    # If this fails, log and continue
                    if self.debug_logger:
                        self.debug_logger.debug(f"Failed to write at ({line_y}, 0): {str(e)}", "INTERFACE")
                    break

            # Force refresh
            self.output_win.refresh()

            if self.debug_logger:
                self.debug_logger.debug(f"DIAGNOSTIC: Displayed {len(messages)} messages, wrote {line_y} lines", "INTERFACE")

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"DIAGNOSTIC display failed: {e}", "INTERFACE")
    
    def _update_input_window(self):
        """Update input window with current input and blocking status"""
        try:
            self.input_win.clear()
            
            # Only show input when not blocked
            if not self.input_blocked:
                input_color = self.color_manager.get_color_pair('user_input')
                
                try:
                    win_height, win_width = self.input_win.getmaxyx()
                    max_input_width = win_width - 4
                    
                    for i, line in enumerate(self.input_lines):
                        if i >= win_height - 1:
                            break
                        
                        display_line = line[:max_input_width] if len(line) > max_input_width else line
                        
                        try:
                            if input_color > 0:
                                self.input_win.attron(curses.color_pair(input_color))
                            self.input_win.addstr(i, 1, display_line)
                            if input_color > 0:
                                self.input_win.attroff(curses.color_pair(input_color))
                        except curses.error:
                            pass
                    
                    # Position cursor
                    if self.cursor_line < win_height - 1:
                        cursor_col = min(self.cursor_col + 1, win_width - 3)
                        try:
                            self.input_win.move(self.cursor_line, cursor_col)
                        except curses.error:
                            pass
                            
                except curses.error:
                    pass
                        
            else:
                # Show blocked message
                try:
                    thinking_color = self.color_manager.get_color_pair('thinking')
                    if thinking_color > 0:
                        self.input_win.attron(curses.color_pair(thinking_color))
                    self.input_win.addstr(1, 1, "Aurora is thinking... Please wait.")
                    if thinking_color > 0:
                        self.input_win.attroff(curses.color_pair(thinking_color))
                except curses.error:
                    pass
            
            self.input_win.refresh()
            
        except curses.error:
            pass
    
    def handle_key_input(self, key):
        """Handle keyboard input"""
        try:
            # Handle Escape key - simplified quit
            if key == 27:  # ESC
                self.running = False
                return
            
            # If input is blocked, only allow quit
            if self.input_blocked:
                if key == ord('q') or key == ord('Q'):
                    self.running = False
                return
            
            # Handle different key types
            if key == 10 or key == 13:  # Enter/Return
                self._submit_input()
            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                self._handle_backspace()
            elif key == curses.KEY_UP:
                # Scroll up in chat
                if self.chat_scroll < 10:
                    self.chat_scroll += 1
                    self.force_refresh()
            elif key == curses.KEY_DOWN:
                # Scroll down in chat
                if self.chat_scroll > 0:
                    self.chat_scroll -= 1
                    self.force_refresh()
            elif 32 <= key <= 126:  # Printable characters
                self._insert_character(chr(key))
            
            # Update input window after any change
            self._update_input_window()
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Error handling key input: {e}", "INTERFACE")
    
    def _submit_input(self):
        """Submit current input"""
        try:
            # Combine all input lines
            full_input = '\n'.join(self.input_lines).strip()
            
            if not full_input:
                return
            
            # Validate input length
            is_valid, warning, preserved = self.input_validator.validate_input_length(full_input)
            
            if not is_valid:
                self.show_error(warning)
                return
            
            # Show user input immediately
            self.show_user_input(full_input)
            
            # Clear input
            self._clear_input()
            
            # Block input during processing
            self.set_input_blocked(True)
            
            # Process the input
            self._process_user_input(full_input)
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Error submitting input: {e}", "INTERFACE")
            self.set_input_blocked(False)
    
    def _process_user_input(self, user_input: str):
        """Process user input with MCP integration"""
        try:
            # Check if it's a command
            if user_input.startswith('/'):
                should_continue, response = self._process_command(user_input)
                
                if not should_continue:
                    self.running = False
                    return
                
                # If command returned text, treat as regular input
                if response and response != user_input:
                    user_input = response
                elif not response:
                    # Command handled, unblock input
                    self.set_input_blocked(False)
                    return
            
            # Show thinking indicator
            self.show_thinking_message("Aurora is thinking...")
            
            # Get conversation history for context
            conversation_history = self.memory_manager.get_conversation_for_mcp()
            
            # Condense memory if needed
            conversation_history = self.memory_manager.condense_if_needed(conversation_history)
            
            # Send to MCP
            try:
                # Get SME context for enhanced prompting
                sme_context = self.sme.get_context_for_mcp()
                
                if sme_context:
                    response = self.mcp_client.send_message_with_sme_context(
                        user_input, conversation_history, sme_context
                    )
                else:
                    response = self.mcp_client.send_message(user_input, conversation_history)
                
                # Remove thinking indicator
                self.remove_last_thinking_message()
                
                # Show assistant response
                self.show_assistant_response(response)
                
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.error(f"MCP communication failed: {e}", "INTERFACE")
                
                # Remove thinking indicator
                self.remove_last_thinking_message()
                
                # Fallback to placeholder response
                response = self._generate_placeholder_response(user_input)
                self.show_assistant_response(response)
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Error processing user input: {e}", "INTERFACE")
            
            # Remove thinking indicator if present
            self.remove_last_thinking_message()
            
            self.show_error(f"Processing error: {str(e)}")
        
        finally:
            # Always unblock input when done
            self.set_input_blocked(False)
    
    def _generate_placeholder_response(self, user_input: str) -> str:
        """Generate placeholder response when MCP fails"""
        import random
        
        try:
            # Get SME status for context-aware responses
            sme_status = self.sme.get_status()
            pressure_level = sme_status.get('pressure_level', 0.0)
            
            # Base responses
            base_responses = [
                f"Aurora considers your words about '{user_input[:30]}...' thoughtfully. "
                f"The mystical energies around you respond, creating new possibilities.",
                
                f"The ancient wisdom flows through Aurora as she processes your mention of '{user_input[:30]}...'. "
                f"New paths reveal themselves in the ethereal mists.",
                
                f"Aurora's eyes glow with understanding as you speak of '{user_input[:30]}...'. "
                f"The fabric of reality shimmers with potential around your words."
            ]
            
            response = random.choice(base_responses)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Generated placeholder response", "INTERFACE")
            
            return response
            
        except Exception:
            return "Aurora nods thoughtfully, preparing her response to your words..."

# nci_nc5_fixed.py - Chunk 4/4
# Command Processing and Message Display Methods - Final Implementation

    def _process_command(self, command: str) -> Tuple[bool, str]:
        """Process user commands"""
        try:
            command = command.strip().lower()
            
            if self.debug_logger:
                self.debug_logger.debug(f"Processing command: {command}", "INTERFACE")
            
            # Handle different commands
            if command == "/quit" or command == "/exit":
                return False, "Goodbye!"
            
            elif command == "/help":
                help_text = self._get_help_text()
                self.show_system_message(help_text)
                return True, ""
            
            elif command == "/color" or command == "/theme":
                new_scheme = self.color_manager.cycle_scheme()
                response = f"Color scheme changed to: {new_scheme}"
                self.show_system_message(response)
                self.force_refresh()  # Immediate redraw with new colors
                return True, ""
            
            elif command == "/status":
                self._show_status_info()
                return True, ""
            
            elif command == "/connection" or command == "/conn":
                self._test_connection_status()
                return True, ""
            
            elif command == "/memory":
                self._show_memory_status()
                return True, ""
            
            elif command == "/sme":
                self._show_sme_status()
                return True, ""
            
            elif command == "/refresh":
                self.force_refresh()
                self.show_system_message("Display refreshed")
                return True, ""
            
            elif command.startswith("/clear"):
                return self._handle_clear_command(command)
            
            else:
                # Unknown command
                self.show_system_message(f"Unknown command: {command.split()[0]}. Type /help for available commands.")
                return True, ""
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Error processing command: {e}", "INTERFACE")
            self.show_error(f"Command processing error: {str(e)}")
            return True, ""
    
    def _get_help_text(self) -> str:
        """Generate help text"""
        return """Aurora RPG Client - Phase 5 Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASIC COMMANDS:
/help - Show this help message
/quit, /exit - Exit the application
/status - Show detailed status information
/connection, /conn - Test MCP server connection

APPEARANCE:
/color, /theme - Cycle through color themes
  • Midnight Aurora (blue)
  • Forest Whisper (green)
  • Dracula Aurora (purple)

SYSTEM:
/refresh - Refresh display
/clear confirm - Clear conversation history
/memory - Show memory management status
/sme - Show Story Momentum Engine status

NAVIGATION:
Arrow Keys - Scroll chat history up/down
Esc - Exit application
Enter - Send message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 5: Modular architecture with enhanced MCP integration"""
    
    # Message display methods
    def show_user_input(self, text: str):
        """Add user input and force immediate display"""
        try:
            self.memory_manager.add_message(text, MessageType.USER)
            self.sme.process_user_input(text)
            
            if self.debug_logger:
                self.debug_logger.debug(f"User input added: {len(text)} chars", "INTERFACE")
            
            self.force_refresh()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show user input: {e}", "INTERFACE")
    
    def show_assistant_response(self, text: str):
        """Add assistant response and force immediate display"""
        try:
            self.memory_manager.add_message(text, MessageType.ASSISTANT)
            self.sme.process_assistant_response(text)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Assistant response added: {len(text)} chars", "INTERFACE")
            
            self.force_refresh()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show assistant response: {e}", "INTERFACE")
    
    def show_system_message(self, text: str):
        """Add system message and force immediate display"""
        try:
            self.memory_manager.add_message(text, MessageType.SYSTEM)
            
            if self.debug_logger:
                self.debug_logger.debug(f"System message: {text}", "INTERFACE")
            
            self.force_refresh()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show system message: {e}", "INTERFACE")
    
    def show_thinking_message(self, text: str = "Aurora is thinking..."):
        """Add thinking indicator and force immediate display"""
        try:
            self.memory_manager.add_message(text, MessageType.THINKING)
            self.force_refresh()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show thinking message: {e}", "INTERFACE")
    
    def remove_last_thinking_message(self):
        """Remove the last thinking message if present"""
        try:
            messages = self.memory_manager.get_chat_history()
            if messages and messages[-1].message_type == MessageType.THINKING:
                self.memory_manager.remove_last_message()
                self.force_refresh()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to remove thinking message: {e}", "INTERFACE")
    
    def show_error(self, text: str):
        """Add error message and force immediate display"""
        try:
            self.memory_manager.add_message(f"[Error] {text}", MessageType.SYSTEM)
            
            if self.debug_logger:
                self.debug_logger.error(text, "INTERFACE")
            
            self.force_refresh()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show error message: {e}", "INTERFACE")
    
    # Status information methods
    def _show_status_info(self):
        """Show comprehensive status information"""
        try:
            messages = self.memory_manager.get_chat_history()
            message_count = len(messages)
            
            # MCP status
            mcp_status = "Connected" if self.mcp_client.connected else "Offline"
            
            # Debug status
            debug_status = "Enabled" if self.debug_logger else "Disabled"
            
            # Current theme
            theme_name = self.color_manager.get_scheme_display_name()
            
            # Input status
            input_status = "Blocked" if self.input_blocked else "Active"
            
            # Memory status
            memory_info = self.memory_manager.get_memory_stats()
            
            # SME status
            sme_status = self.sme.get_status()
            
            status_info = f"""Aurora RPG Client Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Messages in conversation: {message_count}
Memory tokens used: {memory_info.total_tokens}
MCP Server: {mcp_status}
Debug Logging: {debug_status}
Current Theme: {theme_name}
Input Status: {input_status}
Terminal Size: {self.width}x{self.height}
SME Active: {sme_status.get('active', False)}
SME Pressure: {sme_status.get('pressure_level', 0.0):.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use /help for available commands."""
            
            self.show_system_message(status_info)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show status: {e}", "INTERFACE")
    
    def _test_connection_status(self):
        """Test and report MCP connection status"""
        try:
            self.show_system_message("Testing MCP connection...")
            
            if self.mcp_client.test_connection():
                self.show_system_message("MCP connection successful! Aurora is ready to respond.")
                self.mcp_client.connected = True
            else:
                self.show_system_message("MCP connection failed. Using placeholder responses.")
                self.mcp_client.connected = False
        except Exception as e:
            error_msg = f"MCP connection error: {str(e)}"
            self.show_error(error_msg)
            self.mcp_client.connected = False
    
    def _show_memory_status(self):
        """Show memory management status"""
        try:
            memory_info = self.memory_manager.get_memory_stats()
            
            status_text = f"""Enhanced Memory Manager Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total messages: {memory_info.message_count}
Total tokens: {memory_info.total_tokens}
Condensations performed: {memory_info.condensation_count}
Last condensation: {memory_info.last_condensation}
Memory efficiency: {memory_info.efficiency_ratio:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The memory manager automatically condenses old conversations
to maintain optimal performance while preserving context."""
            
            self.show_system_message(status_text)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show memory status: {e}", "INTERFACE")
    
    def _show_sme_status(self):
        """Show Story Momentum Engine status"""
        try:
            sme_status = self.sme.get_status()
            
            status_text = f"""Story Momentum Engine Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Engine Active: {sme_status.get('active', False)}
Pressure Level: {sme_status.get('pressure_level', 0.0):.2f}
Pressure Name: {sme_status.get('pressure_name', 'Calm')}
Current Antagonist: {sme_status.get('antagonist_name', 'None')}
Story Arc: {sme_status.get('story_arc', 'Beginning')}
Last Update: {sme_status.get('last_update', 'Never')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The SME dynamically adjusts narrative tension and
introduces challenges based on conversation flow."""
            
            self.show_system_message(status_text)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to show SME status: {e}", "INTERFACE")
    
    def _handle_clear_command(self, command: str) -> Tuple[bool, str]:
        """Handle clear command with confirmation"""
        try:
            if command.strip() == "/clear":
                messages = self.memory_manager.get_chat_history()
                if messages:
                    self.show_system_message(f"Clear {len(messages)} messages? Type '/clear confirm' to proceed.")
                else:
                    self.show_system_message("Chat history is already empty.")
            
            elif command.strip() == "/clear confirm":
                self.memory_manager.clear_history()
                self.sme.reset()  # Reset SME when clearing chat
                self.show_system_message("Chat history and SME state cleared")
                self.force_refresh()
            
            return True, ""
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to handle clear command: {e}", "INTERFACE")
            return True, ""
    
    # Input handling helper methods (simplified implementations)
    def _clear_input(self):
        """Clear all input and reset to normal mode"""
        self.input_lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.input_mode = "normal"
    
    def _insert_character(self, char: str):
        """Insert character at cursor position"""
        try:
            current_line = self.input_lines[self.cursor_line]
            new_line = current_line[:self.cursor_col] + char + current_line[self.cursor_col:]
            self.input_lines[self.cursor_line] = new_line
            self.cursor_col += 1
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to insert character: {e}", "INTERFACE")
    
    def _handle_backspace(self):
        """Handle backspace key"""
        try:
            if self.cursor_col > 0:
                line = self.input_lines[self.cursor_line]
                self.input_lines[self.cursor_line] = line[:self.cursor_col-1] + line[self.cursor_col:]
                self.cursor_col -= 1
            elif self.cursor_line > 0:
                current_line = self.input_lines[self.cursor_line]
                previous_line = self.input_lines[self.cursor_line - 1]
                self.cursor_col = len(previous_line)
                self.input_lines[self.cursor_line - 1] = previous_line + current_line
                self.input_lines.pop(self.cursor_line)
                self.cursor_line -= 1
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to handle backspace: {e}", "INTERFACE")

# End of nci_nc5_fixed.py - Aurora RPG Client Ncurses Interface Module - FIXED VERSION
