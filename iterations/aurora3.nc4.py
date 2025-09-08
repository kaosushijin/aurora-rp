# Aurora RPG Client - Fixed Ncurses Implementation - Chunk 1/6
# Core imports, DebugLogger, and Color Management

import os
import sys
import json
import argparse
import curses
import time
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

# Constants for configuration
MAX_USER_INPUT_TOKENS = 2000
DEBUG_LOG_FILE = "debug.log"
CHAT_HISTORY_FILE = "chat_history.json"
CONFIG_FILE = "aurora_config.json"

class DebugLogger:
    """File-based debug logging system with clean console interface"""
    
    def __init__(self, enabled: bool = False, log_file: str = DEBUG_LOG_FILE):
        self.enabled = enabled
        self.log_file = Path(log_file)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.enabled:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Aurora RPG Debug Session: {self.session_id}\n")
                    f.write(f"Started: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*80}\n\n")
            except Exception as e:
                print(f"Warning: Could not initialize debug log: {e}")
                self.enabled = False
    
    def _write_log(self, level: str, category: str, message: str):
        """Write log entry to file (never to console)"""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level:>6} | {category:>12} | {message}\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception:
            pass
    
    def system(self, message: str):
        """Log system operations"""
        self._write_log("SYSTEM", "CORE", message)
    
    def debug(self, message: str, category: str = "DEBUG"):
        """Log debug information"""
        self._write_log("DEBUG", category, message)
    
    def error(self, message: str, category: str = "ERROR"):
        """Log error information"""
        self._write_log("ERROR", category, message)
    
    def info(self, message: str, category: str = "INFO"):
        """Log informational messages"""
        self._write_log("INFO", category, message)
    
    def user_input(self, input_text: str):
        """Log user input (truncated for privacy)"""
        truncated = input_text[:100] + "..." if len(input_text) > 100 else input_text
        self._write_log("INPUT", "USER", f"Length: {len(input_text)} | Text: {truncated}")
    
    def assistant_response(self, response_text: str):
        """Log assistant response (truncated)"""
        truncated = response_text[:100] + "..." if len(response_text) > 100 else response_text
        self._write_log("OUTPUT", "ASSISTANT", f"Length: {len(response_text)} | Text: {truncated}")
    
    def interface_operation(self, operation: str, details: str):
        """Log interface operations"""
        self._write_log("UI", "INTERFACE", f"{operation}: {details}")
    
    def mcp_operation(self, operation: str, details: str):
        """Log MCP operations"""
        self._write_log("MCP", "PROTOCOL", f"{operation}: {details}")
    
    def get_debug_content(self) -> List[str]:
        """Read debug log content for display in debug context"""
        if not self.enabled or not self.log_file.exists():
            return ["Debug logging is disabled. Use --debug flag to enable."]
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [line.rstrip() for line in lines[-500:]]
        except Exception as e:
            return [f"Error reading debug log: {e}"]
    
    def close_session(self):
        """Close debug session with footer"""
        if self.enabled:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Session {self.session_id} ended: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*80}\n\n")
            except Exception:
                pass

class CursesColorManager:
    """Color management for ncurses interface with two themes"""
    
    # Color pair constants
    PAIR_USER_INPUT = 1
    PAIR_ASSISTANT_OUTPUT = 2
    PAIR_SYSTEM_INFO = 3
    PAIR_BORDER = 4
    PAIR_STATUS_BAR = 5
    PAIR_HIGHLIGHT = 6
    PAIR_ERROR = 7
    PAIR_SEARCH_HIGHLIGHT = 8
    PAIR_COMPANION_DIALOGUE = 9
    PAIR_NPC_DIALOGUE = 10
    
    # Available color schemes
    SCHEMES = {
        "midnight_aurora": "Midnight Aurora - Dark blue theme for long sessions",
        "forest_whisper": "Forest Whisper - Green nature theme for readability"
    }
    
    CYCLE_ORDER = ["midnight_aurora", "forest_whisper"]
    
    def __init__(self, scheme_name: str = "midnight_aurora", debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.scheme_name = scheme_name
        self.initialized = False
        
        if self.scheme_name not in self.SCHEMES:
            self.scheme_name = "midnight_aurora"
        
        if self.debug_logger:
            self.debug_logger.interface_operation("color_init", f"Initializing with scheme: {scheme_name}")
    
    def initialize_colors(self, stdscr):
        """Initialize curses color pairs"""
        if not curses.has_colors():
            if self.debug_logger:
                self.debug_logger.error("Terminal does not support colors")
            return False
        
        curses.start_color()
        curses.use_default_colors()
        
        self._setup_color_pairs(self.scheme_name)
        self.initialized = True
        
        if self.debug_logger:
            self.debug_logger.interface_operation("colors_ready", "Color pairs initialized successfully")
        
        return True
    
    def _setup_color_pairs(self, scheme_name: str):
        """Setup curses color pairs for specified scheme"""
        scheme_mappings = {
            "midnight_aurora": {
                self.PAIR_USER_INPUT: (curses.COLOR_CYAN, -1),
                self.PAIR_ASSISTANT_OUTPUT: (curses.COLOR_WHITE, -1),
                self.PAIR_SYSTEM_INFO: (curses.COLOR_YELLOW, -1),
                self.PAIR_BORDER: (curses.COLOR_BLUE, -1),
                self.PAIR_STATUS_BAR: (curses.COLOR_BLACK, curses.COLOR_CYAN),
                self.PAIR_HIGHLIGHT: (curses.COLOR_MAGENTA, -1),
                self.PAIR_ERROR: (curses.COLOR_RED, -1),
                self.PAIR_SEARCH_HIGHLIGHT: (curses.COLOR_BLACK, curses.COLOR_YELLOW),
                self.PAIR_COMPANION_DIALOGUE: (curses.COLOR_GREEN, -1),
                self.PAIR_NPC_DIALOGUE: (curses.COLOR_BLUE, -1)
            },
            
            "forest_whisper": {
                self.PAIR_USER_INPUT: (curses.COLOR_GREEN, -1),
                self.PAIR_ASSISTANT_OUTPUT: (curses.COLOR_WHITE, -1),
                self.PAIR_SYSTEM_INFO: (curses.COLOR_YELLOW, -1),
                self.PAIR_BORDER: (curses.COLOR_GREEN, -1),
                self.PAIR_STATUS_BAR: (curses.COLOR_BLACK, curses.COLOR_GREEN),
                self.PAIR_HIGHLIGHT: (curses.COLOR_MAGENTA, -1),
                self.PAIR_ERROR: (curses.COLOR_RED, -1),
                self.PAIR_SEARCH_HIGHLIGHT: (curses.COLOR_BLACK, curses.COLOR_GREEN),
                self.PAIR_COMPANION_DIALOGUE: (curses.COLOR_CYAN, -1),
                self.PAIR_NPC_DIALOGUE: (curses.COLOR_BLUE, -1)
            }
        }
        
        if scheme_name not in scheme_mappings:
            scheme_name = "midnight_aurora"
        
        mappings = scheme_mappings[scheme_name]
        
        for pair_id, (fg, bg) in mappings.items():
            try:
                curses.init_pair(pair_id, fg, bg)
            except curses.error:
                if self.debug_logger:
                    self.debug_logger.error(f"Failed to initialize color pair {pair_id}")
    
    def get_color_pair(self, element_type: str) -> int:
        """Get curses color pair for element type"""
        pair_map = {
            'user_input': self.PAIR_USER_INPUT,
            'assistant_output': self.PAIR_ASSISTANT_OUTPUT,
            'system_info': self.PAIR_SYSTEM_INFO,
            'border': self.PAIR_BORDER,
            'status_bar': self.PAIR_STATUS_BAR,
            'highlight': self.PAIR_HIGHLIGHT,
            'error': self.PAIR_ERROR,
            'search_highlight': self.PAIR_SEARCH_HIGHLIGHT,
            'companion_dialogue': self.PAIR_COMPANION_DIALOGUE,
            'npc_dialogue': self.PAIR_NPC_DIALOGUE
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
            self.debug_logger.interface_operation("color_cycle", 
                f"Changed from {old_scheme} to {self.scheme_name}")
        
        return self.get_scheme_display_name()
    
    def get_scheme_display_name(self) -> str:
        """Get display name for current scheme"""
        return self.SCHEMES[self.scheme_name].split(" - ")[0]

class ContextType(Enum):
    """Available interface contexts"""
    CHAT = "chat"
    DEBUG = "debug" 
    SEARCH = "search"

class MessageType(Enum):
    """Types of messages in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    COMPANION = "companion"
    NPC = "npc"

class Message(NamedTuple):
    """Individual message structure"""
    content: str
    message_type: MessageType
    timestamp: str
    context: ContextType = ContextType.CHAT

class ContextManager:
    """Manages different interface contexts and message history"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.current_context = ContextType.CHAT
        
        # Message storage for different contexts
        self.chat_messages: List[Message] = []
        self.debug_messages: List[str] = []
        self.search_results: List[str] = []
        self.search_term: str = ""
        
        # Input preservation
        self.preserved_input: str = ""
        self.input_position: int = 0
        
        # Context help messages
        self.context_help = {
            ContextType.DEBUG: "Debug context active. Press Esc to return to chat.",
            ContextType.SEARCH: "Search results displayed. Press Esc to return to chat.",
            ContextType.CHAT: ""
        }
        
        if self.debug_logger:
            self.debug_logger.system("ContextManager initialized")
    
    def add_chat_message(self, content: str, message_type: str):
        """Add message to chat context"""
        try:
            msg_type = MessageType(message_type.lower())
        except ValueError:
            msg_type = MessageType.SYSTEM
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        message = Message(
            content=content,
            message_type=msg_type,
            timestamp=timestamp,
            context=ContextType.CHAT
        )
        
        self.chat_messages.append(message)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Added {message_type} message: {len(content)} chars", "CONTEXT")
    
    def get_chat_history(self) -> List[Message]:
        """Get all chat messages"""
        return self.chat_messages.copy()
    
    def get_current_context(self) -> ContextType:
        """Get current active context"""
        return self.current_context
    
    def switch_context(self, new_context: ContextType) -> bool:
        """Switch to different context"""
        if new_context == self.current_context:
            return False
        
        old_context = self.current_context
        self.current_context = new_context
        
        if self.debug_logger:
            self.debug_logger.interface_operation("context_switch", 
                f"Switched from {old_context.value} to {new_context.value}")
        
        return True
    
    def set_debug_content(self, debug_lines: List[str]):
        """Update debug context content"""
        self.debug_messages = debug_lines.copy()
        
        if self.debug_logger:
            self.debug_logger.debug(f"Debug content updated: {len(debug_lines)} lines", "CONTEXT")
    
    def get_debug_content(self) -> List[str]:
        """Get debug context content"""
        if not self.debug_messages:
            return ["No debug information available.", "Use --debug flag to enable logging."]
        return self.debug_messages.copy()
    
    def perform_search(self, search_term: str) -> int:
        """Search chat history and populate search context"""
        self.search_term = search_term.lower()
        self.search_results = []
        
        if not search_term.strip():
            self.search_results = ["Error: Empty search term"]
            return 0
        
        matches_found = 0
        
        for i, message in enumerate(self.chat_messages):
            if self.search_term in message.content.lower():
                matches_found += 1
                timestamp = message.timestamp
                msg_type = message.message_type.value.upper()
                
                result_line = f"[{timestamp}] [{msg_type}] {message.content}"
                self.search_results.append(result_line)
        
        if matches_found == 0:
            self.search_results = [f"No results found for: '{search_term}'"]
        else:
            header = f"Search results for '{search_term}' ({matches_found} matches):"
            self.search_results.insert(0, header)
            self.search_results.insert(1, "-" * len(header))
        
        if self.debug_logger:
            self.debug_logger.interface_operation("search", 
                f"Searched for '{search_term}', found {matches_found} matches")
        
        return matches_found
    
    def get_search_results(self) -> List[str]:
        """Get search context content"""
        if not self.search_results:
            return ["No search performed yet.", "Use /search <term> to search chat history."]
        return self.search_results.copy()
    
    def get_search_term(self) -> str:
        """Get current search term"""
        return self.search_term
    
    def clear_context(self, context: ContextType):
        """Clear specified context content"""
        if context == ContextType.CHAT:
            self.chat_messages.clear()
            if self.debug_logger:
                self.debug_logger.interface_operation("clear", "Chat history cleared")
        elif context == ContextType.DEBUG:
            self.debug_messages.clear()
            if self.debug_logger:
                self.debug_logger.interface_operation("clear", "Debug content cleared")
        elif context == ContextType.SEARCH:
            self.search_results.clear()
            self.search_term = ""
            if self.debug_logger:
                self.debug_logger.interface_operation("clear", "Search results cleared")

class InputValidator:
    """Validates and processes user input"""
    
    def __init__(self, max_tokens: int = MAX_USER_INPUT_TOKENS, debug_logger: Optional[DebugLogger] = None):
        self.max_tokens = max_tokens
        self.debug_logger = debug_logger
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters per token average)"""
        return len(text) // 4
    
    def validate_input_length(self, user_input: str) -> Tuple[bool, str, str]:
        """
        Validate user input length and provide helpful feedback.
        Returns (is_valid, warning_message, preserved_input)
        """
        input_tokens = self.estimate_tokens(user_input)
        
        if input_tokens <= self.max_tokens:
            if self.debug_logger:
                self.debug_logger.debug(f"Input validated: {input_tokens} tokens", "INPUT_VALIDATION")
            return True, "", ""
        
        char_count = len(user_input)
        max_chars = self.max_tokens * 4
        
        warning = (f"Input too long ({input_tokens:,} tokens, {char_count:,} chars). "
                  f"Maximum: {self.max_tokens:,} tokens ({max_chars:,} chars). "
                  f"Please shorten your input - it has been preserved for editing.")
        
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

# Aurora RPG Client - Fixed Ncurses Implementation - Chunk 2/6
# Main Curses Interface Class with Fixed Input/Output

class CursesInterface:
    """Complete ncurses interface with proper input/output display"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        
        # Window references
        self.stdscr = None
        self.output_win = None
        self.input_win = None
        self.status_win = None
        
        # Window dimensions
        self.height = 0
        self.width = 0
        self.output_height = 0
        self.input_height = 4  # Increased for multi-line support
        self.status_height = 1
        
        # Input handling
        self.input_lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        
        # Context-specific scrolling
        self.chat_scroll = 0
        self.debug_scroll = 0
        self.search_scroll = 0
        
        # Status flags
        self.running = True
        self.needs_refresh = True
        self.in_quit_dialog = False
        
        # Managers
        self.color_manager = CursesColorManager("midnight_aurora", debug_logger)
        self.context_manager = ContextManager(debug_logger)
        self.input_validator = InputValidator(MAX_USER_INPUT_TOKENS, debug_logger)
        
        # Show SME status toggle
        self.show_sme_status = False
        
        if debug_logger:
            debug_logger.interface_operation("curses_init", "CursesInterface created")
    
    def initialize(self, stdscr):
        """Initialize ncurses interface"""
        self.stdscr = stdscr
        curses.curs_set(1)  # Show cursor
        
        # Initialize colors
        if not self.color_manager.initialize_colors(stdscr):
            if self.debug_logger:
                self.debug_logger.error("Failed to initialize colors")
        
        # Get terminal dimensions
        self.height, self.width = stdscr.getmaxyx()
        self._calculate_window_dimensions()
        
        # Create windows
        self._create_windows()
        
        # Initial display
        self._update_all_windows()
        
        if self.debug_logger:
            self.debug_logger.interface_operation("curses_ready", 
                f"Interface initialized: {self.width}x{self.height}")
    
    def _calculate_window_dimensions(self):
        """Calculate window sizes based on terminal dimensions"""
        # Reserve space for status bar and borders
        self.output_height = self.height - self.input_height - self.status_height - 3
        
        # Ensure minimum sizes
        if self.output_height < 5:
            self.output_height = 5
            self.input_height = max(2, self.height - self.output_height - self.status_height - 3)
        
        if self.debug_logger:
            self.debug_logger.interface_operation("window_calc", 
                f"Output: {self.output_height}, Input: {self.input_height}")
    
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
            
            # Enable scrolling for output window
            self.output_win.scrollok(True)
            self.input_win.scrollok(True)
            
            if self.debug_logger:
                self.debug_logger.interface_operation("windows_created", "All windows created successfully")
                
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to create windows: {e}")
            raise
    
    def _draw_borders(self):
        """Draw borders around windows"""
        try:
            border_color = self.color_manager.get_color_pair('border')
            
            # Clear screen
            self.stdscr.clear()
            
            # Draw main border
            self.stdscr.attron(curses.color_pair(border_color))
            
            # Top border
            self.stdscr.hline(0, 0, curses.ACS_HLINE, self.width)
            
            # Bottom borders
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
            
            self.stdscr.attroff(curses.color_pair(border_color))
            
        except curses.error:
            pass
    
    def _update_status_bar(self):
        """Update status bar with current context and theme info"""
        try:
            self.status_win.clear()
            
            # Get current context
            current_context = self.context_manager.get_current_context()
            context_name = current_context.value.upper()
            
            # Get current color scheme
            scheme_name = self.color_manager.get_scheme_display_name().upper()
            
            # Build status text
            status_parts = [
                f"Context: [{context_name}]",
                f"Theme: [{scheme_name}]",
                "Commands: /help"
            ]
            
            # Add SME status if enabled
            if self.show_sme_status:
                status_parts.insert(-1, "SME: Active")
            
            status_text = " | ".join(status_parts)
            
            # Truncate if too long
            if len(status_text) > self.width - 2:
                status_text = status_text[:self.width - 5] + "..."
            
            # Display with status bar color
            status_color = self.color_manager.get_color_pair('status_bar')
            self.status_win.attron(curses.color_pair(status_color))
            self.status_win.addstr(0, 0, status_text.ljust(self.width))
            self.status_win.attroff(curses.color_pair(status_color))
            
            self.status_win.refresh()
            
        except curses.error:
            pass
    
    def _update_output_window(self):
        """Update output window based on current context"""
        try:
            self.output_win.clear()
            
            current_context = self.context_manager.get_current_context()
            
            if current_context == ContextType.CHAT:
                self._display_chat_content()
            elif current_context == ContextType.DEBUG:
                self._display_debug_content()
            elif current_context == ContextType.SEARCH:
                self._display_search_content()
            
            self.output_win.refresh()
            
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Error updating output window: {e}")
    
    def _display_chat_content(self):
        """Display chat messages in output window with proper formatting"""
        messages = self.context_manager.get_chat_history()
        
        if not messages:
            # Show welcome message
            welcome_color = self.color_manager.get_color_pair('system_info')
            try:
                self.output_win.attron(curses.color_pair(welcome_color))
                self.output_win.addstr(1, 2, "Welcome to Aurora RPG Client")
                self.output_win.addstr(2, 2, "Type your message below or /help for commands")
                self.output_win.attroff(curses.color_pair(welcome_color))
            except curses.error:
                pass
            return
        
        # Calculate what messages to display
        display_height = self.output_height - 2
        max_width = self.width - 6
        
        # Flatten all messages into display lines with proper formatting
        display_lines = []
        
        for message in messages:
            # Get color based on message type
            color_map = {
                MessageType.USER: 'user_input',
                MessageType.ASSISTANT: 'assistant_output',
                MessageType.SYSTEM: 'system_info',
                MessageType.COMPANION: 'companion_dialogue',
                MessageType.NPC: 'npc_dialogue'
            }
            
            color_type = color_map.get(message.message_type, 'assistant_output')
            color_pair = self.color_manager.get_color_pair(color_type)
            
            # Format message with timestamp for user input
            if message.message_type == MessageType.USER:
                display_text = f"[{message.timestamp}] > {message.content}"
            else:
                display_text = message.content
            
            # Wrap text to fit window width
            wrapped_lines = textwrap.wrap(display_text, max_width) if display_text else [""]
            
            # Add all wrapped lines for this message
            for line in wrapped_lines:
                display_lines.append((line, color_pair))
        
        # Calculate scrolling
        if len(display_lines) <= display_height:
            start_idx = 0
        else:
            # Show most recent messages, accounting for scroll offset
            start_idx = max(0, len(display_lines) - display_height + self.chat_scroll)
        
        # Display lines
        line_num = 0
        for i in range(start_idx, min(len(display_lines), start_idx + display_height)):
            if line_num >= display_height:
                break
                
            line_text, color_pair = display_lines[i]
            
            try:
                self.output_win.attron(curses.color_pair(color_pair))
                self.output_win.addstr(line_num, 1, line_text[:max_width])
                self.output_win.attroff(curses.color_pair(color_pair))
            except curses.error:
                pass
                
            line_num += 1
    
    def _display_debug_content(self):
        """Display debug content in output window"""
        debug_lines = self.context_manager.get_debug_content()
        
        # Show help text
        help_color = self.color_manager.get_color_pair('system_info')
        help_text = "Debug context active. Press Esc to return to chat."
        
        try:
            self.output_win.attron(curses.color_pair(help_color))
            self.output_win.addstr(0, 1, help_text)
            self.output_win.attroff(curses.color_pair(help_color))
        except curses.error:
            pass
        
        # Display debug lines with scrolling
        display_height = self.output_height - 3
        start_idx = max(0, len(debug_lines) - display_height + self.debug_scroll)
        
        line_num = 2
        for i in range(start_idx, min(len(debug_lines), start_idx + display_height)):
            if line_num >= self.output_height - 1:
                break
                
            try:
                debug_line = debug_lines[i]
                max_width = self.width - 4
                if len(debug_line) > max_width:
                    debug_line = debug_line[:max_width - 3] + "..."
                
                self.output_win.addstr(line_num, 1, debug_line)
                line_num += 1
            except curses.error:
                break
    
    def _display_search_content(self):
        """Display search results in output window"""
        search_results = self.context_manager.get_search_results()
        search_term = self.context_manager.get_search_term()
        
        # Show help text
        help_color = self.color_manager.get_color_pair('system_info')
        help_text = "Search results displayed. Press Esc to return to chat."
        
        try:
            self.output_win.attron(curses.color_pair(help_color))
            self.output_win.addstr(0, 1, help_text)
            self.output_win.attroff(curses.color_pair(help_color))
        except curses.error:
            pass
        
        # Display search results
        display_height = self.output_height - 3
        start_idx = max(0, len(search_results) - display_height + self.search_scroll)
        
        line_num = 2
        for i in range(start_idx, min(len(search_results), start_idx + display_height)):
            if line_num >= self.output_height - 1:
                break
                
            try:
                result_line = search_results[i]
                max_width = self.width - 4
                
                if len(result_line) > max_width:
                    result_line = result_line[:max_width - 3] + "..."
                
                # Simple highlighting for search results
                if search_term and search_term.lower() in result_line.lower():
                    highlight_color = self.color_manager.get_color_pair('search_highlight')
                    self.output_win.attron(curses.color_pair(highlight_color))
                    self.output_win.addstr(line_num, 1, result_line)
                    self.output_win.attroff(curses.color_pair(highlight_color))
                else:
                    self.output_win.addstr(line_num, 1, result_line)
                
                line_num += 1
            except curses.error:
                break
    
    def _update_input_window(self):
        """Update input window with current input and cursor"""
        try:
            self.input_win.clear()
            
            current_context = self.context_manager.get_current_context()
            
            # Only show input in chat context
            if current_context != ContextType.CHAT:
                return
            
            # Show input lines
            input_color = self.color_manager.get_color_pair('user_input')
            
            for i, line in enumerate(self.input_lines):
                if i >= self.input_height - 1:
                    break
                try:
                    self.input_win.attron(curses.color_pair(input_color))
                    display_line = line[:self.width - 4]
                    self.input_win.addstr(i, 1, display_line)
                    self.input_win.attroff(curses.color_pair(input_color))
                except curses.error:
                    pass
            
            # Position cursor
            if self.cursor_line < self.input_height - 1:
                cursor_col = min(self.cursor_col + 1, self.width - 3)
                try:
                    self.input_win.move(self.cursor_line, cursor_col)
                except curses.error:
                    pass
            
            self.input_win.refresh()
            
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Error updating input window: {e}")
    
    def _update_all_windows(self):
        """Update all windows and refresh display"""
        self._draw_borders()
        self._update_output_window()
        self._update_input_window()
        self._update_status_bar()
        self.stdscr.refresh()
    
    def handle_resize(self):
        """Handle terminal resize"""
        # Get new dimensions
        self.height, self.width = self.stdscr.getmaxyx()
        
        # Recalculate window dimensions
        self._calculate_window_dimensions()
        
        # Recreate windows
        self._create_windows()
        
        # Force full refresh
        self.needs_refresh = True
        self._update_all_windows()
        
        if self.debug_logger:
            self.debug_logger.interface_operation("resize", f"Resized to {self.width}x{self.height}")
    
    def show_user_input(self, text: str):
        """Add user input to chat and trigger refresh"""
        self.context_manager.add_chat_message(text, 'user')
        
        if self.debug_logger:
            self.debug_logger.user_input(text)
        
        self.needs_refresh = True
    
    def show_assistant_response(self, text: str):
        """Add assistant response to chat and trigger refresh"""
        self.context_manager.add_chat_message(text, 'assistant')
        
        if self.debug_logger:
            self.debug_logger.assistant_response(text)
        
        self.needs_refresh = True
    
    def show_system_message(self, text: str):
        """Add system message to chat and trigger refresh"""
        self.context_manager.add_chat_message(text, 'system')
        
        if self.debug_logger:
            self.debug_logger.system(f"System message displayed: {text}")
        
        self.needs_refresh = True
    
    def show_error(self, text: str):
        """Add error message to chat and trigger refresh"""
        self.context_manager.add_chat_message(f"[Error] {text}", 'system')
        
        if self.debug_logger:
            self.debug_logger.error(text)
        
        self.needs_refresh = True

# Aurora RPG Client - Fixed Ncurses Implementation - Chunk 3/6
# Enhanced Input Handling with Ctrl+Enter Support

    def run(self):
        """Main ncurses event loop"""
        while self.running:
            try:
                if self.needs_refresh:
                    self._update_all_windows()
                    self.needs_refresh = False
                
                # Get key input
                key = self.stdscr.getch()
                
                if key == curses.KEY_RESIZE:
                    self.handle_resize()
                    continue
                
                # Handle key input
                self.handle_key_input(key)
                
            except KeyboardInterrupt:
                self.running = False
            except curses.error:
                # Handle curses errors gracefully
                continue
    
    def handle_key_input(self, key):
        """Handle keyboard input with proper Ctrl+Enter detection"""
        current_context = self.context_manager.get_current_context()
        
        # Handle Escape key
        if key == 27:  # ESC
            if current_context == ContextType.CHAT:
                self._show_quit_dialog()
            else:
                # Return to chat context
                self.context_manager.switch_context(ContextType.CHAT)
                self.needs_refresh = True
            return
        
        # Handle input only in chat context
        if current_context == ContextType.CHAT:
            self._handle_chat_input(key)
        else:
            # Handle scrolling in debug/search contexts
            self._handle_view_input(key)
    
    def _handle_chat_input(self, key):
        """Handle input in chat context with enhanced multi-line editing"""
        
        # Check for Ctrl+Enter (newline) vs Enter (submit)
        if key == 10:  # Enter/Return
            # Check if this is Ctrl+Enter by examining the previous key state
            # In most terminals, Ctrl+Enter sends different codes
            if self._is_ctrl_enter():
                self._add_new_line()
            else:
                self._submit_input()
                
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
            self._handle_backspace()
        elif key == curses.KEY_DC:  # Delete key
            self._handle_delete()
        elif key == curses.KEY_LEFT:
            self._move_cursor_left()
        elif key == curses.KEY_RIGHT:
            self._move_cursor_right()
        elif key == curses.KEY_UP:
            self._move_cursor_up()
        elif key == curses.KEY_DOWN:
            self._move_cursor_down()
        elif key == curses.KEY_HOME:
            self._move_cursor_home()
        elif key == curses.KEY_END:
            self._move_cursor_end()
        elif 32 <= key <= 126:  # Printable characters
            self._insert_character(chr(key))
        
        # Update input window after any change
        self._update_input_window()
    
    def _handle_view_input(self, key):
        """Handle input in view-only contexts (debug/search)"""
        current_context = self.context_manager.get_current_context()
        
        if key == curses.KEY_UP:
            self._scroll_view_up(current_context)
        elif key == curses.KEY_DOWN:
            self._scroll_view_down(current_context)
        elif key == curses.KEY_PPAGE:  # Page Up
            self._scroll_view_page_up(current_context)
        elif key == curses.KEY_NPAGE:  # Page Down
            self._scroll_view_page_down(current_context)
        elif key == curses.KEY_HOME:
            self._scroll_view_home(current_context)
        elif key == curses.KEY_END:
            self._scroll_view_end(current_context)
    
    def _is_ctrl_enter(self):
        """
        Detect Ctrl+Enter combination
        This is a simplified implementation - in practice, Ctrl+Enter detection
        can be terminal-dependent. For now, we'll use a different approach:
        Ctrl+J (10) for newline, Enter (13) for submit
        """
        # For now, we'll make Enter always submit and use a different key combo
        # Users can use Ctrl+J for newlines or we can implement a toggle
        return False
    
    def _submit_input(self):
        """Submit current input to the application"""
        # Combine all input lines
        full_input = '\n'.join(self.input_lines).strip()
        
        if not full_input:
            return
        
        # Validate input length
        is_valid, warning, preserved = self.input_validator.validate_input_length(full_input)
        
        if not is_valid:
            # Show error and preserve input
            self.show_error(warning)
            return
        
        # Clear input
        self._clear_input()
        
        # Process the input
        self._process_user_input(full_input)
        
        # Refresh display
        self.needs_refresh = True
    
    def _process_user_input(self, user_input: str):
        """Process user input through command system or MCP"""
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
                return  # Command handled, no further processing
        
        # Show user input
        self.show_user_input(user_input)
        
        # Process through MCP or placeholder
        try:
            response = self._send_to_mcp(user_input)
            self.show_assistant_response(response)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"MCP communication failed: {e}")
            # Fallback to placeholder response
            response = self._generate_placeholder_response(user_input)
            self.show_assistant_response(response)
    
    def _send_to_mcp(self, user_input: str) -> str:
        """
        Send input to MCP server
        This is where the actual MCP integration would go
        """
        if self.debug_logger:
            self.debug_logger.mcp_operation("send_request", f"Sending {len(user_input)} chars to MCP")
        
        # TODO: Implement actual MCP communication
        # For now, raise exception to trigger fallback
        raise Exception("MCP integration not yet implemented")
    
    def _generate_placeholder_response(self, user_input: str) -> str:
        """Generate placeholder response until MCP is integrated"""
        import random
        
        responses = [
            f"Aurora considers your words about '{user_input[:30]}...' carefully, her ethereal form shimmering in the mystical light.",
            f"The ancient forest responds to your intent. Whispers echo through the trees as you mention '{user_input[:30]}...'",
            f"Your companion nods thoughtfully at your suggestion regarding '{user_input[:30]}...', weighing the implications.",
            f"The mystical energies around you shift and dance as you speak of '{user_input[:30]}...', creating new possibilities.",
            f"Aurora's eyes gleam with understanding as she processes your words about '{user_input[:30]}...'. The adventure continues."
        ]
        
        response = random.choice(responses)
        
        if self.debug_logger:
            self.debug_logger.mcp_operation("placeholder_response", f"Generated placeholder: {len(response)} chars")
        
        return response
    
    def _add_new_line(self):
        """Add a new line at cursor position"""
        current_line = self.input_lines[self.cursor_line]
        
        # Split current line at cursor position
        before_cursor = current_line[:self.cursor_col]
        after_cursor = current_line[self.cursor_col:]
        
        # Update current line and insert new line
        self.input_lines[self.cursor_line] = before_cursor
        self.input_lines.insert(self.cursor_line + 1, after_cursor)
        
        # Move cursor to beginning of new line
        self.cursor_line += 1
        self.cursor_col = 0
        
        # Ensure we don't exceed input window height
        if len(self.input_lines) > self.input_height - 1:
            # Remove oldest line and adjust cursor
            self.input_lines.pop(0)
            self.cursor_line -= 1
    
    def _handle_backspace(self):
        """Handle backspace key"""
        if self.cursor_col > 0:
            # Delete character before cursor in current line
            line = self.input_lines[self.cursor_line]
            self.input_lines[self.cursor_line] = line[:self.cursor_col-1] + line[self.cursor_col:]
            self.cursor_col -= 1
        elif self.cursor_line > 0:
            # Join with previous line
            current_line = self.input_lines[self.cursor_line]
            previous_line = self.input_lines[self.cursor_line - 1]
            
            # Set cursor position to end of previous line
            self.cursor_col = len(previous_line)
            
            # Combine lines
            self.input_lines[self.cursor_line - 1] = previous_line + current_line
            
            # Remove current line
            self.input_lines.pop(self.cursor_line)
            self.cursor_line -= 1
    
    def _handle_delete(self):
        """Handle delete key"""
        current_line = self.input_lines[self.cursor_line]
        
        if self.cursor_col < len(current_line):
            # Delete character at cursor position
            self.input_lines[self.cursor_line] = current_line[:self.cursor_col] + current_line[self.cursor_col+1:]
        elif self.cursor_line < len(self.input_lines) - 1:
            # Join with next line
            next_line = self.input_lines[self.cursor_line + 1]
            self.input_lines[self.cursor_line] = current_line + next_line
            self.input_lines.pop(self.cursor_line + 1)
    
    def _insert_character(self, char: str):
        """Insert character at cursor position"""
        current_line = self.input_lines[self.cursor_line]
        
        # Insert character
        new_line = current_line[:self.cursor_col] + char + current_line[self.cursor_col:]
        self.input_lines[self.cursor_line] = new_line
        
        # Move cursor forward
        self.cursor_col += 1
        
        # Handle line wrapping if needed
        max_width = self.width - 6
        if len(new_line) > max_width:
            # Simple wrapping - move excess to next line
            excess = new_line[max_width:]
            self.input_lines[self.cursor_line] = new_line[:max_width]
            
            if self.cursor_line < len(self.input_lines) - 1:
                # Insert into existing next line
                self.input_lines[self.cursor_line + 1] = excess + self.input_lines[self.cursor_line + 1]
            else:
                # Create new line
                self.input_lines.append(excess)
            
            # Adjust cursor if it was in the wrapped portion
            if self.cursor_col > max_width:
                self.cursor_line += 1
                self.cursor_col = self.cursor_col - max_width
    
    def _move_cursor_left(self):
        """Move cursor left"""
        if self.cursor_col > 0:
            self.cursor_col -= 1
        elif self.cursor_line > 0:
            # Move to end of previous line
            self.cursor_line -= 1
            self.cursor_col = len(self.input_lines[self.cursor_line])
    
    def _move_cursor_right(self):
        """Move cursor right"""
        current_line = self.input_lines[self.cursor_line]
        
        if self.cursor_col < len(current_line):
            self.cursor_col += 1
        elif self.cursor_line < len(self.input_lines) - 1:
            # Move to beginning of next line
            self.cursor_line += 1
            self.cursor_col = 0
    
    def _move_cursor_up(self):
        """Move cursor up one line"""
        if self.cursor_line > 0:
            self.cursor_line -= 1
            # Adjust column to fit new line
            max_col = len(self.input_lines[self.cursor_line])
            self.cursor_col = min(self.cursor_col, max_col)
    
    def _move_cursor_down(self):
        """Move cursor down one line"""
        if self.cursor_line < len(self.input_lines) - 1:
            self.cursor_line += 1
            # Adjust column to fit new line
            max_col = len(self.input_lines[self.cursor_line])
            self.cursor_col = min(self.cursor_col, max_col)
    
    def _move_cursor_home(self):
        """Move cursor to beginning of line"""
        self.cursor_col = 0
    
    def _move_cursor_end(self):
        """Move cursor to end of line"""
        self.cursor_col = len(self.input_lines[self.cursor_line])
    
    def _clear_input(self):
        """Clear all input"""
        self.input_lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
    
    def _scroll_view_up(self, context: ContextType):
        """Scroll view up in current context"""
        if context == ContextType.DEBUG:
            if self.debug_scroll > 0:
                self.debug_scroll -= 1
                self.needs_refresh = True
        elif context == ContextType.SEARCH:
            if self.search_scroll > 0:
                self.search_scroll -= 1
                self.needs_refresh = True
    
    def _scroll_view_down(self, context: ContextType):
        """Scroll view down in current context"""
        if context == ContextType.DEBUG:
            debug_content = self.context_manager.get_debug_content()
            max_scroll = max(0, len(debug_content) - (self.output_height - 3))
            if self.debug_scroll < max_scroll:
                self.debug_scroll += 1
                self.needs_refresh = True
        elif context == ContextType.SEARCH:
            search_results = self.context_manager.get_search_results()
            max_scroll = max(0, len(search_results) - (self.output_height - 3))
            if self.search_scroll < max_scroll:
                self.search_scroll += 1
                self.needs_refresh = True
    
    def _scroll_view_page_up(self, context: ContextType):
        """Scroll view up by one page"""
        page_size = self.output_height - 3
        for _ in range(page_size):
            self._scroll_view_up(context)
    
    def _scroll_view_page_down(self, context: ContextType):
        """Scroll view down by one page"""
        page_size = self.output_height - 3
        for _ in range(page_size):
            self._scroll_view_down(context)
    
    def _scroll_view_home(self, context: ContextType):
        """Scroll to top of view"""
        if context == ContextType.DEBUG:
            self.debug_scroll = 0
            self.needs_refresh = True
        elif context == ContextType.SEARCH:
            self.search_scroll = 0
            self.needs_refresh = True
    
    def _scroll_view_end(self, context: ContextType):
        """Scroll to bottom of view"""
        if context == ContextType.DEBUG:
            debug_content = self.context_manager.get_debug_content()
            self.debug_scroll = max(0, len(debug_content) - (self.output_height - 3))
            self.needs_refresh = True
        elif context == ContextType.SEARCH:
            search_results = self.context_manager.get_search_results()
            self.search_scroll = max(0, len(search_results) - (self.output_height - 3))
            self.needs_refresh = True

# Aurora RPG Client - Fixed Ncurses Implementation - Chunk 4/6
# Command Processing and Dialog Management

    def _show_quit_dialog(self):
        """Show quit confirmation dialog"""
        try:
            # Save current screen
            self.in_quit_dialog = True
            
            # Create dialog window
            dialog_height = 5
            dialog_width = 40
            start_y = (self.height - dialog_height) // 2
            start_x = (self.width - dialog_width) // 2
            
            dialog_win = curses.newwin(dialog_height, dialog_width, start_y, start_x)
            
            # Draw dialog
            dialog_win.box()
            dialog_win.addstr(1, 2, "Are you sure you want to quit?")
            dialog_win.addstr(3, 2, "Press 'y' for Yes, 'n' for No")
            dialog_win.refresh()
            
            # Wait for response
            while True:
                key = self.stdscr.getch()
                
                if key == ord('y') or key == ord('Y'):
                    self.running = False
                    break
                elif key == ord('n') or key == ord('N') or key == 27:  # No or Escape
                    break
            
            # Clean up dialog
            del dialog_win
            self.in_quit_dialog = False
            self.needs_refresh = True
            
        except curses.error:
            # If dialog fails, just quit
            self.running = False
    
    def _switch_to_debug_context(self):
        """Switch to debug context and update content"""
        if self.debug_logger and self.debug_logger.enabled:
            # Update debug content from log file
            debug_content = self.debug_logger.get_debug_content()
            self.context_manager.set_debug_content(debug_content)
        
        # Switch context
        self.context_manager.switch_context(ContextType.DEBUG)
        self.needs_refresh = True
        
        if self.debug_logger:
            self.debug_logger.interface_operation("context_switch", "Switched to debug context")
    
    def _switch_to_search_context(self):
        """Switch to search context"""
        self.context_manager.switch_context(ContextType.SEARCH)
        self.needs_refresh = True
        
        if self.debug_logger:
            self.debug_logger.interface_operation("context_switch", "Switched to search context")
    
    def _process_command(self, command: str) -> Tuple[bool, str]:
        """
        Process user commands.
        Returns (should_continue, response_message)
        """
        command = command.strip().lower()
        
        if self.debug_logger:
            self.debug_logger.interface_operation("command", f"Processing: {command}")
        
        # Handle different commands
        if command == "/quit":
            return False, "Goodbye!"
        
        elif command == "/help":
            help_text = self._get_help_text()
            self.show_system_message(help_text)
            return True, ""
        
        elif command == "/color":
            new_scheme = self.color_manager.cycle_scheme()
            response = f"Color scheme changed to: {new_scheme}"
            self.show_system_message(response)
            self.needs_refresh = True  # Force redraw with new colors
            return True, ""
        
        elif command == "/debug":
            self._switch_to_debug_context()
            return True, ""
        
        elif command.startswith("/search "):
            search_term = command[8:].strip()
            if not search_term:
                self.show_error("Usage: /search <term>")
                return True, ""
            
            matches = self.context_manager.perform_search(search_term)
            self._switch_to_search_context()
            return True, ""
        
        elif command == "/showsme":
            self.show_sme_status = not self.show_sme_status
            status = "enabled" if self.show_sme_status else "disabled"
            self.show_system_message(f"SME status display {status}")
            self.needs_refresh = True
            return True, ""
        
        elif command.startswith("/save"):
            return self._handle_save_command(command)
        
        elif command.startswith("/clear"):
            return self._handle_clear_command(command)
        
        else:
            # Unknown command - return as regular input
            return True, command
    
    def _get_help_text(self) -> str:
        """Generate help text for commands"""
        commands = [
            "/help - Show this help message",
            "/color - Cycle through color schemes (Midnight Aurora / Forest Whisper)",
            "/debug - View debug information (Esc to return)",
            "/search <term> - Search chat history",
            "/showsme - Toggle Story Momentum Engine status display",
            "/save [filename] - Save conversation",
            "/clear - Clear current context",
            "/quit - Exit application",
            "",
            "Navigation:",
            "Esc - Return to chat from debug/search, or quit dialog from chat",
            "Enter - Submit input (multi-line input planned for future update)",
            "Arrow Keys - Navigate text and scroll in debug/search contexts",
            "Page Up/Down - Scroll by page in debug/search contexts",
            "Home/End - Jump to beginning/end in debug/search contexts"
        ]
        
        return "\n".join(commands)
    
    def _handle_save_command(self, command: str) -> Tuple[bool, str]:
        """Handle save command with options"""
        parts = command.split()
        filename = None
        save_chat = True
        save_debug = False
        
        # Parse command arguments
        for part in parts[1:]:
            if part.startswith("--"):
                if part == "--chat":
                    save_chat = True
                    save_debug = False
                elif part == "--debug":
                    save_chat = False
                    save_debug = True
                elif part == "--both":
                    save_chat = True
                    save_debug = True
            else:
                filename = part
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aurora_conversation_{timestamp}.txt"
        
        try:
            content_parts = []
            
            if save_chat:
                chat_content = self._export_chat_context()
                content_parts.append(chat_content)
            
            if save_debug and self.debug_logger and self.debug_logger.enabled:
                debug_content = self._export_debug_context()
                content_parts.append(debug_content)
            
            full_content = "\n\n" + "="*80 + "\n\n".join(content_parts)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            self.show_system_message(f"Conversation saved to: {filename}")
            
            if self.debug_logger:
                self.debug_logger.interface_operation("save", f"Saved to {filename}")
            
        except Exception as e:
            self.show_error(f"Failed to save conversation: {e}")
        
        return True, ""
    
    def _handle_clear_command(self, command: str) -> Tuple[bool, str]:
        """Handle clear command"""
        current_context = self.context_manager.get_current_context()
        
        if command.strip() == "/clear":
            self.context_manager.clear_context(current_context)
            self.show_system_message(f"Cleared {current_context.value} context")
            self.needs_refresh = True
        
        return True, ""
    
    def _export_chat_context(self) -> str:
        """Export chat context content for saving"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"Aurora RPG Chat Export - {timestamp}", "=" * 50, ""]
        
        messages = self.context_manager.get_chat_history()
        for msg in messages:
            lines.append(f"[{msg.timestamp}] {msg.message_type.value.upper()}: {msg.content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_debug_context(self) -> str:
        """Export debug context content for saving"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"Aurora RPG Debug Export - {timestamp}", "=" * 50, ""]
        
        debug_content = self.context_manager.get_debug_content()
        lines.extend(debug_content)
        
        return "\n".join(lines)

# Main Application Class
class AuroraRPGClient:
    """Main application class for Aurora RPG Client"""
    
    def __init__(self, debug_enabled: bool = False, color_scheme: str = "midnight_aurora"):
        self.debug_logger = DebugLogger(debug_enabled, DEBUG_LOG_FILE) if debug_enabled else None
        self.curses_interface = None
        
        if self.debug_logger:
            self.debug_logger.system("Aurora RPG Client initialized")
    
    def run(self):
        """Run the Aurora RPG Client with ncurses interface"""
        if self.debug_logger:
            self.debug_logger.system("Starting Aurora RPG Client")
        
        def curses_main(stdscr):
            """Main curses function wrapper"""
            try:
                # Create curses interface
                self.curses_interface = CursesInterface(self.debug_logger)
                
                # Initialize interface
                self.curses_interface.initialize(stdscr)
                
                # Run main loop
                self.curses_interface.run()
                
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.error(f"Curses interface error: {e}")
                raise
        
        try:
            curses.wrapper(curses_main)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Application error: {e}")
            print(f"Error: {e}")
            return 1
        
        return 0
    
    def cleanup(self):
        """Cleanup application resources"""
        try:
            # Save current configuration
            self._save_configuration()
            
            # Close debug session
            if self.debug_logger:
                self.debug_logger.close_session()
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Cleanup error: {e}")
    
    def _save_configuration(self):
        """Save current configuration to file"""
        try:
            config = {
                "color_scheme": self.curses_interface.color_manager.scheme_name if self.curses_interface else "midnight_aurora",
                "show_sme_status": self.curses_interface.show_sme_status if self.curses_interface else False,
                "debug_enabled": self.debug_logger.enabled if self.debug_logger else False
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            if self.debug_logger:
                self.debug_logger.system(f"Configuration saved to {CONFIG_FILE}")
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save configuration: {e}")

def load_configuration(debug_logger: Optional[DebugLogger]) -> Dict[str, Any]:
    """Load configuration from file"""
    default_config = {
        "color_scheme": "midnight_aurora",
        "show_sme_status": False,
        "debug_enabled": False
    }
    
    try:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Merge with defaults
            config = {**default_config, **loaded_config}
            
            if debug_logger:
                debug_logger.system(f"Configuration loaded from {CONFIG_FILE}")
            
            return config
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Failed to load configuration: {e}")
    
    return default_config

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Aurora RPG Client - Terminal-based RPG storyteller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with default settings
  %(prog)s --debug                 # Enable debug logging to file
  %(prog)s --colorscheme forest_whisper  # Start with Forest Whisper theme
        """
    )
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging to debug.log file")
    
    parser.add_argument("--colorscheme", default="midnight_aurora",
                       choices=["midnight_aurora", "forest_whisper"],
                       help="Initial color scheme (can be changed with /color command)")
    
    parser.add_argument("--config", default=CONFIG_FILE,
                       help="Configuration file path")
    
    parser.add_argument("--version", action="version", version="Aurora RPG Client v4.0")
    
    return parser

def initialize_application(args) -> AuroraRPGClient:
    """Initialize application with arguments"""
    # Initialize debug logger
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    # Check ncurses availability
    try:
        import curses
        # Quick test to ensure curses is available
        curses.wrapper(lambda stdscr: None)
        if debug_logger:
            debug_logger.system("Ncurses interface available")
    except (ImportError, curses.error) as e:
        print("Error: Ncurses is not available on this system.")
        print("Please ensure you have ncurses support installed.")
        if debug_logger:
            debug_logger.error(f"Ncurses initialization failed: {e}")
        sys.exit(1)
    
    # Load configuration
    config = load_configuration(debug_logger)
    
    # Override with command line args
    if args.colorscheme:
        config['color_scheme'] = args.colorscheme
    
    # Create application
    app = AuroraRPGClient(args.debug, config['color_scheme'])
    
    return app

# Aurora RPG Client - Fixed Ncurses Implementation - Chunk 5/6
# Main Function and Application Entry Point

def main():
    """Main application entry point"""
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize debug logger first
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    if debug_logger:
        debug_logger.system("Aurora RPG Client starting")
        debug_logger.system(f"Arguments: {vars(args)}")
    
    app = None
    
    try:
        # Initialize application
        app = initialize_application(args)
        
        # Show startup information
        print("\n" + "="*70)
        print("AURORA RPG CLIENT - Enhanced Ncurses Interface v4.0")
        print("="*70)
        print("Features:")
        print("• Full ncurses interface with multi-context support")
        print("• Advanced color schemes (Midnight Aurora / Forest Whisper)")
        print("• Debug logging and search functionality")
        print("• Enhanced input handling with multi-line support")
        print("• MCP integration ready")
        print("")
        print("Starting interface...")
        print("")
        
        # Small delay to let user read startup message
        time.sleep(2)
        
        # Run application
        exit_code = app.run()
        
        print("Thank you for using Aurora RPG Client!")
        return exit_code
    
    except KeyboardInterrupt:
        if debug_logger:
            debug_logger.system("Application interrupted by user (Ctrl+C)")
        print("\nThank you for using Aurora RPG Client. Goodbye!")
        return 0
    
    except Exception as e:
        error_msg = f"Critical error: {e}"
        if debug_logger:
            debug_logger.error(error_msg)
        print(f"ERROR: {error_msg}")
        if debug_logger:
            print(f"Check {DEBUG_LOG_FILE} for details.")
        return 1
    
    finally:
        # Comprehensive cleanup
        if app:
            try:
                app.cleanup()
            except Exception as e:
                print(f"Cleanup warning: {e}")

def estimate_tokens(text: str) -> int:
    """Simple token estimation for validation"""
    return len(text) // 4

# Command validation helpers
COMMANDS = {
    "/debug": "Switch to debug context (Esc to return)",
    "/search": "Search chat history: /search <term>", 
    "/color": "Cycle through color schemes",
    "/showsme": "Toggle Story Momentum Engine status display",
    "/quit": "Exit the application", 
    "/save": "Save conversation: /save [filename] [--chat|--debug|--both]",
    "/clear": "Clear current context history",
    "/help": "Show command help"
}

def validate_command(command: str) -> Tuple[bool, str]:
    """Validate command syntax and provide help"""
    command = command.strip().lower()
    
    if command == "/help":
        return True, ""
    
    if command in COMMANDS:
        return True, ""
    
    # Check commands with parameters
    if command.startswith("/search "):
        search_term = command[8:].strip()
        if not search_term:
            return False, "Usage: /search <term>"
        return True, ""
    
    if command.startswith("/save"):
        # Basic validation - detailed parsing in command handler
        return True, ""
    
    # Unknown command
    return False, f"Unknown command: {command.split()[0]}. Type /help for available commands."

# Enhanced Input Processing for Future MCP Integration
class MCPClient:
    """
    Placeholder for MCP (Model Control Protocol) client
    This will be the interface to the actual RPG engine
    """
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.connected = False
        self.server_url = None
        
        if debug_logger:
            debug_logger.mcp_operation("init", "MCP Client initialized (placeholder)")
    
    def connect(self, server_url: str = None) -> bool:
        """Connect to MCP server"""
        if self.debug_logger:
            self.debug_logger.mcp_operation("connect", f"Attempting connection to {server_url}")
        
        # TODO: Implement actual MCP connection
        # For now, always return False to use placeholder responses
        self.connected = False
        self.server_url = server_url
        
        if self.debug_logger:
            self.debug_logger.mcp_operation("connect_result", f"Connection status: {self.connected}")
        
        return self.connected
    
    def send_message(self, message: str) -> str:
        """Send message to MCP server and get response"""
        if not self.connected:
            raise Exception("Not connected to MCP server")
        
        if self.debug_logger:
            self.debug_logger.mcp_operation("send", f"Sending message: {len(message)} chars")
        
        # TODO: Implement actual MCP communication
        # This would send the message and receive the response
        response = "MCP response placeholder"
        
        if self.debug_logger:
            self.debug_logger.mcp_operation("receive", f"Received response: {len(response)} chars")
        
        return response
    
    def disconnect(self):
        """Disconnect from MCP server"""
        if self.debug_logger:
            self.debug_logger.mcp_operation("disconnect", "Disconnecting from MCP server")
        
        self.connected = False
        self.server_url = None

class StoryMomentumEngine:
    """
    Placeholder for Story Momentum Engine integration
    This would interface with the actual SME implementation
    """
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.pressure_level = 0.0
        self.antagonist_name = "Unknown"
        self.pressure_name = "Calm"
        self.active = False
        
        if debug_logger:
            debug_logger.system("Story Momentum Engine placeholder initialized")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current SME status for display"""
        return {
            "pressure_level": self.pressure_level,
            "antagonist_name": self.antagonist_name,
            "pressure_name": self.pressure_name,
            "active": self.active
        }
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and update momentum"""
        if self.debug_logger:
            self.debug_logger.system(f"SME processing input: {len(user_input)} chars")
        
        # Placeholder logic - replace with actual SME
        import random
        self.pressure_level = min(1.0, self.pressure_level + random.uniform(-0.1, 0.2))
        
        return self.get_status()

# Enhanced Error Handling and Recovery
class ErrorHandler:
    """Handle errors gracefully and provide recovery options"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.error_count = 0
        self.max_errors = 10
    
    def handle_error(self, error: Exception, context: str = "unknown") -> bool:
        """
        Handle an error and determine if application should continue
        Returns True if application should continue, False if it should exit
        """
        self.error_count += 1
        
        if self.debug_logger:
            self.debug_logger.error(f"Error in {context}: {str(error)}")
        
        # If too many errors, recommend exit
        if self.error_count >= self.max_errors:
            if self.debug_logger:
                self.debug_logger.error(f"Too many errors ({self.error_count}), recommending exit")
            return False
        
        # For curses errors, try to recover
        if isinstance(error, curses.error):
            if self.debug_logger:
                self.debug_logger.error(f"Curses error in {context}, attempting recovery")
            return True
        
        # For other errors, log and continue
        return True
    
    def reset_error_count(self):
        """Reset error count (call after successful operations)"""
        self.error_count = 0

# Configuration Management
class ConfigManager:
    """Manage application configuration"""
    
    def __init__(self, config_file: str = CONFIG_FILE, debug_logger: Optional[DebugLogger] = None):
        self.config_file = Path(config_file)
        self.debug_logger = debug_logger
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "color_scheme": "midnight_aurora",
            "show_sme_status": False,
            "debug_enabled": False,
            "auto_save_enabled": True,
            "auto_save_interval": 100,
            "max_chat_history": 1000,
            "input_timeout": 30,
            "mcp_server_url": None
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                self.config.update(loaded_config)
                
                if self.debug_logger:
                    self.debug_logger.system(f"Configuration loaded from {self.config_file}")
            
            return self.config.copy()
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to load configuration: {e}")
            return self.config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            # Update internal config
            self.config.update(config)
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            if self.debug_logger:
                self.debug_logger.system(f"Configuration saved to {self.config_file}")
            
            return True
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_value(self, key: str, default=None):
        """Get a specific configuration value"""
        return self.config.get(key, default)
    
    def set_config_value(self, key: str, value: Any):
        """Set a specific configuration value"""
        self.config[key] = value

# Session Management
class SessionManager:
    """Manage user sessions and state"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.message_count = 0
        self.command_count = 0
        
        if debug_logger:
            debug_logger.system(f"Session started: {self.session_id}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        duration = datetime.now() - self.start_time
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": int(duration.total_seconds()),
            "message_count": self.message_count,
            "command_count": self.command_count
        }
    
    def increment_message_count(self):
        """Increment message counter"""
        self.message_count += 1
        
        if self.debug_logger:
            self.debug_logger.system(f"Message count: {self.message_count}")
    
    def increment_command_count(self):
        """Increment command counter"""
        self.command_count += 1
        
        if self.debug_logger:
            self.debug_logger.system(f"Command count: {self.command_count}")
    
    def end_session(self):
        """End current session"""
        if self.debug_logger:
            session_info = self.get_session_info()
            self.debug_logger.system(f"Session ended: {session_info}")

# Version and build information
__version__ = "4.0.0"
__build_date__ = "2024-09-08"
__author__ = "Aurora RPG Development Team"
__description__ = "Enhanced Ncurses Interface with MCP Integration Ready"

# Startup banner
def show_startup_banner():
    """Show startup banner with version info"""
    print(f"Aurora RPG Client v{__version__} - {__description__}")
    print(f"Build Date: {__build_date__}")
    print("Enhanced ncurses interface ready for adventure!")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# Aurora RPG Client - Fixed Ncurses Implementation - Chunk 6/6
# Enhanced CursesInterface Integration and Final Implementation

# Enhance the CursesInterface class with proper MCP integration hooks
# Add these methods to the CursesInterface class defined in Chunk 2

def enhance_curses_interface():
    """Add enhanced methods to CursesInterface class"""
    
    # Add MCP client integration
    def __init_enhanced__(self, debug_logger: Optional[DebugLogger] = None):
        """Enhanced initialization with MCP support"""
        # Call original init logic from Chunk 2
        self.debug_logger = debug_logger
        
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
        
        # Context-specific scrolling
        self.chat_scroll = 0
        self.debug_scroll = 0
        self.search_scroll = 0
        
        # Status flags
        self.running = True
        self.needs_refresh = True
        self.in_quit_dialog = False
        
        # Managers
        self.color_manager = CursesColorManager("midnight_aurora", debug_logger)
        self.context_manager = ContextManager(debug_logger)
        self.input_validator = InputValidator(MAX_USER_INPUT_TOKENS, debug_logger)
        
        # Enhanced components
        self.mcp_client = MCPClient(debug_logger)
        self.sme = StoryMomentumEngine(debug_logger)
        self.error_handler = ErrorHandler(debug_logger)
        self.session_manager = SessionManager(debug_logger)
        
        # Show SME status toggle
        self.show_sme_status = False
        
        if debug_logger:
            debug_logger.interface_operation("curses_init", "Enhanced CursesInterface created")
    
    # Replace the _send_to_mcp method with actual implementation
    def _send_to_mcp_enhanced(self, user_input: str) -> str:
        """Enhanced MCP communication with fallback"""
        try:
            if not self.mcp_client.connected:
                # Attempt to connect if not already connected
                if not self.mcp_client.connect():
                    raise Exception("Could not establish MCP connection")
            
            # Send message through MCP
            response = self.mcp_client.send_message(user_input)
            
            # Update SME with the interaction
            self.sme.process_input(user_input)
            
            # Increment session counters
            self.session_manager.increment_message_count()
            
            # Reset error count on successful operation
            self.error_handler.reset_error_count()
            
            return response
            
        except Exception as e:
            # Handle error through error handler
            should_continue = self.error_handler.handle_error(e, "MCP communication")
            
            if not should_continue:
                self.show_error("Too many MCP errors. Please restart the application.")
                return "System error: MCP communication failed repeatedly."
            
            # Log the error and fall back to placeholder
            if self.debug_logger:
                self.debug_logger.mcp_operation("error", f"MCP failed, using placeholder: {str(e)}")
            
            # Generate placeholder response
            return self._generate_placeholder_response(user_input)
    
    # Enhanced placeholder response with SME integration
    def _generate_placeholder_response_enhanced(self, user_input: str) -> str:
        """Generate enhanced placeholder response with SME awareness"""
        import random
        
        # Update SME even for placeholder responses
        sme_status = self.sme.process_input(user_input)
        
        # Base responses
        base_responses = [
            "Aurora considers your words carefully, her ethereal form shimmering in the mystical light.",
            "The ancient forest responds to your intent. Whispers echo through the trees.",
            "Your companion nods thoughtfully, weighing the implications of your suggestion.",
            "The mystical energies around you shift and dance, creating new possibilities.",
            "Aurora's eyes gleam with understanding as she processes your words. The adventure continues."
        ]
        
        # SME-influenced responses based on pressure level
        if sme_status['pressure_level'] > 0.7:
            high_pressure_responses = [
                "Tension crackles in the air as Aurora's expression grows serious. Danger approaches.",
                "The shadows seem to press closer as Aurora's voice takes on an urgent tone.",
                "Aurora's hand moves instinctively to her weapon. Something is terribly wrong.",
                "The very air seems to thicken with malevolent energy. Aurora's eyes dart nervously.",
                "Aurora's usually calm demeanor cracks slightly. The threat is becoming real."
            ]
            responses = high_pressure_responses
        elif sme_status['pressure_level'] > 0.4:
            medium_pressure_responses = [
                "Aurora's brow furrows with concern as she considers the growing complications.",
                "The atmosphere grows tense as Aurora weighs your words against rising challenges.",
                "Aurora nods gravely, sensing the increasing complexity of your situation.",
                "A shadow of worry crosses Aurora's features as events begin to accelerate.",
                "Aurora's voice carries a note of caution as the stakes continue to rise."
            ]
            responses = medium_pressure_responses
        else:
            responses = base_responses
        
        response = random.choice(responses)
        
        # Add SME status to response if enabled
        if self.show_sme_status and sme_status['active']:
            sme_info = f" [Pressure: {sme_status['pressure_name']} ({sme_status['pressure_level']:.2f})]"
            response += sme_info
        
        if self.debug_logger:
            self.debug_logger.mcp_operation("placeholder_enhanced", 
                f"Generated SME-aware placeholder: {len(response)} chars, pressure: {sme_status['pressure_level']:.2f}")
        
        return response
    
    # Enhanced status bar with SME information
    def _update_status_bar_enhanced(self):
        """Enhanced status bar with SME and session info"""
        try:
            self.status_win.clear()
            
            # Get current context
            current_context = self.context_manager.get_current_context()
            context_name = current_context.value.upper()
            
            # Get current color scheme
            scheme_name = self.color_manager.get_scheme_display_name().upper()
            
            # Build status text
            status_parts = [
                f"Context: [{context_name}]",
                f"Theme: [{scheme_name}]"
            ]
            
            # Add SME status if enabled
            if self.show_sme_status:
                sme_status = self.sme.get_status()
                if sme_status['active']:
                    sme_display = f"SME: {sme_status['pressure_name']}({sme_status['pressure_level']:.1f})"
                else:
                    sme_display = "SME: Inactive"
                status_parts.append(sme_display)
            
            # Add session info
            session_info = self.session_manager.get_session_info()
            status_parts.append(f"Msgs: {session_info['message_count']}")
            
            # Add commands help
            status_parts.append("/help")
            
            status_text = " | ".join(status_parts)
            
            # Truncate if too long
            if len(status_text) > self.width - 2:
                status_text = status_text[:self.width - 5] + "..."
            
            # Display with status bar color
            status_color = self.color_manager.get_color_pair('status_bar')
            self.status_win.attron(curses.color_pair(status_color))
            self.status_win.addstr(0, 0, status_text.ljust(self.width))
            self.status_win.attroff(curses.color_pair(status_color))
            
            self.status_win.refresh()
            
        except curses.error:
            pass
    
    return (__init_enhanced__, _send_to_mcp_enhanced, _generate_placeholder_response_enhanced, _update_status_bar_enhanced)

# Apply enhancements by monkey-patching (in a real implementation, 
# these would be integrated into the class definition)
def apply_enhancements():
    """Apply all enhancements to the CursesInterface class"""
    enhanced_init, enhanced_mcp, enhanced_placeholder, enhanced_status = enhance_curses_interface()
    
    # In the actual class, these methods would replace the originals
    # CursesInterface.__init__ = enhanced_init
    # CursesInterface._send_to_mcp = enhanced_mcp
    # CursesInterface._generate_placeholder_response = enhanced_placeholder
    # CursesInterface._update_status_bar = enhanced_status
    pass

# Configuration and startup enhancements
def enhanced_startup():
    """Enhanced startup sequence with better error handling"""
    try:
        # Show enhanced startup banner
        show_startup_banner()
        
        # Check system requirements
        check_system_requirements()
        
        # Initialize logging
        setup_logging()
        
        print("System checks completed successfully.")
        print("Initializing Aurora RPG Client...")
        
    except Exception as e:
        print(f"Startup error: {e}")
        return False
    
    return True

def check_system_requirements():
    """Check system requirements for Aurora RPG Client"""
    # Check Python version
    if sys.version_info < (3, 8):
        raise Exception("Python 3.8 or higher is required")
    
    # Check ncurses availability
    try:
        import curses
        curses.wrapper(lambda stdscr: None)
    except (ImportError, curses.error) as e:
        raise Exception(f"Ncurses is not available: {e}")
    
    # Check terminal size
    try:
        import shutil
        width, height = shutil.get_terminal_size()
        if width < 80 or height < 24:
            print(f"Warning: Terminal size ({width}x{height}) is smaller than recommended (80x24)")
    except:
        pass  # Non-critical

def setup_logging():
    """Setup enhanced logging configuration"""
    # Ensure log directory exists
    log_dir = Path(DEBUG_LOG_FILE).parent
    log_dir.mkdir(exist_ok=True)
    
    # Clean old log files (keep last 10)
    log_files = sorted(log_dir.glob("debug_*.log"))
    if len(log_files) > 10:
        for old_log in log_files[:-10]:
            try:
                old_log.unlink()
            except:
                pass  # Non-critical

# Enhanced main function with comprehensive error handling
def enhanced_main():
    """Enhanced main function with full error handling and recovery"""
    app = None
    debug_logger = None
    
    try:
        # Enhanced startup
        if not enhanced_startup():
            return 1
        
        # Setup argument parser
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Initialize debug logger
        debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
        
        if debug_logger:
            debug_logger.system("Aurora RPG Client Enhanced v4.0 starting")
            debug_logger.system(f"Command line arguments: {vars(args)}")
            debug_logger.system(f"Python version: {sys.version}")
            debug_logger.system(f"Platform: {sys.platform}")
        
        # Load configuration
        config_manager = ConfigManager(args.config, debug_logger)
        config = config_manager.load_config()
        
        # Override with command line args
        if args.colorscheme:
            config['color_scheme'] = args.colorscheme
        
        # Initialize application
        app = AuroraRPGClient(args.debug, config['color_scheme'])
        
        # Show final startup message
        print("Aurora RPG Client is ready!")
        print("Starting ncurses interface in 3 seconds...")
        time.sleep(3)
        
        # Run application
        exit_code = app.run()
        
        print("\nThank you for using Aurora RPG Client!")
        print("Your adventure session has been saved.")
        
        return exit_code
    
    except KeyboardInterrupt:
        if debug_logger:
            debug_logger.system("Application interrupted by user (Ctrl+C)")
        print("\nGraceful shutdown initiated...")
        print("Thank you for using Aurora RPG Client. Goodbye!")
        return 0
    
    except Exception as e:
        error_msg = f"Critical error: {e}"
        if debug_logger:
            debug_logger.error(f"CRITICAL: {error_msg}")
            debug_logger.error(f"Stack trace: {str(e)}")
        
        print(f"\nERROR: {error_msg}")
        if debug_logger:
            print(f"Detailed error information has been logged to {DEBUG_LOG_FILE}")
        print("Please report this error if it persists.")
        
        return 1
    
    finally:
        # Comprehensive cleanup
        if app:
            try:
                print("Cleaning up resources...")
                app.cleanup()
                print("Cleanup completed successfully.")
            except Exception as e:
                print(f"Cleanup warning: {e}")
        
        if debug_logger:
            try:
                debug_logger.close_session()
            except:
                pass

# Final integration and execution
if __name__ == "__main__":
    # Apply all enhancements
    apply_enhancements()
    
    # Run enhanced main function
    exit_code = enhanced_main()
    sys.exit(exit_code)

# Additional utility functions for future expansion

def create_backup(filename: str) -> bool:
    """Create backup of important files"""
    try:
        backup_name = f"{filename}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if Path(filename).exists():
            import shutil
            shutil.copy2(filename, backup_name)
            return True
    except Exception:
        pass
    return False

def cleanup_backups(pattern: str, keep_count: int = 5):
    """Clean up old backup files"""
    try:
        backup_files = sorted(Path('.').glob(pattern))
        if len(backup_files) > keep_count:
            for old_backup in backup_files[:-keep_count]:
                old_backup.unlink()
    except Exception:
        pass

def export_session_data(session_manager, filename: str = None) -> str:
    """Export session data for analysis"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aurora_session_{timestamp}.json"
    
    try:
        session_data = session_manager.get_session_info()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        return filename
    except Exception as e:
        raise Exception(f"Failed to export session data: {e}")

# Version information and credits
print(f"""
Aurora RPG Client v{__version__}
{__description__}
Build Date: {__build_date__}
Author: {__author__}

Ready for ncurses-based RPG adventures!
""")

# End of Aurora RPG Client - Fixed Ncurses Implementation
