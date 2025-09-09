# Chunk 1/4 - nci.py - Core Classes and Dependencies - Rewritten with Immediate Display
#!/usr/bin/env python3
"""
DevName RPG Client - Ncurses Interface Module (nci.py) - REWRITTEN

Module architecture and interconnects documented in genai.txt
Maintains programmatic interfaces with mcp.py, emm.py, and sme.py
Integrates loaded prompts from main.py for enhanced RPG experience

REWRITE CHANGES:
- Immediate window refresh pattern (no batched updates)
- Explicit cursor management after every operation
- True blank lines between user/AI messages (no system prefix)
- Improved message wrapping with textwrap
- Phased initialization with forced immediate display
"""

import curses
import time
import textwrap
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# Import module dependencies
try:
    from mcp import MCPClient
    from emm import EnhancedMemoryManager, MessageType
    from sme import StoryMomentumEngine
except ImportError as e:
    print(f"Module import failed: {e}")
    raise

# Configuration constants
MAX_USER_INPUT_TOKENS = 2000
MIN_SCREEN_WIDTH = 80
MIN_SCREEN_HEIGHT = 24

class ColorTheme(Enum):
    """Available color themes"""
    CLASSIC = "classic"
    DARK = "dark"
    BRIGHT = "bright"

class DisplayMessage:
    """Message for interface display with improved word wrapping"""
    
    def __init__(self, content: str, msg_type: str, timestamp: str = None):
        self.content = content
        self.msg_type = msg_type
        self.timestamp = timestamp or time.strftime("%H:%M:%S")
        self.wrapped_lines = []
    
    def format_for_display(self, max_width: int = 80) -> List[str]:
        """Format message with word wrapping - returns actual display lines"""
        prefix = {
            'user': 'You',
            'assistant': 'AI',
            'system': 'System',
            'error': 'Error'
        }.get(self.msg_type, 'Unknown')
        
        header = f"[{self.timestamp}] {prefix}: "
        
        # Calculate available width for content
        content_width = max(20, max_width - len(header))
        
        # Handle empty content
        if not self.content.strip():
            return [header.rstrip()]
        
        # Wrap the content
        wrapped_content = textwrap.wrap(
            self.content, 
            width=content_width,
            break_long_words=True,
            break_on_hyphens=True,
            expand_tabs=True,
            replace_whitespace=True
        )
        
        if not wrapped_content:
            wrapped_content = [""]
        
        # Format lines
        lines = []
        for i, line in enumerate(wrapped_content):
            if i == 0:
                lines.append(header + line)
            else:
                # Indent continuation lines
                indent = " " * len(header)
                lines.append(indent + line)
        
        self.wrapped_lines = lines
        return lines

class ColorManager:
    """Color management with robust initialization and immediate application"""
    
    def __init__(self, theme: ColorTheme = ColorTheme.CLASSIC):
        self.theme = theme
        self.colors_available = False
        
        # Color pair definitions
        self.USER_COLOR = 1
        self.ASSISTANT_COLOR = 2
        self.SYSTEM_COLOR = 3
        self.ERROR_COLOR = 4
        self.BORDER_COLOR = 5
    
    def init_colors(self):
        """Initialize color pairs with immediate availability check"""
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

class InputValidator:
    """Input validation with detailed feedback"""
    
    def __init__(self, max_tokens: int = MAX_USER_INPUT_TOKENS):
        self.max_tokens = max_tokens
    
    def validate(self, text: str) -> Tuple[bool, str]:
        """Validate input text with specific error messages"""
        if not text.strip():
            return False, "Empty input"
        
        estimated_tokens = len(text) // 4
        if estimated_tokens > self.max_tokens:
            return False, f"Input too long: {estimated_tokens} tokens (max: {self.max_tokens})"
        
        return True, ""

class CursesInterface:
    """
    Rewritten ncurses interface with immediate display pattern and proper cursor management
    
    KEY REWRITE PRINCIPLES:
    1. Immediate window refresh after every change
    2. Explicit cursor positioning after every display operation
    3. True blank lines between messages (no system prefix)
    4. Phased initialization with forced display
    5. Synchronous updates (no batched refresh patterns)
    """
    
    def __init__(self, debug_logger=None, config=None):
        self.debug_logger = debug_logger
        self.config = config or {}
        
        # Core state
        self.running = True
        self.input_blocked = False
        self.mcp_processing = False
        
        # Screen components
        self.stdscr = None
        self.output_win = None
        self.input_win = None
        self.status_win = None
        
        # Screen dimensions
        self.screen_height = 0
        self.screen_width = 0
        self.output_win_height = 0
        self.input_win_height = 4
        self.status_win_height = 1
        
        # Input handling
        self.current_input = ""
        
        # NEW: Separate message storage and display line tracking
        self.display_messages: List[DisplayMessage] = []
        self.display_lines: List[Tuple[str, str]] = []  # (line_text, msg_type)
        self.scroll_offset = 0
        
        # Component managers
        theme = ColorTheme(self.config.get('color_theme', 'classic'))
        self.color_manager = ColorManager(theme)
        self.input_validator = InputValidator(MAX_USER_INPUT_TOKENS)
        
        # Module interconnects
        self.memory_manager = EnhancedMemoryManager(debug_logger=debug_logger)
        self.mcp_client = MCPClient(debug_logger=debug_logger)
        self.sme = StoryMomentumEngine(debug_logger=debug_logger)
        
        # PROMPT INTEGRATION - Load from config passed by main.py
        self.loaded_prompts = self.config.get('prompts', {})
        
        self._configure_components()
    
    def _configure_components(self):
        """Configure modules from config with prompt integration"""
        if not self.config:
            return
        
        # Configure MCP client
        mcp_config = self.config.get('mcp', {})
        if 'server_url' in mcp_config:
            self.mcp_client.server_url = mcp_config['server_url']
        if 'model' in mcp_config:
            self.mcp_client.model = mcp_config['model']
        if 'timeout' in mcp_config:
            self.mcp_client.timeout = mcp_config['timeout']
        
        # Set base system prompt from loaded critrules prompt
        if self.loaded_prompts.get('critrules'):
            self.mcp_client.system_prompt = self.loaded_prompts['critrules']
            self._log_debug("Base system prompt set from critrules")
    
    def _log_debug(self, message: str, category: str = "INTERFACE"):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, category)
    
    def _ensure_cursor_in_input(self):
        """
        CRITICAL: Guarantee cursor is in input window after any operation
        This is called after every display update to prevent cursor wandering
        """
        try:
            if not self.mcp_processing:
                prompt_len = len("Input> ")
                cursor_x = prompt_len + len(self.current_input)
                cursor_x = min(cursor_x, self.screen_width - 1)
                
                # Force cursor to input window
                self.input_win.move(0, cursor_x)
                self.input_win.refresh()
                
                # Ensure cursor is visible
                curses.curs_set(1)
            else:
                # During processing, position cursor at end of "Processing..."
                processing_len = len("Processing... ")
                cursor_x = min(processing_len, self.screen_width - 1)
                self.input_win.move(0, cursor_x)
                self.input_win.refresh()
        except curses.error:
            pass

# Chunk 2/4 - nci.py - Interface Initialization with Immediate Display

    def run(self) -> int:
        """Run interface using curses wrapper"""
        def _curses_main(stdscr):
            try:
                self._initialize_interface_rewrite(stdscr)
                self._run_main_loop()
                return 0
            except Exception as e:
                self._log_debug(f"Interface error: {e}")
                raise
        
        try:
            return curses.wrapper(_curses_main)
        except Exception as e:
            self._log_debug(f"Curses wrapper error: {e}")
            print(f"Interface error: {e}")
            return 1
    
    def _initialize_interface_rewrite(self, stdscr):
        """
        REWRITTEN: Initialize ncurses interface with immediate display and phased setup
        
        PHASES:
        1. Basic ncurses setup and screen clear
        2. Color initialization and dimension calculation  
        3. Window creation with immediate empty display
        4. Content population with immediate refresh
        5. Final cursor positioning
        """
        self.stdscr = stdscr
        
        # PHASE 1: Basic ncurses setup with immediate screen clear
        curses.curs_set(1)  # Show cursor
        curses.noecho()     # Don't echo input automatically
        curses.cbreak()     # React to keys immediately
        stdscr.nodelay(0)   # Block on getch() for input
        stdscr.clear()      # Clear any existing content
        stdscr.refresh()    # Force immediate screen clear
        
        self._log_debug("Phase 1: Basic setup complete")
        
        # PHASE 2: Colors and dimensions
        self.color_manager.init_colors()
        self.screen_height, self.screen_width = stdscr.getmaxyx()
        
        # Validate minimum screen size
        if self.screen_width < MIN_SCREEN_WIDTH or self.screen_height < MIN_SCREEN_HEIGHT:
            raise Exception(f"Terminal too small: {self.screen_width}x{self.screen_height} "
                          f"(minimum: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT})")
        
        self._calculate_window_dimensions()
        self._log_debug("Phase 2: Dimensions calculated")
        
        # PHASE 3: Create windows and display immediately
        self._create_windows_immediate()
        self._log_debug("Phase 3: Windows created with immediate display")
        
        # PHASE 4: Add welcome content with immediate refresh
        self._populate_initial_content()
        self._log_debug("Phase 4: Initial content populated")
        
        # PHASE 5: Test MCP connection and final cursor positioning
        self._test_mcp_connection()
        self._ensure_cursor_in_input()
        self._log_debug("Phase 5: Initialization complete")
        
        self._log_debug(f"Interface initialized: {self.screen_width}x{self.screen_height}")
    
    def _calculate_window_dimensions(self):
        """Calculate window dimensions with better proportions"""
        # Reserve space for borders
        border_space = 3  # top, middle, bottom borders
        
        # Calculate output window height (majority of screen)
        available_height = self.screen_height - self.input_win_height - self.status_win_height - border_space
        self.output_win_height = max(8, available_height)
        
        # Adjust input height if screen is very small
        if self.output_win_height < 12 and self.input_win_height > 2:
            self.input_win_height = 2
            self.output_win_height = self.screen_height - self.input_win_height - self.status_win_height - border_space
    
    def _create_windows_immediate(self):
        """
        REWRITTEN: Create windows and display them immediately with borders
        Each window is created and immediately displayed to avoid blank screen issues
        """
        # Output window (conversation display)
        self.output_win = curses.newwin(
            self.output_win_height,
            self.screen_width,
            0,
            0
        )
        self.output_win.scrollok(True)
        self.output_win.idlok(True)
        
        # IMMEDIATE: Display empty output window
        self.output_win.clear()
        self.output_win.refresh()
        
        # Input window (user input)
        input_y = self.output_win_height + 1
        self.input_win = curses.newwin(
            self.input_win_height,
            self.screen_width,
            input_y,
            0
        )
        
        # IMMEDIATE: Display input window with prompt
        self.input_win.clear()
        self.input_win.addstr(0, 0, "Input> ")
        self.input_win.refresh()
        
        # Status window (bottom line)
        status_y = input_y + self.input_win_height + 1
        self.status_win = curses.newwin(
            self.status_win_height,
            self.screen_width,
            status_y,
            0
        )
        
        # IMMEDIATE: Display status window
        self.status_win.clear()
        self.status_win.addstr(0, 0, "Initializing...")
        self.status_win.refresh()
        
        # Draw borders immediately
        self._draw_borders_immediate()
        
        # Force cursor to input window
        self._ensure_cursor_in_input()
    
    def _draw_borders_immediate(self):
        """Draw borders with immediate refresh"""
        border_color = self.color_manager.get_color('border')
        
        # Apply border color if available
        if border_color and self.color_manager.colors_available:
            self.stdscr.attron(curses.color_pair(border_color))
        
        # Top border
        self.stdscr.hline(self.output_win_height, 0, curses.ACS_HLINE, self.screen_width)
        
        # Bottom border
        status_y = self.output_win_height + self.input_win_height + 1
        self.stdscr.hline(status_y, 0, curses.ACS_HLINE, self.screen_width)
        
        # Remove border color
        if border_color and self.color_manager.colors_available:
            self.stdscr.attroff(curses.color_pair(border_color))
        
        # IMMEDIATE: Refresh borders
        self.stdscr.refresh()
    
    def _populate_initial_content(self):
        """
        REWRITTEN: Add welcome messages with immediate display
        Each message is added and immediately displayed to ensure visibility
        """
        # Welcome message
        welcome_msg = DisplayMessage(
            "DevName RPG Client started. Type /help for commands.",
            "system"
        )
        self._add_message_immediate(welcome_msg)
        
        # Prompt status
        prompt_status = []
        if self.loaded_prompts.get('critrules'):
            prompt_status.append("GM Rules")
        if self.loaded_prompts.get('companion'):
            prompt_status.append("Companion")
        if self.loaded_prompts.get('lowrules'):
            prompt_status.append("Narrative")
        
        if prompt_status:
            status_msg = DisplayMessage(
                f"Active prompts: {', '.join(prompt_status)}",
                "system"
            )
            self._add_message_immediate(status_msg)
        else:
            status_msg = DisplayMessage(
                "Warning: No prompts loaded",
                "system"
            )
            self._add_message_immediate(status_msg)
        
        # Update status to show ready state
        self._update_status_immediate()
    
    def _add_message_immediate(self, message: DisplayMessage):
        """
        REWRITTEN: Add message with immediate display refresh
        This ensures each message appears as soon as it's added
        """
        # Add to message storage
        self.display_messages.append(message)
        
        # Generate wrapped lines
        wrapped_lines = message.format_for_display(self.screen_width - 2)
        
        # Add to display lines cache
        for line in wrapped_lines:
            self.display_lines.append((line, message.msg_type))
        
        # Auto-scroll to bottom
        self._auto_scroll_to_bottom()
        
        # IMMEDIATE: Update and refresh output display
        self._update_output_immediate()
    
    def _add_blank_line_immediate(self):
        """
        NEW: Add a true blank line (no timestamp, no prefix) with immediate display
        This fixes the empty "System:" line issue
        """
        # Add completely empty line to display
        self.display_lines.append(("", ""))
        
        # Auto-scroll and refresh
        self._auto_scroll_to_bottom()
        self._update_output_immediate()
    
    def _update_output_immediate(self):
        """
        REWRITTEN: Update output window with immediate refresh
        No batched updates - each call immediately refreshes the display
        """
        self.output_win.clear()
        
        # Calculate visible lines
        start_idx = self.scroll_offset
        end_idx = start_idx + (self.output_win_height - 1)
        visible_lines = self.display_lines[start_idx:end_idx]
        
        # Display lines with proper colors
        for i, (line_text, msg_type) in enumerate(visible_lines):
            if i >= self.output_win_height - 1:
                break
            
            # Handle empty lines (true blank lines)
            if not line_text and not msg_type:
                try:
                    self.output_win.addstr(i, 0, "")
                except curses.error:
                    pass
                continue
            
            color = self.color_manager.get_color(msg_type)
            display_text = line_text[:self.screen_width - 1] if len(line_text) >= self.screen_width else line_text
            
            try:
                if color and self.color_manager.colors_available:
                    self.output_win.attron(curses.color_pair(color))
                    self.output_win.addstr(i, 0, display_text)
                    self.output_win.attroff(curses.color_pair(color))
                else:
                    self.output_win.addstr(i, 0, display_text)
            except curses.error:
                pass
        
        # IMMEDIATE: Refresh output window
        self.output_win.refresh()
        
        # Ensure cursor returns to input
        self._ensure_cursor_in_input()
    
    def _update_status_immediate(self):
        """Update status with immediate refresh"""
        self.status_win.clear()
        
        status_parts = []
        
        try:
            mem_stats = self.memory_manager.get_memory_stats()
            msg_count = mem_stats.get('message_count', 0)
            status_parts.append(f"Messages: {msg_count}")
        except:
            status_parts.append("Messages: 0")
        
        try:
            sme_stats = self.sme.get_pressure_stats()
            if 'current_pressure' in sme_stats:
                pressure = sme_stats['current_pressure']
                status_parts.append(f"Pressure: {pressure:.2f}")
        except:
            pass
        
        if self.mcp_processing:
            status_parts.append("MCP: Processing")
        else:
            status_parts.append("MCP: Ready")
        
        prompt_count = len([p for p in self.loaded_prompts.values() if p.strip()])
        status_parts.append(f"Prompts: {prompt_count}")
        
        if self.scroll_offset > 0:
            max_scroll = max(0, len(self.display_lines) - (self.output_win_height - 1))
            scroll_percent = int((self.scroll_offset / max_scroll) * 100) if max_scroll > 0 else 0
            status_parts.append(f"Scroll: {scroll_percent}%")
        
        status_text = " | ".join(status_parts)
        
        if len(status_text) > self.screen_width - 1:
            status_text = status_text[:self.screen_width - 4] + "..."
        
        try:
            status_color = self.color_manager.get_color('system')
            if status_color and self.color_manager.colors_available:
                self.status_win.attron(curses.color_pair(status_color))
                self.status_win.addstr(0, 0, status_text)
                self.status_win.attroff(curses.color_pair(status_color))
            else:
                self.status_win.addstr(0, 0, status_text)
        except curses.error:
            pass
        
        # IMMEDIATE: Refresh status
        self.status_win.refresh()
        
        # Ensure cursor returns to input
        self._ensure_cursor_in_input()
    
    def _test_mcp_connection(self):
        """Test MCP server connection with immediate display"""
        try:
            if self.mcp_client.test_connection():
                connection_msg = DisplayMessage("MCP server connected", "system")
                self._add_message_immediate(connection_msg)
            else:
                connection_msg = DisplayMessage("MCP server not available", "system")
                self._add_message_immediate(connection_msg)
        except Exception as e:
            error_msg = DisplayMessage(f"MCP test failed: {e}", "system")
            self._add_message_immediate(error_msg)
    
    def _auto_scroll_to_bottom(self):
        """Auto-scroll to show latest messages"""
        max_scroll = max(0, len(self.display_lines) - (self.output_win_height - 1))
        self.scroll_offset = max_scroll

# Chunk 3/4 - nci.py - Input Processing and MCP Integration with Immediate Display

    def _run_main_loop(self):
        """
        REWRITTEN: Main input processing loop with immediate updates
        Each input change triggers immediate display update with proper cursor management
        """
        while self.running:
            try:
                # Get user input
                key = self.stdscr.getch()
                
                # Only process input if not blocked
                if not self.input_blocked:
                    input_changed = self._handle_key_input(key)
                    
                    # IMMEDIATE: Update input display if input changed
                    if input_changed:
                        self._update_input_immediate()
                
                # Periodic status update (but don't steal cursor)
                self._update_status_immediate()
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self._log_debug(f"Main loop error: {e}")
    
    def _update_input_immediate(self):
        """
        REWRITTEN: Update input window with immediate refresh and proper cursor positioning
        """
        self.input_win.clear()
        
        if self.mcp_processing:
            prompt = "Processing... "
            prompt_color = self.color_manager.get_color('system')
        else:
            prompt = "Input> "
            prompt_color = self.color_manager.get_color('user')
        
        display_input = prompt + self.current_input
        max_display_length = self.screen_width - 1
        
        # Handle long input with scrolling display
        if len(display_input) > max_display_length:
            display_input = "..." + display_input[-(max_display_length-3):]
        
        try:
            if prompt_color and self.color_manager.colors_available:
                self.input_win.attron(curses.color_pair(prompt_color))
                self.input_win.addstr(0, 0, prompt)
                self.input_win.attroff(curses.color_pair(prompt_color))
                if len(prompt) < max_display_length:
                    remaining_input = display_input[len(prompt):max_display_length]
                    self.input_win.addstr(0, len(prompt), remaining_input)
            else:
                self.input_win.addstr(0, 0, display_input)
        except curses.error:
            pass
        
        # IMMEDIATE: Refresh input window
        self.input_win.refresh()
        
        # CRITICAL: Ensure cursor is positioned correctly
        self._ensure_cursor_in_input()
    
    def _handle_key_input(self, key: int) -> bool:
        """Handle keyboard input, return True if input changed"""
        try:
            if key == ord('\n') or key == curses.KEY_ENTER or key == 10:
                self._handle_enter_key()
                return True
            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                return self._handle_backspace_key()
            elif key == curses.KEY_UP:
                self._handle_scroll_up()
                return False
            elif key == curses.KEY_DOWN:
                self._handle_scroll_down()
                return False
            elif 32 <= key <= 126:  # Printable ASCII
                return self._handle_character_input(chr(key))
        except Exception as e:
            self._log_debug(f"Key handling error: {e}")
        
        return False
    
    def _handle_enter_key(self):
        """
        REWRITTEN: Process Enter key with immediate user message display and blank line separation
        """
        if not self.current_input.strip():
            return
        
        # Validate input
        is_valid, error_msg = self.input_validator.validate(self.current_input)
        if not is_valid:
            self.add_error_message_immediate(error_msg)
            return
        
        user_input = self.current_input.strip()
        
        # IMMEDIATE: Display user message
        self.add_user_message_immediate(user_input)
        
        # Clear input and set processing state
        self.current_input = ""
        self.set_processing_state_immediate(True)
        
        # Process input
        self._process_user_input(user_input)
    
    def _handle_backspace_key(self) -> bool:
        """Handle backspace, return True if input changed"""
        if self.current_input:
            self.current_input = self.current_input[:-1]
            return True
        return False
    
    def _handle_scroll_up(self):
        """Scroll chat history up with immediate display"""
        if self.scroll_offset > 0:
            self.scroll_offset -= 1
            self._update_output_immediate()
    
    def _handle_scroll_down(self):
        """Scroll chat history down with immediate display"""
        max_scroll = max(0, len(self.display_lines) - (self.output_win_height - 1))
        if self.scroll_offset < max_scroll:
            self.scroll_offset += 1
            self._update_output_immediate()
    
    def _handle_character_input(self, char: str) -> bool:
        """Handle printable character, return True if input changed"""
        if len(self.current_input) < 1000:
            self.current_input += char
            return True
        return False
    
    def add_user_message_immediate(self, content: str):
        """
        REWRITTEN: Add user message with immediate display
        """
        msg = DisplayMessage(content, "user")
        self._add_message_immediate(msg)
    
    def add_assistant_message_immediate(self, content: str):
        """
        REWRITTEN: Add assistant message with proper blank line separation
        FIXES the empty "System:" line issue by adding true blank line
        """
        # Add TRUE blank line before AI response (no timestamp, no prefix)
        self._add_blank_line_immediate()
        
        # Add the AI response
        msg = DisplayMessage(content, "assistant")
        self._add_message_immediate(msg)
    
    def add_system_message_immediate(self, content: str):
        """Add system message with immediate display"""
        msg = DisplayMessage(content, "system")
        self._add_message_immediate(msg)
    
    def add_error_message_immediate(self, content: str):
        """Add error message with immediate display"""
        msg = DisplayMessage(content, "error")
        self._add_message_immediate(msg)
        self._log_debug(f"Error displayed: {content}")
    
    def set_processing_state_immediate(self, processing: bool):
        """
        REWRITTEN: Set processing state with immediate visual feedback
        """
        self.mcp_processing = processing
        self.input_blocked = processing
        
        # IMMEDIATE: Update input display to show processing state
        self._update_input_immediate()
        
        self._log_debug(f"Processing state: {processing}")
    
    def _process_user_input(self, user_input: str):
        """Process user input - commands or MCP requests"""
        try:
            if user_input.startswith('/'):
                self._process_command(user_input)
                self.set_processing_state_immediate(False)
                return
            
            # Store in memory and update story momentum
            self.memory_manager.add_message(user_input, MessageType.USER)
            self.sme.process_user_input(user_input)
            
            # Send to MCP server
            self._send_mcp_request(user_input)
            
        except Exception as e:
            self.add_error_message_immediate(f"Processing failed: {e}")
            self.set_processing_state_immediate(False)
    
    def _process_command(self, command: str):
        """Process slash commands with immediate display"""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self._show_help()
        elif cmd == '/quit' or cmd == '/exit':
            self.running = False
        elif cmd == '/clear':
            self._clear_display()
        elif cmd == '/stats':
            self._show_stats()
        elif cmd.startswith('/theme '):
            theme_name = cmd[7:].strip()
            self._change_theme(theme_name)
        elif cmd == '/prompts':
            self._show_prompt_status()
        else:
            self.add_error_message_immediate(f"Unknown command: {command}")
    
    def _build_system_messages(self, story_context: str) -> List[Dict[str, str]]:
        """Build system messages with integrated prompts and story context"""
        system_messages = []
        
        if self.loaded_prompts.get('critrules'):
            primary_prompt = self.loaded_prompts['critrules']
            if story_context:
                primary_prompt += f"\n\n**STORY CONTEXT**: {story_context}"
            system_messages.append({"role": "system", "content": primary_prompt})
        
        if self.loaded_prompts.get('companion'):
            system_messages.append({"role": "system", "content": self.loaded_prompts['companion']})
        
        if self.loaded_prompts.get('lowrules'):
            system_messages.append({"role": "system", "content": self.loaded_prompts['lowrules']})
        
        return system_messages
    
    def _send_mcp_request(self, user_input: str):
        """
        REWRITTEN: Send request to MCP server with comprehensive error handling and immediate display
        """
        try:
            # Get context data
            conversation_history = self.memory_manager.get_conversation_for_mcp()
            story_context = self.sme.get_story_context()
            context_str = self._format_story_context(story_context)
            system_messages = self._build_system_messages(context_str)
            
            # Build complete message chain
            all_messages = system_messages + conversation_history + [{"role": "user", "content": user_input}]
            
            try:
                # Try custom MCP request first
                response_data = self.mcp_client._execute_request({
                    "model": self.mcp_client.model,
                    "messages": all_messages,
                    "stream": False
                })
                
                # Store and display response
                self.memory_manager.add_message(response_data, MessageType.ASSISTANT)
                self.add_assistant_message_immediate(response_data)
                
            except Exception as mcp_error:
                self._log_debug(f"Custom MCP call failed, using fallback: {mcp_error}")
                try:
                    # Fallback to standard send_message
                    response = self.mcp_client.send_message(
                        user_input,
                        conversation_history=conversation_history,
                        story_context=context_str
                    )
                    
                    self.memory_manager.add_message(response, MessageType.ASSISTANT)
                    self.add_assistant_message_immediate(response)
                    
                except Exception as fallback_error:
                    self.add_error_message_immediate(f"MCP server connection failed: {fallback_error}")
            
        except Exception as e:
            self.add_error_message_immediate(f"Request processing failed: {e}")
        finally:
            # Always reset processing state
            self.set_processing_state_immediate(False)
    
    def _format_story_context(self, context: Dict[str, Any]) -> str:
        """Format story context for integration"""
        if not context:
            return ""
        
        parts = []
        pressure = context.get('pressure_level', 0.0)
        arc = context.get('story_arc', 'unknown')
        state = context.get('narrative_state', 'unknown')
        
        parts.append(f"Pressure: {pressure:.2f}, Arc: {arc}, State: {state}")
        
        if context.get('antagonist_present'):
            antagonist = context.get('antagonist', {})
            name = antagonist.get('name', 'Unknown')
            threat = antagonist.get('threat_level', 0.0)
            parts.append(f"Antagonist: {name} (threat: {threat:.2f})")
        
        return " | ".join(parts)
    
    # BACKWARD COMPATIBILITY: Keep old method names that call immediate versions
    def add_user_message(self, content: str):
        """Backward compatibility wrapper"""
        self.add_user_message_immediate(content)
    
    def add_assistant_message(self, content: str):
        """Backward compatibility wrapper"""
        self.add_assistant_message_immediate(content)
    
    def add_system_message(self, content: str):
        """Backward compatibility wrapper"""
        self.add_system_message_immediate(content)
    
    def add_error_message(self, content: str):
        """Backward compatibility wrapper"""
        self.add_error_message_immediate(content)
    
    def set_input_blocked(self, blocked: bool):
        """Backward compatibility wrapper"""
        self.set_processing_state_immediate(blocked)

# Chunk 4/4 - nci.py - Utility Methods and Command Processing

    def _clear_display(self):
        """
        REWRITTEN: Clear message display with immediate refresh
        """
        self.display_messages.clear()
        self.display_lines.clear()
        self.scroll_offset = 0
        
        # IMMEDIATE: Clear and refresh output window
        self.output_win.clear()
        self.output_win.refresh()
        
        # Add clear confirmation message
        self.add_system_message_immediate("Display cleared")
    
    def _show_help(self):
        """Show help information with immediate display"""
        help_messages = [
            "Available commands:",
            "/help - Show this help",
            "/quit, /exit - Exit application", 
            "/clear - Clear message display",
            "/stats - Show system statistics",
            "/prompts - Show prompt file status",
            "/theme <name> - Change color theme (classic, dark, bright)",
            "",
            "Use arrow keys to scroll through chat history"
        ]
        
        for msg in help_messages:
            self.add_system_message_immediate(msg)
    
    def _show_stats(self):
        """Show system statistics with immediate display"""
        try:
            mem_stats = self.memory_manager.get_memory_stats()
            self.add_system_message_immediate(f"Memory: {mem_stats.get('message_count', 0)} messages, "
                                           f"{mem_stats.get('total_tokens', 0)} tokens")
        except:
            self.add_system_message_immediate("Memory: Stats unavailable")
        
        try:
            sme_stats = self.sme.get_pressure_stats()
            if 'current_pressure' in sme_stats:
                pressure = sme_stats['current_pressure']
                arc = sme_stats.get('current_arc', 'unknown')
                self.add_system_message_immediate(f"Story: Pressure {pressure:.2f}, Arc {arc}")
        except:
            self.add_system_message_immediate("Story: Stats unavailable")
        
        mcp_info = self.mcp_client.get_server_info()
        self.add_system_message_immediate(f"MCP: {mcp_info.get('server_url', 'unknown')}")
        
        self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, "
                                       f"Scroll offset: {self.scroll_offset}")
        
        # Display screen dimensions
        self.add_system_message_immediate(f"Screen: {self.screen_width}x{self.screen_height}")
    
    def _show_prompt_status(self):
        """Show prompt file status and details with immediate display"""
        self.add_system_message_immediate("Prompt Status:")
        
        prompt_names = {
            'critrules': 'GM Rules (Critical)',
            'companion': 'Companion Character', 
            'lowrules': 'Narrative Guidelines'
        }
        
        for prompt_type, display_name in prompt_names.items():
            content = self.loaded_prompts.get(prompt_type, '')
            if content.strip():
                token_count = len(content) // 4
                self.add_system_message_immediate(f"  {display_name}: Loaded ({token_count:,} tokens)")
            else:
                self.add_system_message_immediate(f"  {display_name}: Missing")
        
        total_tokens = sum(len(content) // 4 for content in self.loaded_prompts.values() if content.strip())
        self.add_system_message_immediate(f"Total prompt tokens: {total_tokens:,}")
    
    def _change_theme(self, theme_name: str):
        """
        REWRITTEN: Change color theme with immediate display refresh
        """
        try:
            theme = ColorTheme(theme_name)
            self.color_manager.theme = theme
            success = self.color_manager.init_colors()
            
            if success:
                self.add_system_message_immediate(f"Theme changed to: {theme_name}")
                
                # IMMEDIATE: Refresh all windows with new colors
                self._refresh_all_immediate()
            else:
                self.add_error_message_immediate("Failed to initialize new theme colors")
        except ValueError:
            self.add_error_message_immediate(f"Unknown theme: {theme_name}. Available: classic, dark, bright")
    
    def _refresh_all_immediate(self):
        """
        REWRITTEN: Refresh all windows immediately (replaces batched refresh pattern)
        Each window is updated and refreshed individually for immediate display
        """
        try:
            # Update and refresh each window individually
            self._update_output_immediate()
            self._update_input_immediate() 
            self._update_status_immediate()
            
            # Redraw borders with new colors
            self._draw_borders_immediate()
            
        except curses.error as e:
            self._log_debug(f"Display refresh error: {e}")
    
    def shutdown(self):
        """
        REWRITTEN: Graceful shutdown with immediate feedback
        """
        self.running = False
        
        # Show shutdown message if interface is still active
        if self.stdscr:
            try:
                self.add_system_message_immediate("Shutting down...")
            except:
                pass
        
        # Auto-save conversation if configured
        if self.config.get('auto_save_conversation', False):
            try:
                filename = f"chat_history_{int(time.time())}.json"
                if self.memory_manager.save_conversation(filename):
                    self._log_debug(f"Conversation saved to {filename}")
                    if self.stdscr:
                        try:
                            self.add_system_message_immediate(f"Conversation saved to {filename}")
                            time.sleep(1)  # Brief pause to show message
                        except:
                            pass
            except Exception as e:
                self._log_debug(f"Failed to save conversation: {e}")
        
        self._log_debug("Interface shutdown complete")
    
    # UTILITY METHODS FOR TESTING AND DEBUGGING
    
    def get_display_state(self) -> Dict[str, Any]:
        """Get current display state for debugging"""
        return {
            "message_count": len(self.display_messages),
            "display_lines": len(self.display_lines),
            "scroll_offset": self.scroll_offset,
            "screen_size": (self.screen_width, self.screen_height),
            "window_heights": {
                "output": self.output_win_height,
                "input": self.input_win_height,
                "status": self.status_win_height
            },
            "processing_state": self.mcp_processing,
            "input_blocked": self.input_blocked,
            "current_input_length": len(self.current_input)
        }
    
    def force_refresh_display(self):
        """Force complete display refresh - useful for recovery"""
        try:
            self.stdscr.clear()
            self.stdscr.refresh()
            
            self._draw_borders_immediate()
            self._refresh_all_immediate()
            
            self._log_debug("Forced complete display refresh")
        except Exception as e:
            self._log_debug(f"Force refresh failed: {e}")
    
    def add_debug_message(self, content: str):
        """Add debug message (only if debug logger is active)"""
        if self.debug_logger:
            self.add_system_message_immediate(f"DEBUG: {content}")
    
    def simulate_user_input(self, text: str):
        """Simulate user input for testing purposes"""
        if not self.input_blocked:
            self.current_input = text
            self._update_input_immediate()
            self._handle_enter_key()

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Ncurses Interface Module - REWRITTEN")
    print("Testing rewritten interface components...")
    
    # Test color manager
    color_mgr = ColorManager()
    print(f"Color themes available: {[theme.value for theme in ColorTheme]}")
    
    # Test input validator
    validator = InputValidator()
    test_input = "Hello, this is a test input!"
    is_valid, msg = validator.validate(test_input)
    print(f"Input validation test: {is_valid} - {msg}")
    
    # Test display message with wrapping
    display_msg = DisplayMessage("This is a very long test message that should wrap properly across multiple lines when displayed in the terminal interface, demonstrating the improved word wrapping functionality.", "user")
    wrapped_lines = display_msg.format_for_display(60)
    print(f"Message wrapping test: {len(wrapped_lines)} lines")
    for i, line in enumerate(wrapped_lines):
        print(f"  Line {i+1}: '{line}'")
    
    # Test empty message formatting (for blank lines)
    empty_msg = DisplayMessage("", "system")
    empty_lines = empty_msg.format_for_display(60)
    print(f"Empty message test: {len(empty_lines)} lines")
    for i, line in enumerate(empty_lines):
        print(f"  Empty line {i+1}: '{line}'")
    
    print("\nREWRITE IMPROVEMENTS:")
    print("- Immediate window refresh pattern implemented")
    print("- Explicit cursor management after every operation")
    print("- True blank lines between messages (no system prefix)")
    print("- Phased initialization with forced immediate display")
    print("- Enhanced error handling with immediate feedback")
    print("- Improved message wrapping with textwrap module")
    print("- Comprehensive debugging and testing utilities")
    
    print("\nInterface module rewrite test completed successfully.")
    print("Run main.py to start the full rewritten application.")

# End of nci.py - DevName RPG Client Ncurses Interface Module - REWRITTEN
# 
# REWRITE SUMMARY:
# ================
# 1. IMMEDIATE DISPLAY PATTERN: All display updates refresh windows immediately
# 2. CURSOR MANAGEMENT: _ensure_cursor_in_input() called after every operation  
# 3. BLANK LINE FIX: True empty lines added via _add_blank_line_immediate()
# 4. PHASED INITIALIZATION: 5-phase startup with immediate display at each step
# 5. ERROR RECOVERY: Comprehensive error handling with immediate user feedback
# 6. TESTING UTILITIES: Debug methods for state inspection and forced refresh
# 7. BACKWARD COMPATIBILITY: Old method names preserved as wrappers
#
# This rewrite addresses all issues identified in NCurses Rewrite Analysis.md:
# - Display update timing problems -> Immediate refresh pattern
# - Cursor management conflicts -> Explicit positioning after every update
# - Window refresh sequence issues -> Individual window refresh instead of batched
# - Memory state vs display state mismatch -> Immediate display of internal changes
# - Blank screen during initialization -> Phased initialization with forced display
# - Empty "System:" lines -> True blank lines with no prefixes
