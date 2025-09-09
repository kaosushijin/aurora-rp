# Chunk 1/4 - nci.py - Core Classes and Dependencies
#!/usr/bin/env python3
"""
DevName RPG Client - Ncurses Interface Module (nci.py)

Module architecture and interconnects documented in genai.txt
Maintains programmatic interfaces with mcp.py, emm.py, and sme.py
"""

import curses
import time
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
    """Message for interface display"""
    
    def __init__(self, content: str, msg_type: str, timestamp: str = None):
        self.content = content
        self.msg_type = msg_type
        self.timestamp = timestamp or time.strftime("%H:%M:%S")
    
    def format_for_display(self) -> str:
        prefix = {
            'user': 'You',
            'assistant': 'AI',
            'system': 'System',
            'error': 'Error'
        }.get(self.msg_type, 'Unknown')
        
        return f"[{self.timestamp}] {prefix}: {self.content}"

class ColorManager:
    """Simplified color management"""
    
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
        """Initialize color pairs"""
        if not curses.has_colors():
            return
        
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
            
        except curses.error:
            self.colors_available = False
    
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
    """Simple input validation"""
    
    def __init__(self, max_tokens: int = MAX_USER_INPUT_TOKENS):
        self.max_tokens = max_tokens
    
    def validate(self, text: str) -> Tuple[bool, str]:
        """Validate input text"""
        if not text.strip():
            return False, "Empty input"
        
        estimated_tokens = len(text) // 4
        if estimated_tokens > self.max_tokens:
            return False, f"Input too long: {estimated_tokens} tokens (max: {self.max_tokens})"
        
        return True, ""

class CursesInterface:
    """Ncurses interface for RPG client"""
    
    def __init__(self, debug_logger=None, config=None):
        self.debug_logger = debug_logger
        self.config = config or {}
        
        # Core state
        self.running = True
        self.input_blocked = False
        
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
        
        # Message display
        self.display_messages: List[DisplayMessage] = []
        self.scroll_offset = 0
        
        # Component managers
        theme = ColorTheme(self.config.get('color_theme', 'classic'))
        self.color_manager = ColorManager(theme)
        self.input_validator = InputValidator(MAX_USER_INPUT_TOKENS)
        
        # Module interconnects
        self.memory_manager = EnhancedMemoryManager(debug_logger=debug_logger)
        self.mcp_client = MCPClient(debug_logger=debug_logger)
        self.sme = StoryMomentumEngine(debug_logger=debug_logger)
        
        self._configure_components()
    
    def _configure_components(self):
        """Configure modules from config"""
        if not self.config:
            return
        
        # Configure MCP client
        mcp_config = self.config.get('mcp', {})
        if 'server_url' in mcp_config:
            self.mcp_client.server_url = mcp_config['server_url']
        if 'model' in mcp_config:
            self.mcp_client.model = mcp_config['model']
    
    def _log_debug(self, message: str, category: str = "INTERFACE"):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, category)

# Chunk 2/4 - nci.py - Interface Initialization and Window Management

    def run(self) -> int:
        """Run interface using curses wrapper"""
        def _curses_main(stdscr):
            try:
                self._initialize_interface(stdscr)
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
    
    def _initialize_interface(self, stdscr):
        """Initialize ncurses interface"""
        self.stdscr = stdscr
        
        # Basic curses setup
        curses.curs_set(1)  # Show cursor
        
        # Initialize colors
        self.color_manager.init_colors()
        
        # Get screen dimensions
        self.screen_height, self.screen_width = stdscr.getmaxyx()
        
        # Validate minimum screen size
        if self.screen_width < MIN_SCREEN_WIDTH or self.screen_height < MIN_SCREEN_HEIGHT:
            raise Exception(f"Terminal too small: {self.screen_width}x{self.screen_height} "
                          f"(minimum: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT})")
        
        # Calculate window dimensions
        self._calculate_window_dimensions()
        
        # Create windows
        self._create_windows()
        
        # Show initial content
        self._show_welcome_message()
        
        # Test MCP connection
        self._test_mcp_connection()
        
        # Initial refresh
        self._refresh_all_windows()
        
        self._log_debug(f"Interface initialized: {self.screen_width}x{self.screen_height}")
    
    def _calculate_window_dimensions(self):
        """Calculate window dimensions"""
        # Reserve space for borders
        border_space = 3  # top, middle, bottom borders
        
        # Calculate output window height (majority of screen)
        available_height = self.screen_height - self.input_win_height - self.status_win_height - border_space
        self.output_win_height = max(5, available_height)
        
        # Adjust input height if needed
        if self.output_win_height < 10 and self.input_win_height > 2:
            self.input_win_height = 2
            self.output_win_height = self.screen_height - self.input_win_height - self.status_win_height - border_space
    
    def _create_windows(self):
        """Create ncurses windows"""
        # Output window (conversation display)
        self.output_win = curses.newwin(
            self.output_win_height,
            self.screen_width,
            0,
            0
        )
        
        # Input window (user input)
        input_y = self.output_win_height + 1
        self.input_win = curses.newwin(
            self.input_win_height,
            self.screen_width,
            input_y,
            0
        )
        
        # Status window (bottom line)
        status_y = input_y + self.input_win_height + 1
        self.status_win = curses.newwin(
            self.status_win_height,
            self.screen_width,
            status_y,
            0
        )
        
        # Draw borders
        self._draw_borders()
    
    def _draw_borders(self):
        """Draw window borders"""
        border_color = self.color_manager.get_color('border')
        
        # Top border
        self.stdscr.hline(self.output_win_height, 0, curses.ACS_HLINE, self.screen_width)
        
        # Bottom border
        status_y = self.output_win_height + self.input_win_height + 1
        self.stdscr.hline(status_y, 0, curses.ACS_HLINE, self.screen_width)
        
        if self.color_manager.colors_available:
            self.stdscr.attron(curses.color_pair(border_color))
            self.stdscr.attroff(curses.color_pair(border_color))
    
    def _show_welcome_message(self):
        """Show initial welcome message"""
        welcome_msg = DisplayMessage(
            "DevName RPG Client started. Type /help for commands.",
            "system"
        )
        self.display_messages.append(welcome_msg)
    
    def _test_mcp_connection(self):
        """Test MCP server connection"""
        try:
            if self.mcp_client.test_connection():
                self.add_system_message("MCP server connected")
            else:
                self.add_system_message("MCP server not available")
        except Exception as e:
            self.add_system_message(f"MCP test failed: {e}")
    
    def _refresh_all_windows(self):
        """Refresh all windows"""
        self._update_output_display()
        self._update_input_display()
        self._update_status_display()
        
        self.output_win.refresh()
        self.input_win.refresh()
        self.status_win.refresh()
        self.stdscr.refresh()

# Chunk 3/4 - nci.py - Input Processing and Display Updates

    def _run_main_loop(self):
        """Main input processing loop"""
        while self.running:
            try:
                # Get user input
                key = self.stdscr.getch()
                
                if not self.input_blocked:
                    self._handle_key_input(key)
                
                # Update displays
                self._refresh_all_windows()
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self._log_debug(f"Main loop error: {e}")
    
    def _handle_key_input(self, key: int):
        """Handle keyboard input"""
        try:
            if key == ord('\n') or key == curses.KEY_ENTER or key == 10:
                self._handle_enter_key()
            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                self._handle_backspace_key()
            elif key == curses.KEY_UP:
                self._handle_scroll_up()
            elif key == curses.KEY_DOWN:
                self._handle_scroll_down()
            elif 32 <= key <= 126:  # Printable ASCII
                self._handle_character_input(chr(key))
        except Exception as e:
            self._log_debug(f"Key handling error: {e}")
    
    def _handle_enter_key(self):
        """Process Enter key - submit input"""
        if not self.current_input.strip():
            return
        
        # Validate input
        is_valid, error_msg = self.input_validator.validate(self.current_input)
        if not is_valid:
            self.add_error_message(error_msg)
            return
        
        user_input = self.current_input.strip()
        self.current_input = ""
        
        # Add user message to display
        self.add_user_message(user_input)
        
        # Block input during processing
        self.set_input_blocked(True)
        
        # Process input
        self._process_user_input(user_input)
    
    def _handle_backspace_key(self):
        """Handle backspace"""
        if self.current_input:
            self.current_input = self.current_input[:-1]
    
    def _handle_scroll_up(self):
        """Scroll chat history up"""
        if self.scroll_offset > 0:
            self.scroll_offset -= 1
    
    def _handle_scroll_down(self):
        """Scroll chat history down"""
        max_scroll = max(0, len(self.display_messages) - (self.output_win_height - 1))
        if self.scroll_offset < max_scroll:
            self.scroll_offset += 1
    
    def _handle_character_input(self, char: str):
        """Handle printable character"""
        self.current_input += char
    
    def _process_user_input(self, user_input: str):
        """Process user input - commands or MCP requests"""
        try:
            # Check for commands
            if user_input.startswith('/'):
                self._process_command(user_input)
                self.set_input_blocked(False)
                return
            
            # Store message in memory
            self.memory_manager.add_message(user_input, MessageType.USER)
            
            # Update story momentum
            self.sme.process_user_input(user_input)
            
            # Send to MCP server
            self._send_mcp_request(user_input)
            
        except Exception as e:
            self.add_error_message(f"Processing failed: {e}")
            self.set_input_blocked(False)
    
    def _process_command(self, command: str):
        """Process slash commands"""
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
        else:
            self.add_error_message(f"Unknown command: {command}")
    
    def _send_mcp_request(self, user_input: str):
        """Send request to MCP server"""
        try:
            # Get conversation history
            conversation_history = self.memory_manager.get_conversation_for_mcp()
            
            # Get story context
            story_context = self.sme.get_story_context()
            context_str = self._format_story_context(story_context)
            
            # Send to MCP
            response = self.mcp_client.send_message(
                user_input,
                conversation_history=conversation_history,
                story_context=context_str
            )
            
            # Store and display response
            self.memory_manager.add_message(response, MessageType.ASSISTANT)
            self.add_assistant_message(response)
            
        except Exception as e:
            self.add_error_message(f"MCP request failed: {e}")
        finally:
            self.set_input_blocked(False)
    
    def _format_story_context(self, context: Dict[str, Any]) -> str:
        """Format story context for MCP"""
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
    
    def _update_output_display(self):
        """Update conversation output display"""
        self.output_win.clear()
        
        # Calculate visible messages based on scroll
        start_idx = self.scroll_offset
        end_idx = start_idx + (self.output_win_height - 1)
        visible_messages = self.display_messages[start_idx:end_idx]
        
        # Display messages
        for i, msg in enumerate(visible_messages):
            if i >= self.output_win_height - 1:
                break
            
            color = self.color_manager.get_color(msg.msg_type)
            display_text = msg.format_for_display()
            
            # Truncate if too long
            if len(display_text) > self.screen_width - 1:
                display_text = display_text[:self.screen_width - 4] + "..."
            
            try:
                if color and self.color_manager.colors_available:
                    self.output_win.attron(curses.color_pair(color))
                    self.output_win.addstr(i, 0, display_text)
                    self.output_win.attroff(curses.color_pair(color))
                else:
                    self.output_win.addstr(i, 0, display_text)
            except curses.error:
                pass  # Ignore display errors
    
    def _update_input_display(self):
        """Update input window display"""
        self.input_win.clear()
        
        # Input prompt
        prompt = "Input> " if not self.input_blocked else "Processing... "
        
        # Display current input
        display_input = prompt + self.current_input
        
        # Truncate if needed
        if len(display_input) > self.screen_width - 1:
            display_input = display_input[:self.screen_width - 4] + "..."
        
        try:
            self.input_win.addstr(0, 0, display_input)
            
            # Position cursor
            if not self.input_blocked:
                cursor_pos = min(len(display_input), self.screen_width - 1)
                self.input_win.move(0, cursor_pos)
        except curses.error:
            pass
    
    def _update_status_display(self):
        """Update status line"""
        self.status_win.clear()
        
        # Build status components
        status_parts = []
        
        # Memory stats
        mem_stats = self.memory_manager.get_memory_stats()
        msg_count = mem_stats.get('message_count', 0)
        status_parts.append(f"Messages: {msg_count}")
        
        # Story pressure
        sme_stats = self.sme.get_pressure_stats()
        if 'current_pressure' in sme_stats:
            pressure = sme_stats['current_pressure']
            status_parts.append(f"Pressure: {pressure:.2f}")
        
        # MCP status
        mcp_info = self.mcp_client.get_server_info()
        if mcp_info.get('connected'):
            status_parts.append("MCP: Connected")
        else:
            status_parts.append("MCP: Offline")
        
        status_text = " | ".join(status_parts)
        
        # Truncate if needed
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

# Chunk 4/4 - nci.py - Message Management and Utility Methods

    def add_user_message(self, content: str):
        """Add user message to display"""
        msg = DisplayMessage(content, "user")
        self.display_messages.append(msg)
        self._auto_scroll_to_bottom()
    
    def add_assistant_message(self, content: str):
        """Add assistant message to display"""
        msg = DisplayMessage(content, "assistant")
        self.display_messages.append(msg)
        self._auto_scroll_to_bottom()
    
    def add_system_message(self, content: str):
        """Add system message to display"""
        msg = DisplayMessage(content, "system")
        self.display_messages.append(msg)
        self._auto_scroll_to_bottom()
    
    def add_error_message(self, content: str):
        """Add error message to display"""
        msg = DisplayMessage(content, "error")
        self.display_messages.append(msg)
        self._auto_scroll_to_bottom()
        self._log_debug(f"Error displayed: {content}")
    
    def set_input_blocked(self, blocked: bool):
        """Set input blocking state"""
        self.input_blocked = blocked
        if blocked:
            self.add_system_message("Processing...")
        self._log_debug(f"Input blocked: {blocked}")
    
    def _auto_scroll_to_bottom(self):
        """Auto-scroll to show latest messages"""
        max_scroll = max(0, len(self.display_messages) - (self.output_win_height - 1))
        self.scroll_offset = max_scroll
    
    def _clear_display(self):
        """Clear message display"""
        self.display_messages.clear()
        self.scroll_offset = 0
        self.add_system_message("Display cleared")
    
    def _show_help(self):
        """Show help information"""
        help_messages = [
            "Available commands:",
            "/help - Show this help",
            "/quit, /exit - Exit application",
            "/clear - Clear message display",
            "/stats - Show system statistics",
            "/theme <name> - Change color theme (classic, dark, bright)",
            "",
            "Use arrow keys to scroll through chat history"
        ]
        
        for msg in help_messages:
            self.add_system_message(msg)
    
    def _show_stats(self):
        """Show system statistics"""
        # Memory stats
        mem_stats = self.memory_manager.get_memory_stats()
        self.add_system_message(f"Memory: {mem_stats.get('message_count', 0)} messages, "
                               f"{mem_stats.get('total_tokens', 0)} tokens")
        
        # Story stats
        sme_stats = self.sme.get_pressure_stats()
        if 'current_pressure' in sme_stats:
            pressure = sme_stats['current_pressure']
            arc = sme_stats.get('current_arc', 'unknown')
            self.add_system_message(f"Story: Pressure {pressure:.2f}, Arc {arc}")
        
        # MCP stats
        mcp_info = self.mcp_client.get_server_info()
        self.add_system_message(f"MCP: {mcp_info.get('server_url', 'unknown')}, "
                               f"Connected: {mcp_info.get('connected', False)}")
        
        # Display stats
        self.add_system_message(f"Display: {len(self.display_messages)} messages, "
                               f"Scroll offset: {self.scroll_offset}")
    
    def _change_theme(self, theme_name: str):
        """Change color theme"""
        try:
            theme = ColorTheme(theme_name)
            self.color_manager.theme = theme
            self.color_manager.init_colors()
            self.add_system_message(f"Theme changed to: {theme_name}")
        except ValueError:
            self.add_error_message(f"Unknown theme: {theme_name}. Available: classic, dark, bright")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        # Save conversation if configured
        if self.config.get('auto_save_conversation', False):
            try:
                filename = f"chat_history_{int(time.time())}.json"
                if self.memory_manager.save_conversation(filename):
                    self._log_debug(f"Conversation saved to {filename}")
            except Exception as e:
                self._log_debug(f"Failed to save conversation: {e}")
        
        self._log_debug("Interface shutdown complete")

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Ncurses Interface Module")
    print("Testing interface components...")
    
    # Test color manager
    color_mgr = ColorManager()
    print(f"Color themes available: {[theme.value for theme in ColorTheme]}")
    
    # Test input validator
    validator = InputValidator()
    test_input = "Hello, this is a test input!"
    is_valid, msg = validator.validate(test_input)
    print(f"Input validation test: {is_valid} - {msg}")
    
    # Test display message
    display_msg = DisplayMessage("Test message", "user")
    print(f"Display message: {display_msg.format_for_display()}")
    
    print("Interface module test completed successfully.")
    print("Run main.py to start the full application.")

# End of nci.py - DevName RPG Client Ncurses Interface Module
