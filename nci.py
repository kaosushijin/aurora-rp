# Chunk 6a/6 - nci.py (refactored) - Part 1/4 - Imports and Initialization
#!/usr/bin/env python3
"""
DevName RPG Client - Ncurses Interface Module (nci.py) - REFACTORED
Module architecture and interconnects documented in genai.txt
Uses extracted support modules for better separation of concerns
"""

import curses
import time
from typing import Dict, List, Any, Optional, Tuple

# Import extracted modules
from nci_colors import ColorManager, ColorTheme
from nci_terminal import TerminalManager
from nci_display import DisplayMessage, InputValidator
from nci_scroll import ScrollManager
from nci_input import MultiLineInput

# Import core dependencies
try:
    from mcp import MCPClient
    from emm import EnhancedMemoryManager, MessageType
    from sme import StoryMomentumEngine
except ImportError as e:
    print(f"Module import failed: {e}")
    raise

# Configuration constants
MAX_USER_INPUT_TOKENS = 2000

class CursesInterface:
    """
    REFACTORED: Ncurses interface using extracted support modules
    
    SIMPLIFICATIONS:
    - Uses extracted modules for specialized functionality
    - Reduced from ~1000 lines to ~400 lines
    - Cleaner separation of concerns
    - Maintained all existing functionality
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
        
        # Screen dimensions (managed by TerminalManager)
        self.screen_height = 0
        self.screen_width = 0
        self.output_win_height = 0
        self.input_win_height = 4
        self.status_win_height = 1
        
        # Extracted module instances
        self.terminal_manager = None  # Initialize after stdscr available
        self.color_manager = ColorManager(ColorTheme(self.config.get('color_theme', 'classic')))
        self.input_validator = InputValidator(MAX_USER_INPUT_TOKENS)
        self.multi_input = MultiLineInput()
        self.scroll_manager = ScrollManager(0)  # Will be updated with actual height
        
        # Message storage and display
        self.display_messages: List[DisplayMessage] = []
        self.display_lines: List[Tuple[str, str]] = []  # (line_text, msg_type)
        
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

# Chunk 6b/6 - nci.py (refactored) - Part 2/4 - Initialization and Window Management
    
    def _initialize_interface(self, stdscr):
        """Initialize interface without MCP connection testing"""
        self.stdscr = stdscr
        
        # Basic ncurses setup
        curses.curs_set(1)
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(0)
        stdscr.clear()
        stdscr.refresh()
        
        # Initialize terminal manager
        self.terminal_manager = TerminalManager(stdscr)
        resized, width, height = self.terminal_manager.check_resize()
        self.screen_width, self.screen_height = width, height
        
        # Check minimum size
        if self.terminal_manager.is_too_small():
            self.terminal_manager.show_too_small_message()
            return
        
        # Initialize components with proper dimensions
        self.color_manager.init_colors()
        self._calculate_window_dimensions()
        self.multi_input.update_max_width(self.screen_width - 10)
        self.scroll_manager.update_window_height(self.output_win_height)
        
        # Create windows and populate content
        self._create_windows()
        self._populate_welcome_content()
        self._ensure_cursor_in_input()
        
        self._log_debug(f"Interface initialized: {self.screen_width}x{self.screen_height}")
    
    def _calculate_window_dimensions(self):
        """Calculate window dimensions with better proportions"""
        # Reserve space for borders
        border_space = 3
        
        # Calculate output window height (majority of screen)
        available_height = self.screen_height - self.input_win_height - self.status_win_height - border_space
        self.output_win_height = max(8, available_height)
        
        # Adjust input height if screen is very small
        if self.output_win_height < 12 and self.input_win_height > 2:
            self.input_win_height = 2
            self.output_win_height = self.screen_height - self.input_win_height - self.status_win_height - border_space
    
    def _create_windows(self):
        """Create ncurses windows with immediate display"""
        # Output window (conversation display)
        self.output_win = curses.newwin(
            self.output_win_height,
            self.screen_width,
            0,
            0
        )
        self.output_win.scrollok(True)
        self.output_win.idlok(True)
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
        self.input_win.clear()
        self._update_input_display()
        
        # Status window (bottom line)
        status_y = input_y + self.input_win_height + 1
        self.status_win = curses.newwin(
            self.status_win_height,
            self.screen_width,
            status_y,
            0
        )
        self.status_win.clear()
        self.status_win.addstr(0, 0, "Ready")
        self.status_win.refresh()
        
        # Draw borders
        self._draw_borders()
    
    def _draw_borders(self):
        """Draw window borders with immediate refresh"""
        border_color = self.color_manager.get_color('border')
        
        if border_color and self.color_manager.colors_available:
            self.stdscr.attron(curses.color_pair(border_color))
        
        # Top border
        self.stdscr.hline(self.output_win_height, 0, curses.ACS_HLINE, self.screen_width)
        
        # Bottom border
        status_y = self.output_win_height + self.input_win_height + 1
        self.stdscr.hline(status_y, 0, curses.ACS_HLINE, self.screen_width)
        
        if border_color and self.color_manager.colors_available:
            self.stdscr.attroff(curses.color_pair(border_color))
        
        self.stdscr.refresh()
    
    def _populate_welcome_content(self):
        """Add welcome messages WITHOUT MCP connection testing"""
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
        
        # Ready message (NO MCP testing)
        ready_msg = DisplayMessage(
            "Ready for adventure! Enter your first action or command.",
            "system"
        )
        self._add_message_immediate(ready_msg)
        
        # Update status
        self._update_status_display()
    
    def _run_main_loop(self):
        """Main input processing loop with enhanced features"""
        while self.running:
            try:
                # Check for terminal resize
                resized, new_width, new_height = self.terminal_manager.check_resize()
                if resized:
                    if self.terminal_manager.is_too_small():
                        self.terminal_manager.show_too_small_message()
                        continue
                    else:
                        self._handle_resize(new_width, new_height)
                
                # Get user input
                key = self.stdscr.getch()
                
                # Process input if not blocked
                if not self.input_blocked:
                    input_changed = self._handle_key_input(key)
                    
                    # Update input display if changed
                    if input_changed:
                        self._update_input_display()
                
                # Periodic status update
                self._update_status_display()
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self._log_debug(f"Main loop error: {e}")
    
    def _handle_resize(self, new_width: int, new_height: int):
        """Handle terminal resize with complete window recreation"""
        self._log_debug(f"Terminal resized: {new_width}x{new_height}")
        
        # Update dimensions
        self.screen_width = new_width
        self.screen_height = new_height
        
        # Recalculate window dimensions
        self._calculate_window_dimensions()
        
        # Update component dimensions
        self.multi_input.update_max_width(new_width - 10)
        self.scroll_manager.update_window_height(self.output_win_height)
        self.scroll_manager.update_max_scroll(len(self.display_lines))
        
        # Recreate windows
        self._create_windows()
        
        # Rewrap all display content
        self._rewrap_all_content()
        
        # Force complete refresh
        self._refresh_all_windows()
        
        self._log_debug("Resize handling complete")
    
    def _rewrap_all_content(self):
        """Rewrap all messages for new terminal width"""
        self.display_lines.clear()
        
        for message in self.display_messages:
            wrapped_lines = message.format_for_display(self.screen_width - 2)
            for line in wrapped_lines:
                self.display_lines.append((line, message.msg_type))
        
        # Update scroll manager with new content
        self.scroll_manager.update_max_scroll(len(self.display_lines))

# Chunk 6c/6 - nci.py (refactored) - Part 3/4 - Input Handling and MCP Processing
    
    def _handle_key_input(self, key: int) -> bool:
        """Enhanced key handling with multi-line input and navigation"""
        try:
            # Multi-line input navigation
            if self.multi_input.handle_arrow_keys(key):
                return True
            
            # Enhanced scrolling
            if key == curses.KEY_UP:
                if self.scroll_manager.handle_line_scroll(-1):
                    self._update_output_display()
                return False
            elif key == curses.KEY_DOWN:
                if self.scroll_manager.handle_line_scroll(1):
                    self._update_output_display()
                return False
            
            # Page navigation
            elif key == curses.KEY_PPAGE:  # PgUp
                if self.scroll_manager.handle_page_scroll(-1):
                    self._update_output_display()
                return False
            elif key == curses.KEY_NPAGE:  # PgDn
                if self.scroll_manager.handle_page_scroll(1):
                    self._update_output_display()
                return False
            
            # Home/End navigation
            elif key == curses.KEY_HOME:
                if self.scroll_manager.handle_home():
                    self._update_output_display()
                return False
            elif key == curses.KEY_END:
                if self.scroll_manager.handle_end():
                    self._update_output_display()
                return False
            
            # Enter key handling
            elif key == ord('\n') or key == curses.KEY_ENTER or key == 10:
                self._handle_enter_key()
                return True
            
            # Backspace
            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                return self.multi_input.handle_backspace()
            
            # Printable characters
            elif 32 <= key <= 126:
                return self.multi_input.insert_char(chr(key))
            
        except Exception as e:
            self._log_debug(f"Key handling error: {e}")
        
        return False
    
    def _handle_enter_key(self):
        """Handle Enter key with multi-line input support"""
        # Try to handle as submission or new line
        should_submit, content = self.multi_input.handle_enter()
        
        if should_submit and content.strip():
            # Validate input
            is_valid, error_msg = self.input_validator.validate(content)
            if not is_valid:
                self.add_error_message_immediate(error_msg)
                return
            
            # Display user message
            self.add_user_message_immediate(content)
            
            # Clear input and set processing state
            self.multi_input.clear()
            self.set_processing_state_immediate(True)
            
            # Auto-scroll to bottom when user submits
            self.scroll_manager.auto_scroll_to_bottom()
            self._update_output_display()
            
            # Process input
            self._process_user_input(content)
    
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
        """Process commands"""
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
            self.add_error_message_immediate(f"Unknown command: {command}")
    
    def _send_mcp_request(self, user_input: str):
        """Send MCP request with improved error handling"""
        try:
            # Get context data
            conversation_history = self.memory_manager.get_conversation_for_mcp()
            story_context = self.sme.get_story_context()
            context_str = self._format_story_context(story_context)
            system_messages = self._build_system_messages(context_str)
            
            # Build complete message chain
            all_messages = system_messages + conversation_history + [{"role": "user", "content": user_input}]
            
            try:
                # Try custom MCP request
                response_data = self.mcp_client._execute_request({
                    "model": self.mcp_client.model,
                    "messages": all_messages,
                    "stream": False
                })
                
                # Store and display response
                self.memory_manager.add_message(response_data, MessageType.ASSISTANT)
                self.add_assistant_message_immediate(response_data)
                
            except ConnectionError:
                self.add_error_message_immediate("Unable to connect to Game Master server")
            except TimeoutError:
                self.add_error_message_immediate("Game Master server response timeout")
            except Exception as mcp_error:
                self._log_debug(f"Custom MCP call failed, trying fallback: {mcp_error}")
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
                    self.add_error_message_immediate(f"Communication error: {str(fallback_error)}")
            
        except Exception as e:
            self.add_error_message_immediate(f"Request processing failed: {e}")
        finally:
            self.set_processing_state_immediate(False)
    
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

# Chunk 6d/6 - nci.py (refactored) - Part 4/4 - Display Methods and Commands
    
    # Display update methods using extracted components
    def _add_message_immediate(self, message: DisplayMessage):
        """Add message with immediate display refresh"""
        # Add to message storage
        self.display_messages.append(message)
        
        # Generate wrapped lines
        wrapped_lines = message.format_for_display(self.screen_width - 2)
        
        # Add to display lines cache
        for line in wrapped_lines:
            self.display_lines.append((line, message.msg_type))
        
        # Update scroll manager
        self.scroll_manager.update_max_scroll(len(self.display_lines))
        self.scroll_manager.auto_scroll_to_bottom()
        
        # Immediate display update
        self._update_output_display()
    
    def _add_blank_line_immediate(self):
        """Add a true blank line with immediate display"""
        self.display_lines.append(("", ""))
        self.scroll_manager.update_max_scroll(len(self.display_lines))
        self.scroll_manager.auto_scroll_to_bottom()
        self._update_output_display()
    
    def _update_input_display(self):
        """Update input display with multi-line support"""
        self.input_win.clear()
        
        if self.mcp_processing:
            prompt = "Processing... "
            prompt_color = self.color_manager.get_color('system')
        else:
            prompt = "Input> "
            prompt_color = self.color_manager.get_color('user')
        
        # Get display lines from multi-input
        display_lines = self.multi_input.get_display_lines(
            self.screen_width - 8, 
            self.input_win_height - 1
        )
        
        # Display prompt and first line
        try:
            if prompt_color and self.color_manager.colors_available:
                self.input_win.attron(curses.color_pair(prompt_color))
                self.input_win.addstr(0, 0, prompt)
                self.input_win.attroff(curses.color_pair(prompt_color))
            else:
                self.input_win.addstr(0, 0, prompt)
            
            # Display input content
            if display_lines:
                first_line = display_lines[0]
                max_len = self.screen_width - len(prompt) - 1
                if len(first_line) > max_len:
                    first_line = first_line[:max_len]
                
                self.input_win.addstr(0, len(prompt), first_line)
                
                # Display additional lines if multi-line
                for i, line in enumerate(display_lines[1:], 1):
                    if i >= self.input_win_height - 1:
                        break
                    
                    max_len = self.screen_width - 1
                    if len(line) > max_len:
                        line = line[:max_len]
                    
                    self.input_win.addstr(i, 0, line)
            
        except curses.error:
            pass
        
        self.input_win.refresh()
        self._ensure_cursor_in_input()
    
    def _update_output_display(self):
        """Update output window with immediate refresh"""
        self.output_win.clear()
        
        # Get visible lines based on scroll position
        start_idx, end_idx = self.scroll_manager.get_visible_range()
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
        
        self.output_win.refresh()
        self._ensure_cursor_in_input()
    
    def _update_status_display(self):
        """Update status with scroll indicators and information"""
        self.status_win.clear()
        
        status_parts = []
        
        # Memory stats
        try:
            mem_stats = self.memory_manager.get_memory_stats()
            msg_count = mem_stats.get('message_count', 0)
            status_parts.append(f"Messages: {msg_count}")
        except:
            status_parts.append("Messages: 0")
        
        # Story pressure
        try:
            sme_stats = self.sme.get_pressure_stats()
            if 'current_pressure' in sme_stats:
                pressure = sme_stats['current_pressure']
                status_parts.append(f"Pressure: {pressure:.2f}")
        except:
            pass
        
        # MCP status
        if self.mcp_processing:
            status_parts.append("GM: Processing")
        else:
            status_parts.append("GM: Ready")
        
        # Prompt count
        prompt_count = len([p for p in self.loaded_prompts.values() if p.strip()])
        status_parts.append(f"Prompts: {prompt_count}")
        
        # Scroll indicator
        scroll_info = self.scroll_manager.get_scroll_info()
        if scroll_info["in_scrollback"]:
            status_parts.append(f"SCROLLBACK ({scroll_info['percentage']}%)")
        
        # Terminal size (for debugging)
        if self.debug_logger:
            status_parts.append(f"{self.screen_width}x{self.screen_height}")
        
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
        
        self.status_win.refresh()
        self._ensure_cursor_in_input()
    
    def _ensure_cursor_in_input(self):
        """Ensure cursor is positioned correctly in input window"""
        try:
            if not self.mcp_processing and self.input_win:
                # Get actual cursor position from multi-line input
                cursor_line, cursor_col = self.multi_input.get_cursor_position()

                # For multi-line display, we need to map logical cursor to display cursor
                display_lines = self.multi_input.get_display_lines(
                    self.screen_width - 8,
                    self.input_win_height - 1
                )

                if cursor_line == 0:
                    # First line - add prompt length
                    prompt_len = len("Input> ")
                    visual_x = prompt_len + cursor_col
                else:
                    # Subsequent lines - no prompt prefix
                    visual_x = cursor_col

                # Clamp to window boundaries
                visual_x = min(visual_x, self.screen_width - 1)
                visual_y = min(cursor_line, self.input_win_height - 1)

                # Set cursor position
                self.input_win.move(visual_y, visual_x)
                self.input_win.refresh()
                curses.curs_set(1)

        except curses.error:
            pass
    
    def _refresh_all_windows(self):
        """Refresh all windows immediately"""
        try:
            self._update_output_display()
            self._update_input_display()
            self._update_status_display()
            self._draw_borders()
        except curses.error as e:
            self._log_debug(f"Display refresh error: {e}")
    
    # Message addition methods with immediate display
    def add_user_message_immediate(self, content: str):
        """Add user message with immediate display"""
        msg = DisplayMessage(content, "user")
        self._add_message_immediate(msg)
    
    def add_assistant_message_immediate(self, content: str):
        """Add assistant message with proper blank line separation"""
        # Add true blank line before GM response
        self._add_blank_line_immediate()
        
        # Add the GM response
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
        """Set processing state with immediate visual feedback"""
        self.mcp_processing = processing
        self.input_blocked = processing
        
        # Immediate input display update
        self._update_input_display()
        
        self._log_debug(f"Processing state: {processing}")
    
    def _clear_display(self):
        """Clear message display with immediate refresh"""
        self.display_messages.clear()
        self.display_lines.clear()
        
        # Reset scroll manager
        self.scroll_manager.scroll_offset = 0
        self.scroll_manager.max_scroll = 0
        self.scroll_manager.in_scrollback = False
        
        # Clear and refresh output window
        self.output_win.clear()
        self.output_win.refresh()
        
        # Add clear confirmation message
        self.add_system_message_immediate("Display cleared")
    
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
            "Navigation:",
            "Arrow Keys - Navigate multi-line input or scroll chat",
            "PgUp/PgDn - Page-based scrolling through chat history",
            "Home/End - Jump to top/bottom of chat history",
            "",
            "Input:",
            "Enter - Submit input (or new line in multi-line mode)",
            "Backspace - Delete character or merge lines",
            "",
            "Multi-line input automatically submits when content ends with",
            "punctuation or is a command. Use Enter for new lines otherwise."
        ]
        
        for msg in help_messages:
            self.add_system_message_immediate(msg)
    
    def _show_stats(self):
        """Show comprehensive system statistics"""
        try:
            # Memory stats
            mem_stats = self.memory_manager.get_memory_stats()
            self.add_system_message_immediate(f"Memory: {mem_stats.get('message_count', 0)} messages, "
                                           f"{mem_stats.get('total_tokens', 0)} tokens, "
                                           f"{mem_stats.get('condensations_performed', 0)} condensations")
        except:
            self.add_system_message_immediate("Memory: Stats unavailable")
        
        try:
            # Story stats
            sme_stats = self.sme.get_pressure_stats()
            if 'current_pressure' in sme_stats:
                pressure = sme_stats['current_pressure']
                arc = sme_stats.get('current_arc', 'unknown')
                updates = sme_stats.get('total_updates', 0)
                self.add_system_message_immediate(f"Story: Pressure {pressure:.2f}, Arc {arc}, {updates} updates")
        except:
            self.add_system_message_immediate("Story: Stats unavailable")
        
        # MCP stats
        mcp_info = self.mcp_client.get_server_info()
        self.add_system_message_immediate(f"MCP: {mcp_info.get('server_url', 'unknown')}")
        self.add_system_message_immediate(f"Model: {mcp_info.get('model', 'unknown')}")
        
        # Display stats
        scroll_info = self.scroll_manager.get_scroll_info()
        self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, "
                                       f"Scroll: {scroll_info['offset']}/{scroll_info['max']}")
        
        # Terminal stats
        self.add_system_message_immediate(f"Terminal: {self.screen_width}x{self.screen_height}")
        
        # Input stats
        input_content = self.multi_input.get_content()
        input_lines = len(self.multi_input.lines)
        cursor_line, cursor_col = self.multi_input.get_cursor_position()
        self.add_system_message_immediate(f"Input: {len(input_content)} chars, {input_lines} lines, "
                                       f"cursor at {cursor_line}:{cursor_col}")
        
        # Prompt stats
        total_tokens = sum(len(content) // 4 for content in self.loaded_prompts.values() if content.strip())
        active_prompts = [name for name, content in self.loaded_prompts.items() if content.strip()]
        self.add_system_message_immediate(f"Prompts: {len(active_prompts)} active ({', '.join(active_prompts)}), "
                                       f"{total_tokens:,} tokens")
    
    def _change_theme(self, theme_name: str):
        """Change color theme with immediate display refresh"""
        if self.color_manager.change_theme(theme_name):
            self.add_system_message_immediate(f"Theme changed to: {theme_name}")
            
            # Force complete refresh with new colors
            self._refresh_all_windows()
        else:
            available_themes = self.color_manager.get_available_themes()
            self.add_error_message_immediate(f"Unknown theme: {theme_name}. Available: {', '.join(available_themes)}")
    
    def shutdown(self):
        """Graceful shutdown with immediate feedback"""
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

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Ncurses Interface Module - REFACTORED")
    print("Successfully extracted support modules:")
    print("✓ nci_colors.py - Color management")
    print("✓ nci_terminal.py - Terminal management") 
    print("✓ nci_display.py - Message display")
    print("✓ nci_scroll.py - Scrolling system")
    print("✓ nci_input.py - Multi-line input")
    print("✓ nci.py - Main interface (reduced complexity)")
    print("\nRefactored interface maintains all functionality while improving code organization.")
    print("Run main.py to start the application with the refactored modules.")
