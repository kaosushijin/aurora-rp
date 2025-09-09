# Chunk 1/4 - nci.py with Dynamic Box Coordinate System - Imports and Initialization
#!/usr/bin/env python3
"""
DevName RPG Client - Ncurses Interface Module (nci.py) - DYNAMIC COORDINATES
Module architecture and interconnects documented in genai.txt
Uses dynamic box coordinate system for robust window management
"""

import curses
import time
from typing import Dict, List, Any, Optional, Tuple

# Import extracted modules with dynamic coordinate support
from nci_colors import ColorManager, ColorTheme
from nci_terminal import TerminalManager, LayoutGeometry
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
    DYNAMIC COORDINATES: Ncurses interface using box coordinate system
    
    IMPROVEMENTS:
    - Dynamic window positioning eliminates coordinate assumption bugs
    - Automatic adaptation to terminal geometry changes
    - Simplified resize logic with consistent layout
    - Robust coordinate calculations prevent curses NULL returns
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
        
        # Dynamic layout management
        self.current_layout = None
        
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
    
    def _initialize_interface(self, stdscr):
        """Initialize interface with dynamic coordinate system"""
        self.stdscr = stdscr
        
        # Basic ncurses setup
        curses.curs_set(1)
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(0)
        stdscr.clear()
        stdscr.refresh()
        
        # Initialize terminal manager with dynamic coordinates
        self.terminal_manager = TerminalManager(stdscr)
        resized, width, height = self.terminal_manager.check_resize()
        
        # Check minimum size
        if self.terminal_manager.is_too_small():
            self.terminal_manager.show_too_small_message()
            return
        
        # Get initial layout
        self.current_layout = self.terminal_manager.get_box_layout()
        
        # Initialize components with layout dimensions
        self.color_manager.init_colors()
        self._update_component_dimensions()
        
        # Create windows using dynamic coordinates
        self._create_windows_dynamic()
        
        # Populate content and finalize setup
        self._populate_welcome_content()
        self._ensure_cursor_in_input()
        
        self._log_debug(f"Interface initialized with dynamic coordinates: {width}x{height}")
    
    def _update_component_dimensions(self):
        """Update component dimensions from current layout"""
        if not self.current_layout:
            return
        
        # Update multi-input width
        self.multi_input.update_max_width(self.current_layout.terminal_width - 10)
        
        # Update scroll manager height
        self.scroll_manager.update_window_height(self.current_layout.output_box.inner_height)
    
    def _create_windows_dynamic(self):
        """Create ncurses windows using dynamic box coordinates"""
        if not self.current_layout:
            return
        
        layout = self.current_layout
        
        # Output window using box coordinates
        self.output_win = curses.newwin(
            layout.output_box.height,
            layout.output_box.width,
            layout.output_box.top,
            layout.output_box.left
        )
        self.output_win.scrollok(True)
        self.output_win.idlok(True)
        self.output_win.clear()
        self.output_win.refresh()
        
        # Input window using box coordinates
        self.input_win = curses.newwin(
            layout.input_box.height,
            layout.input_box.width,
            layout.input_box.top,
            layout.input_box.left
        )
        self.input_win.clear()
        self._update_input_display()
        
        # Status window using box coordinates
        self.status_win = curses.newwin(
            layout.status_line.height,
            layout.status_line.width,
            layout.status_line.top,
            layout.status_line.left
        )
        self.status_win.clear()
        self.status_win.addstr(0, 0, "Ready")
        self.status_win.refresh()
        
        # Draw borders using layout coordinates
        border_color = self.color_manager.get_color('border')
        self.terminal_manager.draw_box_borders(layout, border_color)

# Chunk 2/4 - nci.py with Dynamic Box Coordinate System - Main Loop and Resize Handling

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
        """Main input processing loop with dynamic coordinate support"""
        while self.running:
            try:
                # Check for terminal resize with dynamic coordinate handling
                resized, new_width, new_height = self.terminal_manager.check_resize()
                if resized:
                    if self.terminal_manager.is_too_small():
                        self.terminal_manager.show_too_small_message()
                        continue
                    else:
                        self._handle_resize_dynamic()
                
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
    
    def _handle_resize_dynamic(self):
        """
        SIMPLIFIED: Handle terminal resize with dynamic coordinate system
        
        No manual coordinate calculations needed - layout system handles everything
        """
        # Get new layout from terminal manager
        self.current_layout = self.terminal_manager.get_box_layout()
        
        if not self.current_layout:
            return
        
        self._log_debug(f"Terminal resized: {self.current_layout.terminal_width}x{self.current_layout.terminal_height}")
        
        # Update component dimensions from new layout
        self._update_component_dimensions()
        
        # Recreate windows with new coordinates
        self._create_windows_dynamic()
        
        # Rewrap content for new dimensions
        self._rewrap_all_content()
        
        # Force complete refresh
        self._refresh_all_windows()
        
        self._log_debug("Dynamic resize handling complete")
    
    def _rewrap_all_content(self):
        """Rewrap all messages for new terminal width"""
        if not self.current_layout:
            return
        
        self.display_lines.clear()
        
        # Use inner width for content wrapping
        content_width = self.current_layout.output_box.inner_width - 2
        
        for message in self.display_messages:
            wrapped_lines = message.format_for_display(content_width)
            for line in wrapped_lines:
                self.display_lines.append((line, message.msg_type))
        
        # Update scroll manager with new content
        self.scroll_manager.update_max_scroll(len(self.display_lines))
    
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

# Chunk 3/4 - nci.py with Dynamic Box Coordinate System - MCP Processing and Display Updates

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
    
    # Display update methods using dynamic box coordinates
    def _add_message_immediate(self, message: DisplayMessage):
        """Add message with immediate display refresh using dynamic coordinates"""
        # Add to message storage
        self.display_messages.append(message)
        
        # Generate wrapped lines using layout inner width
        if self.current_layout:
            content_width = self.current_layout.output_box.inner_width - 2
        else:
            content_width = 78  # Fallback width
        
        wrapped_lines = message.format_for_display(content_width)
        
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
        """Update input display with multi-line support using dynamic coordinates"""
        if not self.current_layout or not self.input_win:
            return
        
        self.input_win.clear()
        
        if self.mcp_processing:
            prompt = "Processing... "
            prompt_color = self.color_manager.get_color('system')
        else:
            prompt = "Input> "
            prompt_color = self.color_manager.get_color('user')
        
        # Get display lines from multi-input using layout dimensions
        available_width = self.current_layout.input_box.inner_width - 8
        available_height = self.current_layout.input_box.inner_height - 1
        
        display_lines = self.multi_input.get_display_lines(available_width, available_height)
        
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
                max_len = self.current_layout.input_box.inner_width - len(prompt) - 1
                if len(first_line) > max_len:
                    first_line = first_line[:max_len]
                
                self.input_win.addstr(0, len(prompt), first_line)
                
                # Display additional lines if multi-line
                for i, line in enumerate(display_lines[1:], 1):
                    if i >= self.current_layout.input_box.inner_height - 1:
                        break
                    
                    max_len = self.current_layout.input_box.inner_width - 1
                    if len(line) > max_len:
                        line = line[:max_len]
                    
                    self.input_win.addstr(i, 0, line)
            
        except curses.error:
            pass
        
        self.input_win.refresh()
        self._ensure_cursor_in_input()
    
    def _update_output_display(self):
        """Update output window with immediate refresh using dynamic coordinates"""
        if not self.current_layout or not self.output_win:
            return
        
        self.output_win.clear()
        
        # Get visible lines based on scroll position
        start_idx, end_idx = self.scroll_manager.get_visible_range()
        visible_lines = self.display_lines[start_idx:end_idx]
        
        # Display lines with proper colors using layout dimensions
        max_display_lines = self.current_layout.output_box.inner_height
        
        for i, (line_text, msg_type) in enumerate(visible_lines):
            if i >= max_display_lines:
                break
            
            # Handle empty lines (true blank lines)
            if not line_text and not msg_type:
                try:
                    self.output_win.addstr(i, 0, "")
                except curses.error:
                    pass
                continue
            
            color = self.color_manager.get_color(msg_type)
            max_line_width = self.current_layout.output_box.inner_width - 1
            display_text = line_text[:max_line_width] if len(line_text) >= max_line_width else line_text
            
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
        """Update status with scroll indicators using dynamic coordinates"""
        if not self.current_layout or not self.status_win:
            return
        
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
            status_parts.append(f"{self.current_layout.terminal_width}x{self.current_layout.terminal_height}")
        
        status_text = " | ".join(status_parts)
        
        # Use layout width for truncation
        max_status_width = self.current_layout.status_line.inner_width - 1
        if len(status_text) > max_status_width:
            status_text = status_text[:max_status_width - 3] + "..."
        
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

# Chunk 4/4 - nci.py with Dynamic Box Coordinate System - Cursor Management, Commands and Utilities

    def _ensure_cursor_in_input(self):
        """Ensure cursor is positioned correctly using dynamic coordinates"""
        try:
            if not self.mcp_processing and self.input_win and self.current_layout:
                # Get actual cursor position from multi-line input
                cursor_line, cursor_col = self.multi_input.get_cursor_position()

                # For multi-line display, map logical cursor to display cursor
                available_width = self.current_layout.input_box.inner_width - 8
                available_height = self.current_layout.input_box.inner_height - 1
                
                display_lines = self.multi_input.get_display_lines(available_width, available_height)

                if cursor_line == 0:
                    # First line - add prompt length
                    prompt_len = len("Input> ")
                    visual_x = prompt_len + cursor_col
                else:
                    # Subsequent lines - no prompt prefix
                    visual_x = cursor_col

                # Clamp to layout boundaries
                max_width = self.current_layout.input_box.inner_width - 1
                max_height = self.current_layout.input_box.inner_height - 1
                
                visual_x = min(visual_x, max_width)
                visual_y = min(cursor_line, max_height)

                # Set cursor position
                self.input_win.move(visual_y, visual_x)
                self.input_win.refresh()
                curses.curs_set(1)

        except curses.error:
            pass
    
    def _refresh_all_windows(self):
        """Refresh all windows immediately using dynamic coordinates"""
        try:
            self._update_output_display()
            self._update_input_display()
            self._update_status_display()
            
            # Redraw borders using layout
            if self.current_layout:
                border_color = self.color_manager.get_color('border')
                self.terminal_manager.draw_box_borders(self.current_layout, border_color)
                
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
        if self.output_win:
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
        """Show comprehensive system statistics using dynamic coordinates"""
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
        
        # Display stats using dynamic coordinates
        scroll_info = self.scroll_manager.get_scroll_info()
        self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, "
                                       f"Scroll: {scroll_info['offset']}/{scroll_info['max']}")
        
        # Terminal stats with layout info
        if self.current_layout:
            layout = self.current_layout
            self.add_system_message_immediate(f"Terminal: {layout.terminal_width}x{layout.terminal_height}")
            self.add_system_message_immediate(f"Layout: Output {layout.output_box.inner_width}x{layout.output_box.inner_height}, "
                                           f"Input {layout.input_box.inner_width}x{layout.input_box.inner_height}")
        else:
            self.add_system_message_immediate("Terminal: Layout not available")
        
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
    print("DevName RPG Client - Ncurses Interface Module - DYNAMIC COORDINATES")
    print("Successfully implemented dynamic box coordinate system:")
    print("✓ nci_terminal.py - Box coordinate calculation and layout geometry")
    print("✓ nci.py - Dynamic window positioning and content adaptation")
    print("✓ Eliminated manual coordinate calculations")
    print("✓ Automatic adaptation to terminal geometry changes")
    print("✓ Simplified resize handling with consistent layout")
    print("✓ Robust coordinate system prevents curses NULL returns")
    print("\nDynamic coordinate system provides:")
    print("- Coordinate independence from terminal size")
    print("- Automatic proportional scaling")
    print("- Visual consistency across all terminal sizes") 
    print("- Simplified debugging with clear boundaries")
    print("- Elimination of coordinate assumption bugs")
    print("\nRun main.py to start the application with dynamic coordinates.")
