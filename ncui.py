# Chunk 1/5 - ncui.py - Header, Imports, and Constructor (TerminalManager Fix)
# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - NCurses UI Controller (ncui.py)
Simplified UI management without orchestration logic - business logic moved to orch.py
FIXED: TerminalManager initialization deferred until stdscr is available
"""

import curses
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import consolidated UI library - direct import from current directory
try:
    from uilib import (
        ColorManager, ColorTheme, TerminalManager, LayoutGeometry,
        DisplayMessage, InputValidator, ScrollManager, MultiLineInput,
        MIN_SCREEN_WIDTH, MIN_SCREEN_HEIGHT, MAX_USER_INPUT_TOKENS
    )
except ImportError as e:
    print(f"UI library import failed: {e}")
    print("Ensure uilib.py is present in current directory")
    raise

# =============================================================================
# UI CONTROLLER CLASS
# =============================================================================

class NCursesUIController:
    """
    Pure UI management without business logic.
    Orchestration logic moved to orch.py - this handles only display and input.
    """
    
    def __init__(self, orchestrator_callback: Callable[[str, Dict[str, Any]], None], debug_logger=None):
        """Initialize UI controller with orchestrator callback and debug logger"""
        # Orchestrator interface
        self.callback_handler = orchestrator_callback
        self.debug_logger = debug_logger

        # Core state
        self.stdscr = None
        self.running = False

        # Core UI managers from consolidated uilib
        self.color_manager = ColorManager()
        self.terminal_manager = None  # Created in initialize() when stdscr is available
        self.input_validator = InputValidator()

        # Window objects and layout
        self.output_window = None
        self.input_window = None
        self.status_window = None
        self.current_layout = None

        # UI components
        self.multi_input = MultiLineInput()
        self.scroll_manager = ScrollManager()

        # Display state
        self.display_buffer = []
        self.status_message = "Ready"
        self.processing = False

        # Configuration
        self.debug_mode = bool(debug_logger)
        
    def initialize(self, stdscr) -> bool:
        """Initialize curses interface and UI components"""
        try:
            self.stdscr = stdscr
            
            # NOW create TerminalManager with stdscr available
            self.terminal_manager = TerminalManager(stdscr)
            
            # Initialize color management
            self.color_manager.init_colors()
            
            # Configure curses settings
            curses.curs_set(1)  # Show cursor
            curses.raw(True)    # Raw mode for better key handling
            self.stdscr.nodelay(True)  # Non-blocking input
            self.stdscr.timeout(100)   # 100ms timeout for input
            
            # Create initial layout
            self._create_layout()
            
            # Initialize UI components with layout
            self._initialize_components()
            
            # Draw initial interface
            self._draw_initial_interface()
            
            return True
            
        except Exception as e:
            self._log_debug(f"UI initialization error: {e}")
            return False

    def _log_debug(self, message: str):
        """Debug logging helper - updated to use debug_logger if available"""
        if self.debug_logger:
            self.debug_logger.debug(message, "NCUI")
        elif self.debug_mode:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] MAIN: NCUI: {message}", file=sys.stderr)

# Chunk 2/5 - ncui.py - Layout and Component Management (Fixed)

    def _create_layout(self):
        """Create window layout using terminal manager"""
        try:
            screen_height, screen_width = self.stdscr.getmaxyx()
            
            # Validate minimum screen size
            if screen_height < MIN_SCREEN_HEIGHT or screen_width < MIN_SCREEN_WIDTH:
                raise RuntimeError(f"Terminal too small: {screen_width}x{screen_height}, need {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}")
            
            # Create layout geometry using TerminalManager
            self.current_layout = self.terminal_manager.create_layout(screen_width, screen_height)
            
            self._log_debug(f"Component dimensions updated: scroll_height={self.current_layout.output_box.inner_height}")
            
        except Exception as e:
            self._log_debug(f"Layout creation error: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize UI components with current layout"""
        try:
            if not self.current_layout:
                raise RuntimeError("Layout not created")
            
            # Create windows
            self._create_windows()
            
            # Initialize scroll manager with output dimensions
            output_height = self.current_layout.output_box.inner_height
            self.scroll_manager.reset(max_scroll=0, visible_height=output_height)
            
            # Configure multi-line input
            input_width = self.current_layout.input_box.inner_width
            self.multi_input.update_max_width(input_width)
            
            self._log_debug(f"Windows created - Output: {self.current_layout.output_box.inner_height}x{self.current_layout.output_box.inner_width} Input: {self.current_layout.input_box.inner_height}x{self.current_layout.input_box.inner_width} Status: {self.current_layout.status_box.inner_height}x{self.current_layout.status_box.inner_width}")
            
        except Exception as e:
            self._log_debug(f"Component initialization error: {e}")
            raise
    
    def _create_windows(self):
        """Create curses windows based on layout"""
        try:
            layout = self.current_layout
            
            # Create output window
            self.output_window = curses.newwin(
                layout.output_box.height,
                layout.output_box.width,
                layout.output_box.y,
                layout.output_box.x
            )
            
            # Create input window  
            self.input_window = curses.newwin(
                layout.input_box.height,
                layout.input_box.width,
                layout.input_box.y,
                layout.input_box.x
            )
            
            # Create status window
            self.status_window = curses.newwin(
                layout.status_box.height,
                layout.status_box.width,
                layout.status_box.y,
                layout.status_box.x
            )
            
        except Exception as e:
            self._log_debug(f"Window creation error: {e}")
            raise
    
    def _draw_initial_interface(self):
        """Draw the initial interface elements"""
        try:
            # Clear screen
            self.stdscr.clear()
            self.stdscr.refresh()
            
            # Draw window borders if layout supports it
            if hasattr(self.current_layout, 'draw_borders'):
                self.current_layout.draw_borders(self.stdscr)
            
            # Initial window refreshes
            self._redraw_all_windows()
            
            # Add welcome message
            self.add_message({
                'content': 'DevName RPG Client initialized. Type /help for commands.',
                'type': 'system'
            })
            
        except Exception as e:
            self._log_debug(f"Initial interface error: {e}")

# Chunk 3/5 - ncui.py - Main Loop and Input Handling (Fixed)

    def run(self) -> int:
        """Main UI loop - pure event handling, business logic in orchestrator"""
        try:
            self.running = True
            
            while self.running:
                try:
                    # Handle input events
                    key = self.stdscr.getch()
                    
                    if key != -1:  # Key was pressed
                        self._handle_key_input(key)
                    
                    # Handle terminal resize
                    if key == curses.KEY_RESIZE:
                        self._handle_resize()
                    
                    # Update cursor position
                    self._update_cursor_position()
                    
                    # Brief sleep to prevent high CPU usage
                    time.sleep(0.01)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    self._log_debug(f"Main loop error: {e}")
                    continue
            
            return 0
            
        except Exception as e:
            self._log_debug(f"UI main loop error: {e}")
            return 1
    
    def _handle_key_input(self, key: int):
        """Handle keyboard input"""
        try:
            if self.processing:
                return  # Ignore input while processing
            
            # Handle special keys
            if key == 27:  # ESC
                self.running = False
                return
            elif key == curses.KEY_RESIZE:
                self._handle_resize()
                return
            elif key in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
                self._handle_navigation_keys(key)
                return
            
            # Handle text input
            if key == ord('\n') or key == curses.KEY_ENTER:
                self._handle_enter_key()
            elif key in (curses.KEY_BACKSPACE, ord('\b'), 127):
                self._handle_backspace()
            elif 32 <= key <= 126:  # Printable characters
                self._handle_character_input(chr(key))
            
        except Exception as e:
            self._log_debug(f"Key input error: {e}")
    
    def _handle_enter_key(self):
        """Handle Enter key - submission or new line"""
        try:
            should_submit, content = self.multi_input.handle_enter()
            
            if should_submit and content.strip():
                # Send to orchestrator for processing
                self.callback_handler('user_input', {
                    'content': content,
                    'timestamp': time.time()
                })
                
                # Add to display and clear input
                self.add_message({
                    'content': content,
                    'type': 'user',
                    'timestamp': time.time()
                })
                
                self.multi_input.clear()
                self._redraw_input_area()
            else:
                # Just redraw input area for new line
                self._redraw_input_area()
                
        except Exception as e:
            self._log_debug(f"Enter key error: {e}")
    
    def _handle_backspace(self):
        """Handle backspace key"""
        try:
            if self.multi_input.handle_backspace():
                self._redraw_input_area()
        except Exception as e:
            self._log_debug(f"Backspace error: {e}")
    
    def _handle_character_input(self, char: str):
        """Handle character input"""
        try:
            if self.multi_input.insert_char(char):
                self._redraw_input_area()
        except Exception as e:
            self._log_debug(f"Character input error: {e}")
    
    def _handle_navigation_keys(self, key: int):
        """Handle navigation keys for scrolling"""
        try:
            if key == curses.KEY_UP:
                self.scroll_manager.scroll_up(1)
            elif key == curses.KEY_DOWN:
                self.scroll_manager.scroll_down(1)
            elif key == curses.KEY_PPAGE:  # Page Up
                self.scroll_manager.scroll_up(10)
            elif key == curses.KEY_NPAGE:  # Page Down
                self.scroll_manager.scroll_down(10)
            
            self._redraw_output_area()
            
        except Exception as e:
            self._log_debug(f"Navigation error: {e}")
    
    def _handle_resize(self):
        """Handle terminal resize"""
        try:
            # Recreate layout
            self._create_layout()
            
            # Reinitialize components
            self._initialize_components()
            
            # Redraw everything
            self._redraw_all_windows()
            
        except Exception as e:
            self._log_debug(f"Resize error: {e}")
    
    def _update_cursor_position(self):
        """Update cursor position based on input state"""
        try:
            if self.processing or not self.input_window or not self.current_layout:
                return
            
            # Get cursor position from multi-input - FIXED METHOD NAME
            cursor_line, cursor_col = self.multi_input.get_cursor_position()
            
            # Calculate display position
            prompt_len = len("Input> ")
            display_x = min(prompt_len + cursor_col, self.current_layout.input_box.inner_width - 1)
            display_y = min(cursor_line, self.current_layout.input_box.inner_height - 1)
            
            # Set cursor position
            self.input_window.move(display_y, display_x)
            self.input_window.refresh()
            curses.curs_set(1)
            
        except Exception as e:
            self._log_debug(f"Cursor position error: {e}")

# Chunk 4/5 - ncui.py - Message Handling and Display Updates (Fixed)

    def handle_orchestrator_response(self, response: Dict[str, Any]):
        """Handle response from orchestrator"""
        try:
            response_type = response.get('type', 'unknown')
            
            if response_type == 'message':
                self.add_message({
                    'content': response.get('content', ''),
                    'type': 'assistant',
                    'timestamp': time.time()
                })
            elif response_type == 'status':
                self.update_status(response.get('content', ''))
            elif response_type == 'processing':
                self.set_processing_state(response.get('active', False))
            elif response_type == 'error':
                self.add_message({
                    'content': f"Error: {response.get('content', 'Unknown error')}",
                    'type': 'error',
                    'timestamp': time.time()
                })
            
        except Exception as e:
            self._log_debug(f"Response handling error: {e}")
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to display buffer and update UI - FIXED CONSTRUCTOR"""
        try:
            # Create display message - FIXED: msg_type parameter name
            display_msg = DisplayMessage(
                content=message.get('content', ''),
                msg_type=message.get('type', 'system')
            )
            
            # Add to buffer
            self.display_buffer.append(display_msg)
            
            # Update scroll manager
            self._update_scroll_for_new_content()
            
            # Redraw output area
            self._redraw_output_area()
            
        except Exception as e:
            self._log_debug(f"Add message error: {e}")
    
    def _update_scroll_for_new_content(self):
        """Update scroll manager when new content is added"""
        if not self.current_layout:
            return
        
        try:
            # Calculate total display lines
            total_lines = 0
            output_width = self.current_layout.output_box.inner_width
            
            for msg in self.display_buffer:
                wrapped_lines = msg.wrap_content(output_width)
                total_lines += len(wrapped_lines)
            
            # Update scroll manager
            self.scroll_manager.update_max_scroll(total_lines)
            
        except Exception as e:
            self._log_debug(f"Scroll update error: {e}")
    
    def update_status(self, message: str):
        """Update status message"""
        try:
            self.status_message = message
            self._redraw_status_area()
        except Exception as e:
            self._log_debug(f"Status update error: {e}")
    
    def set_processing_state(self, processing: bool):
        """Set processing state"""
        try:
            self.processing = processing
            if processing:
                self.update_status("Processing...")
                curses.curs_set(0)  # Hide cursor during processing
            else:
                self.update_status("Ready")
                curses.curs_set(1)  # Show cursor when ready
                
            self._redraw_input_area()
            
        except Exception as e:
            self._log_debug(f"Processing state error: {e}")
    
    def _get_color_for_message_type(self, msg_type: str) -> int:
        """Get color attribute for message type"""
        try:
            if not self.color_manager.colors_available:
                return 0
            
            color_code = self.color_manager.get_color(msg_type)
            return curses.color_pair(color_code) if color_code else 0
            
        except Exception:
            return 0
    
    def _redraw_all_windows(self):
        """Redraw all windows after resize or major change"""
        try:
            self.stdscr.clear()
            self.stdscr.refresh()
            
            self._redraw_output_area()
            self._redraw_input_area()
            self._redraw_status_area()
            
        except Exception as e:
            self._log_debug(f"Full redraw error: {e}")
    
    def _redraw_output_area(self):
        """Redraw the output/message area"""
        if not self.output_window or not self.current_layout:
            return
        
        try:
            self.output_window.clear()
            
            # Get display dimensions
            output_box = self.current_layout.output_box
            display_height = output_box.inner_height
            display_width = output_box.inner_width
            
            # Prepare all display lines
            all_lines = []
            for msg in self.display_buffer:
                wrapped_lines = msg.wrap_content(display_width)
                for line in wrapped_lines:
                    all_lines.append((line, msg.msg_type))
            
            # Get visible range from scroll manager
            start_idx, end_idx = self.scroll_manager.get_visible_range()
            visible_lines = all_lines[start_idx:start_idx + display_height]
            
            # Draw visible lines
            for i, (line, msg_type) in enumerate(visible_lines):
                if i >= display_height:
                    break
                
                try:
                    # Apply color based on message type
                    color_attr = self._get_color_for_message_type(msg_type)
                    
                    if color_attr:
                        self.output_window.addstr(i, 0, line[:display_width], color_attr)
                    else:
                        self.output_window.addstr(i, 0, line[:display_width])
                        
                except curses.error:
                    pass  # Ignore positioning errors
            
            # Draw scroll indicator
            self._draw_scroll_indicator()
            
            self.output_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Output redraw error: {e}")
    
    def _draw_scroll_indicator(self):
        """Draw scroll indicator if needed"""
        try:
            if not self.current_layout or not self.scroll_manager.in_scrollback:
                return
            
            # Simple scroll indicator in top-right corner
            indicator = "↑"
            max_x = self.current_layout.output_box.inner_width - 1
            
            try:
                color_attr = self._get_color_for_message_type('system')
                if color_attr:
                    self.output_window.addstr(0, max_x, indicator, color_attr)
                else:
                    self.output_window.addstr(0, max_x, indicator)
            except curses.error:
                pass
                
        except Exception as e:
            self._log_debug(f"Scroll indicator error: {e}")

# Chunk 5/5 - ncui.py - Input/Status Display and Cleanup (Fixed)

    def _redraw_input_area(self):
        """Redraw the input area"""
        if not self.input_window or not self.current_layout:
            return
        
        try:
            self.input_window.clear()
            
            # Get input dimensions
            input_box = self.current_layout.input_box
            available_width = input_box.inner_width
            available_height = input_box.inner_height
            
            # Get display lines from multi-input
            display_lines = self.multi_input.get_display_lines(available_width, available_height)
            
            # Draw input lines
            for i, line in enumerate(display_lines):
                if i >= available_height:
                    break
                
                try:
                    # Add prompt prefix to first line
                    if i == 0:
                        if self.processing:
                            prompt = "Processing... "
                            color_attr = self._get_color_for_message_type('system')
                        else:
                            prompt = "Input> "
                            color_attr = self._get_color_for_message_type('user')
                        
                        # Draw prompt with color
                        if color_attr:
                            self.input_window.addstr(i, 0, prompt, color_attr)
                        else:
                            self.input_window.addstr(i, 0, prompt)
                        
                        # Draw line content
                        content_start = len(prompt)
                        max_content_len = available_width - content_start - 1
                        if line and max_content_len > 0:
                            self.input_window.addstr(i, content_start, line[:max_content_len])
                    else:
                        # Continuation lines without prompt
                        self.input_window.addstr(i, 0, line[:available_width])
                        
                except curses.error:
                    pass  # Ignore positioning errors
            
            self.input_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Input redraw error: {e}")
    
    def _redraw_status_area(self):
        """Redraw the status area"""
        if not self.status_window or not self.current_layout:
            return
        
        try:
            self.status_window.clear()
            
            # Create status text
            status_text = f" {self.status_message}"
            max_width = self.current_layout.status_box.inner_width - 1
            
            # Truncate if too long
            if len(status_text) > max_width:
                status_text = status_text[:max_width-3] + "..."
            
            # Draw status with color
            try:
                color_attr = self._get_color_for_message_type('system')
                if color_attr:
                    self.status_window.addstr(0, 0, status_text, color_attr)
                else:
                    self.status_window.addstr(0, 0, status_text)
            except curses.error:
                pass
            
            self.status_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Status redraw error: {e}")
    
    def shutdown(self):
        """Clean shutdown of UI controller"""
        try:
            self.running = False
            
            # Clean up curses
            if self.stdscr:
                curses.curs_set(1)  # Restore cursor
                curses.raw(False)   # Disable raw mode
                curses.echo()       # Restore echo
                self.stdscr.clear()
                self.stdscr.refresh()
            
            self._log_debug("UI shutdown complete")
            
        except Exception as e:
            self._log_debug(f"UI shutdown error: {e}")
    
    def enable_debug(self):
        """Enable debug logging"""
        self.debug_mode = True
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get current screen information"""
        try:
            if not self.stdscr or not self.current_layout:
                return {}
            
            height, width = self.stdscr.getmaxyx()
            
            return {
                'screen_size': {'width': width, 'height': height},
                'layout': {
                    'output': {
                        'x': self.current_layout.output_box.x,
                        'y': self.current_layout.output_box.y,
                        'width': self.current_layout.output_box.width,
                        'height': self.current_layout.output_box.height
                    },
                    'input': {
                        'x': self.current_layout.input_box.x,
                        'y': self.current_layout.input_box.y,
                        'width': self.current_layout.input_box.width,
                        'height': self.current_layout.input_box.height
                    },
                    'status': {
                        'x': self.current_layout.status_box.x,
                        'y': self.current_layout.status_box.y,
                        'width': self.current_layout.status_box.width,
                        'height': self.current_layout.status_box.height
                    }
                },
                'colors_available': self.color_manager.colors_available,
                'message_count': len(self.display_buffer)
            }
            
        except Exception as e:
            self._log_debug(f"Screen info error: {e}")
            return {}
    
    def clear_display(self):
        """Clear the display buffer"""
        try:
            self.display_buffer.clear()
            self.scroll_manager.reset(max_scroll=0, visible_height=self.current_layout.output_box.inner_height)
            self._redraw_output_area()
        except Exception as e:
            self._log_debug(f"Clear display error: {e}")
    
    def change_theme(self, theme_name: str) -> bool:
        """Change color theme"""
        try:
            success = self.color_manager.change_theme(theme_name)
            if success:
                self._redraw_all_windows()
            return success
        except Exception as e:
            self._log_debug(f"Theme change error: {e}")
            return False
    
    def get_available_themes(self) -> List[str]:
        """Get list of available color themes"""
        try:
            return self.color_manager.get_available_themes()
        except Exception as e:
            self._log_debug(f"Theme list error: {e}")
            return []
