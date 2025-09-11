# Chunk 1/5 - ncui.py - Header, Imports, and Constructor (ScrollManager Fix)

#!/usr/bin/env python3
"""
DevName RPG Client - NCurses UI Controller (ncui.py)
Simplified UI management without orchestration logic - business logic moved to orch.py
FIXED: ScrollManager initialization with placeholder height, updated after layout calculation
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

        # UI components - FIXED: ScrollManager with placeholder height
        self.multi_input = MultiLineInput()
        self.scroll_manager = ScrollManager(0)  # Placeholder height, updated after layout

        # Display state
        self.display_buffer = []
        self.status_message = "Ready"
        self.processing = False

        # Configuration
        self.debug_mode = bool(debug_logger)

# Chunk 2/5 - ncui.py - Initialization and Layout Methods (ScrollManager Update Fix)
        
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
            stdscr.keypad(True)  # Enable special keys
            stdscr.timeout(100)  # Non-blocking input with 100ms timeout
            
            # Create initial layout and components
            self._update_layout()
            self._initialize_components()
            
            # Initial display
            self._initial_display()
            
            self._log_debug("UI initialization complete")
            return True
            
        except Exception as e:
            self._log_debug(f"UI initialization failed: {e}")
            return False
    
    def _update_layout(self):
        """Update layout based on current terminal size"""
        try:
            # Get current terminal dimensions
            screen_height, screen_width = self.stdscr.getmaxyx()
            
            # Validate minimum dimensions
            if screen_width < MIN_SCREEN_WIDTH or screen_height < MIN_SCREEN_HEIGHT:
                raise RuntimeError(f"Terminal too small: {screen_width}x{screen_height} (minimum: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT})")
            
            # Create new layout
            self.current_layout = self.terminal_manager.create_layout(screen_width, screen_height)
            
            self._log_debug(f"Component dimensions updated: scroll_height={self.current_layout.output_box.inner_height}")
            
        except Exception as e:
            self._log_debug(f"Layout creation error: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize UI components with current layout - FIXED: ScrollManager height update"""
        try:
            if not self.current_layout:
                raise RuntimeError("Layout not created")
            
            # Create windows
            self._create_windows()
            
            # FIXED: Update scroll manager with actual output dimensions
            output_height = self.current_layout.output_box.inner_height
            self.scroll_manager.update_window_height(output_height)
            
            # Configure multi-line input
            input_width = self.current_layout.input_box.inner_width
            self.multi_input.update_max_width(input_width)
            
            self._log_debug(f"Windows created - Output: {self.current_layout.output_box.inner_height}x{self.current_layout.output_box.inner_width} Input: {self.current_layout.input_box.inner_height}x{self.current_layout.input_box.inner_width} Status: {self.current_layout.status_box.inner_height}x{self.current_layout.status_box.inner_width}")
            
        except Exception as e:
            self._log_debug(f"Component initialization error: {e}")
            raise
    
    def _create_windows(self):
        """Create curses windows based on current layout"""
        try:
            layout = self.current_layout
            
            # Create windows
            self.output_window = curses.newwin(
                layout.output_box.height, layout.output_box.width,
                layout.output_box.y, layout.output_box.x
            )
            
            self.input_window = curses.newwin(
                layout.input_box.height, layout.input_box.width,
                layout.input_box.y, layout.input_box.x
            )
            
            self.status_window = curses.newwin(
                layout.status_box.height, layout.status_box.width,
                layout.status_box.y, layout.status_box.x
            )
            
            self._log_debug("Curses windows created successfully")
            
        except Exception as e:
            self._log_debug(f"Window creation error: {e}")
            raise

# Chunk 3/5 - ncui.py - Display and Rendering Methods

    def _initial_display(self):
        """Show initial welcome screen"""
        try:
            # Clear all windows
            self.output_window.clear()
            self.input_window.clear()
            self.status_window.clear()
            
            # Draw borders
            self._draw_borders()
            
            # Show welcome message
            welcome_msg = DisplayMessage(
                content="DevName RPG Client - Hub & Spoke Architecture\nType '/help' for commands or start typing to begin...",
                msg_type='system'
            )
            self.display_buffer.append(welcome_msg)
            
            # Initial render
            self._redraw_all()
            
        except Exception as e:
            self._log_debug(f"Initial display error: {e}")
    
    def _draw_borders(self):
        """Draw window borders"""
        try:
            # Draw borders for all windows
            self.output_window.border()
            self.input_window.border()
            self.status_window.border()
            
            # Add window titles
            layout = self.current_layout
            
            # Output window title
            title = " Output "
            title_x = max(1, (layout.output_box.width - len(title)) // 2)
            self.output_window.addstr(0, title_x, title)
            
            # Input window title
            title = " Input "
            title_x = max(1, (layout.input_box.width - len(title)) // 2)
            self.input_window.addstr(0, title_x, title)
            
            # Status window title
            title = " Status "
            title_x = max(1, (layout.status_box.width - len(title)) // 2)
            self.status_window.addstr(0, title_x, title)
            
        except Exception as e:
            self._log_debug(f"Border drawing error: {e}")
    
    def _redraw_all(self):
        """Redraw all windows"""
        try:
            self._redraw_output_area()
            self._redraw_input_area()
            self._redraw_status_area()
            
            # Refresh all windows
            self.output_window.refresh()
            self.input_window.refresh()
            self.status_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Full redraw error: {e}")
    
    def _redraw_output_area(self):
        """Redraw output area with current messages and scroll state"""
        try:
            if not self.output_window or not self.current_layout:
                return
            
            # Clear content area (preserve border)
            layout = self.current_layout.output_box
            for y in range(1, layout.inner_height + 1):
                self.output_window.addstr(y, 1, " " * layout.inner_width)
            
            # Get visible message range
            start_idx, end_idx = self.scroll_manager.get_visible_range()
            
            # Render visible messages
            current_line = 1  # Start after border
            output_width = layout.inner_width
            
            # Collect all wrapped lines first
            all_lines = []
            for msg in self.display_buffer:
                wrapped_lines = msg.wrap_content(output_width)
                for line in wrapped_lines:
                    all_lines.append((line, msg.msg_type))
            
            # Display the visible portion
            visible_lines = all_lines[start_idx:end_idx + 1] if all_lines else []
            
            for line_text, msg_type in visible_lines:
                if current_line > layout.inner_height:
                    break
                
                color_attr = self._get_color_for_message_type(msg_type)
                
                try:
                    self.output_window.addstr(current_line, 1, line_text[:output_width], color_attr)
                except curses.error:
                    pass  # Ignore curses drawing errors at screen edge
                
                current_line += 1
            
            # Update scroll manager with total lines
            self.scroll_manager.update_max_scroll(len(all_lines))
            
            # Add scroll indicator if needed
            self._add_scroll_indicator()
            
        except Exception as e:
            self._log_debug(f"Output redraw error: {e}")
    
    def _add_scroll_indicator(self):
        """Add scroll position indicator"""
        try:
            if not self.scroll_manager.in_scrollback:
                return
            
            layout = self.current_layout.output_box
            scroll_info = self.scroll_manager.get_scroll_info()
            
            if scroll_info.get('scroll_needed', False):
                indicator = f"[{scroll_info['percentage']}%]"
                indicator_x = layout.width - len(indicator) - 1
                
                try:
                    self.output_window.addstr(0, indicator_x, indicator, curses.A_REVERSE)
                except curses.error:
                    pass
                    
        except Exception as e:
            self._log_debug(f"Scroll indicator error: {e}")
    
    def _redraw_input_area(self):
        """Redraw input area with current input and cursor"""
        try:
            if not self.input_window or not self.current_layout:
                return
            
            layout = self.current_layout.input_box
            
            # Clear content area (preserve border)
            for y in range(1, layout.inner_height + 1):
                self.input_window.addstr(y, 1, " " * layout.inner_width)
            
            # Get current input lines
            input_lines = self.multi_input.get_display_lines()
            
            # Display input lines
            for i, line in enumerate(input_lines):
                if i + 1 > layout.inner_height:
                    break
                
                display_line = line[:layout.inner_width]
                try:
                    self.input_window.addstr(i + 1, 1, display_line)
                except curses.error:
                    pass
            
            # Position cursor if not processing
            if not self.processing:
                cursor_y, cursor_x = self.multi_input.get_cursor_position()
                # Adjust for border and clamp to window bounds
                cursor_y = min(cursor_y + 1, layout.inner_height)
                cursor_x = min(cursor_x + 1, layout.inner_width)
                
                try:
                    self.input_window.move(cursor_y, cursor_x)
                except curses.error:
                    pass
                    
        except Exception as e:
            self._log_debug(f"Input redraw error: {e}")
    
    def _redraw_status_area(self):
        """Redraw status area"""
        try:
            if not self.status_window or not self.current_layout:
                return
            
            layout = self.current_layout.status_box
            
            # Clear content area (preserve border)
            self.status_window.addstr(1, 1, " " * layout.inner_width)
            
            # Display status message
            status_text = self.status_message[:layout.inner_width]
            try:
                self.status_window.addstr(1, 1, status_text)
            except curses.error:
                pass
                
        except Exception as e:
            self._log_debug(f"Status redraw error: {e}")

# Chunk 4/5 - ncui.py - Input Handling and Message Management

    def run(self):
        """Main UI loop"""
        try:
            self.running = True
            
            while self.running:
                try:
                    # Get input with timeout
                    key = self.stdscr.getch()
                    
                    if key == -1:  # Timeout, no input
                        continue
                    
                    # Handle input
                    self._handle_input(key)
                    
                    # Refresh display
                    self._redraw_all()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self._log_debug(f"Main loop error: {e}")
                    
        except Exception as e:
            self._log_debug(f"UI run error: {e}")
        finally:
            self.running = False
    
    def _handle_input(self, key: int):
        """Handle keyboard input"""
        try:
            # Navigation keys (scrolling)
            if key == curses.KEY_UP:
                if self.scroll_manager.handle_line_scroll(-1):
                    self._redraw_output_area()
                return
            elif key == curses.KEY_DOWN:
                if self.scroll_manager.handle_line_scroll(1):
                    self._redraw_output_area()
                return
            elif key == curses.KEY_PPAGE:  # Page Up
                if self.scroll_manager.handle_page_scroll(-1):
                    self._redraw_output_area()
                return
            elif key == curses.KEY_NPAGE:  # Page Down
                if self.scroll_manager.handle_page_scroll(1):
                    self._redraw_output_area()
                return
            elif key == curses.KEY_HOME:
                if self.scroll_manager.handle_home():
                    self._redraw_output_area()
                return
            elif key == curses.KEY_END:
                if self.scroll_manager.handle_end():
                    self._redraw_output_area()
                return
            
            # Text input handling
            if key == ord('\n') or key == curses.KEY_ENTER:
                self._handle_enter()
            elif key == curses.KEY_BACKSPACE or key == 127:
                self.multi_input.handle_backspace()
                self._redraw_input_area()
            elif key == curses.KEY_DC:  # Delete key
                self.multi_input.handle_delete()
                self._redraw_input_area()
            elif key == curses.KEY_LEFT:
                self.multi_input.handle_cursor_left()
                self._redraw_input_area()
            elif key == curses.KEY_RIGHT:
                self.multi_input.handle_cursor_right()
                self._redraw_input_area()
            elif 32 <= key <= 126:  # Printable characters
                self.multi_input.handle_char_input(chr(key))
                self._redraw_input_area()
                
                # Auto-scroll to bottom on new input
                if self.scroll_manager.in_scrollback:
                    self.scroll_manager.auto_scroll_to_bottom()
                    self._redraw_output_area()
                    
        except Exception as e:
            self._log_debug(f"Input handling error: {e}")
    
    def _handle_enter(self):
        """Handle enter key - submit input or add newline"""
        try:
            current_text = self.multi_input.get_current_text()
            
            # Check for double enter (submit)
            if self.multi_input.should_submit():
                self._submit_input(current_text)
            else:
                # Single enter - add newline
                self.multi_input.handle_char_input('\n')
                self._redraw_input_area()
                
        except Exception as e:
            self._log_debug(f"Enter handling error: {e}")
    
    def _submit_input(self, text: str):
        """Submit user input to orchestrator"""
        try:
            # Validate input
            is_valid, error_msg = self.input_validator.validate(text.strip())
            
            if not is_valid:
                self.add_message({
                    'content': f"Input error: {error_msg}",
                    'type': 'error'
                })
                return
            
            # Clear input
            self.multi_input.clear()
            self._redraw_input_area()
            
            # Add user message to display
            self.add_message({
                'content': text.strip(),
                'type': 'user'
            })
            
            # Set processing state
            self.set_processing_state(True)
            
            # Send to orchestrator via callback
            self.callback_handler('user_input', {'text': text.strip()})
            
        except Exception as e:
            self._log_debug(f"Input submission error: {e}")
            self.set_processing_state(False)
    
    def add_message(self, message: Dict[str, Any]):
        """Add message to display buffer"""
        try:
            # Create display message
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

# Chunk 5/5 - ncui.py - Utility Methods and Shutdown

    def handle_resize(self):
        """Handle terminal resize"""
        try:
            # Update layout for new terminal size
            self._update_layout()
            
            # Reinitialize components with new layout
            self._initialize_components()
            
            # Redraw everything
            self.stdscr.clear()
            self._draw_borders()
            self._redraw_all()
            
            self._log_debug("Terminal resize handled")
            
        except Exception as e:
            self._log_debug(f"Resize handling error: {e}")
    
    def switch_theme(self, theme_number: int):
        """Switch color theme"""
        try:
            if self.color_manager.switch_theme(theme_number):
                self._redraw_all()
                self.update_status(f"Switched to theme {theme_number}")
                return True
            else:
                self.update_status(f"Invalid theme: {theme_number}")
                return False
                
        except Exception as e:
            self._log_debug(f"Theme switch error: {e}")
            return False
    
    def clear_display(self):
        """Clear the display buffer"""
        try:
            self.display_buffer.clear()
            self.scroll_manager.auto_scroll_to_bottom()
            self._redraw_output_area()
            self.update_status("Display cleared")
            
        except Exception as e:
            self._log_debug(f"Display clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get UI statistics"""
        try:
            stats = {
                'display_buffer_size': len(self.display_buffer),
                'scroll_info': self.scroll_manager.get_scroll_info(),
                'terminal_size': f"{self.stdscr.getmaxyx()[1]}x{self.stdscr.getmaxyx()[0]}",
                'color_theme': self.color_manager.current_theme_name(),
                'processing': self.processing
            }
            return stats
            
        except Exception as e:
            self._log_debug(f"Stats generation error: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown UI gracefully"""
        try:
            self.running = False
            
            # Reset cursor
            if self.stdscr:
                curses.curs_set(1)
                
            self._log_debug("UI shutdown complete")
            
        except Exception as e:
            self._log_debug(f"Shutdown error: {e}")
    
    def _log_debug(self, message: str):
        """Log debug message if debug logger available"""
        if self.debug_logger:
            self.debug_logger(f"[NCUI] {message}")

# =============================================================================
# STANDALONE TESTING
# =============================================================================

def test_ui():
    """Test function for standalone UI testing"""
    def mock_callback(action: str, data: Dict[str, Any]):
        print(f"Callback: {action} with data: {data}")
    
    def mock_logger(message: str):
        print(f"Debug: {message}")
    
    try:
        controller = NCursesUIController(mock_callback, mock_logger)
        
        def ui_wrapper(stdscr):
            if controller.initialize(stdscr):
                controller.add_message({
                    'content': 'Test UI initialized successfully',
                    'type': 'system'
                })
                controller.run()
            else:
                print("UI initialization failed")
        
        curses.wrapper(ui_wrapper)
        
    except Exception as e:
        print(f"UI test error: {e}")
    finally:
        print("UI test complete")

if __name__ == "__main__":
    test_ui()
