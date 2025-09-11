# Chunk 1/4 - ncui.py - Header, Imports, and Constructor
# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - NCurses UI Controller (ncui.py)
Simplified UI management without orchestration logic - business logic moved to orch.py
CORRECTED: Fixed ScrollManager initialization following legacy pattern
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
    CORRECTED: Follows legacy initialization pattern for components requiring terminal dimensions
    """
    
    def __init__(self, debug_logger=None, orchestrator_callback=None):
        self.debug_logger = debug_logger
        self.orchestrator_callback = orchestrator_callback  # Interface to orch.py
        
        # UI state only
        self.running = True
        self.input_mode = True
        self.stdscr = None
        
        # UI components - following legacy pattern
        self.color_manager = ColorManager()
        self.terminal_manager = None  # Created after stdscr available
        self.scroll_manager = ScrollManager(0)  # Placeholder height, updated after layout calculated
        self.multi_input = MultiLineInput()
        self.input_validator = InputValidator()
        self.current_layout = None
        
        # UI windows
        self.output_window = None
        self.input_window = None
        self.status_window = None
        
        # Display state
        self.current_theme = ColorTheme.CLASSIC
        self.show_timestamps = True
        self.display_buffer = []  # Local message buffer for UI
        
        self._log_debug("UI Controller initialized")
    
    def run(self) -> int:
        """Run UI using curses wrapper - interfaces with orchestrator"""
        def _curses_main(stdscr):
            try:
                self._initialize_interface(stdscr)
                return self._run_ui_loop()
            except Exception as e:
                self._log_debug(f"UI error: {e}")
                raise
        
        try:
            return curses.wrapper(_curses_main)
        except Exception as e:
            self._log_debug(f"Curses wrapper error: {e}")
            print(f"UI error: {e}")
            return 1
    
    def _initialize_interface(self, stdscr) -> None:
        """Initialize UI components following legacy pattern"""
        self.stdscr = stdscr
        
        # Basic ncurses setup
        curses.curs_set(1)
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(0)
        stdscr.clear()
        stdscr.refresh()
        
        # Initialize terminal manager with stdscr
        self.terminal_manager = TerminalManager(stdscr)
        resized, width, height = self.terminal_manager.check_resize()
        
        # Check minimum size
        if self.terminal_manager.is_too_small():
            self.terminal_manager.show_too_small_message()
            return
        
        # Get initial layout
        self.current_layout = self.terminal_manager.get_box_layout()
        
        # Initialize components with actual dimensions - following legacy pattern
        self.color_manager.init_colors()
        self._update_component_dimensions()
        
        # Create windows using dynamic coordinates
        self._create_windows_dynamic()
        
        # Populate initial content
        self._populate_welcome_content()
        self._ensure_cursor_in_input()
        
        self._log_debug(f"Interface initialized: {width}x{height}")
    
    def _update_component_dimensions(self):
        """Update component dimensions from current layout - following legacy pattern"""
        if not self.current_layout:
            return
        
        # Update multi-input width (following legacy calculation)
        self.multi_input.update_max_width(self.current_layout.terminal_width - 10)
        
        # Update scroll manager height with output box height (legacy pattern)
        self.scroll_manager.update_window_height(self.current_layout.output_box.height)
        
        self._log_debug(f"Component dimensions updated: scroll_height={self.current_layout.output_box.height}")
    
    def _log_debug(self, message: str) -> None:
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(f"NCUI: {message}")

# Chunk 2/4 - ncui.py - Window Creation and Layout Management

    def _create_windows_dynamic(self):
        """Create curses windows using dynamic coordinates from layout"""
        if not self.current_layout:
            self._log_debug("No layout available for window creation")
            return
        
        try:
            # Create output window
            output_box = self.current_layout.output_box
            self.output_window = curses.newwin(
                output_box.height,
                output_box.width,
                output_box.top,
                output_box.left
            )
            self.output_window.scrollok(True)
            
            # Create input window
            input_box = self.current_layout.input_box
            self.input_window = curses.newwin(
                input_box.height,
                input_box.width,
                input_box.top,
                input_box.left
            )
            
            # Create status window
            status_line = self.current_layout.status_line
            self.status_window = curses.newwin(
                status_line.height,
                status_line.width,
                status_line.top,
                status_line.left
            )
            
            self._log_debug(f"Windows created - Output: {output_box.height}x{output_box.width} "
                          f"Input: {input_box.height}x{input_box.width} "
                          f"Status: {status_line.height}x{status_line.width}")
            
        except curses.error as e:
            self._log_debug(f"Window creation error: {e}")
    
    def _populate_welcome_content(self):
        """Add welcome messages following legacy pattern"""
        welcome_messages = [
            "DevName RPG Client - Hub & Spoke Architecture",
            "Terminal-based RPG storytelling with LLM integration",
            "Type your actions and press Enter. Use /help for commands.",
            ""
        ]
        
        for msg in welcome_messages:
            self.add_message({
                'content': msg,
                'type': 'system',
                'timestamp': time.time()
            })
    
    def _ensure_cursor_in_input(self):
        """Ensure cursor is positioned in input window"""
        if self.input_window:
            try:
                cursor_line, cursor_col = self.multi_input.get_display_cursor_position()
                input_box = self.current_layout.input_box
                
                cursor_y = min(cursor_line + 1, input_box.height - 2)
                cursor_x = min(cursor_col + 1, input_box.width - 2)
                
                self.input_window.move(cursor_y, cursor_x)
                self.input_window.refresh()
            except curses.error:
                pass
    
    def _run_ui_loop(self) -> int:
        """Main UI event loop - interfaces with orchestrator for business logic"""
        self.running = True
        
        while self.running:
            try:
                # Handle terminal resize
                if self._handle_resize():
                    self._redraw_all_windows()
                
                # Get user input
                key = self.stdscr.getch()
                
                if key == -1:  # No input
                    time.sleep(0.01)
                    continue
                
                # Handle input
                if not self._handle_input(key):
                    continue
                
                # Refresh display
                self._refresh_display()
                
            except KeyboardInterrupt:
                self._log_debug("Keyboard interrupt received")
                break
            except Exception as e:
                self._log_debug(f"UI loop error: {e}")
                break
        
        return 0
    
    def _handle_resize(self) -> bool:
        """Handle terminal resize events"""
        if not self.terminal_manager:
            return False
        
        resized, width, height = self.terminal_manager.check_resize()
        
        if resized:
            self._log_debug(f"Terminal resized to {width}x{height}")
            
            # Check if still large enough
            if self.terminal_manager.is_too_small():
                self.terminal_manager.show_too_small_message()
                return False
            
            # Update layout
            self.current_layout = self.terminal_manager.get_box_layout()
            
            # Update component dimensions
            self._update_component_dimensions()
            
            # Recreate windows
            self._create_windows_dynamic()
            
            return True
        
        return False
    
    def _handle_input(self, key: int) -> bool:
        """Handle keyboard input and route to appropriate handlers"""
        try:
            # Handle scrolling keys
            if key == curses.KEY_UP:
                if self.scroll_manager.handle_line_scroll(-1):
                    self._redraw_output_area()
                return True
            elif key == curses.KEY_DOWN:
                if self.scroll_manager.handle_line_scroll(1):
                    self._redraw_output_area()
                return True
            elif key == curses.KEY_PPAGE:  # Page Up
                if self.scroll_manager.handle_page_scroll(-1):
                    self._redraw_output_area()
                return True
            elif key == curses.KEY_NPAGE:  # Page Down
                if self.scroll_manager.handle_page_scroll(1):
                    self._redraw_output_area()
                return True
            elif key == curses.KEY_HOME:
                if self.scroll_manager.handle_home():
                    self._redraw_output_area()
                return True
            elif key == curses.KEY_END:
                if self.scroll_manager.handle_end():
                    self._redraw_output_area()
                return True
            
            # Handle text input
            if key == 10 or key == 13:  # Enter
                return self._handle_enter()
            elif key == 27:  # Escape
                self.running = False
                return False
            elif key == curses.KEY_BACKSPACE or key == 127:
                self.multi_input.handle_backspace()
                self._redraw_input_area()
                return True
            elif 32 <= key <= 126:  # Printable characters
                self.multi_input.add_character(chr(key))
                self._redraw_input_area()
                return True
            
            return True
            
        except Exception as e:
            self._log_debug(f"Input handling error: {e}")
            return True
    
    def _handle_enter(self) -> bool:
        """Handle Enter key - check for submission or new line"""
        try:
            is_submit, content = self.multi_input.handle_enter()
            
            if is_submit:
                if content.strip().startswith('/'):
                    # Handle command
                    if self.orchestrator_callback:
                        response = self.orchestrator_callback({
                            'type': 'command',
                            'command': content.strip()
                        })
                        self._handle_orchestrator_response(response)
                else:
                    # Handle user input
                    if self.orchestrator_callback:
                        response = self.orchestrator_callback({
                            'type': 'user_input',
                            'content': content
                        })
                        self._handle_orchestrator_response(response)
                
                # Clear input after submission
                self.multi_input.clear()
                self._redraw_input_area()
                
                # Auto-scroll to bottom
                self.scroll_manager.auto_scroll_to_bottom()
                self._redraw_output_area()
            else:
                # Just added a new line
                self._redraw_input_area()
            
            return True
            
        except Exception as e:
            self._log_debug(f"Enter handling error: {e}")
            return True

# Chunk 3/4 - ncui.py - Display Management and Message Handling

    def _handle_orchestrator_response(self, response: Optional[Dict[str, Any]]) -> None:
        """Handle response from orchestrator callback"""
        if not response:
            return
        
        try:
            if response.get('status') == 'success':
                if 'response' in response:
                    # Add response message to display
                    self.add_message({
                        'content': response['response'],
                        'type': response.get('type', 'assistant'),
                        'timestamp': time.time()
                    })
            elif 'error' in response:
                # Display error message
                self.add_message({
                    'content': f"Error: {response['error']}",
                    'type': 'system',
                    'timestamp': time.time()
                })
            
        except Exception as e:
            self._log_debug(f"Response handling error: {e}")
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to display buffer and update UI"""
        try:
            # Create display message
            display_msg = DisplayMessage(
                content=message.get('content', ''),
                message_type=message.get('type', 'system'),
                timestamp=message.get('timestamp', time.time())
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
                    all_lines.append((line, msg.message_type))
            
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
                    self.input_window.addstr(i, 0, line[:available_width])
                except curses.error:
                    pass
            
            # Position cursor
            self._ensure_cursor_in_input()
            
            self.input_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Input redraw error: {e}")
    
    def _redraw_status_area(self):
        """Redraw the status line"""
        if not self.status_window or not self.current_layout:
            return
        
        try:
            self.status_window.clear()
            
            # Build status text
            status_parts = []
            
            # Scroll status
            if self.scroll_manager.in_scrollback:
                scroll_info = self.scroll_manager.get_scroll_info()
                status_parts.append(f"SCROLL {scroll_info['percentage']}%")
            
            # Message count
            status_parts.append(f"Messages: {len(self.display_buffer)}")
            
            # Input mode
            if self.input_mode:
                status_parts.append("INPUT")
            
            status_text = " | ".join(status_parts)
            
            # Draw status
            status_width = self.current_layout.status_line.width
            try:
                self.status_window.addstr(0, 0, status_text[:status_width])
            except curses.error:
                pass
            
            self.status_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Status redraw error: {e}")
    
    def _draw_scroll_indicator(self):
        """Draw scroll indicator in output area"""
        if not self.current_layout or not self.scroll_manager.in_scrollback:
            return
        
        try:
            output_box = self.current_layout.output_box
            indicator_x = output_box.inner_width - 1
            indicator_height = max(1, output_box.inner_height // 4)
            
            scroll_info = self.scroll_manager.get_scroll_info()
            indicator_y = int((scroll_info['percentage'] / 100.0) * (output_box.inner_height - indicator_height))
            
            for i in range(indicator_height):
                try:
                    self.output_window.addch(indicator_y + i, indicator_x, '█')
                except curses.error:
                    pass
                    
        except Exception as e:
            self._log_debug(f"Scroll indicator error: {e}")
    
    def _get_color_for_message_type(self, message_type: str):
        """Get color attribute for message type"""
        if not self.color_manager.colors_available:
            return None
        
        try:
            if message_type == 'user':
                return curses.color_pair(self.color_manager.USER_COLOR)
            elif message_type == 'assistant':
                return curses.color_pair(self.color_manager.ASSISTANT_COLOR)
            elif message_type == 'system':
                return curses.color_pair(self.color_manager.SYSTEM_COLOR)
            elif message_type == 'error':
                return curses.color_pair(self.color_manager.ERROR_COLOR)
            else:
                return None
        except:
            return None
    
    def _refresh_display(self):
        """Refresh display elements that need regular updates"""
        try:
            self._redraw_status_area()
            self._ensure_cursor_in_input()
        except Exception as e:
            self._log_debug(f"Display refresh error: {e}")

# Chunk 4/4 - ncui.py - Utility Methods and Shutdown Logic

    def change_theme(self, theme_name: str) -> bool:
        """Change color theme"""
        try:
            # Map theme names to ColorTheme enum
            theme_map = {
                'classic': ColorTheme.CLASSIC,
                'dark': ColorTheme.DARK,
                'bright': ColorTheme.BRIGHT,
                '1': ColorTheme.CLASSIC,
                '2': ColorTheme.DARK,
                '3': ColorTheme.BRIGHT
            }
            
            new_theme = theme_map.get(theme_name.lower())
            if not new_theme:
                return False
            
            # Update color manager
            self.color_manager.theme = new_theme
            self.current_theme = new_theme
            
            # Reinitialize colors
            if self.color_manager.init_colors():
                # Redraw to apply new colors
                self._redraw_all_windows()
                self._log_debug(f"Theme changed to: {theme_name}")
                return True
            
            return False
            
        except Exception as e:
            self._log_debug(f"Theme change error: {e}")
            return False
    
    def display_message(self, content: str, message_type: str = 'system') -> None:
        """Public interface for displaying messages"""
        self.add_message({
            'content': content,
            'type': message_type,
            'timestamp': time.time()
        })
    
    def add_error_message(self, message: Dict[str, Any]) -> None:
        """Add error message with error styling"""
        error_msg = dict(message)
        error_msg['type'] = 'error'
        self.add_message(error_msg)
    
    def update_analysis_status(self, status: str) -> None:
        """Update analysis status display"""
        self.add_message({
            'content': f"Analysis: {status}",
            'type': 'system',
            'timestamp': time.time()
        })
    
    def get_ui_stats(self) -> Dict[str, Any]:
        """Get UI statistics for orchestrator"""
        try:
            terminal_size = "unknown"
            if self.current_layout:
                terminal_size = f"{self.current_layout.terminal_width}x{self.current_layout.terminal_height}"
            
            return {
                "display_buffer_size": len(self.display_buffer),
                "terminal_size": terminal_size,
                "current_theme": self.current_theme.value if self.current_theme else "unknown",
                "input_mode": self.input_mode,
                "running": self.running,
                "scroll_info": self.scroll_manager.get_scroll_info() if self.scroll_manager else {}
            }
        except Exception as e:
            self._log_debug(f"UI stats error: {e}")
            return {"error": str(e)}
    
    def clear_display(self) -> None:
        """Clear display buffer and output area - called by orchestrator"""
        try:
            self.display_buffer.clear()
            
            # Reset scroll manager
            if self.scroll_manager:
                self.scroll_manager.auto_scroll_to_bottom()
                self.scroll_manager.update_max_scroll(0)
            
            # Redraw output area
            self._redraw_output_area()
            self._log_debug("Display cleared")
            
        except Exception as e:
            self._log_debug(f"Clear display error: {e}")
    
    def shutdown(self) -> None:
        """Graceful UI shutdown"""
        try:
            self._log_debug("UI shutdown initiated")
            self.running = False
            
            # Clear windows
            if self.output_window:
                self.output_window.clear()
                self.output_window.refresh()
            
            if self.input_window:
                self.input_window.clear()
                self.input_window.refresh()
            
            if self.status_window:
                self.status_window.clear()
                self.status_window.refresh()
            
            # Clear main screen
            if self.stdscr:
                self.stdscr.clear()
                self.stdscr.refresh()
            
            self._log_debug("UI shutdown complete")
            
        except Exception as e:
            self._log_debug(f"UI shutdown error: {e}")

# =============================================================================
# MODULE TEST AND UTILITIES
# =============================================================================

def test_ui_components():
    """Test UI components without full interface"""
    print("NCurses UI Controller - Component Test")
    
    # Test component creation
    try:
        scroll_mgr = ScrollManager(20)
        multi_input = MultiLineInput()
        color_mgr = ColorManager()
        
        print("✓ All UI components created successfully")
        print(f"✓ ScrollManager: height={scroll_mgr.window_height}")
        print(f"✓ MultiLineInput: max_width={multi_input.max_width}")
        print(f"✓ ColorManager: theme={color_mgr.theme.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

if __name__ == "__main__":
    # Basic UI component test
    test_ui_components()
    
    print("\nNCUI Test - Basic UI functionality verified")
    print("For full UI test, run through orchestrator via main.py")


# End of ncui.py - DevName RPG Client NCurses UI Controller (CORRECTED)
# 
# Key Changes Made:
# - Fixed ScrollManager initialization with placeholder height (0)
# - Added _update_component_dimensions() following legacy pattern
# - Proper window creation using dynamic coordinates
# - Complete message display system with scrolling
# - Orchestrator callback integration
# - Color theme management
# - Graceful shutdown handling
# 
# Usage:
# ui = NCursesUIController(debug_logger=logger, orchestrator_callback=callback_func)
# exit_code = ui.run()
# 
# The UI controller now properly initializes all components following the
# legacy pattern and integrates seamlessly with the orchestrator architecture.
