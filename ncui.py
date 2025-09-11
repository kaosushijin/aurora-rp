# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - NCurses UI Controller (ncui.py)
Simplified UI management without orchestration logic - business logic moved to orch.py
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
    
    def __init__(self, debug_logger=None, orchestrator_callback=None):
        self.debug_logger = debug_logger
        self.orchestrator_callback = orchestrator_callback  # Interface to orch.py
        
        # UI state only
        self.running = True
        self.input_mode = True
        self.stdscr = None
        
        # UI components
        self.color_manager = ColorManager()
        self.terminal_manager = None
        self.scroll_manager = ScrollManager()
        self.multi_input = MultiLineInput()
        self.input_validator = InputValidator()
        self.current_layout = None
        
        # UI windows
        self.output_window = None
        self.input_window = None
        self.status_window = None
        
        # Display state
        self.current_theme = ColorTheme.CHARCOAL
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
        """Initialize UI components without business logic"""
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
        
        # Check minimum size
        if self.terminal_manager.is_too_small():
            self.terminal_manager.show_too_small_message()
            return
        
        # Get initial layout
        self.current_layout = self.terminal_manager.get_box_layout()
        
        # Initialize UI components
        self.color_manager.init_colors()
        self._update_component_dimensions()
        self._create_windows()
        self._populate_welcome_content()
        self._ensure_cursor_in_input()
        
        self._log_debug(f"UI initialized: {width}x{height}")
    
    def _update_component_dimensions(self) -> None:
        """Update component dimensions from current layout"""
        if not self.current_layout:
            return
        
        # Update multi-input width
        self.multi_input.update_max_width(self.current_layout.terminal_width - 10)
        
        # Update scroll manager height
        self.scroll_manager.update_window_height(
            self.current_layout.output_box.inner_height
        )
    
    def _create_windows(self) -> None:
        """Create UI windows using dynamic coordinates"""
        if not self.current_layout:
            return
        
        try:
            # Create output window
            output_box = self.current_layout.output_box
            self.output_window = curses.newwin(
                output_box.height, output_box.width,
                output_box.top, output_box.left
            )
            
            # Create input window
            input_box = self.current_layout.input_box
            self.input_window = curses.newwin(
                input_box.height, input_box.width,
                input_box.top, input_box.left
            )
            
            # Create status window
            status_box = self.current_layout.status_line
            self.status_window = curses.newwin(
                status_box.height, status_box.width,
                status_box.top, status_box.left
            )
            
            # Draw borders
            self.output_window.box()
            self.input_window.box()
            
            # Enable keypad for special keys
            self.input_window.keypad(True)
            
            self._log_debug("UI windows created successfully")
            
        except curses.error as e:
            self._log_debug(f"Window creation failed: {e}")
            raise

# Chunk 2/3 - ncui.py - UI Main Loop and Input Processing
    
    def _run_ui_loop(self) -> int:
        """Main UI loop - handles only display and input, delegates business logic"""
        while self.running:
            try:
                # Handle terminal resize
                if self._check_and_handle_resize():
                    continue
                
                # Process user input
                user_input = self._get_user_input()
                if user_input is not None:
                    # Send input to orchestrator for processing
                    if self.orchestrator_callback:
                        try:
                            self.orchestrator_callback({
                                "action": "process_input",
                                "input": user_input
                            })
                        except Exception as e:
                            self._log_debug(f"Orchestrator callback error: {e}")
                            self.display_message(f"Error processing input: {e}", "error")
                
                # Refresh display
                self._refresh_all_windows()
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                self._log_debug("Keyboard interrupt received")
                break
            except Exception as e:
                self._log_debug(f"UI loop error: {e}")
                self.display_message(f"UI error: {e}", "error")
        
        return 0
    
    def _get_user_input(self) -> Optional[str]:
        """Handle user input and return completed input string"""
        if not self.input_window:
            return None
        
        try:
            # Get character with timeout
            self.input_window.timeout(100)  # 100ms timeout
            key = self.input_window.getch()
            
            if key == -1:  # No input available
                return None
            
            # Handle special keys
            if key == 27:  # ESC key
                self._handle_escape_sequence()
                return None
            elif key == curses.KEY_RESIZE:
                return None  # Handle in resize check
            elif key in (curses.KEY_UP, curses.KEY_DOWN):
                self._handle_scroll_keys(key)
                return None
            elif key == 127 or key == curses.KEY_BACKSPACE:  # Backspace
                self.multi_input.handle_backspace()
                self._redraw_input_area()
                return None
            elif key == curses.KEY_LEFT:
                self.multi_input.move_cursor_left()
                self._redraw_input_area()
                return None
            elif key == curses.KEY_RIGHT:
                self.multi_input.move_cursor_right()
                self._redraw_input_area()
                return None
            elif key == 10 or key == 13:  # Enter key
                should_submit, content = self.multi_input.handle_enter()
                if should_submit and content.strip():
                    # Validate input before submission
                    if self.input_validator.validate_input(content):
                        self.multi_input.clear()
                        self._redraw_input_area()
                        return content.strip()
                    else:
                        self.display_message("Input validation failed", "error")
                else:
                    self._redraw_input_area()
                return None
            elif 32 <= key <= 126:  # Printable characters
                char = chr(key)
                if self.multi_input.insert_char(char):
                    self._redraw_input_area()
                return None
            
        except curses.error:
            return None
        
        return None
    
    def _handle_escape_sequence(self) -> None:
        """Handle ESC key sequences for commands"""
        # Simple ESC handling - could be extended for command mode
        if self.orchestrator_callback:
            self.orchestrator_callback({
                "action": "escape_pressed"
            })
    
    def _handle_scroll_keys(self, key: int) -> None:
        """Handle up/down arrow keys for scrolling"""
        if key == curses.KEY_UP:
            self.scroll_manager.scroll_up(3)
        elif key == curses.KEY_DOWN:
            self.scroll_manager.scroll_down(3)
        
        self._redraw_output_area()
    
    def _check_and_handle_resize(self) -> bool:
        """Check for terminal resize and handle if needed"""
        if not self.terminal_manager:
            return False
        
        resized, width, height = self.terminal_manager.check_resize()
        if resized:
            self._log_debug(f"Terminal resized to {width}x{height}")
            
            # Check if still too small
            if self.terminal_manager.is_too_small():
                self.terminal_manager.show_too_small_message()
                return True
            
            # Update layout and components
            self.current_layout = self.terminal_manager.get_box_layout()
            self._update_component_dimensions()
            self._recreate_windows()
            self._redraw_all_content()
            return True
        
        return False
    
    def _recreate_windows(self) -> None:
        """Recreate windows after resize"""
        # Clear old windows
        if self.output_window:
            self.output_window.clear()
        if self.input_window:
            self.input_window.clear()
        if self.status_window:
            self.status_window.clear()
        
        # Create new windows
        self._create_windows()
    
    # =============================================================================
    # DISPLAY INTERFACE METHODS (Called by orchestrator)
    # =============================================================================
    
    def display_message(self, content: str, msg_type: str = "system", timestamp: Optional[str] = None) -> None:
        """Display message in output area - called by orchestrator"""
        try:
            # Create display message
            display_msg = DisplayMessage(
                content=content,
                msg_type=msg_type,
                color_manager=self.color_manager,
                current_theme=self.current_theme,
                show_timestamp=self.show_timestamps,
                terminal_width=self.current_layout.terminal_width if self.current_layout else 80,
                timestamp=timestamp
            )
            
            # Add to local buffer
            self.display_buffer.append(display_msg)
            
            # Keep buffer manageable
            if len(self.display_buffer) > 1000:
                self.display_buffer = self.display_buffer[-800:]
            
            # Update scroll manager
            formatted_lines = display_msg.get_formatted_lines()
            self.scroll_manager.add_message_lines(formatted_lines)
            
            # Redraw output
            self._redraw_output_area()
            
        except Exception as e:
            self._log_debug(f"Display message error: {e}")
    
    def display_system_status(self, status_data: Dict[str, Any]) -> None:
        """Display system status in status line - called by orchestrator"""
        try:
            if not self.status_window or not status_data:
                return
            
            # Format status information
            message_count = status_data.get("message_count", 0)
            memory_usage = status_data.get("memory_usage", 0.0)
            story_pressure = status_data.get("story_pressure", 0.0)
            background_processing = status_data.get("background_processing", False)
            
            # Create status line
            status_text = f"Messages: {message_count} | Memory: {memory_usage:.1f}% | Pressure: {story_pressure:.2f}"
            if background_processing:
                status_text += " | Processing..."
            
            # Display in status window
            self.status_window.clear()
            self.status_window.addstr(0, 1, status_text[:self.current_layout.terminal_width-2])
            self.status_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Status display error: {e}")
    
    def set_ui_theme(self, theme_name: str) -> bool:
        """Change UI theme - called by orchestrator"""
        try:
            if theme_name.upper() in ColorTheme.__members__:
                self.current_theme = ColorTheme[theme_name.upper()]
                self._redraw_all_content()
                return True
            return False
        except Exception:
            return False
    
    def shutdown_interface(self) -> None:
        """Shutdown UI cleanly - called by orchestrator"""
        self.running = False
        self._log_debug("UI shutdown requested")

# Chunk 3/3 - ncui.py - Display Management and Helper Functions
    
    # =============================================================================
    # DISPLAY MANAGEMENT METHODS
    # =============================================================================
    
    def _redraw_output_area(self) -> None:
        """Redraw the output area with current scroll position"""
        if not self.output_window or not self.current_layout:
            return
        
        try:
            output_box = self.current_layout.output_box
            
            # Clear inner area
            for y in range(1, output_box.height - 1):
                self.output_window.move(y, 1)
                self.output_window.clrtoeol()
            
            # Get visible lines from scroll manager
            visible_lines = self.scroll_manager.get_visible_lines()
            
            # Display lines
            display_y = 1
            for line_data in visible_lines:
                if display_y >= output_box.height - 1:
                    break
                
                try:
                    # Apply color formatting
                    if hasattr(line_data, 'color_pair') and line_data.color_pair:
                        self.output_window.attron(curses.color_pair(line_data.color_pair))
                    
                    # Display line content
                    line_content = str(line_data.content) if hasattr(line_data, 'content') else str(line_data)
                    max_width = output_box.width - 3
                    
                    if len(line_content) > max_width:
                        line_content = line_content[:max_width-3] + "..."
                    
                    self.output_window.addstr(display_y, 1, line_content)
                    
                    # Remove color formatting
                    if hasattr(line_data, 'color_pair') and line_data.color_pair:
                        self.output_window.attroff(curses.color_pair(line_data.color_pair))
                    
                    display_y += 1
                    
                except curses.error:
                    # Skip lines that don't fit
                    continue
            
            # Redraw border
            self.output_window.box()
            self.output_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Output redraw error: {e}")
    
    def _redraw_input_area(self) -> None:
        """Redraw the input area with current input state"""
        if not self.input_window or not self.current_layout:
            return
        
        try:
            input_box = self.current_layout.input_box
            
            # Clear inner area
            for y in range(1, input_box.height - 1):
                self.input_window.move(y, 1)
                self.input_window.clrtoeol()
            
            # Get input lines for display
            input_lines = self.multi_input.get_display_lines()
            cursor_line, cursor_col = self.multi_input.get_display_cursor_position()
            
            # Display input lines
            display_y = 1
            for i, line in enumerate(input_lines):
                if display_y >= input_box.height - 1:
                    break
                
                try:
                    max_width = input_box.width - 3
                    display_line = line[:max_width] if len(line) > max_width else line
                    self.input_window.addstr(display_y, 1, display_line)
                    display_y += 1
                except curses.error:
                    break
            
            # Position cursor
            try:
                cursor_display_y = cursor_line + 1
                cursor_display_x = min(cursor_col + 1, input_box.width - 2)
                
                if cursor_display_y < input_box.height - 1:
                    self.input_window.move(cursor_display_y, cursor_display_x)
            except curses.error:
                pass
            
            # Redraw border
            self.input_window.box()
            self.input_window.refresh()
            
        except Exception as e:
            self._log_debug(f"Input redraw error: {e}")
    
    def _redraw_all_content(self) -> None:
        """Redraw all UI content after resize or theme change"""
        try:
            self.stdscr.clear()
            self.stdscr.refresh()
            
            self._redraw_output_area()
            self._redraw_input_area()
            
            # Redraw status if available
            if hasattr(self, 'last_status_data'):
                self.display_system_status(self.last_status_data)
            
            self._ensure_cursor_in_input()
            
        except Exception as e:
            self._log_debug(f"Full redraw error: {e}")
    
    def _refresh_all_windows(self) -> None:
        """Refresh all windows for display updates"""
        try:
            if self.output_window:
                self.output_window.refresh()
            if self.input_window:
                self.input_window.refresh()
            if self.status_window:
                self.status_window.refresh()
        except curses.error:
            pass
    
    def _populate_welcome_content(self) -> None:
        """Add welcome message to display"""
        welcome_lines = [
            "DevName RPG Client - Ready",
            "Type your actions and press Enter to interact",
            "Use arrow keys to scroll through conversation history",
            "Press ESC for commands",
            ""
        ]
        
        for line in welcome_lines:
            self.display_message(line, "system")
    
    def _ensure_cursor_in_input(self) -> None:
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
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_ui_stats(self) -> Dict[str, Any]:
        """Get UI statistics for orchestrator"""
        return {
            "display_buffer_size": len(self.display_buffer),
            "terminal_size": f"{self.current_layout.terminal_width}x{self.current_layout.terminal_height}" if self.current_layout else "unknown",
            "current_theme": self.current_theme.name if self.current_theme else "unknown",
            "input_mode": self.input_mode,
            "running": self.running
        }
    
    def clear_display(self) -> None:
        """Clear display buffer and output area - called by orchestrator"""
        self.display_buffer.clear()
        self.scroll_manager.clear_content()
        self._redraw_output_area()
        self._log_debug("Display cleared")
    
    def _log_debug(self, message: str) -> None:
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(f"NCUI: {message}")

# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Basic UI test without orchestrator
    def dummy_callback(data):
        print(f"Test callback received: {data}")
    
    ui = NCursesUIController(orchestrator_callback=dummy_callback)
    
    # Add some test messages
    ui.display_message("Test message 1", "system")
    ui.display_message("Test message 2", "user")
    ui.display_message("Test message 3", "assistant")
    
    print("NCUI Test - Basic UI functionality verified")
    print(f"UI Stats: {ui.get_ui_stats()}")
