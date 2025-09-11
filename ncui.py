# Chunk 1/6 - ncui.py - Header, Imports, and Constructor (Method Signature Fixes)

#!/usr/bin/env python3
"""
DevName RPG Client - NCurses UI Controller (ncui.py)
Simplified UI management without orchestration logic - business logic moved to orch.py
FIXED: All method calls corrected to match actual uilib.py signatures
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

        self._log_debug("UI controller created")

# Chunk 2/6 - ncui.py - Initialization and Layout Methods (Method Signature Fixes)
        
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
            self._log_error(f"UI initialization failed: {e}")
            return False
    
    def _update_layout(self):
        """Update layout based on current terminal size"""
        try:
            # FIXED: Use correct method name from uilib.py
            self.current_layout = self.terminal_manager.get_box_layout()

            # FIXED: Update ScrollManager height after layout calculation
            if self.current_layout and hasattr(self.current_layout, 'output_box'):
                # Use output_box.inner_height for content area (excluding borders)
                self.scroll_manager.update_window_height(self.current_layout.output_box.inner_height)

            # Create/recreate windows with new layout
            self._create_windows()

            self._log_debug("Layout updated successfully")

        except Exception as e:
            self._log_error(f"Layout update failed: {e}")
    
    def _create_windows(self):
        """Create or recreate curses windows based on current layout"""
        if not self.current_layout or not self.stdscr:
            return

        try:
            layout = self.current_layout

            # FIXED: Use correct LayoutGeometry structure from uilib.py
            # Create output window (main display area)
            self.output_window = curses.newwin(
                layout.output_box.height,
                layout.output_box.width,
                layout.output_box.top,
                layout.output_box.left
            )

            # Create input window
            self.input_window = curses.newwin(
                layout.input_box.height,
                layout.input_box.width,
                layout.input_box.top,
                layout.input_box.left
            )

            # Create status window
            self.status_window = curses.newwin(
                layout.status_line.height,
                layout.status_line.width,
                layout.status_line.top,
                layout.status_line.left
            )

            # Enable scrolling for output window
            self.output_window.scrollok(True)
            self.output_window.idlok(True)

            # Draw borders
            self._draw_borders()

            self._log_debug("Windows created successfully")

        except Exception as e:
            self._log_error(f"Window creation failed: {e}")
    
    def _initialize_components(self):
        """Initialize UI components after windows are created"""
        try:
            # Reset multi-line input for new layout
            if self.current_layout:
                # FIXED: Use correct layout structure - input_box instead of input_width
                max_width = self.current_layout.input_box.inner_width - 4  # Account for borders
                self.multi_input.update_max_width(max_width)

            # Update scroll manager with proper content height
            if self.current_layout:
                # FIXED: Use correct method name - update_window_height instead of update_height
                self.scroll_manager.update_window_height(self.current_layout.output_box.inner_height - 2)  # Account for borders

            self._log_debug("UI components initialized")

        except Exception as e:
            self._log_error(f"Component initialization failed: {e}")
    
    def _draw_borders(self):
        """Draw window borders"""
        try:
            if self.output_window:
                self.output_window.box()
                self.output_window.addstr(0, 2, " Story ")
            
            if self.input_window:
                self.input_window.box()
                self.input_window.addstr(0, 2, " Input ")
            
            if self.status_window:
                self.status_window.box()
                self.status_window.addstr(0, 2, " Status ")
                
        except curses.error:
            pass  # Ignore curses drawing errors
    
    def _initial_display(self):
        """Show initial display content"""
        try:
            # FIXED: Use correct DisplayMessage constructor parameters
            welcome_msg = DisplayMessage(
                content="Welcome to DevName RPG Client",
                msg_type="system"  # FIXED: parameter name changed from message_type to msg_type
            )
            # FIXED: Add timestamp as attribute after creation (not in constructor)
            welcome_msg.timestamp = time.time()
            self.display_buffer.append(welcome_msg)

            # Show initial status
            self.status_message = "Ready - Type your message and press Enter twice to send"

            # Refresh all displays
            self._refresh_all_windows()

            self._log_debug("Initial display complete")

        except Exception as e:
            self._log_error(f"Initial display failed: {e}")
    
    def _refresh_all_windows(self):
        """Refresh all windows to show current content"""
        try:
            # Refresh output window with current buffer
            self._refresh_output_window()
            
            # Refresh input window with current input
            self._refresh_input_window()
            
            # Refresh status window
            self._refresh_status_window()
            
            # Update screen
            curses.doupdate()
            
        except Exception as e:
            self._log_error(f"Window refresh failed: {e}")

# Chunk 3/6 - ncui.py - Main Run Loop and Input Processing (Method Signature Fixes)

    def run(self) -> int:
        """Run interface using curses wrapper"""
        def _curses_main(stdscr):
            try:
                # Initialize the interface
                if not self.initialize(stdscr):
                    self._log_error("UI initialization failed")
                    return 1

                self.running = True
                self._log_debug("Starting UI main loop")

                # Main event loop
                while self.running:
                    try:
                        # Process user input
                        if self._handle_input():
                            # Input was processed, refresh display
                            self._refresh_all_windows()

                        # Handle any pending display updates
                        self._process_display_updates()

                        # Brief sleep to prevent CPU spinning
                        time.sleep(0.01)

                    except KeyboardInterrupt:
                        self._log_debug("Keyboard interrupt received")
                        break
                    except Exception as e:
                        self._log_error(f"Error in main loop: {e}")
                        # Don't break on individual errors, just log and continue

                self._log_debug("UI main loop ended")
                return 0

            except Exception as e:
                self._log_error(f"Interface error: {e}")
                return 1
            finally:
                self.running = False
                if hasattr(self, '_cleanup'):
                    self._cleanup()

        try:
            self._log_debug("Starting curses wrapper")
            return curses.wrapper(_curses_main)
        except Exception as e:
            self._log_error(f"Curses wrapper error: {e}")
            return 1
    
    def _handle_input(self) -> bool:
        """
        Handle keyboard input and return True if display needs refresh
        """
        try:
            # Get character with timeout
            ch = self.stdscr.getch()

            if ch == -1:  # No input available
                return False

            # Handle special keys
            if ch == curses.KEY_RESIZE:
                self._handle_resize()
                return True
            elif ch == 27:  # Escape key
                return self._handle_escape()
            elif ch in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
                return self._handle_scroll_keys(ch)
            elif ch == 10 or ch == 13:  # Enter key
                return self._handle_enter()
            elif ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
                return self._handle_backspace()
            elif ch == 9:  # Tab key
                return self._handle_tab()
            elif 32 <= ch <= 126:  # Printable characters
                return self._handle_character(chr(ch))
            else:
                # Unknown key, log for debugging
                self._log_debug(f"Unknown key code: {ch}")
                return False

        except Exception as e:
            self._log_error(f"Input handling error: {e}")
            return False
    
    def _handle_character(self, char: str) -> bool:
        """Handle printable character input"""
        try:
            # FIXED: Use correct method name - insert_char instead of insert_character
            self.multi_input.insert_char(char)
            return True
            
        except Exception as e:
            self._log_error(f"Character handling error: {e}")
            return False
    
    def _handle_enter(self) -> bool:
        """Handle Enter key - check for submission or new line"""
        try:
            # Check if input should be submitted (double-enter)
            should_submit, content = self.multi_input.handle_enter()
            
            if should_submit and content.strip():
                # Submit input through orchestrator callback
                self._submit_user_input(content.strip())
                return True
            else:
                # Just added a new line, refresh input display
                return True
                
        except Exception as e:
            self._log_error(f"Enter key handling error: {e}")
            return False
    
    def _handle_backspace(self) -> bool:
        """Handle backspace key"""
        try:
            self.multi_input.handle_backspace()
            return True
            
        except Exception as e:
            self._log_error(f"Backspace handling error: {e}")
            return False
    
    def _handle_tab(self) -> bool:
        """Handle tab key - could be used for autocomplete in future"""
        try:
            # FIXED: Use correct method name - insert_char instead of insert_character
            self.multi_input.insert_char("    ")
            return True
            
        except Exception as e:
            self._log_error(f"Tab handling error: {e}")
            return False
    
    def _handle_escape(self) -> bool:
        """Handle escape key - show options or exit"""
        try:
            # For now, treat as exit request
            self._log_debug("Escape key pressed - requesting shutdown")
            
            # Call orchestrator to handle shutdown
            if self.callback_handler:
                self.callback_handler("shutdown", {})
            
            self.running = False
            return True
            
        except Exception as e:
            self._log_error(f"Escape key handling error: {e}")
            return False
    
    def _handle_scroll_keys(self, key: int) -> bool:
        """Handle scrolling keys"""
        try:
            # FIXED: Use correct ScrollManager method names from uilib.py
            if key == curses.KEY_UP:
                self.scroll_manager.handle_line_scroll(-1)
            elif key == curses.KEY_DOWN:
                self.scroll_manager.handle_line_scroll(1)
            elif key == curses.KEY_PPAGE:  # Page Up
                self.scroll_manager.handle_page_scroll(-1)
            elif key == curses.KEY_NPAGE:  # Page Down
                self.scroll_manager.handle_page_scroll(1)

            return True

        except Exception as e:
            self._log_error(f"Scroll key handling error: {e}")
            return False
    
    def _handle_resize(self):
        """Handle terminal resize"""
        try:
            self._log_debug("Terminal resize detected")
            
            # Update layout for new terminal size
            self._update_layout()
            
            # Redraw everything
            self.stdscr.clear()
            self._refresh_all_windows()
            
            self._log_debug("Resize handling complete")
            
        except Exception as e:
            self._log_error(f"Resize handling error: {e}")
    
    def _submit_user_input(self, content: str):
        """Submit user input through orchestrator callback"""
        try:
            self._log_debug("Submitting user input")
            
            # Clear the input area
            self.multi_input.clear()
            
            # Set processing state
            self.processing = True
            self.status_message = "Processing..."
            
            # Call orchestrator to process input
            if self.callback_handler:
                result = self.callback_handler("user_input", {"input": content})
                
                if result and result.get("success", False):
                    # Add response to display
                    response = result.get("response", "")
                    if response:
                        self._add_message(content, "user")
                        self._add_message(response, "assistant")
                    
                    self.status_message = "Ready"
                else:
                    error_msg = result.get("error", "Unknown error") if result else "No response from orchestrator"
                    self._add_message(f"Error: {error_msg}", "error")
                    self.status_message = "Error occurred"
            else:
                self._log_error("No orchestrator callback available")
                self.status_message = "Error: No orchestrator"
            
        except Exception as e:
            self._log_error(f"Input submission error: {e}")
            self.status_message = "Error occurred"
        finally:
            self.processing = False

# Chunk 4/6 - ncui.py - Display and Message Management (Method Signature Fixes)

    def _add_message(self, content: str, message_type: str):
        """Add message to display buffer"""
        try:
            # FIXED: Use correct DisplayMessage constructor and add timestamp after
            message = DisplayMessage(
                content=content,
                msg_type=message_type  # FIXED: Use msg_type parameter name
            )
            # FIXED: Add timestamp as attribute after creation
            message.timestamp = time.time()

            self.display_buffer.append(message)

            # Update scroll manager with new content
            self.scroll_manager.update_max_scroll(len(self.display_buffer))

            # Auto-scroll to bottom for new messages
            self.scroll_manager.scroll_to_bottom()

            self._log_debug(f"Added {message_type} message")

        except Exception as e:
            self._log_error(f"Message addition error: {e}")
    
    def _process_display_updates(self):
        """Process any pending display updates from orchestrator"""
        try:
            # Check for messages from orchestrator
            if self.callback_handler:
                result = self.callback_handler("get_messages", {"limit": 10})
                
                if result and result.get("success", False):
                    messages = result.get("messages", [])
                    
                    # Add any new messages to display
                    for msg in messages:
                        if msg not in self.display_buffer:
                            self._add_message(
                                msg.get("content", ""),
                                msg.get("type", "unknown")
                            )
            
        except Exception as e:
            self._log_error(f"Display update error: {e}")
    
    def _refresh_output_window(self):
        """Refresh the output window with current message buffer"""
        try:
            if not self.output_window or not self.current_layout:
                return

            # Clear the window
            self.output_window.clear()
            self._draw_borders()

            # FIXED: Use correct layout structure - output_box instead of output_height/width
            # Calculate display area (account for borders)
            display_height = self.current_layout.output_box.inner_height
            display_width = self.current_layout.output_box.inner_width

            # Get visible messages based on scroll position
            start_line = self.scroll_manager.scroll_offset
            visible_messages = self.display_buffer[start_line:start_line + display_height]

            # Display messages
            y_pos = 1  # Start after top border
            for message in visible_messages:
                if y_pos >= display_height + 1:  # Stop before bottom border
                    break

                # FIXED: Handle timestamp attribute properly
                if hasattr(message, 'timestamp') and message.timestamp:
                    timestamp_str = time.strftime("%H:%M:%S", time.localtime(message.timestamp))
                else:
                    timestamp_str = time.strftime("%H:%M:%S")

                # FIXED: Use msg_type attribute name
                message_type = getattr(message, 'msg_type', 'user')

                # Choose color based on message type - FIXED: Use get_color instead of get_color_pair
                color_pair = self.color_manager.get_color(message_type)

                # Wrap long messages
                wrapped_lines = self._wrap_text(message.content, display_width - 12)  # Account for timestamp

                for i, line in enumerate(wrapped_lines):
                    if y_pos >= display_height + 1:
                        break

                    try:
                        if i == 0:
                            # First line includes timestamp
                            display_line = f"[{timestamp_str}] {line}"
                        else:
                            # Continuation lines are indented
                            display_line = f"           {line}"

                        # Truncate if too long
                        if len(display_line) > display_width:
                            display_line = display_line[:display_width-3] + "..."

                        self.output_window.addstr(y_pos, 1, display_line, color_pair)
                        y_pos += 1

                    except curses.error:
                        # Ignore drawing errors (e.g., writing outside window)
                        break

            # Show scroll indicator if needed
            if len(self.display_buffer) > display_height:
                scroll_info = f"({self.scroll_manager.scroll_offset + 1}/{len(self.display_buffer)})"
                try:
                    self.output_window.addstr(0, display_width - len(scroll_info) - 1, scroll_info)
                except curses.error:
                    pass

            self.output_window.noutrefresh()

        except Exception as e:
            self._log_error(f"Output window refresh error: {e}")
    
    def _refresh_input_window(self):
        """Refresh the input window with current input content"""
        try:
            if not self.input_window or not self.current_layout:
                return

            # Clear the window
            self.input_window.clear()
            self._draw_borders()

            # FIXED: Use correct layout structure - input_box instead of input_height/width
            # Get input display area
            display_height = self.current_layout.input_box.inner_height
            display_width = self.current_layout.input_box.inner_width

            # FIXED: Get current input lines with required parameters
            input_lines = self.multi_input.get_display_lines(display_width, display_height)
            
            # FIXED: Get cursor position from direct attributes instead of method call
            cursor_pos = (self.multi_input.cursor_line, self.multi_input.cursor_col)

            # Display input lines
            y_pos = 1
            for i, line in enumerate(input_lines):
                if y_pos >= display_height + 1:
                    break

                try:
                    # Truncate line if too long
                    display_line = line[:display_width] if len(line) > display_width else line
                    self.input_window.addstr(y_pos, 1, display_line)
                    y_pos += 1

                except curses.error:
                    break

            # Position cursor
            try:
                cursor_y = cursor_pos[0] + 1  # Account for border
                cursor_x = cursor_pos[1] + 1  # Account for border

                if 1 <= cursor_y <= display_height and 1 <= cursor_x <= display_width:
                    self.input_window.move(cursor_y, cursor_x)

            except curses.error:
                pass

            self.input_window.noutrefresh()

        except Exception as e:
            self._log_error(f"Input window refresh error: {e}")
    
    def _refresh_status_window(self):
        """Refresh the status window with current status"""
        try:
            if not self.status_window or not self.current_layout:
                return

            # Clear the window
            self.status_window.clear()
            self._draw_borders()

            # FIXED: Use correct layout structure - status_line instead of status_width
            # Display status message
            status_width = self.current_layout.status_line.inner_width

            # Truncate status if too long
            display_status = self.status_message
            if len(display_status) > status_width:
                display_status = display_status[:status_width-3] + "..."

            try:
                # Add processing indicator
                if self.processing:
                    display_status = f"Processing... {display_status}"

                self.status_window.addstr(1, 1, display_status)

            except curses.error:
                pass

            self.status_window.noutrefresh()

        except Exception as e:
            self._log_error(f"Status window refresh error: {e}")
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to fit within specified width"""
        if not text or width <= 0:
            return [""]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            
            if len(test_line) <= width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                
                # Handle very long words
                if len(word) > width:
                    # Split long word
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]

# Chunk 5/6 - ncui.py - Command Handling and Utility Methods (Method Signature Fixes)

    def _handle_command(self, command: str) -> bool:
        """Handle special commands starting with /"""
        try:
            command = command.lower().strip()
            
            if command == "/help":
                self._show_help()
                return True
            elif command == "/clear":
                self._clear_display()
                return True
            elif command == "/stats":
                self._show_stats()
                return True
            elif command == "/quit" or command == "/exit":
                self._handle_quit()
                return True
            elif command.startswith("/theme"):
                parts = command.split()
                if len(parts) > 1:
                    self._change_theme(parts[1])
                else:
                    self._list_themes()
                return True
            elif command == "/analyze":
                self._trigger_analysis()
                return True
            else:
                self._add_message(f"Unknown command: {command}. Type /help for available commands.", "error")
                return True
                
        except Exception as e:
            self._log_error(f"Command handling error: {e}")
            return False
    
    def _show_help(self):
        """Display help information"""
        try:
            help_text = [
                "Available Commands:",
                "/help - Show this help message",
                "/clear - Clear the display",
                "/stats - Show system statistics",
                "/theme [name] - Change color theme or list themes",
                "/analyze - Trigger immediate analysis",
                "/quit or /exit - Exit the application",
                "",
                "Input Controls:",
                "Enter - New line (double-enter to submit)",
                "Backspace - Delete character",
                "Up/Down arrows - Scroll through messages",
                "Page Up/Down - Scroll faster",
                "Escape - Exit application"
            ]
            
            for line in help_text:
                self._add_message(line, "system")
                
            self._log_debug("Help displayed")
            
        except Exception as e:
            self._log_error(f"Help display error: {e}")
    
    def _clear_display(self):
        """Clear the display buffer"""
        try:
            self.display_buffer.clear()
            self.scroll_manager.update_max_scroll(0)
            self._add_message("Display cleared.", "system")
            
            self._log_debug("Display cleared by user")
            
        except Exception as e:
            self._log_error(f"Display clear error: {e}")
    
    def _show_stats(self):
        """Request and display system statistics"""
        try:
            if self.callback_handler:
                result = self.callback_handler("get_stats", {})
                
                if result and result.get("success", False):
                    stats = result.get("stats", {})
                    
                    self._add_message("System Statistics:", "system")
                    self._add_message(f"Messages: {stats.get('message_count', 0)}", "system")
                    self._add_message(f"Phase: {stats.get('phase', 'unknown')}", "system")
                    self._add_message(f"Analysis in progress: {stats.get('analysis_in_progress', False)}", "system")
                    
                    # Show memory stats if available
                    memory_stats = stats.get("memory", {})
                    if memory_stats:
                        self._add_message(f"Memory usage: {memory_stats.get('usage', 'unknown')}", "system")
                    
                else:
                    self._add_message("Failed to retrieve statistics.", "error")
            else:
                self._add_message("No orchestrator connection available.", "error")
                
            self._log_debug("Statistics displayed")
            
        except Exception as e:
            self._log_error(f"Statistics display error: {e}")
    
    def _change_theme(self, theme_name: str):
        """Change the color theme"""
        try:
            # Get available themes
            available_themes = self.color_manager.get_available_themes()
            
            if theme_name in available_themes:
                self.color_manager.set_theme(theme_name)
                self._add_message(f"Theme changed to: {theme_name}", "system")
                
                # Refresh display with new colors
                self._refresh_all_windows()
                
                self._log_debug(f"Theme changed to: {theme_name}")
            else:
                self._add_message(f"Unknown theme: {theme_name}", "error")
                self._list_themes()
                
        except Exception as e:
            self._log_error(f"Theme change error: {e}")
    
    def _list_themes(self):
        """List available color themes"""
        try:
            themes = self.color_manager.get_available_themes()
            current_theme = self.color_manager.current_theme.name
            
            self._add_message("Available themes:", "system")
            for theme in themes:
                marker = " (current)" if theme == current_theme else ""
                self._add_message(f"  {theme}{marker}", "system")
                
        except Exception as e:
            self._log_error(f"Theme listing error: {e}")
    
    def _trigger_analysis(self):
        """Trigger immediate analysis"""
        try:
            if self.callback_handler:
                result = self.callback_handler("analyze_now", {})
                
                if result and result.get("success", False):
                    self._add_message("Analysis triggered successfully.", "system")
                else:
                    error_msg = result.get("error", "Unknown error") if result else "No response"
                    self._add_message(f"Analysis failed: {error_msg}", "error")
            else:
                self._add_message("No orchestrator connection available.", "error")
                
            self._log_debug("Analysis trigger requested")
            
        except Exception as e:
            self._log_error(f"Analysis trigger error: {e}")
    
    def _handle_quit(self):
        """Handle quit command"""
        try:
            self._log_debug("Quit command received")
            
            # Call orchestrator shutdown
            if self.callback_handler:
                self.callback_handler("shutdown", {})
            
            self.running = False
            
        except Exception as e:
            self._log_error(f"Quit handling error: {e}")
    
    def _cleanup(self):
        """Clean up resources before shutdown"""
        try:
            self._log_debug("Starting UI cleanup")
            
            # Clear any pending refresh operations
            if self.stdscr:
                try:
                    self.stdscr.clear()
                    self.stdscr.refresh()
                except curses.error:
                    pass
            
            # Reset cursor
            try:
                curses.curs_set(1)
            except curses.error:
                pass
            
            self._log_debug("UI cleanup complete")
            
        except Exception as e:
            self._log_error(f"Cleanup error: {e}")
    
    def shutdown(self):
        """External shutdown method"""
        try:
            self._log_debug("External shutdown requested")
            self.running = False
            
        except Exception as e:
            self._log_error(f"Shutdown error: {e}")
    
    def display_message(self, content: str, message_type: str = "system"):
        """External method to display a message"""
        try:
            self._add_message(content, message_type)
            self._refresh_all_windows()
            
        except Exception as e:
            self._log_error(f"External message display error: {e}")
    
    def set_status(self, status: str):
        """External method to set status message"""
        try:
            self.status_message = status
            self._refresh_status_window()
            
        except Exception as e:
            self._log_error(f"Status setting error: {e}")
    
    def is_running(self) -> bool:
        """Check if UI is currently running"""
        return self.running

# Chunk 6/6 - ncui.py - Debug Logging Helper Methods and Module Exports (Method Signature Fixes)

    def _log_debug(self, message: str):
        """
        Standardized debug logging with null safety
        Uses method pattern: self.debug_logger.debug(message, "NCUI")
        """
        if self.debug_logger:
            self.debug_logger.debug(message, "NCUI")
    
    def _log_error(self, message: str):
        """
        Standardized error logging with null safety
        Uses method pattern: self.debug_logger.error(message, "NCUI")
        """
        if self.debug_logger:
            self.debug_logger.error(message, "NCUI")
    
    def _log_system(self, message: str):
        """
        Standardized system logging with null safety
        Uses method pattern: self.debug_logger.system(message)
        """
        if self.debug_logger:
            self.debug_logger.system(message)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'NCursesUIController'
]
