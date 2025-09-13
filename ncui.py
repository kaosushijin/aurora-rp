# Chunk 1/6 - ncui.py - Imports and Class Initialization (Comprehensive Fix)

import curses
import time
import threading
from typing import Callable, Dict, Any, List, Optional, Tuple

from uilib import (
    TerminalManager, ColorManager, ScrollManager, MultiLineInput,
    InputValidator, DisplayMessage, ColorTheme,
    MIN_SCREEN_WIDTH, MIN_SCREEN_HEIGHT
)

class NCursesUIController:
    """
    Primary UI Controller for DevName RPG Client
    Orchestration logic moved to orch.py - this handles only display and input.
    FULLY RESTORED: Complete display functionality, cursor positioning, message tracking
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

        # UI components - RESTORED: Complete initialization
        self.multi_input = MultiLineInput()
        self.scroll_manager = ScrollManager(0)  # Updated after layout calculation

        # Display state - RESTORED: Complete message tracking system
        self.display_buffer = []
        self.displayed_message_ids = set()  # CRITICAL: Track displayed messages for deduplication
        self.status_message = "Ready"
        self.processing = False

        # Configuration
        self.debug_mode = bool(debug_logger)

        self._log_debug("UI controller created")

    def _log_debug(self, message: str):
        """Log debug message with NCUI prefix"""
        if self.debug_logger:
            self.debug_logger.debug(f"NCUI: {message}")

    def _log_error(self, message: str):
        """Log error message with NCUI prefix"""
        if self.debug_logger:
            self.debug_logger.error(f"NCUI: {message}")

# Chunk 2/6 - ncui.py - Initialization and Layout Methods (Comprehensive Fix)

    def initialize(self, stdscr) -> bool:
        """Initialize the UI with proper terminal size detection and initial display"""
        try:
            self.stdscr = stdscr

            # RESTORED: Complete terminal initialization sequence
            try:
                height, width = stdscr.getmaxyx()
                self._log_debug(f"Terminal size detected: {width}x{height}")
            except curses.error as e:
                self._log_error(f"Failed to get terminal size: {e}")
                return False

            # Validate minimum terminal size
            if width < MIN_SCREEN_WIDTH or height < MIN_SCREEN_HEIGHT:
                self._log_error(f"Terminal too small: {width}x{height} (minimum: {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT})")
                try:
                    stdscr.clear()
                    msg = f"Terminal too small! Need {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}, got {width}x{height}"
                    stdscr.addstr(height//2, max(0, (width-len(msg))//2), msg)
                    stdscr.refresh()
                except curses.error:
                    pass
                return False

            # RESTORED: Complete color initialization sequence
            if self.color_manager.init_colors():
                self.color_manager.change_theme("classic")
                self._log_debug("Color support initialized with classic theme")
            else:
                self._log_debug("Running without color support")

            # Create terminal manager with validated dimensions
            self.terminal_manager = TerminalManager(stdscr)
            self._log_debug("Terminal manager created")

            # Calculate layout using terminal manager
            self.terminal_manager.check_resize()
            self.current_layout = self.terminal_manager.get_box_layout()
            if not self.current_layout:
                self._log_error("Layout calculation failed")
                return False

            # Update scroll manager with actual output height
            self.scroll_manager = ScrollManager(self.current_layout.output_box.inner_height)
            self._log_debug("Layout calculated successfully")
            # Update multi-input with correct maximum width
            self.multi_input.update_max_width(self.current_layout.input_box.inner_width - 10)  # Account for prompt and margins
            self._log_debug(f"MultiLineInput width set to: {self.current_layout.input_box.inner_width - 10}")

            # Create windows
            if not self._create_windows():
                self._log_error("Window creation failed")
                return False

            # RESTORED: Initialize display with welcome content
            self._populate_welcome_content()

            # RESTORED: Complete initial refresh sequence
            self._refresh_all_windows()
            self._ensure_cursor_in_input()

            self._log_debug("UI initialization complete")
            return True

        except Exception as e:
            self._log_error(f"UI initialization error: {e}")
            return False

    def _create_windows(self) -> bool:
        """Create curses windows using calculated layout"""
        try:
            if not self.current_layout:
                return False

            # Create output window (story display)
            self.output_window = curses.newwin(
                self.current_layout.output_box.height,
                self.current_layout.output_box.width,
                self.current_layout.output_box.top,
                self.current_layout.output_box.left
            )

            # Create input window
            self.input_window = curses.newwin(
                self.current_layout.input_box.height,
                self.current_layout.input_box.width,
                self.current_layout.input_box.top,
                self.current_layout.input_box.left
            )

            # Create status window - FIXED: Use correct attribute name
            self.status_window = curses.newwin(
                self.current_layout.status_line.height,
                self.current_layout.status_line.width,
                self.current_layout.status_line.top,
                self.current_layout.status_line.left
            )

            # Enable keypad for all windows
            self.output_window.keypad(True)
            self.input_window.keypad(True)
            self.status_window.keypad(True)

            self._log_debug("Windows created successfully")
            return True

        except curses.error as e:
            self._log_error(f"Window creation error: {e}")
            return False

    def _populate_welcome_content(self):
        """RESTORED: Add initial welcome message to prevent blank screen"""
        try:
            welcome_time = time.strftime("%H:%M:%S")
            welcome_msg = f"[{welcome_time}] Welcome to DevName RPG Client"
            
            # Create welcome message object
            welcome_display_msg = DisplayMessage(
                content="Welcome to DevName RPG Client",
                msg_type="system"
            )
            welcome_display_msg.timestamp = time.time()
            
            # Add to display buffer with unique ID tracking
            message_id = f"welcome_{welcome_display_msg.timestamp}"
            self.display_buffer.append(welcome_display_msg)
            self.displayed_message_ids.add(message_id)
            
            # Update scroll manager
            self.scroll_manager.update_max_scroll(len(self.display_buffer))
            self.scroll_manager.scroll_to_bottom()
            
            self._log_debug("Welcome content populated")
            
        except Exception as e:
            self._log_error(f"Welcome content population error: {e}")

# Chunk 3/6 - ncui.py - Main Loop and Input Handling (Comprehensive Fix)

    def run(self) -> int:
        """Run interface using curses wrapper - RESTORED: Complete main loop"""
        def _curses_main(stdscr):
            try:
                # Initialize the interface
                if not self.initialize(stdscr):
                    self._log_error("UI initialization failed")
                    return 1

                self.running = True
                self._log_debug("Starting UI main loop")

                while self.running:
                    try:
                        # RESTORED: Process display updates periodically
                        self._process_display_updates()

                        # Get user input with timeout
                        self.input_window.timeout(100)  # 100ms timeout
                        key = self.input_window.getch()

                        if key == -1:  # Timeout, no input
                            continue

                        # Handle special keys
                        if key == 27:  # Escape key
                            self._handle_quit()
                            break
                        elif key == curses.KEY_RESIZE:
                            self._handle_resize()
                            continue
                        elif key in [curses.KEY_PPAGE, curses.KEY_NPAGE]:  # Page Up/Down
                            self._handle_scroll(key)
                            continue

                        # Process input through multi-input handler
                        input_result = self.multi_input.handle_input(key)

                        if input_result.submitted:
                            # User submitted input
                            self._handle_user_input(input_result.content)
                        else:
                            # Just update input display and cursor
                            self._refresh_input_window()
                            self._ensure_cursor_in_input()

                    except curses.error as e:
                        self._log_error(f"Curses error in main loop: {e}")
                        continue
                    except KeyboardInterrupt:
                        self._log_debug("Keyboard interrupt received")
                        break

                return 0  # Success exit code

            except Exception as e:
                self._log_error(f"Main loop error: {e}")
                return 1
            finally:
                self.running = False
                self._cleanup()

        # Use curses wrapper to properly initialize terminal
        try:
            return curses.wrapper(_curses_main)
        except Exception as e:
            self._log_error(f"Curses wrapper failed: {e}")
            return 1

    def _ensure_cursor_in_input(self):
        """CORRECTED: Robust cursor positioning with processing state check"""
        try:
            # CRITICAL: Only position cursor when not processing AND input window exists
            if self.processing or not self.input_window or not self.current_layout:
                return

            # Get cursor position from multi-line input
            cursor_line, cursor_col = self.multi_input.get_cursor_position()

            # Calculate visual position with border offset
            visual_x = 1 + cursor_col  # +1 for left border
            visual_y = 1 + cursor_line  # +1 for top border

            # Clamp to layout boundaries
            max_width = self.current_layout.input_box.inner_width
            max_height = self.current_layout.input_box.inner_height

            visual_x = min(visual_x, max_width)
            visual_y = min(visual_y, max_height)

            # Position cursor with error handling
            try:
                self.input_window.move(visual_y, visual_x)
                curses.curs_set(1)  # Make cursor visible
                curses.doupdate()   # Force screen update
            except curses.error:
                # Fallback positioning
                try:
                    self.input_window.move(1, 1)
                    curses.curs_set(1)
                    curses.doupdate()
                except curses.error:
                    pass

        except Exception as e:
            self._log_error(f"Cursor positioning error: {e}")

    def _handle_user_input(self, user_input: str):
        """CORRECTED: Simplified user input handling with proper flow"""
        try:
            # 1. Clear input field IMMEDIATELY
            self.multi_input.clear()
            self._refresh_input_window()

            # 2. Handle commands locally (these don't go to orchestrator)
            if user_input.startswith('/'):
                if self._handle_command(user_input):
                    self._ensure_cursor_in_input()
                    return

            self._log_debug("Submitting user input")

            # 3. Set processing state and show "Processing..."
            self.processing = True
            self.status_message = "Processing..."
            self._refresh_status_window()

            # 4. Send to orchestrator (which now handles immediate user echo)
            if self.callback_handler:
                result = self.callback_handler("user_input", {"input": user_input})

                # Don't need to handle response here - the background processing
                # in orchestrator will make messages available via get_messages

            else:
                self.status_message = "No orchestrator connection available"
                self.processing = False
                self._refresh_status_window()
                self._ensure_cursor_in_input()

        except Exception as e:
            # Error handling - ensure we reset processing state
            self.processing = False
            self.status_message = f"Input error: {e}"
            self._refresh_status_window()
            self._ensure_cursor_in_input()
            self._log_error(f"Input handling error: {e}")

    def _handle_scroll(self, key: int):
        """Handle page up/down scrolling - FIXED: Use correct ScrollManager methods"""
        try:
            if key == curses.KEY_PPAGE:  # Page Up
                if self.scroll_manager.handle_page_scroll(-1):  # Use handle_page_scroll instead of scroll_up()
                    self._refresh_output_window()
            elif key == curses.KEY_NPAGE:  # Page Down
                if self.scroll_manager.handle_page_scroll(1):   # Use handle_page_scroll instead of scroll_down()
                    self._refresh_output_window()

            self._ensure_cursor_in_input()

        except Exception as e:
            self._log_error(f"Scroll handling error: {e}")

    def _handle_resize(self):
        """Handle terminal resize events"""
        try:
            self._log_debug("Terminal resize detected")

            # Get new terminal size
            curses.update_lines_cols()
            height, width = self.stdscr.getmaxyx()

            self.terminal_manager.check_resize()

            # Check if terminal is too small BEFORE trying to use layout
            if self.terminal_manager.is_too_small():
                self.terminal_manager.show_too_small_message()
                return

            self.current_layout = self.terminal_manager.get_box_layout()

            # Only proceed if we have a valid layout
            if not self.current_layout:
                return

            # Recreate windows with new layout
            self._create_windows()

            # Update scroll manager
            self.scroll_manager = ScrollManager(self.current_layout.output_box.inner_height)

            # Force complete refresh
            self.stdscr.clear()
            self.stdscr.refresh()
            self._refresh_all_windows()
            self._ensure_cursor_in_input()

        except Exception as e:
            self._log_error(f"Resize handling error: {e}")

# Chunk 4/6 - ncui.py - Display and Message Management (Comprehensive Fix)

    def _add_message(self, content: str, message_type: str, message_id: str = None):
        """FINAL: Add message with enhanced deduplication to prevent double echo"""
        try:
            # Generate message ID if not provided
            if not message_id:
                timestamp = time.time()
                message_id = f"{message_type}_{int(timestamp * 1000000)}_{len(content)}"

            # Enhanced deduplication: check both ID and content hash
            content_hash = f"{message_type}_{hash(content) % 100000}"

            # Skip if already displayed by ID or content
            if message_id in self.displayed_message_ids or content_hash in self.displayed_message_ids:
                return

            # Create and add message
            message = DisplayMessage(content=content, msg_type=message_type)
            message.timestamp = time.time()

            self.display_buffer.append(message)
            self.displayed_message_ids.add(message_id)
            self.displayed_message_ids.add(content_hash)

            # Update scroll and refresh
            self.scroll_manager.update_max_scroll(len(self.display_buffer))
            if not self.scroll_manager.in_scrollback:
                self.scroll_manager.auto_scroll_to_bottom()

            self._refresh_output_window()
            self._log_debug(f"Added {message_type} message (ID: {message_id})")

        except Exception as e:
            self._log_error(f"Message addition error: {e}")
    
    def _process_display_updates(self):
        """CORRECTED: Properly handle processing state transitions"""
        try:
            # Check for messages from orchestrator
            if self.callback_handler:
                result = self.callback_handler("get_messages", {"limit": 10})

                if result and result.get("success", False):
                    messages = result.get("messages", [])

                    # Track if we added any new messages
                    new_messages_added = False
                    assistant_message_received = False

                    # Add any new messages to display with deduplication
                    for msg in messages:
                        # Create unique message ID
                        msg_id = msg.get("id")
                        if not msg_id:
                            content = msg.get("content", "")
                            msg_type = msg.get("type", "unknown")
                            timestamp = msg.get("timestamp", time.time())
                            msg_id = f"{msg_type}_{timestamp}_{hash(content) % 10000}"

                        # Only add if we haven't seen this message ID before
                        if msg_id not in self.displayed_message_ids:
                            self._add_message(
                                msg.get("content", ""),
                                msg.get("type", "unknown"),
                                msg_id
                            )
                            new_messages_added = True

                            # Check if this is an assistant message (LLM response)
                            if msg.get("type") == "assistant":
                                assistant_message_received = True

                    # CRITICAL: Clear processing state ONLY when assistant message received
                    if new_messages_added and assistant_message_received and self.processing:
                        self.processing = False
                        self.status_message = "Ready"
                        self._refresh_status_window()
                        self._ensure_cursor_in_input()

        except Exception as e:
            self._log_error(f"Display update error: {e}")
    
    def _refresh_output_window(self):
        """FINAL: Refresh output window with proper scrolling and auto-scroll"""
        try:
            if not self.output_window:
                return

            height, width = self.output_window.getmaxyx()
            display_height = height - 2
            display_width = width - 2

            self.output_window.clear()
            self._draw_borders()  # Use custom themed borders

            # Convert messages to display lines
            all_display_lines = []
            for message in self.display_buffer:
                timestamp = time.strftime("%H:%M:%S", time.localtime(message.timestamp))
                content_width = display_width - 12
                if content_width < 20:
                    content_width = 20

                wrapped_lines = self._wrap_text(message.content, content_width)

                for i, line in enumerate(wrapped_lines):
                    if i == 0:
                        display_line = f"[{timestamp}] {line}"
                    else:
                        display_line = f"           {line}"
                    all_display_lines.append((display_line, message.msg_type))

            # Update scroll manager and ensure auto-scroll works
            self.scroll_manager.update_max_scroll(len(all_display_lines))
            self.scroll_manager.update_window_height(display_height)

            if not self.scroll_manager.in_scrollback:
                self.scroll_manager.auto_scroll_to_bottom()

            # Display visible lines
            start_line = self.scroll_manager.scroll_offset
            visible_lines = all_display_lines[start_line:start_line + display_height]

            for row, (display_line, msg_type) in enumerate(visible_lines):
                if row >= display_height:
                    break

                color_pair = self._get_color_for_message_type(msg_type)

                if len(display_line) > display_width:
                    display_line = display_line[:display_width-3] + "..."

                try:
                    self.output_window.addstr(row + 1, 1, display_line, color_pair)
                except curses.error:
                    pass

            # Show scroll indicators if in scrollback
            if self.scroll_manager.in_scrollback:
                scroll_info = self.scroll_manager.get_scroll_info()
                if scroll_info.get("scroll_needed", False):
                    indicator = f"({scroll_info['offset'] + 1}/{len(all_display_lines)})"
                    try:
                        self.output_window.addstr(0, width - len(indicator) - 1, indicator)
                    except curses.error:
                        pass

            self.output_window.refresh()

        except Exception as e:
            if hasattr(self, '_log_error'):
                self._log_error(f"Error in _refresh_output_window: {e}")

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width, preserving word boundaries"""
        if not text:
            return [""]

        import textwrap

        # Split by existing newlines first to preserve paragraph structure
        paragraphs = text.split('\n')
        wrapped_lines = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                wrapped_lines.append("")  # Preserve empty lines
            else:
                # Wrap each paragraph
                paragraph_lines = textwrap.wrap(
                    paragraph,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                    expand_tabs=False
                )
                wrapped_lines.extend(paragraph_lines if paragraph_lines else [""])

        return wrapped_lines

# Chunk 5/6 - ncui.py - Window Refresh and Drawing Methods (Comprehensive Fix)

    def _refresh_input_window(self):
        """CORRECTED: Input window refresh to show typed text"""
        try:
            if not self.input_window or not self.current_layout:
                return

            self.input_window.clear()

            # Draw input window borders first
            self._draw_input_borders()

            # Get current input content and cursor position
            content = self.multi_input.get_content()
            cursor_line, cursor_col = self.multi_input.get_cursor_position()

            # Display content if any exists
            if content:
                # Split content into lines for display
                content_lines = content.split('\n')

                # Display each line within the available space
                for line_idx, line_content in enumerate(content_lines):
                    display_y = line_idx + 1  # +1 for top border

                    # Only display if within window bounds
                    if display_y < self.current_layout.input_box.height - 1:  # -1 for bottom border
                        try:
                            # Truncate line if too long for width
                            max_width = self.current_layout.input_box.inner_width
                            display_content = line_content[:max_width] if len(line_content) > max_width else line_content

                            # Display the line content
                            if display_content or line_idx == cursor_line:  # Always show cursor line even if empty
                                self.input_window.addstr(display_y, 1, display_content)

                        except curses.error:
                            # If we can't write here, stop trying
                            break
                    else:
                        # No more room for display
                        break

            self.input_window.noutrefresh()

        except Exception as e:
            self._log_error(f"Input window refresh error: {e}")

    def _refresh_status_window(self):
        """CORRECTED: Status window without automatic cursor positioning"""
        try:
            if not self.status_window or not self.current_layout:
                return

            self.status_window.clear()

            # Truncate status message if too long
            max_width = self.current_layout.status_line.inner_width
            status_text = self.status_message[:max_width-2] if len(self.status_message) > max_width-2 else self.status_message

            # Choose status color
            if self.processing:
                status_color = self.color_manager.get_color('system')
            elif "Error" in self.status_message:
                status_color = self.color_manager.get_color('error')
            else:
                status_color = self.color_manager.get_color('user')

            try:
                if status_color and self.color_manager.colors_available:
                    self.status_window.addstr(0, 1, status_text, curses.color_pair(status_color))
                else:
                    self.status_window.addstr(0, 1, status_text)
            except curses.error:
                pass

            self.status_window.noutrefresh()

            # REMOVED: Automatic cursor positioning call that was causing conflicts

        except Exception as e:
            self._log_error(f"Status window refresh error: {e}")

    def _draw_borders(self):
        """RESTORED: Draw borders around windows using current layout"""
        try:
            if not self.current_layout or not self.output_window:
                return

            # Output window border (Story window)
            border_color = self.color_manager.get_color('border')
            
            if border_color and self.color_manager.colors_available:
                self.output_window.attron(curses.color_pair(border_color))

            # Draw border characters
            try:
                # Top border
                self.output_window.hline(0, 0, curses.ACS_HLINE, self.current_layout.output_box.width)
                self.output_window.addch(0, 0, curses.ACS_ULCORNER)
                self.output_window.addch(0, self.current_layout.output_box.width-1, curses.ACS_URCORNER)

                # Side borders
                for y in range(1, self.current_layout.output_box.height-1):
                    self.output_window.addch(y, 0, curses.ACS_VLINE)
                    self.output_window.addch(y, self.current_layout.output_box.width-1, curses.ACS_VLINE)

                # Bottom border
                self.output_window.hline(self.current_layout.output_box.height-1, 0, curses.ACS_HLINE, self.current_layout.output_box.width)
                self.output_window.addch(self.current_layout.output_box.height-1, 0, curses.ACS_LLCORNER)
                self.output_window.addch(self.current_layout.output_box.height-1, self.current_layout.output_box.width-1, curses.ACS_LRCORNER)

                # Add title
                title = "── Story "
                title_x = 2
                if title_x + len(title) < self.current_layout.output_box.width:
                    self.output_window.addstr(0, title_x, title)

            except curses.error:
                pass

            if border_color and self.color_manager.colors_available:
                self.output_window.attroff(curses.color_pair(border_color))

        except Exception as e:
            self._log_error(f"Border drawing error: {e}")

    def _draw_input_borders(self):
        """Draw borders around input window matching output window style"""
        try:
            if not self.current_layout or not self.input_window:
                return

            border_color = self.color_manager.get_color('border')

            if border_color and self.color_manager.colors_available:
                self.input_window.attron(curses.color_pair(border_color))

            # Draw border characters for input window (same as output window)
            try:
                # Top border
                self.input_window.hline(0, 0, curses.ACS_HLINE, self.current_layout.input_box.width)
                self.input_window.addch(0, 0, curses.ACS_ULCORNER)
                self.input_window.addch(0, self.current_layout.input_box.width-1, curses.ACS_URCORNER)

                # Side borders
                for y in range(1, self.current_layout.input_box.height-1):
                    self.input_window.addch(y, 0, curses.ACS_VLINE)
                    self.input_window.addch(y, self.current_layout.input_box.width-1, curses.ACS_VLINE)

                # Bottom border
                self.input_window.hline(self.current_layout.input_box.height-1, 0, curses.ACS_HLINE, self.current_layout.input_box.width)
                self.input_window.addch(self.current_layout.input_box.height-1, 0, curses.ACS_LLCORNER)
                self.input_window.addch(self.current_layout.input_box.height-1, self.current_layout.input_box.width-1, curses.ACS_LRCORNER)

                # Add title
                title = "── Input "
                title_x = 2
                if title_x + len(title) < self.current_layout.input_box.width:
                    self.input_window.addstr(0, title_x, title)

            except curses.error:
                pass

            if border_color and self.color_manager.colors_available:
                self.input_window.attroff(curses.color_pair(border_color))

        except Exception as e:
            self._log_error(f"Input border drawing error: {e}")

    def _refresh_all_windows(self):
        """RESTORED: Refresh all windows with proper sequence and cursor positioning"""
        try:
            self._refresh_output_window()
            self._refresh_input_window()
            self._refresh_status_window()
            
            # Use noutrefresh for all windows, then single doupdate for efficiency
            curses.doupdate()
            
            # Always end with proper cursor positioning
            self._ensure_cursor_in_input()
            
        except Exception as e:
            self._log_error(f"Window refresh error: {e}")

    def scroll_up(self):
        """Public method for scrolling up"""
        try:
            if self.scroll_manager.handle_page_scroll(-1):
                self._refresh_output_window()
                self._ensure_cursor_in_input()
        except Exception as e:
            self._log_error(f"Scroll up error: {e}")

    def scroll_down(self):
        """Public method for scrolling down"""
        try:
            if self.scroll_manager.handle_page_scroll(1):
                self._refresh_output_window()
                self._ensure_cursor_in_input()
        except Exception as e:
            self._log_error(f"Scroll down error: {e}")

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

# Chunk 6/6 - ncui.py - Command Handling and Cleanup Methods (Comprehensive Fix)

    def _handle_command(self, command: str) -> bool:
        """Handle slash commands with complete functionality - UPDATED: Support all 6 themes"""
        try:
            command = command.lower().strip()

            if command == "/quit" or command == "/exit":
                self._handle_quit()
                return True

            elif command == "/clear":
                self.display_buffer.clear()
                self.displayed_message_ids.clear()
                self.scroll_manager = ScrollManager(self.current_layout.output_box.inner_height)
                self._refresh_output_window()
                self._add_message("Display cleared", "system")
                self._refresh_output_window()  # Force immediate display
                return True

            elif command.startswith("/theme"):
                parts = command.split()
                if len(parts) > 1:
                    theme_name = parts[1].lower()
                    # UPDATED: Support all 6 themes instead of hardcoded 3
                    available_themes = self.color_manager.get_available_themes()
                    if theme_name in available_themes:
                        if self.color_manager.change_theme(theme_name):
                            self._refresh_all_windows()
                            self._add_message(f"Theme changed to {theme_name}", "system")
                            self._refresh_output_window()  # Force immediate display
                        else:
                            self._add_message(f"Failed to initialize {theme_name} theme", "error")
                            self._refresh_output_window()  # Force immediate display
                    else:
                        available_list = ", ".join(available_themes)
                        self._add_message(f"Unknown theme: {parts[1]}. Available: {available_list}", "error")
                        self._refresh_output_window()  # Force immediate display
                else:
                    available_themes = self.color_manager.get_available_themes()
                    available_list = ", ".join(available_themes)
                    self._add_message(f"Usage: /theme <{available_list}>", "system")
                    self._refresh_output_window()  # Force immediate display
                return True

            elif command == "/help":
                help_text = [
                    "Available commands:",
                    "/help - Show this help",
                    "/quit - Exit the application",
                    "/clear - Clear message history",
                    "/theme <name> - Change color theme",
                    "  Available themes: classic, dark, bright, nord, solarized, monokai",
                    "/stats - Show system statistics",
                    "/analyze - Trigger immediate analysis"
                ]
                for line in help_text:
                    self._add_message(line, "system")
                    self._refresh_output_window()  # Force immediate display
                return True

            elif command == "/stats":
                if self.callback_handler:
                    result = self.callback_handler("get_stats", {})
                    if result and result.get("success", False):
                        stats = result.get("stats", {})
                        self._add_message("=== System Statistics ===", "system")
                        for key, value in stats.items():
                            self._add_message(f"{key}: {value}", "system")
                            self._refresh_output_window()  # Force immediate display
                    else:
                        self._add_message("Failed to get system statistics", "error")
                        self._refresh_output_window()  # Force immediate display
                else:
                    self._add_message("No orchestrator connection available", "error")
                    self._refresh_output_window()  # Force immediate display
                return True

            elif command == "/analyze":
                if self.callback_handler:
                    result = self.callback_handler("force_analysis", {})
                    if result and result.get("success", False):
                        self._add_message("Analysis triggered", "system")
                        self._refresh_output_window()  # Force immediate display
                    else:
                        self._add_message("Failed to trigger analysis", "error")
                        self._refresh_output_window()  # Force immediate display
                else:
                    self._add_message("No orchestrator connection available", "error")
                    self._refresh_output_window()  # Force immediate display
                return True

            return False

        except Exception as e:
            self._log_error(f"Command handling error: {e}")
            self._add_message(f"Command error: {e}", "error")
            self._refresh_output_window()
            return True
    
    def _cleanup(self):
        """RESTORED: Clean up resources before shutdown"""
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

    def get_input_content(self) -> str:
        """Get current input content"""
        try:
            return self.multi_input.get_content()
        except Exception as e:
            self._log_error(f"Get input content error: {e}")
            return ""

    def clear_input(self):
        """Clear current input"""
        try:
            self.multi_input.clear()
            self._refresh_input_window()
            self._ensure_cursor_in_input()
        except Exception as e:
            self._log_error(f"Clear input error: {e}")

    def add_system_message(self, message: str):
        """Add a system message to display"""
        try:
            self._add_message(message, "system")
            self._refresh_output_window()
        except Exception as e:
            self._log_error(f"Add system message error: {e}")

    def force_refresh(self):
        """Force a complete refresh of all windows"""
        try:
            self._refresh_all_windows()
        except Exception as e:
            self._log_error(f"Force refresh error: {e}")

    def _get_color_for_message_type(self, msg_type: str) -> int:
        """Get color pair for message type"""
        # This method should return appropriate color pairs based on your color manager
        # Placeholder implementation - adjust based on your ColorManager setup
        color_map = {
            'user': curses.color_pair(1),      # User messages
            'assistant': curses.color_pair(2), # LLM responses
            'system': curses.color_pair(3),    # System messages
            'error': curses.color_pair(4),     # Error messages
        }
        return color_map.get(msg_type, curses.color_pair(0))

# RESTORED: Complete NCursesUIController class with full functionality
# - Message display with deduplication
# - Proper cursor positioning
# - Complete input handling
# - Color theme support
# - Command system
# - Terminal resize handling
# - Error handling and logging
# - External interface methods
