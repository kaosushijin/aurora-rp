# Chunk 1/6 - ncui.py - Imports and Class Initialization (Comprehensive Fix)
# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

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

        # UI components
        self.multi_input = MultiLineInput()
        self.scroll_manager = ScrollManager(0)  # Updated after layout calculation

        # REMOVED: display_buffer and displayed_message_ids
        # Messages now come from orchestrator on-demand
        self.status_message = "Ready"
        self.processing = False
        self.processing_started_time = None  # ADD: Timestamp tracking for processing timeout

        # NEW: Message change tracking to prevent unnecessary clearing
        self._last_displayed_messages = []

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
            # Set viewport height for scrolling input
            self.multi_input.set_viewport_height(self.current_layout.input_box.inner_height)
            self._log_debug(f"MultiLineInput viewport height set to: {self.current_layout.input_box.inner_height}")

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
        """FIXED: Conditional welcome with system-only memory cleanup"""
        try:
            # Check if there's meaningful existing memory through orchestrator
            has_meaningful_memory = False
            system_only_memory = False

            if self.callback_handler:
                result = self.callback_handler("get_messages", {"limit": 10})
                if result and result.get("success", False):
                    messages = result.get("messages", [])

                    if messages:
                        # Check message composition
                        user_messages = [msg for msg in messages if msg.get("type") == "user"]
                        assistant_messages = [msg for msg in messages if msg.get("type") == "assistant"]
                        system_messages = [msg for msg in messages if msg.get("type") == "system"]

                        # Meaningful memory = has user/assistant conversation
                        has_meaningful_memory = len(user_messages) > 0 or len(assistant_messages) > 0

                        # System-only memory = only system messages exist
                        system_only_memory = len(system_messages) > 0 and len(user_messages) == 0 and len(assistant_messages) == 0

            # Clear system-only memory if detected
            if system_only_memory and not has_meaningful_memory:
                self._log_debug("Detected system-only memory, clearing for fresh start")
                if self.callback_handler:
                    self.callback_handler("clear_memory", {})
                has_meaningful_memory = False

            # Send appropriate welcome message
            if self.callback_handler:
                if has_meaningful_memory:
                    welcome_message = "Resuming previous session."
                else:
                    welcome_message = "Starting new session. Type /help for commands."

                # Send welcome message as system message through orchestrator
                result = self.callback_handler("add_system_message", {
                    "content": welcome_message,
                    "message_type": "system"
                })

                # Force initial display refresh to show welcome message
                self._process_display_updates()

            self._log_debug("Welcome content populated")

        except Exception as e:
            self._log_error(f"Welcome content population error: {e}")

# Chunk 3/6 - ncui.py - Main Loop and Input Handling (Comprehensive Fix)

    def run(self) -> int:
        """Run interface using curses wrapper - FIXED: Handle too-small terminal state"""
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
                        # FIXED: Check if terminal is too small before processing display updates
                        if self.terminal_manager and self.terminal_manager.is_too_small():
                            # Terminal is too small - skip normal display updates
                            # Just handle resize events and quit commands
                            pass
                        else:
                            # Terminal is big enough - process normal display updates
                            self._process_display_updates()

                        # Get user input with timeout
                        self.input_window.timeout(100) if self.input_window else None

                        # FIXED: Handle input differently based on terminal size
                        if self.terminal_manager and self.terminal_manager.is_too_small():
                            # Terminal too small - only listen for resize and quit
                            try:
                                key = stdscr.getch()  # Use stdscr instead of input_window
                                stdscr.timeout(100)
                            except (curses.error, AttributeError):
                                key = -1
                        else:
                            # Normal input handling
                            key = self.input_window.getch() if self.input_window else -1

                        if key == -1:  # Timeout, no input
                            continue

                        # Handle special keys (these work regardless of terminal size)
                        if key == 27:  # Escape key
                            self._handle_quit()
                            break
                        elif key == curses.KEY_RESIZE:
                            self._handle_resize()
                            continue

                        # FIXED: Only handle normal input if terminal is big enough
                        if self.terminal_manager and not self.terminal_manager.is_too_small():
                            if key in [curses.KEY_PPAGE, curses.KEY_NPAGE]:  # Page Up/Down
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
        """FIXED: More reliable cursor positioning with processing state check"""
        try:
            # CRITICAL: Only position cursor when not processing AND input window exists
            if self.processing or not self.input_window or not self.current_layout:
                return

            # Get cursor position from multi-line input
            cursor_line, cursor_col = self.multi_input.get_cursor_position()

            # Calculate visual position relative to viewport, not absolute buffer position
            scroll_offset = self.multi_input.scroll_offset
            visual_cursor_line = cursor_line - scroll_offset  # Adjust for scroll

            # Calculate visual position with border offset
            visual_x = 1 + cursor_col  # +1 for left border
            visual_y = 1 + visual_cursor_line  # +1 for top border, adjusted for scroll

            # Clamp to layout boundaries
            max_width = self.current_layout.input_box.inner_width
            max_height = self.current_layout.input_box.inner_height

            visual_x = min(visual_x, max_width)
            visual_y = min(visual_y, max_height)

            # Position cursor with error handling
            try:
                self.input_window.move(visual_y, visual_x)
                # Make cursor visible ONLY when not processing
                curses.curs_set(1)
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
        """ENHANCED: Clear message cache when user submits input to ensure immediate echo"""
        try:
            # 1. Exit scrollback mode when user submits input
            if self.scroll_manager.in_scrollback:
                self.scroll_manager.auto_scroll_to_bottom()
                self._log_debug("Exited scrollback mode - user submitted input")

            # 2. Clear input field IMMEDIATELY
            self.multi_input.clear()
            self._refresh_input_window()

            # 3. Handle commands locally (these don't go to orchestrator)
            if user_input.startswith('/'):
                if self._handle_command(user_input):
                    self._ensure_cursor_in_input()
                    return

            self._log_debug("Submitting user input")

            # 4. CRITICAL: Clear message cache BEFORE sending to orchestrator
            # This ensures the user's message will be detected as a change
            self._last_displayed_messages = []

            # 5. Send to orchestrator (which stores user message immediately)
            if self.callback_handler:
                result = self.callback_handler("user_input", {"input": user_input})

                if result and result.get("success", False):
                    # SUCCESS: User message stored in orchestrator memory

                    # 6. Set processing state AFTER orchestrator confirms storage
                    self.processing = True
                    self.processing_started_time = time.time()
                    self.status_message = "Processing..."
                    self._refresh_status_window()

                    # Hide cursor during processing
                    try:
                        curses.curs_set(0)
                    except curses.error:
                        pass

                    # 7. CRITICAL: Force immediate display update to show user message echo
                    # Since we cleared message cache, this will detect the change and refresh
                    self._process_display_updates()

                    # Force screen update
                    curses.doupdate()

                else:
                    # Orchestrator failed - show error immediately
                    error_msg = result.get("error", "Unknown error") if result else "No response"
                    self.status_message = f"Error: {error_msg}"
                    self._refresh_status_window()
                    self._ensure_cursor_in_input()
            else:
                # No orchestrator - show error
                self.status_message = "No orchestrator connection available"
                self._refresh_status_window()
                self._ensure_cursor_in_input()

        except Exception as e:
            # Error handling - ensure we don't get stuck in processing state
            self.processing = False
            self.processing_started_time = None
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
        """Handle terminal resize events - FIXED: Stateless too-small message handling"""
        try:
            self._log_debug("Terminal resize detected")

            # Get new terminal size
            curses.update_lines_cols()
            height, width = self.stdscr.getmaxyx()

            self.terminal_manager.check_resize()

            # Check if terminal is too small BEFORE trying to use layout
            if self.terminal_manager.is_too_small():
                # FIXED: Handle "too small" message through orchestrator instead of terminal_manager
                self._show_terminal_too_small_message(width, height)
                return

            # Terminal is big enough - proceed with normal layout
            self.current_layout = self.terminal_manager.get_box_layout()

            # Only proceed if we have a valid layout
            if not self.current_layout:
                return

            # Recreate windows with new layout
            self._create_windows()

            # Update scroll manager
            self.scroll_manager = ScrollManager(self.current_layout.output_box.inner_height)

            # Update multi-input width
            self.multi_input.update_max_width(self.current_layout.input_box.inner_width - 10)
            self.multi_input.set_viewport_height(self.current_layout.input_box.inner_height)

            # Force complete refresh
            self.stdscr.clear()
            self.stdscr.refresh()
            self._refresh_all_windows()
            self._ensure_cursor_in_input()

        except Exception as e:
            self._log_error(f"Resize handling error: {e}")

# Chunk 4/6 - ncui.py - Display and Message Management (Comprehensive Fix)

    def _process_display_updates(self):
        """FIXED: Only refresh display when content actually changes"""
        try:
            # Check if user is in scrollback mode
            if self.scroll_manager.in_scrollback:
                # User is viewing history - only handle processing state, don't refresh display
                self._handle_processing_state_minimal()
                return

            # Get ALL messages - no limit for full scrollback capability
            messages_to_display = []
            if self.callback_handler:
                result = self.callback_handler("get_messages", {})  # ← REMOVED LIMIT
                if result and result.get("success", False):
                    messages_to_display = result.get("messages", [])

            # CRITICAL: Only refresh if content actually changed
            if self._has_message_content_changed(messages_to_display):
                self._refresh_output_with_messages(messages_to_display)
                self._last_displayed_messages = [
                    {
                        'content': msg.get('content', ''),
                        'type': msg.get('type', ''),
                        'timestamp': msg.get('timestamp', ''),
                        'id': msg.get('id', '')
                    }
                    for msg in messages_to_display
                ]
                self._log_debug(f"Display refreshed - content changed ({len(messages_to_display)} messages)")

            # Handle processing state without triggering display refresh
            self._handle_processing_state_minimal()

        except Exception as e:
            self._log_error(f"Display update error: {e}")

    def _has_message_content_changed(self, new_messages: List[Dict]) -> bool:
        """Check if message content has actually changed"""
        try:
            # First run - always consider changed
            if not self._last_displayed_messages:
                return len(new_messages) > 0

            # Quick check: different number of messages
            if len(new_messages) != len(self._last_displayed_messages):
                return True

            # Deep check: compare each message
            for new_msg, old_msg in zip(new_messages, self._last_displayed_messages):
                if (new_msg.get('content', '') != old_msg.get('content', '') or
                    new_msg.get('type', '') != old_msg.get('type', '') or
                    new_msg.get('timestamp', '') != old_msg.get('timestamp', '') or
                    new_msg.get('id', '') != old_msg.get('id', '')):
                    return True

            # No changes detected
            return False

        except Exception as e:
            self._log_error(f"Message change detection error: {e}")
            return True  # On error, assume change to be safe

    def _handle_processing_state_minimal(self):
        """Handle processing state changes without triggering display refresh"""
        try:
            if not self.processing:
                # Not processing - ensure cursor is positioned
                self._ensure_cursor_in_input()
                return

            # Check for processing timeout
            if (self.processing_started_time and
                time.time() - self.processing_started_time > 30):
                # Timeout occurred
                self._clear_processing_state()
                if self.callback_handler:
                    self.callback_handler("add_system_message", {
                        "content": "Processing timeout - please try again",
                        "message_type": "system"
                    })
                # Force message change detection on next cycle
                self._last_displayed_messages = []
                self._log_debug("Processing timeout - cleared message cache")
                return

            # Check if assistant response arrived (without full message fetch)
            if self.callback_handler:
                result = self.callback_handler("get_messages", {"limit": 1})
                if result and result.get("success", False):
                    messages = result.get("messages", [])
                    if messages:
                        latest_message = messages[-1]
                        if latest_message and latest_message.get("type") == "assistant":
                            # Assistant response received
                            self._clear_processing_state()
                            # Force refresh on next cycle by clearing cache
                            self._last_displayed_messages = []
                            self._log_debug("Assistant response detected - cleared message cache")

        except Exception as e:
            self._log_error(f"Minimal processing state error: {e}")

    def _clear_processing_state(self):
        """FIXED: Clear processing state and reset timestamp"""
        try:
            # Only clear if we're actually in processing state
            if self.processing:
                self.processing = False
                self.processing_started_time = None  # FIXED: Reset timestamp when clearing
                self.status_message = "Ready"
                self._refresh_status_window()

                # CRITICAL: Restore cursor immediately after clearing processing
                self._ensure_cursor_in_input()

                # Force screen update to show cursor
                try:
                    curses.doupdate()
                except curses.error:
                    pass

                self._log_debug("Processing state cleared - UI ready for input")

        except Exception as e:
            self._log_error(f"Processing state clear error: {e}")
    
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

                # Get viewport info from multi_input
                scroll_offset = self.multi_input.scroll_offset
                viewport_height = self.multi_input.viewport_height

                # Calculate which lines to show based on scroll offset
                start_line = scroll_offset
                end_line = min(start_line + viewport_height, len(content_lines))
                visible_lines = content_lines[start_line:end_line]

                # Display visible lines within the available space
                for display_idx, line_content in enumerate(visible_lines):
                    display_y = display_idx + 1  # +1 for top border

                    # Only display if within window bounds
                    if display_y < self.current_layout.input_box.height - 1:  # -1 for bottom border
                        try:
                            # Truncate line if too long for width
                            max_width = self.current_layout.input_box.inner_width
                            display_content = line_content[:max_width] if len(line_content) > max_width else line_content

                            # Display the line content
                            # Calculate the actual buffer line index for this display line
                            actual_line_idx = scroll_offset + display_idx
                            if display_content or actual_line_idx == cursor_line:  # Always show cursor line even if empty
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
        """FIXED: Ensure proper cursor positioning after all window refreshes"""
        try:
            self._refresh_output_window()
            self._refresh_input_window()
            self._refresh_status_window()

            # Use noutrefresh for all windows, then single doupdate for efficiency
            curses.doupdate()

            # CRITICAL: Always end with proper cursor positioning
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
        """FIXED: Use stateless approach but force immediate display updates"""
        try:
            command = command.lower().strip()

            if command == "/quit" or command == "/exit":
                self._handle_quit()
                return True

            elif command == "/clear":
                # Clear through orchestrator instead of local buffer
                if self.callback_handler:
                    result = self.callback_handler("clear_memory", {})
                    if result and result.get("success", False):
                        # Reset scroll manager and CLEAR MESSAGE CACHE
                        self.scroll_manager = ScrollManager(self.current_layout.output_box.inner_height)
                        self._last_displayed_messages = []  # CRITICAL: Clear cache
                        self._process_display_updates()  # Force immediate refresh
                return True

            elif command.startswith("/theme"):
                parts = command.split()
                if len(parts) > 1:
                    theme_name = parts[1].lower()
                    available_themes = self.color_manager.get_available_themes()
                    if theme_name in available_themes:
                        if self.color_manager.change_theme(theme_name):
                            self._refresh_all_windows()
                            # Add system message through orchestrator
                            if self.callback_handler:
                                self.callback_handler("add_system_message", {
                                    "content": f"Theme changed to {theme_name}",
                                    "message_type": "system"
                                })
                                self._process_display_updates()
                        else:
                            if self.callback_handler:
                                self.callback_handler("add_system_message", {
                                    "content": f"Failed to initialize {theme_name} theme",
                                    "message_type": "system"
                                })
                                self._process_display_updates()
                    else:
                        available_list = ", ".join(available_themes)
                        if self.callback_handler:
                            self.callback_handler("add_system_message", {
                                "content": f"Unknown theme: {parts[1]}. Available: {available_list}",
                                "message_type": "system"
                            })
                            self._process_display_updates()
                else:
                    available_themes = self.color_manager.get_available_themes()
                    available_list = ", ".join(available_themes)
                    if self.callback_handler:
                        self.callback_handler("add_system_message", {
                            "content": f"Usage: /theme <{available_list}>",
                            "message_type": "system"
                        })
                        self._process_display_updates()
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
                    "/analyze - Trigger immediate analysis",
                    "/semantic - Show semantic analysis for recent messages"  # NEW
                ]
                for line in help_text:
                    if self.callback_handler:
                        self.callback_handler("add_system_message", {
                            "content": line,
                            "message_type": "system"
                        })
                self._process_display_updates()  # Show all help text
                return True

            elif command == "/stats":
                if self.callback_handler:
                    result = self.callback_handler("get_stats", {})
                    if result and result.get("success", False):
                        stats = result.get("stats", {})
                        self.callback_handler("add_system_message", {
                            "content": "=== System Statistics ===",
                            "message_type": "system"
                        })
                        for key, value in stats.items():
                            self.callback_handler("add_system_message", {
                                "content": f"{key}: {value}",
                                "message_type": "system"
                            })
                        self._process_display_updates()
                    else:
                        self.callback_handler("add_system_message", {
                            "content": "Failed to get system statistics",
                            "message_type": "system"
                        })
                        self._process_display_updates()
                else:
                    # Fallback when no orchestrator
                    pass
                return True

            elif command == "/analyze":
                if self.callback_handler:
                    result = self.callback_handler("force_analysis", {})
                    if result and result.get("success", False):
                        self.callback_handler("add_system_message", {
                            "content": "Analysis triggered",
                            "message_type": "system"
                        })
                    else:
                        self.callback_handler("add_system_message", {
                            "content": "Failed to trigger analysis",
                            "message_type": "system"
                        })
                    self._process_display_updates()
                return True

            elif command == "/semantic":
                # Show semantic analysis for recent messages
                if self.callback_handler:
                    result = self.callback_handler("show_semantic_analysis", {"limit": 10})
                    if result and result.get("success", False):
                        # Results will be displayed as system messages by orchestrator
                        pass
                    else:
                        self.callback_handler("add_system_message", {
                            "content": "Failed to retrieve semantic analysis data",
                            "message_type": "system"
                        })
                    self._process_display_updates()
                return True

            return False

        except Exception as e:
            self._log_error(f"Command handling error: {e}")
            if self.callback_handler:
                self.callback_handler("add_system_message", {
                    "content": f"Command error: {e}",
                    "message_type": "system"
                })
                self._process_display_updates()
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
        """SIMPLIFIED: Display message through orchestrator"""
        try:
            if self.callback_handler:
                self.callback_handler("add_system_message", {
                    "content": content,
                    "message_type": message_type
                })
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
        """SIMPLIFIED: Add system message through orchestrator"""
        try:
            if self.callback_handler:
                self.callback_handler("add_system_message", {
                    "content": message,
                    "message_type": "system"
                })
        except Exception as e:
            self._log_error(f"Add system message error: {e}")

    def force_refresh(self):
        """Force a complete refresh of all windows"""
        try:
            self._refresh_all_windows()
        except Exception as e:
            self._log_error(f"Force refresh error: {e}")

    def _get_color_for_message_type(self, msg_type: str) -> int:
        """FIXED: Return curses color pairs instead of color manager strings"""
        try:
            if not self.color_manager or not self.color_manager.colors_available:
                return 0  # Default no-color

            # Map message types to color manager color names
            color_name_map = {
                'user': 'user',
                'assistant': 'assistant',
                'system': 'system',
                'error': 'error'
            }

            # Get color name and convert to curses color pair
            color_name = color_name_map.get(msg_type, 'user')
            color_id = self.color_manager.get_color(color_name)

            if color_id and isinstance(color_id, int):
                return curses.color_pair(color_id)
            else:
                return 0  # Safe fallback - no color

        except Exception as e:
            self._log_error(f"Color lookup error for {msg_type}: {e}")
            return 0  # Safe fallback

    def _refresh_output_window(self):
        """FIXED: Don't leave cursor in output window after refresh"""
        try:
            if not self.output_window:
                return

            height, width = self.output_window.getmaxyx()
            display_height = height - 2
            display_width = width - 2

            self.output_window.clear()
            self._draw_borders()

            # Get ALL messages from orchestrator - no limit
            messages = []
            if self.callback_handler:
                result = self.callback_handler("get_messages", {})  # ← REMOVED LIMIT
                if result and result.get("success", False):
                    messages = result.get("messages", [])

            # Convert messages to display lines (like .bak version)
            all_display_lines = []
            for message in messages:
                # FIXED: Handle both ISO string timestamps and float timestamps
                raw_timestamp = message.get("timestamp", time.time())
                try:
                    if isinstance(raw_timestamp, str):
                        # Parse ISO format timestamp from orchestrator
                        from datetime import datetime
                        dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                        timestamp_float = dt.timestamp()
                    else:
                        # Already a float timestamp
                        timestamp_float = float(raw_timestamp)

                    timestamp = time.strftime("%H:%M:%S", time.localtime(timestamp_float))
                except (ValueError, OSError):
                    # Fallback to current time if parsing fails
                    timestamp = time.strftime("%H:%M:%S")

                content = message.get("content", "")
                msg_type = message.get("type", "unknown")

                content_width = display_width - 12
                if content_width < 20:
                    content_width = 20

                wrapped_lines = self._wrap_text(content, content_width)

                for i, line in enumerate(wrapped_lines):
                    if i == 0:
                        display_line = f"[{timestamp}] {line}"
                    else:
                        display_line = f"           {line}"
                    all_display_lines.append((display_line, msg_type))

            # Update scroll manager with current line count
            self.scroll_manager.update_max_scroll(len(all_display_lines))
            self.scroll_manager.update_window_height(display_height)

            # Auto-scroll to bottom unless user is in scrollback
            if not self.scroll_manager.in_scrollback:
                self.scroll_manager.auto_scroll_to_bottom()

            # Display visible lines based on scroll position
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

            # CRITICAL: Use noutrefresh instead of refresh to not change cursor position
            self.output_window.noutrefresh()

        except Exception as e:
            self._log_error(f"Error in _refresh_output_window: {e}")

    def _refresh_output_with_messages(self, messages: List[Dict]):
        """STATELESS: Render output window directly from message list without local buffer"""
        try:
            if not self.output_window:
                return

            height, width = self.output_window.getmaxyx()
            display_height = height - 2
            display_width = width - 2

            self.output_window.clear()
            self._draw_borders()

            # Convert messages to display lines
            all_display_lines = []
            for message in messages:
                # Handle timestamp conversion
                raw_timestamp = message.get("timestamp", time.time())
                try:
                    if isinstance(raw_timestamp, str):
                        from datetime import datetime
                        dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                        timestamp_float = dt.timestamp()
                    else:
                        timestamp_float = float(raw_timestamp)
                    timestamp = time.strftime("%H:%M:%S", time.localtime(timestamp_float))
                except (ValueError, OSError):
                    timestamp = time.strftime("%H:%M:%S")

                content = message.get("content", "")
                msg_type = message.get("type", "unknown")

                content_width = display_width - 12
                if content_width < 20:
                    content_width = 20

                wrapped_lines = self._wrap_text(content, content_width)

                for i, line in enumerate(wrapped_lines):
                    if i == 0:
                        display_line = f"[{timestamp}] {line}"
                    else:
                        display_line = f"           {line}"
                    all_display_lines.append((display_line, msg_type))

            # Update scroll manager
            self.scroll_manager.update_max_scroll(len(all_display_lines))
            self.scroll_manager.update_window_height(display_height)

            # Auto-scroll to bottom unless user is in scrollback
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

            # Use noutrefresh to avoid cursor position changes
            self.output_window.noutrefresh()

        except Exception as e:
            self._log_error(f"Error in _refresh_output_with_messages: {e}")

    def _show_terminal_too_small_message(self, current_width: int, current_height: int):
        """FIXED: Display terminal too small message using stateless approach"""
        try:
            # Clear the screen and show a simple message directly
            self.stdscr.clear()

            # Create the message
            msg = f"Terminal needs {MIN_SCREEN_WIDTH}x{MIN_SCREEN_HEIGHT}, got {current_width}x{current_height}"
            help_msg = "Resize your terminal or press Ctrl+C to quit."

            # Calculate position to center the message
            try:
                # Position main message
                msg_y = max(0, current_height // 2)
                msg_x = max(0, (current_width - len(msg)) // 2)

                # Position help message
                help_y = msg_y + 1
                help_x = max(0, (current_width - len(help_msg)) // 2)

                # Display messages directly on stdscr
                self.stdscr.addstr(msg_y, msg_x, msg)
                if help_y < current_height:
                    self.stdscr.addstr(help_y, help_x, help_msg)

            except curses.error:
                # If we can't position it nicely, just put it at 0,0
                try:
                    self.stdscr.addstr(0, 0, msg[:current_width-1])
                except curses.error:
                    pass

            # Refresh to show the message
            self.stdscr.refresh()

            # Hide cursor since we're in an error state
            try:
                curses.curs_set(0)
            except curses.error:
                pass

            self._log_debug(f"Displayed terminal too small message: {current_width}x{current_height}")

        except Exception as e:
            self._log_error(f"Failed to show terminal too small message: {e}")
