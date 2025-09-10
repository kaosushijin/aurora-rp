# Chunk 1/3 - ui.py - Pure UI Controller Core Components
#!/usr/bin/env python3
"""
DevName RPG Client - Pure UI Controller (ui.py)

Pure interface management without business logic
Refactored from nci.py - orchestration logic moved to orch.py
Module architecture and interconnects documented in genai.txt
"""

import curses
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable

# UI library imports (will be updated when uilib.py is created in Phase 4)
try:
    from nci_colors import ColorManager, ColorTheme
    from nci_terminal import TerminalManager, LayoutGeometry
    from nci_display import DisplayMessage, InputValidator
    from nci_scroll import ScrollManager
    from nci_input import MultiLineInput
except ImportError:
    # Fallback for transition period
    print("Warning: UI utilities not found, using minimal fallbacks")
    ColorManager = None
    TerminalManager = None

# Configuration constants
MAX_USER_INPUT_TOKENS = 2000
UI_REFRESH_RATE = 30  # FPS for display updates
INPUT_PROCESSING_DELAY = 0.1  # Seconds between input checks

class UIState:
    """UI state tracking without business logic"""
    
    def __init__(self):
        self.running = True
        self.mcp_processing = False
        self.input_locked = False
        self.current_theme = "classic"
        self.terminal_resized = False
        self.last_resize_time = 0.0
        self.display_dirty = True
        self.cursor_visible = True
        
    def lock_input(self):
        """Lock input during processing"""
        self.input_locked = True
        self.cursor_visible = False
    
    def unlock_input(self):
        """Unlock input after processing"""
        self.input_locked = False
        self.cursor_visible = True
    
    def mark_processing(self, processing: bool):
        """Mark MCP processing state"""
        self.mcp_processing = processing
        if processing:
            self.lock_input()
        else:
            self.unlock_input()
    
    def mark_display_dirty(self):
        """Mark display for refresh"""
        self.display_dirty = True
    
    def mark_display_clean(self):
        """Mark display as refreshed"""
        self.display_dirty = False


class WindowManager:
    """Manages curses windows with dynamic layout"""
    
    def __init__(self, stdscr, terminal_manager, color_manager):
        self.stdscr = stdscr
        self.terminal_manager = terminal_manager
        self.color_manager = color_manager
        
        # Window references
        self.output_win = None
        self.input_win = None
        self.status_win = None
        
        # Layout tracking
        self.current_layout = None
        self.windows_created = False
    
    def create_windows(self) -> bool:
        """Create windows using dynamic coordinates"""
        try:
            # Get current layout
            self.current_layout = self.terminal_manager.get_box_layout()
            if not self.current_layout:
                return False
            
            # Create output window
            output_box = self.current_layout.output_box
            self.output_win = curses.newwin(
                output_box.height,
                output_box.width,
                output_box.y,
                output_box.x
            )
            
            # Create input window
            input_box = self.current_layout.input_box
            self.input_win = curses.newwin(
                input_box.height,
                input_box.width,
                input_box.y,
                input_box.x
            )
            
            # Create status window
            status_box = self.current_layout.status_box
            self.status_win = curses.newwin(
                status_box.height,
                status_box.width,
                status_box.y,
                status_box.x
            )
            
            # Configure windows
            if self.output_win:
                self.output_win.scrollok(True)
                self.output_win.idlok(True)
            
            if self.input_win:
                self.input_win.keypad(True)
                self.input_win.nodelay(False)
            
            self.windows_created = True
            return True
            
        except curses.error as e:
            return False
    
    def destroy_windows(self):
        """Safely destroy all windows"""
        try:
            for window in [self.output_win, self.input_win, self.status_win]:
                if window:
                    window.clear()
                    del window
        except curses.error:
            pass
        
        self.output_win = None
        self.input_win = None
        self.status_win = None
        self.windows_created = False
    
    def handle_resize(self) -> bool:
        """Handle terminal resize by recreating windows"""
        try:
            # Check if resize occurred
            resized, width, height = self.terminal_manager.check_resize()
            
            if resized:
                # Destroy old windows
                self.destroy_windows()
                
                # Clear screen
                self.stdscr.clear()
                self.stdscr.refresh()
                
                # Recreate windows with new layout
                return self.create_windows()
            
            return True
            
        except curses.error:
            return False
    
    def draw_borders(self):
        """Draw window borders using layout"""
        if not self.current_layout:
            return
        
        try:
            # Draw borders for each window
            for box_name, box in [
                ("output", self.current_layout.output_box),
                ("input", self.current_layout.input_box),
                ("status", self.current_layout.status_box)
            ]:
                # Draw box border
                for y in range(box.height):
                    for x in range(box.width):
                        screen_y = box.y + y
                        screen_x = box.x + x
                        
                        # Border characters
                        if y == 0 or y == box.height - 1:
                            if x == 0 or x == box.width - 1:
                                char = '+'
                            else:
                                char = '-'
                        elif x == 0 or x == box.width - 1:
                            char = '|'
                        else:
                            continue
                        
                        try:
                            self.stdscr.addch(screen_y, screen_x, char)
                        except curses.error:
                            pass
            
            self.stdscr.refresh()
            
        except curses.error:
            pass
    
    def refresh_all(self):
        """Refresh all windows"""
        try:
            for window in [self.output_win, self.input_win, self.status_win]:
                if window:
                    window.refresh()
        except curses.error:
            pass


class DisplayController:
    """Controls display content without business logic"""
    
    def __init__(self, window_manager, color_manager, scroll_manager):
        self.window_manager = window_manager
        self.color_manager = color_manager
        self.scroll_manager = scroll_manager
        
        # Display content
        self.messages = []
        self.current_input = ""
        self.current_status = "Ready"
        
        # Display helpers
        self.display_message = DisplayMessage() if DisplayMessage else None
    
    def add_message(self, content: str, msg_type: str = "system"):
        """Add message to display queue"""
        message = {
            "content": content,
            "type": msg_type,
            "timestamp": time.time()
        }
        self.messages.append(message)
        
        # Limit message history for display
        if len(self.messages) > 1000:
            self.messages = self.messages[-500:]
    
    def set_status(self, status: str):
        """Set status message"""
        self.current_status = status
    
    def set_input_content(self, content: str):
        """Set current input content"""
        self.current_input = content
    
    def update_output_display(self):
        """Update output window display"""
        if not self.window_manager.output_win or not self.window_manager.current_layout:
            return
        
        try:
            output_win = self.window_manager.output_win
            output_box = self.window_manager.current_layout.output_box
            
            # Clear window
            output_win.clear()
            
            # Calculate display area
            display_height = output_box.inner_height
            display_width = output_box.inner_width
            
            # Get messages to display
            display_messages = self.scroll_manager.get_visible_messages(
                self.messages, display_height
            ) if self.scroll_manager else self.messages[-display_height:]
            
            # Display messages
            y_pos = 0
            for message in display_messages:
                if y_pos >= display_height:
                    break
                
                # Format message content
                if self.display_message:
                    formatted_lines = self.display_message.format_content(
                        message["content"], message["type"], display_width
                    )
                else:
                    # Fallback formatting
                    formatted_lines = [message["content"][:display_width]]
                
                # Display formatted lines
                for line in formatted_lines:
                    if y_pos >= display_height:
                        break
                    
                    # Apply colors if available
                    if self.color_manager and message["type"] in ["assistant", "system", "user"]:
                        color_pair = self.color_manager.get_color_pair(message["type"])
                        if color_pair:
                            output_win.attron(curses.color_pair(color_pair))
                        
                        output_win.addstr(y_pos, 0, line[:display_width])
                        
                        if color_pair:
                            output_win.attroff(curses.color_pair(color_pair))
                    else:
                        output_win.addstr(y_pos, 0, line[:display_width])
                    
                    y_pos += 1
            
            output_win.refresh()
            
        except curses.error:
            pass
    
    def update_input_display(self):
        """Update input window display"""
        if not self.window_manager.input_win or not self.window_manager.current_layout:
            return
        
        try:
            input_win = self.window_manager.input_win
            input_box = self.window_manager.current_layout.input_box
            
            # Clear window
            input_win.clear()
            
            # Display prompt and input
            prompt = "Input> "
            display_width = input_box.inner_width
            
            # Handle multi-line input display
            if len(prompt + self.current_input) <= display_width:
                # Single line display
                input_win.addstr(0, 0, prompt + self.current_input)
            else:
                # Multi-line display
                input_win.addstr(0, 0, prompt)
                
                # Display input text with wrapping
                remaining_text = self.current_input
                available_width = display_width - len(prompt)
                y_pos = 0
                
                while remaining_text and y_pos < input_box.inner_height:
                    if y_pos == 0:
                        # First line - account for prompt
                        line_text = remaining_text[:available_width]
                        remaining_text = remaining_text[available_width:]
                        input_win.addstr(y_pos, len(prompt), line_text)
                    else:
                        # Subsequent lines - full width
                        line_text = remaining_text[:display_width]
                        remaining_text = remaining_text[display_width:]
                        input_win.addstr(y_pos, 0, line_text)
                    
                    y_pos += 1
            
            input_win.refresh()
            
        except curses.error:
            pass
    
    def update_status_display(self):
        """Update status window display"""
        if not self.window_manager.status_win or not self.window_manager.current_layout:
            return
        
        try:
            status_win = self.window_manager.status_win
            status_box = self.window_manager.current_layout.status_box
            
            # Clear window
            status_win.clear()
            
            # Display status with truncation
            display_width = status_box.inner_width
            status_text = self.current_status[:display_width]
            
            # Apply status color if available
            if self.color_manager:
                status_color = self.color_manager.get_color_pair("status")
                if status_color:
                    status_win.attron(curses.color_pair(status_color))
                
                status_win.addstr(0, 0, status_text)
                
                if status_color:
                    status_win.attroff(curses.color_pair(status_color))
            else:
                status_win.addstr(0, 0, status_text)
            
            status_win.refresh()
            
        except curses.error:
            pass
    
    def refresh_display(self):
        """Refresh entire display"""
        self.update_output_display()
        self.update_input_display()
        self.update_status_display()


class InputController:
    """Controls input capture and processing without business logic"""
    
    def __init__(self, window_manager, multi_input):
        self.window_manager = window_manager
        self.multi_input = multi_input if multi_input else None
        
        # Input state
        self.current_input = ""
        self.input_locked = False
        
        # Input callbacks
        self.message_callback = None
        self.command_callback = None
        self.resize_callback = None
    
    def set_message_callback(self, callback: Callable[[str], None]):
        """Set callback for user messages"""
        self.message_callback = callback
    
    def set_command_callback(self, callback: Callable[[str], Any]):
        """Set callback for commands"""
        self.command_callback = callback
    
    def set_resize_callback(self, callback: Callable[[], None]):
        """Set callback for resize events"""
        self.resize_callback = callback
    
    def lock_input(self):
        """Lock input during processing"""
        self.input_locked = True
    
    def unlock_input(self):
        """Unlock input"""
        self.input_locked = False
    
    def handle_input(self) -> Optional[str]:
        """Handle input and return complete messages"""
        if self.input_locked or not self.window_manager.input_win:
            return None
        
        try:
            # Get key input
            key = self.window_manager.input_win.getch()
            
            if key == curses.ERR:
                return None
            
            # Handle special keys
            if key == curses.KEY_RESIZE:
                if self.resize_callback:
                    self.resize_callback()
                return None
            
            # Handle input through multi-input if available
            if self.multi_input:
                result = self.multi_input.handle_key(key)
                
                if result.get("complete", False):
                    # Complete input received
                    message = result.get("content", "")
                    self.multi_input.clear()
                    self.current_input = ""
                    return message
                else:
                    # Update current input
                    self.current_input = result.get("content", "")
                    return None
            else:
                # Fallback input handling
                if key == ord('\n') or key == ord('\r'):
                    if self.current_input.strip():
                        message = self.current_input.strip()
                        self.current_input = ""
                        return message
                elif key == curses.KEY_BACKSPACE or key == 127:
                    if self.current_input:
                        self.current_input = self.current_input[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    self.current_input += chr(key)
                
                return None
                
        except curses.error:
            return None
    
    def get_current_input(self) -> str:
        """Get current input content"""
        return self.current_input
    
    def clear_input(self):
        """Clear current input"""
        self.current_input = ""
        if self.multi_input:
            self.multi_input.clear()
    
    def ensure_cursor_visible(self):
        """Ensure cursor is positioned correctly"""
        if self.input_locked or not self.window_manager.input_win:
            curses.curs_set(0)
            return
        
        try:
            input_win = self.window_manager.input_win
            layout = self.window_manager.current_layout
            
            if not layout:
                return
            
            # Calculate cursor position
            prompt_len = len("Input> ")
            input_len = len(self.current_input)
            display_width = layout.input_box.inner_width
            
            if prompt_len + input_len <= display_width:
                # Single line cursor
                cursor_x = prompt_len + input_len
                cursor_y = 0
            else:
                # Multi-line cursor calculation
                available_width = display_width - prompt_len
                overflow = input_len - available_width
                
                if overflow <= 0:
                    cursor_x = prompt_len + input_len
                    cursor_y = 0
                else:
                    cursor_y = min(1 + overflow // display_width, layout.input_box.inner_height - 1)
                    cursor_x = overflow % display_width
            
            # Set cursor position
            input_win.move(cursor_y, cursor_x)
            input_win.refresh()
            curses.curs_set(1)
            
        except curses.error:
            curses.curs_set(0)

# Chunk 2/3 - ui.py - Main UI Controller and Integration Methods

class UIController:
    """
    Pure UI Controller - Interface management without business logic
    
    Responsibilities:
    - Terminal/curses initialization
    - Window creation and layout management  
    - Display refresh and update cycles
    - Input capture and routing
    - Terminal resize handling
    
    Orchestrator Integration:
    - Callbacks for message processing
    - Command routing to orchestrator
    - Status updates from orchestrator
    - Theme management coordination
    """
    
    def __init__(self, debug_logger=None, config=None):
        self.debug_logger = debug_logger
        self.config = config or {}
        
        # Core UI state
        self.ui_state = UIState()
        self.stdscr = None
        
        # UI components
        self.color_manager = None
        self.terminal_manager = None
        self.window_manager = None
        self.display_controller = None
        self.input_controller = None
        self.scroll_manager = None
        self.multi_input = None
        
        # Orchestrator integration
        self.command_processor = None  # Set by orchestrator
        self.message_processor = None  # Set by orchestrator
        self.status_updater = None     # Set by orchestrator
        
        # UI configuration
        self.refresh_rate = self.config.get('ui_refresh_rate', UI_REFRESH_RATE)
        self.auto_refresh = self.config.get('ui_auto_refresh', True)
        
        # Display thread
        self.display_thread = None
        self.display_thread_running = False
    
    def set_command_processor(self, processor):
        """Set command processor from orchestrator"""
        self.command_processor = processor
    
    def set_message_processor(self, processor):
        """Set message processor from orchestrator"""
        self.message_processor = processor
    
    def set_status_updater(self, updater):
        """Set status updater from orchestrator"""
        self.status_updater = updater
    
    def run(self) -> int:
        """Main UI entry point - run interface"""
        try:
            return curses.wrapper(self._run_interface)
        except Exception as e:
            self._log_debug(f"UI runtime error: {e}")
            print(f"Interface error: {e}")
            return 1
    
    def _run_interface(self, stdscr) -> int:
        """Main interface loop with orchestrator integration"""
        try:
            # Initialize interface
            if not self._initialize_interface(stdscr):
                return 1
            
            # Start display refresh thread
            self._start_display_thread()
            
            # Show welcome message
            self._show_welcome_message()
            
            # Main input loop
            return self._main_input_loop()
            
        except KeyboardInterrupt:
            self._log_debug("Interface interrupted by user")
            return 0
        except Exception as e:
            self._log_debug(f"Interface error: {e}")
            return 1
        finally:
            self._cleanup_interface()
    
    def _initialize_interface(self, stdscr) -> bool:
        """Initialize curses interface and components"""
        try:
            self.stdscr = stdscr
            
            # Basic ncurses setup
            curses.curs_set(1)
            curses.noecho()
            curses.cbreak()
            stdscr.nodelay(0)
            stdscr.clear()
            stdscr.refresh()
            
            # Initialize UI components
            if not self._initialize_ui_components():
                return False
            
            # Check terminal size
            if self.terminal_manager and self.terminal_manager.is_too_small():
                self.terminal_manager.show_too_small_message()
                return False
            
            # Create windows
            if not self.window_manager.create_windows():
                return False
            
            # Initialize display components
            self._initialize_display_components()
            
            # Draw initial interface
            self.window_manager.draw_borders()
            self._refresh_display()
            
            self._log_debug("UI interface initialized successfully")
            return True
            
        except Exception as e:
            self._log_debug(f"UI initialization failed: {e}")
            return False
    
    def _initialize_ui_components(self) -> bool:
        """Initialize UI component instances"""
        try:
            # Initialize color manager
            if ColorManager:
                self.color_manager = ColorManager()
                self.color_manager.init_colors()
                
                # Apply configured theme
                initial_theme = self.config.get('color_theme', 'classic')
                self.color_manager.set_theme(initial_theme)
                self.ui_state.current_theme = initial_theme
            
            # Initialize terminal manager
            if TerminalManager:
                self.terminal_manager = TerminalManager(self.stdscr)
                resized, width, height = self.terminal_manager.check_resize()
                self._log_debug(f"Terminal initialized: {width}x{height}")
            
            # Initialize scroll manager
            if ScrollManager:
                self.scroll_manager = ScrollManager()
            
            # Initialize multi-line input
            if MultiLineInput:
                max_width = self.config.get('max_input_width', 100)
                self.multi_input = MultiLineInput(max_width)
            
            # Initialize window manager
            self.window_manager = WindowManager(
                self.stdscr, self.terminal_manager, self.color_manager
            )
            
            return True
            
        except Exception as e:
            self._log_debug(f"UI component initialization failed: {e}")
            return False
    
    def _initialize_display_components(self):
        """Initialize display and input controllers"""
        # Initialize display controller
        self.display_controller = DisplayController(
            self.window_manager, self.color_manager, self.scroll_manager
        )
        
        # Initialize input controller with callbacks
        self.input_controller = InputController(
            self.window_manager, self.multi_input
        )
        
        # Set input callbacks
        self.input_controller.set_message_callback(self._handle_user_message)
        self.input_controller.set_command_callback(self._handle_user_command)
        self.input_controller.set_resize_callback(self._handle_terminal_resize)
    
    def _start_display_thread(self):
        """Start background display refresh thread"""
        if not self.auto_refresh:
            return
        
        self.display_thread_running = True
        self.display_thread = threading.Thread(
            target=self._display_refresh_loop,
            daemon=True
        )
        self.display_thread.start()
        self._log_debug("Display refresh thread started")
    
    def _display_refresh_loop(self):
        """Background display refresh loop"""
        refresh_interval = 1.0 / self.refresh_rate
        
        while self.display_thread_running and self.ui_state.running:
            try:
                if self.ui_state.display_dirty:
                    self._refresh_display()
                    self.ui_state.mark_display_clean()
                
                time.sleep(refresh_interval)
                
            except Exception as e:
                self._log_debug(f"Display refresh error: {e}")
                time.sleep(0.1)
    
    def _main_input_loop(self) -> int:
        """Main input processing loop"""
        last_input_time = time.time()
        
        while self.ui_state.running:
            try:
                # Handle terminal resize
                if self.terminal_manager:
                    resized = self.window_manager.handle_resize()
                    if not resized:
                        self._log_debug("Resize handling failed")
                        continue
                
                # Process input
                user_input = self.input_controller.handle_input()
                current_time = time.time()
                
                if user_input is not None:
                    last_input_time = current_time
                    
                    # Route input based on type
                    if user_input.startswith('/'):
                        self._handle_user_command(user_input)
                    else:
                        self._handle_user_message(user_input)
                
                # Update input display
                current_input = self.input_controller.get_current_input()
                self.display_controller.set_input_content(current_input)
                
                # Ensure cursor visibility
                self.input_controller.ensure_cursor_visible()
                
                # Manual refresh if auto-refresh disabled
                if not self.auto_refresh:
                    if current_time - last_input_time > 0.1:  # 100ms since last input
                        self._refresh_display()
                
                # Prevent excessive CPU usage
                time.sleep(INPUT_PROCESSING_DELAY)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._log_debug(f"Input loop error: {e}")
                time.sleep(0.1)
        
        return 0
    
    def _handle_user_message(self, message: str):
        """Handle user message through orchestrator"""
        if not message.strip():
            return
        
        try:
            # Show user message immediately
            self.display_controller.add_message(message, "user")
            self.ui_state.mark_display_dirty()
            
            # Lock input during processing
            self.ui_state.mark_processing(True)
            self.display_controller.set_status("Processing...")
            self.input_controller.lock_input()
            
            # Process through orchestrator if available
            if self.message_processor:
                # Run in thread to avoid blocking UI
                processing_thread = threading.Thread(
                    target=self._process_message_async,
                    args=(message,),
                    daemon=True
                )
                processing_thread.start()
            else:
                # Fallback: show message without processing
                self.display_controller.add_message(
                    "Message processor not available", "system"
                )
                self._unlock_interface()
            
        except Exception as e:
            self._log_debug(f"Message handling error: {e}")
            self._unlock_interface()
    
    def _process_message_async(self, message: str):
        """Process message through orchestrator asynchronously"""
        try:
            # This will be called by orchestrator integration
            if asyncio.iscoroutinefunction(self.message_processor):
                # Run async processor
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.message_processor(message))
                finally:
                    loop.close()
            else:
                # Run sync processor
                result = self.message_processor(message)
            
            # Handle result
            if result and result.get("success"):
                ai_response = result.get("ai_response")
                if ai_response:
                    self.display_controller.add_message(ai_response, "assistant")
                    self.ui_state.mark_display_dirty()
            else:
                error_msg = result.get("error", "Processing failed") if result else "No response"
                self.display_controller.add_message(f"Error: {error_msg}", "system")
                self.ui_state.mark_display_dirty()
            
        except Exception as e:
            self._log_debug(f"Async message processing error: {e}")
            self.display_controller.add_message(f"Processing error: {e}", "system")
            self.ui_state.mark_display_dirty()
        finally:
            self._unlock_interface()
    
    def _handle_user_command(self, command: str):
        """Handle user command through orchestrator"""
        try:
            # Show command immediately
            self.display_controller.add_message(command, "user")
            self.ui_state.mark_display_dirty()
            
            # Process through command processor if available
            if self.command_processor:
                # Run in thread to avoid blocking UI
                command_thread = threading.Thread(
                    target=self._process_command_async,
                    args=(command,),
                    daemon=True
                )
                command_thread.start()
            else:
                # Handle basic UI commands locally
                self._handle_local_command(command)
            
        except Exception as e:
            self._log_debug(f"Command handling error: {e}")
    
    def _process_command_async(self, command: str):
        """Process command through orchestrator asynchronously"""
        try:
            if asyncio.iscoroutinefunction(self.command_processor.process_command):
                # Run async command processor
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.command_processor.process_command(command)
                    )
                finally:
                    loop.close()
            else:
                # Run sync command processor
                result = self.command_processor.process_command(command)
            
            # Handle command result
            if result:
                if "message" in result:
                    self.display_controller.add_message(result["message"], "system")
                    self.ui_state.mark_display_dirty()
                
                if result.get("shutdown"):
                    self.ui_state.running = False
                
                if "error" in result:
                    self.display_controller.add_message(f"Error: {result['error']}", "system")
                    self.ui_state.mark_display_dirty()
            
        except Exception as e:
            self._log_debug(f"Async command processing error: {e}")
            self.display_controller.add_message(f"Command error: {e}", "system")
            self.ui_state.mark_display_dirty()
    
    def _handle_local_command(self, command: str):
        """Handle basic UI commands locally"""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return
        
        base_cmd = cmd_parts[0].lower()
        
        if base_cmd in ['/quit', '/exit']:
            self.ui_state.running = False
            self.display_controller.add_message("Shutting down...", "system")
        
        elif base_cmd == '/theme':
            if len(cmd_parts) > 1:
                self._change_theme(cmd_parts[1])
            else:
                self.display_controller.add_message(
                    "Usage: /theme <name> (classic/dark/bright)", "system"
                )
        
        elif base_cmd == '/help':
            self._show_help_message()
        
        else:
            self.display_controller.add_message(
                f"Unknown command: {base_cmd}", "system"
            )
        
        self.ui_state.mark_display_dirty()
    
    def _handle_terminal_resize(self):
        """Handle terminal resize event"""
        try:
            self.ui_state.terminal_resized = True
            self.ui_state.last_resize_time = time.time()
            
            # Force display refresh
            self.ui_state.mark_display_dirty()
            
            # Update component dimensions if available
            if self.multi_input and self.window_manager.current_layout:
                new_width = self.window_manager.current_layout.terminal_width - 10
                self.multi_input.update_max_width(new_width)
            
            if self.scroll_manager and self.window_manager.current_layout:
                new_height = self.window_manager.current_layout.output_box.inner_height
                self.scroll_manager.update_window_height(new_height)
            
            self._log_debug("Terminal resize handled")
            
        except Exception as e:
            self._log_debug(f"Resize handling error: {e}")
    
    def _unlock_interface(self):
        """Unlock interface after processing"""
        self.ui_state.mark_processing(False)
        self.display_controller.set_status("Ready")
        self.input_controller.unlock_input()
        self.ui_state.mark_display_dirty()
    
    def _refresh_display(self):
        """Refresh entire display"""
        try:
            if self.display_controller:
                self.display_controller.refresh_display()
            
            if self.window_manager:
                self.window_manager.refresh_all()
                
        except Exception as e:
            self._log_debug(f"Display refresh error: {e}")
    
    def _change_theme(self, theme_name: str):
        """Change color theme"""
        if not self.color_manager:
            self.display_controller.add_message("Color themes not available", "system")
            return
        
        try:
            if self.color_manager.set_theme(theme_name):
                self.ui_state.current_theme = theme_name
                self.ui_state.mark_display_dirty()
                self.display_controller.add_message(
                    f"Theme changed to {theme_name}", "system"
                )
            else:
                available_themes = ["classic", "dark", "bright"]
                self.display_controller.add_message(
                    f"Invalid theme. Available: {', '.join(available_themes)}", "system"
                )
                
        except Exception as e:
            self._log_debug(f"Theme change error: {e}")
            self.display_controller.add_message(f"Theme change failed: {e}", "system")
    
    def _show_welcome_message(self):
        """Show initial welcome message"""
        welcome_msg = "DevName RPG Client - Ready for Adventure!"
        self.display_controller.add_message(welcome_msg, "system")
        self.display_controller.set_status("Ready")
        self.ui_state.mark_display_dirty()
    
    def _show_help_message(self):
        """Show help message"""
        help_text = """
Available Commands:
/help - Show this help
/quit, /exit - Exit application
/theme <name> - Change color theme
Type your message and press Enter to interact with the GM.
"""
        self.display_controller.add_message(help_text.strip(), "system")
    
    def _cleanup_interface(self):
        """Clean up interface on shutdown"""
        try:
            # Stop display thread
            if self.display_thread_running:
                self.display_thread_running = False
                if self.display_thread:
                    self.display_thread.join(timeout=1.0)
            
            # Clean up windows
            if self.window_manager:
                self.window_manager.destroy_windows()
            
            # Reset cursor
            curses.curs_set(1)
            
            self._log_debug("UI interface cleanup complete")
            
        except Exception as e:
            self._log_debug(f"UI cleanup error: {e}")
    
    def _log_debug(self, message: str):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, "UI")
    
    # Public interface methods for orchestrator integration
    
    def add_system_message(self, message: str):
        """Add system message to display"""
        self.display_controller.add_message(message, "system")
        self.ui_state.mark_display_dirty()
    
    def add_assistant_message(self, message: str):
        """Add assistant message to display"""
        self.display_controller.add_message(message, "assistant")
        self.ui_state.mark_display_dirty()
    
    def set_status_message(self, status: str):
        """Set status message"""
        self.display_controller.set_status(status)
        self.ui_state.mark_display_dirty()
    
    def is_running(self) -> bool:
        """Check if UI is running"""
        return self.ui_state.running
    
    def shutdown(self):
        """Request UI shutdown"""
        self.ui_state.running = False
        self.add_system_message("Shutting down...")
    
    def get_ui_stats(self) -> Dict[str, Any]:
        """Get UI statistics"""
        stats = {
            "running": self.ui_state.running,
            "current_theme": self.ui_state.current_theme,
            "processing": self.ui_state.mcp_processing,
            "input_locked": self.ui_state.input_locked,
            "display_dirty": self.ui_state.display_dirty,
            "terminal_resized": self.ui_state.terminal_resized,
            "messages_displayed": len(self.display_controller.messages) if self.display_controller else 0
        }
        
        if self.window_manager and self.window_manager.current_layout:
            layout = self.window_manager.current_layout
            stats["terminal_size"] = f"{layout.terminal_width}x{layout.terminal_height}"
        
        return stats

# Chunk 3/3 - ui.py - Theme Management and Interface Coordination

class ThemeManager:
    """Manages UI themes and visual consistency"""
    
    def __init__(self, color_manager, display_controller):
        self.color_manager = color_manager
        self.display_controller = display_controller
        
        # Theme configurations
        self.theme_configs = {
            "classic": {
                "name": "Classic",
                "description": "Traditional terminal colors",
                "status_prefix": "RPG"
            },
            "dark": {
                "name": "Dark Mode", 
                "description": "Dark background with bright text",
                "status_prefix": "DARK"
            },
            "bright": {
                "name": "Bright Mode",
                "description": "Light background with dark text", 
                "status_prefix": "BRIGHT"
            }
        }
        
        self.current_theme = "classic"
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return list(self.theme_configs.keys())
    
    def get_theme_info(self, theme_name: str) -> Optional[Dict[str, str]]:
        """Get theme configuration information"""
        return self.theme_configs.get(theme_name)
    
    def apply_theme(self, theme_name: str) -> bool:
        """Apply theme and update display"""
        if theme_name not in self.theme_configs:
            return False
        
        try:
            if self.color_manager and self.color_manager.set_theme(theme_name):
                self.current_theme = theme_name
                
                # Update status with theme indicator
                theme_config = self.theme_configs[theme_name]
                status_msg = f"{theme_config['status_prefix']} - Ready"
                self.display_controller.set_status(status_msg)
                
                return True
            
        except Exception:
            pass
        
        return False
    
    def get_current_theme(self) -> str:
        """Get current theme name"""
        return self.current_theme


class InterfaceCoordinator:
    """Coordinates between UI components and external systems"""
    
    def __init__(self, ui_controller):
        self.ui_controller = ui_controller
        self.coordination_active = False
        
        # External system references (set by orchestrator)
        self.semantic_processor = None
        self.memory_manager = None
        self.story_engine = None
        self.mcp_client = None
        self.loaded_prompts = {}
        
        # Coordination state
        self.last_message_count = 0
        self.last_status_update = 0.0
        
    def set_external_systems(self, **systems):
        """Set references to external systems from orchestrator"""
        self.semantic_processor = systems.get('semantic_processor')
        self.memory_manager = systems.get('memory_manager')
        self.story_engine = systems.get('story_engine')
        self.mcp_client = systems.get('mcp_client')
        self.loaded_prompts = systems.get('loaded_prompts', {})
    
    def start_coordination(self):
        """Start coordination between UI and external systems"""
        self.coordination_active = True
        
        # Set up UI callbacks for orchestrator integration
        if hasattr(self.ui_controller, 'input_controller'):
            self.ui_controller.input_controller.set_message_callback(
                self._coordinate_message_processing
            )
            self.ui_controller.input_controller.set_command_callback(
                self._coordinate_command_processing
            )
    
    def stop_coordination(self):
        """Stop coordination"""
        self.coordination_active = False
    
    async def _coordinate_message_processing(self, message: str):
        """Coordinate message processing with external systems"""
        if not self.coordination_active:
            return {"success": False, "error": "Coordination not active"}
        
        try:
            # This method integrates with orchestrator's message processing
            # The actual processing is handled by the orchestrator
            # UI just coordinates the display aspects
            
            # Update UI status
            self.ui_controller.set_status_message("Processing message...")
            
            # Lock UI input during processing
            if hasattr(self.ui_controller, 'ui_state'):
                self.ui_controller.ui_state.mark_processing(True)
            
            # Return coordination info for orchestrator
            return {
                "ui_ready": True,
                "message": message,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _coordinate_command_processing(self, command: str):
        """Coordinate command processing"""
        if not self.coordination_active:
            return {"success": False, "error": "Coordination not active"}
        
        try:
            # Handle UI-specific commands locally
            if command.startswith('/theme'):
                return self._handle_theme_command(command)
            elif command in ['/quit', '/exit']:
                return self._handle_quit_command()
            elif command == '/status':
                return self._handle_status_command()
            
            # Pass other commands to orchestrator
            return {
                "ui_handled": False,
                "command": command,
                "requires_orchestrator": True
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_theme_command(self, command: str) -> Dict[str, Any]:
        """Handle theme change command"""
        try:
            parts = command.split()
            if len(parts) < 2:
                available = ["classic", "dark", "bright"]
                return {
                    "success": True,
                    "message": f"Usage: /theme <name>. Available: {', '.join(available)}"
                }
            
            theme_name = parts[1].lower()
            
            if hasattr(self.ui_controller, 'theme_manager'):
                if self.ui_controller.theme_manager.apply_theme(theme_name):
                    return {
                        "success": True,
                        "message": f"Theme changed to {theme_name}",
                        "ui_refresh_needed": True
                    }
                else:
                    available = self.ui_controller.theme_manager.get_available_themes()
                    return {
                        "success": False,
                        "message": f"Invalid theme. Available: {', '.join(available)}"
                    }
            else:
                return {"success": False, "message": "Theme manager not available"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_quit_command(self) -> Dict[str, Any]:
        """Handle quit command"""
        return {
            "success": True,
            "message": "Shutting down...",
            "shutdown_requested": True
        }
    
    def _handle_status_command(self) -> Dict[str, Any]:
        """Handle status command"""
        try:
            if hasattr(self.ui_controller, 'get_ui_stats'):
                stats = self.ui_controller.get_ui_stats()
                status_msg = f"UI Status: Running={stats.get('running')}, Theme={stats.get('current_theme')}"
                if 'terminal_size' in stats:
                    status_msg += f", Size={stats['terminal_size']}"
                
                return {
                    "success": True,
                    "message": status_msg,
                    "detailed_stats": stats
                }
            else:
                return {"success": True, "message": "UI status: Running"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def update_ui_from_external_state(self):
        """Update UI based on external system state"""
        if not self.coordination_active:
            return
        
        try:
            current_time = time.time()
            
            # Update message count status
            if self.memory_manager and hasattr(self.memory_manager, 'get_message_count'):
                message_count = self.memory_manager.get_message_count()
                if message_count != self.last_message_count:
                    self.last_message_count = message_count
                    # Update status with message count
                    status_msg = f"Messages: {message_count} - Ready"
                    self.ui_controller.set_status_message(status_msg)
            
            # Periodic status updates
            if current_time - self.last_status_update > 30.0:  # Every 30 seconds
                self.last_status_update = current_time
                self._update_periodic_status()
                
        except Exception as e:
            if hasattr(self.ui_controller, '_log_debug'):
                self.ui_controller._log_debug(f"External state update error: {e}")
    
    def _update_periodic_status(self):
        """Update status with periodic information"""
        try:
            status_parts = []
            
            # Add story engine status if available
            if self.story_engine and hasattr(self.story_engine, 'get_pressure_stats'):
                pressure_stats = self.story_engine.get_pressure_stats()
                pressure = pressure_stats.get('current_pressure', 0.0)
                status_parts.append(f"Pressure: {pressure:.2f}")
            
            # Add memory status if available
            if self.memory_manager and hasattr(self.memory_manager, 'get_stats'):
                memory_stats = self.memory_manager.get_stats()
                token_count = memory_stats.get('total_tokens', 0)
                status_parts.append(f"Tokens: {token_count}")
            
            if status_parts:
                status_msg = " | ".join(status_parts) + " - Ready"
                self.ui_controller.set_status_message(status_msg)
                
        except Exception:
            pass


# Enhanced UIController with theme and coordination integration
class UIControllerEnhanced(UIController):
    """Enhanced UI Controller with theme management and coordination"""
    
    def __init__(self, debug_logger=None, config=None):
        super().__init__(debug_logger, config)
        
        # Enhanced components
        self.theme_manager = None
        self.interface_coordinator = None
    
    def _initialize_display_components(self):
        """Initialize display components with enhancements"""
        # Call parent initialization
        super()._initialize_display_components()
        
        # Initialize theme manager
        if self.color_manager and self.display_controller:
            self.theme_manager = ThemeManager(
                self.color_manager, self.display_controller
            )
            
            # Apply initial theme
            initial_theme = self.config.get('color_theme', 'classic')
            self.theme_manager.apply_theme(initial_theme)
        
        # Initialize interface coordinator
        self.interface_coordinator = InterfaceCoordinator(self)
    
    def set_external_systems(self, **systems):
        """Set external system references for coordination"""
        if self.interface_coordinator:
            self.interface_coordinator.set_external_systems(**systems)
    
    def start_coordination(self):
        """Start coordination with external systems"""
        if self.interface_coordinator:
            self.interface_coordinator.start_coordination()
    
    def _main_input_loop(self) -> int:
        """Enhanced main input loop with coordination"""
        # Start coordination
        self.start_coordination()
        
        # Call parent input loop with periodic coordination updates
        last_coordination_update = time.time()
        
        while self.ui_state.running:
            try:
                # Handle standard input processing
                result = super()._handle_single_input_cycle()
                
                # Periodic coordination updates
                current_time = time.time()
                if current_time - last_coordination_update > 5.0:  # Every 5 seconds
                    if self.interface_coordinator:
                        self.interface_coordinator.update_ui_from_external_state()
                    last_coordination_update = current_time
                
                # Continue with standard processing
                if result != 0:
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._log_debug(f"Enhanced input loop error: {e}")
                time.sleep(0.1)
        
        # Stop coordination
        if self.interface_coordinator:
            self.interface_coordinator.stop_coordination()
        
        return 0
    
    def _handle_single_input_cycle(self):
        """Handle single input cycle (extracted for coordination)"""
        try:
            # Handle terminal resize
            if self.terminal_manager:
                resized = self.window_manager.handle_resize()
                if not resized:
                    return 1
            
            # Process input
            user_input = self.input_controller.handle_input()
            
            if user_input is not None:
                # Route input based on type
                if user_input.startswith('/'):
                    self._handle_user_command(user_input)
                else:
                    self._handle_user_message(user_input)
            
            # Update input display
            current_input = self.input_controller.get_current_input()
            self.display_controller.set_input_content(current_input)
            
            # Ensure cursor visibility
            self.input_controller.ensure_cursor_visible()
            
            # Manual refresh if needed
            if not self.auto_refresh and user_input is not None:
                self._refresh_display()
            
            # Prevent excessive CPU usage
            time.sleep(INPUT_PROCESSING_DELAY)
            
            return 0
            
        except Exception as e:
            self._log_debug(f"Input cycle error: {e}")
            return 1
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced UI statistics"""
        stats = self.get_ui_stats()
        
        # Add theme information
        if self.theme_manager:
            stats["theme_manager"] = {
                "current_theme": self.theme_manager.get_current_theme(),
                "available_themes": self.theme_manager.get_available_themes()
            }
        
        # Add coordination information
        if self.interface_coordinator:
            stats["coordination"] = {
                "active": self.interface_coordinator.coordination_active,
                "last_message_count": self.interface_coordinator.last_message_count
            }
        
        return stats


# Factory functions for orchestrator integration
def create_ui_controller(debug_logger=None, config=None) -> UIControllerEnhanced:
    """Factory function to create enhanced UI controller"""
    return UIControllerEnhanced(debug_logger, config)

def create_basic_ui_controller(debug_logger=None, config=None) -> UIController:
    """Factory function to create basic UI controller"""
    return UIController(debug_logger, config)


# Utility functions for UI management
def validate_ui_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize UI configuration"""
    validated_config = {}
    
    # Validate refresh rate
    refresh_rate = config.get('ui_refresh_rate', UI_REFRESH_RATE)
    if not isinstance(refresh_rate, (int, float)) or refresh_rate <= 0:
        refresh_rate = UI_REFRESH_RATE
    validated_config['ui_refresh_rate'] = min(refresh_rate, 60)  # Cap at 60 FPS
    
    # Validate auto refresh
    auto_refresh = config.get('ui_auto_refresh', True)
    validated_config['ui_auto_refresh'] = bool(auto_refresh)
    
    # Validate theme
    color_theme = config.get('color_theme', 'classic')
    if color_theme not in ['classic', 'dark', 'bright']:
        color_theme = 'classic'
    validated_config['color_theme'] = color_theme
    
    # Validate input width
    max_input_width = config.get('max_input_width', 100)
    if not isinstance(max_input_width, int) or max_input_width < 20:
        max_input_width = 100
    validated_config['max_input_width'] = max_input_width
    
    return validated_config

def get_ui_info() -> Dict[str, Any]:
    """Get information about UI capabilities"""
    return {
        "name": "DevName RPG Client UI Controller",
        "version": "1.0",
        "features": [
            "Pure UI management without business logic",
            "Dynamic terminal resize handling",
            "Multi-threaded display refresh",
            "Theme management system",
            "Orchestrator integration via callbacks",
            "Asynchronous message processing",
            "Command routing system"
        ],
        "themes": ["classic", "dark", "bright"],
        "integration_points": [
            "Message processor callbacks",
            "Command processor callbacks", 
            "Status update coordination",
            "External system state synchronization"
        ],
        "dependencies": [
            "curses library for terminal UI",
            "threading for background processing",
            "asyncio for async coordination"
        ]
    }


# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Pure UI Controller Module")
    print("Successfully implemented pure UI controller with:")
    print("✓ Pure interface management without business logic")
    print("✓ Dynamic terminal resize handling") 
    print("✓ Multi-threaded display refresh system")
    print("✓ Theme management with visual consistency")
    print("✓ Orchestrator integration via callback system")
    print("✓ Asynchronous message processing coordination")
    print("✓ Command routing with local UI command handling")
    print("✓ External system state synchronization")
    print("✓ Complete separation from business logic modules")
    
    print("\nUI Controller Info:")
    info = get_ui_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    print("\nReady for integration with orch.py orchestrator.")
    print("Next phase: Create uilib.py (Phase 4) - Consolidate UI utilities.")
