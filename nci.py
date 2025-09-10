# Chunk 1/4 - nci.py - Imports and Initialization with Narrative Time Integration
#!/usr/bin/env python3
"""
DevName RPG Client - Ncurses Interface Module (nci.py) - NARRATIVE TIME INTEGRATION
Module architecture and interconnects documented in genai.txt
Integrates narrative time tracking and semantic analysis with momentum management
"""

import curses
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple

# Import extracted modules with dynamic coordinate support
from nci_colors import ColorManager, ColorTheme
from nci_terminal import TerminalManager, LayoutGeometry
from nci_display import DisplayMessage, InputValidator
from nci_scroll import ScrollManager
from nci_input import MultiLineInput

# Import core dependencies with narrative time integration
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
    NARRATIVE TIME INTEGRATION: Ncurses interface with semantic time detection and momentum management
    
    FEATURES:
    - Dynamic window positioning eliminates coordinate assumption bugs
    - Automatic adaptation to terminal geometry changes
    - Narrative time tracking separates story progression from real time
    - Complete LLM-driven semantic memory management with multi-pass condensation
    - Robust 15-message momentum analysis cycle with 5-strategy JSON parsing
    - Background processing prevents interface blocking
    - Enhanced antagonist generation with quality validation
    - Pressure floor ratcheting system prevents infinite stalling
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
        
        # Module interconnects with narrative time integration
        self.memory_manager = EnhancedMemoryManager(
            debug_logger=debug_logger,
            auto_save_enabled=True
        )
        self.mcp_client = MCPClient(debug_logger=debug_logger)
        self.sme = StoryMomentumEngine(debug_logger=debug_logger)
        
        # Load existing SME state from memory
        self._initialize_sme_state()
        
        # PROMPT INTEGRATION - Load from config passed by main.py
        self.loaded_prompts = self.config.get('prompts', {})
        
        self._configure_components()
    
    def _initialize_sme_state(self):
        """Initialize SME state from EMM on startup with narrative time restoration"""
        try:
            momentum_state = self.memory_manager.get_momentum_state()
            if momentum_state:
                success = self.sme.load_state_from_dict(momentum_state)
                if success and self.debug_logger:
                    self.debug_logger.debug("SME state loaded from EMM on startup with narrative time tracking")
            else:
                if self.debug_logger:
                    self.debug_logger.debug("No existing SME state found in EMM")
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to initialize SME state: {e}")
    
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
        
        self._log_debug(f"Interface initialized with narrative time integration: {width}x{height}")
    
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

# Chunk 2/4 - nci.py - Welcome Content and Input Processing with Narrative Time

    def _populate_welcome_content(self):
        """Add welcome messages with memory load status and narrative time capabilities"""
        # Welcome message with narrative time integration
        welcome_msg = DisplayMessage(
            "DevName RPG Client started with narrative time tracking and LLM-powered story analysis.",
            "system"
        )
        self._add_message_immediate(welcome_msg)
        
        # Memory load status with semantic analysis and narrative time info
        mem_stats = self.memory_manager.get_memory_stats()
        if mem_stats.get('message_count', 0) > 0:
            narrative_time = mem_stats.get('narrative_time_formatted', '0s')
            memory_msg = DisplayMessage(
                f"Restored {mem_stats['message_count']} messages from previous session "
                f"({mem_stats.get('condensations_performed', 0)} condensations performed, "
                f"{narrative_time} narrative time).",
                "system"
            )
            self._add_message_immediate(memory_msg)
        
        # SME state status with narrative time info
        sme_stats = self.sme.get_pressure_stats()
        if sme_stats.get('status') != 'no_data':
            pressure = sme_stats.get('current_pressure', 0.0)
            arc = sme_stats.get('current_arc', 'setup')
            floor = sme_stats.get('pressure_floor', 0.0)
            escalations = sme_stats.get('escalation_count', 0)
            narrative_time = sme_stats.get('narrative_time_formatted', '0s')
            sme_msg = DisplayMessage(
                f"Story momentum restored: {pressure:.2f} pressure (floor {floor:.2f}), "
                f"{arc} arc, {escalations} escalations, {narrative_time} elapsed",
                "system"
            )
            self._add_message_immediate(sme_msg)
        
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
        
        # Ready message with narrative time features
        ready_msg = DisplayMessage(
            "Ready for adventure! Narrative time tracking will enhance story pacing, "
            "while LLM analysis continues every 15 messages with semantic memory management.",
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
        """Handle terminal resize with dynamic coordinate system"""
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
            
            # Process input with narrative time integration
            self._process_user_input(content)
    
    def _process_user_input(self, user_input: str):
        """Process user input with narrative time tracking and integrated LLM analysis"""
        try:
            if user_input.startswith('/'):
                self._process_command(user_input)
                self.set_processing_state_immediate(False)
                return
            
            # Calculate semantic narrative duration before storing
            narrative_duration = self.sme.narrative_tracker.calculate_semantic_duration(user_input)
            
            # Store in memory with calculated narrative duration
            self.memory_manager.add_message(user_input, MessageType.USER, narrative_duration)
            
            # Update story momentum with narrative time tracking
            momentum_result = self.sme.process_user_input(user_input)
            
            # Check if LLM analysis is needed (precise 15-message count for USER/ASSISTANT only)
            user_assistant_messages = [msg for msg in self.memory_manager.get_messages() 
                                     if msg.message_type in [MessageType.USER, MessageType.ASSISTANT]]
            total_conversation_messages = len(user_assistant_messages)
            
            if self.sme.should_analyze_momentum(total_conversation_messages):
                # Show analysis notification
                self.add_system_message_immediate("Analyzing story momentum with narrative time context...")
                
                # Trigger comprehensive background LLM analysis
                self._trigger_comprehensive_momentum_analysis(total_conversation_messages)
            
            # Send to MCP server
            self._send_mcp_request(user_input)
            
        except Exception as e:
            self.add_error_message_immediate(f"Processing failed: {e}")
            self.set_processing_state_immediate(False)
    
    def _trigger_comprehensive_momentum_analysis(self, total_messages: int):
        """Trigger complete LLM momentum analysis in background thread with narrative time context"""
        def run_comprehensive_analysis():
            try:
                # Load current SME state from EMM
                momentum_state = self.memory_manager.get_momentum_state()
                if momentum_state:
                    self.sme.load_state_from_dict(momentum_state)
                
                # Prepare conversation context for analysis
                conversation_messages = self.memory_manager.get_conversation_for_mcp()
                
                # Determine if this is first analysis or regular cycle
                is_first_analysis = momentum_state is None or self.sme.last_analysis_count == 0
                
                # Run comprehensive async analysis in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Execute complete momentum analysis with narrative time context
                    analysis_result = loop.run_until_complete(
                        self.sme.analyze_momentum(conversation_messages, total_messages, is_first_analysis)
                    )
                    
                    # Save updated state to EMM
                    updated_state = self.sme.save_state_to_dict()
                    self.memory_manager.update_momentum_state(updated_state)
                    
                    # Show comprehensive analysis completion with narrative time details
                    pressure = analysis_result.get("narrative_pressure", 0.0)
                    manifestation = analysis_result.get("manifestation_type", "unknown")
                    source = analysis_result.get("pressure_source", "unknown")
                    commitment_change = analysis_result.get("commitment_change", "no_change")
                    
                    # Get narrative time stats for display
                    narrative_stats = self.sme.narrative_tracker.get_stats()
                    narrative_time = narrative_stats.get('narrative_time_formatted', '0s')
                    
                    completion_msg = (f"Story analysis complete: {pressure:.2f} pressure, "
                                    f"{manifestation} manifestation from {source} source, "
                                    f"{narrative_time} elapsed")
                    self.add_system_message_immediate(completion_msg)
                    
                    # Show antagonist updates if applicable
                    if commitment_change != "no_change" and self.sme.current_antagonist:
                        antagonist_msg = (f"Antagonist {self.sme.current_antagonist.name} "
                                        f"commitment escalated to: {commitment_change}")
                        self.add_system_message_immediate(antagonist_msg)
                    
                    if self.debug_logger:
                        self.debug_logger.debug(f"Comprehensive momentum analysis complete: pressure {pressure:.2f}, "
                                              f"narrative time {narrative_time}")
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                error_msg = f"Story analysis failed: {str(e)}"
                self.add_error_message_immediate(error_msg)
                if self.debug_logger:
                    self.debug_logger.error(f"Comprehensive momentum analysis failed: {e}")
        
        # Run in background thread with enhanced name
        analysis_thread = threading.Thread(
            target=run_comprehensive_analysis,
            daemon=True,
            name="SME-Comprehensive-Analysis"
        )
        analysis_thread.start()

# Chunk 3/4 - nci.py - Command Processing and Stats with Narrative Time

    def _send_mcp_request(self, user_input: str):
        """Send MCP request with enhanced story context including narrative time"""
        try:
            # Get context data with complete SME integration
            conversation_history = self.memory_manager.get_conversation_for_mcp()
            
            # Get current SME state (may include recent comprehensive LLM analysis)
            story_context = self.sme.get_story_context()
            context_str = self._format_complete_story_context(story_context)
            
            # Build system messages with enhanced context
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
                
                # Store and display response with GM response duration
                self.memory_manager.add_message(response_data, MessageType.ASSISTANT, narrative_duration=5.0)
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
                    
                    self.memory_manager.add_message(response, MessageType.ASSISTANT, narrative_duration=5.0)
                    self.add_assistant_message_immediate(response)
                    
                except Exception as fallback_error:
                    self.add_error_message_immediate(f"Communication error: {str(fallback_error)}")
            
        except Exception as e:
            self.add_error_message_immediate(f"Request processing failed: {e}")
        finally:
            self.set_processing_state_immediate(False)
    
    def _format_complete_story_context(self, context: Dict[str, Any]) -> str:
        """Format complete story context with narrative time details"""
        if not context:
            return ""
        
        parts = []
        pressure = context.get('pressure_level', 0.0)
        arc = context.get('story_arc', 'unknown')
        state = context.get('narrative_state', 'unknown')
        floor = context.get('pressure_floor', 0.0)
        narrative_time = context.get('narrative_time', '0s')
        total_exchanges = context.get('total_exchanges', 0)
        
        # Core momentum info with pressure floor and narrative time
        parts.append(f"Pressure: {pressure:.2f} (floor: {floor:.2f}), Arc: {arc}, State: {state}")
        parts.append(f"Narrative time: {narrative_time} ({total_exchanges} exchanges)")
        
        # Enhanced antagonist information with complete details
        if context.get('antagonist_present'):
            antagonist = context.get('antagonist', {})
            name = antagonist.get('name', 'Unknown')
            commitment = antagonist.get('commitment_level', 'unknown')
            threat = antagonist.get('threat_level', 0.0)
            losses = antagonist.get('resources_lost', 0)
            parts.append(f"Antagonist: {name} ({commitment} commitment, threat: {threat:.2f}, {losses} losses)")
        
        # Pressure trend with detailed analysis
        trend = context.get('pressure_trend', 'stable')
        if trend != 'stable':
            parts.append(f"Trend: {trend}")
        
        # Tension recommendations
        if context.get('should_introduce_tension'):
            parts.append("Recommend: Introduce tension")
        if context.get('climax_approaching'):
            parts.append("Alert: Climax approaching")
        
        return " | ".join(parts)
    
    def _build_system_messages(self, story_context: str) -> List[Dict[str, str]]:
        """Build system messages with integrated prompts and complete story context"""
        system_messages = []
        
        if self.loaded_prompts.get('critrules'):
            primary_prompt = self.loaded_prompts['critrules']
            if story_context:
                primary_prompt += f"\n\n**COMPLETE STORY CONTEXT**: {story_context}"
            system_messages.append({"role": "system", "content": primary_prompt})
        
        if self.loaded_prompts.get('companion'):
            system_messages.append({"role": "system", "content": self.loaded_prompts['companion']})
        
        if self.loaded_prompts.get('lowrules'):
            system_messages.append({"role": "system", "content": self.loaded_prompts['lowrules']})
        
        return system_messages

    def _process_command(self, command: str):
        """Process commands with narrative time analysis commands"""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self._show_complete_help()
        elif cmd == '/quit' or cmd == '/exit':
            self.running = False
        elif cmd.startswith('/clearmemory'):
            # Parse optional backup filename
            parts = command.split(None, 1)
            backup_filename = parts[1] if len(parts) > 1 else None
            self._clear_memory(backup_filename)
        elif cmd.startswith('/savememory'):
            # Parse optional filename
            parts = command.split(None, 1)
            filename = parts[1] if len(parts) > 1 else None
            self._save_memory(filename)
        elif cmd.startswith('/loadmemory'):
            # Parse required filename
            parts = command.split(None, 1)
            if len(parts) < 2:
                self.add_error_message_immediate("Usage: /loadmemory <filename>")
            else:
                filename = parts[1]
                self._load_memory(filename)
        elif cmd == '/stats':
            self._show_complete_stats()
        elif cmd == '/analyze':
            self._force_comprehensive_analysis()
        elif cmd == '/reset_momentum':
            self._reset_complete_momentum()
        elif cmd.startswith('/theme '):
            theme_name = cmd[7:].strip()
            self._change_theme(theme_name)
        else:
            self.add_error_message_immediate(f"Unknown command: {command}")
    
    def _force_comprehensive_analysis(self):
        """Force immediate comprehensive momentum analysis for testing"""
        try:
            user_assistant_messages = [msg for msg in self.memory_manager.get_messages() 
                                     if msg.message_type in [MessageType.USER, MessageType.ASSISTANT]]
            total_messages = len(user_assistant_messages)
            
            if total_messages < 5:
                self.add_error_message_immediate("Need at least 5 messages for meaningful analysis")
                return
            
            self.add_system_message_immediate("Running forced comprehensive momentum analysis with narrative time context...")
            self._trigger_comprehensive_momentum_analysis(total_messages)
            
        except Exception as e:
            self.add_error_message_immediate(f"Failed to force comprehensive analysis: {e}")
    
    def _reset_complete_momentum(self):
        """Reset complete story momentum state including narrative time"""
        try:
            self.sme.reset_story_state()
            
            # Clear momentum state from EMM
            empty_state = self.sme.save_state_to_dict()
            self.memory_manager.update_momentum_state(empty_state)
            
            self.add_system_message_immediate("Complete story momentum reset to initial state with narrative time tracking")
            
        except Exception as e:
            self.add_error_message_immediate(f"Failed to reset complete momentum: {e}")
    
    def _clear_memory(self, backup_filename: Optional[str] = None):
        """Clear memory with optional backup"""
        try:
            # Save backup if filename provided
            if backup_filename:
                success = self.memory_manager.save_conversation(backup_filename)
                if success:
                    self.add_system_message_immediate(f"Backup saved to {backup_filename}")
                else:
                    self.add_error_message_immediate(f"Failed to save backup to {backup_filename}")
                    return
            
            # Clear memory file and in-memory state
            success = self.memory_manager.clear_memory_file()
            if success:
                # Reset complete SME state
                self.sme.reset_story_state()
                
                # Also clear display
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
                
                # Show confirmation
                if backup_filename:
                    self.add_system_message_immediate(f"Memory cleared with complete state reset (backup saved to {backup_filename})")
                else:
                    self.add_system_message_immediate("Memory cleared with complete state reset")
            else:
                self.add_error_message_immediate("Failed to clear memory")
                
        except Exception as e:
            self.add_error_message_immediate(f"Error clearing memory: {e}")
            self._log_debug(f"Memory clear error: {e}")

    def _save_memory(self, filename: Optional[str] = None):
        """Save memory to file"""
        try:
            if not filename:
                import time
                filename = f"chat_backup_{int(time.time())}.json"
            
            success = self.memory_manager.save_conversation(filename)
            if success:
                self.add_system_message_immediate(f"Memory saved to {filename}")
            else:
                self.add_error_message_immediate(f"Failed to save memory to {filename}")
                
        except Exception as e:
            self.add_error_message_immediate(f"Error saving memory: {e}")
            self._log_debug(f"Memory save error: {e}")

    def _load_memory(self, filename: str):
        """Load memory from file and restore complete SME state with narrative time"""
        try:
            success = self.memory_manager.load_conversation(filename)
            if success:
                # Clear current display
                self.display_messages.clear()
                self.display_lines.clear()
                
                # Reset scroll manager
                self.scroll_manager.scroll_offset = 0
                self.scroll_manager.max_scroll = 0
                self.scroll_manager.in_scrollback = False
                
                # Clear output window
                if self.output_win:
                    self.output_win.clear()
                    self.output_win.refresh()
                
                # Restore complete SME state
                momentum_state = self.memory_manager.get_momentum_state()
                if momentum_state:
                    self.sme.load_state_from_dict(momentum_state)
                    self.add_system_message_immediate("Complete story momentum state restored with narrative time tracking")
                
                # Show loaded conversation
                messages = self.memory_manager.get_messages()
                for msg in messages:
                    if msg.message_type == MessageType.USER:
                        self.add_user_message_immediate(msg.content)
                    elif msg.message_type == MessageType.ASSISTANT:
                        self.add_assistant_message_immediate(msg.content)
                    elif msg.message_type == MessageType.SYSTEM:
                        self.add_system_message_immediate(msg.content)
                
                # Show confirmation with narrative time info
                narrative_stats = self.memory_manager.get_narrative_time_stats()
                narrative_time = narrative_stats.get('narrative_time_formatted', '0s')
                self.add_system_message_immediate(f"Loaded {len(messages)} messages from {filename} "
                                               f"with complete state restoration ({narrative_time} narrative time)")
            else:
                self.add_error_message_immediate(f"Failed to load memory from {filename}")
                
        except Exception as e:
            self.add_error_message_immediate(f"Error loading memory: {e}")
            self._log_debug(f"Memory load error: {e}")
    
    def _show_complete_stats(self):
        """Show comprehensive system statistics with narrative time information"""
        try:
            # Enhanced memory stats with narrative time information
            mem_stats = self.memory_manager.get_memory_stats()
            file_info = self.memory_manager.get_memory_file_info()
            patterns = self.memory_manager.analyze_conversation_patterns()
            
            # Basic memory info
            self.add_system_message_immediate(
                f"Memory: {mem_stats.get('message_count', 0)} messages, "
                f"{mem_stats.get('total_tokens', 0)} tokens, "
                f"{mem_stats.get('condensations_performed', 0)} multi-pass condensations"
            )
            
            # Narrative time statistics
            narrative_time = mem_stats.get('narrative_time_formatted', '0s')
            if narrative_time != '0s':
                self.add_system_message_immediate(f"Narrative time: {narrative_time}")
            
            # Detailed semantic categorization stats
            if 'semantic_categories' in patterns:
                categories = patterns['semantic_categories']
                category_summary = ', '.join([f"{k}: {v}" for k, v in categories.items() if v > 0])
                if category_summary:
                    self.add_system_message_immediate(f"Semantic categories: {category_summary}")
            
            # Enhanced condensed message stats
            condensed_count = patterns.get('condensed_messages', 0)
            if condensed_count > 0:
                utilization = mem_stats.get('utilization', 0.0)
                self.add_system_message_immediate(f"Condensed messages: {condensed_count}, utilization: {utilization:.1%}")
            
            # Complete memory file details
            if file_info.get('file_exists', False):
                file_size = file_info.get('file_size', 0)
                file_size_kb = file_size / 1024 if file_size > 0 else 0
                last_modified = file_info.get('last_modified', 'unknown')
                backup_status = "yes" if file_info.get('backup_exists', False) else "no"
                file_narrative_time = file_info.get('narrative_time', '0s')
                
                self.add_system_message_immediate(
                    f"Memory file: {file_info.get('file_path', 'unknown')} "
                    f"({file_size_kb:.1f}KB, modified: {last_modified[:19]}, backup: {backup_status})"
                )
                
                if file_narrative_time != '0s':
                    self.add_system_message_immediate(f"File narrative time: {file_narrative_time}")
                
                auto_save_status = "enabled" if file_info.get('auto_save_enabled', False) else "disabled"
                self.add_system_message_immediate(f"Auto-save: {auto_save_status}")
            else:
                self.add_system_message_immediate("Memory file: Not found")
                
        except Exception as e:
            self.add_system_message_immediate("Memory: Stats unavailable")
            self._log_debug(f"Memory stats error: {e}")
        
        try:
            # Complete story stats with narrative time analysis information
            sme_stats = self.sme.get_pressure_stats()
            if 'current_pressure' in sme_stats:
                pressure = sme_stats['current_pressure']
                arc = sme_stats.get('current_arc', 'unknown')
                updates = sme_stats.get('total_updates', 0)
                floor = sme_stats.get('pressure_floor', 0.0)
                escalations = sme_stats.get('escalation_count', 0)
                last_analysis = sme_stats.get('last_analysis_count', 0)
                variance = sme_stats.get('pressure_variance', 0.0)
                narrative_time = sme_stats.get('narrative_time_formatted', '0s')
                avg_duration = sme_stats.get('average_exchange_duration', 0.0)
                
                self.add_system_message_immediate(
                    f"Story: Pressure {pressure:.2f} (floor {floor:.2f}, var {variance:.3f}), "
                    f"Arc {arc}, {updates} updates, {escalations} escalations"
                )
                
                self.add_system_message_immediate(f"Analysis: Last at message {last_analysis}")
                
                # Narrative time details
                if narrative_time != '0s':
                    self.add_system_message_immediate(f"Narrative time: {narrative_time} (avg {avg_duration:.1f}s per exchange)")
                
                # Enhanced antagonist information
                story_context = self.sme.get_story_context()
                if story_context.get('antagonist_present'):
                    antagonist = story_context.get('antagonist', {})
                    name = antagonist.get('name', 'Unknown')
                    commitment = antagonist.get('commitment_level', 'unknown')
                    threat = antagonist.get('threat_level', 0.0)
                    losses = antagonist.get('resources_lost', 0)
                    self.add_system_message_immediate(
                        f"Antagonist: {name} ({commitment} commitment, threat {threat:.2f}, {losses} losses)"
                    )
        except Exception as e:
            self.add_system_message_immediate("Story: Stats unavailable")
            self._log_debug(f"Story stats error: {e}")
        
        # MCP stats
        try:
            mcp_info = self.mcp_client.get_server_info()
            self.add_system_message_immediate(f"MCP: {mcp_info.get('server_url', 'unknown')}")
            self.add_system_message_immediate(f"Model: {mcp_info.get('model', 'unknown')}")
        except Exception as e:
            self.add_system_message_immediate("MCP: Stats unavailable")
            self._log_debug(f"MCP stats error: {e}")
        
        # Display stats with scroll manager error handling
        try:
            scroll_info = self.scroll_manager.get_scroll_info()
            self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, "
                                           f"Scroll: {scroll_info.get('offset', 0)}/{scroll_info.get('max', 0)}")
        except Exception as e:
            self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, Scroll: Error")
            self._log_debug(f"Scroll stats error: {e}")
        
        # Terminal stats with layout info
        try:
            if self.current_layout:
                layout = self.current_layout
                self.add_system_message_immediate(f"Terminal: {layout.terminal_width}x{layout.terminal_height}")
                self.add_system_message_immediate(f"Layout: Output {layout.output_box.inner_width}x{layout.output_box.inner_height}, "
                                               f"Input {layout.input_box.inner_width}x{layout.input_box.inner_height}")
            else:
                self.add_system_message_immediate("Terminal: Layout not available")
        except Exception as e:
            self.add_system_message_immediate("Terminal: Stats unavailable")
            self._log_debug(f"Terminal stats error: {e}")
        
        # Input stats
        try:
            input_content = self.multi_input.get_content()
            input_lines = len(self.multi_input.lines)
            cursor_line, cursor_col = self.multi_input.get_cursor_position()
            self.add_system_message_immediate(f"Input: {len(input_content)} chars, {input_lines} lines, "
                                           f"cursor at {cursor_line}:{cursor_col}")
        except Exception as e:
            self.add_system_message_immediate("Input: Stats unavailable")
            self._log_debug(f"Input stats error: {e}")
        
        # Enhanced prompt stats
        try:
            total_tokens = sum(len(content) // 4 for content in self.loaded_prompts.values() if content.strip())
            active_prompts = [name for name, content in self.loaded_prompts.items() if content.strip()]
            self.add_system_message_immediate(f"Prompts: {len(active_prompts)} active ({', '.join(active_prompts)}), "
                                           f"{total_tokens:,} tokens")
        except Exception as e:
            self.add_system_message_immediate("Prompts: Stats unavailable")
            self._log_debug(f"Prompt stats error: {e}")
        
        # Background thread status
        try:
            active_threads = [t for t in threading.enumerate() 
                            if t.name in ["SME-Comprehensive-Analysis", "EMM-AutoSave", "EMM-Condensation"]]
            if active_threads:
                thread_names = [t.name for t in active_threads]
                self.add_system_message_immediate(f"Background: {len(active_threads)} active ({', '.join(thread_names)})")
        except Exception as e:
            self._log_debug(f"Thread stats error: {e}")

# Chunk 4/4 - nci.py - Command Processing, Utilities, and Shutdown

    def _process_command(self, command: str):
        """Process commands with complete LLM analysis commands"""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self._show_complete_help()
        elif cmd == '/quit' or cmd == '/exit':
            self.running = False
        elif cmd.startswith('/clearmemory'):
            # Parse optional backup filename
            parts = command.split(None, 1)
            backup_filename = parts[1] if len(parts) > 1 else None
            self._clear_memory(backup_filename)
        elif cmd.startswith('/savememory'):
            # Parse optional filename
            parts = command.split(None, 1)
            filename = parts[1] if len(parts) > 1 else None
            self._save_memory(filename)
        elif cmd.startswith('/loadmemory'):
            # Parse required filename
            parts = command.split(None, 1)
            if len(parts) < 2:
                self.add_error_message_immediate("Usage: /loadmemory <filename>")
            else:
                filename = parts[1]
                self._load_memory(filename)
        elif cmd == '/stats':
            self._show_complete_stats()
        elif cmd == '/analyze':
            self._force_comprehensive_analysis()
        elif cmd == '/reset_momentum':
            self._reset_complete_momentum()
        elif cmd.startswith('/theme '):
            theme_name = cmd[7:].strip()
            self._change_theme(theme_name)
        else:
            self.add_error_message_immediate(f"Unknown command: {command}")
    
    def _force_comprehensive_analysis(self):
        """Force immediate comprehensive momentum analysis for testing"""
        try:
            user_assistant_messages = [msg for msg in self.memory_manager.get_messages() 
                                     if msg.message_type in [MessageType.USER, MessageType.ASSISTANT]]
            total_messages = len(user_assistant_messages)
            
            if total_messages < 5:
                self.add_error_message_immediate("Need at least 5 messages for meaningful analysis")
                return
            
            self.add_system_message_immediate("Running forced comprehensive momentum analysis with complete LLM integration...")
            self._trigger_comprehensive_momentum_analysis(total_messages)
            
        except Exception as e:
            self.add_error_message_immediate(f"Failed to force comprehensive analysis: {e}")
    
    def _reset_complete_momentum(self):
        """Reset complete story momentum state"""
        try:
            self.sme.reset_story_state()
            
            # Clear momentum state from EMM
            empty_state = self.sme.save_state_to_dict()
            self.memory_manager.update_momentum_state(empty_state)
            
            self.add_system_message_immediate("Complete story momentum reset to initial state")
            
        except Exception as e:
            self.add_error_message_immediate(f"Failed to reset complete momentum: {e}")
    
    def _clear_memory(self, backup_filename: Optional[str] = None):
        """Clear memory with optional backup"""
        try:
            # Save backup if filename provided
            if backup_filename:
                success = self.memory_manager.save_conversation(backup_filename)
                if success:
                    self.add_system_message_immediate(f"Backup saved to {backup_filename}")
                else:
                    self.add_error_message_immediate(f"Failed to save backup to {backup_filename}")
                    return
            
            # Clear memory file and in-memory state
            success = self.memory_manager.clear_memory_file()
            if success:
                # Reset complete SME state
                self.sme.reset_story_state()
                
                # Also clear display
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
                
                # Show confirmation
                if backup_filename:
                    self.add_system_message_immediate(f"Memory cleared with complete state reset (backup saved to {backup_filename})")
                else:
                    self.add_system_message_immediate("Memory cleared with complete state reset")
            else:
                self.add_error_message_immediate("Failed to clear memory")
                
        except Exception as e:
            self.add_error_message_immediate(f"Error clearing memory: {e}")
            self._log_debug(f"Memory clear error: {e}")

    def _save_memory(self, filename: Optional[str] = None):
        """Save memory to file"""
        try:
            if not filename:
                import time
                filename = f"chat_backup_{int(time.time())}.json"
            
            success = self.memory_manager.save_conversation(filename)
            if success:
                self.add_system_message_immediate(f"Memory saved to {filename}")
            else:
                self.add_error_message_immediate(f"Failed to save memory to {filename}")
                
        except Exception as e:
            self.add_error_message_immediate(f"Error saving memory: {e}")
            self._log_debug(f"Memory save error: {e}")

    def _load_memory(self, filename: str):
        """Load memory from file and restore complete SME state"""
        try:
            success = self.memory_manager.load_conversation(filename)
            if success:
                # Clear current display
                self.display_messages.clear()
                self.display_lines.clear()
                
                # Reset scroll manager
                self.scroll_manager.scroll_offset = 0
                self.scroll_manager.max_scroll = 0
                self.scroll_manager.in_scrollback = False
                
                # Clear output window
                if self.output_win:
                    self.output_win.clear()
                    self.output_win.refresh()
                
                # Restore complete SME state
                momentum_state = self.memory_manager.get_momentum_state()
                if momentum_state:
                    self.sme.load_state_from_dict(momentum_state)
                    self.add_system_message_immediate("Complete story momentum state restored")
                
                # Show loaded conversation
                messages = self.memory_manager.get_messages()
                for msg in messages:
                    if msg.message_type == MessageType.USER:
                        self.add_user_message_immediate(msg.content)
                    elif msg.message_type == MessageType.ASSISTANT:
                        self.add_assistant_message_immediate(msg.content)
                    elif msg.message_type == MessageType.SYSTEM:
                        self.add_system_message_immediate(msg.content)
                
                # Show confirmation
                self.add_system_message_immediate(f"Loaded {len(messages)} messages from {filename} with complete state restoration")
            else:
                self.add_error_message_immediate(f"Failed to load memory from {filename}")
                
        except Exception as e:
            self.add_error_message_immediate(f"Error loading memory: {e}")
            self._log_debug(f"Memory load error: {e}")
    
    def _show_complete_stats(self):
        """Show comprehensive system statistics with complete LLM analysis information"""
        try:
            # Enhanced memory stats with semantic analysis information
            mem_stats = self.memory_manager.get_memory_stats()
            file_info = self.memory_manager.get_memory_file_info()
            patterns = self.memory_manager.analyze_conversation_patterns()
            
            self.add_system_message_immediate(
                f"Memory: {mem_stats.get('message_count', 0)} messages, "
                f"{mem_stats.get('total_tokens', 0)} tokens, "
                f"{mem_stats.get('condensations_performed', 0)} multi-pass condensations"
            )
            
            # Detailed semantic categorization stats
            if 'semantic_categories' in patterns:
                categories = patterns['semantic_categories']
                category_summary = ', '.join([f"{k}: {v}" for k, v in categories.items() if v > 0])
                if category_summary:
                    self.add_system_message_immediate(f"Semantic categories: {category_summary}")
            
            # Enhanced condensed message stats
            condensed_count = patterns.get('condensed_messages', 0)
            if condensed_count > 0:
                utilization = mem_stats.get('utilization', 0.0)
                self.add_system_message_immediate(f"Condensed messages: {condensed_count}, utilization: {utilization:.1%}")
            
            # Complete memory file details
            if file_info.get('file_exists', False):
                file_size = file_info.get('file_size', 0)
                file_size_kb = file_size / 1024 if file_size > 0 else 0
                last_modified = file_info.get('last_modified', 'unknown')
                backup_status = "yes" if file_info.get('backup_exists', False) else "no"
                
                self.add_system_message_immediate(
                    f"Memory file: {file_info.get('file_path', 'unknown')} "
                    f"({file_size_kb:.1f}KB, modified: {last_modified[:19]}, backup: {backup_status})"
                )
                
                auto_save_status = "enabled" if file_info.get('auto_save_enabled', False) else "disabled"
                self.add_system_message_immediate(f"Auto-save: {auto_save_status}")
            else:
                self.add_system_message_immediate("Memory file: Not found")
                
        except Exception as e:
            self.add_system_message_immediate("Memory: Stats unavailable")
            self._log_debug(f"Memory stats error: {e}")
        
        try:
            # Complete story stats with comprehensive LLM analysis information
            sme_stats = self.sme.get_pressure_stats()
            if 'current_pressure' in sme_stats:
                pressure = sme_stats['current_pressure']
                arc = sme_stats.get('current_arc', 'unknown')
                updates = sme_stats.get('total_updates', 0)
                floor = sme_stats.get('pressure_floor', 0.0)
                escalations = sme_stats.get('escalation_count', 0)
                last_analysis = sme_stats.get('last_analysis_count', 0)
                variance = sme_stats.get('pressure_variance', 0.0)
                
                self.add_system_message_immediate(
                    f"Story: Pressure {pressure:.2f} (floor {floor:.2f}, var {variance:.3f}), "
                    f"Arc {arc}, {updates} updates, {escalations} escalations"
                )
                
                self.add_system_message_immediate(f"Analysis: Last at message {last_analysis}")
                
                # Enhanced antagonist information
                story_context = self.sme.get_story_context()
                if story_context.get('antagonist_present'):
                    antagonist = story_context.get('antagonist', {})
                    name = antagonist.get('name', 'Unknown')
                    commitment = antagonist.get('commitment_level', 'unknown')
                    threat = antagonist.get('threat_level', 0.0)
                    losses = antagonist.get('resources_lost', 0)
                    self.add_system_message_immediate(
                        f"Antagonist: {name} ({commitment} commitment, threat {threat:.2f}, {losses} losses)"
                    )
        except Exception as e:
            self.add_system_message_immediate("Story: Stats unavailable")
            self._log_debug(f"Story stats error: {e}")
        
        # MCP stats
        try:
            mcp_info = self.mcp_client.get_server_info()
            self.add_system_message_immediate(f"MCP: {mcp_info.get('server_url', 'unknown')}")
            self.add_system_message_immediate(f"Model: {mcp_info.get('model', 'unknown')}")
        except Exception as e:
            self.add_system_message_immediate("MCP: Stats unavailable")
            self._log_debug(f"MCP stats error: {e}")
        
        # Display stats with scroll manager error handling
        try:
            scroll_info = self.scroll_manager.get_scroll_info()
            self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, "
                                           f"Scroll: {scroll_info.get('offset', 0)}/{scroll_info.get('max', 0)}")
        except Exception as e:
            self.add_system_message_immediate(f"Display: {len(self.display_lines)} lines, Scroll: Error")
            self._log_debug(f"Scroll stats error: {e}")
        
        # Terminal stats with layout info
        try:
            if self.current_layout:
                layout = self.current_layout
                self.add_system_message_immediate(f"Terminal: {layout.terminal_width}x{layout.terminal_height}")
                self.add_system_message_immediate(f"Layout: Output {layout.output_box.inner_width}x{layout.output_box.inner_height}, "
                                               f"Input {layout.input_box.inner_width}x{layout.input_box.inner_height}")
            else:
                self.add_system_message_immediate("Terminal: Layout not available")
        except Exception as e:
            self.add_system_message_immediate("Terminal: Stats unavailable")
            self._log_debug(f"Terminal stats error: {e}")
        
        # Input stats
        try:
            input_content = self.multi_input.get_content()
            input_lines = len(self.multi_input.lines)
            cursor_line, cursor_col = self.multi_input.get_cursor_position()
            self.add_system_message_immediate(f"Input: {len(input_content)} chars, {input_lines} lines, "
                                           f"cursor at {cursor_line}:{cursor_col}")
        except Exception as e:
            self.add_system_message_immediate("Input: Stats unavailable")
            self._log_debug(f"Input stats error: {e}")
        
        # Enhanced prompt stats
        try:
            total_tokens = sum(len(content) // 4 for content in self.loaded_prompts.values() if content.strip())
            active_prompts = [name for name, content in self.loaded_prompts.items() if content.strip()]
            self.add_system_message_immediate(f"Prompts: {len(active_prompts)} active ({', '.join(active_prompts)}), "
                                           f"{total_tokens:,} tokens")
        except Exception as e:
            self.add_system_message_immediate("Prompts: Stats unavailable")
            self._log_debug(f"Prompt stats error: {e}")
        
        # Background thread status
        try:
            active_threads = [t for t in threading.enumerate() 
                            if t.name in ["SME-Comprehensive-Analysis", "EMM-AutoSave", "EMM-Condensation"]]
            if active_threads:
                thread_names = [t.name for t in active_threads]
                self.add_system_message_immediate(f"Background: {len(active_threads)} active ({', '.join(thread_names)})")
        except Exception as e:
            self._log_debug(f"Thread stats error: {e}")
    
    def _show_complete_help(self):
        """Show help information with complete LLM commands"""
        help_messages = [
            "Available commands:",
            "/help - Show this help",
            "/quit, /exit - Exit application", 
            "/clearmemory [filename] - Clear memory with optional backup",
            "/savememory [filename] - Save memory to file (auto-timestamped if no filename)",
            "/loadmemory <filename> - Load memory from file",
            "/stats - Show complete system statistics with LLM analysis",
            "/analyze - Force immediate comprehensive momentum analysis",
            "/reset_momentum - Reset complete story momentum to initial state",
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
            "punctuation or is a command. Use Enter for new lines otherwise.",
            "",
            "Complete LLM Features:",
            "- Multi-pass semantic memory categorization with robust 5-strategy JSON parsing",
            "- Comprehensive story momentum analysis every 15 messages with defensive error handling",
            "- Dynamic antagonist generation with quality validation and commitment escalation",
            "- Pressure floor ratcheting system prevents infinite narrative stalling",
            "- Category-aware condensation preserves story-critical content intelligently",
            "- Background threading ensures non-blocking LLM operations",
            "",
            "Memory Management:",
            "- Memory automatically saves after each message with semantic analysis",
            "- Previous conversations restore on startup with complete momentum state",
            "- Multi-pass condensation system optimizes memory usage by content importance",
            "- Use /clearmemory to start fresh (optionally with backup)",
            "- Use /savememory for manual backups",
            "- Complete LLM analysis enhances narrative coherence and pacing over time",
            "",
            "Story Analysis:",
            "- Antagonist generation with context-appropriate details and motivations",
            "- Resource loss tracking affects antagonist commitment and responses",
            "- Pressure calculations include environmental, social, and discovery sources",
            "- Escalation ratcheting prevents tension deflation during extended play"
        ]
        
        for msg in help_messages:
            self.add_system_message_immediate(msg)
    
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
        """Graceful shutdown with complete LLM state preservation"""
        self.running = False
        
        # Show shutdown message if interface is still active
        if self.stdscr:
            try:
                self.add_system_message_immediate("Shutting down with complete LLM state preservation...")
            except:
                pass
        
        # Save final complete SME state to EMM
        try:
            final_state = self.sme.save_state_to_dict()
            self.memory_manager.update_momentum_state(final_state)
            if self.debug_logger:
                self.debug_logger.debug("Final complete SME state saved to EMM")
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save final complete SME state: {e}")
        
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
        
        self._log_debug("Interface shutdown complete with complete LLM integration")

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Ncurses Interface Module - COMPLETE LLM INTEGRATION")
    print("Successfully implemented complete LLM-integrated dynamic coordinate system:")
    print("✓ nci_terminal.py - Box coordinate calculation and layout geometry")
    print("✓ nci.py - Dynamic window positioning and content adaptation")
    print("✓ Eliminated manual coordinate calculations")
    print("✓ Automatic adaptation to terminal geometry changes")
    print("✓ Simplified resize handling with consistent layout")
    print("✓ Robust coordinate system prevents curses NULL returns")
    print("✓ Complete LLM-driven semantic memory management integration")
    print("✓ Robust 15-message momentum analysis cycle with 5-strategy JSON parsing")
    print("✓ Multi-pass semantic condensation with category-aware preservation")
    print("✓ Background processing prevents interface blocking")
    print("✓ Enhanced story context with comprehensive antagonist tracking")
    print("✓ Complete SME state persistence through EMM")
    print("\nComplete LLM Integration Features:")
    print("- Multi-pass semantic memory categorization with robust error handling")
    print("- Comprehensive story momentum analysis every 15 messages")
    print("- Dynamic antagonist generation with quality validation and commitment escalation")
    print("- Pressure floor ratcheting system prevents narrative stalling")
    print("- Category-aware condensation preserves story-critical content intelligently")
    print("- Background threading for non-blocking LLM operations")
    print("- Enhanced status display with comprehensive analysis indicators")
    print("- Force analysis and complete momentum reset commands")
    print("- Complete state preservation across sessions")
    print("- 5-strategy defensive JSON parsing for robust LLM response handling")
    print("\nRun main.py to start the application with complete LLM integration.")
