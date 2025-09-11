# Chunk 1/4 - Central Orchestrator Hub
# orch.py - DevName RPG Client Central Orchestrator
"""
Central Hub Orchestrator for DevName RPG Client
Coordinates all service modules and contains main program logic
ONLY module that communicates with mcp.py for LLM requests
"""

import asyncio
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

# Service module imports (spoke modules)
from remod_staging.ncui import NCursesUIController
from remod_staging.emm import EnhancedMemoryManager
from remod_staging.sme import StoryMomentumEngine
from remod_staging.sem import SemanticAnalysisEngine
from remod_staging.uilib import TerminalManager
import mcp

@dataclass
class OrchestrationState:
    """Central state management for orchestrator"""
    running: bool = False
    input_blocked: bool = False
    analysis_in_progress: bool = False
    ui_initialized: bool = False
    modules_initialized: bool = False
    last_analysis_time: float = 0.0
    message_count: int = 0
    
class Orchestrator:
    """
    Central Hub Orchestrator
    
    Responsibilities:
    - Initialize and coordinate all service modules
    - Main program loop coordination
    - ONLY module that calls mcp.py for LLM requests
    - Process user input through service modules
    - Trigger periodic analysis and updates
    - Manage graceful shutdown
    """
    
    def __init__(self, config: Dict[str, Any], loaded_prompts: Dict[str, str], debug_logger=None):
        """Initialize orchestrator with configuration and prompts"""
        self.config = config
        self.loaded_prompts = loaded_prompts
        self.debug_logger = debug_logger
        self.state = OrchestrationState()
        
        # Service modules (spokes)
        self.ui_controller: Optional[NCursesUIController] = None
        self.memory_manager: Optional[EnhancedMemoryManager] = None
        self.momentum_engine: Optional[StoryMomentumEngine] = None
        self.semantic_engine: Optional[SemanticAnalysisEngine] = None
        self.terminal_manager: Optional[TerminalManager] = None
        
        # MCP client (exclusive access)
        self.mcp_client: Optional[mcp.MCPClient] = None
        
        # Analysis threading
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_shutdown_event = threading.Event()
        
        # Constants
        self.ANALYSIS_INTERVAL = 15  # Messages between analysis cycles
        self.ANALYSIS_TIMEOUT = 30.0  # Seconds
        
    def initialize_modules(self) -> bool:
        """Initialize all service modules in dependency order"""
        try:
            self._log_debug("Starting module initialization")
            
            # 1. Terminal manager (no dependencies)
            self.terminal_manager = TerminalManager()
            if self.terminal_manager.is_too_small():
                self._log_debug("Terminal too small for initialization")
                return False
            
            # 2. Memory manager (minimal dependencies)
            self.memory_manager = EnhancedMemoryManager(debug_logger=self.debug_logger)
            if not self.memory_manager.initialize():
                self._log_debug("Memory manager initialization failed")
                return False
            
            # 3. Semantic engine (no LLM dependencies)
            self.semantic_engine = SemanticAnalysisEngine(debug_logger=self.debug_logger)
            
            # 4. Momentum engine (state management only)
            self.momentum_engine = StoryMomentumEngine(debug_logger=self.debug_logger)
            
            # Load SME state from memory if available
            sme_state = self.memory_manager.get_sme_state()
            if sme_state:
                self.momentum_engine.load_state(sme_state)
                self._log_debug("SME state loaded from memory")
            
            # 5. MCP client (exclusive orchestrator access)
            self.mcp_client = mcp.MCPClient(debug_logger=self.debug_logger)
            self._configure_mcp_client()
            
            # 6. UI controller (depends on other modules for callbacks)
            self.ui_controller = NCursesUIController(
                config=self.config,
                debug_logger=self.debug_logger,
                terminal_manager=self.terminal_manager,
                # UI callback handlers
                input_callback=self.process_user_input,
                command_callback=self.process_command,
                resize_callback=self.handle_resize,
                shutdown_callback=self.initiate_shutdown
            )
            
            self.state.modules_initialized = True
            self._log_debug("All modules initialized successfully")
            return True
            
        except Exception as e:
            self._log_debug(f"Module initialization failed: {e}")
            return False
    
    def _configure_mcp_client(self):
        """Configure MCP client from config and prompts"""
        if not self.mcp_client:
            return
        
        # Configure from config
        mcp_config = self.config.get('mcp', {})
        if 'server_url' in mcp_config:
            self.mcp_client.server_url = mcp_config['server_url']
        if 'model' in mcp_config:
            self.mcp_client.model = mcp_config['model']
        if 'timeout' in mcp_config:
            self.mcp_client.timeout = mcp_config['timeout']
        
        # Set base system prompt from loaded prompts
        if self.loaded_prompts.get('critrules'):
            self.mcp_client.system_prompt = self.loaded_prompts['critrules']
            self._log_debug("Base system prompt configured from critrules")
    
    def run_main_loop(self) -> int:
        """
        Main program loop coordination
        Delegates UI management to ncui while retaining control flow
        """
        try:
            if not self.state.modules_initialized:
                self._log_debug("Modules not initialized")
                return 1
            
            self.state.running = True
            self._log_debug("Starting main orchestration loop")
            
            # Start background analysis thread
            self._start_analysis_thread()
            
            # Initialize UI and transfer control to UI main loop
            # UI controller will call back to orchestrator for business logic
            result = self.ui_controller.run_interface()
            
            self._log_debug("UI main loop ended")
            return result
            
        except Exception as e:
            self._log_debug(f"Main loop error: {e}")
            return 1
        finally:
            self.shutdown_gracefully()
    
    def _start_analysis_thread(self):
        """Start background analysis thread for periodic processing"""
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            name="AnalysisWorker",
            daemon=True
        )
        self.analysis_thread.start()
        self._log_debug("Background analysis thread started")
    
    def _analysis_worker(self):
        """Background worker for periodic analysis"""
        while self.state.running and not self.analysis_shutdown_event.is_set():
            try:
                time.sleep(1.0)  # Check every second
                
                # Check if analysis is due
                if self._should_trigger_analysis():
                    self.trigger_periodic_analysis()
                
            except Exception as e:
                self._log_debug(f"Analysis worker error: {e}")
    
    def _should_trigger_analysis(self) -> bool:
        """Check if periodic analysis should be triggered"""
        if self.state.analysis_in_progress:
            return False
        
        if self.state.message_count % self.ANALYSIS_INTERVAL == 0 and self.state.message_count > 0:
            return True
        
        return False

# Chunk 2/4 - Core Business Logic
# orch.py - DevName RPG Client Central Orchestrator (continued)

    def process_user_input(self, input_text: str) -> bool:
        """
        Process user input through service modules
        Called by UI controller via callback
        Returns True if input was processed successfully
        """
        try:
            if not input_text.strip():
                return False
            
            self._log_debug(f"Processing user input: {input_text[:50]}...")
            
            # Block further input during processing
            self.state.input_blocked = True
            
            # Add user message to memory
            user_message = {
                "role": "user", 
                "content": input_text.strip(),
                "timestamp": time.time()
            }
            self.memory_manager.add_message(user_message)
            
            # Update message count for analysis triggers
            self.state.message_count += 1
            
            # Send to LLM for response
            llm_response = self._send_user_message_to_llm(input_text)
            
            if llm_response:
                # Add assistant response to memory
                assistant_message = {
                    "role": "assistant",
                    "content": llm_response,
                    "timestamp": time.time()
                }
                self.memory_manager.add_message(assistant_message)
                
                # Update UI with assistant response
                if self.ui_controller:
                    self.ui_controller.add_message(assistant_message)
                
                # Update momentum state
                self.momentum_engine.update_momentum_state({
                    "user_input": input_text,
                    "assistant_response": llm_response
                })
                
                self._log_debug("User input processed successfully")
                return True
            else:
                # Handle LLM failure
                error_msg = {
                    "role": "system",
                    "content": "LLM request failed. Please try again.",
                    "timestamp": time.time()
                }
                if self.ui_controller:
                    self.ui_controller.add_error_message(error_msg)
                
                self._log_debug("LLM request failed for user input")
                return False
                
        except Exception as e:
            self._log_debug(f"User input processing error: {e}")
            return False
        finally:
            self.state.input_blocked = False
    
    def process_command(self, command: str) -> bool:
        """
        Process user commands
        Called by UI controller via callback
        """
        try:
            cmd = command.lower().strip()
            self._log_debug(f"Processing command: {cmd}")
            
            if cmd == '/help':
                self._show_help()
            elif cmd == '/quit' or cmd == '/exit':
                self.initiate_shutdown()
            elif cmd.startswith('/clearmemory'):
                parts = command.split(None, 1)
                backup_filename = parts[1] if len(parts) > 1 else None
                self._clear_memory(backup_filename)
            elif cmd.startswith('/savememory'):
                parts = command.split(None, 1)
                filename = parts[1] if len(parts) > 1 else None
                self._save_memory(filename)
            elif cmd.startswith('/loadmemory'):
                parts = command.split(None, 1)
                if len(parts) < 2:
                    self._send_error_to_ui("Usage: /loadmemory <filename>")
                else:
                    self._load_memory(parts[1])
            elif cmd == '/stats':
                self._show_stats()
            elif cmd == '/analyze':
                self._force_analysis()
            elif cmd == '/reset_momentum':
                self._reset_momentum()
            elif cmd.startswith('/theme '):
                theme_name = cmd[7:].strip()
                self._change_theme(theme_name)
            else:
                self._send_error_to_ui(f"Unknown command: {command}")
                return False
            
            return True
            
        except Exception as e:
            self._log_debug(f"Command processing error: {e}")
            self._send_error_to_ui(f"Command error: {e}")
            return False
    
    def _send_user_message_to_llm(self, user_input: str) -> Optional[str]:
        """
        Send user message to LLM with full context
        ONLY method that calls mcp.py - exclusive LLM access
        """
        try:
            if not self.mcp_client:
                self._log_debug("MCP client not available")
                return None
            
            # Gather context from service modules
            context = self.gather_context_for_llm()
            
            # Build system messages with prompts and context
            system_messages = self._build_system_messages(context.get('story_context', ''))
            
            # Get recent conversation history
            recent_messages = self.memory_manager.get_recent_messages(limit=20)
            
            # Combine system + conversation + current input
            all_messages = system_messages + recent_messages + [
                {"role": "user", "content": user_input}
            ]
            
            # Send to LLM
            self._log_debug("Sending request to LLM")
            response = self.mcp_client.send_message_sync(all_messages)
            
            if response:
                self._log_debug("LLM response received")
                return response
            else:
                self._log_debug("LLM returned empty response")
                return None
                
        except Exception as e:
            self._log_debug(f"LLM request error: {e}")
            return None
    
    def gather_context_for_llm(self) -> Dict[str, Any]:
        """
        Gather context from all service modules for LLM requests
        Coordinates information from memory, momentum, and semantic engines
        """
        context = {}
        
        try:
            # Get story context from memory
            if self.memory_manager:
                story_summary = self.memory_manager.get_story_summary()
                if story_summary:
                    context['story_context'] = story_summary
            
            # Get momentum state
            if self.momentum_engine:
                momentum_state = self.momentum_engine.get_current_state()
                if momentum_state:
                    context['momentum_state'] = momentum_state
            
            # Get semantic categorization hints
            if self.semantic_engine:
                recent_messages = self.memory_manager.get_recent_messages(limit=5)
                if recent_messages:
                    categories = []
                    for msg in recent_messages:
                        category = self.semantic_engine.categorize_message(msg.get('content', ''))
                        if category:
                            categories.append(category)
                    if categories:
                        context['recent_categories'] = categories
            
            self._log_debug(f"Context gathered: {list(context.keys())}")
            return context
            
        except Exception as e:
            self._log_debug(f"Context gathering error: {e}")
            return {}
    
    def _build_system_messages(self, story_context: str) -> List[Dict[str, str]]:
        """Build system messages with integrated prompts and story context"""
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
    
    def trigger_periodic_analysis(self) -> None:
        """
        Trigger periodic semantic and momentum analysis
        Coordinates analysis through service modules
        """
        if self.state.analysis_in_progress:
            self._log_debug("Analysis already in progress, skipping")
            return
        
        self.state.analysis_in_progress = True
        self.state.last_analysis_time = time.time()
        
        try:
            self._log_debug("Starting periodic analysis")
            
            # Get messages for analysis
            messages_for_analysis = self.memory_manager.get_messages_for_analysis()
            
            if not messages_for_analysis:
                self._log_debug("No messages available for analysis")
                return
            
            # Semantic analysis through semantic engine
            if self.semantic_engine:
                condensation_needed = self.semantic_engine.check_condensation_needed(messages_for_analysis)
                
                if condensation_needed:
                    self._perform_semantic_condensation(messages_for_analysis)
            
            # Momentum analysis through momentum engine and LLM
            if self.momentum_engine:
                self._perform_momentum_analysis(messages_for_analysis)
            
            # Update UI with analysis results
            if self.ui_controller:
                self.ui_controller.update_analysis_status("Analysis complete")
            
            self._log_debug("Periodic analysis completed")
            
        except Exception as e:
            self._log_debug(f"Periodic analysis error: {e}")
        finally:
            self.state.analysis_in_progress = False

# Chunk 3/4 - Analysis and Command Methods
# orch.py - DevName RPG Client Central Orchestrator (continued)

    def _perform_semantic_condensation(self, messages: List[Dict[str, Any]]) -> None:
        """Perform semantic condensation through semantic engine and LLM"""
        try:
            self._log_debug("Starting semantic condensation")
            
            # Prepare condensation request through semantic engine
            condensation_request = self.semantic_engine.prepare_condensation_request(
                messages, target_tokens=2000
            )
            
            if not condensation_request:
                self._log_debug("No condensation request prepared")
                return
            
            # Send condensation request to LLM (orchestrator exclusive access)
            condensation_messages = [
                {"role": "system", "content": condensation_request}
            ]
            
            condensed_content = self.mcp_client.send_message_sync(condensation_messages)
            
            if condensed_content:
                # Process condensation result through semantic engine
                success = self.semantic_engine.process_condensation_result(
                    condensed_content, messages
                )
                
                if success:
                    # Update memory with condensed content
                    self.memory_manager.apply_condensation(condensed_content)
                    self._log_debug("Semantic condensation completed successfully")
                else:
                    self._log_debug("Condensation result processing failed")
            else:
                self._log_debug("LLM condensation request failed")
                
        except Exception as e:
            self._log_debug(f"Semantic condensation error: {e}")
    
    def _perform_momentum_analysis(self, messages: List[Dict[str, Any]]) -> None:
        """Perform momentum analysis through momentum engine and LLM"""
        try:
            self._log_debug("Starting momentum analysis")
            
            # Prepare momentum analysis request through momentum engine
            analysis_request = self.momentum_engine.prepare_analysis_request(messages)
            
            if not analysis_request:
                self._log_debug("No momentum analysis request prepared")
                return
            
            # Send analysis request to LLM (orchestrator exclusive access)
            analysis_messages = [
                {"role": "system", "content": analysis_request}
            ]
            
            analysis_result = self.mcp_client.send_message_sync(analysis_messages)
            
            if analysis_result:
                # Process analysis result through momentum engine
                momentum_data = self.momentum_engine.process_analysis_result(analysis_result)
                
                if momentum_data:
                    # Update memory with momentum state
                    self.memory_manager.update_sme_state(momentum_data)
                    self._log_debug("Momentum analysis completed successfully")
                else:
                    self._log_debug("Momentum analysis result processing failed")
            else:
                self._log_debug("LLM momentum analysis request failed")
                
        except Exception as e:
            self._log_debug(f"Momentum analysis error: {e}")
    
    # Command implementation methods
    
    def _show_help(self) -> None:
        """Display help information through UI"""
        help_content = """
DevName RPG Client - Commands:
/help - Show this help message
/quit, /exit - Exit the application
/stats - Show comprehensive statistics
/analyze - Force immediate analysis
/reset_momentum - Reset story momentum state
/clearmemory [backup_file] - Clear memory with optional backup
/savememory [filename] - Save memory to file
/loadmemory <filename> - Load memory from file
/theme <name> - Change color theme

During conversation:
- Type messages and press Enter to send
- Use Ctrl+C to interrupt/exit
- Terminal will auto-resize dynamically
"""
        
        help_message = {
            "role": "system",
            "content": help_content,
            "timestamp": time.time()
        }
        
        if self.ui_controller:
            self.ui_controller.add_message(help_message)
    
    def _show_stats(self) -> None:
        """Display comprehensive statistics through UI"""
        try:
            stats = []
            
            # Memory statistics
            if self.memory_manager:
                memory_stats = self.memory_manager.get_statistics()
                stats.append(f"Memory: {memory_stats.get('total_messages', 0)} messages")
                stats.append(f"Storage: {memory_stats.get('storage_size', 0)} bytes")
            
            # Momentum statistics
            if self.momentum_engine:
                momentum_stats = self.momentum_engine.get_statistics()
                stats.append(f"Momentum: {momentum_stats.get('current_pressure', 0):.2f}")
                stats.append(f"Antagonist: {momentum_stats.get('antagonist_name', 'None')}")
            
            # Analysis statistics
            stats.append(f"Message count: {self.state.message_count}")
            stats.append(f"Last analysis: {time.time() - self.state.last_analysis_time:.1f}s ago")
            stats.append(f"Analysis in progress: {self.state.analysis_in_progress}")
            
            # MCP statistics
            if self.mcp_client:
                mcp_stats = self.mcp_client.get_statistics()
                stats.append(f"LLM requests: {mcp_stats.get('total_requests', 0)}")
                stats.append(f"LLM failures: {mcp_stats.get('failed_requests', 0)}")
            
            stats_content = "DevName RPG Client Statistics:\n" + "\n".join(f"• {stat}" for stat in stats)
            
            stats_message = {
                "role": "system", 
                "content": stats_content,
                "timestamp": time.time()
            }
            
            if self.ui_controller:
                self.ui_controller.add_message(stats_message)
                
        except Exception as e:
            self._log_debug(f"Stats display error: {e}")
            self._send_error_to_ui("Failed to display statistics")
    
    def _force_analysis(self) -> None:
        """Force immediate comprehensive analysis"""
        try:
            self._log_debug("Forcing immediate analysis")
            
            if self.state.analysis_in_progress:
                self._send_error_to_ui("Analysis already in progress")
                return
            
            # Trigger analysis regardless of message count
            self.trigger_periodic_analysis()
            
            analysis_message = {
                "role": "system",
                "content": "Comprehensive analysis initiated manually.",
                "timestamp": time.time()
            }
            
            if self.ui_controller:
                self.ui_controller.add_message(analysis_message)
                
        except Exception as e:
            self._log_debug(f"Force analysis error: {e}")
            self._send_error_to_ui("Failed to force analysis")
    
    def _reset_momentum(self) -> None:
        """Reset story momentum state"""
        try:
            if self.momentum_engine:
                self.momentum_engine.reset_state()
                
            if self.memory_manager:
                self.memory_manager.clear_sme_state()
            
            reset_message = {
                "role": "system",
                "content": "Story momentum state has been reset.",
                "timestamp": time.time()
            }
            
            if self.ui_controller:
                self.ui_controller.add_message(reset_message)
                
            self._log_debug("Momentum state reset completed")
            
        except Exception as e:
            self._log_debug(f"Momentum reset error: {e}")
            self._send_error_to_ui("Failed to reset momentum")
    
    def _clear_memory(self, backup_filename: Optional[str] = None) -> None:
        """Clear memory with optional backup"""
        try:
            if backup_filename and self.memory_manager:
                # Save backup first
                if not self.memory_manager.save_to_file(backup_filename):
                    self._send_error_to_ui(f"Failed to create backup: {backup_filename}")
                    return
            
            if self.memory_manager:
                self.memory_manager.clear_all()
            
            if self.momentum_engine:
                self.momentum_engine.reset_state()
            
            clear_message = {
                "role": "system",
                "content": f"Memory cleared{'with backup: ' + backup_filename if backup_filename else ''}.",
                "timestamp": time.time()
            }
            
            if self.ui_controller:
                self.ui_controller.add_message(clear_message)
                
            self._log_debug(f"Memory cleared {backup_filename or ''}")
            
        except Exception as e:
            self._log_debug(f"Clear memory error: {e}")
            self._send_error_to_ui("Failed to clear memory")
    
    def _save_memory(self, filename: Optional[str] = None) -> None:
        """Save memory to file"""
        try:
            if not self.memory_manager:
                self._send_error_to_ui("Memory manager not available")
                return
            
            actual_filename = filename or f"memory_save_{int(time.time())}.json"
            
            if self.memory_manager.save_to_file(actual_filename):
                save_message = {
                    "role": "system",
                    "content": f"Memory saved to: {actual_filename}",
                    "timestamp": time.time()
                }
                
                if self.ui_controller:
                    self.ui_controller.add_message(save_message)
                    
                self._log_debug(f"Memory saved to {actual_filename}")
            else:
                self._send_error_to_ui(f"Failed to save memory to: {actual_filename}")
                
        except Exception as e:
            self._log_debug(f"Save memory error: {e}")
            self._send_error_to_ui("Failed to save memory")
    
    def _load_memory(self, filename: str) -> None:
        """Load memory from file"""
        try:
            if not self.memory_manager:
                self._send_error_to_ui("Memory manager not available")
                return
            
            if self.memory_manager.load_from_file(filename):
                # Reload SME state from restored memory
                if self.momentum_engine:
                    sme_state = self.memory_manager.get_sme_state()
                    if sme_state:
                        self.momentum_engine.load_state(sme_state)
                
                load_message = {
                    "role": "system",
                    "content": f"Memory loaded from: {filename}",
                    "timestamp": time.time()
                }
                
                if self.ui_controller:
                    self.ui_controller.add_message(load_message)
                    
                self._log_debug(f"Memory loaded from {filename}")
            else:
                self._send_error_to_ui(f"Failed to load memory from: {filename}")
                
        except Exception as e:
            self._log_debug(f"Load memory error: {e}")
            self._send_error_to_ui("Failed to load memory")
    
    def _change_theme(self, theme_name: str) -> None:
        """Change UI color theme"""
        try:
            if self.ui_controller:
                success = self.ui_controller.change_theme(theme_name)
                
                if success:
                    theme_message = {
                        "role": "system",
                        "content": f"Theme changed to: {theme_name}",
                        "timestamp": time.time()
                    }
                    self.ui_controller.add_message(theme_message)
                    self._log_debug(f"Theme changed to {theme_name}")
                else:
                    self._send_error_to_ui(f"Invalid theme: {theme_name}")
            else:
                self._send_error_to_ui("UI controller not available")
                
        except Exception as e:
            self._log_debug(f"Theme change error: {e}")
            self._send_error_to_ui("Failed to change theme")
    
    def _send_error_to_ui(self, error_message: str) -> None:
        """Send error message to UI"""
        error_msg = {
            "role": "system",
            "content": f"Error: {error_message}",
            "timestamp": time.time()
        }
        
        if self.ui_controller:
            self.ui_controller.add_error_message(error_msg)
    
    # Callback handlers for UI controller
    
    def handle_resize(self) -> None:
        """Handle terminal resize event"""
        self._log_debug("Handling terminal resize")
        # UI controller handles the actual resize, orchestrator just logs
    
    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown"""
        self._log_debug("Shutdown initiated")
        self.state.running = False

# Chunk 4/4 - Shutdown and Utilities
# orch.py - DevName RPG Client Central Orchestrator (continued)

    def shutdown_gracefully(self) -> None:
        """
        Graceful shutdown of all service modules
        Ensures data persistence and proper cleanup
        """
        try:
            self._log_debug("Starting graceful shutdown")
            self.state.running = False
            
            # Signal analysis thread to stop
            if self.analysis_shutdown_event:
                self.analysis_shutdown_event.set()
            
            # Wait for analysis thread to complete
            if self.analysis_thread and self.analysis_thread.is_alive():
                self._log_debug("Waiting for analysis thread to complete")
                self.analysis_thread.join(timeout=5.0)
                
                if self.analysis_thread.is_alive():
                    self._log_debug("Analysis thread did not complete within timeout")
            
            # Save current state to memory before shutdown
            if self.momentum_engine and self.memory_manager:
                try:
                    current_sme_state = self.momentum_engine.get_current_state()
                    if current_sme_state:
                        self.memory_manager.update_sme_state(current_sme_state)
                        self._log_debug("SME state saved to memory before shutdown")
                except Exception as e:
                    self._log_debug(f"Failed to save SME state before shutdown: {e}")
            
            # Shutdown service modules in reverse initialization order
            
            # 1. UI controller shutdown
            if self.ui_controller:
                try:
                    self.ui_controller.shutdown()
                    self._log_debug("UI controller shutdown complete")
                except Exception as e:
                    self._log_debug(f"UI controller shutdown error: {e}")
            
            # 2. MCP client cleanup
            if self.mcp_client:
                try:
                    self.mcp_client.cleanup()
                    self._log_debug("MCP client cleanup complete")
                except Exception as e:
                    self._log_debug(f"MCP client cleanup error: {e}")
            
            # 3. Momentum engine shutdown
            if self.momentum_engine:
                try:
                    self.momentum_engine.shutdown()
                    self._log_debug("Momentum engine shutdown complete")
                except Exception as e:
                    self._log_debug(f"Momentum engine shutdown error: {e}")
            
            # 4. Semantic engine shutdown
            if self.semantic_engine:
                try:
                    self.semantic_engine.shutdown()
                    self._log_debug("Semantic engine shutdown complete")
                except Exception as e:
                    self._log_debug(f"Semantic engine shutdown error: {e}")
            
            # 5. Memory manager final save and shutdown
            if self.memory_manager:
                try:
                    self.memory_manager.force_save()  # Final auto-save
                    self.memory_manager.shutdown()
                    self._log_debug("Memory manager shutdown complete")
                except Exception as e:
                    self._log_debug(f"Memory manager shutdown error: {e}")
            
            # 6. Terminal manager cleanup
            if self.terminal_manager:
                try:
                    self.terminal_manager.cleanup()
                    self._log_debug("Terminal manager cleanup complete")
                except Exception as e:
                    self._log_debug(f"Terminal manager cleanup error: {e}")
            
            self.state.modules_initialized = False
            self.state.ui_initialized = False
            
            self._log_debug("Graceful shutdown completed")
            
        except Exception as e:
            self._log_debug(f"Shutdown error: {e}")
    
    def update_memory_and_state(self, data: Dict[str, Any]) -> None:
        """
        Update memory and state from external sources
        Used by UI controller for direct updates
        """
        try:
            if 'message' in data and self.memory_manager:
                self.memory_manager.add_message(data['message'])
                self.state.message_count += 1
            
            if 'momentum_update' in data and self.momentum_engine:
                self.momentum_engine.update_momentum_state(data['momentum_update'])
            
            if 'sme_state' in data and self.memory_manager:
                self.memory_manager.update_sme_state(data['sme_state'])
            
            self._log_debug("Memory and state updated from external data")
            
        except Exception as e:
            self._log_debug(f"Memory/state update error: {e}")
    
    def send_llm_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """
        Public interface for LLM requests from service modules
        Maintains exclusive orchestrator access to mcp.py
        """
        try:
            if not self.mcp_client:
                self._log_debug("MCP client not available for request")
                return None
            
            # Validate request structure
            if 'messages' not in request_data:
                self._log_debug("Invalid LLM request: missing messages")
                return None
            
            messages = request_data['messages']
            if not isinstance(messages, list):
                self._log_debug("Invalid LLM request: messages not a list")
                return None
            
            # Add context if requested
            if request_data.get('include_context', False):
                context = self.gather_context_for_llm()
                
                # Prepend story context if available
                if context.get('story_context'):
                    context_message = {
                        "role": "system",
                        "content": f"Story Context: {context['story_context']}"
                    }
                    messages = [context_message] + messages
            
            # Send request through MCP client
            self._log_debug("Sending coordinated LLM request")
            response = self.mcp_client.send_message_sync(messages)
            
            if response:
                self._log_debug("LLM request completed successfully")
                return response
            else:
                self._log_debug("LLM request returned empty response")
                return None
                
        except Exception as e:
            self._log_debug(f"LLM request coordination error: {e}")
            return None
    
    def handle_llm_response(self, response: str) -> bool:
        """
        Handle LLM response processing
        Coordinates response through relevant service modules
        """
        try:
            if not response:
                return False
            
            # Process through semantic engine for categorization
            if self.semantic_engine:
                category = self.semantic_engine.categorize_message(response)
                self._log_debug(f"LLM response categorized as: {category}")
            
            # Update momentum based on response
            if self.momentum_engine:
                momentum_update = {
                    "llm_response": response,
                    "timestamp": time.time()
                }
                self.momentum_engine.update_momentum_state(momentum_update)
            
            self._log_debug("LLM response processing completed")
            return True
            
        except Exception as e:
            self._log_debug(f"LLM response handling error: {e}")
            return False
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Get current orchestrator status for UI display
        Provides centralized status information
        """
        try:
            status = {
                'running': self.state.running,
                'modules_initialized': self.state.modules_initialized,
                'ui_initialized': self.state.ui_initialized,
                'analysis_in_progress': self.state.analysis_in_progress,
                'input_blocked': self.state.input_blocked,
                'message_count': self.state.message_count,
                'last_analysis_time': self.state.last_analysis_time,
                'time_since_analysis': time.time() - self.state.last_analysis_time,
                'next_analysis_in': self.ANALYSIS_INTERVAL - (self.state.message_count % self.ANALYSIS_INTERVAL)
            }
            
            # Add module status
            status['modules'] = {
                'memory_manager': self.memory_manager is not None,
                'momentum_engine': self.momentum_engine is not None,
                'semantic_engine': self.semantic_engine is not None,
                'ui_controller': self.ui_controller is not None,
                'mcp_client': self.mcp_client is not None,
                'terminal_manager': self.terminal_manager is not None
            }
            
            return status
            
        except Exception as e:
            self._log_debug(f"Status retrieval error: {e}")
            return {'error': str(e)}
    
    def _log_debug(self, message: str, category: str = "ORCHESTRATOR"):
        """Debug logging helper with orchestrator category"""
        if self.debug_logger:
            self.debug_logger.debug(message, category)


# End of orch.py - DevName RPG Client Central Orchestrator
# 
# Usage Example:
# 
# from orch import Orchestrator
# 
# # Initialize orchestrator
# config = {...}  # Application configuration
# prompts = {...}  # Loaded prompt files
# orchestrator = Orchestrator(config, prompts, debug_logger)
# 
# # Initialize all service modules
# if orchestrator.initialize_modules():
#     # Run main program loop
#     exit_code = orchestrator.run_main_loop()
# else:
#     exit_code = 1
# 
# # Orchestrator handles graceful shutdown automatically
