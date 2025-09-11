# Chunk 1/5 - orch.py - Header, Imports, and Core Class Definition (Debug Logger Fix)

#!/usr/bin/env python3
"""
Central Hub Orchestrator for DevName RPG Client - CORRECTED VERSION
Coordinates all service modules and contains main program logic
ONLY module that communicates with mcp.py for LLM requests
FIXED: Debug logger interface consistency - uses callable pattern
"""

import asyncio
import threading
import time
import sys
import curses
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Service module imports (spoke modules) - direct imports from current directory
try:
    from ncui import NCursesUIController
except ImportError as e:
    print(f"Failed to import ncui: {e}")
    sys.exit(1)

try:
    from emm import EnhancedMemoryManager, MessageType
except ImportError as e:
    print(f"Failed to import emm: {e}")
    sys.exit(1)

try:
    from sme import StoryMomentumEngine
except ImportError as e:
    print(f"Failed to import sme: {e}")
    sys.exit(1)

try:
    from sem import SemanticAnalysisEngine
except ImportError as e:
    print(f"Failed to import sem: {e}")
    sys.exit(1)

try:
    import mcp
except ImportError as e:
    print(f"Failed to import mcp: {e}")
    sys.exit(1)

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
    Central Hub Orchestrator - CORRECTED VERSION
    
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
        
        # Analysis configuration
        self.ANALYSIS_INTERVAL = 15  # Trigger analysis every 15 messages
        self.analysis_shutdown_event = threading.Event()
        self.analysis_thread = None
        
        # Service modules (spoke modules)
        self.memory_manager = None
        self.momentum_engine = None
        self.semantic_engine = None
        self.ui_controller = None
        self.mcp_client = None
        
        self._log_debug("Orchestrator created")

# Chunk 2/5 - orch.py - Module Initialization (Debug Logger Fix)

    def initialize_modules(self) -> bool:
        """
        Initialize all service modules in correct dependency order
        Hub-and-spoke pattern: orchestrator coordinates all modules
        """
        try:
            self._log_debug("Starting module initialization")
            
            # 1. Enhanced Memory Manager (storage only, no dependencies)
            self.memory_manager = EnhancedMemoryManager(debug_logger=self.debug_logger)
            self._log_debug("Memory manager initialized")
            
            # 2. Semantic Analysis Engine (analysis only, no dependencies)
            self.semantic_engine = SemanticAnalysisEngine(debug_logger=self.debug_logger)
            self._log_debug("Semantic engine initialized")
            
            # 3. Story Momentum Engine (state tracking with threading.Lock fix)
            self.momentum_engine = StoryMomentumEngine(debug_logger=self.debug_logger)
            self._log_debug("Momentum engine initialized")
            
            # 4. MCP client (exclusive orchestrator access)
            self.mcp_client = mcp.MCPClient(debug_logger=self.debug_logger)
            self._configure_mcp_client()
            self._log_debug("MCP client initialized")
            
            # 5. UI controller (depends on orchestrator callback)
            self.ui_controller = NCursesUIController(
                orchestrator_callback=self.handle_ui_request,
                debug_logger=self.debug_logger
            )
            self._log_debug("UI controller initialized")
            
            # Set orchestrator callbacks for service modules that support them
            if hasattr(self.memory_manager, 'set_orchestrator_callback'):
                self.memory_manager.set_orchestrator_callback(self.handle_service_request)
            
            if hasattr(self.semantic_engine, 'set_orchestrator_callback'):
                self.semantic_engine.set_orchestrator_callback(self.handle_service_request)
            
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
            
            # Run UI using curses wrapper - this will handle the main curses loop
            def ui_wrapper(stdscr):
                try:
                    # Initialize UI controller with curses screen
                    if self.ui_controller.initialize(stdscr):
                        self.state.ui_initialized = True
                        self._log_debug("UI initialized successfully")
                        
                        # Run the main UI loop
                        self.ui_controller.run()
                        return 0
                    else:
                        self._log_debug("UI initialization failed")
                        return 1
                        
                except Exception as e:
                    self._log_debug(f"UI wrapper error: {e}")
                    return 1
            
            # Use curses wrapper to handle UI
            result = curses.wrapper(ui_wrapper)
            
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

# Chunk 3/5 - orch.py - UI Request Handling (Debug Logger Fix)

    def handle_ui_request(self, action: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle requests from UI controller
        Main callback interface between UI and orchestrator
        """
        try:
            self._log_debug(f"UI request: {action}")
            
            if action == 'user_input':
                return self._handle_user_input(data)
            elif action == 'command':
                return self._handle_command(data)
            elif action == 'get_messages':
                return self._handle_get_messages(data)
            elif action == 'get_status':
                return self._handle_get_status(data)
            elif action == 'shutdown':
                self.initiate_shutdown()
                return {'status': 'shutdown_initiated'}
            else:
                self._log_debug(f"Unknown UI request: {action}")
                return {'error': f'Unknown request: {action}'}
                
        except Exception as e:
            self._log_debug(f"UI request handling error: {e}")
            return {'error': str(e)}
    
    def _handle_user_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user text input through service modules"""
        try:
            user_text = data.get('text', '').strip()
            if not user_text:
                return {'error': 'Empty input'}
            
            self._log_debug(f"Processing user input: {len(user_text)} chars")
            
            # Increment message count
            self.state.message_count += 1
            
            # Store user message in memory manager
            if self.memory_manager:
                self.memory_manager.add_message(user_text, MessageType.USER)
            
            # Check if this is a command
            if user_text.startswith('/'):
                return self._handle_command({'command': user_text})
            
            # Process through semantic analysis
            if self.semantic_engine:
                analysis = self.semantic_engine.analyze_message(user_text, 'user')
                self._log_debug(f"Semantic analysis: {analysis.get('category', 'unknown')}")
            
            # Update momentum engine
            if self.momentum_engine:
                self.momentum_engine.update_momentum(user_text, 'user')
            
            # Generate LLM response through MCP client
            response = self._generate_llm_response(user_text)
            
            if response.get('success'):
                # Store assistant response in memory
                if self.memory_manager:
                    self.memory_manager.add_message(response['content'], MessageType.ASSISTANT)
                
                # Update UI with response
                if self.ui_controller:
                    self.ui_controller.add_message({
                        'content': response['content'],
                        'type': 'assistant'
                    })
                    self.ui_controller.set_processing_state(False)
                
                return {'success': True, 'response': response['content']}
            else:
                # Handle error response
                error_msg = response.get('error', 'Unknown error occurred')
                if self.ui_controller:
                    self.ui_controller.add_message({
                        'content': f"Error: {error_msg}",
                        'type': 'error'
                    })
                    self.ui_controller.set_processing_state(False)
                
                return {'error': error_msg}
                
        except Exception as e:
            self._log_debug(f"User input handling error: {e}")
            return {'error': str(e)}
    
    def _handle_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle slash commands"""
        try:
            command = data.get('command', '').strip()
            if not command.startswith('/'):
                return {'error': 'Invalid command format'}
            
            cmd_parts = command[1:].split()
            cmd_name = cmd_parts[0].lower() if cmd_parts else ''
            
            self._log_debug(f"Processing command: {cmd_name}")
            
            if cmd_name == 'help':
                return self._handle_help_command()
            elif cmd_name == 'stats':
                return self._handle_stats_command()
            elif cmd_name == 'analyze':
                return self._handle_analyze_command()
            elif cmd_name == 'theme':
                theme_num = int(cmd_parts[1]) if len(cmd_parts) > 1 else 1
                return self._handle_theme_command(theme_num)
            elif cmd_name == 'clearmemory':
                return self._handle_clear_memory_command()
            elif cmd_name == 'quit' or cmd_name == 'exit':
                self.initiate_shutdown()
                return {'status': 'shutdown_initiated'}
            else:
                return {'error': f'Unknown command: {cmd_name}'}
                
        except Exception as e:
            self._log_debug(f"Command handling error: {e}")
            return {'error': str(e)}
    
    def _handle_help_command(self) -> Dict[str, Any]:
        """Handle help command"""
        help_text = """DevName RPG Client - Available Commands:

/help           - Show this help message
/stats          - Display system statistics
/analyze        - Trigger immediate semantic analysis
/theme <n>      - Switch color theme (1-4)
/clearmemory    - Clear conversation memory
/quit or /exit  - Exit the application

Navigation:
- Arrow Keys    - Scroll through message history
- PgUp/PgDn     - Page through messages quickly
- Home/End      - Jump to top/bottom of history
- Double Enter  - Submit multi-line input"""

        if self.ui_controller:
            self.ui_controller.add_message({
                'content': help_text,
                'type': 'system'
            })
        
        return {'success': True, 'content': help_text}
    
    def _handle_stats_command(self) -> Dict[str, Any]:
        """Handle stats command"""
        try:
            stats = []
            
            # Orchestrator stats
            stats.append(f"Orchestrator Status: {'Running' if self.state.running else 'Stopped'}")
            stats.append(f"Messages Processed: {self.state.message_count}")
            stats.append(f"Analysis Interval: {self.ANALYSIS_INTERVAL} messages")
            
            # Memory manager stats
            if self.memory_manager:
                memory_stats = self.memory_manager.get_stats()
                stats.append(f"Memory: {memory_stats.get('total_messages', 0)} messages stored")
            
            # UI stats
            if self.ui_controller:
                ui_stats = self.ui_controller.get_stats()
                stats.append(f"Display Buffer: {ui_stats.get('display_buffer_size', 0)} messages")
                stats.append(f"Terminal: {ui_stats.get('terminal_size', 'unknown')}")
            
            # Module status
            stats.append("\nModule Status:")
            stats.append(f"  Memory Manager: {'✓' if self.memory_manager else '✗'}")
            stats.append(f"  Semantic Engine: {'✓' if self.semantic_engine else '✗'}")
            stats.append(f"  Momentum Engine: {'✓' if self.momentum_engine else '✗'}")
            stats.append(f"  MCP Client: {'✓' if self.mcp_client else '✗'}")
            stats.append(f"  UI Controller: {'✓' if self.ui_controller else '✗'}")
            
            stats_text = '\n'.join(stats)
            
            if self.ui_controller:
                self.ui_controller.add_message({
                    'content': stats_text,
                    'type': 'system'
                })
            
            return {'success': True, 'content': stats_text}
            
        except Exception as e:
            self._log_debug(f"Stats command error: {e}")
            return {'error': str(e)}

# Chunk 4/5 - orch.py - LLM Communication and Analysis (Debug Logger Fix)

    def _handle_analyze_command(self) -> Dict[str, Any]:
        """Handle analyze command - trigger immediate analysis"""
        try:
            self.trigger_periodic_analysis()
            return {'success': True, 'content': 'Analysis triggered'}
        except Exception as e:
            self._log_debug(f"Analyze command error: {e}")
            return {'error': str(e)}
    
    def _handle_theme_command(self, theme_number: int) -> Dict[str, Any]:
        """Handle theme switching command"""
        try:
            if self.ui_controller and hasattr(self.ui_controller, 'switch_theme'):
                if self.ui_controller.switch_theme(theme_number):
                    return {'success': True, 'content': f'Switched to theme {theme_number}'}
                else:
                    return {'error': f'Invalid theme: {theme_number}'}
            else:
                return {'error': 'UI controller does not support theme changes'}
                
        except Exception as e:
            self._log_debug(f"Theme change error: {e}")
            return {'error': 'Failed to change theme'}
    
    def _handle_clear_memory_command(self) -> Dict[str, Any]:
        """Handle clear memory command"""
        try:
            if self.memory_manager:
                self.memory_manager.clear_memory()
                
            if self.ui_controller:
                self.ui_controller.clear_display()
                self.ui_controller.add_message({
                    'content': 'Memory and display cleared',
                    'type': 'system'
                })
            
            self.state.message_count = 0
            return {'success': True, 'content': 'Memory cleared'}
            
        except Exception as e:
            self._log_debug(f"Clear memory error: {e}")
            return {'error': str(e)}
    
    def _handle_get_messages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request for message history"""
        try:
            if not self.memory_manager:
                return {'error': 'Memory manager not available'}
            
            limit = data.get('limit', 50)
            messages = self.memory_manager.get_recent_messages(limit)
            
            return {
                'success': True,
                'messages': [msg.to_dict() for msg in messages]
            }
            
        except Exception as e:
            self._log_debug(f"Get messages error: {e}")
            return {'error': str(e)}
    
    def _handle_get_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request"""
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
            
            return {'success': True, 'status': status}
            
        except Exception as e:
            self._log_debug(f"Status request error: {e}")
            return {'error': str(e)}
    
    def _generate_llm_response(self, user_input: str) -> Dict[str, Any]:
        """
        Generate LLM response through MCP client
        EXCLUSIVE orchestrator access to mcp.py
        """
        try:
            if not self.mcp_client:
                return {'error': 'MCP client not available'}
            
            self._log_debug("Generating LLM response via MCP")
            
            # Gather context from service modules
            context = self._gather_context()
            
            # Build full prompt with context
            full_prompt = self._build_contextual_prompt(user_input, context)
            
            # Make LLM request through MCP client
            response = self.mcp_client.send_request(full_prompt)
            
            if response.get('success'):
                self._log_debug("LLM response received successfully")
                return {
                    'success': True,
                    'content': response['content']
                }
            else:
                self._log_debug(f"LLM request failed: {response.get('error')}")
                return {
                    'error': response.get('error', 'LLM request failed')
                }
                
        except Exception as e:
            self._log_debug(f"LLM response generation error: {e}")
            return {'error': str(e)}
    
    def _gather_context(self) -> Dict[str, Any]:
        """Gather context from all service modules"""
        context = {}
        
        try:
            # Memory context
            if self.memory_manager:
                recent_messages = self.memory_manager.get_recent_messages(10)
                context['recent_messages'] = [msg.to_dict() for msg in recent_messages]
                context['memory_stats'] = self.memory_manager.get_stats()
            
            # Momentum context
            if self.momentum_engine:
                context['momentum'] = self.momentum_engine.get_current_state()
            
            # Semantic context
            if self.semantic_engine:
                context['semantic_state'] = self.semantic_engine.get_analysis_summary()
            
        except Exception as e:
            self._log_debug(f"Context gathering error: {e}")
        
        return context
    
    def _build_contextual_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Build full prompt with context for LLM"""
        try:
            prompt_parts = []
            
            # Base system prompt
            if self.loaded_prompts.get('critrules'):
                prompt_parts.append(self.loaded_prompts['critrules'])
            
            # Add companion prompt if available
            if self.loaded_prompts.get('companion'):
                prompt_parts.append(self.loaded_prompts['companion'])
            
            # Add lowrules prompt if available
            if self.loaded_prompts.get('lowrules'):
                prompt_parts.append(self.loaded_prompts['lowrules'])
            
            # Add context information
            if context.get('recent_messages'):
                prompt_parts.append("Recent conversation context:")
                for msg in context['recent_messages'][-5:]:  # Last 5 messages
                    prompt_parts.append(f"{msg['type']}: {msg['content']}")
            
            # Add momentum context
            if context.get('momentum'):
                momentum = context['momentum']
                prompt_parts.append(f"Story momentum: {momentum.get('level', 'unknown')}")
            
            # Current user input
            prompt_parts.append(f"User: {user_input}")
            prompt_parts.append("Assistant:")
            
            return '\n\n'.join(prompt_parts)
            
        except Exception as e:
            self._log_debug(f"Prompt building error: {e}")
            return user_input
    
    def trigger_periodic_analysis(self):
        """Trigger periodic analysis of conversation"""
        try:
            if self.state.analysis_in_progress:
                return
            
            self.state.analysis_in_progress = True
            self.state.last_analysis_time = time.time()
            
            self._log_debug("Starting periodic analysis")
            
            # Run analysis in background thread to avoid blocking UI
            analysis_thread = threading.Thread(
                target=self._run_analysis_cycle,
                name="PeriodicAnalysis",
                daemon=True
            )
            analysis_thread.start()
            
        except Exception as e:
            self._log_debug(f"Analysis trigger error: {e}")
            self.state.analysis_in_progress = False
    
    def _run_analysis_cycle(self):
        """Run complete analysis cycle"""
        try:
            # Gather messages for analysis
            if self.memory_manager:
                recent_messages = self.memory_manager.get_recent_messages(self.ANALYSIS_INTERVAL)
                
                # Run semantic analysis
                if self.semantic_engine:
                    for msg in recent_messages:
                        self.semantic_engine.analyze_message(msg.content, msg.message_type.value)
                
                # Update momentum
                if self.momentum_engine:
                    self.momentum_engine.analyze_conversation_momentum(recent_messages)
            
            self._log_debug("Analysis cycle completed")
            
        except Exception as e:
            self._log_debug(f"Analysis cycle error: {e}")
        finally:
            self.state.analysis_in_progress = False

# Chunk 5/5 - orch.py - Service Requests and Shutdown (Debug Logger Fix)

    def handle_service_request(self, module_name: str, request_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle requests from service modules
        Enables service modules to request orchestrator coordination
        """
        try:
            self._log_debug(f"Service request from {module_name}: {request_type}")
            
            if request_type == 'memory_save':
                return self._handle_memory_save_request(data)
            elif request_type == 'analysis_request':
                return self._handle_analysis_request(data)
            elif request_type == 'llm_request':
                # Only orchestrator can make LLM requests
                return self._generate_llm_response(data.get('prompt', ''))
            elif request_type == 'ui_update':
                return self._handle_ui_update_request(data)
            else:
                self._log_debug(f"Unknown service request: {request_type}")
                return {'error': f'Unknown request type: {request_type}'}
                
        except Exception as e:
            self._log_debug(f"Service request handling error: {e}")
            return {'error': str(e)}
    
    def _handle_memory_save_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory save requests from service modules"""
        try:
            if self.memory_manager:
                filename = data.get('filename', 'memory.json')
                success = self.memory_manager.save_conversation(filename)
                return {'success': success}
            else:
                return {'error': 'Memory manager not available'}
                
        except Exception as e:
            self._log_debug(f"Memory save request error: {e}")
            return {'error': str(e)}
    
    def _handle_analysis_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis requests from service modules"""
        try:
            analysis_type = data.get('type', 'general')
            
            if analysis_type == 'semantic':
                if self.semantic_engine:
                    text = data.get('text', '')
                    msg_type = data.get('message_type', 'user')
                    result = self.semantic_engine.analyze_message(text, msg_type)
                    return {'success': True, 'analysis': result}
                else:
                    return {'error': 'Semantic engine not available'}
            
            elif analysis_type == 'momentum':
                if self.momentum_engine:
                    result = self.momentum_engine.get_current_state()
                    return {'success': True, 'momentum': result}
                else:
                    return {'error': 'Momentum engine not available'}
            
            else:
                return {'error': f'Unknown analysis type: {analysis_type}'}
                
        except Exception as e:
            self._log_debug(f"Analysis request error: {e}")
            return {'error': str(e)}
    
    def _handle_ui_update_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle UI update requests from service modules"""
        try:
            if not self.ui_controller:
                return {'error': 'UI controller not available'}
            
            update_type = data.get('type', '')
            
            if update_type == 'add_message':
                message = data.get('message', {})
                self.ui_controller.add_message(message)
                return {'success': True}
            
            elif update_type == 'update_status':
                status = data.get('status', '')
                self.ui_controller.update_status(status)
                return {'success': True}
            
            elif update_type == 'set_processing':
                processing = data.get('processing', False)
                self.ui_controller.set_processing_state(processing)
                return {'success': True}
            
            else:
                return {'error': f'Unknown UI update type: {update_type}'}
                
        except Exception as e:
            self._log_debug(f"UI update request error: {e}")
            return {'error': str(e)}
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown sequence"""
        try:
            self._log_debug("Initiating graceful shutdown")
            self.state.running = False
            
            # Signal UI to stop
            if self.ui_controller:
                self.ui_controller.shutdown()
            
        except Exception as e:
            self._log_debug(f"Shutdown initiation error: {e}")
    
    def shutdown_gracefully(self):
        """Perform graceful shutdown of all modules"""
        try:
            self._log_debug("Starting graceful shutdown")
            
            # Stop analysis thread
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_shutdown_event.set()
                self._log_debug("Waiting for analysis thread to complete")
                self.analysis_thread.join(timeout=2.0)
            
            # Shutdown UI controller
            if self.ui_controller:
                try:
                    self.ui_controller.shutdown()
                    self._log_debug("UI controller shutdown complete")
                except Exception as e:
                    self._log_debug(f"UI controller shutdown error: {e}")
            
            # Shutdown MCP client
            if self.mcp_client:
                try:
                    self.mcp_client.cleanup()
                    self._log_debug("MCP client cleanup complete")
                except Exception as e:
                    self._log_debug(f"MCP client cleanup error: {e}")
            
            # Shutdown momentum engine
            if self.momentum_engine:
                try:
                    self.momentum_engine.shutdown()
                    self._log_debug("Momentum engine shutdown complete")
                except Exception as e:
                    self._log_debug(f"Momentum engine shutdown error: {e}")
            
            # Shutdown semantic engine
            if self.semantic_engine:
                try:
                    self.semantic_engine.shutdown()
                    self._log_debug("Semantic engine shutdown complete")
                except Exception as e:
                    self._log_debug(f"Semantic engine shutdown error: {e}")
            
            # Shutdown memory manager (save final state)
            if self.memory_manager:
                try:
                    self.memory_manager.save_conversation()
                    self.memory_manager.shutdown()
                    self._log_debug("Memory manager shutdown complete")
                except Exception as e:
                    self._log_debug(f"Memory manager shutdown error: {e}")
            
            self._log_debug("Graceful shutdown completed")
            
        except Exception as e:
            self._log_debug(f"Shutdown error: {e}")
        finally:
            self.state.running = False
    
    def _log_debug(self, message: str):
        """Debug logging helper - FIXED: uses callable interface"""
        if self.debug_logger:
            # Use callable interface to match ncui.py expectations
            self.debug_logger(f"[ORCHESTRATOR] {message}")

# =============================================================================
# STANDALONE TESTING
# =============================================================================

def test_orchestrator():
    """Test function for standalone orchestrator testing"""
    def mock_logger(message: str):
        print(f"Debug: {message}")
    
    try:
        config = {
            'mcp': {
                'server_url': 'http://localhost:3001/v1/chat/completions',
                'model': 'llama3.2',
                'timeout': 30
            }
        }
        
        prompts = {
            'critrules': 'Test system prompt for RPG client',
            'lowrules': 'Additional test rules'
        }
        
        orchestrator = Orchestrator(config, prompts, mock_logger)
        
        if orchestrator.initialize_modules():
            print("Orchestrator test: All modules initialized successfully")
            
            # Test status
            status = orchestrator.get_orchestrator_status()
            print(f"Orchestrator status: {status}")
            
        else:
            print("Orchestrator test: Module initialization failed")
            
    except Exception as e:
        print(f"Orchestrator test error: {e}")
    finally:
        print("Orchestrator test complete")

if __name__ == "__main__":
    test_orchestrator()
