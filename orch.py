# Chunk 1/5 - orch.py - Header, Imports, and Core Class Definition
# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
Central Hub Orchestrator for DevName RPG Client - CORRECTED VERSION
Coordinates all service modules and contains main program logic
ONLY module that communicates with mcp.py for LLM requests
"""

import asyncio
import threading
import time
import sys
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
        
        # Service modules (spokes)
        self.ui_controller: Optional[NCursesUIController] = None
        self.memory_manager: Optional[EnhancedMemoryManager] = None
        self.momentum_engine: Optional[StoryMomentumEngine] = None
        self.semantic_engine: Optional[SemanticAnalysisEngine] = None
        
        # MCP client (exclusive access)
        self.mcp_client: Optional[mcp.MCPClient] = None
        
        # Analysis threading
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_shutdown_event = threading.Event()
        
        # Constants
        self.ANALYSIS_INTERVAL = 15  # Messages between analysis cycles
        self.ANALYSIS_TIMEOUT = 30.0  # Seconds
        
    def initialize_modules(self) -> bool:
        """Initialize all service modules in corrected dependency order"""
        try:
            self._log_debug("Starting module initialization")
            
            # 1. Memory manager (storage only, no dependencies)
            self.memory_manager = EnhancedMemoryManager(debug_logger=self.debug_logger)
            self._log_debug("Memory manager initialized")
            
            # 2. Semantic engine (analysis only, no dependencies)
            self.semantic_engine = SemanticAnalysisEngine(debug_logger=self.debug_logger)
            self._log_debug("Semantic engine initialized")
            
            # 3. Momentum engine (state tracking only)
            self.momentum_engine = StoryMomentumEngine(debug_logger=self.debug_logger)
            
            # Load SME state from memory if available
            try:
                if hasattr(self.memory_manager, 'get_sme_state'):
                    sme_state = self.memory_manager.get_sme_state()
                    if sme_state and hasattr(self.momentum_engine, 'load_state'):
                        self.momentum_engine.load_state(sme_state)
                        self._log_debug("SME state loaded from memory")
            except Exception as e:
                self._log_debug(f"SME state loading failed (non-critical): {e}")
            
            self._log_debug("Momentum engine initialized")
            
            # 4. MCP client (exclusive orchestrator access)
            self.mcp_client = mcp.MCPClient(debug_logger=self.debug_logger)
            self._configure_mcp_client()
            self._log_debug("MCP client initialized")
            
            # 5. UI controller (depends on orchestrator callback)
            self.ui_controller = NCursesUIController(
                debug_logger=self.debug_logger,
                orchestrator_callback=self.handle_ui_request
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
    
    def _log_debug(self, message: str, category: str = "ORCHESTRATOR"):
        """Debug logging helper with orchestrator category"""
        if self.debug_logger:
            self.debug_logger.debug(message, category)

# Chunk 2/5 - orch.py - Main Program Loop and Background Threading

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
            
            # Run UI - this will handle the main curses loop
            result = self.ui_controller.run()
            
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
    
    def handle_ui_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle requests from UI controller
        Main callback interface between UI and orchestrator
        """
        try:
            request_type = request.get('type', '')
            self._log_debug(f"UI request: {request_type}")
            
            if request_type == 'user_input':
                return self._handle_user_input_request(request)
            elif request_type == 'command':
                return self._handle_command_request(request)
            elif request_type == 'get_messages':
                return self._handle_get_messages_request(request)
            elif request_type == 'get_status':
                return self._handle_status_request(request)
            elif request_type == 'shutdown':
                self.initiate_shutdown()
                return {'status': 'shutdown_initiated'}
            else:
                self._log_debug(f"Unknown UI request type: {request_type}")
                return {'error': f'Unknown request type: {request_type}'}
                
        except Exception as e:
            self._log_debug(f"UI request handling error: {e}")
            return {'error': str(e)}
    
    def handle_service_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle requests from service modules
        Callback interface for service module coordination
        """
        try:
            request_type = request.get('type', '')
            self._log_debug(f"Service request: {request_type}")
            
            if request_type == 'llm_request':
                return self._handle_llm_request(request)
            elif request_type == 'memory_update':
                return self._handle_memory_update(request)
            elif request_type == 'analysis_request':
                return self._handle_analysis_request(request)
            else:
                self._log_debug(f"Unknown service request type: {request_type}")
                return {'error': f'Unknown request type: {request_type}'}
                
        except Exception as e:
            self._log_debug(f"Service request handling error: {e}")
            return {'error': str(e)}
    
    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown"""
        self._log_debug("Shutdown initiated")
        self.state.running = False

# Chunk 3/5 - orch.py - User Input Processing and LLM Communication

    def _handle_user_input_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input through service modules"""
        try:
            user_input = request.get('content', '')
            if not user_input.strip():
                return {'error': 'Empty input'}
            
            self._log_debug(f"Processing user input: {user_input[:50]}...")
            
            # Block further input during processing
            self.state.input_blocked = True
            
            # Add user message to memory
            if self.memory_manager:
                try:
                    self.memory_manager.add_message(user_input, MessageType.USER)
                    self.state.message_count += 1
                    self._log_debug("User message added to memory")
                except Exception as e:
                    self._log_debug(f"Memory add error: {e}")
            
            # Send to LLM for response
            llm_response = self._send_user_message_to_llm(user_input)
            
            if llm_response:
                # Add assistant response to memory
                if self.memory_manager:
                    try:
                        self.memory_manager.add_message(llm_response, MessageType.ASSISTANT)
                        self._log_debug("Assistant message added to memory")
                    except Exception as e:
                        self._log_debug(f"Memory add error: {e}")
                
                # Update momentum state
                if self.momentum_engine:
                    try:
                        if hasattr(self.momentum_engine, 'process_user_input'):
                            self.momentum_engine.process_user_input(user_input, self.state.message_count)
                        self._log_debug("Momentum state updated")
                    except Exception as e:
                        self._log_debug(f"Momentum update error: {e}")
                
                return {
                    'status': 'success',
                    'response': llm_response,
                    'type': 'assistant'
                }
            else:
                return {
                    'status': 'error',
                    'error': 'LLM request failed'
                }
                
        except Exception as e:
            self._log_debug(f"User input processing error: {e}")
            return {'error': str(e)}
        finally:
            self.state.input_blocked = False
    
    def _send_user_message_to_llm(self, user_input: str) -> Optional[str]:
        """Send user message to LLM with full context"""
        try:
            if not self.mcp_client:
                self._log_debug("MCP client not available")
                return None
            
            # Gather context from service modules
            context = self.gather_context_for_llm()
            
            # Get recent conversation history
            recent_messages = []
            if self.memory_manager and hasattr(self.memory_manager, 'get_messages'):
                try:
                    messages = self.memory_manager.get_messages(limit=20)
                    for msg in messages:
                        if hasattr(msg, 'message_type') and hasattr(msg, 'content'):
                            role = 'user' if msg.message_type == MessageType.USER else 'assistant'
                            if msg.message_type in [MessageType.USER, MessageType.ASSISTANT]:
                                recent_messages.append({
                                    'role': role,
                                    'content': str(msg.content)
                                })
                except Exception as e:
                    self._log_debug(f"Error getting conversation history: {e}")
            
            # Extract story context for MCP client
            story_context = context.get('story_context', '')
            
            # Send to LLM using correct MCP client interface
            self._log_debug("Sending request to LLM")
            response = self.mcp_client.send_message(
                user_input=user_input,
                conversation_history=recent_messages,
                story_context=story_context
            )
            
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
        """Gather context from all service modules for LLM requests"""
        context = {}
        
        try:
            # Get story summary from memory if available
            if self.memory_manager and hasattr(self.memory_manager, 'get_story_summary'):
                try:
                    story_summary = self.memory_manager.get_story_summary()
                    if story_summary:
                        context['story_context'] = story_summary
                except Exception as e:
                    self._log_debug(f"Story summary error: {e}")
            
            # Get momentum state if available
            if self.momentum_engine and hasattr(self.momentum_engine, 'get_current_state'):
                try:
                    momentum_state = self.momentum_engine.get_current_state()
                    if momentum_state:
                        context['momentum_state'] = momentum_state
                except Exception as e:
                    self._log_debug(f"Momentum state error: {e}")
            
            # Get semantic categories if available
            if self.semantic_engine and hasattr(self.semantic_engine, 'categorize_message'):
                try:
                    recent_messages = []
                    if self.memory_manager and hasattr(self.memory_manager, 'get_messages'):
                        recent_messages = self.memory_manager.get_messages(limit=5)
                    
                    categories = []
                    for msg in recent_messages:
                        if hasattr(msg, 'content'):
                            category = self.semantic_engine.categorize_message(str(msg.content))
                            if category:
                                categories.append(category)
                    
                    if categories:
                        context['recent_categories'] = categories
                except Exception as e:
                    self._log_debug(f"Semantic categorization error: {e}")
            
            self._log_debug(f"Context gathered: {list(context.keys())}")
            return context
            
        except Exception as e:
            self._log_debug(f"Context gathering error: {e}")
            return {}
    
    def _handle_llm_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM requests from service modules"""
        try:
            messages = request.get('messages', [])
            include_context = request.get('include_context', False)
            
            if not messages:
                return {'error': 'No messages provided'}
            
            # For service module LLM requests, extract user input from messages
            user_input = ""
            conversation_history = []
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'user':
                    user_input = content
                elif role == 'assistant':
                    conversation_history.append(msg)
            
            # Add context if requested
            story_context = ""
            if include_context:
                context = self.gather_context_for_llm()
                story_context = context.get('story_context', '')
            
            # Send to LLM through MCP client
            response = self.mcp_client.send_message(
                user_input=user_input,
                conversation_history=conversation_history,
                story_context=story_context
            )
            
            if response:
                return {
                    'status': 'success',
                    'response': response
                }
            else:
                return {'error': 'LLM request failed'}
                
        except Exception as e:
            self._log_debug(f"LLM request handling error: {e}")
            return {'error': str(e)}

# Chunk 4/5 - orch.py - Command Processing and Analysis Methods

    def _handle_command_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process user commands"""
        try:
            command = request.get('command', '').lower().strip()
            self._log_debug(f"Processing command: {command}")
            
            if command == '/help':
                return self._show_help()
            elif command == '/quit' or command == '/exit':
                self.initiate_shutdown()
                return {'status': 'shutdown_initiated'}
            elif command.startswith('/clearmemory'):
                parts = command.split(None, 1)
                backup_filename = parts[1] if len(parts) > 1 else None
                return self._clear_memory(backup_filename)
            elif command.startswith('/savememory'):
                parts = command.split(None, 1)
                filename = parts[1] if len(parts) > 1 else None
                return self._save_memory(filename)
            elif command.startswith('/loadmemory'):
                parts = command.split(None, 1)
                if len(parts) < 2:
                    return {'error': 'Usage: /loadmemory <filename>'}
                return self._load_memory(parts[1])
            elif command == '/stats':
                return self._show_stats()
            elif command == '/analyze':
                return self._force_analysis()
            elif command == '/reset_momentum':
                return self._reset_momentum()
            elif command.startswith('/theme '):
                theme_name = command[7:].strip()
                return self._change_theme(theme_name)
            else:
                return {'error': f'Unknown command: {command}'}
                
        except Exception as e:
            self._log_debug(f"Command processing error: {e}")
            return {'error': str(e)}
    
    def _handle_get_messages_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get messages from memory for UI display"""
        try:
            limit = request.get('limit', 50)
            
            if self.memory_manager and hasattr(self.memory_manager, 'get_messages'):
                messages = self.memory_manager.get_messages(limit=limit)
                
                # Convert Message objects to dictionaries for UI
                message_dicts = []
                for msg in messages:
                    if hasattr(msg, 'to_dict'):
                        message_dicts.append(msg.to_dict())
                    else:
                        # Fallback for basic message format
                        message_dicts.append({
                            'content': str(msg.content) if hasattr(msg, 'content') else str(msg),
                            'type': msg.message_type.value if hasattr(msg, 'message_type') else 'unknown',
                            'timestamp': getattr(msg, 'timestamp', time.time())
                        })
                
                return {
                    'status': 'success',
                    'messages': message_dicts
                }
            else:
                return {
                    'status': 'success',
                    'messages': []
                }
                
        except Exception as e:
            self._log_debug(f"Get messages error: {e}")
            return {'error': str(e)}
    
    def _handle_status_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get orchestrator status"""
        try:
            return {
                'status': 'success',
                'data': self.get_orchestrator_status()
            }
        except Exception as e:
            self._log_debug(f"Status request error: {e}")
            return {'error': str(e)}
    
    def _handle_memory_update(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory update requests from service modules"""
        try:
            update_type = request.get('update_type', '')
            data = request.get('data', {})
            
            if update_type == 'sme_state' and self.memory_manager:
                if hasattr(self.memory_manager, 'update_sme_state'):
                    self.memory_manager.update_sme_state(data)
                    return {'status': 'success'}
            
            return {'error': f'Unknown memory update type: {update_type}'}
            
        except Exception as e:
            self._log_debug(f"Memory update error: {e}")
            return {'error': str(e)}
    
    def _handle_analysis_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis requests from service modules"""
        try:
            analysis_type = request.get('analysis_type', '')
            
            if analysis_type == 'trigger_periodic':
                self.trigger_periodic_analysis()
                return {'status': 'success'}
            
            return {'error': f'Unknown analysis type: {analysis_type}'}
            
        except Exception as e:
            self._log_debug(f"Analysis request error: {e}")
            return {'error': str(e)}
    
    def trigger_periodic_analysis(self) -> None:
        """Trigger periodic semantic and momentum analysis"""
        if self.state.analysis_in_progress:
            self._log_debug("Analysis already in progress, skipping")
            return
        
        self.state.analysis_in_progress = True
        self.state.last_analysis_time = time.time()
        
        try:
            self._log_debug("Starting periodic analysis")
            
            # Get messages for analysis
            messages_for_analysis = []
            if self.memory_manager and hasattr(self.memory_manager, 'get_messages'):
                try:
                    messages_for_analysis = self.memory_manager.get_messages(limit=50)
                except Exception as e:
                    self._log_debug(f"Error getting messages for analysis: {e}")
            
            if not messages_for_analysis:
                self._log_debug("No messages available for analysis")
                return
            
            # Semantic analysis through semantic engine
            if self.semantic_engine:
                try:
                    self._perform_semantic_analysis(messages_for_analysis)
                except Exception as e:
                    self._log_debug(f"Semantic analysis error: {e}")
            
            # Momentum analysis through momentum engine
            if self.momentum_engine:
                try:
                    self._perform_momentum_analysis(messages_for_analysis)
                except Exception as e:
                    self._log_debug(f"Momentum analysis error: {e}")
            
            self._log_debug("Periodic analysis completed")
            
        except Exception as e:
            self._log_debug(f"Periodic analysis error: {e}")
        finally:
            self.state.analysis_in_progress = False
    
    def _perform_semantic_analysis(self, messages: List[Any]) -> None:
        """Perform semantic analysis through semantic engine"""
        try:
            self._log_debug("Starting semantic analysis")
            
            # Basic semantic categorization without LLM for now
            # Full implementation would coordinate through semantic engine
            for msg in messages[-10:]:  # Analyze last 10 messages
                if hasattr(msg, 'content') and self.semantic_engine:
                    try:
                        if hasattr(self.semantic_engine, 'categorize_message'):
                            category = self.semantic_engine.categorize_message(str(msg.content))
                            self._log_debug(f"Message categorized as: {category}")
                    except Exception as e:
                        self._log_debug(f"Message categorization error: {e}")
            
        except Exception as e:
            self._log_debug(f"Semantic analysis error: {e}")
    
    def _perform_momentum_analysis(self, messages: List[Any]) -> None:
        """Perform momentum analysis through momentum engine"""
        try:
            self._log_debug("Starting momentum analysis")
            
            # Update momentum based on recent messages
            if self.momentum_engine and hasattr(self.momentum_engine, 'process_user_input'):
                for i, msg in enumerate(messages[-5:]):  # Process last 5 messages
                    if hasattr(msg, 'content') and hasattr(msg, 'message_type'):
                        if msg.message_type == MessageType.USER:
                            try:
                                self.momentum_engine.process_user_input(str(msg.content), i)
                            except Exception as e:
                                self._log_debug(f"Momentum processing error: {e}")
            
        except Exception as e:
            self._log_debug(f"Momentum analysis error: {e}")
    
    def _show_help(self) -> Dict[str, Any]:
        """Display help information"""
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
        
        return {
            'status': 'success',
            'response': help_content,
            'type': 'system'
        }
    
    def _show_stats(self) -> Dict[str, Any]:
        """Display comprehensive statistics"""
        try:
            stats = []
            
            # Memory statistics
            if self.memory_manager and hasattr(self.memory_manager, 'get_statistics'):
                try:
                    memory_stats = self.memory_manager.get_statistics()
                    stats.append(f"Memory: {memory_stats.get('total_messages', 0)} messages")
                    stats.append(f"Storage: {memory_stats.get('storage_size', 0)} bytes")
                except Exception as e:
                    stats.append(f"Memory: Error getting stats - {e}")
            
            # Momentum statistics
            if self.momentum_engine and hasattr(self.momentum_engine, 'get_statistics'):
                try:
                    momentum_stats = self.momentum_engine.get_statistics()
                    stats.append(f"Momentum: {momentum_stats.get('current_pressure', 0):.2f}")
                    stats.append(f"Antagonist: {momentum_stats.get('antagonist_name', 'None')}")
                except Exception as e:
                    stats.append(f"Momentum: Error getting stats - {e}")
            
            # Analysis statistics
            stats.append(f"Message count: {self.state.message_count}")
            stats.append(f"Last analysis: {time.time() - self.state.last_analysis_time:.1f}s ago")
            stats.append(f"Analysis in progress: {self.state.analysis_in_progress}")
            
            # MCP statistics
            if self.mcp_client and hasattr(self.mcp_client, 'get_statistics'):
                try:
                    mcp_stats = self.mcp_client.get_statistics()
                    stats.append(f"LLM requests: {mcp_stats.get('total_requests', 0)}")
                    stats.append(f"LLM failures: {mcp_stats.get('failed_requests', 0)}")
                except Exception as e:
                    stats.append(f"MCP: Error getting stats - {e}")
            
            stats_content = "DevName RPG Client Statistics:\n" + "\n".join(f"• {stat}" for stat in stats)
            
            return {
                'status': 'success',
                'response': stats_content,
                'type': 'system'
            }
                
        except Exception as e:
            self._log_debug(f"Stats display error: {e}")
            return {'error': 'Failed to display statistics'}

# Chunk 5/5 - orch.py - Command Implementations and Shutdown Logic

    def _force_analysis(self) -> Dict[str, Any]:
        """Force immediate comprehensive analysis"""
        try:
            self._log_debug("Forcing immediate analysis")
            
            if self.state.analysis_in_progress:
                return {'error': 'Analysis already in progress'}
            
            # Trigger analysis regardless of message count
            self.trigger_periodic_analysis()
            
            return {
                'status': 'success',
                'response': 'Comprehensive analysis initiated manually.',
                'type': 'system'
            }
                
        except Exception as e:
            self._log_debug(f"Force analysis error: {e}")
            return {'error': 'Failed to force analysis'}
    
    def _reset_momentum(self) -> Dict[str, Any]:
        """Reset story momentum state"""
        try:
            if self.momentum_engine and hasattr(self.momentum_engine, 'reset_state'):
                self.momentum_engine.reset_state()
                
            if self.memory_manager and hasattr(self.memory_manager, 'clear_sme_state'):
                self.memory_manager.clear_sme_state()
            
            return {
                'status': 'success',
                'response': 'Story momentum state has been reset.',
                'type': 'system'
            }
                
        except Exception as e:
            self._log_debug(f"Momentum reset error: {e}")
            return {'error': 'Failed to reset momentum'}
    
    def _clear_memory(self, backup_filename: Optional[str] = None) -> Dict[str, Any]:
        """Clear memory with optional backup"""
        try:
            if backup_filename and self.memory_manager:
                # Save backup first
                if hasattr(self.memory_manager, 'save_to_file'):
                    if not self.memory_manager.save_to_file(backup_filename):
                        return {'error': f'Failed to create backup: {backup_filename}'}
            
            if self.memory_manager and hasattr(self.memory_manager, 'clear_all'):
                self.memory_manager.clear_all()
            
            if self.momentum_engine and hasattr(self.momentum_engine, 'reset_state'):
                self.momentum_engine.reset_state()
            
            backup_msg = f' with backup: {backup_filename}' if backup_filename else ''
            return {
                'status': 'success',
                'response': f'Memory cleared{backup_msg}.',
                'type': 'system'
            }
                
        except Exception as e:
            self._log_debug(f"Clear memory error: {e}")
            return {'error': 'Failed to clear memory'}
    
    def _save_memory(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Save memory to file"""
        try:
            if not self.memory_manager:
                return {'error': 'Memory manager not available'}
            
            actual_filename = filename or f"memory_save_{int(time.time())}.json"
            
            if hasattr(self.memory_manager, 'save_to_file'):
                if self.memory_manager.save_to_file(actual_filename):
                    return {
                        'status': 'success',
                        'response': f'Memory saved to: {actual_filename}',
                        'type': 'system'
                    }
                else:
                    return {'error': f'Failed to save memory to: {actual_filename}'}
            else:
                return {'error': 'Memory manager does not support file saving'}
                
        except Exception as e:
            self._log_debug(f"Save memory error: {e}")
            return {'error': 'Failed to save memory'}
    
    def _load_memory(self, filename: str) -> Dict[str, Any]:
        """Load memory from file"""
        try:
            if not self.memory_manager:
                return {'error': 'Memory manager not available'}
            
            if hasattr(self.memory_manager, 'load_from_file'):
                if self.memory_manager.load_from_file(filename):
                    # Reload SME state from restored memory
                    if self.momentum_engine and hasattr(self.memory_manager, 'get_sme_state'):
                        try:
                            sme_state = self.memory_manager.get_sme_state()
                            if sme_state and hasattr(self.momentum_engine, 'load_state'):
                                self.momentum_engine.load_state(sme_state)
                        except Exception as e:
                            self._log_debug(f"SME state reload error: {e}")
                    
                    return {
                        'status': 'success',
                        'response': f'Memory loaded from: {filename}',
                        'type': 'system'
                    }
                else:
                    return {'error': f'Failed to load memory from: {filename}'}
            else:
                return {'error': 'Memory manager does not support file loading'}
                
        except Exception as e:
            self._log_debug(f"Load memory error: {e}")
            return {'error': 'Failed to load memory'}
    
    def _change_theme(self, theme_name: str) -> Dict[str, Any]:
        """Change UI color theme"""
        try:
            if self.ui_controller and hasattr(self.ui_controller, 'change_theme'):
                success = self.ui_controller.change_theme(theme_name)
                
                if success:
                    return {
                        'status': 'success',
                        'response': f'Theme changed to: {theme_name}',
                        'type': 'system'
                    }
                else:
                    return {'error': f'Invalid theme: {theme_name}'}
            else:
                return {'error': 'UI controller does not support theme changes'}
                
        except Exception as e:
            self._log_debug(f"Theme change error: {e}")
            return {'error': 'Failed to change theme'}
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status for UI display"""
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
                'mcp_client': self.mcp_client is not None
            }
            
            return status
            
        except Exception as e:
            self._log_debug(f"Status retrieval error: {e}")
            return {'error': str(e)}
    
    def shutdown_gracefully(self) -> None:
        """Graceful shutdown of all service modules"""
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
                    if hasattr(self.momentum_engine, 'get_current_state') and hasattr(self.memory_manager, 'update_sme_state'):
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
                    if hasattr(self.ui_controller, 'shutdown'):
                        self.ui_controller.shutdown()
                    self._log_debug("UI controller shutdown complete")
                except Exception as e:
                    self._log_debug(f"UI controller shutdown error: {e}")
            
            # 2. MCP client cleanup
            if self.mcp_client:
                try:
                    if hasattr(self.mcp_client, 'cleanup'):
                        self.mcp_client.cleanup()
                    elif hasattr(self.mcp_client, 'shutdown'):
                        self.mcp_client.shutdown()
                    self._log_debug("MCP client cleanup complete")
                except Exception as e:
                    self._log_debug(f"MCP client cleanup error: {e}")
            
            # 3. Momentum engine shutdown
            if self.momentum_engine:
                try:
                    if hasattr(self.momentum_engine, 'shutdown'):
                        self.momentum_engine.shutdown()
                    self._log_debug("Momentum engine shutdown complete")
                except Exception as e:
                    self._log_debug(f"Momentum engine shutdown error: {e}")
            
            # 4. Semantic engine shutdown
            if self.semantic_engine:
                try:
                    if hasattr(self.semantic_engine, 'shutdown'):
                        self.semantic_engine.shutdown()
                    self._log_debug("Semantic engine shutdown complete")
                except Exception as e:
                    self._log_debug(f"Semantic engine shutdown error: {e}")
            
            # 5. Memory manager final save and shutdown
            if self.memory_manager:
                try:
                    if hasattr(self.memory_manager, 'force_save'):
                        self.memory_manager.force_save()  # Final auto-save
                    if hasattr(self.memory_manager, 'shutdown'):
                        self.memory_manager.shutdown()
                    self._log_debug("Memory manager shutdown complete")
                except Exception as e:
                    self._log_debug(f"Memory manager shutdown error: {e}")
            
            self.state.modules_initialized = False
            self.state.ui_initialized = False
            
            self._log_debug("Graceful shutdown completed")
            
        except Exception as e:
            self._log_debug(f"Shutdown error: {e}")


# End of orch.py - DevName RPG Client Central Orchestrator (CORRECTED)
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
