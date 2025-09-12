# Chunk 1/5 - orch.py - Header, Imports, and Initial Class Setup (Debug Logger Fix)

#!/usr/bin/env python3
"""
DevName RPG Client - Orchestrator Module (orch.py)
Hub-and-spoke coordination for all service modules with standardized debug logging
FIXED: All debug logger calls use method pattern with null safety checks
"""

import threading
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass, field

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import service modules - orchestrator coordinates everything
try:
    import mcp
    from emm import EnhancedMemoryManager, MessageType
    from sme import StoryMomentumEngine  
    from sem import SemanticAnalysisEngine
    from ncui import NCursesUIController
except ImportError as e:
    print(f"Failed to import required service module: {e}")
    print("Ensure all remodularized module files are present in current directory")
    print("Required files: mcp.py, emm.py, sme.py, sem.py, ncui.py")
    raise

# =============================================================================
# ORCHESTRATOR STATE MANAGEMENT
# =============================================================================

class OrchestrationPhase(Enum):
    """Current operational phase of the orchestrator"""
    INITIALIZING = "initializing"
    ACTIVE = "active" 
    ANALYZING = "analyzing"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class OrchestrationState:
    """Central state tracking for orchestrator operations"""
    phase: OrchestrationPhase = OrchestrationPhase.INITIALIZING
    message_count: int = 0
    last_analysis_count: int = 0
    analysis_in_progress: bool = False
    startup_complete: bool = False
    
    # Analysis tracking
    pending_analysis_requests: List[str] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    
    # Threading coordination
    background_threads: List[threading.Thread] = field(default_factory=list)
    shutdown_requested: bool = False

# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================

class Orchestrator:
    """
    Central coordination hub for all service modules.
    Implements hub-and-spoke pattern - no direct module-to-module communication.
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
            
            # 5. UI Controller (receives orchestrator callback)
            self.ui_controller = NCursesUIController(
                orchestrator_callback=self._handle_ui_callback,
                debug_logger=self.debug_logger
            )
            self._log_debug("UI controller initialized")
            
            # 6. Set orchestrator callbacks for all modules that need them
            self._setup_module_callbacks()
            
            # 7. Start background threads
            self._start_background_services()
            
            self.state.phase = OrchestrationPhase.ACTIVE
            self.state.startup_complete = True
            
            self._log_debug("All modules initialized successfully")
            return True
            
        except Exception as e:
            self._log_error(f"Module initialization failed: {e}")
            return False
    
    def _configure_mcp_client(self):
        """Configure MCP client with loaded prompts and settings"""
        if not self.mcp_client:
            return

        try:
            # Set system prompt from critrules (primary prompt)
            if self.loaded_prompts.get('critrules'):
                self.mcp_client.system_prompt = self.loaded_prompts['critrules']
                self._log_debug("System prompt set from critrules")

            # Configure server settings from config using direct property assignment
            mcp_config = self.config.get('mcp', {})
            if 'server_url' in mcp_config:
                self.mcp_client.server_url = mcp_config['server_url']
            if 'model' in mcp_config:
                self.mcp_client.model = mcp_config['model']
            if 'timeout' in mcp_config:
                self.mcp_client.timeout = mcp_config['timeout']

            self._log_debug("MCP client configured with prompts and settings")

        except Exception as e:
            self._log_error(f"MCP client configuration failed: {e}")
    
    def _setup_module_callbacks(self):
        """Set up orchestrator callbacks for modules that need cross-module communication"""
        try:
            # Memory manager needs no callbacks (storage only)
            
            # Semantic engine needs no callbacks (analysis only)
            
            # Momentum engine needs no callbacks (state tracking only)
            
            # UI controller callback already set in constructor
            
            self._log_debug("Module callbacks configured")
            
        except Exception as e:
            self._log_error(f"Callback setup failed: {e}")
    
    def _start_background_services(self):
        """Start background threads for periodic operations"""
        try:
            # Start memory auto-save thread
            if self.memory_manager and hasattr(self.memory_manager, 'start_auto_save'):
                self.memory_manager.start_auto_save()
                self._log_debug("Memory auto-save thread started")
            
            # Start periodic analysis thread
            self.analysis_thread = threading.Thread(
                target=self._analysis_worker,
                daemon=True,
                name="AnalysisWorker"
            )
            self.analysis_thread.start()
            self.state.background_threads.append(self.analysis_thread)
            self._log_debug("Background analysis thread started")
            
        except Exception as e:
            self._log_error(f"Background service startup failed: {e}")
    
    def _analysis_worker(self):
        """Background worker for periodic analysis operations"""
        self._log_debug("Analysis worker thread started")
        
        while not self.analysis_shutdown_event.is_set():
            try:
                # Check if analysis is needed
                if (self.state.message_count - self.state.last_analysis_count >= self.ANALYSIS_INTERVAL 
                    and not self.state.analysis_in_progress 
                    and self.state.phase == OrchestrationPhase.ACTIVE):
                    
                    self._trigger_periodic_analysis()
                
                # Sleep for short interval to avoid busy waiting
                self.analysis_shutdown_event.wait(timeout=5.0)
                
            except Exception as e:
                self._log_error(f"Analysis worker error: {e}")
                time.sleep(1.0)  # Prevent rapid error loops
        
        self._log_debug("Analysis worker thread stopped")

# Chunk 3/5 - orch.py - Core Processing Methods (Debug Logger Fix)

    def run(self) -> int:
        """
        Main orchestrator run loop - initializes and coordinates all modules
        Returns exit code for main.py
        """
        try:
            self._log_debug("Starting orchestrator run sequence")
            
            # Initialize all modules
            if not self.initialize_modules():
                self._log_error("Module initialization failed")
                return 1
            
            # Transfer control to UI controller
            if not self.ui_controller:
                self._log_error("UI controller not available")
                return 1
            
            self._log_debug("Transferring control to UI controller")
            exit_code = self.ui_controller.run()
            
            # Cleanup after UI exits
            self._shutdown_modules()
            
            self._log_debug(f"Orchestrator run completed with exit code: {exit_code}")
            return exit_code
            
        except Exception as e:
            self._log_error(f"Orchestrator run failed: {e}")
            return 1
    
    def _handle_ui_callback(self, action: str, data: Dict[str, Any]) -> Any:
        """
        Handle callbacks from UI controller
        This is the main coordination point for user actions
        """
        try:
            self._log_debug(f"Processing UI callback: {action}")
            
            if action == "user_input":
                return self._process_user_input(data)
            elif action == "get_messages":
                return self._get_message_history(data)
            elif action == "clear_memory":
                return self._clear_memory()
            elif action == "get_stats":
                return self._get_system_stats()
            elif action == "analyze_now":
                return self._trigger_immediate_analysis()
            elif action == "shutdown":
                return self._handle_shutdown()
            else:
                self._log_error(f"Unknown UI callback action: {action}")
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            self._log_error(f"UI callback handling failed for {action}: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_user_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input through the full pipeline
        FIXED: Use correct MessageType enum values from emm.py
        """
        try:
            user_input = data.get("input", "").strip()
            if not user_input:
                return {"success": False, "error": "Empty input"}

            self._log_debug("Processing user input")

            # 1. Validate input through semantic engine
            if self.semantic_engine:
                validation_result = self.semantic_engine.validate_input(user_input)
                if not validation_result.get("valid", True):
                    return {"success": False, "error": validation_result.get("error", "Input validation failed")}

            # 2. Store user message in memory - FIXED: Use MessageType.USER enum not string
            if self.memory_manager:
                self.memory_manager.add_message(user_input, MessageType.USER)
                self.state.message_count += 1
                self._log_debug("User message stored in memory")

            # 3. Send to LLM for response
            llm_response = self._make_llm_request(user_input)
            if not llm_response.get("success", False):
                return llm_response

            # 4. Store LLM response in memory - FIXED: Use MessageType.ASSISTANT enum not string
            response_text = llm_response.get("response", "")
            if self.memory_manager and response_text:
                self.memory_manager.add_message(response_text, MessageType.ASSISTANT)
                self.state.message_count += 1
                self._log_debug("LLM response stored in memory")

            # 5. Return success with response
            return {
                "success": True,
                "response": response_text,
                "message_count": self.state.message_count
            }

        except Exception as e:
            self._log_error(f"User input processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _make_llm_request(self, user_input: str) -> Dict[str, Any]:
        """
        Make LLM request - ONLY function that calls mcp.py
        FIXED: Handle both string and dictionary responses from MCP client
        """
        try:
            if not self.mcp_client:
                return {"success": False, "error": "MCP client not available"}

            self._log_debug("Making LLM request")

            # Get conversation context from memory - FIXED method name
            context = []
            if self.memory_manager:
                # Use get_conversation_for_mcp() instead of get_conversation_context()
                if hasattr(self.memory_manager, 'get_conversation_for_mcp'):
                    context = self.memory_manager.get_conversation_for_mcp()
                elif hasattr(self.memory_manager, 'get_messages'):
                    # Fallback to get_messages if available
                    messages = self.memory_manager.get_messages()
                    # Convert Message objects to MCP format
                    context = []
                    for msg in messages:
                        context.append({
                            "role": msg.message_type.value.lower(),  # Convert enum to string
                            "content": msg.content
                        })

            # Get current momentum state for context - FIXED: Handle missing methods gracefully
            momentum_data = {}
            if self.momentum_engine:
                try:
                    if hasattr(self.momentum_engine, 'get_current_state'):
                        momentum_data = self.momentum_engine.get_current_state()
                    elif hasattr(self.momentum_engine, 'get_state'):
                        momentum_data = self.momentum_engine.get_state()
                    else:
                        # Fallback: provide empty momentum context
                        momentum_data = {}
                        self._log_debug("Momentum engine has no state retrieval method")
                except Exception as e:
                    self._log_debug(f"Failed to get momentum state: {e}")
                    momentum_data = {}

            # FIXED: Use correct MCP client method call - try different possible parameter names
            raw_response = None
            try:
                # Try the most common parameter names for MCP clients
                if hasattr(self.mcp_client, 'send_message'):
                    # Attempt 1: Try with 'message' parameter
                    try:
                        raw_response = self.mcp_client.send_message(
                            message=user_input,
                            conversation_context=context,
                            momentum_context=momentum_data
                        )
                    except TypeError:
                        # Attempt 2: Try with 'content' parameter
                        try:
                            raw_response = self.mcp_client.send_message(
                                content=user_input,
                                conversation_context=context,
                                momentum_context=momentum_data
                            )
                        except TypeError:
                            # Attempt 3: Try with just the message as positional parameter
                            try:
                                raw_response = self.mcp_client.send_message(
                                    user_input,
                                    conversation_context=context,
                                    momentum_context=momentum_data
                                )
                            except TypeError:
                                # Attempt 4: Try without additional context parameters
                                raw_response = self.mcp_client.send_message(user_input)
                else:
                    return {"success": False, "error": "MCP client has no send_message method"}

            except Exception as method_error:
                self._log_error(f"MCP method call failed: {method_error}")
                return {"success": False, "error": f"MCP call error: {str(method_error)}"}

            # FIXED: Handle both string and dictionary responses from MCP client
            if raw_response is None:
                return {"success": False, "error": "No response from MCP client"}

            # Convert response to standard format
            if isinstance(raw_response, str):
                # MCP client returned a string - assume it's the LLM response content
                response = {
                    "success": True,
                    "response": raw_response,
                    "error": None
                }
                self._log_debug("LLM request completed successfully (string response)")
            elif isinstance(raw_response, dict):
                # MCP client returned a dictionary - use as-is or normalize
                response = raw_response
                if response.get("success", False):
                    self._log_debug("LLM request completed successfully (dict response)")
                else:
                    self._log_error(f"LLM request failed: {response.get('error', 'Unknown error')}")
            else:
                # Unexpected response type
                self._log_error(f"Unexpected MCP response type: {type(raw_response)}")
                return {"success": False, "error": f"Invalid response type: {type(raw_response)}"}

            return response

        except Exception as e:
            self._log_error(f"LLM request error: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_message_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get message history from memory manager
        FIXED: Handle Message objects correctly and convert to UI format
        """
        try:
            if not self.memory_manager:
                return {"success": False, "error": "Memory manager not available"}

            # Get parameters from data
            limit = data.get("limit", 50)

            messages = []

            # Try to get messages from memory manager
            if hasattr(self.memory_manager, 'get_messages'):
                # Get Message objects
                message_objects = self.memory_manager.get_messages(limit)

                # Convert Message objects to UI format
                for msg in message_objects:
                    # Message objects have attributes, not dictionary keys
                    messages.append({
                        "content": msg.content,
                        "type": msg.message_type.value.lower(),  # Convert enum to string
                        "timestamp": msg.timestamp,
                        "id": getattr(msg, 'id', None)
                    })

            elif hasattr(self.memory_manager, 'get_conversation_for_mcp'):
                # Get MCP format and convert to UI format
                mcp_messages = self.memory_manager.get_conversation_for_mcp()

                for msg in mcp_messages[-limit:]:
                    messages.append({
                        "content": msg.get("content", ""),
                        "type": msg.get("role", "user"),
                        "timestamp": None
                    })

            return {
                "success": True,
                "messages": messages,
                "total_count": len(messages)
            }

        except Exception as e:
            self._log_error(f"Message history retrieval failed: {e}")
            return {"success": False, "error": str(e)}

# Chunk 4/5 - orch.py - Analysis and Utility Methods (Debug Logger Fix)

    def _trigger_periodic_analysis(self):
        """
        Trigger comprehensive analysis every 15 messages
        FIXED: Use correct memory manager method name
        """
        if self.state.analysis_in_progress:
            self._log_debug("Analysis already in progress, skipping")
            return

        try:
            self.state.analysis_in_progress = True
            self.state.phase = OrchestrationPhase.ANALYZING

            self._log_debug("Starting periodic analysis")

            # Get conversation context for analysis - FIXED method name
            analysis_context = []
            if self.memory_manager:
                if hasattr(self.memory_manager, 'get_conversation_for_mcp'):
                    analysis_context = self.memory_manager.get_conversation_for_mcp()
                elif hasattr(self.memory_manager, 'get_messages'):
                    # Convert Message objects to analysis format
                    messages = self.memory_manager.get_messages()
                    for msg in messages:
                        analysis_context.append({
                            "id": getattr(msg, 'id', None),
                            "content": msg.content,
                            "role": msg.message_type.value.lower(),
                            "timestamp": msg.timestamp
                        })

            # Run semantic analysis
            semantic_results = {}
            if self.semantic_engine and analysis_context:
                semantic_results = self.semantic_engine.analyze_conversation(analysis_context)
                self._log_debug("Semantic analysis completed")

            # Run momentum analysis
            momentum_results = {}
            if self.momentum_engine and analysis_context:
                momentum_results = self.momentum_engine.analyze_momentum(analysis_context)
                self._log_debug("Momentum analysis completed")

            # Store analysis results
            self.state.analysis_results = {
                "semantic": semantic_results,
                "momentum": momentum_results,
                "timestamp": time.time(),
                "message_count": self.state.message_count
            }

            # Update analysis tracking
            self.state.last_analysis_count = self.state.message_count

            self._log_debug("Periodic analysis completed successfully")

        except Exception as e:
            self._log_error(f"Periodic analysis failed: {e}")
        finally:
            self.state.analysis_in_progress = False
            self.state.phase = OrchestrationPhase.ACTIVE
    
    def _trigger_immediate_analysis(self) -> Dict[str, Any]:
        """Trigger immediate analysis requested by user"""
        try:
            if self.state.analysis_in_progress:
                return {"success": False, "error": "Analysis already in progress"}
            
            self._log_debug("Starting immediate analysis")
            
            # Run analysis in current thread since user requested it
            self._trigger_periodic_analysis()
            
            return {
                "success": True,
                "results": self.state.analysis_results
            }
            
        except Exception as e:
            self._log_error(f"Immediate analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _clear_memory(self) -> Dict[str, Any]:
        """Clear conversation memory"""
        try:
            if not self.memory_manager:
                return {"success": False, "error": "Memory manager not available"}
            
            self._log_debug("Clearing conversation memory")
            
            # Clear memory and reset counters
            self.memory_manager.clear_conversation()
            self.state.message_count = 0
            self.state.last_analysis_count = 0
            self.state.analysis_results = {}
            
            # Reset momentum engine state
            if self.momentum_engine:
                self.momentum_engine.reset_state()
            
            self._log_debug("Memory cleared successfully")
            
            return {"success": True}
            
        except Exception as e:
            self._log_error(f"Memory clear failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """
        Get current system statistics
        FIXED: Handle missing get_stats() methods gracefully
        """
        try:
            stats = {
                "message_count": self.state.message_count,
                "last_analysis_count": self.state.last_analysis_count,
                "analysis_in_progress": self.state.analysis_in_progress,
                "phase": self.state.phase.value,
                "startup_complete": self.state.startup_complete
            }

            # Add memory stats with graceful fallback
            if self.memory_manager:
                try:
                    if hasattr(self.memory_manager, 'get_stats'):
                        memory_stats = self.memory_manager.get_stats()
                        stats["memory"] = memory_stats
                    else:
                        # Fallback: provide basic memory info
                        stats["memory"] = {
                            "available": True,
                            "message_count": getattr(self.memory_manager, 'message_count', 0)
                        }
                except Exception as e:
                    self._log_debug(f"Memory stats unavailable: {e}")
                    stats["memory"] = {"error": "stats unavailable"}

            # Add momentum stats with graceful fallback
            if self.momentum_engine:
                try:
                    if hasattr(self.momentum_engine, 'get_stats'):
                        momentum_stats = self.momentum_engine.get_stats()
                        stats["momentum"] = momentum_stats
                    elif hasattr(self.momentum_engine, 'get_current_state'):
                        # Fallback: get current state instead of stats
                        current_state = self.momentum_engine.get_current_state()
                        stats["momentum"] = {
                            "available": True,
                            "current_state": current_state
                        }
                    else:
                        stats["momentum"] = {"available": True, "stats": "method not available"}
                except Exception as e:
                    self._log_debug(f"Momentum stats unavailable: {e}")
                    stats["momentum"] = {"error": "stats unavailable"}

            # Add MCP stats with graceful fallback
            if self.mcp_client:
                try:
                    if hasattr(self.mcp_client, 'get_stats'):
                        mcp_stats = self.mcp_client.get_stats()
                        stats["mcp"] = mcp_stats
                    else:
                        # Fallback: provide basic MCP info
                        stats["mcp"] = {
                            "available": True,
                            "server_url": getattr(self.mcp_client, 'server_url', 'unknown'),
                            "model": getattr(self.mcp_client, 'model', 'unknown')
                        }
                except Exception as e:
                    self._log_debug(f"MCP stats unavailable: {e}")
                    stats["mcp"] = {"error": "stats unavailable"}

            return {"success": True, "stats": stats}

        except Exception as e:
            self._log_error(f"Stats retrieval failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_shutdown(self) -> Dict[str, Any]:
        """Handle shutdown request from UI"""
        try:
            self._log_debug("Shutdown requested")
            self.state.shutdown_requested = True
            return {"success": True}
            
        except Exception as e:
            self._log_error(f"Shutdown handling failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _shutdown_modules(self):
        """Clean shutdown of all modules and background threads"""
        try:
            self._log_debug("Starting module shutdown")
            self.state.phase = OrchestrationPhase.SHUTTING_DOWN
            
            # Stop background analysis thread
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_shutdown_event.set()
                self.analysis_thread.join(timeout=5.0)
                self._log_debug("Analysis thread stopped")
            
            # Shutdown memory manager (stop auto-save)
            if self.memory_manager and hasattr(self.memory_manager, 'shutdown'):
                self.memory_manager.shutdown()
                self._log_debug("Memory manager shutdown")
            
            # Shutdown momentum engine
            if self.momentum_engine and hasattr(self.momentum_engine, 'shutdown'):
                self.momentum_engine.shutdown()
                self._log_debug("Momentum engine shutdown")
            
            # Shutdown semantic engine
            if self.semantic_engine and hasattr(self.semantic_engine, 'shutdown'):
                self.semantic_engine.shutdown()
                self._log_debug("Semantic engine shutdown")
            
            # Shutdown MCP client
            if self.mcp_client and hasattr(self.mcp_client, 'shutdown'):
                self.mcp_client.shutdown()
                self._log_debug("MCP client shutdown")
            
            # Wait for remaining background threads
            for thread in self.state.background_threads:
                if thread.is_alive():
                    thread.join(timeout=2.0)
            
            self._log_debug("All modules shutdown completed")
            
        except Exception as e:
            self._log_error(f"Module shutdown failed: {e}")

# Chunk 5/5 - orch.py - Debug Logging Helper Methods (Debug Logger Fix)

    def _log_debug(self, message: str):
        """
        Standardized debug logging with null safety
        Uses method pattern: self.debug_logger.debug(message, "ORCHESTRATOR")
        """
        if self.debug_logger:
            self.debug_logger.debug(message, "ORCHESTRATOR")
    
    def _log_error(self, message: str):
        """
        Standardized error logging with null safety
        Uses method pattern: self.debug_logger.error(message, "ORCHESTRATOR")
        """
        if self.debug_logger:
            self.debug_logger.error(message, "ORCHESTRATOR")
    
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
    'Orchestrator',
    'OrchestrationPhase', 
    'OrchestrationState'
]
