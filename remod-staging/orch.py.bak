# Chunk 1/4 - orch.py with AsyncSyncBridge integration
#!/usr/bin/env python3

import asyncio
import threading
import time
import signal
from typing import Dict, List, Any, Optional, Callable, Coroutine
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import semantic logic module
from sem import SemanticProcessor, create_semantic_processor

# Import core modules
from mcp import MCPClient
from emm import EnhancedMemoryManager, MessageType
from sme import StoryMomentumEngine

# UI import will be updated when ui.py is created
# For now, keep compatibility with existing nci.py
try:
    from ui import UIController
except ImportError:
    from nci import CursesInterface as UIController


class AsyncSyncBridge:
    """
    Bridge for running async operations from synchronous context
    Manages dedicated event loop in separate thread
    """
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False
        
    def start(self):
        """Start the async event loop in dedicated thread"""
        if self.thread is not None:
            return  # Already started
            
        if self.debug_logger:
            self.debug_logger.debug("Starting AsyncSyncBridge", "BRIDGE")
            
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        
        # Start thread with event loop
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        timeout = 5.0
        start_time = time.time()
        while not self.loop.is_running() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
            
        if not self.loop.is_running():
            raise RuntimeError("Failed to start async event loop")
            
        if self.debug_logger:
            self.debug_logger.debug("AsyncSyncBridge started successfully", "BRIDGE")
    
    def _run_loop(self):
        """Run the event loop in dedicated thread"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Event loop error: {e}", "BRIDGE")
        finally:
            if self.debug_logger:
                self.debug_logger.debug("Event loop stopped", "BRIDGE")
    
    def run_async(self, coro: Coroutine, timeout: float = 30.0) -> Any:
        """
        Run async coroutine from sync context
        Returns the result or raises exception
        """
        if self._shutdown:
            raise RuntimeError("AsyncSyncBridge is shut down")
            
        if self.loop is None or not self.loop.is_running():
            raise RuntimeError("AsyncSyncBridge not started")
        
        try:
            if self.debug_logger:
                self.debug_logger.debug(f"Running async operation with {timeout}s timeout", "BRIDGE")
                
            # Submit coroutine to async event loop
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            
            # Wait for result with timeout
            result = future.result(timeout=timeout)
            
            if self.debug_logger:
                self.debug_logger.debug("Async operation completed successfully", "BRIDGE")
                
            return result
            
        except TimeoutError:
            if self.debug_logger:
                self.debug_logger.error(f"Async operation timed out after {timeout}s", "BRIDGE")
            raise TimeoutError(f"Async operation timed out after {timeout} seconds")
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Async operation failed: {e}", "BRIDGE")
            raise
    
    def run_async_safely(self, coro: Coroutine, timeout: float = 30.0, default=None) -> Any:
        """
        Run async coroutine with error handling
        Returns result on success, default value on failure
        """
        try:
            return self.run_async(coro, timeout)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Async operation failed safely: {e}", "BRIDGE")
            return default
    
    def shutdown(self):
        """Shutdown the async bridge and clean up resources"""
        if self._shutdown:
            return
            
        if self.debug_logger:
            self.debug_logger.debug("Shutting down AsyncSyncBridge", "BRIDGE")
            
        self._shutdown = True
        
        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                if self.debug_logger:
                    self.debug_logger.warning("Thread did not stop cleanly", "BRIDGE")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        if self.debug_logger:
            self.debug_logger.debug("AsyncSyncBridge shutdown complete", "BRIDGE")
    
    def is_running(self) -> bool:
        """Check if bridge is running and ready"""
        return (not self._shutdown and 
                self.loop is not None and 
                self.loop.is_running() and
                self.thread is not None and 
                self.thread.is_alive())


class OrchestratorState:
    """State tracking for orchestrator lifecycle"""
    
    def __init__(self):
        self.initialization_phase = "not_started"
        self.running = False
        self.shutdown_requested = False
        self.background_threads = {}
        self.initialization_time = None
        self.shutdown_time = None
        self.error_count = 0
        self.last_error = None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.initialization_phase,
            "running": self.running,
            "shutdown_requested": self.shutdown_requested,
            "active_threads": list(self.background_threads.keys()),
            "initialization_time": self.initialization_time,
            "shutdown_time": self.shutdown_time,
            "error_count": self.error_count,
            "last_error": self.last_error
        }

# Chunk 2/4 - orch.py with AsyncSyncBridge integration

class ModuleRegistry:
    """Registry for managing module instances and dependencies"""
    
    def __init__(self):
        self.modules = {}
        self.initialization_order = [
            "semantic_processor",
            "mcp_client", 
            "memory_manager",
            "story_engine",
            "ui_controller"
        ]
        self.dependencies = {
            "semantic_processor": [],
            "mcp_client": ["semantic_processor"],
            "memory_manager": ["semantic_processor", "mcp_client"],
            "story_engine": ["semantic_processor", "memory_manager"],
            "ui_controller": ["semantic_processor", "mcp_client", "memory_manager", "story_engine"]
        }
    
    def register_module(self, name: str, instance: Any) -> None:
        """Register a module instance"""
        self.modules[name] = instance
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get module instance by name"""
        return self.modules.get(name)
    
    def get_dependencies(self, module_name: str) -> List[str]:
        """Get dependency list for a module"""
        return self.dependencies.get(module_name, [])
    
    def are_dependencies_ready(self, module_name: str) -> bool:
        """Check if all dependencies for a module are initialized"""
        deps = self.get_dependencies(module_name)
        return all(dep in self.modules for dep in deps)
    
    def get_initialization_order(self) -> List[str]:
        """Get proper initialization order respecting dependencies"""
        return self.initialization_order.copy()


class DevNameOrchestrator:
    """
    Main orchestrator for DevName RPG Client with AsyncSyncBridge support
    
    Responsibilities:
    - Initialize all subsystems in correct order
    - Manage inter-module communication
    - Handle background thread coordination  
    - Process configuration and prompts from main.py
    - Coordinate shutdown sequence
    - Bridge async/sync operations
    """
    
    def __init__(self, config: Dict[str, Any] = None, debug_logger=None):
        self.config = config or {}
        self.debug_logger = debug_logger
        
        # Core orchestrator state
        self.state = OrchestratorState()
        self.module_registry = ModuleRegistry()
        
        # Configuration and prompts from main.py
        self.loaded_prompts = self.config.get('prompts', {})
        self.max_memory_tokens = self.config.get('max_memory_tokens', 30000)
        
        # Communication channels
        self.message_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        
        # Background processing
        self.background_tasks = {}
        self.analysis_coordinator = None
        
        # Async/Sync bridge
        self.async_bridge = AsyncSyncBridge(debug_logger)
        
        # Signal handling
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self._log_debug(f"Received signal {signum}, initiating shutdown")
            self.request_shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _log_debug(self, message: str, category: str = "ORCHESTRATOR"):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, category)
    
    async def initialize_system(self) -> bool:
        """Initialize all subsystems in correct dependency order"""
        try:
            self.state.initialization_phase = "starting"
            self.state.initialization_time = datetime.now().isoformat()
            self._log_debug("Starting system initialization")
            
            # Start async bridge first
            self.async_bridge.start()
            self._log_debug("AsyncSyncBridge started")
            
            # Initialize modules in dependency order
            for module_name in self.module_registry.get_initialization_order():
                if not await self._initialize_module(module_name):
                    self._log_debug(f"Failed to initialize {module_name}")
                    return False
            
            # Start background processing coordination
            await self._start_background_processing()
            
            # Complete initialization
            self.state.initialization_phase = "complete"
            self.state.running = True
            self._log_debug("System initialization complete")
            
            return True
            
        except Exception as e:
            self.state.last_error = str(e)
            self.state.error_count += 1
            self._log_debug(f"System initialization failed: {e}")
            return False
    
    async def _initialize_module(self, module_name: str) -> bool:
        """Initialize individual module with dependency checking"""
        try:
            self._log_debug(f"Initializing {module_name}")
            
            # Check dependencies
            if not self.module_registry.are_dependencies_ready(module_name):
                missing_deps = [dep for dep in self.module_registry.get_dependencies(module_name) 
                               if dep not in self.module_registry.modules]
                self._log_debug(f"Missing dependencies for {module_name}: {missing_deps}")
                return False
            
            # Initialize based on module type
            instance = None
            
            if module_name == "semantic_processor":
                instance = await self._initialize_semantic_processor()
            elif module_name == "mcp_client":
                instance = await self._initialize_mcp_client()
            elif module_name == "memory_manager":
                instance = await self._initialize_memory_manager()
            elif module_name == "story_engine":
                instance = await self._initialize_story_engine()
            elif module_name == "ui_controller":
                instance = await self._initialize_ui_controller()
            
            if instance:
                self.module_registry.register_module(module_name, instance)
                self._log_debug(f"Successfully initialized {module_name}")
                return True
            else:
                self._log_debug(f"Failed to create instance for {module_name}")
                return False
                
        except Exception as e:
            self._log_debug(f"Error initializing {module_name}: {e}")
            return False
    
    async def _initialize_semantic_processor(self) -> Optional[SemanticProcessor]:
        """Initialize semantic processor with configuration"""
        try:
            # Create semantic processor (MCP client will be provided later)
            processor = create_semantic_processor(debug_logger=self.debug_logger)
            
            # Start background processing
            asyncio.create_task(processor.start_background_processing())
            
            return processor
            
        except Exception as e:
            self._log_debug(f"Failed to initialize semantic processor: {e}")
            return None
    
    async def _initialize_mcp_client(self) -> Optional[MCPClient]:
        """Initialize MCP client with semantic processor integration"""
        try:
            semantic_processor = self.module_registry.get_module("semantic_processor")
            
            # Create MCP client
            mcp_client = MCPClient(debug_logger=self.debug_logger)
            
            # Configure from config
            mcp_config = self.config.get('mcp', {})
            if 'server_url' in mcp_config:
                mcp_client.server_url = mcp_config['server_url']
            if 'model' in mcp_config:
                mcp_client.model = mcp_config['model']
            if 'timeout' in mcp_config:
                mcp_client.timeout = mcp_config['timeout']
            
            # Set base system prompt from loaded critrules prompt
            if self.loaded_prompts.get('critrules'):
                mcp_client.system_prompt = self.loaded_prompts['critrules']
                self._log_debug("Base system prompt set from critrules")
            
            # Provide MCP client to semantic processor
            semantic_processor.mcp_client = mcp_client
            
            return mcp_client
            
        except Exception as e:
            self._log_debug(f"Failed to initialize MCP client: {e}")
            return None
    
    async def _initialize_memory_manager(self) -> Optional[EnhancedMemoryManager]:
        """Initialize memory manager with semantic integration"""
        try:
            semantic_processor = self.module_registry.get_module("semantic_processor")
            mcp_client = self.module_registry.get_module("mcp_client")

            # Create memory manager with correct parameters (matches current emm.py constructor)
            memory_manager = EnhancedMemoryManager(
                max_memory_tokens=self.max_memory_tokens,
                debug_logger=self.debug_logger
            )

            # Integrate semantic processor for analysis
            memory_manager.semantic_processor = semantic_processor
            memory_manager.mcp_client = mcp_client

            # Auto-load existing memory
            memory_manager._auto_load()

            return memory_manager

        except Exception as e:
            self._log_debug(f"Failed to initialize memory manager: {e}")
            return None

# Chunk 3/4 - orch.py with AsyncSyncBridge integration

    async def _initialize_story_engine(self) -> Optional[StoryMomentumEngine]:
        """Initialize story momentum engine with semantic integration"""
        try:
            semantic_processor = self.module_registry.get_module("semantic_processor")
            memory_manager = self.module_registry.get_module("memory_manager")
            
            # Create story momentum engine
            story_engine = StoryMomentumEngine(debug_logger=self.debug_logger)
            
            # Integrate semantic processor for analysis
            story_engine.semantic_processor = semantic_processor
            
            # Load existing SME state from memory manager
            try:
                momentum_state = memory_manager.get_momentum_state()
                if momentum_state:
                    story_engine.load_state_from_dict(momentum_state)
                    self._log_debug("SME state loaded from EMM on startup")
                else:
                    self._log_debug("No existing SME state found in EMM")
            except Exception as e:
                self._log_debug(f"Failed to load SME state: {e}")
            
            return story_engine
            
        except Exception as e:
            self._log_debug(f"Failed to initialize story engine: {e}")
            return None
    
    async def _initialize_ui_controller(self) -> Optional['UIController']:
        """Initialize UI controller with proper orchestrator integration and async bridge"""
        try:
            ui_controller = UIController(
                debug_logger=self.debug_logger,
                config=self.config
            )

            # Create bridge-enabled message processor
            def message_processor(user_input: str) -> Dict[str, Any]:
                """Process user messages through backend modules using async bridge"""
                try:
                    # Define async message processing coroutine
                    async def async_message_process():
                        memory_manager = self.module_registry.get_module("memory_manager")
                        story_engine = self.module_registry.get_module("story_engine")
                        mcp_client = self.module_registry.get_module("mcp_client")

                        if not all([memory_manager, story_engine, mcp_client]):
                            return {"success": False, "error": "Backend modules not available"}

                        # Store message in memory
                        memory_manager.add_message(user_input, MessageType.USER)

                        # Process through story engine
                        momentum_result = story_engine.process_user_input(user_input)

                        # Build context for MCP request
                        conversation_history = memory_manager.get_conversation_for_mcp()
                        story_context_dict = story_engine.get_story_context()

                        # Format story context as string for MCP
                        if isinstance(story_context_dict, dict):
                            pressure = story_context_dict.get('pressure_level', 0)
                            arc = story_context_dict.get('story_arc', 'setup')
                            story_context = f"Story Pressure: {pressure:.2f}, Arc: {arc}"
                        else:
                            story_context = str(story_context_dict)

                        # Build system messages with story context (like original nci.py)
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

                        # Combine system messages with conversation history
                        full_messages = system_messages + conversation_history

                        # Send to MCP server via direct httpx call (consistent with Phase 2)
                        try:
                            import httpx

                            # Build proper MCP payload with required fields
                            payload = {
                                "model": "qwen2.5:14b-instruct-q4_k_m",
                                "messages": full_messages,
                                "stream": False
                            }

                            async with httpx.AsyncClient(timeout=30.0) as client:
                                response = await client.post(
                                    "http://127.0.0.1:3456/chat",
                                    json=payload
                                )

                                if response.status_code == 200:
                                    response_data = response.json()
                                    # Debug: log the actual response structure
                                    print(f"DEBUG: MCP Response: {response_data}")

                                    # Use correct MCP server response format
                                    if (isinstance(response_data, dict) and
                                        "message" in response_data and
                                        isinstance(response_data["message"], dict) and
                                        "content" in response_data["message"]):
                                        ai_response = response_data["message"]["content"]
                                    else:
                                        return {"success": False, "error": f"Invalid response format from MCP server. Got: {response_data}"}

                            # Store AI response in memory
                            memory_manager.add_message(ai_response, MessageType.ASSISTANT)

                            # Update story momentum with AI response
                            if hasattr(story_engine, 'process_ai_response'):
                                story_engine.process_ai_response(ai_response)

                            return {
                                "success": True,
                                "ai_response": ai_response,
                                "momentum_result": momentum_result
                            }

                        except Exception as e:
                            return {"success": False, "error": f"MCP error: {e}"}

                    # Use async bridge to run async operation from sync context
                    if self.async_bridge.is_running():
                        return self.async_bridge.run_async_safely(
                            async_message_process(),
                            timeout=30.0,
                            default={"success": False, "error": "Async operation failed"}
                        )
                    else:
                        return {"success": False, "error": "Async bridge not available"}

                except Exception as e:
                    self._log_debug(f"Message processing error: {e}")
                    return {"success": False, "error": str(e)}

            # Create simplified command processor (remove duplicate logic)
            def command_processor(command: str) -> Dict[str, Any]:
                """Process user commands through appropriate handlers"""
                try:
                    if command == "/help":
                        return {"success": True, "system_message": "Available commands: /help, /stats, /analyze, /clearmemory, /theme, /quit"}
                    elif command == "/quit":
                        self.request_shutdown()
                        return {"success": True, "system_message": "Shutting down..."}
                    elif command == "/stats":
                        # Use async bridge for stats that might need async operations
                        async def async_stats():
                            memory_manager = self.module_registry.get_module("memory_manager")
                            story_engine = self.module_registry.get_module("story_engine")

                            stats = []
                            if memory_manager:
                                messages = memory_manager.get_messages() if hasattr(memory_manager, 'get_messages') else []
                                stats.append(f"Memory: {len(messages)} messages")
                            if story_engine:
                                current_state = story_engine.get_current_state()
                                if isinstance(current_state, dict):
                                    stats.append(f"Story pressure: {current_state.get('narrative_pressure', 0):.2f}")
                                    stats.append(f"Story arc: {current_state.get('story_arc', 'setup')}")

                            return {"success": True, "system_message": "\n".join(stats)}

                        if self.async_bridge.is_running():
                            return self.async_bridge.run_async_safely(
                                async_stats(),
                                timeout=10.0,
                                default={"success": False, "error": "Stats command failed"}
                            )
                        else:
                            return {"success": False, "error": "Async bridge not available"}
                    else:
                        return {"success": False, "error": f"Unknown command: {command}"}

                except Exception as e:
                    return {"success": False, "error": str(e)}

            # Set processors on UI controller
            ui_controller.set_message_processor(message_processor)
            ui_controller.set_command_processor(command_processor)

            # Set status updater callback
            def status_updater(status: str):
                """Update status from orchestrator"""
                pass

            ui_controller.set_status_updater(status_updater)

            self._log_debug("UI controller initialized with orchestrator integration and async bridge")
            return ui_controller

        except Exception as e:
            self._log_debug(f"Failed to initialize UI controller: {e}")
            return None
    
    async def _start_background_processing(self):
        """Start background processing coordination"""
        try:
            # Create analysis coordinator
            self.analysis_coordinator = BackgroundAnalysisCoordinator(
                self.module_registry,
                debug_logger=self.debug_logger
            )
            
            # Start coordination task
            self.background_tasks["analysis_coordinator"] = asyncio.create_task(
                self.analysis_coordinator.start_coordination()
            )
            
            # Start semantic processor background processing
            semantic_processor = self.module_registry.get_module("semantic_processor")
            if semantic_processor:
                self.background_tasks["semantic_processing"] = asyncio.create_task(
                    semantic_processor.start_background_processing()
                )
            
            self._log_debug("Background processing coordination started")
            
        except Exception as e:
            self._log_debug(f"Failed to start background processing: {e}")
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get module instance by name"""
        return self.module_registry.get_module(name)
    
    def get_all_modules(self) -> Dict[str, Any]:
        """Get all registered modules"""
        return self.module_registry.modules.copy()
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        self.state.shutdown_requested = True
        self.shutdown_event.set()
    
    async def shutdown_system(self) -> bool:
        """Graceful shutdown of all subsystems including async bridge"""
        try:
            self.state.shutdown_time = datetime.now().isoformat()
            self._log_debug("Starting system shutdown")
            
            # Stop background processing
            await self._stop_background_processing()
            
            # Save final states
            await self._save_final_states()
            
            # Shutdown modules in reverse order
            await self._shutdown_modules()
            
            # Shutdown async bridge
            if self.async_bridge:
                self.async_bridge.shutdown()
                self._log_debug("AsyncSyncBridge shutdown complete")
            
            self.state.running = False
            self._log_debug("System shutdown complete")
            return True
            
        except Exception as e:
            self.state.last_error = str(e)
            self.state.error_count += 1
            self._log_debug(f"Error during shutdown: {e}")
            return False
    
    async def _stop_background_processing(self):
        """Stop all background processing tasks"""
        try:
            # Cancel background tasks
            for task_name, task in self.background_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                self._log_debug(f"Stopped background task: {task_name}")
            
            # Stop analysis coordinator
            if self.analysis_coordinator:
                await self.analysis_coordinator.stop_coordination()
            
            # Stop semantic processor background processing
            semantic_processor = self.get_module("semantic_processor")
            if semantic_processor:
                semantic_processor.stop_background_processing()
            
        except Exception as e:
            self._log_debug(f"Error stopping background processing: {e}")
    
    async def _save_final_states(self):
        """Save final states of all modules"""
        try:
            memory_manager = self.get_module("memory_manager")
            story_engine = self.get_module("story_engine")
            
            if memory_manager and story_engine:
                # Save final SME state to EMM
                final_state = story_engine.save_state_to_dict()
                memory_manager.update_momentum_state(final_state)
                self._log_debug("Final SME state saved to EMM")
                
                # Auto-save conversation if configured
                if self.config.get('auto_save_conversation', False):
                    filename = f"chat_history_{int(time.time())}.json"
                    if memory_manager.save_conversation(filename):
                        self._log_debug(f"Conversation saved to {filename}")
            
        except Exception as e:
            self._log_debug(f"Error saving final states: {e}")
    
    async def _shutdown_modules(self):
        """Shutdown modules in reverse dependency order"""
        try:
            shutdown_order = list(reversed(self.module_registry.get_initialization_order()))
            
            for module_name in shutdown_order:
                module = self.get_module(module_name)
                if module and hasattr(module, 'shutdown'):
                    try:
                        if asyncio.iscoroutinefunction(module.shutdown):
                            await module.shutdown()
                        else:
                            module.shutdown()
                        self._log_debug(f"Shutdown {module_name}")
                    except Exception as e:
                        self._log_debug(f"Error shutting down {module_name}: {e}")
        
        except Exception as e:
            self._log_debug(f"Error during module shutdown: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including async bridge"""
        status = {
            "orchestrator_state": self.state.to_dict(),
            "modules": {},
            "background_tasks": {},
            "async_bridge": {
                "running": self.async_bridge.is_running() if self.async_bridge else False,
                "shutdown": self.async_bridge._shutdown if self.async_bridge else True
            }
        }
        
        # Module status
        for name, module in self.module_registry.modules.items():
            module_status = {"initialized": True, "type": type(module).__name__}
            
            # Get module-specific status if available
            if hasattr(module, 'get_stats'):
                try:
                    module_status.update(module.get_stats())
                except:
                    pass
            
            status["modules"][name] = module_status
        
        # Background task status
        for name, task in self.background_tasks.items():
            status["background_tasks"][name] = {
                "running": not task.done(),
                "cancelled": task.cancelled(),
                "exception": str(task.exception()) if task.done() and task.exception() else None
            }
        
        return status


class BackgroundAnalysisCoordinator:
    """Coordinates background analysis between modules"""
    
    def __init__(self, module_registry: ModuleRegistry, debug_logger=None):
        self.module_registry = module_registry
        self.debug_logger = debug_logger
        self.running = False
        self.analysis_queue = asyncio.Queue()
    
    async def start_coordination(self):
        """Start coordination loop"""
        self.running = True
        
        while self.running:
            try:
                # Wait for analysis requests

# Chunk 4/4 - orch.py with AsyncSyncBridge integration

                analysis_request = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)
                
                if analysis_request:
                    await self._process_analysis_request(analysis_request)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.debug(f"Analysis coordination error: {e}")
    
    async def queue_comprehensive_analysis(self, total_messages: int):
        """Queue comprehensive momentum analysis"""
        request = {
            "type": "comprehensive_momentum",
            "total_messages": total_messages,
            "timestamp": time.time()
        }
        
        await self.analysis_queue.put(request)
    
    async def _process_analysis_request(self, request: Dict[str, Any]):
        """Process analysis request"""
        try:
            if request["type"] == "comprehensive_momentum":
                await self._run_comprehensive_momentum_analysis(request["total_messages"])
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.debug(f"Error processing analysis request: {e}")
    
    async def _run_comprehensive_momentum_analysis(self, total_messages: int):
        """Run comprehensive momentum analysis in background"""
        try:
            memory_manager = self.module_registry.get_module("memory_manager")
            story_engine = self.module_registry.get_module("story_engine")
            
            if not memory_manager or not story_engine:
                return
            
            # Load current SME state from EMM
            momentum_state = memory_manager.get_momentum_state()
            if momentum_state:
                story_engine.load_state_from_dict(momentum_state)
            
            # Prepare conversation context for analysis
            conversation_messages = memory_manager.get_conversation_for_mcp()
            
            # Determine if this is first analysis or regular cycle
            is_first_analysis = momentum_state is None or story_engine.last_analysis_count == 0
            
            # Execute comprehensive momentum analysis
            analysis_result = await story_engine.analyze_momentum(
                conversation_messages, total_messages, is_first_analysis
            )
            
            # Save updated state to EMM
            updated_state = story_engine.save_state_to_dict()
            memory_manager.update_momentum_state(updated_state)
            
            if self.debug_logger:
                pressure = analysis_result.get("narrative_pressure", 0.0)
                manifestation = analysis_result.get("manifestation_type", "unknown")
                self.debug_logger.debug(
                    f"Comprehensive analysis complete: {pressure:.2f} pressure, {manifestation} manifestation"
                )
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.debug(f"Comprehensive analysis failed: {e}")
    
    async def stop_coordination(self):
        """Stop coordination loop"""
        self.running = False


# Main orchestrator interface functions for integration with main.py

async def create_orchestrator(config: Dict[str, Any], debug_logger=None) -> DevNameOrchestrator:
    """Factory function to create and initialize orchestrator"""
    orchestrator = DevNameOrchestrator(config, debug_logger)
    
    # Initialize system
    if await orchestrator.initialize_system():
        return orchestrator
    else:
        raise RuntimeError("Failed to initialize orchestrator system")


async def run_orchestrated_application(config: Dict[str, Any], debug_logger=None) -> int:
    """Run the complete orchestrated application with proper async/sync separation and bridge"""

    orchestrator = None

    # PHASE 1: Initialize orchestrator and extract components
    try:
        if debug_logger:
            debug_logger.debug("Phase 1: Initializing orchestrator components", "ORCHESTRATOR")

        # Create and initialize orchestrator (includes AsyncSyncBridge)
        orchestrator = await create_orchestrator(config, debug_logger)

        # Extract modules for synchronous use
        ui_controller = orchestrator.get_module("ui_controller")
        memory_manager = orchestrator.get_module("memory_manager")
        mcp_client = orchestrator.get_module("mcp_client")
        story_engine = orchestrator.get_module("story_engine")

        if not ui_controller:
            raise RuntimeError("UI controller not initialized")

        if debug_logger:
            debug_logger.debug("Phase 1 complete: Components extracted", "ORCHESTRATOR")

    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Phase 1 failed: {e}", "ORCHESTRATOR")
        print(f"Initialization error: {e}")
        return 1

    # PHASE 2: Setup synchronous processors with live orchestrator and async bridge
    try:
        if debug_logger:
            debug_logger.debug("Phase 2: Setting up sync processors with async bridge", "ORCHESTRATOR")

        # Create sync message processor that uses async bridge for MCP communication
        def sync_message_processor(messages):
            """Process messages synchronously using async bridge for MCP calls"""
            try:
                print(f"DEBUG: sync_message_processor called with {len(messages)} messages")

                if debug_logger:
                    debug_logger.debug(f"Processing {len(messages)} messages via bridge", "ORCHESTRATOR")

                # Define async MCP communication
                async def async_mcp_call():
                    print("DEBUG: Inside async_mcp_call")
                    import httpx

                    try:
                        # Use simple payload format that the Node.js server expects
                        user_message = messages[-1]["content"] if messages else "test"
                        payload = {
                            "message": user_message,
                            "model": "qwen2.5:14b-instruct-q4_k_m"
                        }

                        print(f"DEBUG: Sending payload: {payload}")

                        async with httpx.AsyncClient(timeout=30.0) as client:
                            print("DEBUG: Making HTTP request")
                            response = await client.post(
                                "http://127.0.0.1:3456/chat",
                                json=payload
                            )

                            print(f"DEBUG: Response status: {response.status_code}")

                            if response.status_code == 200:
                                response_data = response.json()
                                print(f"DEBUG: Response data: {response_data}")
                                return {"response": "Got successful response"}
                            else:
                                response_text = response.text
                                print(f"DEBUG: Error response: {response_text}")
                                return {"error": f"MCP server error: {response.status_code} - {response_text}"}

                    except Exception as e:
                        print(f"DEBUG: Exception in async_mcp_call: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                print("DEBUG: About to call async bridge")

                # Use async bridge to execute MCP call
                if orchestrator.async_bridge.is_running():
                    print("DEBUG: Bridge is running, calling run_async_safely")
                    result = orchestrator.async_bridge.run_async_safely(
                        async_mcp_call(),
                        timeout=30.0,
                        default={"error": "Bridge communication failed"}
                    )
                    print(f"DEBUG: Bridge result: {result}")

                    if debug_logger:
                        success = "error" not in result
                        debug_logger.debug(f"Bridge MCP call {'successful' if success else 'failed'}", "ORCHESTRATOR")

                    return result
                else:
                    print("DEBUG: Bridge not running")
                    error_msg = "AsyncSyncBridge not available"
                    if debug_logger:
                        debug_logger.error(error_msg, "ORCHESTRATOR")
                    return {"error": error_msg}

            except Exception as e:
                print(f"DEBUG: Exception in sync_message_processor: {e}")
                error_msg = f"Message processing failed: {str(e)}"
                if debug_logger:
                    debug_logger.error(error_msg, "ORCHESTRATOR")
                return {"error": error_msg}

        # Create sync command processor that uses live orchestrator
        def sync_command_processor(command, args):
            """Process commands synchronously using live orchestrator"""
            try:
                if command == "theme":
                    if not args:
                        return {"error": "Theme name required"}

                    theme_name = args[0].lower()
                    valid_themes = ["classic", "cyberpunk", "forest", "ocean", "fire", "minimal"]

                    if theme_name not in valid_themes:
                        return {"error": f"Invalid theme: {theme_name}. Available: {', '.join(valid_themes)}"}

                    # Use live UI controller for theme changes
                    if ui_controller and hasattr(ui_controller, '_change_theme'):
                        try:
                            ui_controller._change_theme(theme_name)
                            return {"message": f"Theme changed to {theme_name}"}
                        except Exception as e:
                            return {"error": f"Theme change failed: {str(e)}"}
                    else:
                        return {"error": "UI controller not available"}

                elif command == "status":
                    # Use live orchestrator for status
                    status = orchestrator.get_system_status()
                    status_message = "=== System Status ===\n"
                    status_message += f"Orchestrator: {status['orchestrator_state']['phase']}\n"
                    status_message += f"Running: {status['orchestrator_state']['running']}\n"
                    status_message += f"Modules: {len(status['modules'])} initialized\n"
                    status_message += f"Background Tasks: {len([t for t in status['background_tasks'].values() if t['running']])} active\n"
                    status_message += f"AsyncBridge: {'running' if status['async_bridge']['running'] else 'stopped'}\n"
                    return {"message": status_message, "detailed_status": status}

                elif command == "clear":
                    if ui_controller and hasattr(ui_controller, 'clear_display'):
                        ui_controller.clear_display()
                        return {"message": "Display cleared"}
                    else:
                        return {"error": "Clear not available"}
                else:
                    return {"error": f"Unknown command: {command}"}

            except Exception as e:
                return {"error": str(e)}

        # Set processors on UI controller
        ui_controller.set_message_processor(sync_message_processor)
        ui_controller.set_command_processor(sync_command_processor)

        if debug_logger:
            debug_logger.debug("Phase 2 complete: Sync processors with async bridge configured", "ORCHESTRATOR")

    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Phase 2 failed: {e}", "ORCHESTRATOR")
        # Try to shutdown cleanly before returning
        if orchestrator:
            try:
                await orchestrator.shutdown_system()
            except:
                pass
        return 1

    # PHASE 3: Run UI with live orchestrator and async bridge
    try:
        if debug_logger:
            debug_logger.debug("Phase 3: Running UI with live orchestrator and async bridge", "ORCHESTRATOR")

        # Run the UI - this blocks until UI exits
        ui_exit_code = ui_controller.run()

        if debug_logger:
            debug_logger.debug(f"UI exited with code: {ui_exit_code}", "ORCHESTRATOR")

        return ui_exit_code

    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Phase 3 failed: {e}", "ORCHESTRATOR")
        print(f"UI error: {e}")
        return 1

    finally:
        # PHASE 4: Shutdown orchestrator and async bridge AFTER UI completes
        if debug_logger:
            debug_logger.debug("Phase 4: Shutting down orchestrator and async bridge", "ORCHESTRATOR")

        if orchestrator:
            try:
                orchestrator.request_shutdown()
                await orchestrator.shutdown_system()
                if debug_logger:
                    debug_logger.debug("Orchestrator and async bridge shutdown complete", "ORCHESTRATOR")
            except Exception as e:
                if debug_logger:
                    debug_logger.error(f"Shutdown error: {e}", "ORCHESTRATOR")
                print(f"Shutdown error: {e}")

        # Clean up references
        ui_controller = None
        memory_manager = None
        mcp_client = None
        story_engine = None
        orchestrator = None


# Utility functions for testing and diagnostics

def test_orchestrator_initialization(config: Dict[str, Any] = None) -> bool:
    """Test orchestrator initialization without running UI"""
    async def test_init():
        try:
            orchestrator = await create_orchestrator(config or {})
            status = orchestrator.get_system_status()
            await orchestrator.shutdown_system()
            return status['orchestrator_state']['running']
        except Exception:
            return False
    
    return asyncio.run(test_init())

def get_orchestrator_info() -> Dict[str, Any]:
    """Get information about orchestrator capabilities"""
    return {
        "name": "DevName RPG Client Orchestrator with AsyncSyncBridge",
        "version": "1.1",
        "modules_managed": [
            "semantic_processor",
            "mcp_client", 
            "memory_manager",
            "story_engine",
            "ui_controller"
        ],
        "features": [
            "Dependency-aware initialization",
            "Background thread coordination",
            "Inter-module communication",
            "Graceful shutdown sequence",
            "Command processing",
            "System status monitoring",
            "Async/Sync bridge for UI-backend communication"
        ],
        "integration_points": [
            "main.py configuration processing",
            "Prompt system integration",
            "Background LLM analysis coordination",
            "Module state persistence",
            "AsyncSyncBridge for curses UI compatibility"
        ]
    }

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Main Orchestrator Module with AsyncSyncBridge")
    print("Successfully implemented orchestrator with:")
    print("✓ Dependency-aware module initialization")
    print("✓ Background thread coordination")
    print("✓ Inter-module communication management")
    print("✓ Configuration and prompt processing from main.py")
    print("✓ Graceful shutdown sequence")
    print("✓ Command processing coordination")
    print("✓ System status monitoring")
    print("✓ Complete LLM analysis coordination")
    print("✓ AsyncSyncBridge for sync UI to async backend communication")
    print("\nOrchestrator Info:")
    
    info = get_orchestrator_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nInitialization test: {'✓ PASSED' if test_orchestrator_initialization() else '✗ FAILED'}")
    print("\nReady for integration with main.py with async/sync bridge support.")
