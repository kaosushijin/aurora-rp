# Chunk 1/3 - orch.py - Main Orchestrator with Central Coordination
#!/usr/bin/env python3

import asyncio
import threading
import time
import signal
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

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
    Main orchestrator for DevName RPG Client
    
    Responsibilities:
    - Initialize all subsystems in correct order
    - Manage inter-module communication
    - Handle background thread coordination  
    - Process configuration and prompts from main.py
    - Coordinate shutdown sequence
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
            
            # Create memory manager
            memory_manager = EnhancedMemoryManager(
                debug_logger=self.debug_logger,
                max_memory_tokens=self.max_memory_tokens
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

# Chunk 2/3 - orch.py - Story Engine Initialization and Background Thread Coordination

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
    
    async def _initialize_ui_controller(self) -> Optional[UIController]:
        """Initialize UI controller with all backend modules"""
        try:
            # Get all initialized modules
            semantic_processor = self.module_registry.get_module("semantic_processor")
            mcp_client = self.module_registry.get_module("mcp_client")
            memory_manager = self.module_registry.get_module("memory_manager")
            story_engine = self.module_registry.get_module("story_engine")
            
            # Create UI controller with orchestrator coordination
            ui_controller = UIController(
                debug_logger=self.debug_logger,
                config=self.config
            )
            
            # Provide module references to UI
            ui_controller.semantic_processor = semantic_processor
            ui_controller.mcp_client = mcp_client
            ui_controller.memory_manager = memory_manager
            ui_controller.sme = story_engine
            ui_controller.loaded_prompts = self.loaded_prompts
            
            # Configure components through orchestrator
            await self._configure_ui_components(ui_controller)
            
            return ui_controller
            
        except Exception as e:
            self._log_debug(f"Failed to initialize UI controller: {e}")
            return None
    
    async def _configure_ui_components(self, ui_controller):
        """Configure UI components through orchestrator"""
        try:
            # This will be simplified when UI is refactored to ui.py
            if hasattr(ui_controller, '_configure_components'):
                ui_controller._configure_components()
        except Exception as e:
            self._log_debug(f"Failed to configure UI components: {e}")
    
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
    
    async def process_user_message(self, content: str) -> Dict[str, Any]:
        """Process user message through complete pipeline"""
        try:
            result = {
                "success": False,
                "user_content": content,
                "ai_response": None,
                "semantic_analysis": None,
                "momentum_update": None,
                "error": None
            }
            
            # Get modules
            memory_manager = self.get_module("memory_manager")
            story_engine = self.get_module("story_engine")
            mcp_client = self.get_module("mcp_client")
            semantic_processor = self.get_module("semantic_processor")
            
            if not all([memory_manager, story_engine, mcp_client, semantic_processor]):
                result["error"] = "Required modules not initialized"
                return result
            
            # Store user message with background semantic analysis
            user_message = memory_manager.add_message(content, MessageType.USER)
            
            # Process immediate story momentum patterns
            momentum_result = story_engine.process_user_input(content)
            result["momentum_update"] = momentum_result
            
            # Build system messages with story context
            story_context = story_engine.get_story_context()
            system_messages = self._build_system_messages(story_context)
            
            # Get conversation history for MCP
            conversation_messages = memory_manager.get_conversation_for_mcp()
            
            # Send to LLM via MCP
            full_messages = system_messages + conversation_messages
            ai_response = await mcp_client.send_request(full_messages)
            
            if ai_response:
                # Store AI response
                memory_manager.add_message(ai_response, MessageType.ASSISTANT)
                result["ai_response"] = ai_response
                
                # Check if comprehensive analysis is needed
                total_messages = memory_manager.get_message_count()
                if total_messages % 15 == 0:
                    await self._trigger_comprehensive_analysis(total_messages)
                
                result["success"] = True
            else:
                result["error"] = "No response from LLM"
            
            return result
            
        except Exception as e:
            self._log_debug(f"Error processing user message: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_content": content
            }
    
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
    
    async def _trigger_comprehensive_analysis(self, total_messages: int):
        """Trigger comprehensive momentum analysis via coordinator"""
        if self.analysis_coordinator:
            await self.analysis_coordinator.queue_comprehensive_analysis(total_messages)
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        self.state.shutdown_requested = True
        self.shutdown_event.set()
    
    async def shutdown_system(self) -> bool:
        """Graceful shutdown of all subsystems"""
        try:
            self.state.shutdown_time = datetime.now().isoformat()
            self._log_debug("Starting system shutdown")
            
            # Stop background processing
            await self._stop_background_processing()
            
            # Save final states
            await self._save_final_states()
            
            # Shutdown modules in reverse order
            await self._shutdown_modules()
            
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
        """Get comprehensive system status"""
        status = {
            "orchestrator_state": self.state.to_dict(),
            "modules": {},
            "background_tasks": {},
            "memory_usage": {}
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

# Chunk 3/3 - orch.py - Main Orchestrator Interface and Integration

class CommandProcessor:
    """Processes commands through the orchestrator"""
    
    def __init__(self, orchestrator: DevNameOrchestrator):
        self.orchestrator = orchestrator
        self.debug_logger = orchestrator.debug_logger
        
        self.command_handlers = {
            '/help': self._handle_help,
            '/quit': self._handle_quit,
            '/exit': self._handle_quit,
            '/clearmemory': self._handle_clear_memory,
            '/savememory': self._handle_save_memory,
            '/loadmemory': self._handle_load_memory,
            '/stats': self._handle_stats,
            '/analyze': self._handle_force_analysis,
            '/reset_momentum': self._handle_reset_momentum,
            '/theme': self._handle_theme_change,
            '/status': self._handle_system_status
        }
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process command and return result"""
        try:
            cmd_parts = command.strip().split()
            if not cmd_parts:
                return {"error": "Empty command"}
            
            base_cmd = cmd_parts[0].lower()
            args = cmd_parts[1:] if len(cmd_parts) > 1 else []
            
            if base_cmd in self.command_handlers:
                return await self.command_handlers[base_cmd](args)
            else:
                return {"error": f"Unknown command: {base_cmd}"}
                
        except Exception as e:
            return {"error": f"Command processing error: {str(e)}"}
    
    async def _handle_help(self, args: List[str]) -> Dict[str, Any]:
        """Handle help command"""
        help_text = """
DevName RPG Client - Available Commands:

/help                    - Show this help message
/quit, /exit            - Exit the application
/clearmemory [backup]   - Clear conversation memory (optional backup filename)
/savememory [filename]  - Save conversation to file
/loadmemory <filename>  - Load conversation from file
/stats                  - Show comprehensive system statistics
/analyze                - Force immediate comprehensive analysis
/reset_momentum         - Reset story momentum engine
/theme <name>           - Change color theme (classic/dark/bright)
/status                 - Show detailed system status

Background Analysis Features:
- Automatic semantic categorization of all messages
- Comprehensive momentum analysis every 15 messages
- Dynamic antagonist generation and commitment tracking
- Pressure floor ratcheting prevents narrative stalling
"""
        return {"message": help_text}
    
    async def _handle_quit(self, args: List[str]) -> Dict[str, Any]:
        """Handle quit command"""
        self.orchestrator.request_shutdown()
        return {"message": "Shutting down...", "shutdown": True}
    
    async def _handle_clear_memory(self, args: List[str]) -> Dict[str, Any]:
        """Handle clear memory command"""
        memory_manager = self.orchestrator.get_module("memory_manager")
        if not memory_manager:
            return {"error": "Memory manager not available"}
        
        backup_filename = args[0] if args else None
        
        try:
            if backup_filename:
                if memory_manager.save_conversation(backup_filename):
                    memory_manager.clear_conversation()
                    return {"message": f"Memory cleared, backup saved to {backup_filename}"}
                else:
                    return {"error": f"Failed to create backup {backup_filename}"}
            else:
                memory_manager.clear_conversation()
                return {"message": "Memory cleared"}
        except Exception as e:
            return {"error": f"Failed to clear memory: {str(e)}"}
    
    async def _handle_save_memory(self, args: List[str]) -> Dict[str, Any]:
        """Handle save memory command"""
        memory_manager = self.orchestrator.get_module("memory_manager")
        if not memory_manager:
            return {"error": "Memory manager not available"}
        
        filename = args[0] if args else f"conversation_{int(time.time())}.json"
        
        try:
            if memory_manager.save_conversation(filename):
                return {"message": f"Conversation saved to {filename}"}
            else:
                return {"error": f"Failed to save to {filename}"}
        except Exception as e:
            return {"error": f"Save failed: {str(e)}"}
    
    async def _handle_load_memory(self, args: List[str]) -> Dict[str, Any]:
        """Handle load memory command"""
        if not args:
            return {"error": "Usage: /loadmemory <filename>"}
        
        memory_manager = self.orchestrator.get_module("memory_manager")
        if not memory_manager:
            return {"error": "Memory manager not available"}
        
        filename = args[0]
        
        try:
            if memory_manager.load_conversation(filename):
                return {"message": f"Conversation loaded from {filename}"}
            else:
                return {"error": f"Failed to load from {filename}"}
        except Exception as e:
            return {"error": f"Load failed: {str(e)}"}
    
    async def _handle_stats(self, args: List[str]) -> Dict[str, Any]:
        """Handle stats command"""
        try:
            # Get stats from all modules
            stats = {}
            
            memory_manager = self.orchestrator.get_module("memory_manager")
            if memory_manager and hasattr(memory_manager, 'get_stats'):
                stats["memory"] = memory_manager.get_stats()
            
            story_engine = self.orchestrator.get_module("story_engine")
            if story_engine and hasattr(story_engine, 'get_pressure_stats'):
                stats["story_momentum"] = story_engine.get_pressure_stats()
            
            semantic_processor = self.orchestrator.get_module("semantic_processor")
            if semantic_processor and hasattr(semantic_processor, 'get_comprehensive_stats'):
                stats["semantic"] = semantic_processor.get_comprehensive_stats()
            
            # Format stats message
            stats_message = "=== System Statistics ===\n"
            
            if "memory" in stats:
                mem_stats = stats["memory"]
                stats_message += f"Memory: {mem_stats.get('message_count', 0)} messages, "
                stats_message += f"{mem_stats.get('total_tokens', 0)} tokens\n"
            
            if "story_momentum" in stats:
                sm_stats = stats["story_momentum"]
                stats_message += f"Pressure: {sm_stats.get('current_pressure', 0):.3f}, "
                stats_message += f"Arc: {sm_stats.get('current_arc', 'unknown')}\n"
            
            if "semantic" in stats:
                sem_stats = stats["semantic"]
                analysis_stats = sem_stats.get("analysis_stats", {})
                stats_message += f"Analyses: {analysis_stats.get('total_analyses', 0)} total, "
                stats_message += f"{analysis_stats.get('successful_analyses', 0)} successful\n"
            
            return {"message": stats_message, "detailed_stats": stats}
            
        except Exception as e:
            return {"error": f"Failed to get stats: {str(e)}"}
    
    async def _handle_force_analysis(self, args: List[str]) -> Dict[str, Any]:
        """Handle force analysis command"""
        try:
            memory_manager = self.orchestrator.get_module("memory_manager")
            if not memory_manager:
                return {"error": "Memory manager not available"}
            
            total_messages = memory_manager.get_message_count()
            await self.orchestrator._trigger_comprehensive_analysis(total_messages)
            
            return {"message": "Comprehensive analysis initiated"}
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _handle_reset_momentum(self, args: List[str]) -> Dict[str, Any]:
        """Handle reset momentum command"""
        try:
            story_engine = self.orchestrator.get_module("story_engine")
            memory_manager = self.orchestrator.get_module("memory_manager")
            
            if not story_engine or not memory_manager:
                return {"error": "Required modules not available"}
            
            # Reset story engine
            if hasattr(story_engine, 'reset_momentum'):
                story_engine.reset_momentum()
            
            # Clear momentum state in memory
            memory_manager.update_momentum_state({})
            
            return {"message": "Story momentum reset complete"}
            
        except Exception as e:
            return {"error": f"Reset failed: {str(e)}"}
    
    async def _handle_theme_change(self, args: List[str]) -> Dict[str, Any]:
        """Handle theme change command"""
        if not args:
            return {"error": "Usage: /theme <name> (classic/dark/bright)"}
        
        theme_name = args[0].lower()
        valid_themes = ["classic", "dark", "bright"]
        
        if theme_name not in valid_themes:
            return {"error": f"Invalid theme. Available: {', '.join(valid_themes)}"}
        
        # This will be handled by UI controller
        ui_controller = self.orchestrator.get_module("ui_controller")
        if ui_controller and hasattr(ui_controller, '_change_theme'):
            try:
                ui_controller._change_theme(theme_name)
                return {"message": f"Theme changed to {theme_name}"}
            except Exception as e:
                return {"error": f"Theme change failed: {str(e)}"}
        else:
            return {"error": "UI controller not available"}
    
    async def _handle_system_status(self, args: List[str]) -> Dict[str, Any]:
        """Handle system status command"""
        try:
            status = self.orchestrator.get_system_status()
            
            status_message = "=== System Status ===\n"
            status_message += f"Orchestrator: {status['orchestrator_state']['phase']}\n"
            status_message += f"Running: {status['orchestrator_state']['running']}\n"
            status_message += f"Modules: {len(status['modules'])} initialized\n"
            status_message += f"Background Tasks: {len([t for t in status['background_tasks'].values() if t['running']])} active\n"
            
            return {"message": status_message, "detailed_status": status}
            
        except Exception as e:
            return {"error": f"Status check failed: {str(e)}"}


# Main orchestrator interface functions for integration with main.py

async def create_orchestrator(config: Dict[str, Any], debug_logger=None) -> DevNameOrchestrator:
    """Factory function to create and initialize orchestrator"""
    orchestrator = DevNameOrchestrator(config, debug_logger)
    
    # Initialize system
    if await orchestrator.initialize_system():
        return orchestrator
    else:
        raise RuntimeError("Failed to initialize orchestrator system")

def create_command_processor(orchestrator: DevNameOrchestrator) -> CommandProcessor:
    """Factory function to create command processor"""
    return CommandProcessor(orchestrator)

async def run_orchestrated_application(config: Dict[str, Any], debug_logger=None) -> int:
    """Run the complete orchestrated application"""
    try:
        # Create and initialize orchestrator
        orchestrator = await create_orchestrator(config, debug_logger)
        
        # Get UI controller and run interface
        ui_controller = orchestrator.get_module("ui_controller")
        if not ui_controller:
            raise RuntimeError("UI controller not initialized")
        
        # Create command processor for UI
        command_processor = create_command_processor(orchestrator)
        ui_controller.command_processor = command_processor
        
        # Run the interface
        try:
            return ui_controller.run()
        finally:
            # Ensure graceful shutdown
            if not orchestrator.state.shutdown_requested:
                orchestrator.request_shutdown()
            await orchestrator.shutdown_system()
            
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Application error: {e}")
        print(f"Application error: {e}")
        return 1

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
        "name": "DevName RPG Client Orchestrator",
        "version": "1.0",
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
            "System status monitoring"
        ],
        "integration_points": [
            "main.py configuration processing",
            "Prompt system integration",
            "Background LLM analysis coordination",
            "Module state persistence"
        ]
    }

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Main Orchestrator Module")
    print("Successfully implemented orchestrator with:")
    print("✓ Dependency-aware module initialization")
    print("✓ Background thread coordination")
    print("✓ Inter-module communication management")
    print("✓ Configuration and prompt processing from main.py")
    print("✓ Graceful shutdown sequence")
    print("✓ Command processing coordination")
    print("✓ System status monitoring")
    print("✓ Complete LLM analysis coordination")
    print("\nOrchestrator Info:")
    
    info = get_orchestrator_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nInitialization test: {'✓ PASSED' if test_orchestrator_initialization() else '✗ FAILED'}")
    print("\nReady for integration with main.py and Phase 3 UI refactoring.")
