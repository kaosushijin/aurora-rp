#!/usr/bin/env python3
"""
DevName RPG Client - Main Application Entry Point (main.py)
Updated for hub-and-spoke architecture with orch.py orchestrator
Located in /remod-staging/ directory with all other modules

Module architecture and interconnects documented in genai.txt
Now coordinates through orch.py instead of nci.py directly
"""

import sys
import os
import json
import signal
import argparse
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Import application modules - all in same directory
try:
    from orch import Orchestrator
    from mcp import MCPClient
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Ensure all remodularized module files are present in current directory")
    sys.exit(1)

# Configuration constants
DEFAULT_CONFIG_FILE = "devname_config.json"
DEBUG_LOG_FILE = "debug.log"
MAX_LOG_AGE_DAYS = 7

# Prompt file configuration - look in parent directory
PROMPT_FILES = {
    'critrules': Path("../critrules.prompt"),
    'companion': Path("../companion.prompt"), 
    'lowrules': Path("../lowrules.prompt")
}

# Token allocation for prompt system
CONTEXT_WINDOW = 32000
SYSTEM_PROMPT_TOKENS = 5000  # Token budget for all prompts combined

class DebugLogger:
    """Simple debug logging functionality"""
    
    def __init__(self, enabled: bool = False, log_file: str = DEBUG_LOG_FILE):
        self.enabled = enabled
        self.log_file = log_file
        
        if enabled:
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize debug log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"\n--- Debug session started: {datetime.now().isoformat()} ---\n")
        except Exception:
            pass
    
    def debug(self, message: str, category: str = "MAIN"):
        """Log debug message"""
        if not self.enabled:
            return
        
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] {category}: {message}\n"
            
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass
    
    def error(self, message: str, category: str = "ERROR"):
        """Log error message"""
        self.debug(f"ERROR: {message}", category)

    def memory(self, message: str):
        """Log memory message"""
        self.debug(message, "MEMORY")
    
    def system(self, message: str):
        """Log system message"""
        self.debug(message, "SYSTEM")

class ApplicationConfig:
    """Hardcoded configuration values - no file creation"""
    
    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        # Hardcoded configuration - no file reading
        self.config_data = {
            "mcp": {
                "server_url": "http://localhost:3000/v1/chat/completions",
                "model": "gpt-4o",
                "timeout": 30
            },
            "interface": {
                "theme": "default",
                "output_ratio": 0.85,
                "auto_scroll": True,
                "max_history": 1000
            },
            "memory": {
                "auto_save_interval": 30,
                "max_messages": 1000,
                "backup_count": 5
            },
            "analysis": {
                "trigger_interval": 15,
                "max_analysis_time": 30,
                "enable_momentum": True,
                "enable_semantic": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

def estimate_tokens(text: str) -> int:
    """Conservative token estimation"""
    if not text:
        return 0
    return max(1, len(text) // 4)

class PromptManager:
    """Manages loading and condensation of prompt files"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.mcp_client = None
        
    def _log_debug(self, message: str):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, "PROMPT")
    
    def load_prompt_file(self, file_path: Path) -> str:
        """Load prompt file with graceful handling of missing files"""
        if not file_path.exists():
            self._log_debug(f"Prompt file not found: {file_path}")
            print(f"Warning: Prompt file not found: {file_path}")
            return ""
        
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            self._log_debug(f"Loaded {file_path.name}: {len(content)} chars")
            return content
        except Exception as e:
            self._log_debug(f"Failed to load {file_path}: {e}")
            print(f"Warning: Failed to load {file_path}: {e}")
            return ""
    
    async def condense_prompt(self, content: str, prompt_type: str) -> str:
        """Condense a single prompt file while preserving essential functionality"""
        if not self.mcp_client:
            self._log_debug("No MCP client available for condensation")
            return content
        
        self._log_debug(f"Condensing {prompt_type} prompt ({len(content)} chars)")
        
        # Design prompt-specific condensation instructions
        condensation_prompts = {
            "critrules": (
                "You are optimizing a Game Master system prompt for an RPG. "
                "Condense the following prompt while preserving all essential game master rules, "
                "narrative generation guidelines, and core functionality. Maintain the same purpose "
                "and effectiveness while reducing token count. Keep all critical instructions intact:\n\n"
                f"{content}\n\n"
                "Provide only the condensed prompt text that maintains full GM functionality."
            ),
            "companion": (
                "You are optimizing a character definition prompt for an RPG companion. "
                "Condense the following prompt while preserving the companion's complete personality, "
                "appearance, abilities, relationship dynamics, and behavioral patterns. "
                "Maintain character consistency while reducing token count:\n\n"
                f"{content}\n\n"
                "Provide only the condensed character prompt that preserves full personality."
            ),
            "lowrules": (
                "You are optimizing supplementary game rules for an RPG. "
                "Condense the following rules while preserving all mechanical functionality, "
                "edge case handling, and rule interactions. Keep essential game balance:\n\n"
                f"{content}\n\n"
                "Provide only the condensed rules that maintain complete functionality."
            )
        }
        
        condensation_prompt = condensation_prompts.get(prompt_type, 
            f"Condense the following text while preserving essential meaning and functionality:\n\n{content}"
        )
        
        try:
            messages = [{"role": "user", "content": condensation_prompt}]
            condensed = await self.mcp_client.send_message(messages)
            
            if condensed and len(condensed) < len(content):
                self._log_debug(f"Condensed {prompt_type}: {len(content)} -> {len(condensed)} chars")
                return condensed
            else:
                self._log_debug(f"Condensation failed or ineffective for {prompt_type}")
                return content
                
        except Exception as e:
            self._log_debug(f"Condensation error for {prompt_type}: {e}")
            return content
    
    async def load_and_optimize_prompts(self, mcp_client: MCPClient) -> Dict[str, str]:
        """Load all prompt files and apply condensation if needed"""
        self.mcp_client = mcp_client
        
        # Load all prompt files
        prompts = {}
        for prompt_type, file_path in PROMPT_FILES.items():
            content = self.load_prompt_file(file_path)
            prompts[prompt_type] = content
        
        # Calculate combined token count
        total_tokens = sum(estimate_tokens(content) for content in prompts.values() if content)
        
        self._log_debug(f"Loaded prompts: {total_tokens:,} tokens total")
        for prompt_type, content in prompts.items():
            if content:
                tokens = estimate_tokens(content)
                self._log_debug(f"  {prompt_type}: {tokens:,} tokens ({len(content):,} chars)")
            else:
                self._log_debug(f"  {prompt_type}: MISSING")
        
        # Apply condensation if budget exceeded
        if total_tokens > SYSTEM_PROMPT_TOKENS:
            print(f"Prompt files exceed token budget ({total_tokens:,} > {SYSTEM_PROMPT_TOKENS:,})")
            print("Applying intelligent condensation...")
            
            # Condense any file that's more than 1/3 of the total budget
            individual_threshold = SYSTEM_PROMPT_TOKENS // 3
            
            for prompt_type, content in prompts.items():
                if content and estimate_tokens(content) > individual_threshold:
                    self._log_debug(f"Condensing {prompt_type} prompt...")
                    prompts[prompt_type] = await self.condense_prompt(content, prompt_type)
            
            # Verify condensation worked
            new_total = sum(estimate_tokens(content) for content in prompts.values() if content)
            self._log_debug(f"Condensation complete: {new_total:,} tokens (saved {total_tokens - new_total:,})")
        
        return prompts

class DevNameRPGClient:
    """
    Main application class with orchestrator integration
    Updated to use orch.py instead of nci.py directly
    """
    
    def __init__(self, config: ApplicationConfig, debug_logger: Optional[DebugLogger] = None):
        self.config = config
        self.debug_logger = debug_logger
        self.orchestrator = None
        self.running = False
        self.prompt_manager = PromptManager(debug_logger)
        self.loaded_prompts = {}
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self.debug_logger:
            self.debug_logger.system(f"Received signal {signum}")
        
        self.shutdown()
    
    async def _load_prompts(self):
        """Load and process prompt files using temporary MCP client"""
        # Create temporary MCP client for prompt condensation
        temp_mcp = MCPClient(
            server_url=self.config.get('mcp.server_url'),
            model=self.config.get('mcp.model'),
            debug_logger=self.debug_logger
        )
        
        try:
            self.loaded_prompts = await self.prompt_manager.load_and_optimize_prompts(temp_mcp)
            
            if self.debug_logger:
                total_tokens = sum(estimate_tokens(content) for content in self.loaded_prompts.values() if content)
                self.debug_logger.system(f"Prompts loaded: {total_tokens} tokens")
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Prompt loading failed: {e}")
            raise
        finally:
            # Clean up temporary client
            temp_mcp.cleanup()
    
    def run(self) -> int:
        """
        Run the application with orchestrator coordination
        Updated to use orch.py instead of nci.py
        """
        try:
            self.running = True
            
            if self.debug_logger:
                self.debug_logger.system("DevName RPG Client starting with orchestrator")
            
            # Load prompts with async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self._load_prompts())
            except Exception as e:
                print(f"Failed to load prompts: {e}")
                return 1
            finally:
                loop.close()
            
            # Convert config to dict for orchestrator
            config_dict = self.config.config_data
            
            # Create orchestrator with loaded prompts
            self.orchestrator = Orchestrator(
                config=config_dict,
                loaded_prompts=self.loaded_prompts,
                debug_logger=self.debug_logger
            )
            
            if self.debug_logger:
                self.debug_logger.system("Orchestrator created")
            
            # Initialize all service modules through orchestrator
            if not self.orchestrator.initialize_modules():
                print("Failed to initialize service modules")
                if self.debug_logger:
                    self.debug_logger.error("Module initialization failed")
                return 1
            
            if self.debug_logger:
                self.debug_logger.system("All modules initialized successfully")
            
            # Run main program loop through orchestrator
            print("Starting main program loop...")
            exit_code = self.orchestrator.run_main_loop()
            
            if self.debug_logger:
                self.debug_logger.system(f"Main loop ended with exit code: {exit_code}")
            
            return exit_code
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Application run error: {e}")
            print(f"Application error: {e}")
            return 1
        finally:
            self.running = False
    
    def shutdown(self):
        """Graceful shutdown through orchestrator"""
        if self.debug_logger:
            self.debug_logger.system("Application shutdown initiated")
        
        self.running = False
        
        if self.orchestrator:
            self.orchestrator.shutdown_gracefully()
        
        if self.debug_logger:
            self.debug_logger.system("Application shutdown complete")

def verify_modules() -> bool:
    """Verify all required modules are available in current directory"""
    required_modules = [
        'orch',
        'ncui', 
        'emm',
        'sme',
        'sem',
        'uilib',
        'mcp'
    ]
    
    missing_modules = []
    
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(module_name)
    
    if missing_modules:
        print("ERROR: Missing required modules in current directory:")
        for module in missing_modules:
            print(f"  - {module}.py")
        print("\nEnsure all remodularized files are in the current directory")
        return False
    
    return True

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check for optional dependencies"""
    optional_deps = []
    
    try:
        import httpx
    except ImportError:
        optional_deps.append('httpx')
    
    try:
        import curses
    except ImportError:
        print("ERROR: curses module not available")
        print("Install with: pip install windows-curses (Windows) or ensure ncurses is installed (Unix)")
        return False, optional_deps
    
    return True, optional_deps

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="DevName RPG Client - Hub and Spoke Architecture with Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help=f'Configuration file (default: {DEFAULT_CONFIG_FILE}) - NOTE: Uses hardcoded values'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='DevName RPG Client 1.0.0 - Remodularized Hub & Spoke'
    )
    
    return parser

def initialize_environment():
    """Initialize application environment"""
    # Clean up old log files in current directory
    cleanup_old_files()
    
    # Ensure current directory is ready
    Path('.').mkdir(exist_ok=True)

def cleanup_old_files():
    """Clean up old log and history files in current directory"""
    import time
    
    current_time = time.time()
    cutoff_time = current_time - (MAX_LOG_AGE_DAYS * 24 * 60 * 60)
    
    # Clean up old debug logs
    for log_file in Path('.').glob('debug_*.log'):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
        except Exception:
            pass
    
    # Clean up old conversation history
    for history_file in Path('.').glob('conversation_*.json'):
        try:
            if history_file.stat().st_mtime < cutoff_time:
                history_file.unlink()
        except Exception:
            pass

def show_startup_info():
    """Show startup information with remodularization notes"""
    print("DevName RPG Client - Hub & Spoke Architecture")
    print("Running from remod-staging/ with orchestrator coordination...")
    print("✓ Remodularized with central hub orchestration")

def initialize_application(args) -> DevNameRPGClient:
    """Initialize main application with orchestrator"""
    # Initialize debug logger
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    if debug_logger:
        debug_logger.system("DevName RPG Client starting - remodularized version")
        debug_logger.system(f"Arguments: {vars(args)}")
        debug_logger.system("Using hub-and-spoke architecture with orchestrator")
        debug_logger.system("Running from remod-staging/ directory")
    
    # Load configuration (hardcoded values)
    config = ApplicationConfig(args.config)
    
    if debug_logger:
        debug_logger.system("Configuration loaded with hardcoded values")
    
    # Create application with orchestrator support
    app = DevNameRPGClient(config, debug_logger)
    
    return app

def check_terminal_requirements():
    """Check terminal capabilities"""
    try:
        # Get terminal size
        if hasattr(os, 'get_terminal_size'):
            size = os.get_terminal_size()
            if size.columns < 80 or size.lines < 24:
                print(f"Warning: Terminal size {size.columns}x{size.lines} may be too small")
                print("Recommended minimum: 80x24")
    except Exception:
        pass

def validate_prompt_files():
    """Validate prompt file existence in parent directory"""
    missing_files = []
    existing_files = []
    
    for prompt_type, file_path in PROMPT_FILES.items():
        if file_path.exists():
            existing_files.append(f"{prompt_type} ({file_path})")
        else:
            missing_files.append(f"{prompt_type} ({file_path})")
    
    if existing_files:
        print(f"Found prompt files: {', '.join([f.split()[0] for f in existing_files])}")
    
    if missing_files:
        print(f"Missing prompt files: {', '.join([f.split()[0] for f in missing_files])}")
        print("Note: Looking for prompt files in parent directory (../)")
        
        # Check for critical missing file
        if not PROMPT_FILES['critrules'].exists():
            print("WARNING: critrules.prompt is required for core functionality")
            return False
    
    return True

def main():
    """Main application entry point"""
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize environment
        initialize_environment()
        
        # Verify all remodularized modules are available
        if not verify_modules():
            print("Module verification failed")
            print("Ensure all remodularized files are in the current directory:")
            print("  - orch.py, ncui.py, emm.py, sme.py, sem.py, uilib.py, mcp.py")
            return 1
        
        # Check dependencies
        deps_ok, missing_optional = check_dependencies()
        if not deps_ok:
            return 1
        
        if missing_optional:
            print(f"Optional dependencies missing: {', '.join(missing_optional)}")
            if 'httpx' in missing_optional:
                print("MCP functionality will be limited. Install with: pip install httpx")
        
        # Check terminal
        check_terminal_requirements()
        
        # Validate prompt files in parent directory
        if not validate_prompt_files():
            print("Critical prompt files missing - application cannot start")
            print("Ensure critrules.prompt exists in parent directory")
            return 1
        
        # Show startup info
        show_startup_info()
        
        # Initialize application with orchestrator
        app = initialize_application(args)
        
        # Small delay for user to read messages
        time.sleep(1)
        
        # Run application through orchestrator
        exit_code = app.run()
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# End of main.py - DevName RPG Client Main Application
# Updated for hub-and-spoke architecture with orch.py orchestrator
# Designed as root codebase with all files in same directory
