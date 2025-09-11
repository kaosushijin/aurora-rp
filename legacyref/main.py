# Chunk 1/3 - main.py - Core Dependencies and Configuration
#!/usr/bin/env python3
"""
DevName RPG Client - Main Application Entry Point (main.py)

Module architecture and interconnects documented in genai.txt
Coordinates all modules: nci.py, mcp.py, emm.py, sme.py
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

# Import application modules
try:
    from nci import CursesInterface
    from mcp import MCPClient
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Ensure all module files are present in the current directory")
    sys.exit(1)

# Configuration constants
DEFAULT_CONFIG_FILE = "devname_config.json"
DEBUG_LOG_FILE = "debug.log"
MAX_LOG_AGE_DAYS = 7

# Prompt file configuration
PROMPT_FILES = {
    'critrules': Path("critrules.prompt"),
    'companion': Path("companion.prompt"), 
    'lowrules': Path("lowrules.prompt")
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
    """Application configuration management - hardcoded values"""
    
    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        self.config_file = config_file
        self.config_data = self._get_hardcoded_config()
    
    def _get_hardcoded_config(self) -> Dict[str, Any]:
        """Return hardcoded configuration - no file creation"""
        return {
            "mcp": {
                "server_url": "http://127.0.0.1:3456/chat",
                "model": "qwen2.5:14b-instruct-q4_k_m",
                "timeout": 300
            },
            "interface": {
                "color_theme": "classic",
                "auto_save_conversation": False
            },
            "memory": {
                "max_tokens": 16000
            },
            "story": {
                "pressure_decay_rate": 0.05,
                "antagonist_threshold": 0.6
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
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
                "Maintain all essential character traits while reducing token count:\n\n"
                f"{content}\n\n"
                "Provide only the condensed prompt text that fully preserves the companion character."
            ),
            "lowrules": (
                "You are optimizing a narrative generation prompt for an RPG system. "
                "Condense the following prompt while preserving all narrative guidelines, "
                "storytelling rules, and generation principles. Maintain effectiveness "
                "in guiding story creation while reducing token count:\n\n"
                f"{content}\n\n"
                "Provide only the condensed prompt text that maintains narrative quality."
            )
        }
        
        prompt_instruction = condensation_prompts.get(prompt_type, condensation_prompts["critrules"])
        
        try:
            condensed = await self.mcp_client.send_message(
                "Please condense this prompt.",
                conversation_history=[],
                story_context=""
            )
            
            # Override system prompt temporarily for condensation
            original_prompt = self.mcp_client.system_prompt
            self.mcp_client.system_prompt = prompt_instruction
            
            # Get condensed version
            response = await self.mcp_client._execute_request({
                "model": self.mcp_client.model,
                "messages": [{"role": "system", "content": prompt_instruction}],
                "stream": False
            })
            
            # Restore original prompt
            self.mcp_client.system_prompt = original_prompt
            
            self._log_debug(f"Condensed {prompt_type}: {len(content)} -> {len(response)} chars")
            return response.strip()
            
        except Exception as e:
            self._log_debug(f"Condensation failed for {prompt_type}: {e}")
            print(f"Warning: Failed to condense {prompt_type} prompt: {e}")
            return content
    
    async def load_and_optimize_prompts(self, mcp_client: MCPClient) -> Dict[str, str]:
        """Load all prompt files and apply condensation if they exceed the token budget"""
        self.mcp_client = mcp_client
        
        # Load all prompt files with graceful handling
        prompts = {}
        for prompt_type, file_path in PROMPT_FILES.items():
            prompts[prompt_type] = self.load_prompt_file(file_path)
        
        # Check if any required prompts are missing
        missing_prompts = [name for name, content in prompts.items() if not content.strip()]
        if missing_prompts:
            print(f"Warning: Missing prompt files: {', '.join(missing_prompts)}")
            print("System will continue with available prompts")
        
        # Critical validation - critrules is required
        if not prompts.get('critrules', '').strip():
            raise ValueError(
                "Critical prompt file 'critrules.prompt' is missing!\n"
                "The system cannot function without the core game master rules.\n"
                "Please ensure 'critrules.prompt' exists in the current directory."
            )
        
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

# Chunk 2/3 - main.py - Application Classes and Initialization

class DevNameRPGClient:
    """Main application class with prompt management integration"""
    
    def __init__(self, config: ApplicationConfig, debug_logger: Optional[DebugLogger] = None):
        self.config = config
        self.debug_logger = debug_logger
        self.interface = None
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
        """Load and process prompt files"""
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
    
    def run(self) -> int:
        """Run the application with async prompt loading"""
        try:
            if self.debug_logger:
                self.debug_logger.system("Application starting")
            
            # Run async prompt loading
            asyncio.run(self._load_prompts())
            
            # Initialize interface with loaded prompts
            interface_config = {
                'color_theme': self.config.get('interface.color_theme', 'classic'),
                'auto_save_conversation': self.config.get('interface.auto_save_conversation', False),
                'mcp': {
                    'server_url': self.config.get('mcp.server_url'),
                    'model': self.config.get('mcp.model'),
                    'timeout': self.config.get('mcp.timeout')
                },
                'prompts': self.loaded_prompts  # Pass prompts to interface
            }
            
            self.interface = CursesInterface(
                debug_logger=self.debug_logger,
                config=interface_config
            )
            
            self.running = True
            
            if self.debug_logger:
                self.debug_logger.system("Starting interface")
            
            # Run interface
            exit_code = self.interface.run()
            
            if self.debug_logger:
                self.debug_logger.system(f"Application finished with exit code: {exit_code}")
            
            return exit_code
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Application error: {e}")
            
            print(f"Application error: {e}")
            return 1
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        if self.interface:
            self.interface.shutdown()
        
        if self.debug_logger:
            self.debug_logger.system("Application shutdown complete")

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
        description="DevName RPG Client - Modular ncurses RPG interface",
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
        version='DevName RPG Client 1.0.0'
    )
    
    return parser

def initialize_environment():
    """Initialize application environment"""
    # Clean up old log files
    cleanup_old_files()
    
    # Ensure required directories exist
    Path('.').mkdir(exist_ok=True)

def cleanup_old_files():
    """Clean up old log and history files"""
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
    for history_file in Path('.').glob('chat_history_*.json'):
        try:
            if history_file.stat().st_mtime < cutoff_time:
                history_file.unlink()
        except Exception:
            pass

def verify_modules() -> bool:
    """Verify all required modules are available"""
    required_modules = {
        'nci': 'Ncurses interface module',
        'mcp': 'MCP communication module', 
        'emm': 'Enhanced memory manager',
        'sme': 'Story momentum engine'
    }
    
    missing_modules = []
    
    for module_name, description in required_modules.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(f"{module_name}.py ({description})")
    
    if missing_modules:
        print("Missing required modules:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nEnsure all module files are present in the current directory.")
        return False
    
    return True

def show_startup_info():
    """Show minimal startup information"""
    print("DevName RPG Client")
    print("Starting modular RPG interface...")

def initialize_application(args) -> DevNameRPGClient:
    """Initialize main application"""
    # Initialize debug logger
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    if debug_logger:
        debug_logger.system("DevName RPG Client starting")
        debug_logger.system(f"Arguments: {vars(args)}")
        debug_logger.system("Using hardcoded configuration values")
    
    # Load configuration (hardcoded values)
    config = ApplicationConfig(args.config)
    
    if debug_logger:
        debug_logger.system("Configuration loaded with hardcoded values")
    
    # Create application
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
    """Validate prompt file existence and provide helpful feedback"""
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
        
        # Check for critical missing file
        if not PROMPT_FILES['critrules'].exists():
            print("WARNING: critrules.prompt is required for core functionality")
            return False
    
    return True

# Chunk 3/3 - main.py - Main Entry Point and Application Execution

def main():
    """Main application entry point"""
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize environment
        initialize_environment()
        
        # Verify all modules are available
        if not verify_modules():
            print("Module verification failed")
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
        
        # Validate prompt files
        if not validate_prompt_files():
            print("Critical prompt files missing - application cannot start")
            return 1
        
        # Show startup info
        show_startup_info()
        
        # Initialize application
        app = initialize_application(args)
        
        # Small delay for user to read messages
        time.sleep(1)
        
        # Run application
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
