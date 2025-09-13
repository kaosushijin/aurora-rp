# Chunk 1/4 - main.py - Imports and Core Classes
# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.
#!/usr/bin/env python3
"""
DevName RPG Client - Main Application Entry Point (main.py)

Remodularized version with hub-and-spoke architecture via orchestrator.
All modules in root directory with flat file structure.
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

# Verify current directory has all required modules
current_dir = Path.cwd()
required_modules = ['orch', 'mcp', 'ncui', 'emm', 'sme', 'sem', 'uilib']
missing_modules = []

for module_name in required_modules:
    module_file = current_dir / f"{module_name}.py"
    if not module_file.exists():
        missing_modules.append(f"{module_name}.py (expected at {module_file})")

# Import application modules from current directory
try:
    from orch import Orchestrator
except ImportError as e:
    missing_modules.append(f"orch.py import failed: {e}")

# Exit if critical modules failed to import
if missing_modules:
    print("Failed to import required modules:")
    for module_error in missing_modules:
        print(f"  - {module_error}")
    print("\nEnsure all remodularized module files are present in current directory:")
    print("Required files: orch.py, mcp.py, ncui.py, emm.py, sme.py, sem.py, uilib.py")
    print(f"Current directory: {current_dir}")
    sys.exit(1)

# Configuration constants
DEFAULT_CONFIG_FILE = "devname_config.json"
DEBUG_LOG_FILE = "debug.log"
MAX_LOG_AGE_DAYS = 7

# Prompt file configuration - all files in current directory
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

    def system(self, message: str):
        """Log system message"""
        self.debug(message, "SYSTEM")

class ApplicationConfig:
    """Configuration management with hardcoded defaults"""
    
    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        self.config_file = config_file
        self.config_data = self._load_hardcoded_config()
    
    def _load_hardcoded_config(self) -> Dict[str, Any]:
        """Return hardcoded configuration values - FIXED: Correct MCP server port"""
        return {
            "mcp": {
                "server_url": "http://localhost:3456/chat",  # FIXED: Changed from 3000 to 3456
                "model": "qwen2.5:14b-instruct-q4_k_m",    # FIXED: Use correct model name from logs
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 2
            },
            "ui": {
                "refresh_rate": 0.1,
                "auto_save_interval": 30,
                "max_display_messages": 50
            },
            "memory": {
                "max_tokens": 20000,
                "condensation_threshold": 0.8,
                "auto_save": True
            },
            "analysis": {
                "momentum_threshold": 15,
                "semantic_analysis": True,
                "background_processing": True
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

# Chunk 2/4 - main.py - Utility Functions and DevNameRPGClient

def estimate_tokens(text: str) -> int:
    """Conservative token estimation"""
    if not text:
        return 0
    return max(1, len(text) // 4)

class DevNameRPGClient:
    """Main application coordinator using orchestrator pattern"""
    
    def __init__(self, config: ApplicationConfig, debug_logger: Optional[DebugLogger] = None):
        self.config = config
        self.debug_logger = debug_logger
        self.orchestrator = None
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._log_system("DevName RPG Client starting with orchestrator")
    
    def _log_system(self, message: str):
        """System logging helper"""
        if self.debug_logger:
            self.debug_logger.system(message)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self._log_system(f"Signal {signum} received, requesting shutdown")
        self.shutdown_requested = True
    
    def run(self) -> int:
        """Main application run with orchestrator coordination"""
        try:
            self._log_system("Starting orchestrator run sequence")

            # Load prompt files first
            prompt_data = load_prompt_files()

            # Create and run orchestrator with prompts
            self.orchestrator = Orchestrator(
                config=self.config.config_data,  # Pass config data, not config object
                loaded_prompts=prompt_data,      # Pass the loaded prompts
                debug_logger=self.debug_logger
            )
            exit_code = self.orchestrator.run()

            self._log_system(f"Orchestrator run completed with exit code: {exit_code}")
            return exit_code

        except KeyboardInterrupt:
            self._log_system("Keyboard interrupt received")
            return 0
        except Exception as e:
            self._log_system(f"Application error: {e}")
            return 1
        finally:
            if self.orchestrator:
                # Orchestrator handles its own cleanup
                pass

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="DevName RPG Client - Hub & Spoke Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --debug              # Run with debug logging
  python main.py --config custom.json # Use custom config file (unused in current version)
  
Note: All modules must be present in the current directory.
Required files: orch.py, mcp.py, ncui.py, emm.py, sme.py, sem.py, uilib.py
        """
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging to debug.log'
    )
    
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help=f'Configuration file (default: {DEFAULT_CONFIG_FILE}) - currently uses hardcoded values'
    )
    
    return parser

def verify_modules() -> bool:
    """Verify all remodularized modules are available"""
    required_files = [
        'orch.py', 'mcp.py', 'ncui.py', 'emm.py', 
        'sme.py', 'sem.py', 'uilib.py'
    ]
    
    missing_files = []
    for filename in required_files:
        if not Path(filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"Missing module files: {', '.join(missing_files)}")
        return False
    
    return True

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check for required and optional dependencies"""
    missing_required = []
    missing_optional = []
    
    # Check optional dependencies
    try:
        import httpx
    except ImportError:
        missing_optional.append('httpx')
    
    # All other dependencies are standard library
    return len(missing_required) == 0, missing_optional

def initialize_environment():
    """Initialize application environment"""
    # Create logs directory if needed
    Path('logs').mkdir(exist_ok=True)

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

# Chunk 3/4 - main.py - Startup Functions and Validation

def show_startup_info():
    """Show startup information with remodularization notes"""
    print("DevName RPG Client - Hub & Spoke Architecture")
    print("Running from root directory with orchestrator coordination...")
    print("✓ Remodularized with central hub orchestration")

def initialize_application(args) -> DevNameRPGClient:
    """Initialize main application with orchestrator"""
    # Initialize debug logger
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    if debug_logger:
        debug_logger.system("DevName RPG Client starting - remodularized version")
        debug_logger.system(f"Arguments: {vars(args)}")
        debug_logger.system("Using hub-and-spoke architecture with orchestrator")
        debug_logger.system(f"Running from root directory: {Path.cwd()}")
    
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
    """Validate prompt file existence in current directory"""
    missing_files = []
    existing_files = []
    
    for prompt_type, file_path in PROMPT_FILES.items():
        if file_path.exists():
            existing_files.append(prompt_type)
        else:
            missing_files.append(prompt_type)
    
    if existing_files:
        print(f"Found prompt files: {', '.join(existing_files)}")
    
    if missing_files:
        print(f"Missing prompt files: {', '.join(missing_files)}")
        
        # Check if critrules is missing (critical)
        if 'critrules' in missing_files:
            print("WARNING: critrules.prompt is required for core functionality")
            print("The system may not function properly without core game master rules")
            # Don't prevent startup, but warn user
            return True
        else:
            print("Note: Missing files are optional, system will continue")
    
    return True

def load_prompt_files() -> Dict[str, str]:
    """Load all prompt files with graceful handling of missing files"""
    prompts = {}
    total_tokens = 0
    
    for prompt_type, file_path in PROMPT_FILES.items():
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                tokens = estimate_tokens(content)
                prompts[prompt_type] = content
                total_tokens += tokens
                print(f"Loaded {prompt_type}: {len(content)} chars ({tokens} tokens)")
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                prompts[prompt_type] = ""
        else:
            prompts[prompt_type] = ""
    
    print(f"Total prompt tokens: {total_tokens}/{SYSTEM_PROMPT_TOKENS}")
    
    if total_tokens > SYSTEM_PROMPT_TOKENS:
        print(f"Warning: Prompts exceed token budget by {total_tokens - SYSTEM_PROMPT_TOKENS} tokens")
        print("Consider condensing prompts or increasing budget")
    
    return prompts

# Chunk 4/4 - main.py - Main Function and Entry Point

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

        # Validate prompt files in current directory
        if not validate_prompt_files():
            print("Critical prompt file validation failed")
            return 1

        # Load prompt files (informational only - orchestrator handles actual loading)
        prompt_data = load_prompt_files()

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
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# End of main.py - DevName RPG Client Main Application
# Updated for hub-and-spoke architecture with orch.py orchestrator
# All files in root directory with flat file structure
