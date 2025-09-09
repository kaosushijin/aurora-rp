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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import application modules
try:
    from nci import CursesInterface
except ImportError as e:
    print(f"Failed to import interface module: {e}")
    print("Ensure nci.py is present in the current directory")
    sys.exit(1)

# Configuration constants
DEFAULT_CONFIG_FILE = "aurora_config.json"
DEBUG_LOG_FILE = "debug.log"
MAX_LOG_AGE_DAYS = 7

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
    """Application configuration management"""
    
    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        self.config_file = config_file
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "mcp": {
                "server_url": "http://localhost:11434/api/chat",
                "model": "qwen2.5:14b-instruct",
                "timeout": 30
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
        
        # Save default config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception:
            pass
        
        return default_config
    
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

class DevNameRPGClient:
    """Main application class"""
    
    def __init__(self, config: ApplicationConfig, debug_logger: Optional[DebugLogger] = None):
        self.config = config
        self.debug_logger = debug_logger
        self.interface = None
        self.running = False
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self.debug_logger:
            self.debug_logger.system(f"Received signal {signum}")
        
        self.shutdown()
    
    def run(self) -> int:
        """Run the application"""
        try:
            if self.debug_logger:
                self.debug_logger.system("Application starting")
            
            # Initialize interface
            interface_config = {
                'color_theme': self.config.get('interface.color_theme', 'classic'),
                'auto_save_conversation': self.config.get('interface.auto_save_conversation', False),
                'mcp': {
                    'server_url': self.config.get('mcp.server_url'),
                    'model': self.config.get('mcp.model'),
                    'timeout': self.config.get('mcp.timeout')
                }
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

# Chunk 2/3 - main.py - Argument Parsing and Environment Setup

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
        help=f'Configuration file (default: {DEFAULT_CONFIG_FILE})'
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
    
    # Load configuration
    config = ApplicationConfig(args.config)
    
    if debug_logger:
        debug_logger.system(f"Configuration loaded from {args.config}")
    
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
