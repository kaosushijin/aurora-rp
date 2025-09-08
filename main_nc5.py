#!/usr/bin/env python3
"""
Aurora RPG Client - Main Application Entry Point (main_nc5.py) - FIXED Chunk 1/3

CRITICAL: This comment block must be preserved in all files to ensure proper
understanding of the modular architecture when analyzed by generative models.

MODULAR ARCHITECTURE OVERVIEW:
This project uses a modular architecture with the following interconnected files:

1. main_nc5.py (THIS FILE): Main executable and application coordination
   - Handles command-line arguments, configuration, and application lifecycle
   - Imports and coordinates all other modules
   - Manages session state and graceful shutdown
   - Contains startup/shutdown logic and error handling

2. nci_nc5.py: Ncurses Interface Module
   - Complete ncurses interface implementation with fixed display pipeline
   - Input handling, screen management, color themes, context switching
   - Called by main_nc5.py for all user interface operations
   - Coordinates with other modules for display updates

3. mcp_nc5.py: MCP Communication Module  
   - HTTP client for Ollama/MCP server communication
   - Message formatting, retry logic, connection management
   - Called by nci_nc5.py when sending user messages
   - Provides enhanced context from sme_nc5.py

4. emm_nc5.py: Enhanced Memory Manager Module
   - Conversation history storage with semantic condensation
   - Token estimation and memory optimization
   - Called by nci_nc5.py for message storage/retrieval
   - Provides conversation context to mcp_nc5.py

5. sme_nc5.py: Story Momentum Engine Module
   - Dynamic narrative pressure and antagonist management
   - Analyzes conversation for story pacing
   - Called by nci_nc5.py to update pressure based on user input
   - Provides context enhancement for mcp_nc5.py requests

PROGRAMMATIC INTERCONNECTS:
- main_nc5.py → nci_nc5.py: Creates and runs CursesInterface
- nci_nc5.py → mcp_nc5.py: Sends messages via MCPClient
- nci_nc5.py → emm_nc5.py: Stores/retrieves messages via EnhancedMemoryManager
- nci_nc5.py → sme_nc5.py: Updates pressure via StoryMomentumEngine
- mcp_nc5.py ← sme_nc5.py: Receives story context for enhanced prompting
- mcp_nc5.py ← emm_nc5.py: Receives conversation history for context

PRESERVATION NOTICE:
When modifying any file in this project, you MUST preserve this comment block
to ensure that future analysis (human or AI) understands the full architecture
and interdependencies. Breaking these interconnects will cause system failures.

FIXES IN THIS VERSION:
- Added missing get_debug_content() method to DebugLogger class
- Fixed debug content display compatibility with nci_nc5.py
- Enhanced debug line storage and retrieval for /debug command
- Improved error handling in DebugLogger class

Main responsibilities of this file:
- Application initialization and configuration management
- Command-line argument parsing and validation
- Dependency checking and environment setup
- Session management and statistics tracking
- Graceful shutdown and resource cleanup
- Error handling and recovery coordination
"""

import os
import sys
import json
import argparse
import time
import signal
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Configuration constants
CONFIG_FILE = "aurora_config.json"
DEBUG_LOG_FILE = "debug.log"
CHAT_HISTORY_FILE = "chat_history.json"

class DebugLogger:
    """Centralized debug logging system for all modules - FIXED VERSION with get_debug_content()"""
    
    def __init__(self, enabled: bool = False, log_file: str = DEBUG_LOG_FILE):
        self.enabled = enabled
        self.log_file = Path(log_file)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_lines = []  # Store debug lines in memory for display - CRITICAL FIX
        
        if self.enabled:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Aurora RPG Phase 5 Debug Session: {self.session_id}\n")
                    f.write(f"Started: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*80}\n\n")
            except Exception as e:
                print(f"Warning: Could not initialize debug log: {e}")
                self.enabled = False
    
    def _write_log(self, level: str, category: str, message: str):
        """Write log entry to file and store in memory for display - FIXED"""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level:>6} | {category:>12} | {message}"
        
        # Store in memory for debug display (keep last 100 entries) - CRITICAL FIX
        self.debug_lines.append(log_entry)
        if len(self.debug_lines) > 100:
            self.debug_lines.pop(0)
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception:
            pass
    
    def system(self, message: str):
        self._write_log("SYSTEM", "MAIN", message)
    
    def debug(self, message: str, category: str = "DEBUG"):
        self._write_log("DEBUG", category, message)
    
    def error(self, message: str, category: str = "ERROR"):
        self._write_log("ERROR", category, message)
    
    def info(self, message: str, category: str = "INFO"):
        self._write_log("INFO", category, message)
    
    def get_debug_content(self) -> List[str]:
        """Get debug content for display - CRITICAL MISSING METHOD ADDED"""
        if not self.enabled:
            return [
                "Debug logging is not enabled.",
                "Start the application with --debug flag to enable detailed logging.",
                "",
                "Basic system information:",
                f"Session ID: {self.session_id}",
                f"Debug file: {self.log_file}",
                f"Current time: {datetime.now().strftime('%H:%M:%S')}",
                "",
                "To enable debug logging:",
                "python main_nc5.py --debug"
            ]
        
        # Return stored debug lines with header
        header_lines = [
            "Aurora RPG Client - Debug Information",
            f"Session: {self.session_id}",
            f"Log file: {self.log_file}",
            f"Total entries: {len(self.debug_lines)}",
            "=" * 50,
            ""
        ]
        
        return header_lines + self.debug_lines
    
    def close_session(self):
        if self.enabled:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Session {self.session_id} ended: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*80}\n\n")
            except Exception:
                pass

class SessionManager:
    """Session tracking and statistics management"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.message_count = 0
        self.command_count = 0
        self.mcp_requests = 0
        self.errors_encountered = 0
        
        if debug_logger:
            debug_logger.system(f"Session started: {self.session_id}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information"""
        duration = datetime.now() - self.start_time
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": int(duration.total_seconds()),
            "duration_formatted": str(duration).split('.')[0],
            "message_count": self.message_count,
            "command_count": self.command_count,
            "mcp_requests": self.mcp_requests,
            "errors_encountered": self.errors_encountered
        }
    
    def increment_message_count(self):
        self.message_count += 1
    
    def increment_command_count(self):
        self.command_count += 1
    
    def increment_mcp_requests(self):
        self.mcp_requests += 1
    
    def increment_error_count(self):
        self.errors_encountered += 1
    
    def end_session(self):
        if self.debug_logger:
            session_info = self.get_session_info()
            self.debug_logger.system(f"Session ended: {session_info}")

class ConfigManager:
    """Configuration management and persistence"""
    
    def __init__(self, config_file: str = CONFIG_FILE, debug_logger: Optional[DebugLogger] = None):
        self.config_file = Path(config_file)
        self.debug_logger = debug_logger
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            "color_scheme": "midnight_aurora",
            "debug_enabled": False,
            "auto_save_enabled": True,
            "auto_save_interval": 100,
            "max_chat_history": 1000,
            "input_timeout": 30,
            "mcp_server_url": "http://127.0.0.1:3456/chat",
            "mcp_model": "qwen2.5:14b-instruct-q4_k_m",
            "mcp_timeout": 300.0,
            "input_mode": "normal",
            "last_session": None,
            "memory_condensation_threshold": 8000,
            "sme_enabled": True
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                self.config.update(loaded_config)
                
                if self.debug_logger:
                    self.debug_logger.system(f"Configuration loaded from {self.config_file}")
            
            return self.config.copy()
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to load configuration: {e}")
            return self.config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            # Update internal config
            self.config.update(config)
            
            # Add current session info
            self.config["last_session"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            if self.debug_logger:
                self.debug_logger.system(f"Configuration saved to {self.config_file}")
            
            return True
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_value(self, key: str, default=None):
        """Get a specific configuration value"""
        return self.config.get(key, default)
    
    def set_config_value(self, key: str, value: Any):
        """Set a specific configuration value"""
        self.config[key] = value

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check for required and optional dependencies"""
    missing_required = []
    missing_optional = []
    
    # Check for ncurses (required)
    try:
        import curses
        # Quick test to ensure curses is available
        curses.wrapper(lambda stdscr: None)
    except (ImportError, curses.error):
        missing_required.append("ncurses")
    
    # Check for httpx (optional but preferred for MCP)
    try:
        import httpx
    except ImportError:
        missing_optional.append("httpx")
    
    # Check that our modular components can be imported
    module_errors = []
    try:
        import nci_nc5
    except ImportError as e:
        module_errors.append(f"nci_nc5: {e}")
    
    try:
        import mcp_nc5
    except ImportError as e:
        module_errors.append(f"mcp_nc5: {e}")
    
    try:
        import emm_nc5
    except ImportError as e:
        module_errors.append(f"emm_nc5: {e}")
    
    try:
        import sme_nc5
    except ImportError as e:
        module_errors.append(f"sme_nc5: {e}")
    
    if module_errors:
        missing_required.extend(module_errors)
    
    # Print dependency status
    if missing_required:
        print("Missing required dependencies:")
        for dep in missing_required:
            print(f"  - {dep}")
        return False, missing_optional
    
    print("All required dependencies available")
    
    if missing_optional:
        print("Missing optional dependencies:")
        for dep in missing_optional:
            print(f"  - {dep}")
        print("\nTo install optional dependencies:")
        print(f"  pip install {' '.join(missing_optional)}")
        print("\nThe application will work with limited functionality without these.")
    
    return True, missing_optional

# main_nc5.py - FIXED Chunk 2/3
# Main Application Class and Error Handling

def show_startup_banner():
    """Show enhanced startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                    AURORA RPG CLIENT - PHASE 5                  ║
║                    Modular Architecture Edition                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌟 Enhanced Features:                                           ║
║    • Modular design with clear separation of concerns           ║
║    • Ncurses interface with fixed display pipeline              ║
║    • Input blocking during MCP processing                       ║
║    • Advanced memory management with condensation               ║
║    • Story Momentum Engine with pressure dynamics               ║
║    • Three beautiful color themes                               ║
║                                                                  ║
║  🎮 Ready for Adventure:                                         ║
║    Aurora awaits with enhanced intelligence and memory          ║
║    Type /help for commands or start your journey                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup enhanced command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Aurora RPG Client Phase 5 - Modular Terminal RPG with MCP Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with default settings
  %(prog)s --debug                 # Enable debug logging to file
  %(prog)s --colorscheme forest_whisper  # Start with Forest Whisper theme
  %(prog)s --config custom.json    # Use custom configuration file

Phase 5 Features:
  • Modular architecture (main/nci/mcp/emm/sme modules)
  • Enhanced MCP integration with input blocking
  • Advanced memory management with condensation
  • Story Momentum Engine with pressure dynamics
  • Fixed display pipeline and immediate refresh
        """
    )
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging to debug.log file")
    
    parser.add_argument("--colorscheme", "--theme", default="midnight_aurora",
                       choices=["midnight_aurora", "forest_whisper", "dracula_aurora"],
                       help="Initial color scheme (can be changed with /color command)")
    
    parser.add_argument("--config", default=CONFIG_FILE,
                       help="Configuration file path")
    
    parser.add_argument("--mcp-url", default="http://127.0.0.1:3456/chat",
                       help="MCP server URL")
    
    parser.add_argument("--mcp-model", default="qwen2.5:14b-instruct-q4_k_m",
                       help="MCP model name")
    
    parser.add_argument("--version", action="version", 
                       version="Aurora RPG Client Phase 5 v5.0.0")
    
    return parser

class AuroraRPGClient:
    """Main application class for Aurora RPG Client Phase 5"""
    
    def __init__(self, debug_enabled: bool = False, color_scheme: str = "midnight_aurora"):
        self.debug_logger = DebugLogger(debug_enabled, DEBUG_LOG_FILE) if debug_enabled else None
        self.session_manager = SessionManager(self.debug_logger)
        self.config_manager = ConfigManager(CONFIG_FILE, self.debug_logger)
        
        # Load configuration
        self.config = self.config_manager.load_config()
        
        # Override with parameters
        if color_scheme:
            self.config['color_scheme'] = color_scheme
        
        # Interface will be created in run()
        self.curses_interface = None
        
        if self.debug_logger:
            self.debug_logger.system("Aurora RPG Client Phase 5 initialized")
    
    def run(self) -> int:
        """Run the Aurora RPG Client with modular ncurses interface"""
        if self.debug_logger:
            self.debug_logger.system("Starting Aurora RPG Client Phase 5")
        
        try:
            # Import the ncurses interface module
            from nci_nc5 import CursesInterface
            
            # Create and configure the interface
            self.curses_interface = CursesInterface(
                debug_logger=self.debug_logger,
                config=self.config
            )
            
            # Run the interface (this handles curses.wrapper internally)
            exit_code = self.curses_interface.run()
            
            print("\n" + "="*60)
            print("Thank you for your adventure with Aurora!")
            
            # Show session summary
            session_info = self.session_manager.get_session_info()
            print(f"Session Duration: {session_info['duration_formatted']}")
            print(f"Messages Exchanged: {session_info['message_count']}")
            print(f"Commands Used: {session_info['command_count']}")
            print(f"MCP Requests: {session_info['mcp_requests']}")
            
            print("Your conversation has been saved.")
            print("="*60)
            
            return exit_code
            
        except ImportError as e:
            print(f"Error: Cannot import required module: {e}")
            print("Please ensure all Aurora RPG modules are present:")
            print("  - nci_nc5.py (Ncurses interface)")
            print("  - mcp_nc5.py (MCP communication)")
            print("  - emm_nc5.py (Enhanced memory manager)")
            print("  - sme_nc5.py (Story Momentum Engine)")
            return 1
            
        except Exception as e:
            error_msg = f"Critical error: {e}"
            if self.debug_logger:
                self.debug_logger.error(f"CRITICAL: {error_msg}")
                import traceback
                self.debug_logger.error(f"Stack trace: {traceback.format_exc()}")
            
            print(f"\nERROR: {error_msg}")
            if self.debug_logger:
                print(f"Detailed error information has been logged to {DEBUG_LOG_FILE}")
            print("Please report this error if it persists.")
            
            return 1
    
    def cleanup(self):
        """Cleanup application resources"""
        try:
            # End session tracking
            self.session_manager.end_session()
            
            # Save current configuration
            self._save_configuration()
            
            # Save conversation state
            self._save_conversation_state()
            
            # Close debug session
            if self.debug_logger:
                self.debug_logger.close_session()
        
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Cleanup error: {e}")
    
    def _save_configuration(self):
        """Save current configuration to file"""
        try:
            if self.curses_interface:
                # Update config with current interface state
                interface_config = self.curses_interface.get_config_updates()
                self.config.update(interface_config)
            
            # Add session statistics
            session_info = self.session_manager.get_session_info()
            self.config["last_session_stats"] = session_info
            
            # Save configuration
            self.config_manager.save_config(self.config)
            
            if self.debug_logger:
                self.debug_logger.system("Configuration saved successfully")
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save configuration: {e}")
    
    def _save_conversation_state(self):
        """Save conversation state for potential restoration"""
        try:
            if self.curses_interface:
                # Get conversation export data
                conversation_data = self.curses_interface.export_conversation_state()
                
                # Add session info
                conversation_data.update({
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_manager.session_id,
                    "session_stats": self.session_manager.get_session_info()
                })
                
                with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2)
                
                if self.debug_logger:
                    self.debug_logger.system(f"Conversation state saved to {CHAT_HISTORY_FILE}")
                    
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save conversation state: {e}")

class ErrorRecovery:
    """Enhanced error recovery system"""
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        self.debug_logger = debug_logger
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = None
    
    def handle_error(self, error: Exception, context: str = "unknown") -> bool:
        """
        Handle an error and determine if application should continue
        Returns True if application should continue, False if it should exit
        """
        current_time = time.time()
        self.error_count += 1
        
        if self.debug_logger:
            self.debug_logger.error(f"Error #{self.error_count} in {context}: {str(error)}")
        
        # Reset error count if enough time has passed
        if self.last_error_time and (current_time - self.last_error_time) > 60:
            self.error_count = 1
        
        self.last_error_time = current_time
        
        # If too many errors in short time, recommend exit
        if self.error_count >= self.max_errors:
            if self.debug_logger:
                self.debug_logger.error(f"Too many errors ({self.error_count}), recommending exit")
            return False
        
        return True
    
    def reset_error_count(self):
        """Reset error count after successful operations"""
        if self.error_count > 0:
            if self.debug_logger:
                self.debug_logger.system(f"Resetting error count from {self.error_count}")
            self.error_count = 0

def initialize_application(args) -> AuroraRPGClient:
    """Initialize application with enhanced setup"""
    # Check dependencies first
    deps_ok, missing_optional = check_dependencies()
    if not deps_ok:
        print("Dependency check failed. Exiting.")
        sys.exit(1)
    
    # Initialize debug logger
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    if debug_logger:
        debug_logger.system("Phase 5 modular initialization starting")
        debug_logger.system(f"Arguments: {vars(args)}")
        debug_logger.system(f"Python version: {sys.version}")
        debug_logger.system(f"Platform: {sys.platform}")
    
    # Create application
    app = AuroraRPGClient(args.debug, args.colorscheme)
    
    # Override MCP settings if provided
    if args.mcp_url != "http://127.0.0.1:3456/chat":
        app.config['mcp_server_url'] = args.mcp_url
    if args.mcp_model != "qwen2.5:14b-instruct-q4_k_m":
        app.config['mcp_model'] = args.mcp_model
    
    if debug_logger:
        debug_logger.system(f"Application initialized with MCP URL: {app.config.get('mcp_server_url')}")
    
    return app

def setup_signal_handlers(app: AuroraRPGClient):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        if app.debug_logger:
            app.debug_logger.system(f"Received signal {signum}, shutting down gracefully")
        
        # Cleanup application
        try:
            app.cleanup()
        except:
            pass
        
        print("\nAurora RPG Client shutdown complete. Goodbye!")
        sys.exit(0)
    
    # Handle common termination signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

def initialize_environment():
    """Initialize Phase 5 environment and perform setup"""
    # Create necessary directories
    directories = ['backups', 'exports', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Clean up old files (older than 30 days)
    cleanup_old_files(max_age_days=30)
    
    return True

def cleanup_old_files(max_age_days: int = 30):
    """Clean up old log files and backups"""
    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    
    # Clean up old debug logs
    for log_file in Path('.').glob('debug_*.log'):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
        except Exception:
            pass
    
    # Clean up old conversation history files
    for history_file in Path('.').glob('chat_history_*.json'):
        try:
            if history_file.stat().st_mtime < cutoff_time:
                history_file.unlink()
        except Exception:
            pass

# main_nc5.py - FIXED Chunk 3/3
# Main Entry Point and Application Execution

def generate_error_report(error: Exception, context: str, debug_logger: DebugLogger = None) -> str:
    """Generate comprehensive error report"""
    import traceback
    import platform
    
    report_lines = [
        "Aurora RPG Client Phase 5 - Error Report",
        "=" * 50,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Version: 5.0.0",
        f"Context: {context}",
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        "",
        "Error Details:",
        f"Type: {type(error).__name__}",
        f"Message: {str(error)}",
        "",
        "Stack Trace:",
        traceback.format_exc(),
        "",
        "System Information:",
        f"Terminal Size: {os.get_terminal_size() if hasattr(os, 'get_terminal_size') else 'Unknown'}",
        "",
        "Module Status:",
        "- main_nc5.py: Main application (this module)",
        "- nci_nc5.py: Ncurses interface module",
        "- mcp_nc5.py: MCP communication module",
        "- emm_nc5.py: Enhanced memory management module", 
        "- sme_nc5.py: Story Momentum Engine module",
        ""
    ]
    
    if debug_logger and hasattr(debug_logger, 'get_debug_content'):
        report_lines.extend([
            "Recent Debug Log Entries:",
            "-" * 30
        ])
        
        # Add last few debug entries
        try:
            debug_content = debug_logger.get_debug_content()
            report_lines.extend([line for line in debug_content[-20:]])  # Last 20 entries
        except Exception:
            report_lines.append("Could not read debug log")
    
    return "\n".join(report_lines)

def verify_modular_implementation() -> bool:
    """Verify all required modules are available"""
    required_modules = ['nci_nc5', 'mcp_nc5', 'emm_nc5', 'sme_nc5']
    
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            print(f"Missing required module: {module_name}.py")
            return False
    
    return True

def main():
    """Enhanced main application entry point"""
    # Show startup banner
    show_startup_banner()
    
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize debug logger first for early logging
    debug_logger = DebugLogger(args.debug, DEBUG_LOG_FILE) if args.debug else None
    
    if debug_logger:
        debug_logger.system("Aurora RPG Client Phase 5 starting")
        debug_logger.system(f"Command line arguments: {vars(args)}")
    
    app = None
    
    try:
        # Verify modular implementation
        if not verify_modular_implementation():
            print("Modular implementation incomplete")
            print("Please ensure all module files are present:")
            print("  - main_nc5.py (this file)")
            print("  - nci_nc5.py (Ncurses interface module)")
            print("  - mcp_nc5.py (MCP client module)")
            print("  - emm_nc5.py (Enhanced memory manager)")
            print("  - sme_nc5.py (Story Momentum Engine)")
            print("\nSome features may not work correctly.")
            print("Press Enter to continue anyway or Ctrl+C to exit...")
            try:
                input()
            except KeyboardInterrupt:
                return 1
        
        # Initialize environment
        initialize_environment()
        
        # Initialize application
        app = initialize_application(args)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(app)
        
        # Show dependency status
        deps_ok, missing_optional = check_dependencies()
        if missing_optional:
            print(f"MCP integration limited ({', '.join(missing_optional)} not found)")
            if 'httpx' in missing_optional:
                print("  Install with: pip install httpx")
        else:
            print("MCP integration available")
        
        print("Modular architecture loaded:")
        print("  - main_nc5.py: Core application coordination")
        print("  - nci_nc5.py: Ncurses interface management")
        print("  - mcp_nc5.py: MCP server communication")
        print("  - emm_nc5.py: Enhanced memory management")
        print("  - sme_nc5.py: Story Momentum Engine")
        
        print("\nStarting Aurora RPG Client...")
        print("Press Ctrl+C at any time for graceful shutdown.")
        
        # Small delay to let user read startup message
        time.sleep(2)
        
        # Run application
        exit_code = app.run()
        
        return exit_code
    
    except KeyboardInterrupt:
        if debug_logger:
            debug_logger.system("Application interrupted by user (Ctrl+C)")
        print("\n\nGraceful shutdown initiated...")
        print("Thank you for using Aurora RPG Client Phase 5!")
        return 0
    
    except Exception as e:
        error_msg = f"Critical error: {e}"
        if debug_logger:
            debug_logger.error(f"CRITICAL: {error_msg}")
            debug_logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            debug_logger.error(f"Stack trace: {traceback.format_exc()}")
        
        print(f"\nERROR: {error_msg}")
        if debug_logger:
            print(f"Detailed error information has been logged to {DEBUG_LOG_FILE}")
        print("Please report this error if it persists.")
        
        return 1
    
    finally:
        # Comprehensive cleanup
        if app:
            try:
                print("Cleaning up resources...")
                app.cleanup()
                print("Cleanup completed successfully.")
            except Exception as e:
                print(f"Cleanup warning: {e}")

# Utility functions for token estimation and validation
def estimate_tokens(text: str) -> int:
    """Enhanced token estimation for validation"""
    # More accurate estimation considering word boundaries
    words = text.split()
    chars = len(text)
    
    # Average tokens per word is about 1.3, plus character-based estimation
    word_tokens = len(words) * 1.3
    char_tokens = chars / 4
    
    # Use the higher estimate for safety
    return int(max(word_tokens, char_tokens))

def validate_mcp_config(config: Dict[str, Any]) -> bool:
    """Validate MCP configuration settings"""
    required_keys = ['mcp_server_url', 'mcp_model']
    
    for key in required_keys:
        if key not in config:
            return False
    
    # Validate URL format
    url = config['mcp_server_url']
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    return True

def check_for_updates() -> Dict[str, Any]:
    """Check for Phase 5 updates (placeholder for future implementation)"""
    return {
        "current_version": "5.0.0",
        "latest_version": "5.0.0",
        "update_available": False,
        "update_url": None,
        "changelog": [
            "Phase 5.0.0: Modular architecture with enhanced MCP integration",
            "- Fixed display pipeline with immediate refresh",
            "- Input blocking during MCP processing", 
            "- Advanced memory management with condensation",
            "- Story Momentum Engine with pressure dynamics",
            "- Comprehensive error handling and recovery"
        ]
    }

# Version and build information
__version__ = "5.0.0"
__build_date__ = "2024-09-08"
__author__ = "Aurora RPG Development Team"
__description__ = "Phase 5 - Modular Architecture with Enhanced MCP Integration"

# Phase 5 feature flags and constants
PHASE_5_FEATURES = {
    "modular_architecture": True,
    "enhanced_mcp_integration": True,
    "input_blocking_support": True,
    "fixed_display_pipeline": True,
    "advanced_memory_management": True,
    "story_momentum_engine": True,
    "comprehensive_error_handling": True,
    "session_management": True,
    "configuration_persistence": True
}

# Entry point verification and execution
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# End of main_nc5.py - Aurora RPG Client Phase 5 Main Application - FIXED VERSION
