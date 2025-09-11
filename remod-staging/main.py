#!/usr/bin/env python3
"""
DevName RPG Client - Main Entry Point (Phase 6: Orchestrator Integration)

Updated to use the new orchestrator architecture from remodularization
Phase 6 completes the transition from nci.py to orch.py coordination
"""

import os
import sys
import asyncio
from typing import Dict, Any, Optional

# Import new orchestrator (Phase 6 update)
try:
    from orch import run_orchestrated_application, create_orchestrator
except ImportError:
    print("Error: New orchestrator modules not found. Ensure orch.py is present.")
    sys.exit(1)

# Configuration and logging imports
try:
    from debug_logger import DebugLogger
except ImportError:
    # Fallback if debug logger not available
    class DebugLogger:
        def debug(self, message, category="MAIN"):
            print(f"[{category}] {message}")
        def error(self, message, category="MAIN"):
            print(f"[{category}] ERROR: {message}")

# Constants for configuration
PROMPT_DIRECTORY = "prompts"
REQUIRED_PROMPTS = ["critrules.txt", "companion.txt", "lowrules.txt"]
CONFIG_DEFAULTS = {
    "max_memory_tokens": 30000,
    "color_theme": "classic",
    "auto_save_conversation": False,
    "ui_refresh_rate": 30,
    "ui_auto_refresh": True,
    "max_input_width": 100,
    "submission_mode": "smart"
}

def load_prompts() -> Dict[str, str]:
    """
    Load system prompts from files
    Maintains existing prompt loading logic for compatibility
    """
    prompts = {}
    
    if not os.path.exists(PROMPT_DIRECTORY):
        print(f"Warning: Prompts directory '{PROMPT_DIRECTORY}' not found")
        return prompts
    
    for prompt_file in REQUIRED_PROMPTS:
        prompt_path = os.path.join(PROMPT_DIRECTORY, prompt_file)
        
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    # Use filename without extension as key
                    prompt_name = os.path.splitext(prompt_file)[0]
                    prompts[prompt_name] = content
                    print(f"Loaded prompt: {prompt_name} ({len(content)} chars)")
                else:
                    print(f"Warning: Empty prompt file: {prompt_file}")
                    
            except Exception as e:
                print(f"Error loading prompt {prompt_file}: {e}")
        else:
            print(f"Warning: Prompt file not found: {prompt_file}")
    
    if not prompts:
        print("Warning: No prompts loaded. Application will use minimal defaults.")
    
    return prompts

def validate_prompts(prompts: Dict[str, str]) -> bool:
    """Validate that essential prompts are present"""
    if not prompts.get("critrules"):
        print("Critical: 'critrules.txt' prompt is required for proper operation")
        return False
    
    return True

def create_configuration(prompts: Dict[str, str]) -> Dict[str, Any]:
    """
    Create application configuration for orchestrator
    Combines loaded prompts with system defaults
    """
    config = CONFIG_DEFAULTS.copy()
    
    # Add prompts to configuration
    config["prompts"] = prompts
    
    # Environment-based overrides
    if "DEV_MODE" in os.environ:
        config["auto_save_conversation"] = True
        config["ui_refresh_rate"] = 60  # Higher refresh rate for development
        print("Development mode enabled")
    
    # Terminal-specific adjustments
    try:
        terminal_width = os.get_terminal_size().columns
        if terminal_width < 100:
            config["max_input_width"] = max(20, terminal_width - 20)
    except:
        pass  # Use default if terminal size unavailable
    
    return config

def setup_debug_logging() -> Optional[DebugLogger]:
    """Setup debug logging for application monitoring"""
    try:
        debug_logger = DebugLogger()
        debug_logger.debug("Debug logging initialized", "MAIN")
        return debug_logger
    except Exception as e:
        print(f"Warning: Debug logging setup failed: {e}")
        return None

def validate_system_requirements() -> bool:
    """Validate system requirements for application"""
    try:
        import curses
        import json
        import time
        import threading
        import asyncio
        
        # Test curses capability
        if not curses.has_colors():
            print("Warning: Terminal does not support colors")
        
        # Test async capability
        loop = asyncio.new_event_loop()
        loop.close()
        
        return True
        
    except ImportError as e:
        print(f"Critical: Missing required dependency: {e}")
        return False
    except Exception as e:
        print(f"System validation error: {e}")
        return False

def show_startup_banner():
    """Display application startup banner"""
    print("DevName RPG Client - Remodularized Architecture")
    print("=" * 50)
    print("Phase 6: Complete integration with orchestrator")
    print("• Modular architecture with clear separation")
    print("• Centralized semantic analysis")
    print("• Enhanced UI with consolidated utilities")
    print("• Background processing coordination")
    print("=" * 50)

def main() -> int:
    """
    Main application entry point - Phase 6 Integration
    
    Updated to use new orchestrator architecture instead of direct nci.py
    """
    try:
        # Display startup information
        show_startup_banner()
        
        # Validate system requirements
        if not validate_system_requirements():
            print("System requirements not met. Exiting.")
            return 1
        
        # Setup debug logging
        debug_logger = setup_debug_logging()
        
        # Load and validate prompts
        print("Loading system prompts...")
        prompts = load_prompts()
        
        if not validate_prompts(prompts):
            print("Essential prompts missing. Exiting.")
            return 1
        
        # Create configuration
        config = create_configuration(prompts)
        
        if debug_logger:
            debug_logger.debug(f"Configuration created with {len(prompts)} prompts", "MAIN")
            debug_logger.debug(f"Max memory tokens: {config['max_memory_tokens']}", "MAIN")
            debug_logger.debug(f"UI theme: {config['color_theme']}", "MAIN")
        
        # Run application through orchestrator (Phase 6 change)
        print("Starting orchestrated application...")
        
        return asyncio.run(run_orchestrated_application(config, debug_logger))
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Critical application error: {e}")
        if "debug_logger" in locals() and debug_logger:
            debug_logger.error(f"Critical application error: {e}", "MAIN")
        return 1

def test_orchestrator_integration() -> bool:
    """
    Test orchestrator integration without running full UI
    Phase 6 testing function
    """
    try:
        print("Testing orchestrator integration...")
        
        # Create minimal test configuration
        test_config = {
            "prompts": {"critrules": "Test system prompt"},
            "max_memory_tokens": 1000,
            "color_theme": "classic"
        }
        
        # Test orchestrator creation
        async def test_init():
            try:
                orchestrator = await create_orchestrator(test_config)
                status = orchestrator.get_system_status()
                await orchestrator.shutdown_system()
                return status["orchestrator_state"]["running"]
            except Exception as e:
                print(f"Test failed: {e}")
                return False
        
        result = asyncio.run(test_init())
        
        if result:
            print("✓ Orchestrator integration test passed")
        else:
            print("✗ Orchestrator integration test failed")
        
        return result
        
    except Exception as e:
        print(f"Integration test error: {e}")
        return False

def show_help():
    """Display help information"""
    help_text = """
DevName RPG Client - Phase 6 Integrated Version

Usage:
    python main.py              - Run the application
    python main.py --test       - Test orchestrator integration
    python main.py --help       - Show this help

Configuration:
    Prompts loaded from 'prompts/' directory:
    • critrules.txt (required) - Core system rules
    • companion.txt (optional) - Companion guidelines  
    • lowrules.txt (optional) - Additional rules

Environment Variables:
    DEV_MODE=1 - Enable development mode with enhanced debugging

Phase 6 Changes:
    • Updated to use orch.py orchestrator instead of nci.py
    • Centralized initialization and coordination
    • Enhanced error handling and testing
    • Modular architecture with clear dependencies
    
System Requirements:
    • Python 3.8+
    • curses library (usually included)
    • No external dependencies required
    """
    print(help_text)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_help()
            sys.exit(0)
        elif sys.argv[1] == "--test":
            result = test_orchestrator_integration()
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)
