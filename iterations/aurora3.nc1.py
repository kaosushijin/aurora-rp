import asyncio
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import httpx
import sys
from colorama import init, Fore, Style
import textwrap
import shutil
import argparse

# ------------------ Initialization ------------------ #
init(autoreset=True)

# Parse command line arguments early
parser = argparse.ArgumentParser(description="Aurora RPG Client with Story Momentum Engine")
parser.add_argument("--bypasscurses", action="store_true",
                   help="Use simple terminal interface instead of ncurses")
parser.add_argument("--debug", action="store_true",
                   help="Enable debug logging to debug.log file")
parser.add_argument("--colorscheme", default="midnight_aurora",
                   choices=["midnight_aurora", "forest_whisper", "dracula_aurora"],
                   help="Initial color scheme (can be changed with /color command)")

args = parser.parse_args()

# ------------------ Configuration ------------------ #
MCP_URL = "http://127.0.0.1:3456/chat"
MODEL = "qwen2.5:14b-instruct-q4_k_m"
SAVE_FILE = Path("memory.json")
TIMEOUT = 300.0  # seconds

# Context window and token allocation
CONTEXT_WINDOW = 32000
SYSTEM_PROMPT_TOKENS = 5000  # Token budget for all system prompts combined
MOMENTUM_ANALYSIS_TOKENS = 6000  # Token budget for momentum analysis

REMAINING_TOKENS = CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS - MOMENTUM_ANALYSIS_TOKENS
MEMORY_FRACTION = 0.7
MAX_MEMORY_TOKENS = int(REMAINING_TOKENS * MEMORY_FRACTION)  # ~14,700 tokens
MAX_USER_INPUT_TOKENS = int(REMAINING_TOKENS * (1 - MEMORY_FRACTION))  # ~6,300 tokens

# Memory condensation strategies by content type
CONDENSATION_STRATEGIES = {
    "story_critical": {
        "threshold": 100,  # Start condensing after 100 messages
        "preservation_ratio": 0.8,  # Keep 80% of content
        "instruction": (
            "Preserve all major plot developments, character deaths, world-changing events, "
            "key player decisions, and their consequences. Use decisive language highlighting "
            "the significance of events. Compress dialogue while maintaining essential meaning."
        )
    },
    "character_focused": {
        "threshold": 80,
        "preservation_ratio": 0.7,  # Keep 70% of content
        "instruction": (
            "Preserve relationship changes, trust/betrayal moments, character motivations, "
            "personality reveals, Aurora's development, and NPC traits. Emphasize emotional "
            "weight and relationship dynamics. Condense descriptions while keeping character essence."
        )
    },
    "world_building": {
        "threshold": 60,
        "preservation_ratio": 0.6,  # Keep 60% of content
        "instruction": (
            "Preserve new locations, lore revelations, cultural information, political changes, "
            "economic systems, magical discoveries, and historical context. Provide rich "
            "foundational details. Compress atmospheric descriptions while keeping key world facts."
        )
    },
    "standard": {
        "threshold": 40,
        "preservation_ratio": 0.4,  # Keep 40% of content
        "instruction": (
            "Preserve player actions and immediate consequences for continuity. Compress "
            "everything else aggressively while maintaining basic story flow."
        )
    }
}

# Color scheme definitions
MIDNIGHT_AURORA = {
    "background": "#1e1e2e",
    "surface": "#313244", 
    "text_primary": "#cdd6f4",
    "text_secondary": "#a6adc8",
    "user_input": "#89dceb",
    "user_prompt": "#74c7ec",
    "assistant_primary": "#a6e3a1",
    "assistant_secondary": "#94e2d5",
    "system_info": "#b4befe",
    "system_success": "#a6e3a1",
    "debug_info": "#f9e2af",
    "debug_warning": "#fab387",
    "debug_error": "#f38ba8",
    "debug_critical": "#f38ba8",
}

FOREST_WHISPER = {
    "background": "#1a1b26",
    "surface": "#24283b",
    "text_primary": "#c0caf5",
    "text_secondary": "#9aa5ce",
    "user_input": "#7dcfff",
    "user_prompt": "#2ac3de",
    "assistant_primary": "#9ece6a",
    "assistant_secondary": "#73daca",
    "system_info": "#bb9af7",
    "system_success": "#9ece6a",
    "debug_info": "#e0af68",
    "debug_warning": "#ff9e64",
    "debug_error": "#f7768e",
    "debug_critical": "#ff757f",
}

DRACULA_AURORA = {
    "background": "#282a36",
    "surface": "#44475a",
    "text_primary": "#f8f8f2",
    "text_secondary": "#6272a4",
    "user_input": "#8be9fd",
    "user_prompt": "#50fa7b",
    "assistant_primary": "#bd93f9",
    "assistant_secondary": "#ff79c6",
    "system_info": "#bd93f9",
    "system_success": "#50fa7b",
    "debug_info": "#f1fa8c",
    "debug_warning": "#ffb86c",
    "debug_error": "#ff5555",
    "debug_critical": "#ff5555",
}

COLOR_SCHEMES = {
    "midnight_aurora": MIDNIGHT_AURORA,
    "forest_whisper": FOREST_WHISPER,
    "dracula_aurora": DRACULA_AURORA
}

COLOR_CYCLE_ORDER = ["midnight_aurora", "forest_whisper", "dracula_aurora"]

# Command definitions
COMMANDS = {
    "/debug": "Switch to debug context (Esc to return)",
    "/search <term>": "Search chat history in separate context", 
    "/color": "Cycle through color schemes (midnight→forest→dracula)",
    "/showsme": "Toggle Story Momentum Engine status display in status bar",
    "/quit": "Exit the application", 
    "/save <filename>": "Save conversation (optional: --chat, --debug, --both)",
    "/clear": "Clear current context history",
    "/help": "Show command help"
}

# Context instruction messages
CONTEXT_HELP = {
    "debug": "Debug context active. Type /debug or press Esc to return to chat.",
    "search": "Search results displayed. Press Esc to return to chat."
}

# SME Status Display
SME_STATUS_FORMAT = "SME: {pressure_name}({pressure:.2f}) | Villain: {antagonist_name}"

# ------------------ Prompt Files ------------------ #
PROMPT_TOP_FILE = Path("critrules.prompt")
PROMPT_MID_FILE = Path("companion.prompt")
PROMPT_LOW_FILE = Path("lowrules.prompt")

def load_prompt(file_path: Path, debug_logger=None) -> str:
    """Load prompt file with graceful handling of missing files."""
    if not file_path.exists():
        warning_msg = f"Prompt file not found: {file_path}"
        if debug_logger:
            debug_logger.error(warning_msg)
        print(Fore.YELLOW + f"[Warning] {warning_msg}")
        print(Fore.YELLOW + f"[Warning] Using empty prompt for {file_path.stem}")
        return ""
    
    content = file_path.read_text(encoding="utf-8").strip()
    if debug_logger:
        debug_logger.system(f"Loaded prompt file {file_path}: {len(content)} characters")
    return content

# ------------------ Debug Logger System ------------------ #
class DebugLogger:
    """File-based debug logging system with per-run reinitialization"""
    
    def __init__(self, debug_file_enabled=False, log_file_path="debug.log"):
        self.debug_file_enabled = debug_file_enabled
        self.log_file_path = log_file_path
        self.log_file = None
        
        if debug_file_enabled:
            self._init_log_file()
    
    def _init_log_file(self):
        """Initialize debug log file, overwriting any previous run"""
        try:
            # Overwrite mode ('w') ensures fresh log for each run
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"Aurora RPG Client Debug Log - Session Started: {timestamp}\n")
            self.log_file.write("=" * 60 + "\n")
            self.log_file.write("Log reinitialized for this session - previous debug data cleared\n\n")
            self.log_file.flush()
        except IOError as e:
            print(f"Warning: Could not create debug log file: {e}")
            self.debug_file_enabled = False
    
    def system(self, message):
        """System-level debug messages (startup, configuration)"""
        if self.debug_file_enabled:
            self._log_to_file("SYSTEM", message, severity="info")
    
    def memory(self, message):
        """Memory management debug messages"""
        if self.debug_file_enabled:
            self._log_to_file("MEMORY", message, severity="warning")
    
    def momentum(self, message):
        """Story momentum engine debug messages"""
        if self.debug_file_enabled:
            self._log_to_file("MOMENTUM", message, severity="info")
    
    def error(self, message):
        """Error-level debug messages"""
        if self.debug_file_enabled:
            self._log_to_file("ERROR", message, severity="error")
    
    def network(self, message):
        """Network/MCP communication debug messages"""
        if self.debug_file_enabled:
            self._log_to_file("NETWORK", message, severity="info")
    
    def _log_to_file(self, category, message, severity):
        """Write debug message to log file"""
        if not self.debug_file_enabled or not self.log_file:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        formatted_msg = f"[{timestamp}] {severity.upper():<7} {category:<8}: {message}\n"
        
        try:
            self.log_file.write(formatted_msg)
            self.log_file.flush()  # Ensure immediate write
        except IOError:
            # Silently disable logging if file becomes unavailable
            self.debug_file_enabled = False
    
    def close(self):
        """Close debug log file gracefully"""
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"\nDebug session ended: {timestamp}\n")
            self.log_file.close()
            self.log_file = None

# ------------------ Color Manager (COMPLETELY FIXED) ------------------ #
class ColorManager:
    """Manages multiple color schemes with dynamic switching - ALWAYS returns strings for simple interface"""
    
    def __init__(self, initial_scheme="midnight_aurora", interface_type="simple"):
        self.schemes = COLOR_SCHEMES
        self.current_scheme_name = initial_scheme
        self.current_scheme = self.schemes[initial_scheme]
        # FORCE simple interface for Phase 1 - this prevents the int/string bug completely
        self.interface_type = "simple"
        self.color_pairs = {}  # For future ncurses implementation
    
    def switch_scheme(self, scheme_name):
        """Switch to a different color scheme dynamically"""
        if scheme_name in self.schemes:
            self.current_scheme_name = scheme_name
            self.current_scheme = self.schemes[scheme_name]
            
            # Future ncurses support
            if self.interface_type == "curses":
                self._reinit_curses_colors()
            
            return True
        return False
    
    def get_available_schemes(self):
        """Return list of available color schemes"""
        return list(self.schemes.keys())
    
    def get_current_scheme_name(self):
        """Return current scheme name for status display"""
        return self.current_scheme_name.upper()
    
    def get_color(self, color_name):
        """Get color for current interface type - ALWAYS returns string for Phase 1"""
        # ALWAYS return ANSI color string for Phase 1 to prevent int/string concatenation error
        return self._hex_to_ansi(self.current_scheme.get(color_name, "#ffffff"))
    
    def _hex_to_ansi(self, hex_color):
        """Convert hex color to closest ANSI color for simple terminal"""
        # Comprehensive mapping of hex colors to ANSI colors
        color_map = {
            # White variations
            "#ffffff": Fore.WHITE, "#f8f8f2": Fore.WHITE, "#cdd6f4": Fore.WHITE,
            "#a6adc8": Fore.WHITE, "#c0caf5": Fore.WHITE, "#9aa5ce": Fore.WHITE,
            
            # Red variations
            "#ff0000": Fore.RED, "#ff5555": Fore.RED, "#f38ba8": Fore.RED,
            "#f7768e": Fore.RED, "#ff757f": Fore.RED,
            
            # Green variations
            "#00ff00": Fore.GREEN, "#50fa7b": Fore.GREEN, "#a6e3a1": Fore.GREEN,
            "#9ece6a": Fore.GREEN,
            
            # Blue variations
            "#0000ff": Fore.BLUE, "#6272a4": Fore.BLUE,
            
            # Cyan variations
            "#8be9fd": Fore.CYAN, "#89dceb": Fore.CYAN, "#7dcfff": Fore.CYAN,
            "#2ac3de": Fore.CYAN, "#74c7ec": Fore.CYAN, "#94e2d5": Fore.CYAN,
            "#73daca": Fore.CYAN,
            
            # Yellow variations
            "#ffff00": Fore.YELLOW, "#f1fa8c": Fore.YELLOW, "#f9e2af": Fore.YELLOW,
            "#e0af68": Fore.YELLOW, "#ff9e64": Fore.YELLOW, "#ffb86c": Fore.YELLOW,
            "#fab387": Fore.YELLOW,
            
            # Magenta/Purple variations
            "#ff00ff": Fore.MAGENTA, "#bd93f9": Fore.MAGENTA, "#ff79c6": Fore.MAGENTA,
            "#bb9af7": Fore.MAGENTA, "#b4befe": Fore.MAGENTA,
            
            # Background colors (typically map to white/default for text)
            "#1e1e2e": Fore.WHITE, "#313244": Fore.WHITE, "#1a1b26": Fore.WHITE,
            "#24283b": Fore.WHITE, "#282a36": Fore.WHITE, "#44475a": Fore.WHITE,
        }
        
        # Return the mapped color or default to white
        return color_map.get(hex_color, Fore.WHITE)
    
    # Placeholder methods for future ncurses implementation
    def _reinit_curses_colors(self):
        """Reinitialize color pairs when scheme changes (placeholder for Phase 4)"""
        pass

# ------------------ Context Manager ------------------ #
class ContextManager:
    """Enhanced context manager supporting three contexts with input preservation"""
    
    def __init__(self):
        self.chat_history = []
        self.debug_history = [] 
        self.search_results = []
        self.current_search_term = ""
        self.current_context = "chat"  # "chat", "debug", "search"
        self.max_history = 1000
        self.sme_status_visible = False  # Toggle for SME status display
    
    def add_chat_message(self, message, message_type):
        """Add message to chat history"""
        self.chat_history.append({
            'content': message,
            'type': message_type,  # 'user', 'assistant', 'system'
            'timestamp': datetime.now()
        })
        self._trim_history('chat')
    
    def add_debug_message(self, message, severity):
        """Add message to debug history"""
        self.debug_history.append({
            'content': message,
            'severity': severity,  # 'info', 'warning', 'error'
            'timestamp': datetime.now()
        })
        self._trim_history('debug')
    
    def switch_context(self, new_context):
        """Switch between contexts"""
        old_context = self.current_context
        self.current_context = new_context
        return self.get_current_history()
    
    def get_current_history(self):
        """Get history for current context"""
        if self.current_context == "chat":
            return self.chat_history
        elif self.current_context == "debug":
            return self.debug_history
        else:  # search
            return self.search_results
    
    def search_chat_history(self, search_term):
        """Search only chat history and return results with context"""
        results = []
        for i, entry in enumerate(self.chat_history):
            if search_term.lower() in entry['content'].lower():
                results.append({
                    'index': i,
                    'content': entry['content'],
                    'type': entry['type'],
                    'timestamp': entry['timestamp'],
                    'source_context': 'CHAT'
                })
        return results
    
    def set_search_results(self, search_term, results):
        """Set current search term and results"""
        self.current_search_term = search_term
        self.search_results = results
    
    def get_message_count(self):
        """Get message count for current context"""
        if self.current_context == "chat":
            return len(self.chat_history)
        elif self.current_context == "debug":
            return len(self.debug_history)
        else:  # search
            return len(self.search_results)
    
    def toggle_sme_status(self):
        """Toggle SME status visibility"""
        self.sme_status_visible = not self.sme_status_visible
        return self.sme_status_visible
    
    def clear_current_context(self):
        """Clear current context history"""
        if self.current_context == "chat":
            self.chat_history.clear()
        elif self.current_context == "debug":
            self.debug_history.clear()
        else:  # search
            self.search_results.clear()
            self.current_search_term = ""
    
    def _trim_history(self, context_type):
        """Keep history within limits"""
        if context_type == "chat" and len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        elif context_type == "debug" and len(self.debug_history) > self.max_history:
            self.debug_history = self.debug_history[-self.max_history:]

# ------------------ Display Manager (COMPLETELY FIXED) ------------------ #
class DisplayManager:
    """Abstraction layer with file-based debug logging and clean user interface"""
    
    def __init__(self, interface_type="simple", debug_logger=None):
        self.interface_type = "simple"  # FORCE simple interface for Phase 1
        self.debug_logger = debug_logger
        # ColorManager ALWAYS uses simple interface for Phase 1
        self.color_manager = ColorManager(args.colorscheme, "simple")
        self.context_manager = ContextManager()
        self.preserved_input = ""  # Preserve input when switching contexts
        self.quit_dialog_active = False
        
        # Debug logger receives all debug messages for file logging
        if debug_logger:
            debug_logger.system("Display manager initialized with simple interface")
    
    def show_user_input(self, text):
        """Display user input (no debug output to console)"""
        self.context_manager.add_chat_message(text, 'user')
        
        if self.debug_logger:
            self.debug_logger.system(f"User input: {text[:50]}...")
        
        # ALWAYS use simple interface for Phase 1
        color = self.color_manager.get_color('user_input')
        print(color + f"> {text}" + Style.RESET_ALL)
    
    def show_assistant_response(self, text):
        """Display assistant response (no debug output to console)"""
        self.context_manager.add_chat_message(text, 'assistant')
        
        if self.debug_logger:
            response_length = len(text)
            self.debug_logger.system(f"Assistant response: {response_length} characters")
        
        # ALWAYS use simple interface for Phase 1
        self._simple_show_response(text)
    
    def show_system_message(self, text, log_only=False):
        """Show system messages with clean interface - log_only for internal operations"""
        if log_only:
            # Only log to debug file, don't show to user
            if self.debug_logger:
                self.debug_logger.system(text)
            return
        
        # Show to user in current interface
        self.context_manager.add_chat_message(text, 'system')
        
        if self.debug_logger:
            self.debug_logger.system(f"System message displayed: {text}")
        
        # ALWAYS use simple interface for Phase 1
        color = self.color_manager.get_color('system_info')
        print(color + f"[System] {text}" + Style.RESET_ALL)
    
    def handle_command(self, command):
        """Handle special commands with enhanced cycling and search"""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        if cmd == "/debug":
            return self._toggle_debug_context()
        elif cmd == "/color":
            return self._cycle_color_scheme()
        elif cmd.startswith("/search") and len(parts) > 1:
            search_term = " ".join(parts[1:])
            return self._open_search_context(search_term)
        elif cmd == "/showsme":
            return self._toggle_sme_status()
        elif cmd == "/help":
            return self._show_help()
        elif cmd in ["/quit", "/exit"]:
            return {"action": "quit"}
        elif cmd == "/clear":
            return self._clear_current_context()
        elif cmd.startswith("/save"):
            filename = parts[1] if len(parts) > 1 else "conversation.txt"
            return self._save_conversation(filename)
        else:
            return {"action": "invalid_command", "message": f"Unknown command: {cmd}"}
    
    def _toggle_debug_context(self):
        """Switch to debug context with instruction message"""
        if self.context_manager.current_context == "chat":
            self.context_manager.switch_context("debug")
            
            # Show instructional message in debug context
            instruction = CONTEXT_HELP["debug"]
            self.context_manager.add_debug_message(instruction, 'info')
            
            return {"action": "context_switched", "context": "debug", "message": "Switched to debug context"}
        else:
            # Return to chat from debug - switch back to chat context
            self.context_manager.switch_context("chat")
            return {"action": "context_switched", "context": "chat", "message": "Returned to chat context"}
    
    def _cycle_color_scheme(self):
        """Cycle through color schemes in predetermined order"""
        current_scheme = self.color_manager.current_scheme_name
        current_index = COLOR_CYCLE_ORDER.index(current_scheme)
        next_index = (current_index + 1) % len(COLOR_CYCLE_ORDER)
        next_scheme = COLOR_CYCLE_ORDER[next_index]
        
        self.color_manager.switch_scheme(next_scheme)
        
        return {
            "action": "color_cycled",
            "scheme": next_scheme,
            "message": f"Color scheme: {next_scheme.replace('_', ' ').title()}"
        }
    
    def _toggle_sme_status(self):
        """Toggle SME status display in status bar"""
        visible = self.context_manager.toggle_sme_status()
        status = "enabled" if visible else "disabled"
        return {
            "action": "sme_toggled",
            "visible": visible,
            "message": f"SME status display {status}"
        }
    
    def _open_search_context(self, search_term):
        """Open search context with results"""
        # Search only chat history
        results = self.context_manager.search_chat_history(search_term)
        
        self.context_manager.switch_context("search")
        self.context_manager.set_search_results(search_term, results)
        
        return {
            "action": "search_opened",
            "term": search_term,
            "results_count": len(results),
            "message": f"Search results for '{search_term}': {len(results)} matches found"
        }
    
    def _show_help(self):
        """Show command help - FIXED to include message key"""
        help_text = "Available commands:\n"
        for cmd, desc in COMMANDS.items():
            help_text += f"  {cmd}: {desc}\n"
        
        self.show_system_message(help_text.strip())
        return {"action": "help_shown", "message": "Command help displayed"}
    
    def _clear_current_context(self):
        """Clear current context history"""
        self.context_manager.clear_current_context()
        return {"action": "context_cleared", "context": self.context_manager.current_context, "message": f"Cleared {self.context_manager.current_context} context"}
    
    def _save_conversation(self, filename):
        """Save conversation history (placeholder for later phases)"""
        return {"action": "save_requested", "filename": filename, "message": "Save functionality coming in later phase"}
    
    def get_user_input(self):
        """Get user input with multi-line support and command handling"""
        # ALWAYS use simple interface for Phase 1
        return self._simple_get_input()
    
    def _simple_get_input(self):
        """Simple terminal input collection"""
        user_lines = []
        color = self.color_manager.get_color('user_prompt')
        print(color + "> ", end="", flush=True)
        
        while True:
            try:
                line = input()
                if line.lower().strip() == 'quit':
                    return "/quit"
                
                if line == "" and user_lines:  # Double enter to submit
                    break
                
                user_lines.append(line)
                print(color + "> ", end="", flush=True)
                
            except KeyboardInterrupt:
                return "/quit"
        
        return "\n".join(user_lines).strip()
    
    def _simple_show_response(self, text):
        """Display assistant response in simple terminal mode"""
        color = self.color_manager.get_color('assistant_primary')
        print_wrapped(text, color)
    
    def generate_status_bar(self, momentum_state=None):
        """Generate status bar text"""
        context_name = self.context_manager.current_context.upper()
        theme_name = self.color_manager.get_current_scheme_name()
        
        base_status = f"Window: [{context_name}] | Theme: [{theme_name}] | Commands: /help"
        
        # Add SME status if enabled and available
        if self.context_manager.sme_status_visible and momentum_state:
            antagonist = momentum_state.get("antagonist", {})
            pressure = momentum_state.get("narrative_pressure", 0.0)
            pressure_name = get_pressure_name(pressure)
            antagonist_name = antagonist.get("name", "Unknown")
            
            sme_status = f" | SME: {pressure_name.title()}({pressure:.2f}) | Villain: {antagonist_name}"
            base_status += sme_status
        
        return base_status

# ------------------ Utility Functions ------------------ #
def get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def print_wrapped(text: str, color=Fore.GREEN, indent: int = 0):
    width = get_terminal_width() - indent
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=' ' * indent)
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        lines = para.splitlines()
        wrapped_para = "\n".join(wrapper.fill(line) for line in lines)
        print(color + wrapped_para)
        if i < len(paragraphs) - 1:
            print("")

# ------------------ Token Estimation ------------------ #
estimate_tokens = lambda text: max(1, len(text) // 4)

# ------------------ Input Validation ------------------ #
def validate_user_input_length(user_input: str) -> tuple[bool, str, str]:
    """
    Validate user input length and provide helpful feedback if too long.
    Returns (is_valid, warning_message, preserved_input)
    """
    input_tokens = estimate_tokens(user_input)

    if input_tokens <= MAX_USER_INPUT_TOKENS:
        return True, "", ""

    char_count = len(user_input)
    max_chars = MAX_USER_INPUT_TOKENS * 4

    warning = (f"Input too long ({input_tokens:,} tokens, {char_count:,} chars). "
              f"Maximum: {MAX_USER_INPUT_TOKENS:,} tokens ({max_chars:,} chars). "
              f"Please shorten your input - it has been preserved for editing.")

    return False, warning, user_input  # Preserve the input for user editing

# ------------------ Configuration Validation ------------------ #
def validate_token_allocation(debug_logger=None):
    """Ensure token allocation doesn't exceed context window"""
    total_allocated = (SYSTEM_PROMPT_TOKENS + MOMENTUM_ANALYSIS_TOKENS +
                      MAX_MEMORY_TOKENS + MAX_USER_INPUT_TOKENS)

    if total_allocated > CONTEXT_WINDOW:
        raise ValueError(f"Token allocation ({total_allocated:,}) exceeds context window ({CONTEXT_WINDOW:,})")

    if debug_logger:
        utilization = (total_allocated / CONTEXT_WINDOW) * 100
        debug_logger.system(f"Token allocation validated: {utilization:.1f}% utilization")
        debug_logger.system(f"System prompts: {SYSTEM_PROMPT_TOKENS:,} tokens")
        debug_logger.system(f"Momentum analysis: {MOMENTUM_ANALYSIS_TOKENS:,} tokens")
        debug_logger.system(f"Memory: {MAX_MEMORY_TOKENS:,} tokens")
        debug_logger.system(f"User input: {MAX_USER_INPUT_TOKENS:,} tokens")
        debug_logger.system(f"Total: {total_allocated:,} tokens")
        debug_logger.system(f"Safety margin: {CONTEXT_WINDOW - total_allocated:,} tokens")

    return True

# ------------------ Memory Management ------------------ #
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_memory(debug_logger=None):
    if not SAVE_FILE.exists():
        if debug_logger:
            debug_logger.system("No existing memory file found")
        return []
    try:
        with open(SAVE_FILE, 'r', encoding='utf-8') as f:
            memories = json.load(f)
            if debug_logger:
                debug_logger.system(f"Loaded {len(memories)} memories from {SAVE_FILE}")
            return memories
    except (json.JSONDecodeError, FileNotFoundError) as e:
        if debug_logger:
            debug_logger.error(f"Failed to load memory file: {e}")
        return []

def save_memory(memories, debug_logger=None):
    try:
        with open(SAVE_FILE, 'w', encoding='utf-8') as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)
            if debug_logger:
                debug_logger.system(f"Saved {len(memories)} memories to {SAVE_FILE}")
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Failed to save memory file: {e}")

def add_memory(memories, role, content, debug_logger=None):
    memory = {
        "id": str(uuid4()),
        "role": role,
        "content": content,
        "timestamp": now_iso()
    }
    memories.append(memory)
    save_memory(memories, debug_logger)
    if debug_logger:
        debug_logger.memory(f"Added {role} memory: {content[:50]}...")

# ------------------ MCP Communication ------------------ #
async def call_mcp(messages, debug_logger=None, max_retries=3):
    """Call MCP with automatic retry logic for robustness."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    
    if debug_logger:
        debug_logger.network(f"MCP call with {len(messages)} messages")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(MCP_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                if debug_logger:
                    debug_logger.network(f"MCP response: {len(content)} characters")
                
                return content
        except (httpx.TimeoutException, httpx.RequestError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                if debug_logger:
                    debug_logger.network(f"MCP call failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                if debug_logger:
                    debug_logger.error(f"MCP call failed after {max_retries} attempts: {e}")
                raise e

# ------------------ Antagonist System Functions ------------------ #
async def generate_antagonist(memories, debug_logger=None, max_attempts=3):
    """Generate a high-quality antagonist based on story context."""
    
    if debug_logger:
        debug_logger.momentum("Starting antagonist generation")
    
    # Prepare story context from recent memories
    story_context = "\n".join([
        f"{mem['role']}: {mem['content'][:300]}"  # Truncate for token efficiency
        for mem in memories[-15:]  # Recent context
        if mem.get("role") in ["user", "assistant"]
    ])
    
    antagonist_prompt = f"""
You are creating an antagonist for an ongoing RPG story. Based on the story context, 
generate a compelling antagonist that fits naturally into the narrative.

Recent Story Context:
{story_context}

Create an antagonist with:
1. Name: A fitting name for the setting
2. Motivation: Clear, understandable goals that conflict with the player
3. Commitment Level: Start at "testing" (will escalate based on story events)
4. Resources: What power, influence, or assets do they have?
5. Personality: Key traits that drive their behavior
6. Background: Brief history that explains their motivation

Provide a JSON response with these fields:
{{
    "name": "Antagonist Name",
    "motivation": "Clear motivation that opposes player goals",
    "commitment_level": "testing",
    "resources_available": ["resource1", "resource2", "resource3"],
    "resources_lost": [],
    "personality_traits": ["trait1", "trait2", "trait3"],
    "background": "Brief background story",
    "threat_level": "moderate"
}}
"""
    
    for attempt in range(max_attempts):
        try:
            response = await call_mcp([{"role": "system", "content": antagonist_prompt}], debug_logger)
            antagonist_data = json.loads(response)
            
            # Validate required fields
            required_fields = ["name", "motivation", "commitment_level"]
            if all(field in antagonist_data for field in required_fields):
                # Ensure lists exist
                antagonist_data.setdefault("resources_available", [])
                antagonist_data.setdefault("resources_lost", [])
                antagonist_data.setdefault("personality_traits", [])
                
                if debug_logger:
                    debug_logger.momentum(f"Generated antagonist: {antagonist_data['name']}")
                
                return antagonist_data
                
        except (json.JSONDecodeError, KeyError) as e:
            if debug_logger:
                debug_logger.momentum(f"Antagonist generation attempt {attempt + 1} failed: {e}")
    
    # Fallback antagonist if generation fails
    fallback = {
        "name": "The Shadow",
        "motivation": "seeks to disrupt the player's journey",
        "commitment_level": "testing",
        "resources_available": ["stealth", "cunning", "local knowledge"],
        "resources_lost": [],
        "personality_traits": ["mysterious", "patient", "observant"],
        "background": "A mysterious figure who opposes those who disturb the natural order",
        "threat_level": "moderate"
    }
    
    if debug_logger:
        debug_logger.momentum("Using fallback antagonist due to generation failures")
    
    return fallback

def get_pressure_name(pressure_level):
    """Convert pressure level to named range."""
    if pressure_level < 0.1:
        return "low"
    elif pressure_level < 0.3:
        return "building" 
    elif pressure_level < 0.6:
        return "critical"
    else:
        return "explosive"

# ------------------ Simplified Application Initialization (COMPLETELY FIXED) ------------------ #
def initialize_application():
    """Initialize application with simple interface only for Phase 1"""
    
    # Initialize debug logger (file-based, no console output)
    debug_logger = DebugLogger(args.debug, "debug.log") if args.debug else None
    
    if debug_logger:
        debug_logger.system("Aurora RPG Client starting...")
        debug_logger.system(f"Arguments: debug={args.debug}, bypasscurses={args.bypasscurses}, colorscheme={args.colorscheme}")
        debug_logger.system("Phase 1: Using simple interface only")
    
    # ALWAYS use simple interface for Phase 1 - no ncurses detection needed
    interface_type = "simple"
    
    # Initialize display manager with simple interface
    display_manager = DisplayManager(interface_type, debug_logger)
    
    # Show clean startup message
    print("Aurora RPG Client ready. Type /help for commands.")
    
    return display_manager, debug_logger

# ------------------ FIXED Main Function for Phase 1 ------------------ #
async def main():
    """Main application entry point - Phase 1 implementation with FIXED command handling"""
    try:
        # Initialize application with SIMPLE interface only
        display_manager, debug_logger = initialize_application()
        
        # Validate configuration
        validate_token_allocation(debug_logger)
        
        # Load conversation memory
        memories = load_memory(debug_logger)
        display_manager.show_system_message(f"Loaded {len(memories)} memories from previous sessions", log_only=True)
        
        # Display session info
        print(Fore.GREEN + "\n" + "="*60)
        print(Fore.GREEN + "Aurora RPG Client - Phase 1 (Debug System & Display Abstraction)")
        print(Fore.GREEN + "="*60)
        
        # Show status bar
        status = display_manager.generate_status_bar()
        print(Fore.CYAN + status)
        
        print(Fore.WHITE + "\nType your message (press Enter twice to send, '/quit' to exit):")
        print(Fore.WHITE + "New commands: /debug, /color, /showsme, /help")
        print(Fore.WHITE + "-" * 60 + "\n")
        
        # Main conversation loop
        while True:
            try:
                # Get user input
                raw_input_text = display_manager.get_user_input()
                
                if not raw_input_text:
                    continue
                
                # Handle commands
                if raw_input_text.startswith('/'):
                    result = display_manager.handle_command(raw_input_text)
                    
                    if result["action"] == "quit":
                        display_manager.show_system_message("Goodbye!")
                        break
                    elif result["action"] == "invalid_command":
                        display_manager.show_system_message(result["message"])
                    elif result["action"] in ["color_cycled", "sme_toggled", "context_switched"]:
                        # FIXED: Only access message if it exists
                        if "message" in result:
                            display_manager.show_system_message(result["message"])
                        # Update status bar
                        status = display_manager.generate_status_bar()
                        print(Fore.CYAN + "\n" + status + "\n")
                    elif result["action"] == "search_opened":
                        display_manager.show_system_message(result["message"])
                        print(Fore.CYAN + f"Found {result['results_count']} results for '{result['term']}'")
                        print(Fore.CYAN + "Use /debug to return to chat, or type a new command.\n")
                    elif result["action"] == "help_shown":
                        # FIXED: Help command is handled properly - no additional message needed
                        pass
                    elif result["action"] == "context_cleared":
                        # FIXED: Access message safely
                        if "message" in result:
                            display_manager.show_system_message(result["message"])
                    elif result["action"] == "save_requested":
                        # FIXED: Access message safely
                        if "message" in result:
                            display_manager.show_system_message(result["message"])
                    
                    continue
                
                # Validate input length
                is_valid, warning, preserved_input = validate_user_input_length(raw_input_text)
                if not is_valid:
                    display_manager.show_system_message(warning)
                    # TODO Phase 2: Implement input preservation for editing
                    continue
                
                # Add user input to memory and display
                add_memory(memories, "user", raw_input_text, debug_logger)
                display_manager.show_user_input(raw_input_text)
                
                # Simple response for Phase 1 testing
                print(Fore.MAGENTA + "\n[Phase 1: Debug system active, SME integration coming in Phase 2...]")
                
                # Simulate assistant response
                test_response = f"Phase 1 implementation received your message: '{raw_input_text[:50]}...'"
                add_memory(memories, "assistant", test_response, debug_logger)
                display_manager.show_assistant_response(test_response)
                
                print(Fore.GREEN + "="*60 + "\n")
                
            except KeyboardInterrupt:
                display_manager.show_system_message("Goodbye!")
                break
    
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Fatal error in main(): {e}")
        print(Fore.RED + f"[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if debug_logger:
            debug_logger.close()

# ------------------ Entry Point ------------------ #
if __name__ == "__main__":
    if args.debug:
        print(Fore.CYAN + "[Info] Debug logging enabled - check debug.log for detailed information")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Fore.CYAN + "\n[System] Shutdown complete.")
    except Exception as e:
        print(Fore.RED + f"[Fatal Error] {e}")
        sys.exit(1)
