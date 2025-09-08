# Chunk 1: Imports and Configuration (FIXED)
import asyncio
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import httpx
import sys
from colorama import init, Fore, Style
import textwrap
import shutil
import argparse

# Phase 2: Ncurses imports with graceful fallback
try:
    import curses
    import curses.textpad
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    curses = None

# ------------------ Initialization ------------------ #
init(autoreset=True)

# Parse command line arguments early
parser = argparse.ArgumentParser(description="Aurora RPG Client with Story Momentum Engine - Phase 2")
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
SYSTEM_PROMPT_TOKENS = 5000
MOMENTUM_ANALYSIS_TOKENS = 6000

REMAINING_TOKENS = CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS - MOMENTUM_ANALYSIS_TOKENS
MEMORY_FRACTION = 0.7
MAX_MEMORY_TOKENS = int(REMAINING_TOKENS * MEMORY_FRACTION)
MAX_USER_INPUT_TOKENS = int(REMAINING_TOKENS * (1 - MEMORY_FRACTION))

# Memory condensation strategies
CONDENSATION_STRATEGIES = {
    "story_critical": {
        "threshold": 100,
        "preservation_ratio": 0.8,
        "instruction": (
            "Preserve all major plot developments, character deaths, world-changing events, "
            "key player decisions, and their consequences. Use decisive language highlighting "
            "the significance of events. Compress dialogue while maintaining essential meaning."
        )
    },
    "character_focused": {
        "threshold": 80,
        "preservation_ratio": 0.7,
        "instruction": (
            "Preserve relationship changes, trust/betrayal moments, character motivations, "
            "personality reveals, Aurora's development, and NPC traits. Emphasize emotional "
            "weight and relationship dynamics. Condense descriptions while keeping character essence."
        )
    },
    "world_building": {
        "threshold": 60,
        "preservation_ratio": 0.6,
        "instruction": (
            "Preserve new locations, lore revelations, cultural information, political changes, "
            "economic systems, magical discoveries, and historical context. Provide rich "
            "foundational details. Compress atmospheric descriptions while keeping key world facts."
        )
    },
    "standard": {
        "threshold": 40,
        "preservation_ratio": 0.4,
        "instruction": (
            "Preserve player actions and immediate consequences for continuity. Compress "
            "everything else aggressively while maintaining basic story flow."
        )
    }
}

# Chunk 2: Color Schemes and Commands (FIXED)
# Phase 2: Enhanced Color Scheme Definitions with ncurses support
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
    "border": "#6c7086",
    "active_border": "#89b4fa",
    "status_bar": "#45475a",
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
    "border": "#565f89",
    "active_border": "#7aa2f7",
    "status_bar": "#32344a",
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
    "border": "#6272a4",
    "active_border": "#bd93f9",
    "status_bar": "#44475a",
}

COLOR_SCHEMES = {
    "midnight_aurora": MIDNIGHT_AURORA,
    "forest_whisper": FOREST_WHISPER,
    "dracula_aurora": DRACULA_AURORA
}

COLOR_CYCLE_ORDER = ["midnight_aurora", "forest_whisper", "dracula_aurora"]

# Phase 2: Enhanced Commands and Layout
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

CONTEXT_HELP = {
    "debug": "Debug context active. Type /debug or press Esc to return to chat.",
    "search": "Search results displayed. Press Esc to return to chat."
}

SME_STATUS_FORMAT = "SME: {pressure_name}({pressure:.2f}) | Villain: {antagonist_name}"

# Phase 2: Ncurses Layout Configuration
LAYOUT_CONFIG = {
    "status_bar_height": 1,
    "input_pane_min_height": 3,
    "input_pane_max_height": 8,
    "border_width": 1,
    "min_terminal_width": 80,
    "min_terminal_height": 24,
    "output_pane_padding": 1,
}

# Prompt files
PROMPT_TOP_FILE = Path("critrules.prompt")
PROMPT_MID_FILE = Path("companion.prompt")
PROMPT_LOW_FILE = Path("lowrules.prompt")

def load_prompt(file_path: Path, debug_logger=None) -> str:
    """Load prompt file with graceful handling of missing files."""
    if not file_path.exists():
        warning_msg = f"Prompt file not found: {file_path}"
        if debug_logger:
            debug_logger.error(warning_msg)
        return ""
    
    content = file_path.read_text(encoding="utf-8").strip()
    if debug_logger:
        debug_logger.system(f"Loaded prompt file {file_path}: {len(content)} characters")
    return content

# Chunk 3: DebugLogger and FIXED ColorManager
# ------------------ Enhanced Debug Logger System (Phase 2) ------------------ #
class DebugLogger:
    """Enhanced file-based debug logging system with ncurses integration"""
    
    def __init__(self, debug_file_enabled=False, log_file_path="debug.log"):
        self.debug_file_enabled = debug_file_enabled
        self.log_file_path = log_file_path
        self.log_file = None
        self.message_buffer = []  # Buffer for ncurses debug context
        self.max_buffer_size = 1000
        
        if debug_file_enabled:
            self._init_log_file()
    
    def _init_log_file(self):
        """Initialize debug log file, overwriting any previous run"""
        try:
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
        """System-level debug messages"""
        self._log_message("SYSTEM", message, "info")
    
    def memory(self, message):
        """Memory management debug messages"""
        self._log_message("MEMORY", message, "warning")
    
    def momentum(self, message):
        """Story momentum engine debug messages"""
        self._log_message("MOMENTUM", message, "info")
    
    def error(self, message):
        """Error-level debug messages"""
        self._log_message("ERROR", message, "error")
    
    def network(self, message):
        """Network/MCP communication debug messages"""
        self._log_message("NETWORK", message, "info")
    
    def interface(self, message):
        """Interface/display debug messages"""
        self._log_message("INTERFACE", message, "info")
    
    def _log_message(self, category, message, severity):
        """Log message to both file and memory buffer"""
        if self.debug_file_enabled:
            self._log_to_file(category, message, severity)
        
        # Always add to buffer for ncurses debug context
        self._add_to_buffer(category, message, severity)
    
    def _log_to_file(self, category, message, severity):
        """Write debug message to log file"""
        if not self.log_file:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] {severity.upper():<7} {category:<8}: {message}\n"
        
        try:
            self.log_file.write(formatted_msg)
            self.log_file.flush()
        except IOError:
            self.debug_file_enabled = False
    
    def _add_to_buffer(self, category, message, severity):
        """Add message to ncurses debug buffer"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_entry = {
            'timestamp': timestamp,
            'category': category,
            'message': message,
            'severity': severity
        }
        
        self.message_buffer.append(debug_entry)
        
        # Keep buffer size manageable
        if len(self.message_buffer) > self.max_buffer_size:
            self.message_buffer = self.message_buffer[-self.max_buffer_size:]
    
    def get_debug_messages(self):
        """Get all debug messages for ncurses debug context"""
        return self.message_buffer.copy()
    
    def close(self):
        """Close debug log file gracefully"""
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"\nDebug session ended: {timestamp}\n")
            self.log_file.close()
            self.log_file = None

# ------------------ FIXED Color Manager (Phase 2) ------------------ #
class ColorManager:
    """Enhanced color manager with ncurses support and dynamic switching - FIXED"""
    
    def __init__(self, initial_scheme="midnight_aurora", interface_type="simple"):
        self.schemes = COLOR_SCHEMES
        self.current_scheme_name = initial_scheme
        self.current_scheme = self.schemes[initial_scheme]
        self.interface_type = interface_type
        self.color_pairs = {}  # For ncurses color pairs
        self.pair_counter = 1  # Start from 1 (0 is reserved)
        
        # Only initialize ncurses colors if we're actually using ncurses
        if interface_type == "curses" and CURSES_AVAILABLE:
            self._init_curses_colors()
    
    def switch_scheme(self, scheme_name):
        """Switch to a different color scheme dynamically"""
        if scheme_name in self.schemes:
            old_scheme = self.current_scheme_name
            self.current_scheme_name = scheme_name
            self.current_scheme = self.schemes[scheme_name]
            
            if self.interface_type == "curses" and CURSES_AVAILABLE:
                self._reinit_curses_colors()
            
            return True
        return False
    
    def cycle_scheme(self):
        """Cycle to the next color scheme in order"""
        current_index = COLOR_CYCLE_ORDER.index(self.current_scheme_name)
        next_index = (current_index + 1) % len(COLOR_CYCLE_ORDER)
        next_scheme = COLOR_CYCLE_ORDER[next_index]
        return self.switch_scheme(next_scheme)
    
    def get_available_schemes(self):
        """Return list of available color schemes"""
        return list(self.schemes.keys())
    
    def get_current_scheme_name(self):
        """Return current scheme name for status display"""
        return self.current_scheme_name.replace('_', ' ').title()
    
    def get_color(self, color_name):
        """Get color for current interface type - FIXED to always return strings for simple interface"""
        # CRITICAL FIX: Always return ANSI string for simple interface
        if self.interface_type == "simple":
            hex_color = self.current_scheme.get(color_name, "#ffffff")
            return self._hex_to_ansi(hex_color)
        elif self.interface_type == "curses" and CURSES_AVAILABLE and self.color_pairs:
            return self.color_pairs.get(color_name, 1)
        else:
            # Fallback to ANSI string in all other cases
            hex_color = self.current_scheme.get(color_name, "#ffffff")
            return self._hex_to_ansi(hex_color)
    
    def _init_curses_colors(self):
        """Initialize all color pairs for ncurses"""
        if not CURSES_AVAILABLE:
            return
        
        try:
            curses.start_color()
            curses.use_default_colors()
            self._setup_color_pairs()
        except curses.error:
            # Fallback if color initialization fails
            pass
    
    def _reinit_curses_colors(self):
        """Reinitialize color pairs when scheme changes"""
        if CURSES_AVAILABLE and self.interface_type == "curses":
            self.color_pairs.clear()
            self.pair_counter = 1
            self._setup_color_pairs()
    
    def _setup_color_pairs(self):
        """Set up color pairs for the current scheme"""
        if not CURSES_AVAILABLE:
            return
        
        for color_name, hex_color in self.current_scheme.items():
            try:
                curses_color = self._hex_to_curses_color(hex_color)
                curses.init_pair(self.pair_counter, curses_color, -1)
                self.color_pairs[color_name] = curses.color_pair(self.pair_counter)
                self.pair_counter += 1
            except (curses.error, ValueError):
                # Fallback to default color
                self.color_pairs[color_name] = curses.color_pair(0) if CURSES_AVAILABLE else 1
    
    def _hex_to_curses_color(self, hex_color):
        """Convert hex color to closest curses color"""
        r, g, b = self._hex_to_rgb(hex_color)
        
        # Simple heuristic for color mapping
        if r > 200 and g < 100 and b < 100:
            return curses.COLOR_RED
        elif r < 100 and g > 200 and b < 100:
            return curses.COLOR_GREEN
        elif r > 200 and g > 200 and b < 100:
            return curses.COLOR_YELLOW
        elif r < 100 and g < 100 and b > 200:
            return curses.COLOR_BLUE
        elif r > 200 and g < 100 and b > 200:
            return curses.COLOR_MAGENTA
        elif r < 100 and g > 200 and b > 200:
            return curses.COLOR_CYAN
        elif r > 150 and g > 150 and b > 150:
            return curses.COLOR_WHITE
        else:
            return curses.COLOR_BLACK
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _hex_to_ansi(self, hex_color):
        """Convert hex color to closest ANSI color for simple terminal - COMPREHENSIVE MAPPING"""
        color_map = {
            # White/Light colors
            "#ffffff": Fore.WHITE, "#f8f8f2": Fore.WHITE, "#cdd6f4": Fore.WHITE,
            "#a6adc8": Fore.WHITE, "#c0caf5": Fore.WHITE, "#9aa5ce": Fore.WHITE,
            
            # Red colors
            "#ff0000": Fore.RED, "#ff5555": Fore.RED, "#f38ba8": Fore.RED,
            "#f7768e": Fore.RED, "#ff757f": Fore.RED,
            
            # Green colors
            "#00ff00": Fore.GREEN, "#50fa7b": Fore.GREEN, "#a6e3a1": Fore.GREEN,
            "#9ece6a": Fore.GREEN,
            
            # Blue colors
            "#0000ff": Fore.BLUE, "#6272a4": Fore.BLUE,
            
            # Cyan colors
            "#8be9fd": Fore.CYAN, "#89dceb": Fore.CYAN, "#7dcfff": Fore.CYAN,
            "#2ac3de": Fore.CYAN, "#74c7ec": Fore.CYAN, "#94e2d5": Fore.CYAN,
            "#73daca": Fore.CYAN,
            
            # Yellow colors
            "#ffff00": Fore.YELLOW, "#f1fa8c": Fore.YELLOW, "#f9e2af": Fore.YELLOW,
            "#e0af68": Fore.YELLOW, "#ff9e64": Fore.YELLOW, "#ffb86c": Fore.YELLOW,
            "#fab387": Fore.YELLOW,
            
            # Magenta/Purple colors
            "#ff00ff": Fore.MAGENTA, "#bd93f9": Fore.MAGENTA, "#ff79c6": Fore.MAGENTA,
            "#bb9af7": Fore.MAGENTA, "#b4befe": Fore.MAGENTA,
            
            # Background colors (map to white for text)
            "#1e1e2e": Fore.WHITE, "#313244": Fore.WHITE, "#1a1b26": Fore.WHITE,
            "#24283b": Fore.WHITE, "#282a36": Fore.WHITE, "#44475a": Fore.WHITE,
            "#6c7086": Fore.WHITE, "#89b4fa": Fore.WHITE, "#45475a": Fore.WHITE,
            "#565f89": Fore.WHITE, "#7aa2f7": Fore.WHITE, "#32344a": Fore.WHITE,
        }
        
        # Return mapped color or default to WHITE if not found
        return color_map.get(hex_color, Fore.WHITE)

# Chunk 4: Utility Functions and ContextManager
# ------------------ Utility Functions ------------------ #
def get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def get_terminal_height(default=24):
    try:
        return shutil.get_terminal_size((80, default)).lines
    except Exception:
        return default

def print_wrapped(text: str, color=Fore.GREEN, indent: int = 0):
    """Print text with word wrapping for simple interface"""
    width = get_terminal_width() - indent
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=' ' * indent)
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        lines = para.splitlines()
        wrapped_para = "\n".join(wrapper.fill(line) for line in lines)
        print(color + wrapped_para)
        if i < len(paragraphs) - 1:
            print("")

def estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    return max(1, len(text) // 4)

def validate_user_input_length(user_input: str) -> tuple[bool, str, str]:
    """Validate user input length and provide helpful feedback if too long"""
    input_tokens = estimate_tokens(user_input)

    if input_tokens <= MAX_USER_INPUT_TOKENS:
        return True, "", ""

    char_count = len(user_input)
    max_chars = MAX_USER_INPUT_TOKENS * 4

    warning = (f"Input too long ({input_tokens:,} tokens, {char_count:,} chars). "
              f"Maximum: {MAX_USER_INPUT_TOKENS:,} tokens ({max_chars:,} chars). "
              f"Please shorten your input - it has been preserved for editing.")

    return False, warning, user_input

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

# ------------------ Enhanced Context Manager (Phase 2) ------------------ #
class ContextManager:
    """Enhanced context manager supporting three contexts with advanced features"""
    
    def __init__(self):
        self.chat_history = []
        self.debug_history = [] 
        self.search_results = []
        self.current_search_term = ""
        self.current_context = "chat"  # "chat", "debug", "search"
        self.max_history = 1000
        self.sme_status_visible = False
        self.scroll_position = 0  # For ncurses scrolling
    
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
        """Switch between contexts and reset scroll position"""
        old_context = self.current_context
        self.current_context = new_context
        self.scroll_position = 0  # Reset scroll when switching contexts
        return old_context
    
    def get_current_history(self):
        """Get history for current context"""
        if self.current_context == "chat":
            return self.chat_history
        elif self.current_context == "debug":
            return self.debug_history
        else:  # search
            return self.search_results
    
    def search_chat_history(self, search_term):
        """Search chat history and return results with context"""
        results = []
        search_lower = search_term.lower()
        
        for i, entry in enumerate(self.chat_history):
            if search_lower in entry['content'].lower():
                # Add context around the match
                context_start = max(0, i - 2)
                context_end = min(len(self.chat_history), i + 3)
                
                result = {
                    'index': i,
                    'content': entry['content'],
                    'type': entry['type'],
                    'timestamp': entry['timestamp'],
                    'context_messages': self.chat_history[context_start:context_end],
                    'match_highlight': self._highlight_match(entry['content'], search_term)
                }
                results.append(result)
        
        return results
    
    def _highlight_match(self, text, search_term):
        """Create highlighted version of text for search results"""
        # Simple case-insensitive highlighting
        lower_text = text.lower()
        lower_term = search_term.lower()
        
        highlighted = ""
        last_end = 0
        
        start = lower_text.find(lower_term)
        while start != -1:
            highlighted += text[last_end:start]
            highlighted += f"[MATCH]{text[start:start+len(search_term)]}[/MATCH]"
            last_end = start + len(search_term)
            start = lower_text.find(lower_term, last_end)
        
        highlighted += text[last_end:]
        return highlighted
    
    def set_search_results(self, search_term, results):
        """Set current search term and results"""
        self.current_search_term = search_term
        self.search_results = results
        self.scroll_position = 0
    
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
        
        self.scroll_position = 0
    
    def scroll_up(self, lines=1):
        """Scroll up in current context"""
        self.scroll_position = max(0, self.scroll_position - lines)
    
    def scroll_down(self, lines=1):
        """Scroll down in current context"""
        max_scroll = max(0, self.get_message_count() - 1)
        self.scroll_position = min(max_scroll, self.scroll_position + lines)
    
    def _trim_history(self, context_type):
        """Keep history within limits"""
        if context_type == "chat" and len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        elif context_type == "debug" and len(self.debug_history) > self.max_history:
            self.debug_history = self.debug_history[-self.max_history:]

# Chunk 5: Interface Detection and DisplayManager Foundation
# ------------------ Interface Detection and Initialization ------------------ #
def initialize_interface(bypass_curses=False, debug_logger=None):
    """Initialize interface with graceful fallback and comprehensive logging"""
    if bypass_curses:
        if debug_logger:
            debug_logger.interface("Using simple interface (--bypasscurses flag)")
        return "simple"
    
    if not CURSES_AVAILABLE:
        if debug_logger:
            debug_logger.interface("Ncurses not available, using simple interface")
        print("Note: Ncurses unavailable, using simple interface.")
        print("Use --bypasscurses to suppress this message.")
        return "simple"
    
    # Test terminal compatibility
    try:
        # Quick ncurses compatibility test
        curses.wrapper(lambda stdscr: None)
        
        # Check minimum terminal size
        height, width = get_terminal_height(), get_terminal_width()
        min_height = LAYOUT_CONFIG["min_terminal_height"]
        min_width = LAYOUT_CONFIG["min_terminal_width"]
        
        if height < min_height or width < min_width:
            if debug_logger:
                debug_logger.interface(f"Terminal too small ({width}x{height}), minimum {min_width}x{min_height}")
            print(f"Terminal too small ({width}x{height}), using simple interface.")
            print(f"Minimum size: {min_width}x{min_height}")
            return "simple"
        
        if debug_logger:
            debug_logger.interface(f"Ncurses interface initialized successfully ({width}x{height})")
        return "curses"
        
    except (curses.error, Exception) as e:
        if debug_logger:
            debug_logger.interface(f"Ncurses initialization failed: {e}")
            debug_logger.interface("Falling back to simple interface")
        
        print("Note: Ncurses initialization failed, using simple interface.")
        print("Use --bypasscurses to suppress this message.")
        return "simple"

# ------------------ FIXED Display Manager (Phase 2) ------------------ #
class DisplayManager:
    """Enhanced display manager with ncurses support and clean abstraction - FIXED"""
    
    def __init__(self, interface_type="simple", debug_logger=None):
        self.interface_type = interface_type
        self.debug_logger = debug_logger
        # CRITICAL FIX: Ensure ColorManager gets correct interface_type
        self.color_manager = ColorManager(args.colorscheme, interface_type)
        self.context_manager = ContextManager()
        self.preserved_input = ""
        self.quit_dialog_active = False
        
        # Ncurses-specific attributes
        self.stdscr = None
        self.windows = {}
        self.current_input = ""
        self.input_cursor_pos = 0
        self.input_lines = [""]
        self.input_line_index = 0
        
        if debug_logger:
            debug_logger.interface(f"Display manager initialized with {interface_type} interface")
            debug_logger.interface(f"ColorManager interface type: {self.color_manager.interface_type}")
    
    def initialize_ncurses(self, stdscr):
        """Initialize ncurses interface with proper setup"""
        self.stdscr = stdscr
        
        # Basic ncurses setup
        curses.curs_set(1)  # Show cursor
        curses.noecho()     # Don't echo keys
        curses.cbreak()     # React to keys immediately
        stdscr.keypad(True) # Enable special keys
        stdscr.timeout(100) # Non-blocking input with 100ms timeout
        
        # Initialize colors
        self.color_manager._init_curses_colors()
        
        # Create window layout
        self._create_windows()
        
        if self.debug_logger:
            self.debug_logger.interface("Ncurses interface fully initialized")
    
    def _create_windows(self):
        """Create the window layout for ncurses interface"""
        if not self.stdscr:
            return
        
        height, width = self.stdscr.getmaxyx()
        
        # Calculate layout dimensions
        status_height = LAYOUT_CONFIG["status_bar_height"]
        input_height = LAYOUT_CONFIG["input_pane_min_height"]
        output_height = height - status_height - input_height - 2  # Account for borders
        
        # Create windows
        try:
            # Status bar at top
            self.windows['status'] = curses.newwin(status_height, width, 0, 0)
            
            # Output pane in middle
            self.windows['output'] = curses.newwin(output_height, width, status_height, 0)
            
            # Input pane at bottom (only visible in chat context)
            input_y = height - input_height - 1
            self.windows['input'] = curses.newwin(input_height, width, input_y, 0)
            
            # Add borders to windows
            for window_name, window in self.windows.items():
                if window_name != 'status':  # Status bar has no border
                    window.border()
            
            if self.debug_logger:
                self.debug_logger.interface(f"Created ncurses windows: {width}x{height}")
                
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to create ncurses windows: {e}")
            raise
    
    def show_user_input(self, text):
        """Display user input in current interface"""
        self.context_manager.add_chat_message(text, 'user')
        
        if self.debug_logger:
            self.debug_logger.system(f"User input: {text[:50]}...")
        
        if self.interface_type == "curses":
            self._ncurses_show_user_input(text)
        else:
            self._simple_show_user_input(text)
    
    def show_assistant_response(self, text):
        """Display assistant response in current interface"""
        self.context_manager.add_chat_message(text, 'assistant')
        
        if self.debug_logger:
            self.debug_logger.system(f"Assistant response: {len(text)} characters")
        
        if self.interface_type == "curses":
            self._ncurses_show_assistant_response(text)
        else:
            self._simple_show_assistant_response(text)
    
    def show_system_message(self, text, log_only=False):
        """Show system messages with appropriate interface"""
        if log_only:
            if self.debug_logger:
                self.debug_logger.system(text)
            return
        
        self.context_manager.add_chat_message(text, 'system')
        
        if self.debug_logger:
            self.debug_logger.system(f"System message: {text}")
        
        if self.interface_type == "curses":
            self._ncurses_show_system_message(text)
        else:
            self._simple_show_system_message(text)
    
    def _simple_show_user_input(self, text):
        """Show user input in simple terminal"""
        color = self.color_manager.get_color('user_input')
        print(color + f"> {text}" + Style.RESET_ALL)
    
    def _simple_show_assistant_response(self, text):
        """Show assistant response in simple terminal"""
        color = self.color_manager.get_color('assistant_primary')
        print_wrapped(text, color)
    
    def _simple_show_system_message(self, text):
        """Show system message in simple terminal"""
        color = self.color_manager.get_color('system_info')
        print(color + f"[System] {text}" + Style.RESET_ALL)
    
    def _ncurses_show_user_input(self, text):
        """Show user input in ncurses interface"""
        # Implementation will be added in future phases
        pass
    
    def _ncurses_show_assistant_response(self, text):
        """Show assistant response in ncurses interface"""
        # Implementation will be added in future phases
        pass
    
    def _ncurses_show_system_message(self, text):
        """Show system message in ncurses interface"""
        # Implementation will be added in future phases
        pass

# Chunk 6: DisplayManager Command Handling with FIXED Input Logic
    def handle_command(self, command):
        """Handle special commands with enhanced functionality"""
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
        """Switch to debug context or return to chat"""
        if self.context_manager.current_context == "chat":
            old_context = self.context_manager.switch_context("debug")
            
            # Populate debug context with messages from debug logger
            if self.debug_logger:
                debug_messages = self.debug_logger.get_debug_messages()
                for msg in debug_messages:
                    self.context_manager.add_debug_message(
                        f"[{msg['category']}] {msg['message']}", 
                        msg['severity']
                    )
            
            # Add instruction message
            instruction = CONTEXT_HELP["debug"]
            self.context_manager.add_debug_message(instruction, 'info')
            
            return {"action": "context_switched", "context": "debug", "message": "Switched to debug context"}
        else:
            self.context_manager.switch_context("chat")
            return {"action": "context_switched", "context": "chat", "message": "Returned to chat context"}
    
    def _cycle_color_scheme(self):
        """Cycle through color schemes"""
        old_scheme = self.color_manager.current_scheme_name
        self.color_manager.cycle_scheme()
        new_scheme = self.color_manager.current_scheme_name
        
        return {
            "action": "color_cycled",
            "old_scheme": old_scheme,
            "new_scheme": new_scheme,
            "message": f"Color scheme: {self.color_manager.get_current_scheme_name()}"
        }
    
    def _toggle_sme_status(self):
        """Toggle SME status display"""
        visible = self.context_manager.toggle_sme_status()
        status = "enabled" if visible else "disabled"
        return {
            "action": "sme_toggled",
            "visible": visible,
            "message": f"SME status display {status}"
        }
    
    def _open_search_context(self, search_term):
        """Open search context with results"""
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
        """Show command help"""
        help_text = "Available commands:\n"
        for cmd, desc in COMMANDS.items():
            help_text += f"  {cmd}: {desc}\n"
        
        self.show_system_message(help_text.strip())
        return {"action": "help_shown", "message": "Command help displayed"}
    
    def _clear_current_context(self):
        """Clear current context history"""
        context_name = self.context_manager.current_context
        self.context_manager.clear_current_context()
        return {
            "action": "context_cleared", 
            "context": context_name, 
            "message": f"Cleared {context_name} context"
        }
    
    def _save_conversation(self, filename):
        """Save conversation (placeholder for future implementation)"""
        return {
            "action": "save_requested", 
            "filename": filename, 
            "message": "Save functionality coming in later phase"
        }
    
    def get_user_input(self):
        """Get user input with FIXED interface routing"""
        # CRITICAL FIX: Properly route based on interface_type
        if self.debug_logger:
            self.debug_logger.interface(f"get_user_input called with interface_type: {self.interface_type}")
        
        if self.interface_type == "curses":
            return self._ncurses_get_input()
        else:
            return self._simple_get_input()
    
    def _simple_get_input(self):
        """Simple terminal input collection - FIXED"""
        user_lines = []
        
        # CRITICAL FIX: Ensure we get a string color, not an integer
        try:
            color = self.color_manager.get_color('user_prompt')
            if self.debug_logger:
                self.debug_logger.interface(f"Got color for user_prompt: {repr(color)} (type: {type(color)})")
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to get color: {e}")
            color = Fore.CYAN  # Fallback color
        
        # Ensure color is a string
        if not isinstance(color, str):
            if self.debug_logger:
                self.debug_logger.error(f"Color is not string: {repr(color)}, using fallback")
            color = Fore.CYAN
        
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
    
    def _ncurses_get_input(self):
        """Ncurses input collection (placeholder for future implementation)"""
        # CRITICAL FIX: Don't call simple input from ncurses method
        # This was causing the wrong interface routing
        if self.debug_logger:
            self.debug_logger.interface("Ncurses input not implemented, falling back to simple")
        # For Phase 2, fallback to simple but log the issue
        return self._simple_get_input()
    
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
    
    def refresh_display(self):
        """Refresh the display (ncurses-specific)"""
        if self.interface_type == "curses" and self.stdscr:
            try:
                for window in self.windows.values():
                    window.refresh()
                self.stdscr.refresh()
            except curses.error:
                pass

# Chunk 7: Memory Management, MCP Communication, and Support Functions
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
    """Call MCP with automatic retry logic"""
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
                wait_time = 2 ** attempt
                if debug_logger:
                    debug_logger.network(f"MCP call failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                if debug_logger:
                    debug_logger.error(f"MCP call failed after {max_retries} attempts: {e}")
                raise e

# ------------------ Antagonist System Functions ------------------ #
def get_pressure_name(pressure_level):
    """Convert pressure level to named range"""
    if pressure_level < 0.1:
        return "low"
    elif pressure_level < 0.3:
        return "building" 
    elif pressure_level < 0.6:
        return "critical"
    else:
        return "explosive"

# ------------------ Application Initialization ------------------ #
def initialize_application():
    """Initialize application with enhanced interface detection"""
    debug_logger = DebugLogger(args.debug, "debug.log") if args.debug else None
    
    if debug_logger:
        debug_logger.system("Aurora RPG Client Phase 2 starting...")
        debug_logger.system(f"Arguments: debug={args.debug}, bypasscurses={args.bypasscurses}, colorscheme={args.colorscheme}")
    
    # Determine interface type
    interface_type = initialize_interface(args.bypasscurses, debug_logger)
    
    if debug_logger:
        debug_logger.system(f"Interface type determined: {interface_type}")
    
    # Initialize display manager
    display_manager = DisplayManager(interface_type, debug_logger)
    
    # Verify the interface type was set correctly
    if debug_logger:
        debug_logger.system(f"DisplayManager interface_type: {display_manager.interface_type}")
        debug_logger.system(f"ColorManager interface_type: {display_manager.color_manager.interface_type}")
    
    # Show clean startup message
    if interface_type == "curses":
        # Ncurses initialization will be handled in main loop
        pass
    else:
        print("Aurora RPG Client Phase 2 ready. Type /help for commands.")
    
    return display_manager, debug_logger

# Chunk 8: Main Application Logic with FIXED Interface Routing (Final)
# ------------------ FIXED Main Application Logic ------------------ #
async def main():
    """Main application entry point - Phase 2 with FIXED interface routing"""
    try:
        # Initialize application
        display_manager, debug_logger = initialize_application()
        
        # Validate configuration
        validate_token_allocation(debug_logger)
        
        # Load conversation memory
        memories = load_memory(debug_logger)
        display_manager.show_system_message(f"Loaded {len(memories)} memories from previous sessions", log_only=True)
        
        # CRITICAL FIX: Properly route based on interface_type
        if debug_logger:
            debug_logger.system(f"Main loop routing - interface_type: {display_manager.interface_type}")
        
        # Handle ncurses vs simple interface - FIXED LOGIC
        if display_manager.interface_type == "curses":
            if debug_logger:
                debug_logger.system("Routing to ncurses interface")
            await run_curses_interface(display_manager, debug_logger, memories)
        else:
            if debug_logger:
                debug_logger.system("Routing to simple interface")
            await run_simple_interface(display_manager, debug_logger, memories)
    
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Fatal error in main(): {e}")
        print(Fore.RED + f"[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if debug_logger:
            debug_logger.close()

async def run_simple_interface(display_manager, debug_logger, memories):
    """Run the simple terminal interface - VERIFIED SIMPLE ONLY"""
    if debug_logger:
        debug_logger.system("Starting simple interface loop")
        debug_logger.system(f"DisplayManager interface_type: {display_manager.interface_type}")
    
    # Display session info
    print(Fore.GREEN + "\n" + "="*60)
    print(Fore.GREEN + "Aurora RPG Client - Phase 2 (Enhanced Display & Context Management)")
    print(Fore.GREEN + "="*60)
    
    # Show status bar
    status = display_manager.generate_status_bar()
    print(Fore.CYAN + status)
    
    print(Fore.WHITE + "\nType your message (press Enter twice to send, '/quit' to exit):")
    print(Fore.WHITE + "Commands: /debug, /color, /showsme, /search <term>, /help")
    print(Fore.WHITE + "-" * 60 + "\n")
    
    # Main conversation loop
    while True:
        try:
            # Get user input - this should now work properly
            if debug_logger:
                debug_logger.system("Calling get_user_input()")
            
            raw_input_text = display_manager.get_user_input()
            
            if debug_logger:
                debug_logger.system(f"Received input: {raw_input_text[:50]}...")
            
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
                elif "message" in result:
                    display_manager.show_system_message(result["message"])
                    
                    # Update status bar for relevant commands
                    if result["action"] in ["color_cycled", "sme_toggled", "context_switched"]:
                        status = display_manager.generate_status_bar()
                        print(Fore.CYAN + "\n" + status + "\n")
                
                continue
            
            # Validate input length
            is_valid, warning, preserved_input = validate_user_input_length(raw_input_text)
            if not is_valid:
                display_manager.show_system_message(warning)
                continue
            
            # Add user input to memory and display
            add_memory(memories, "user", raw_input_text, debug_logger)
            display_manager.show_user_input(raw_input_text)
            
            # Phase 2: Enhanced test response with MCP integration placeholder
            print(Fore.MAGENTA + "\n[Phase 2: Enhanced display system active, full MCP integration coming in Phase 3...]")
            
            test_response = f"Phase 2 enhanced response to: '{raw_input_text[:50]}...'\nContext: {display_manager.context_manager.current_context}"
            add_memory(memories, "assistant", test_response, debug_logger)
            display_manager.show_assistant_response(test_response)
            
            print(Fore.GREEN + "="*60 + "\n")
            
        except KeyboardInterrupt:
            display_manager.show_system_message("Goodbye!")
            break
        except Exception as e:
            if debug_logger:
                debug_logger.error(f"Error in simple interface loop: {e}")
            print(Fore.RED + f"[Error] {e}")

async def run_curses_interface(display_manager, debug_logger, memories):
    """Run the ncurses interface - FIXED to not call simple interface"""
    if debug_logger:
        debug_logger.interface("Ncurses interface requested but not fully implemented in Phase 2")
    
    # CRITICAL FIX: Don't call run_simple_interface from here
    # This was causing the interface routing confusion
    print("Ncurses interface coming in Phase 3. Falling back to simple interface...")
    
    # Temporarily change interface type to simple for fallback
    display_manager.interface_type = "simple"
    display_manager.color_manager.interface_type = "simple"
    
    if debug_logger:
        debug_logger.interface("Temporarily switched to simple interface for Phase 2 fallback")
    
    await run_simple_interface(display_manager, debug_logger, memories)

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
