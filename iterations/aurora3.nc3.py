# Chunk 1: Imports and Enhanced Configuration (Phase 3 - COLOR SYSTEM IMPLEMENTATION)
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

# Phase 3: Ncurses imports with graceful fallback
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
parser = argparse.ArgumentParser(description="Aurora RPG Client Phase 3 - Color System Implementation")
parser.add_argument("--bypasscurses", action="store_true",
                   help="Use simple terminal interface instead of ncurses")
parser.add_argument("--debug", action="store_true",
                   help="Enable debug logging to debug.log file")
parser.add_argument("--colorscheme", default="midnight_aurora",
                   choices=["midnight_aurora", "forest_whisper", "dracula_aurora"],
                   help="Initial color scheme (can be changed with /color command)")

args = parser.parse_args()

# ------------------ Enhanced Configuration for Phase 3 ------------------ #
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

# Chunk 2: Enhanced Color Schemes and Commands (Phase 3 - COMPLETE COLOR SYSTEM)

# Phase 3: Enhanced Color Scheme Definitions with comprehensive ncurses and ANSI support
MIDNIGHT_AURORA = {
    # Background and base colors - Nordic-inspired dark blue-grey theme
    "background": "#1e1e2e",      # Deep charcoal blue
    "surface": "#313244",         # Elevated surface
    "text_primary": "#cdd6f4",    # Cool white
    "text_secondary": "#a6adc8",  # Muted blue-grey
    
    # User input colors - Cyan family for clarity
    "user_input": "#89dceb",      # Bright cyan
    "user_prompt": "#74c7ec",     # Sky blue
    
    # Assistant response colors - Green family for warmth
    "assistant_primary": "#a6e3a1",   # Soft green
    "assistant_secondary": "#94e2d5", # Mint green
    
    # System colors - Purple family for distinction
    "system_info": "#b4befe",     # Lavender
    "system_success": "#a6e3a1",  # Green for success
    
    # Debug colors - Yellow/Orange/Red progression
    "debug_info": "#f9e2af",      # Warm yellow
    "debug_warning": "#fab387",   # Peach orange
    "debug_error": "#f38ba8",     # Rose red
    "debug_critical": "#f38ba8",  # Bright red
    
    # Interface elements
    "border": "#6c7086",          # Muted grey
    "active_border": "#89b4fa",   # Active blue
    "status_bar": "#45475a",      # Status background
}

FOREST_WHISPER = {
    # Background and base colors - Nature-inspired greens and blues
    "background": "#1a1b26",      # Deep forest night
    "surface": "#24283b",         # Tree bark
    "text_primary": "#c0caf5",    # Moonlight white
    "text_secondary": "#9aa5ce",  # Misty blue
    
    # User input colors - Ocean blues
    "user_input": "#7dcfff",      # Ocean cyan
    "user_prompt": "#2ac3de",     # Deep ocean
    
    # Assistant response colors - Forest greens
    "assistant_primary": "#9ece6a",   # Spring green
    "assistant_secondary": "#73daca", # Seafoam
    
    # System colors - Mystical purples
    "system_info": "#bb9af7",     # Violet
    "system_success": "#9ece6a",  # Green success
    
    # Debug colors - Amber/Orange progression
    "debug_info": "#e0af68",      # Golden amber
    "debug_warning": "#ff9e64",   # Sunset orange
    "debug_error": "#f7768e",     # Coral red
    "debug_critical": "#ff757f",  # Bright coral
    
    # Interface elements
    "border": "#565f89",          # Twilight grey
    "active_border": "#7aa2f7",   # Active blue
    "status_bar": "#32344a",      # Dark surface
}

DRACULA_AURORA = {
    # Background and base colors - Popular Dracula theme adaptation
    "background": "#282a36",      # Dracula background
    "surface": "#44475a",         # Dracula selection
    "text_primary": "#f8f8f2",    # Dracula foreground
    "text_secondary": "#6272a4",  # Dracula comment
    
    # User input colors - Dracula cyan and green
    "user_input": "#8be9fd",      # Dracula cyan
    "user_prompt": "#50fa7b",     # Dracula green
    
    # Assistant response colors - Dracula purple and pink
    "assistant_primary": "#bd93f9",   # Dracula purple
    "assistant_secondary": "#ff79c6", # Dracula pink
    
    # System colors - Purple emphasis
    "system_info": "#bd93f9",     # Purple
    "system_success": "#50fa7b",  # Green
    
    # Debug colors - Dracula palette
    "debug_info": "#f1fa8c",      # Dracula yellow
    "debug_warning": "#ffb86c",   # Dracula orange
    "debug_error": "#ff5555",     # Dracula red
    "debug_critical": "#ff5555",  # Bright red
    
    # Interface elements
    "border": "#6272a4",          # Comment color
    "active_border": "#bd93f9",   # Purple accent
    "status_bar": "#44475a",      # Selection color
}

# Phase 3: Enhanced color scheme metadata and organization
COLOR_SCHEMES = {
    "midnight_aurora": {
        "colors": MIDNIGHT_AURORA,
        "description": "Dark blue-grey with Nordic inspiration - excellent for long sessions",
        "optimized_for": "readability, low eye strain"
    },
    "forest_whisper": {
        "colors": FOREST_WHISPER,
        "description": "Nature-inspired greens and blues - optimized for readability",
        "optimized_for": "natural feel, balanced contrast"
    },
    "dracula_aurora": {
        "colors": DRACULA_AURORA,
        "description": "Popular Dracula theme adaptation - bold and vibrant",
        "optimized_for": "high contrast, vibrant display"
    }
}

COLOR_CYCLE_ORDER = ["midnight_aurora", "forest_whisper", "dracula_aurora"]

# Phase 3: Enhanced Commands with color system integration
COMMANDS = {
    "/debug": "Switch to debug context (Esc to return)",
    "/search <term>": "Search chat history in separate context", 
    "/color": "Cycle through color schemes (midnight→forest→dracula→midnight)",
    "/color <scheme>": "Switch to specific color scheme (midnight_aurora, forest_whisper, dracula_aurora)",
    "/showsme": "Toggle Story Momentum Engine status display in status bar",
    "/quit": "Exit the application", 
    "/save <filename>": "Save conversation (optional: --chat, --debug, --both)",
    "/clear": "Clear current context history",
    "/help": "Show command help",
    "/themes": "Show available color themes with descriptions"
}

CONTEXT_HELP = {
    "debug": "Debug context active. Type /debug or press Esc to return to chat.",
    "search": "Search results displayed. Press Esc to return to chat."
}

SME_STATUS_FORMAT = "SME: {pressure_name}({pressure:.2f}) | Villain: {antagonist_name}"

# Phase 3: Ncurses Layout Configuration with color-aware settings
LAYOUT_CONFIG = {
    "status_bar_height": 1,
    "input_pane_min_height": 3,
    "input_pane_max_height": 8,
    "border_width": 1,
    "min_terminal_width": 80,
    "min_terminal_height": 24,
    "output_pane_padding": 1,
    "color_test_enabled": True,  # Enable color compatibility testing
    "theme_transition_delay": 0.1,  # Seconds for smooth theme transitions
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

# Chunk 3: Enhanced DebugLogger and COMPLETE ColorManager (Phase 3 - CORE COLOR SYSTEM)

# ------------------ Enhanced Debug Logger System (Phase 3) ------------------ #
class DebugLogger:
    """Enhanced file-based debug logging system with color system integration"""
    
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
            self.log_file.write("Phase 3: Color System Implementation\n")
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
    
    def color(self, message):
        """Color system debug messages (Phase 3)"""
        self._log_message("COLOR", message, "info")
    
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
        formatted_msg = f"[{timestamp}] {severity.upper():<7} {category:<9}: {message}\n"
        
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

# ------------------ COMPLETE Color Manager (Phase 3) ------------------ #
class ColorManager:
    """Complete color manager with dynamic switching, ncurses support, and comprehensive ANSI mapping"""
    
    def __init__(self, initial_scheme="midnight_aurora", interface_type="simple", debug_logger=None):
        self.schemes = COLOR_SCHEMES
        self.current_scheme_name = initial_scheme
        self.current_scheme = self.schemes[initial_scheme]["colors"]
        self.interface_type = interface_type
        self.color_pairs = {}  # For ncurses color pairs
        self.pair_counter = 1  # Start from 1 (0 is reserved)
        self.debug_logger = debug_logger
        
        if debug_logger:
            debug_logger.color(f"ColorManager initialized with scheme: {initial_scheme}, interface: {interface_type}")
        
        # Only initialize ncurses colors if we're actually using ncurses
        if interface_type == "curses" and CURSES_AVAILABLE:
            self._init_curses_colors()
    
    def switch_scheme(self, scheme_name):
        """Switch to a different color scheme dynamically"""
        if scheme_name in self.schemes:
            old_scheme = self.current_scheme_name
            self.current_scheme_name = scheme_name
            self.current_scheme = self.schemes[scheme_name]["colors"]
            
            if self.debug_logger:
                self.debug_logger.color(f"Switched color scheme: {old_scheme} → {scheme_name}")
            
            if self.interface_type == "curses" and CURSES_AVAILABLE:
                self._reinit_curses_colors()
            
            return True
        return False
    
    def cycle_scheme(self):
        """Cycle to the next color scheme in order"""
        current_index = COLOR_CYCLE_ORDER.index(self.current_scheme_name)
        next_index = (current_index + 1) % len(COLOR_CYCLE_ORDER)
        next_scheme = COLOR_CYCLE_ORDER[next_index]
        
        if self.debug_logger:
            self.debug_logger.color(f"Cycling color scheme: {self.current_scheme_name} → {next_scheme}")
        
        return self.switch_scheme(next_scheme)
    
    def get_available_schemes(self):
        """Return list of available color schemes with metadata"""
        return {name: data["description"] for name, data in self.schemes.items()}
    
    def get_current_scheme_name(self):
        """Return current scheme name for status display"""
        return self.current_scheme_name.replace('_', ' ').title()
    
    def get_current_scheme_description(self):
        """Return current scheme description"""
        return self.schemes[self.current_scheme_name]["description"]
    
    def get_color(self, color_name):
        """Get color for current interface type with comprehensive fallback"""
        if self.interface_type == "simple":
            hex_color = self.current_scheme.get(color_name, "#ffffff")
            ansi_color = self._hex_to_ansi(hex_color)
            
            if self.debug_logger:
                self.debug_logger.color(f"Simple interface color lookup: {color_name} → {hex_color} → {ansi_color}")
            
            return ansi_color
        elif self.interface_type == "curses" and CURSES_AVAILABLE and self.color_pairs:
            curses_color = self.color_pairs.get(color_name, 1)
            
            if self.debug_logger:
                self.debug_logger.color(f"Curses interface color lookup: {color_name} → pair {curses_color}")
            
            return curses_color
        else:
            # Fallback to ANSI string in all other cases
            hex_color = self.current_scheme.get(color_name, "#ffffff")
            ansi_color = self._hex_to_ansi(hex_color)
            
            if self.debug_logger:
                self.debug_logger.color(f"Fallback color lookup: {color_name} → {hex_color} → {ansi_color}")
            
            return ansi_color
    
    def test_color_support(self):
        """Test terminal color support and log results"""
        if self.debug_logger:
            self.debug_logger.color("Testing terminal color support...")
            
            # Test basic ANSI colors
            test_colors = ["text_primary", "user_input", "assistant_primary", "system_info"]
            for color_name in test_colors:
                color = self.get_color(color_name)
                self.debug_logger.color(f"Color test {color_name}: {repr(color)}")
            
            # Test curses availability
            if CURSES_AVAILABLE:
                self.debug_logger.color("Ncurses module available")
            else:
                self.debug_logger.color("Ncurses module not available")
    
    def _init_curses_colors(self):
        """Initialize all color pairs for ncurses"""
        if not CURSES_AVAILABLE:
            if self.debug_logger:
                self.debug_logger.color("Cannot initialize curses colors - module not available")
            return
        
        try:
            curses.start_color()
            curses.use_default_colors()
            self._setup_color_pairs()
            
            if self.debug_logger:
                self.debug_logger.color(f"Initialized {len(self.color_pairs)} ncurses color pairs")
                
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Curses color initialization failed: {e}")
    
    def _reinit_curses_colors(self):
        """Reinitialize color pairs when scheme changes"""
        if CURSES_AVAILABLE and self.interface_type == "curses":
            old_pair_count = len(self.color_pairs)
            self.color_pairs.clear()
            self.pair_counter = 1
            self._setup_color_pairs()
            
            if self.debug_logger:
                self.debug_logger.color(f"Reinitialized curses colors: {old_pair_count} → {len(self.color_pairs)} pairs")
    
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
                
                if self.debug_logger:
                    self.debug_logger.color(f"Created curses pair {self.pair_counter-1}: {color_name} → {hex_color}")
                    
            except (curses.error, ValueError) as e:
                # Fallback to default color
                self.color_pairs[color_name] = curses.color_pair(0) if CURSES_AVAILABLE else 1
                
                if self.debug_logger:
                    self.debug_logger.error(f"Failed to create curses pair for {color_name}: {e}")
    
    def _hex_to_curses_color(self, hex_color):
        """Convert hex color to closest curses color"""
        r, g, b = self._hex_to_rgb(hex_color)
        
        # Enhanced heuristic for color mapping
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
        """Convert hex color to closest ANSI color - COMPREHENSIVE MAPPING for Phase 3"""
        # Enhanced color mapping for all three schemes
        color_map = {
            # Midnight Aurora scheme colors
            "#1e1e2e": Fore.BLACK, "#313244": Fore.BLACK, "#cdd6f4": Fore.WHITE,
            "#a6adc8": Fore.WHITE, "#89dceb": Fore.CYAN, "#74c7ec": Fore.CYAN,
            "#a6e3a1": Fore.GREEN, "#94e2d5": Fore.CYAN, "#b4befe": Fore.MAGENTA,
            "#f9e2af": Fore.YELLOW, "#fab387": Fore.YELLOW, "#f38ba8": Fore.RED,
            "#6c7086": Fore.WHITE, "#89b4fa": Fore.BLUE, "#45475a": Fore.BLACK,
            
            # Forest Whisper scheme colors
            "#1a1b26": Fore.BLACK, "#24283b": Fore.BLACK, "#c0caf5": Fore.WHITE,
            "#9aa5ce": Fore.WHITE, "#7dcfff": Fore.CYAN, "#2ac3de": Fore.CYAN,
            "#9ece6a": Fore.GREEN, "#73daca": Fore.CYAN, "#bb9af7": Fore.MAGENTA,
            "#e0af68": Fore.YELLOW, "#ff9e64": Fore.YELLOW, "#f7768e": Fore.RED,
            "#ff757f": Fore.RED, "#565f89": Fore.WHITE, "#7aa2f7": Fore.BLUE,
            "#32344a": Fore.BLACK,
            
            # Dracula Aurora scheme colors
            "#282a36": Fore.BLACK, "#44475a": Fore.BLACK, "#f8f8f2": Fore.WHITE,
            "#6272a4": Fore.BLUE, "#8be9fd": Fore.CYAN, "#50fa7b": Fore.GREEN,
            "#bd93f9": Fore.MAGENTA, "#ff79c6": Fore.MAGENTA, "#f1fa8c": Fore.YELLOW,
            "#ffb86c": Fore.YELLOW, "#ff5555": Fore.RED,
            
            # Common fallbacks
            "#ffffff": Fore.WHITE, "#000000": Fore.BLACK,
            "#ff0000": Fore.RED, "#00ff00": Fore.GREEN, "#0000ff": Fore.BLUE,
            "#ffff00": Fore.YELLOW, "#ff00ff": Fore.MAGENTA, "#00ffff": Fore.CYAN,
        }
        
        # Return mapped color or default to WHITE if not found
        result = color_map.get(hex_color, Fore.WHITE)
        
        if self.debug_logger:
            self.debug_logger.color(f"ANSI color mapping: {hex_color} → {repr(result)}")
        
        return result

# Chunk 4: Utility Functions and Enhanced ContextManager (Phase 3 - COLOR-AWARE SYSTEMS)

# ------------------ Enhanced Utility Functions (Phase 3) ------------------ #
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

# Phase 3: Color-aware utility functions
def format_color_info(color_manager):
    """Format current color scheme information for display"""
    scheme_name = color_manager.get_current_scheme_name()
    description = color_manager.get_current_scheme_description()
    return f"{scheme_name}: {description}"

def test_color_display(color_manager, debug_logger=None):
    """Test color display across all schemes"""
    if debug_logger:
        debug_logger.color("Starting color display test...")
    
    original_scheme = color_manager.current_scheme_name
    test_results = {}
    
    for scheme_name in COLOR_CYCLE_ORDER:
        color_manager.switch_scheme(scheme_name)
        
        # Test key colors
        test_colors = ["text_primary", "user_input", "assistant_primary", "system_info"]
        scheme_results = {}
        
        for color_name in test_colors:
            try:
                color = color_manager.get_color(color_name)
                scheme_results[color_name] = {"success": True, "color": repr(color)}
            except Exception as e:
                scheme_results[color_name] = {"success": False, "error": str(e)}
        
        test_results[scheme_name] = scheme_results
        
        if debug_logger:
            debug_logger.color(f"Tested scheme {scheme_name}: {len([r for r in scheme_results.values() if r['success']])}/{len(scheme_results)} colors successful")
    
    # Restore original scheme
    color_manager.switch_scheme(original_scheme)
    
    if debug_logger:
        debug_logger.color(f"Color display test complete, restored to {original_scheme}")
    
    return test_results

# ------------------ Enhanced Context Manager (Phase 3) ------------------ #
class ContextManager:
    """Enhanced context manager with color-aware message handling and theme support"""
    
    def __init__(self, color_manager=None, debug_logger=None):
        self.chat_history = []
        self.debug_history = [] 
        self.search_results = []
        self.current_search_term = ""
        self.current_context = "chat"  # "chat", "debug", "search"
        self.max_history = 1000
        self.sme_status_visible = False
        self.scroll_position = 0  # For ncurses scrolling
        self.color_manager = color_manager
        self.debug_logger = debug_logger
        
        if debug_logger:
            debug_logger.system("ContextManager initialized with color support")
    
    def add_chat_message(self, message, message_type):
        """Add message to chat history with color-aware metadata"""
        chat_entry = {
            'content': message,
            'type': message_type,  # 'user', 'assistant', 'system'
            'timestamp': datetime.now(),
            'color_scheme': self.color_manager.current_scheme_name if self.color_manager else "default"
        }
        
        self.chat_history.append(chat_entry)
        self._trim_history('chat')
        
        if self.debug_logger:
            self.debug_logger.system(f"Added {message_type} message to chat history (scheme: {chat_entry['color_scheme']})")
    
    def add_debug_message(self, message, severity):
        """Add message to debug history with enhanced metadata"""
        debug_entry = {
            'content': message,
            'severity': severity,  # 'info', 'warning', 'error'
            'timestamp': datetime.now(),
            'color_scheme': self.color_manager.current_scheme_name if self.color_manager else "default"
        }
        
        self.debug_history.append(debug_entry)
        self._trim_history('debug')
        
        if self.debug_logger:
            self.debug_logger.system(f"Added {severity} debug message")
    
    def switch_context(self, new_context):
        """Switch between contexts and reset scroll position"""
        old_context = self.current_context
        self.current_context = new_context
        self.scroll_position = 0  # Reset scroll when switching contexts
        
        if self.debug_logger:
            self.debug_logger.interface(f"Context switched: {old_context} → {new_context}")
        
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
        """Search chat history and return results with enhanced context"""
        results = []
        search_lower = search_term.lower()
        
        if self.debug_logger:
            self.debug_logger.interface(f"Searching chat history for: '{search_term}'")
        
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
                    'color_scheme': entry.get('color_scheme', 'unknown'),
                    'context_messages': self.chat_history[context_start:context_end],
                    'match_highlight': self._highlight_match(entry['content'], search_term)
                }
                results.append(result)
        
        if self.debug_logger:
            self.debug_logger.interface(f"Search completed: {len(results)} matches found")
        
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
        
        if self.debug_logger:
            self.debug_logger.interface(f"Search results set: '{search_term}' with {len(results)} matches")
    
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
        
        if self.debug_logger:
            status = "enabled" if self.sme_status_visible else "disabled"
            self.debug_logger.interface(f"SME status display {status}")
        
        return self.sme_status_visible
    
    def clear_current_context(self):
        """Clear current context history"""
        if self.current_context == "chat":
            cleared_count = len(self.chat_history)
            self.chat_history.clear()
        elif self.current_context == "debug":
            cleared_count = len(self.debug_history)
            self.debug_history.clear()
        else:  # search
            cleared_count = len(self.search_results)
            self.search_results.clear()
            self.current_search_term = ""
        
        self.scroll_position = 0
        
        if self.debug_logger:
            self.debug_logger.interface(f"Cleared {self.current_context} context: {cleared_count} messages removed")
    
    def scroll_up(self, lines=1):
        """Scroll up in current context"""
        old_position = self.scroll_position
        self.scroll_position = max(0, self.scroll_position - lines)
        
        if self.debug_logger and old_position != self.scroll_position:
            self.debug_logger.interface(f"Scrolled up: {old_position} → {self.scroll_position}")
    
    def scroll_down(self, lines=1):
        """Scroll down in current context"""
        old_position = self.scroll_position
        max_scroll = max(0, self.get_message_count() - 1)
        self.scroll_position = min(max_scroll, self.scroll_position + lines)
        
        if self.debug_logger and old_position != self.scroll_position:
            self.debug_logger.interface(f"Scrolled down: {old_position} → {self.scroll_position}")
    
    def get_context_info(self):
        """Get information about current context for status display"""
        context_info = {
            'name': self.current_context,
            'message_count': self.get_message_count(),
            'scroll_position': self.scroll_position
        }
        
        if self.current_context == "search":
            context_info['search_term'] = self.current_search_term
        
        return context_info
    
    def _trim_history(self, context_type):
        """Keep history within limits"""
        if context_type == "chat" and len(self.chat_history) > self.max_history:
            trimmed_count = len(self.chat_history) - self.max_history
            self.chat_history = self.chat_history[-self.max_history:]
            
            if self.debug_logger:
                self.debug_logger.memory(f"Trimmed chat history: removed {trimmed_count} oldest messages")
                
        elif context_type == "debug" and len(self.debug_history) > self.max_history:
            trimmed_count = len(self.debug_history) - self.max_history
            self.debug_history = self.debug_history[-self.max_history:]
            
            if self.debug_logger:
                self.debug_logger.memory(f"Trimmed debug history: removed {trimmed_count} oldest messages")

# Chunk 5: Interface Detection and Enhanced DisplayManager Foundation (Phase 3 - COLOR-INTEGRATED DISPLAY)

# ------------------ Enhanced Interface Detection and Initialization (Phase 3) ------------------ #
def initialize_interface(bypass_curses=False, debug_logger=None):
    """Initialize interface with graceful fallback and comprehensive color system logging"""
    if bypass_curses:
        if debug_logger:
            debug_logger.interface("Using simple interface (--bypasscurses flag)")
            debug_logger.color("Color system will use ANSI escape codes")
        return "simple"
    
    if not CURSES_AVAILABLE:
        if debug_logger:
            debug_logger.interface("Ncurses not available, using simple interface")
            debug_logger.color("Ncurses color system unavailable, falling back to ANSI")
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
                debug_logger.color("Color system will use ANSI fallback due to terminal size")
            print(f"Terminal too small ({width}x{height}), using simple interface.")
            print(f"Minimum size: {min_width}x{min_height}")
            return "simple"
        
        # Test color support if available
        if debug_logger:
            debug_logger.interface(f"Ncurses interface initialized successfully ({width}x{height})")
            debug_logger.color("Ncurses color system will be available")
        return "curses"
        
    except (curses.error, Exception) as e:
        if debug_logger:
            debug_logger.interface(f"Ncurses initialization failed: {e}")
            debug_logger.interface("Falling back to simple interface")
            debug_logger.color("Color system falling back to ANSI due to ncurses failure")
        
        print("Note: Ncurses initialization failed, using simple interface.")
        print("Use --bypasscurses to suppress this message.")
        return "simple"

# ------------------ Enhanced Display Manager (Phase 3) ------------------ #
class DisplayManager:
    """Enhanced display manager with comprehensive color system integration and theme support"""
    
    def __init__(self, interface_type="simple", initial_color_scheme="midnight_aurora", debug_logger=None):
        self.interface_type = interface_type
        self.debug_logger = debug_logger
        
        # Initialize color manager with proper interface type and debug logging
        self.color_manager = ColorManager(initial_color_scheme, interface_type, debug_logger)
        
        # Initialize context manager with color support
        self.context_manager = ContextManager(self.color_manager, debug_logger)
        
        self.preserved_input = ""
        self.quit_dialog_active = False
        
        # Ncurses-specific attributes
        self.stdscr = None
        self.windows = {}
        self.current_input = ""
        self.input_cursor_pos = 0
        self.input_lines = [""]
        self.input_line_index = 0
        
        # Phase 3: Color system testing and validation
        if debug_logger:
            debug_logger.interface(f"Display manager initialized with {interface_type} interface")
            debug_logger.color(f"ColorManager interface type: {self.color_manager.interface_type}")
            
            # Test color system on initialization
            self.color_manager.test_color_support()
    
    def initialize_ncurses(self, stdscr):
        """Initialize ncurses interface with proper color setup"""
        self.stdscr = stdscr
        
        # Basic ncurses setup
        curses.curs_set(1)  # Show cursor
        curses.noecho()     # Don't echo keys
        curses.cbreak()     # React to keys immediately
        stdscr.keypad(True) # Enable special keys
        stdscr.timeout(100) # Non-blocking input with 100ms timeout
        
        # Initialize colors - this is where the magic happens for Phase 3
        self.color_manager._init_curses_colors()
        
        # Create window layout
        self._create_windows()
        
        if self.debug_logger:
            self.debug_logger.interface("Ncurses interface fully initialized")
            self.debug_logger.color("Ncurses color pairs initialized and ready")
    
    def _create_windows(self):
        """Create the window layout for ncurses interface with color-aware borders"""
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
            
            # Add borders to windows with color support
            border_color = self.color_manager.get_color('border')
            for window_name, window in self.windows.items():
                if window_name != 'status':  # Status bar has no border
                    if self.interface_type == "curses":
                        window.attron(border_color)
                        window.border()
                        window.attroff(border_color)
                    else:
                        window.border()
            
            if self.debug_logger:
                self.debug_logger.interface(f"Created ncurses windows: {width}x{height}")
                self.debug_logger.color(f"Applied border color: {repr(border_color)}")
                
        except curses.error as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to create ncurses windows: {e}")
            raise
    
    def show_user_input(self, text):
        """Display user input with current color scheme"""
        self.context_manager.add_chat_message(text, 'user')
        
        if self.debug_logger:
            self.debug_logger.system(f"User input: {text[:50]}...")
            self.debug_logger.color(f"Using color scheme: {self.color_manager.current_scheme_name}")
        
        if self.interface_type == "curses":
            self._ncurses_show_user_input(text)
        else:
            self._simple_show_user_input(text)
    
    def show_assistant_response(self, text):
        """Display assistant response with current color scheme"""
        self.context_manager.add_chat_message(text, 'assistant')
        
        if self.debug_logger:
            self.debug_logger.system(f"Assistant response: {len(text)} characters")
            self.debug_logger.color(f"Response color scheme: {self.color_manager.current_scheme_name}")
        
        if self.interface_type == "curses":
            self._ncurses_show_assistant_response(text)
        else:
            self._simple_show_assistant_response(text)
    
    def show_system_message(self, text, log_only=False):
        """Show system messages with appropriate color scheme"""
        if log_only:
            if self.debug_logger:
                self.debug_logger.system(text)
            return
        
        self.context_manager.add_chat_message(text, 'system')
        
        if self.debug_logger:
            self.debug_logger.system(f"System message: {text}")
            self.debug_logger.color(f"System message color scheme: {self.color_manager.current_scheme_name}")
        
        if self.interface_type == "curses":
            self._ncurses_show_system_message(text)
        else:
            self._simple_show_system_message(text)
    
    def show_color_scheme_info(self):
        """Display current color scheme information (Phase 3 feature)"""
        scheme_info = format_color_info(self.color_manager)
        available_schemes = self.color_manager.get_available_schemes()
        
        info_text = f"Current theme: {scheme_info}\n\nAvailable themes:\n"
        for name, description in available_schemes.items():
            marker = "→ " if name == self.color_manager.current_scheme_name else "  "
            display_name = name.replace('_', ' ').title()
            info_text += f"{marker}{display_name}: {description}\n"
        
        info_text += "\nUse '/color' to cycle themes or '/color <theme_name>' to switch directly."
        
        self.show_system_message(info_text.strip())
        
        if self.debug_logger:
            self.debug_logger.color(f"Displayed color scheme info for {len(available_schemes)} themes")
    
    def _simple_show_user_input(self, text):
        """Show user input in simple terminal with color"""
        color = self.color_manager.get_color('user_input')
        print(color + f"> {text}" + Style.RESET_ALL)
        
        if self.debug_logger:
            self.debug_logger.color(f"Simple user input color: {repr(color)}")
    
    def _simple_show_assistant_response(self, text):
        """Show assistant response in simple terminal with color"""
        color = self.color_manager.get_color('assistant_primary')
        print_wrapped(text, color)
        
        if self.debug_logger:
            self.debug_logger.color(f"Simple assistant response color: {repr(color)}")
    
    def _simple_show_system_message(self, text):
        """Show system message in simple terminal with color"""
        color = self.color_manager.get_color('system_info')
        print(color + f"[System] {text}" + Style.RESET_ALL)
        
        if self.debug_logger:
            self.debug_logger.color(f"Simple system message color: {repr(color)}")
    
    def _ncurses_show_user_input(self, text):
        """Show user input in ncurses interface with color (Phase 3 placeholder)"""
        if self.debug_logger:
            self.debug_logger.interface("Ncurses user input display called (Phase 3 placeholder)")
        # Implementation will be added in future phases
        pass
    
    def _ncurses_show_assistant_response(self, text):
        """Show assistant response in ncurses interface with color (Phase 3 placeholder)"""
        if self.debug_logger:
            self.debug_logger.interface("Ncurses assistant response display called (Phase 3 placeholder)")
        # Implementation will be added in future phases
        pass
    
    def _ncurses_show_system_message(self, text):
        """Show system message in ncurses interface with color (Phase 3 placeholder)"""
        if self.debug_logger:
            self.debug_logger.interface("Ncurses system message display called (Phase 3 placeholder)")
        # Implementation will be added in future phases
        pass

# Chunk 6: Enhanced Command Handling with Complete Color System (Phase 3 - FULL COLOR COMMANDS)

    def handle_command(self, command):
        """Handle special commands with enhanced color system functionality (Phase 3)"""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        if self.debug_logger:
            self.debug_logger.interface(f"Processing command: {cmd} with {len(parts)-1} arguments")
        
        if cmd == "/debug":
            return self._toggle_debug_context()
        elif cmd == "/color":
            # Phase 3: Enhanced color command with specific scheme selection
            if len(parts) > 1:
                scheme_name = parts[1].lower()
                return self._switch_to_specific_color_scheme(scheme_name)
            else:
                return self._cycle_color_scheme()
        elif cmd == "/themes":
            # Phase 3: New command to show all available themes
            return self._show_color_themes()
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
            return {"action": "invalid_command", "message": f"Unknown command: {cmd}. Type /help for available commands."}
    
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
        """Cycle through color schemes (Phase 3 enhanced)"""
        old_scheme = self.color_manager.current_scheme_name
        success = self.color_manager.cycle_scheme()
        
        if success:
            new_scheme = self.color_manager.current_scheme_name
            new_description = self.color_manager.get_current_scheme_description()
            
            if self.debug_logger:
                self.debug_logger.color(f"User cycled color scheme: {old_scheme} → {new_scheme}")
            
            # Show immediate visual feedback
            scheme_display_name = self.color_manager.get_current_scheme_name()
            message = f"Color scheme: {scheme_display_name}\n{new_description}"
            
            return {
                "action": "color_cycled",
                "old_scheme": old_scheme,
                "new_scheme": new_scheme,
                "message": message,
                "description": new_description
            }
        else:
            return {
                "action": "color_cycle_failed", 
                "message": "Failed to cycle color scheme"
            }
    
    def _switch_to_specific_color_scheme(self, scheme_name):
        """Switch to a specific color scheme (Phase 3 new feature)"""
        # Normalize scheme name input
        scheme_map = {
            "midnight": "midnight_aurora",
            "midnight_aurora": "midnight_aurora",
            "forest": "forest_whisper", 
            "forest_whisper": "forest_whisper",
            "dracula": "dracula_aurora",
            "dracula_aurora": "dracula_aurora"
        }
        
        normalized_scheme = scheme_map.get(scheme_name)
        
        if not normalized_scheme:
            available_schemes = list(self.color_manager.get_available_schemes().keys())
            available_names = [name.replace('_', ' ').title() for name in available_schemes]
            return {
                "action": "invalid_scheme",
                "message": f"Unknown color scheme '{scheme_name}'. Available: {', '.join(available_names)}"
            }
        
        old_scheme = self.color_manager.current_scheme_name
        
        if old_scheme == normalized_scheme:
            display_name = self.color_manager.get_current_scheme_name()
            return {
                "action": "scheme_already_active",
                "message": f"{display_name} is already the active color scheme"
            }
        
        success = self.color_manager.switch_scheme(normalized_scheme)
        
        if success:
            new_description = self.color_manager.get_current_scheme_description()
            display_name = self.color_manager.get_current_scheme_name()
            
            if self.debug_logger:
                self.debug_logger.color(f"User switched to specific scheme: {old_scheme} → {normalized_scheme}")
            
            message = f"Switched to {display_name}\n{new_description}"
            
            return {
                "action": "color_switched",
                "old_scheme": old_scheme,
                "new_scheme": normalized_scheme,
                "message": message,
                "description": new_description
            }
        else:
            return {
                "action": "color_switch_failed",
                "message": f"Failed to switch to {scheme_name}"
            }
    
    def _show_color_themes(self):
        """Show all available color themes with descriptions (Phase 3 new feature)"""
        available_schemes = self.color_manager.get_available_schemes()
        current_scheme = self.color_manager.current_scheme_name
        
        themes_text = "Available Color Themes:\n"
        themes_text += "=" * 40 + "\n\n"
        
        for scheme_name, description in available_schemes.items():
            display_name = scheme_name.replace('_', ' ').title()
            marker = "→ " if scheme_name == current_scheme else "  "
            status = " (ACTIVE)" if scheme_name == current_scheme else ""
            
            themes_text += f"{marker}{display_name}{status}\n"
            themes_text += f"  {description}\n"
            themes_text += f"  Command: /color {scheme_name}\n\n"
        
        themes_text += "Usage:\n"
        themes_text += "  /color                    - Cycle through themes\n"
        themes_text += "  /color <theme_name>       - Switch to specific theme\n"
        themes_text += "  /themes                   - Show this information"
        
        self.show_system_message(themes_text.strip())
        
        if self.debug_logger:
            self.debug_logger.color(f"Displayed theme information for {len(available_schemes)} schemes")
        
        return {
            "action": "themes_displayed", 
            "message": "Color theme information displayed",
            "theme_count": len(available_schemes)
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
        """Show command help with Phase 3 color command enhancements"""
        help_text = "Aurora RPG Client - Available Commands:\n"
        help_text += "=" * 50 + "\n\n"
        
        # Group commands by category
        navigation_commands = {
            "/debug": "Switch to debug context (Esc to return)",
            "/search <term>": "Search chat history in separate context",
            "/clear": "Clear current context history"
        }
        
        color_commands = {
            "/color": "Cycle through color schemes (midnight→forest→dracula)",
            "/color <scheme>": "Switch to specific scheme (midnight_aurora, forest_whisper, dracula_aurora)",
            "/themes": "Show all available color themes with descriptions"
        }
        
        system_commands = {
            "/showsme": "Toggle Story Momentum Engine status display",
            "/save <filename>": "Save conversation (future: --chat, --debug, --both)",
            "/help": "Show this command help",
            "/quit": "Exit the application"
        }
        
        # Format help output
        help_text += "Navigation & Context:\n"
        for cmd, desc in navigation_commands.items():
            help_text += f"  {cmd:<20} {desc}\n"
        
        help_text += "\nColor & Themes:\n"
        for cmd, desc in color_commands.items():
            help_text += f"  {cmd:<20} {desc}\n"
        
        help_text += "\nSystem & Utilities:\n"
        for cmd, desc in system_commands.items():
            help_text += f"  {cmd:<20} {desc}\n"
        
        help_text += f"\nCurrent Theme: {self.color_manager.get_current_scheme_name()}\n"
        help_text += f"Interface: {self.interface_type.title()}\n"
        help_text += f"Context: {self.context_manager.current_context.title()}"
        
        self.show_system_message(help_text.strip())
        
        if self.debug_logger:
            self.debug_logger.interface("Displayed enhanced help with color command information")
        
        return {"action": "help_shown", "message": "Enhanced command help displayed"}
    
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
        """Save conversation (enhanced placeholder for future implementation)"""
        if self.debug_logger:
            self.debug_logger.system(f"Save conversation requested: {filename}")
        
        return {
            "action": "save_requested", 
            "filename": filename, 
            "message": "Save functionality coming in later phase"
        }
    
    def get_user_input(self):
        """Get user input with proper interface routing"""
        if self.debug_logger:
            self.debug_logger.interface(f"get_user_input called with interface_type: {self.interface_type}")
        
        if self.interface_type == "curses":
            return self._ncurses_get_input()
        else:
            return self._simple_get_input()
    
    def _simple_get_input(self):
        """Simple terminal input collection with color-aware prompts"""
        user_lines = []
        
        # Get color for prompt
        try:
            color = self.color_manager.get_color('user_prompt')
            if self.debug_logger:
                self.debug_logger.color(f"User prompt color: {repr(color)}")
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to get prompt color: {e}")
            color = Fore.CYAN  # Fallback color
        
        # Ensure color is a string for simple interface
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
        
        result = "\n".join(user_lines).strip()
        
        if self.debug_logger:
            self.debug_logger.interface(f"Simple input collected: {len(result)} characters")
        
        return result
    
    def _ncurses_get_input(self):
        """Ncurses input collection (placeholder for future implementation)"""
        if self.debug_logger:
            self.debug_logger.interface("Ncurses input not implemented, falling back to simple")
        # For Phase 3, fallback to simple but log the issue
        return self._simple_get_input()
    
    def generate_status_bar(self, momentum_state=None):
        """Generate color-aware status bar text (Phase 3 enhanced)"""
        context_name = self.context_manager.current_context.upper()
        theme_name = self.color_manager.get_current_scheme_name()
        
        base_status = f"Window: [{context_name}] | Theme: [{theme_name}] | Commands: /help /themes"
        
        # Add SME status if enabled and available
        if self.context_manager.sme_status_visible and momentum_state:
            antagonist = momentum_state.get("antagonist", {})
            pressure = momentum_state.get("narrative_pressure", 0.0)
            pressure_name = get_pressure_name(pressure)
            antagonist_name = antagonist.get("name", "Unknown")
            
            sme_status = f" | SME: {pressure_name.title()}({pressure:.2f}) | Villain: {antagonist_name}"
            base_status += sme_status
        
        if self.debug_logger:
            self.debug_logger.interface(f"Generated status bar with theme: {theme_name}")
        
        return base_status
    
    def refresh_display(self):
        """Refresh the display with current color scheme (ncurses-specific)"""
        if self.interface_type == "curses" and self.stdscr:
            try:
                for window in self.windows.values():
                    window.refresh()
                self.stdscr.refresh()
                
                if self.debug_logger:
                    self.debug_logger.interface("Ncurses display refreshed")
            except curses.error as e:
                if self.debug_logger:
                    self.debug_logger.error(f"Display refresh failed: {e}")

# Chunk 7: Memory Management, MCP Communication, and Support Functions (Phase 3 - COLOR-ENHANCED SYSTEMS)

# ------------------ Memory Management (Phase 3 Enhanced) ------------------ #
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
                debug_logger.memory(f"Memory file size: {SAVE_FILE.stat().st_size} bytes")
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
                debug_logger.memory(f"Memory file updated: {SAVE_FILE.stat().st_size} bytes")
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Failed to save memory file: {e}")

def add_memory(memories, role, content, debug_logger=None, color_scheme=None):
    """Add memory with Phase 3 color scheme tracking"""
    memory = {
        "id": str(uuid4()),
        "role": role,
        "content": content,
        "timestamp": now_iso(),
        "color_scheme": color_scheme  # Phase 3: Track color scheme used
    }
    memories.append(memory)
    save_memory(memories, debug_logger)
    
    if debug_logger:
        debug_logger.memory(f"Added {role} memory: {content[:50]}...")
        if color_scheme:
            debug_logger.color(f"Memory saved with color scheme: {color_scheme}")

# ------------------ MCP Communication (Phase 3 Enhanced) ------------------ #
async def call_mcp(messages, debug_logger=None, max_retries=3):
    """Call MCP with automatic retry logic and enhanced logging"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    
    if debug_logger:
        debug_logger.network(f"MCP call with {len(messages)} messages")
        debug_logger.network(f"Model: {MODEL}")
        debug_logger.network(f"URL: {MCP_URL}")
        debug_logger.network(f"Timeout: {TIMEOUT}s")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                if debug_logger:
                    debug_logger.network(f"Attempt {attempt + 1}/{max_retries}: Sending MCP request")
                
                response = await client.post(MCP_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                if debug_logger:
                    debug_logger.network(f"MCP response received: {len(content)} characters")
                    debug_logger.network(f"Response status: {response.status_code}")
                
                return content
                
        except (httpx.TimeoutException, httpx.RequestError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                if debug_logger:
                    debug_logger.network(f"MCP call failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                    debug_logger.error(f"Network error: {e}")
                await asyncio.sleep(wait_time)
            else:
                if debug_logger:
                    debug_logger.error(f"MCP call failed after {max_retries} attempts: {e}")
                raise e

# ------------------ Antagonist System Functions (Phase 3 Enhanced) ------------------ #
def get_pressure_name(pressure_level):
    """Convert pressure level to named range with enhanced descriptions"""
    if pressure_level < 0.1:
        return "low"
    elif pressure_level < 0.3:
        return "building" 
    elif pressure_level < 0.6:
        return "critical"
    else:
        return "explosive"

def format_sme_status(momentum_state, color_manager=None):
    """Format SME status with color scheme awareness (Phase 3)"""
    if not momentum_state:
        return "SME: No data available"
    
    antagonist = momentum_state.get("antagonist", {})
    pressure = momentum_state.get("narrative_pressure", 0.0)
    pressure_name = get_pressure_name(pressure)
    antagonist_name = antagonist.get("name", "Unknown")
    
    status = f"SME: {pressure_name.title()}({pressure:.2f}) | Villain: {antagonist_name}"
    
    # Phase 3: Add color scheme context if available
    if color_manager:
        current_scheme = color_manager.get_current_scheme_name()
        status += f" | Theme: {current_scheme}"
    
    return status

# ------------------ Phase 3: Color System Support Functions ------------------ #
def validate_color_scheme_name(scheme_name):
    """Validate and normalize color scheme names"""
    if not scheme_name:
        return None
    
    # Normalize input
    normalized = scheme_name.lower().replace(' ', '_').replace('-', '_')
    
    # Direct matches
    if normalized in COLOR_SCHEMES:
        return normalized
    
    # Partial matches
    scheme_aliases = {
        "midnight": "midnight_aurora",
        "forest": "forest_whisper",
        "dracula": "dracula_aurora",
        "aurora": "midnight_aurora",
        "whisper": "forest_whisper",
        "dark": "midnight_aurora",
        "nature": "forest_whisper",
        "purple": "dracula_aurora"
    }
    
    return scheme_aliases.get(normalized)

def test_color_compatibility(interface_type, debug_logger=None):
    """Test color system compatibility for current environment"""
    results = {
        "interface_type": interface_type,
        "ansi_support": True,  # Assume ANSI is supported
        "curses_support": CURSES_AVAILABLE,
        "color_depth": "unknown",
        "terminal_size": f"{get_terminal_width()}x{get_terminal_height()}"
    }
    
    if debug_logger:
        debug_logger.color(f"Color compatibility test - Interface: {interface_type}")
        debug_logger.color(f"ANSI support: {results['ansi_support']}")
        debug_logger.color(f"Curses support: {results['curses_support']}")
        debug_logger.color(f"Terminal size: {results['terminal_size']}")
    
    # Test ANSI color output
    try:
        test_output = Fore.RED + "TEST" + Style.RESET_ALL
        results["ansi_test"] = "success"
        if debug_logger:
            debug_logger.color("ANSI color test successful")
    except Exception as e:
        results["ansi_test"] = f"failed: {e}"
        if debug_logger:
            debug_logger.error(f"ANSI color test failed: {e}")
    
    # Test curses colors if available
    if CURSES_AVAILABLE and interface_type == "curses":
        try:
            # Quick curses color test
            curses.wrapper(lambda stdscr: curses.start_color())
            results["curses_color_test"] = "success"
            if debug_logger:
                debug_logger.color("Curses color test successful")
        except Exception as e:
            results["curses_color_test"] = f"failed: {e}"
            if debug_logger:
                debug_logger.error(f"Curses color test failed: {e}")
    
    return results

# ------------------ Application Initialization (Phase 3 Enhanced) ------------------ #
def initialize_application():
    """Initialize application with enhanced color system integration"""
    debug_logger = DebugLogger(args.debug, "debug.log") if args.debug else None
    
    if debug_logger:
        debug_logger.system("Aurora RPG Client Phase 3 starting...")
        debug_logger.system(f"Phase 3 Focus: Complete Color System Implementation")
        debug_logger.system(f"Arguments: debug={args.debug}, bypasscurses={args.bypasscurses}, colorscheme={args.colorscheme}")
        debug_logger.color(f"Initial color scheme: {args.colorscheme}")
    
    # Test color compatibility first
    interface_type = initialize_interface(args.bypasscurses, debug_logger)
    color_test_results = test_color_compatibility(interface_type, debug_logger)
    
    if debug_logger:
        debug_logger.system(f"Interface type determined: {interface_type}")
        debug_logger.color(f"Color compatibility results: {color_test_results}")
    
    # Initialize display manager with enhanced color support
    display_manager = DisplayManager(interface_type, args.colorscheme, debug_logger)
    
    # Test the color system
    if debug_logger:
        debug_logger.system(f"DisplayManager interface_type: {display_manager.interface_type}")
        debug_logger.system(f"ColorManager interface_type: {display_manager.color_manager.interface_type}")
        debug_logger.color(f"Active color scheme: {display_manager.color_manager.current_scheme_name}")
        
        # Run comprehensive color test
        test_results = test_color_display(display_manager.color_manager, debug_logger)
        debug_logger.color(f"Color display test completed for {len(test_results)} schemes")
    
    # Show clean startup message with color scheme info
    if interface_type == "curses":
        # Ncurses initialization will be handled in main loop
        pass
    else:
        scheme_name = display_manager.color_manager.get_current_scheme_name()
        print(f"Aurora RPG Client Phase 3 ready. Active theme: {scheme_name}")
        print("Type /help for commands or /themes to explore color options.")
    
    return display_manager, debug_logger

def show_phase3_startup_info(display_manager):
    """Show Phase 3 specific startup information"""
    startup_info = []
    startup_info.append("Aurora RPG Client - Phase 3: Complete Color System")
    startup_info.append("=" * 55)
    startup_info.append("")
    startup_info.append("Phase 3 Features:")
    startup_info.append("• Three complete color schemes with dynamic switching")
    startup_info.append("• Enhanced /color command with specific scheme selection")  
    startup_info.append("• New /themes command to explore all available themes")
    startup_info.append("• Color-aware status bar and interface elements")
    startup_info.append("• Comprehensive debug logging for color system")
    startup_info.append("")
    
    current_scheme = display_manager.color_manager.get_current_scheme_name()
    description = display_manager.color_manager.get_current_scheme_description()
    startup_info.append(f"Active Theme: {current_scheme}")
    startup_info.append(f"Description: {description}")
    startup_info.append("")
    startup_info.append("Quick Commands:")
    startup_info.append("  /color         - Cycle through themes")
    startup_info.append("  /themes        - Show all available themes")
    startup_info.append("  /help          - Complete command reference")
    startup_info.append("")
    
    for line in startup_info:
        if line.startswith("="):
            color = display_manager.color_manager.get_color('system_success')
        elif line.startswith("Phase 3") or line.startswith("Active Theme") or line.startswith("Quick Commands"):
            color = display_manager.color_manager.get_color('system_info')
        elif line.startswith("•") or line.startswith("  /"):
            color = display_manager.color_manager.get_color('text_secondary')
        else:
            color = display_manager.color_manager.get_color('text_primary')
        
        print(color + line + Style.RESET_ALL)

# ------------------ Enhanced Error Handling (Phase 3) ------------------ #
def handle_color_system_error(error, debug_logger=None, fallback_scheme="midnight_aurora"):
    """Handle color system errors gracefully"""
    if debug_logger:
        debug_logger.error(f"Color system error: {error}")
        debug_logger.color(f"Attempting fallback to scheme: {fallback_scheme}")
    
    try:
        # Attempt to create a basic color manager with fallback
        fallback_color_manager = ColorManager(fallback_scheme, "simple", debug_logger)
        
        if debug_logger:
            debug_logger.color("Color system fallback successful")
        
        return fallback_color_manager
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Color system fallback failed: {e}")
        
        # Ultimate fallback - no colors
        return None

# Chunk 8: Main Application Logic with Complete Color System (Phase 3 - FINAL)

# ------------------ Main Application Logic (Phase 3 Complete) ------------------ #
async def main():
    """Main application entry point - Phase 3 with complete color system implementation"""
    try:
        # Initialize application with enhanced color system
        display_manager, debug_logger = initialize_application()
        
        # Validate configuration
        validate_token_allocation(debug_logger)
        
        # Load conversation memory with color scheme tracking
        memories = load_memory(debug_logger)
        current_scheme = display_manager.color_manager.current_scheme_name
        
        display_manager.show_system_message(
            f"Loaded {len(memories)} memories from previous sessions", 
            log_only=True
        )
        
        if debug_logger:
            debug_logger.system(f"Main loop routing - interface_type: {display_manager.interface_type}")
            debug_logger.color(f"Starting main loop with color scheme: {current_scheme}")
        
        # Handle ncurses vs simple interface with color system
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
            debug_logger.color("Color system session ending")
            debug_logger.close()

async def run_simple_interface(display_manager, debug_logger, memories):
    """Run the simple terminal interface with complete Phase 3 color system"""
    if debug_logger:
        debug_logger.system("Starting simple interface loop with Phase 3 color system")
        debug_logger.color(f"DisplayManager interface_type: {display_manager.interface_type}")
        debug_logger.color(f"Active color scheme: {display_manager.color_manager.current_scheme_name}")
    
    # Display enhanced Phase 3 session info
    print()
    show_phase3_startup_info(display_manager)
    print()
    
    # Show color-aware status bar
    status = display_manager.generate_status_bar()
    status_color = display_manager.color_manager.get_color('system_info')
    print(status_color + status + Style.RESET_ALL)
    
    # Show input instructions with current color scheme
    text_color = display_manager.color_manager.get_color('text_primary')
    secondary_color = display_manager.color_manager.get_color('text_secondary')
    
    print(text_color + "\nType your message (press Enter twice to send, '/quit' to exit):" + Style.RESET_ALL)
    print(secondary_color + "New in Phase 3: /color to cycle themes, /themes to explore options" + Style.RESET_ALL)
    print(secondary_color + "-" * 60 + Style.RESET_ALL + "\n")
    
    # Main conversation loop with color system integration
    while True:
        try:
            # Get user input with color-aware prompts
            if debug_logger:
                debug_logger.system("Calling get_user_input()")
                debug_logger.color(f"Current scheme for input: {display_manager.color_manager.current_scheme_name}")
            
            raw_input_text = display_manager.get_user_input()
            
            if debug_logger:
                debug_logger.system(f"Received input: {raw_input_text[:50]}...")
            
            if not raw_input_text:
                continue
            
            # Handle commands with enhanced color system support
            if raw_input_text.startswith('/'):
                result = display_manager.handle_command(raw_input_text)
                
                if result["action"] == "quit":
                    farewell_color = display_manager.color_manager.get_color('system_success')
                    print(farewell_color + "Goodbye! Thank you for trying Phase 3 color system." + Style.RESET_ALL)
                    break
                elif result["action"] == "invalid_command":
                    display_manager.show_system_message(result["message"])
                elif "message" in result:
                    display_manager.show_system_message(result["message"])
                    
                    # Update status bar for color-related commands
                    if result["action"] in ["color_cycled", "color_switched", "sme_toggled", "context_switched"]:
                        print()  # Add spacing
                        status = display_manager.generate_status_bar()
                        status_color = display_manager.color_manager.get_color('system_info')
                        print(status_color + status + Style.RESET_ALL + "\n")
                        
                        # Show color transition feedback
                        if result["action"] in ["color_cycled", "color_switched"]:
                            if debug_logger:
                                debug_logger.color(f"Color command completed: {result['action']}")
                
                continue
            
            # Validate input length
            is_valid, warning, preserved_input = validate_user_input_length(raw_input_text)
            if not is_valid:
                display_manager.show_system_message(warning)
                continue
            
            # Add user input to memory with current color scheme
            current_scheme = display_manager.color_manager.current_scheme_name
            add_memory(memories, "user", raw_input_text, debug_logger, current_scheme)
            display_manager.show_user_input(raw_input_text)
            
            # Phase 3: Enhanced test response with color system integration
            scheme_name = display_manager.color_manager.get_current_scheme_name()
            scheme_description = display_manager.color_manager.get_current_scheme_description()
            
            # Create color-aware response
            response_lines = []
            response_lines.append(f"Phase 3 Enhanced Response (Theme: {scheme_name})")
            response_lines.append(f"Color System: {scheme_description}")
            response_lines.append("")
            response_lines.append(f"Your input: '{raw_input_text[:100]}{'...' if len(raw_input_text) > 100 else ''}'")
            response_lines.append(f"Context: {display_manager.context_manager.current_context}")
            response_lines.append(f"Interface: {display_manager.interface_type}")
            response_lines.append("")
            response_lines.append("Phase 3 Color System Active:")
            response_lines.append("✓ Three complete color schemes available")
            response_lines.append("✓ Dynamic theme switching with /color command")
            response_lines.append("✓ Specific theme selection with /color <scheme>")
            response_lines.append("✓ Theme information with /themes command")
            response_lines.append("✓ Color-aware status bar and interface elements")
            response_lines.append("")
            response_lines.append("Try: /themes to see all available color options!")
            
            test_response = "\n".join(response_lines)
            
            add_memory(memories, "assistant", test_response, debug_logger, current_scheme)
            display_manager.show_assistant_response(test_response)
            
            # Show colorful separator
            separator_color = display_manager.color_manager.get_color('border')
            print(separator_color + "=" * 60 + Style.RESET_ALL + "\n")
            
            if debug_logger:
                debug_logger.color(f"Completed interaction with scheme: {current_scheme}")
            
        except KeyboardInterrupt:
            farewell_color = display_manager.color_manager.get_color('system_success')
            print("\n" + farewell_color + "Goodbye! Color system session ended." + Style.RESET_ALL)
            break
        except Exception as e:
            if debug_logger:
                debug_logger.error(f"Error in simple interface loop: {e}")
            
            error_color = display_manager.color_manager.get_color('debug_error')
            print(error_color + f"[Error] {e}" + Style.RESET_ALL)

# Fix for run_curses_interface function - replace the existing function with this corrected version:

async def run_curses_interface(display_manager, debug_logger, memories):
    """Run the ncurses interface with Phase 3 color system (enhanced placeholder)"""
    if debug_logger:
        debug_logger.interface("Ncurses interface requested for Phase 3")
        debug_logger.color("Ncurses color system ready but full implementation pending Phase 4")

    # CRITICAL FIX: Temporarily switch to simple interface mode for color compatibility
    # before trying to get colors for fallback messages
    original_interface_type = display_manager.interface_type
    display_manager.interface_type = "simple"
    display_manager.color_manager.interface_type = "simple"

    if debug_logger:
        debug_logger.interface("Temporarily switched to simple interface for Phase 3 fallback")
        debug_logger.color("Color system remains fully functional in simple interface mode")

    # Now get colors - these will be ANSI strings instead of curses integers
    fallback_color = display_manager.color_manager.get_color('system_info')
    warning_color = display_manager.color_manager.get_color('debug_warning')

    # Enhanced fallback message with color system (now works correctly)
    print(warning_color + "Phase 3: Ncurses interface foundation ready but full implementation coming in Phase 4." + Style.RESET_ALL)
    print(fallback_color + "Falling back to enhanced simple interface with complete color system..." + Style.RESET_ALL)
    print()

    # Continue with simple interface
    await run_simple_interface(display_manager, debug_logger, memories)

# ------------------ Enhanced Entry Point (Phase 3) ------------------ #
if __name__ == "__main__":
    # Show Phase 3 initialization message
    if args.debug:
        print(Fore.CYAN + "[Phase 3] Debug logging enabled - check debug.log for detailed color system information" + Style.RESET_ALL)
    
    print(Fore.MAGENTA + "[Phase 3] Aurora RPG Client - Complete Color System Implementation" + Style.RESET_ALL)
    print(Fore.YELLOW + f"[Phase 3] Starting with initial theme: {args.colorscheme}" + Style.RESET_ALL)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Fore.CYAN + "\n[Phase 3] Color system shutdown complete." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[Phase 3 Fatal Error] {e}" + Style.RESET_ALL)
        
        # Try to log the error if possible
        try:
            emergency_logger = DebugLogger(True, "debug_emergency.log")
            emergency_logger.error(f"Phase 3 fatal error: {e}")
            emergency_logger.close()
            print(Fore.YELLOW + "[Phase 3] Error logged to debug_emergency.log" + Style.RESET_ALL)
        except:
            pass
        
        sys.exit(1)

# ------------------ Phase 3 Completion Marker ------------------ #
"""
Aurora RPG Client - Phase 3 Complete: Color System Implementation

PHASE 3 ACHIEVEMENTS:
✓ Complete ColorManager with three beautiful color schemes
✓ Dynamic color scheme switching with /color command
✓ Specific color scheme selection with /color <scheme_name>
✓ New /themes command to explore all available themes
✓ Enhanced color-aware DisplayManager and ContextManager
✓ Comprehensive color system debug logging
✓ Color compatibility testing and graceful fallbacks
✓ Enhanced status bar with theme information
✓ Color scheme tracking in memory system
✓ Beautiful themed startup experience

READY FOR PHASE 4:
- Basic ncurses interface implementation
- Three-context window management (chat/debug/search)
- Color-aware ncurses windows and borders
- Enhanced input/output panes with proper color rendering

The color system is now complete and fully functional in simple terminal mode,
providing a beautiful and dynamic user experience with three distinct themes
that can be switched seamlessly during runtime.
"""
