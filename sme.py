# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

# Chunk 1/3 - sme.py - Basic State Management and Data Structures
#!/usr/bin/env python3
"""
DevName RPG Client - Story Momentum Engine (sme.py)
Remodularized for hub-and-spoke architecture - LLM analysis moved to sem.py
"""

import json
import threading
import time
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# =============================================================================
# DATA STRUCTURES AND ENUMS
# =============================================================================

class StoryArc(Enum):
    """Story progression phases"""
    SETUP = "setup"
    RISING = "rising"
    CLIMAX = "climax"
    RESOLUTION = "resolution"

@dataclass
class Antagonist:
    """Antagonist state tracking"""
    name: str = ""
    threat_level: float = 0.0
    active: bool = False
    last_mention: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Antagonist':
        return cls(**data)

class NarrativeTimeTracker:
    """Tracks narrative time progression using basic pattern detection"""
    
    def __init__(self):
        self.total_narrative_seconds = 0.0
        self.exchange_count = 0
        self.sequence_history = []  # (sequence_number, duration, cumulative_time)
        self.lock = threading.Lock()
    
    def detect_duration_from_text(self, text: str) -> float:
        """Basic pattern-based duration detection without LLM analysis"""
        duration = 0.0
        text_lower = text.lower()
        
        # Time-specific patterns (basic regex detection)
        time_patterns = {
            r'\b(\d+)\s*hours?\b': lambda m: float(m.group(1)) * 3600,
            r'\b(\d+)\s*minutes?\b': lambda m: float(m.group(1)) * 60,
            r'\b(\d+)\s*seconds?\b': lambda m: float(m.group(1)),
            r'\bhalf\s+hour\b': lambda m: 1800,
            r'\bquarter\s+hour\b': lambda m: 900,
            r'\ba\s+while\b': lambda m: 300,
            r'\bsome\s+time\b': lambda m: 600,
            r'\ba\s+moment\b': lambda m: 30,
            r'\bbriefly\b': lambda m: 15,
            r'\bquickly\b': lambda m: 5,
        }
        
        for pattern, time_func in time_patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                duration += time_func(match)
        
        # Activity-based duration estimates (basic heuristics)
        activity_durations = {
            r'\b(examine|search|investigate|study)\b': 120,  # 2 minutes
            r'\b(travel|walk|journey|move)\b': 600,          # 10 minutes
            r'\b(rest|sleep|wait)\b': 1800,                  # 30 minutes
            r'\b(fight|battle|combat)\b': 180,               # 3 minutes
            r'\b(talk|speak|discuss|converse)\b': 300,       # 5 minutes
        }
        
        for pattern, base_duration in activity_durations.items():
            if re.search(pattern, text_lower):
                duration += base_duration
                break  # Only count one primary activity
        
        # Default duration if no patterns found
        if duration == 0.0:
            duration = 30.0  # 30 seconds default
        
        return duration
    
    def add_exchange(self, input_text: str, sequence_number: int) -> float:
        """Add exchange and return narrative duration"""
        with self.lock:
            duration = self.detect_duration_from_text(input_text)
            self.total_narrative_seconds += duration
            self.exchange_count += 1
            
            # Store sequence history
            self.sequence_history.append((sequence_number, duration, self.total_narrative_seconds))
            
            # Keep only recent history (last 50 exchanges)
            if len(self.sequence_history) > 50:
                self.sequence_history = self.sequence_history[-50:]
            
            return duration

    def get_stats(self) -> Dict[str, Any]:
        """Get narrative time statistics"""
        with self.lock:
            # Format total time for display
            total_seconds = self.total_narrative_seconds
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)

            if hours > 0:
                time_formatted = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_formatted = f"{minutes}m {seconds}s"
            else:
                time_formatted = f"{seconds}s"

            return {
                "total_narrative_seconds": self.total_narrative_seconds,
                "narrative_time_formatted": time_formatted,
                "exchange_count": self.exchange_count,
                "recent_sequences": len(self.sequence_history),
                "average_duration": (self.total_narrative_seconds / max(1, self.exchange_count))
            }

# Chunk 2/3 - sme.py - Core StoryMomentumEngine Class
class StoryMomentumEngine:
    """
    Story state management with basic pattern detection.
    Semantic analysis moved to sem.py - this handles only basic state tracking.
    """
    
    def __init__(self, debug_logger=None, orchestrator_callback=None):
        self.debug_logger = debug_logger
        self.orchestrator_callback = orchestrator_callback
        
        # Core state tracking
        self.pressure_level = 0.0
        self.escalation_count = 0
        self.base_pressure_floor = 0.0
        self.last_analysis_count = 0
        self.last_pressure_decay_sequence = 0
        
        # Story progression
        self.story_arc = StoryArc.SETUP
        self.current_antagonist = None
        self.pressure_history = []
        
        # Narrative time tracking
        self.narrative_tracker = NarrativeTimeTracker()
        
        # Thread safety
        self.lock = threading.Lock()
    
    def process_user_input(self, input_text: str, sequence_number: int = 0) -> Dict[str, Any]:
        """Process user input and update state using basic pattern detection"""
        with self.lock:
            # Add narrative time
            duration = self.narrative_tracker.add_exchange(input_text, sequence_number)
            
            # Basic pressure adjustment using pattern detection
            pressure_change = self._calculate_pressure_change(input_text)
            self.pressure_level = max(0.0, min(1.0, self.pressure_level + pressure_change))
            
            # Update story arc based on pressure trends
            self._update_story_arc()
            
            # Update antagonist state if mentioned
            self._update_antagonist_state(input_text)
            
            # Store pressure history
            self.pressure_history.append({
                "sequence": sequence_number,
                "pressure": self.pressure_level,
                "duration": duration,
                "timestamp": time.time()
            })
            
            # Keep only recent history
            if len(self.pressure_history) > 100:
                self.pressure_history = self.pressure_history[-100:]
            
            self._log_debug(f"Processed input: pressure={self.pressure_level:.3f}, duration={duration:.1f}s")
            
            return {
                "pressure": self.pressure_level,
                "narrative_duration": duration,
                "total_narrative_time": self.narrative_tracker.total_narrative_seconds,
                "story_arc": self.story_arc.value,
                "antagonist_active": self.current_antagonist.active if self.current_antagonist else False
            }
    
    def _calculate_pressure_change(self, text: str) -> float:
        """Calculate pressure change using basic pattern detection"""
        text_lower = text.lower()
        pressure_change = 0.0
        
        # Conflict/tension patterns (increase pressure)
        tension_patterns = [
            (r'\b(attack|fight|battle|combat|danger|threat)\b', 0.15),
            (r'\b(enemy|hostile|aggressive|violent)\b', 0.12),
            (r'\b(afraid|fear|scared|terror|panic)\b', 0.10),
            (r'\b(chase|pursue|hunt|escape|flee)\b', 0.08),
            (r'\b(angry|rage|fury|mad)\b', 0.06),
        ]
        
        for pattern, weight in tension_patterns:
            if re.search(pattern, text_lower):
                pressure_change += weight
        
        # Resolution/calm patterns (decrease pressure)
        calm_patterns = [
            (r'\b(rest|sleep|peaceful|calm|quiet)\b', -0.08),
            (r'\b(safe|secure|protected|relief)\b', -0.10),
            (r'\b(victory|win|succeed|triumph)\b', -0.15),
            (r'\b(friend|ally|help|support)\b', -0.05),
            (r'\b(heal|recover|restore)\b', -0.06),
        ]
        
        for pattern, weight in calm_patterns:
            if re.search(pattern, text_lower):
                pressure_change += weight
        
        # Natural decay over narrative time
        if pressure_change == 0.0:
            pressure_change = -0.02  # Slight natural decay
        
        return pressure_change
    
    def _update_story_arc(self) -> None:
        """Update story arc based on pressure level"""
        if self.pressure_level < 0.2:
            self.story_arc = StoryArc.SETUP
        elif self.pressure_level < 0.6:
            self.story_arc = StoryArc.RISING
        elif self.pressure_level < 0.9:
            self.story_arc = StoryArc.CLIMAX
        else:
            self.story_arc = StoryArc.RESOLUTION
    
    def _update_antagonist_state(self, text: str) -> None:
        """Update antagonist state based on mentions in text"""
        text_lower = text.lower()
        
        # Look for antagonist mentions
        antagonist_patterns = [
            r'\b(enemy|foe|villain|monster|beast|dragon|demon|orc|goblin)\b',
            r'\b(bandit|thief|assassin|warrior|guard|soldier)\b',
            r'\b(dark|shadow|evil|corrupt|twisted)\b.*\b(lord|king|master|mage)\b'
        ]
        
        antagonist_found = False
        for pattern in antagonist_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                antagonist_found = True
                # Create or update antagonist
                if not self.current_antagonist:
                    self.current_antagonist = Antagonist()
                
                self.current_antagonist.active = True
                self.current_antagonist.last_mention = text[:100]  # First 100 chars
                self.current_antagonist.threat_level = min(1.0, self.current_antagonist.threat_level + 0.1)
                
                if not self.current_antagonist.name:
                    # Try to extract a name from the match
                    self.current_antagonist.name = matches[0] if matches else "Unknown Threat"
                break
        
        # Decay antagonist presence if not mentioned
        if not antagonist_found and self.current_antagonist:
            self.current_antagonist.threat_level = max(0.0, self.current_antagonist.threat_level - 0.05)
            if self.current_antagonist.threat_level <= 0.1:
                self.current_antagonist.active = False

# Chunk 3/3 - sme.py - State Management Interface and Utilities
    
    # =============================================================================
    # STATE MANAGEMENT INTERFACE
    # =============================================================================
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current momentum state for orchestrator"""
        with self.lock:
            return {
                "narrative_pressure": self.pressure_level,
                "pressure_source": self._get_pressure_source(),
                "manifestation_type": self._get_manifestation_type(),
                "escalation_count": self.escalation_count,
                "base_pressure_floor": self.base_pressure_floor,
                "last_analysis_count": self.last_analysis_count,
                "antagonist": self.current_antagonist.to_dict() if self.current_antagonist else None,
                "story_arc": self.story_arc.value,
                "narrative_time_stats": self.narrative_tracker.get_stats()
            }
    
    def get_story_context(self) -> Dict[str, Any]:
        """Generate story context for orchestrator (replaces MCP integration)"""
        narrative_stats = self.narrative_tracker.get_stats()
        
        with self.lock:
            context = {
                "pressure_level": round(self.pressure_level, 3),
                "story_arc": self.story_arc.value,
                "narrative_state": self._get_narrative_state_description(),
                "should_introduce_tension": self._should_introduce_tension(),
                "current_narrative_time": narrative_stats["narrative_time_formatted"],
                "total_exchanges": narrative_stats["exchange_count"],
                "antagonist_status": self._get_antagonist_status()
            }
        
        return context
    
    def get_pressure_stats(self) -> Dict[str, Any]:
        """Get pressure statistics for debugging"""
        with self.lock:
            return {
                "current_pressure": self.pressure_level,
                "story_arc": self.story_arc.value,
                "escalation_count": self.escalation_count,
                "base_pressure_floor": self.base_pressure_floor,
                "pressure_history_length": len(self.pressure_history),
                "narrative_time": self.narrative_tracker.get_stats()
            }
    
    def save_state(self) -> Dict[str, Any]:
        """Save current state to dict for persistence"""
        with self.lock:
            state_data = {
                "pressure_level": self.pressure_level,
                "escalation_count": self.escalation_count,
                "base_pressure_floor": self.base_pressure_floor,
                "last_analysis_count": self.last_analysis_count,
                "last_pressure_decay_sequence": self.last_pressure_decay_sequence,
                "story_arc": self.story_arc.value,
                "antagonist": self.current_antagonist.to_dict() if self.current_antagonist else None,
                "pressure_history": self.pressure_history[-10:],  # Save recent history
                "narrative_time_total": self.narrative_tracker.total_narrative_seconds,
                "narrative_exchange_count": self.narrative_tracker.exchange_count
            }
        
        self._log_debug("SME state saved")
        return state_data
    
    def load_state(self, state_data: Dict[str, Any]) -> bool:
        """Load state from dict"""
        try:
            with self.lock:
                # Load basic state
                self.pressure_level = max(0.0, min(1.0, state_data.get("pressure_level", 0.0)))
                self.escalation_count = max(0, state_data.get("escalation_count", 0))
                self.base_pressure_floor = max(0.0, min(1.0, state_data.get("base_pressure_floor", 0.0)))
                self.last_analysis_count = max(0, state_data.get("last_analysis_count", 0))
                self.last_pressure_decay_sequence = max(0, state_data.get("last_pressure_decay_sequence", 0))
                
                # Load narrative time tracking
                self.narrative_tracker.total_narrative_seconds = state_data.get("narrative_time_total", 0.0)
                self.narrative_tracker.exchange_count = state_data.get("narrative_exchange_count", 0)
                
                # Load story arc
                arc_value = state_data.get("story_arc", "setup")
                try:
                    self.story_arc = StoryArc(arc_value)
                except ValueError:
                    self.story_arc = StoryArc.SETUP
                
                # Load antagonist
                antagonist_data = state_data.get("antagonist")
                if antagonist_data:
                    self.current_antagonist = Antagonist.from_dict(antagonist_data)
                else:
                    self.current_antagonist = None
                
                # Load pressure history
                pressure_history = state_data.get("pressure_history", [])
                if isinstance(pressure_history, list):
                    self.pressure_history = pressure_history
            
            self._log_debug("SME state loaded successfully")
            return True
            
        except Exception as e:
            self._log_debug(f"Failed to load SME state: {e}")
            return False
    
    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================
    
    def _get_pressure_source(self) -> str:
        """Determine current pressure source"""
        if self.current_antagonist and self.current_antagonist.active:
            return "antagonist"
        elif self.pressure_level > 0.5:
            return "environment"
        else:
            return "exploration"
    
    def _get_manifestation_type(self) -> str:
        """Determine how momentum is currently manifesting"""
        if self.story_arc == StoryArc.SETUP:
            return "exploration"
        elif self.story_arc == StoryArc.RISING:
            return "tension"
        elif self.story_arc == StoryArc.CLIMAX:
            return "conflict"
        else:
            return "resolution"
    
    def _get_narrative_state_description(self) -> str:
        """Get human-readable narrative state"""
        pressure_desc = "low" if self.pressure_level < 0.3 else "medium" if self.pressure_level < 0.7 else "high"
        return f"{pressure_desc} tension in {self.story_arc.value} phase"
    
    def _should_introduce_tension(self) -> bool:
        """Basic heuristic for tension introduction"""
        return self.pressure_level < 0.3 and self.story_arc == StoryArc.SETUP
    
    def _get_antagonist_status(self) -> str:
        """Get antagonist status description"""
        if not self.current_antagonist:
            return "none"
        elif self.current_antagonist.active:
            return f"active ({self.current_antagonist.name})"
        else:
            return f"inactive ({self.current_antagonist.name})"
    
    def _log_debug(self, message: str) -> None:
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(f"SME: {message}")

# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Basic functionality test
    sme = StoryMomentumEngine()
    
    test_inputs = [
        "I look around the room carefully",
        "I hear strange noises coming from the shadows", 
        "I draw my weapon and prepare for battle",
        "The enemy attacks with fierce intensity!",
        "After some time, we continue our journey",
        "I rest peacefully by the campfire"
    ]
    
    print("SME Test Run (Simplified):")
    for i, input_text in enumerate(test_inputs):
        result = sme.process_user_input(input_text, i)
        print(f"Input: {input_text}")
        print(f"Pressure: {result['pressure']:.3f} | Arc: {result['story_arc']} | Duration: {result['narrative_duration']:.1f}s")
        print("---")
    
    print(f"Final Stats: {sme.get_pressure_stats()}")
    print(f"Story Context: {sme.get_story_context()}")
