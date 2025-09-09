#!/usr/bin/env python3
"""
# Chunk 1/3 - Core Classes and Enums

DevName RPG Client - Story Momentum Engine Module (sme.py)

For complete architecture documentation, see genai.txt
Programmatic interconnects: Called by nci.py, provides context to mcp.py
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import re

class StoryArc(Enum):
    """Narrative progression states"""
    SETUP = "setup"
    RISING = "rising_action" 
    CLIMAX = "climax"
    RESOLUTION = "resolution"

class Antagonist:
    """Dynamic story opposition element"""
    
    def __init__(self, name: str, motivation: str, threat_level: float, context: str):
        self.name = name
        self.motivation = motivation
        self.threat_level = threat_level  # 0.0-1.0
        self.context = context
        self.introduction_time = time.time()
        self.active = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "motivation": self.motivation,
            "threat_level": self.threat_level,
            "context": self.context,
            "introduction_time": self.introduction_time,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Antagonist':
        antagonist = cls(
            data["name"],
            data["motivation"], 
            data["threat_level"],
            data["context"]
        )
        antagonist.introduction_time = data.get("introduction_time", time.time())
        antagonist.active = data.get("active", True)
        return antagonist

class StoryMomentumEngine:
    """
    Dynamic narrative pressure and antagonist management system.
    Analyzes conversation for story pacing and provides context enhancement.
    """
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.pressure_level = 0.0  # 0.0-1.0 scale
        self.story_arc = StoryArc.SETUP
        self.current_antagonist: Optional[Antagonist] = None
        self.pressure_history: List[Tuple[float, float]] = []  # (timestamp, pressure)
        self.last_analysis_time = 0.0
        self.user_input_buffer: List[str] = []
        
        # Pressure calculation parameters
        self.pressure_decay_rate = 0.05
        self.pressure_threshold_antagonist = 0.6
        self.pressure_threshold_climax = 0.8
        self.analysis_cooldown = 2.0  # seconds
        
        # Story momentum patterns
        self.momentum_patterns = {
            "conflict": ["fight", "attack", "defend", "battle", "combat", "strike"],
            "exploration": ["examine", "search", "look", "investigate", "explore", "discover"],
            "social": ["talk", "speak", "negotiate", "persuade", "convince", "ask"],
            "mystery": ["strange", "unusual", "mysterious", "hidden", "secret", "whisper"],
            "tension": ["danger", "threat", "fear", "worry", "concern", "risk"],
            "resolution": ["resolve", "solution", "answer", "complete", "finish", "end"]
        }
    
    def _log_debug(self, message: str):
        """Internal debug logging"""
        if self.debug_logger:
            self.debug_logger.log_debug(f"SME: {message}")
    
    def _calculate_pressure_change(self, input_text: str) -> float:
        """Calculate pressure change based on user input patterns"""
        if not input_text.strip():
            return 0.0
        
        text_lower = input_text.lower()
        pressure_delta = 0.0
        
        # Pattern-based pressure calculation
        for pattern_type, keywords in self.momentum_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if pattern_type == "conflict":
                pressure_delta += matches * 0.15
            elif pattern_type == "tension":
                pressure_delta += matches * 0.12
            elif pattern_type == "mystery":
                pressure_delta += matches * 0.08
            elif pattern_type == "exploration":
                pressure_delta += matches * 0.05
            elif pattern_type == "social":
                pressure_delta += matches * 0.03
            elif pattern_type == "resolution":
                pressure_delta -= matches * 0.10
        
        # Length and complexity factors
        word_count = len(text_lower.split())
        if word_count > 20:
            pressure_delta += 0.05
        
        # Question pattern detection
        if "?" in input_text:
            pressure_delta += 0.03
        
        # Exclamation intensity
        exclamation_count = input_text.count("!")
        pressure_delta += min(exclamation_count * 0.02, 0.08)
        
        return min(pressure_delta, 0.3)  # Cap maximum single increase

# Chunk 2/3 - Main Processing Methods and Antagonist Generation

    def process_user_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process user input and update story momentum.
        Called by nci.py when user provides input.
        """
        current_time = time.time()
        
        # Rate limiting for analysis
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return {"status": "rate_limited", "pressure": self.pressure_level}
        
        self.last_analysis_time = current_time
        self.user_input_buffer.append(input_text)
        
        # Keep buffer manageable
        if len(self.user_input_buffer) > 50:
            self.user_input_buffer = self.user_input_buffer[-25:]
        
        # Apply pressure decay
        self._apply_pressure_decay()
        
        # Calculate pressure change
        pressure_change = self._calculate_pressure_change(input_text)
        old_pressure = self.pressure_level
        self.pressure_level = max(0.0, min(1.0, self.pressure_level + pressure_change))
        
        # Record pressure history
        self.pressure_history.append((current_time, self.pressure_level))
        if len(self.pressure_history) > 200:
            self.pressure_history = self.pressure_history[-100:]
        
        # Update story arc
        old_arc = self.story_arc
        self._update_story_arc()
        
        # Antagonist management
        antagonist_introduced = self._manage_antagonist()
        
        self._log_debug(f"Pressure: {old_pressure:.3f} → {self.pressure_level:.3f} (+{pressure_change:.3f})")
        
        return {
            "status": "processed",
            "pressure": self.pressure_level,
            "pressure_change": pressure_change,
            "arc": self.story_arc.value,
            "arc_changed": old_arc != self.story_arc,
            "antagonist_introduced": antagonist_introduced,
            "antagonist_active": self.current_antagonist is not None
        }
    
    def _apply_pressure_decay(self):
        """Apply natural pressure decay over time"""
        current_time = time.time()
        if self.pressure_history:
            last_update = self.pressure_history[-1][0]
            time_delta = current_time - last_update
            decay = self.pressure_decay_rate * (time_delta / 60.0)  # Per minute
            self.pressure_level = max(0.0, self.pressure_level - decay)
    
    def _update_story_arc(self):
        """Update story arc based on pressure level"""
        if self.pressure_level < 0.3:
            self.story_arc = StoryArc.SETUP
        elif self.pressure_level < 0.7:
            self.story_arc = StoryArc.RISING
        elif self.pressure_level < 0.9:
            self.story_arc = StoryArc.CLIMAX
        else:
            self.story_arc = StoryArc.RESOLUTION
    
    def _manage_antagonist(self) -> bool:
        """Manage antagonist introduction and lifecycle"""
        if (self.pressure_level >= self.pressure_threshold_antagonist and 
            self.current_antagonist is None):
            self.current_antagonist = self._generate_antagonist()
            self._log_debug(f"Antagonist introduced: {self.current_antagonist.name}")
            return True
        
        # Deactivate antagonist during resolution
        if (self.story_arc == StoryArc.RESOLUTION and 
            self.current_antagonist and self.current_antagonist.active):
            self.current_antagonist.active = False
            self._log_debug("Antagonist deactivated for resolution")
        
        return False
    
    def _generate_antagonist(self) -> Antagonist:
        """Generate dynamic antagonist using LLM-oriented prompting"""
        # Analyze recent user inputs for context
        recent_inputs = self.user_input_buffer[-10:] if self.user_input_buffer else []
        context_keywords = []
        
        for input_text in recent_inputs:
            words = input_text.lower().split()
            context_keywords.extend(words)
        
        # Determine threat level based on current pressure
        threat_level = min(0.9, self.pressure_level + 0.1)
        
        # Context-driven antagonist generation
        if any(keyword in context_keywords for keyword in ["magic", "spell", "wizard", "arcane"]):
            antagonist_context = "magical_opposition"
            base_name = "Corrupted Mage"
            motivation = "seeks to drain magical essence"
        elif any(keyword in context_keywords for keyword in ["explore", "dungeon", "cave", "ruins"]):
            antagonist_context = "environmental_threat"
            base_name = "Ancient Guardian"
            motivation = "protects forbidden knowledge"
        elif any(keyword in context_keywords for keyword in ["town", "city", "people", "merchant"]):
            antagonist_context = "social_conflict"
            base_name = "Corrupt Official"
            motivation = "maintains oppressive control"
        else:
            # Default context-adaptive antagonist
            antagonist_context = "adaptive_threat"
            base_name = "Shadow Entity"
            motivation = "feeds on narrative tension"
        
        return Antagonist(
            name=base_name,
            motivation=motivation,
            threat_level=threat_level,
            context=antagonist_context
        )
    
    def get_story_context(self) -> Dict[str, Any]:
        """
        Generate story context for MCP module integration.
        Called by mcp.py to enhance prompting.
        """
        context = {
            "pressure_level": round(self.pressure_level, 3),
            "story_arc": self.story_arc.value,
            "narrative_state": self._get_narrative_state_description(),
            "should_introduce_tension": self.pressure_level < 0.4,
            "climax_approaching": self.pressure_level > 0.7,
            "antagonist_present": self.current_antagonist is not None
        }
        
        if self.current_antagonist:
            context["antagonist"] = {
                "name": self.current_antagonist.name,
                "motivation": self.current_antagonist.motivation,
                "threat_level": self.current_antagonist.threat_level,
                "active": self.current_antagonist.active
            }
        
        # Recent pressure trend
        if len(self.pressure_history) >= 3:
            recent_pressures = [p[1] for p in self.pressure_history[-3:]]
            if recent_pressures[-1] > recent_pressures[0]:
                context["pressure_trend"] = "rising"
            elif recent_pressures[-1] < recent_pressures[0]:
                context["pressure_trend"] = "falling"
            else:
                context["pressure_trend"] = "stable"
        else:
            context["pressure_trend"] = "initializing"
        
        return context
    
    def _get_narrative_state_description(self) -> str:
        """Generate narrative state description for context"""
        if self.story_arc == StoryArc.SETUP:
            if self.pressure_level < 0.2:
                return "calm_exploration"
            else:
                return "building_tension"
        elif self.story_arc == StoryArc.RISING:
            return "escalating_conflict"
        elif self.story_arc == StoryArc.CLIMAX:
            return "peak_intensity"
        else:  # RESOLUTION
            return "concluding_action"

# Chunk 3/3 - State Management and Utility Methods

    def reset_story_state(self):
        """Reset story state for new session"""
        self.pressure_level = 0.0
        self.story_arc = StoryArc.SETUP
        self.current_antagonist = None
        self.pressure_history.clear()
        self.user_input_buffer.clear()
        self.last_analysis_time = 0.0
        self._log_debug("Story state reset")
    
    def get_pressure_stats(self) -> Dict[str, Any]:
        """Get pressure statistics for analysis"""
        if not self.pressure_history:
            return {"status": "no_data"}
        
        pressures = [p[1] for p in self.pressure_history]
        timestamps = [p[0] for p in self.pressure_history]
        
        stats = {
            "current_pressure": self.pressure_level,
            "average_pressure": sum(pressures) / len(pressures),
            "max_pressure": max(pressures),
            "min_pressure": min(pressures),
            "pressure_variance": self._calculate_variance(pressures),
            "session_duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
            "total_updates": len(self.pressure_history),
            "current_arc": self.story_arc.value
        }
        
        return stats
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of pressure values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        return sum(squared_diffs) / len(squared_diffs)
    
    def save_state(self, filepath: str) -> bool:
        """Save SME state to file"""
        try:
            state_data = {
                "pressure_level": self.pressure_level,
                "story_arc": self.story_arc.value,
                "pressure_history": self.pressure_history,
                "user_input_buffer": self.user_input_buffer[-10:],  # Save last 10 inputs
                "last_analysis_time": self.last_analysis_time,
                "current_antagonist": self.current_antagonist.to_dict() if self.current_antagonist else None,
                "save_timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self._log_debug(f"State saved to {filepath}")
            return True
            
        except Exception as e:
            self._log_debug(f"State save failed: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load SME state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            self.pressure_level = state_data.get("pressure_level", 0.0)
            self.story_arc = StoryArc(state_data.get("story_arc", "setup"))
            self.pressure_history = state_data.get("pressure_history", [])
            self.user_input_buffer = state_data.get("user_input_buffer", [])
            self.last_analysis_time = state_data.get("last_analysis_time", 0.0)
            
            antagonist_data = state_data.get("current_antagonist")
            if antagonist_data:
                self.current_antagonist = Antagonist.from_dict(antagonist_data)
            else:
                self.current_antagonist = None
            
            self._log_debug(f"State loaded from {filepath}")
            return True
            
        except Exception as e:
            self._log_debug(f"State load failed: {e}")
            return False
    
    def force_pressure_level(self, new_pressure: float):
        """Force specific pressure level (debug/testing use)"""
        self.pressure_level = max(0.0, min(1.0, new_pressure))
        self._update_story_arc()
        self._log_debug(f"Pressure forced to {self.pressure_level}")
    
    def force_antagonist_introduction(self):
        """Force antagonist introduction (debug/testing use)"""
        if self.current_antagonist is None:
            self.current_antagonist = self._generate_antagonist()
            self._log_debug("Antagonist introduction forced")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        return {
            "pressure_level": self.pressure_level,
            "story_arc": self.story_arc.value,
            "antagonist_active": self.current_antagonist is not None,
            "pressure_history_count": len(self.pressure_history),
            "input_buffer_count": len(self.user_input_buffer),
            "last_analysis_time": self.last_analysis_time,
            "pressure_decay_rate": self.pressure_decay_rate,
            "antagonist_threshold": self.pressure_threshold_antagonist,
            "climax_threshold": self.pressure_threshold_climax,
            "analysis_cooldown": self.analysis_cooldown
        }

# Standalone utility functions for testing and analysis
def test_pressure_scenario(scenario: str) -> Dict[str, Any]:
    """Test SME with predefined scenario inputs"""
    sme = StoryMomentumEngine()
    initial_pressure = sme.pressure_level
    
    scenarios = {
        "combat": ["I draw my sword", "I attack the enemy", "The battle intensifies"],
        "exploration": ["I examine the room", "I search for clues", "I investigate further"],
        "social": ["I talk to the guard", "I negotiate peacefully", "I make an offer"],
        "mystery": ["Something feels wrong", "I hear whispers", "Shadows move strangely"]
    }
    
    if scenario not in scenarios:
        return {"error": f"Unknown scenario: {scenario}"}
    
    for input_text in scenarios[scenario]:
        sme.process_user_input(input_text)
    
    return {
        "scenario": scenario,
        "initial_pressure": initial_pressure,
        "final_pressure": sme.pressure_level,
        "pressure_change": sme.pressure_level - initial_pressure,
        "current_arc": sme.story_arc.value,
        "antagonist_present": sme.current_antagonist is not None
    }

def analyze_text_momentum(text: str) -> Dict[str, Any]:
    """Analyze text for story momentum patterns"""
    sme = StoryMomentumEngine()
    
    # Process text and get detailed analysis
    result = sme.process_user_input(text)
    context = sme.get_story_context()
    
    return {
        "input_text": text,
        "momentum_analysis": result,
        "story_context": context,
        "detected_patterns": sme._analyze_patterns_in_text(text)
    }

# Extension method for pattern analysis
StoryMomentumEngine._analyze_patterns_in_text = lambda self, text: {
    pattern_type: [keyword for keyword in keywords if keyword in text.lower()]
    for pattern_type, keywords in self.momentum_patterns.items()
}

if __name__ == "__main__":
    # Basic functionality test
    sme = StoryMomentumEngine()
    
    test_inputs = [
        "I look around the room carefully",
        "I hear strange noises coming from the shadows", 
        "I draw my weapon and prepare for battle",
        "The enemy attacks with fierce intensity!"
    ]
    
    print("SME Test Run:")
    for input_text in test_inputs:
        result = sme.process_user_input(input_text)
        print(f"Input: {input_text}")
        print(f"Pressure: {result['pressure']:.3f} | Arc: {result['arc']}")
        print(f"Context: {sme.get_story_context()['narrative_state']}")
        print("---")
    
    print(f"Final Stats: {sme.get_pressure_stats()}")
