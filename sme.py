# Chunk 1/1 - sme.py - Story Momentum Engine with Complete LLM Analysis
#!/usr/bin/env python3

import json
import time
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import re
import httpx

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
        self.commitment_level = "testing"  # testing → engaged → desperate → cornered
        self.resources_available = []
        self.resources_lost = []
        self.personality_traits = []
        self.background = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "motivation": self.motivation,
            "threat_level": self.threat_level,
            "context": self.context,
            "introduction_time": self.introduction_time,
            "active": self.active,
            "commitment_level": self.commitment_level,
            "resources_available": self.resources_available,
            "resources_lost": self.resources_lost,
            "personality_traits": self.personality_traits,
            "background": self.background
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
        antagonist.commitment_level = data.get("commitment_level", "testing")
        antagonist.resources_available = data.get("resources_available", [])
        antagonist.resources_lost = data.get("resources_lost", [])
        antagonist.personality_traits = data.get("personality_traits", [])
        antagonist.background = data.get("background", "")
        return antagonist

class StoryMomentumEngine:
    """
    Dynamic narrative pressure and antagonist management system with complete LLM analysis.
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
        
        # Analysis cycle tracking (every 15 messages)
        self.last_analysis_count = 0
        self.escalation_count = 0
        self.base_pressure_floor = 0.0
        
        # Pressure calculation parameters
        self.pressure_decay_rate = 0.05
        self.pressure_threshold_antagonist = 0.6
        self.pressure_threshold_climax = 0.8
        self.analysis_cooldown = 2.0  # seconds
        
        # MCP configuration for LLM calls
        self.mcp_config = {
            "server_url": "http://127.0.0.1:3456/chat",
            "model": "qwen2.5:14b-instruct-q4_k_m",
            "timeout": 300
        }
        
        # Story momentum patterns (legacy pattern matching for immediate feedback)
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
            self.debug_logger.debug(f"SME: {message}", "SME")
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make LLM request for momentum analysis using working MCP format"""
        try:
            async with httpx.AsyncClient(timeout=self.mcp_config.get("timeout", 30)) as client:
                payload = {
                    "model": self.mcp_config["model"],
                    "messages": messages,
                    "stream": False
                }

                response = await client.post(self.mcp_config["server_url"], json=payload)
                response.raise_for_status()

                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"SME LLM call failed: {e}")
            return None
    
    def _calculate_pressure_change(self, input_text: str) -> float:
        """Calculate immediate pressure change based on user input patterns (legacy system)"""
        if not input_text.strip():
            return 0.0
        
        text_lower = input_text.lower()
        pressure_delta = 0.0
        
        # Pattern-based pressure calculation for immediate feedback
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

    def process_user_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process user input and update story momentum.
        Provides immediate feedback while deferring LLM analysis to 15-message cycles.
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
        
        # Calculate immediate pressure change using legacy patterns
        pressure_change = self._calculate_pressure_change(input_text)
        old_pressure = self.pressure_level
        
        # Apply pressure floor ratcheting
        effective_floor = max(self.base_pressure_floor, 0.0)
        self.pressure_level = max(effective_floor, min(1.0, self.pressure_level + pressure_change))
        
        # Record pressure history
        self.pressure_history.append((current_time, self.pressure_level))
        if len(self.pressure_history) > 200:
            self.pressure_history = self.pressure_history[-100:]
        
        # Update story arc
        old_arc = self.story_arc
        self._update_story_arc()
        
        # Antagonist management (basic threshold check)
        antagonist_introduced = self._manage_antagonist_threshold()
        
        self._log_debug(f"Pressure: {old_pressure:.3f} → {self.pressure_level:.3f} (+{pressure_change:.3f})")
        
        return {
            "status": "processed",
            "pressure": self.pressure_level,
            "pressure_change": pressure_change,
            "arc": self.story_arc.value,
            "arc_changed": old_arc != self.story_arc,
            "antagonist_introduced": antagonist_introduced,
            "antagonist_active": self.current_antagonist is not None,
            "needs_llm_analysis": False  # Will be determined by message count
        }
    
    def _apply_pressure_decay(self):
        """Apply natural pressure decay over time"""
        current_time = time.time()
        if self.pressure_history:
            last_update = self.pressure_history[-1][0]
            time_delta = current_time - last_update
            decay = self.pressure_decay_rate * (time_delta / 60.0)  # Per minute
            floor_pressure = max(self.base_pressure_floor, 0.0)
            self.pressure_level = max(floor_pressure, self.pressure_level - decay)
    
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
    
    def _manage_antagonist_threshold(self) -> bool:
        """Basic antagonist introduction based on pressure threshold"""
        if (self.pressure_level >= self.pressure_threshold_antagonist and 
            self.current_antagonist is None):
            # Create basic antagonist - will be enhanced by LLM analysis
            self.current_antagonist = self._create_basic_antagonist()
            self._log_debug(f"Basic antagonist introduced: {self.current_antagonist.name}")
            return True
        
        # Deactivate antagonist during resolution
        if (self.story_arc == StoryArc.RESOLUTION and 
            self.current_antagonist and self.current_antagonist.active):
            self.current_antagonist.active = False
            self._log_debug("Antagonist deactivated for resolution")
        
        return False
    
    def _create_basic_antagonist(self) -> Antagonist:
        """Create basic antagonist for immediate use (enhanced later by LLM)"""
        antagonist = Antagonist(
            name="Shadow Entity",
            motivation="opposes the player's progress",
            threat_level=min(0.9, self.pressure_level + 0.1),
            context="adaptive_threat"
        )
        antagonist.commitment_level = "testing"
        antagonist.resources_available = ["stealth", "cunning", "persistence"]
        antagonist.personality_traits = ["mysterious", "adaptive", "persistent"]
        return antagonist

    def should_analyze_momentum(self, total_message_count: int) -> bool:
        """Check if momentum analysis should be triggered (every 15 messages)"""
        if total_message_count < 15:
            return False  # Grace period for initial conversation
        
        return total_message_count - self.last_analysis_count >= 15
    
    def prepare_momentum_analysis_context(self, conversation_messages: List[Dict[str, Any]], max_tokens: int = 6000) -> Tuple[List[Dict[str, Any]], int]:
        """Prepare context for momentum analysis within allocated budget"""
        # Reserve 25% for analysis prompt overhead
        analysis_overhead = max_tokens // 4
        available_tokens = max_tokens - analysis_overhead
        
        # Start with recent messages and work backwards
        context_messages = []
        total_tokens = 0
        
        for message in reversed(conversation_messages):
            content = message.get("content", "")
            message_tokens = len(content) // 4  # Conservative token estimation
            
            if total_tokens + message_tokens <= available_tokens:
                context_messages.insert(0, message)
                total_tokens += message_tokens
            else:
                break
        
        self._log_debug(f"Momentum analysis context: {len(context_messages)} messages, {total_tokens} tokens")
        return context_messages, total_tokens
    
    async def analyze_momentum(self, conversation_messages: List[Dict[str, Any]], total_message_count: int, is_first_analysis: bool = False) -> Dict[str, Any]:
        """
        Unified momentum analysis function that handles both first-time and regular analysis.
        """
        
        # 1. Prepare context within token budget
        context_messages, context_tokens = self.prepare_momentum_analysis_context(
            conversation_messages, max_tokens=6000
        )
        
        # 2. Handle antagonist generation/validation
        if is_first_analysis or not self.current_antagonist or not self.validate_antagonist_quality(self.current_antagonist):
            self._log_debug("Generating/enhancing antagonist for momentum analysis...")
            antagonist = await self.generate_antagonist(context_messages)
            if antagonist:
                self.current_antagonist = antagonist
        
        # 3. Resource loss analysis
        conversation_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in context_messages[-10:]  # Last 10 messages
        ])
        
        events_occurred = await self.analyze_resource_loss(conversation_text, self.current_antagonist)
        
        # 4. Calculate pressure floor ratcheting
        new_pressure_floor = self.calculate_pressure_floor_ratchet(events_occurred.get("events", []))
        
        # 5. Main momentum analysis
        current_state = self.get_current_state()
        momentum_prompt = self._create_momentum_analysis_prompt(current_state, context_messages, events_occurred, new_pressure_floor)
        
        self._log_debug(f"Running momentum analysis with {len(context_messages)} context messages")
        
        # 6. Execute analysis with error handling
        try:
            response = await self._call_llm([{"role": "system", "content": momentum_prompt}])
            if response:
                analysis_result = self._parse_momentum_response_robust(response)
                if analysis_result:
                    # Update state with analysis results
                    self._update_state_from_analysis(analysis_result, new_pressure_floor, total_message_count)
                    self._log_debug(f"Momentum analysis complete. Pressure: {self.pressure_level:.2f}")
                    return analysis_result
        
        except Exception as e:
            self._log_debug(f"Momentum analysis failed: {e}")
        
        # Return safe updated state on failure
        safe_state = current_state.copy()
        safe_state["last_analysis_count"] = total_message_count
        safe_state["base_pressure_floor"] = new_pressure_floor
        return safe_state
    
    def _create_momentum_analysis_prompt(self, current_state: Dict[str, Any], context_messages: List[Dict[str, Any]], events_occurred: Dict[str, Any], new_pressure_floor: float) -> str:
        """Create comprehensive momentum analysis prompt"""
        
        conversation_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')[:300]}"  # Truncate for efficiency
            for m in context_messages
        ])
        
        antagonist_info = "None"
        if self.current_antagonist:
            antagonist_info = f"{self.current_antagonist.name} - {self.current_antagonist.motivation} (commitment: {self.current_antagonist.commitment_level})"
        
        return f"""You are analyzing story momentum in an ongoing RPG narrative. Based on the conversation 
and current momentum state, provide updated momentum metrics.

Current Momentum State:
- Narrative Pressure: {self.pressure_level:.2f}
- Story Arc: {self.story_arc.value}
- Antagonist: {antagonist_info}
- Escalation Count: {self.escalation_count}
- Pressure Floor: {self.base_pressure_floor:.2f} → {new_pressure_floor:.2f}

Recent Events Detected: {events_occurred}

Recent Conversation:
{conversation_text}

Analyze and provide updated momentum state:

1. How has narrative pressure changed? (0.0-1.0 scale, considering floor of {new_pressure_floor:.2f})
2. What is the pressure source? (antagonist/environment/social/discovery)
3. How is momentum manifesting? (exploration/tension/conflict/resolution)
4. What is the player's behavioral pattern? (aggressive/cautious/avoidant/diplomatic)
5. How should the antagonist respond given their commitment level?
6. Should the antagonist's commitment level change? (testing/engaged/desperate/cornered)

Return JSON format:
{{
  "narrative_pressure": 0.0-1.0,
  "pressure_source": "antagonist|environment|social|discovery",
  "manifestation_type": "exploration|tension|conflict|resolution",
  "player_behavior": "aggressive|cautious|avoidant|diplomatic",
  "antagonist_response": "description of how antagonist should respond",
  "commitment_change": "testing|engaged|desperate|cornered|no_change",
  "escalation_events": ["event1", "event2"],
  "pressure_reasoning": "explanation of pressure changes"
}}"""
    
    def _parse_momentum_response_robust(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM momentum analysis response with 5-strategy defensive handling"""
        
        # Strategy 1: Direct JSON parsing
        try:
            data = json.loads(response.strip())
            if self._validate_momentum_data(data):
                return self._inject_momentum_defaults(data)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Substring extraction
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                if self._validate_momentum_data(data):
                    return self._inject_momentum_defaults(data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 3: Field validation and pattern extraction
        try:
            extracted_data = {}
            
            # Extract pressure using regex
            import re
            pressure_match = re.search(r'"?narrative_pressure"?\s*:\s*([0-9.]+)', response)
            if pressure_match:
                extracted_data["narrative_pressure"] = float(pressure_match.group(1))
            
            # Extract source
            source_match = re.search(r'"?pressure_source"?\s*:\s*"([^"]+)"', response)
            if source_match:
                extracted_data["pressure_source"] = source_match.group(1)
            
            # Extract manifestation
            manifest_match = re.search(r'"?manifestation_type"?\s*:\s*"([^"]+)"', response)
            if manifest_match:
                extracted_data["manifestation_type"] = manifest_match.group(1)
            
            if extracted_data:
                return self._inject_momentum_defaults(extracted_data)
        except:
            pass
        
        # Strategy 4: Default injection based on keywords
        try:
            response_lower = response.lower()
            extracted_data = {}
            
            # Infer pressure from keywords
            if "high" in response_lower or "intense" in response_lower:
                extracted_data["narrative_pressure"] = 0.7
            elif "moderate" in response_lower or "medium" in response_lower:
                extracted_data["narrative_pressure"] = 0.5
            elif "low" in response_lower or "calm" in response_lower:
                extracted_data["narrative_pressure"] = 0.3
            
            # Infer source from keywords
            if "antagonist" in response_lower:
                extracted_data["pressure_source"] = "antagonist"
            elif "environment" in response_lower:
                extracted_data["pressure_source"] = "environment"
            elif "social" in response_lower:
                extracted_data["pressure_source"] = "social"
            elif "discovery" in response_lower:
                extracted_data["pressure_source"] = "discovery"
            
            if extracted_data:
                return self._inject_momentum_defaults(extracted_data)
        except:
            pass
        
        # Strategy 5: Complete fallback
        return self._create_fallback_momentum_state()
    
    def _validate_momentum_data(self, data: Dict[str, Any]) -> bool:
        """Validate that momentum analysis data has core required fields"""
        if not isinstance(data, dict):
            return False
        
        # Must have at least narrative_pressure
        if "narrative_pressure" not in data:
            return False
        
        pressure = data["narrative_pressure"]
        if not isinstance(pressure, (int, float)) or pressure < 0 or pressure > 1:
            return False
        
        return True
    
    def _inject_momentum_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject missing fields with sensible defaults"""
        defaults = {
            "narrative_pressure": self.pressure_level,
            "pressure_source": self._get_pressure_source(),
            "manifestation_type": self._get_manifestation_type(),
            "player_behavior": "cautious",
            "antagonist_response": "maintaining current strategy",
            "commitment_change": "no_change",
            "escalation_events": [],
            "pressure_reasoning": "analysis completed"
        }
        
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        # Validate pressure value
        pressure = data.get("narrative_pressure", self.pressure_level)
        if not isinstance(pressure, (int, float)) or pressure < 0 or pressure > 1:
            data["narrative_pressure"] = self.pressure_level
        
        return data
    
    def _create_fallback_momentum_state(self) -> Dict[str, Any]:
        """Create complete fallback momentum state"""
        return {
            "narrative_pressure": self.pressure_level,
            "pressure_source": self._get_pressure_source(),
            "manifestation_type": self._get_manifestation_type(),
            "player_behavior": "cautious",
            "antagonist_response": "maintaining current strategy",
            "commitment_change": "no_change",
            "escalation_events": [],
            "pressure_reasoning": "fallback analysis - LLM response parsing failed"
        }
    
    def _update_state_from_analysis(self, analysis: Dict[str, Any], new_pressure_floor: float, total_message_count: int):
        """Update internal state from LLM analysis results"""
        # Update pressure with floor constraint
        new_pressure = analysis.get("narrative_pressure", self.pressure_level)
        self.pressure_level = max(new_pressure, new_pressure_floor)
        
        # Update pressure floor
        self.base_pressure_floor = new_pressure_floor
        
        # Update analysis tracking
        self.last_analysis_count = total_message_count
        
        # Update antagonist commitment if specified
        if self.current_antagonist and "commitment_change" in analysis:
            commitment = analysis["commitment_change"]
            if commitment != "no_change":
                self.current_antagonist.commitment_level = commitment
                self._log_debug(f"Antagonist commitment updated to: {commitment}")
        
        # Track escalation events
        escalation_events = analysis.get("escalation_events", [])
        if escalation_events:
            self.escalation_count += len(escalation_events)
    
    async def generate_antagonist(self, context_messages: List[Dict[str, Any]], max_attempts: int = 3) -> Optional[Antagonist]:
        """Generate high-quality antagonist based on story context"""
        
        # Prepare story context from recent messages
        story_context = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:300]}"
            for msg in context_messages[-15:]  # Recent context
        ])
        
        antagonist_prompt = f"""You are creating an antagonist for an ongoing RPG story. Based on the story context, 
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
}}"""
        
        for attempt in range(max_attempts):
            try:
                response = await self._call_llm([{"role": "system", "content": antagonist_prompt}])
                if response:
                    antagonist_data = self._parse_antagonist_response_robust(response)
                    if antagonist_data:
                        antagonist = self._create_antagonist_from_data(antagonist_data)
                        if self.validate_antagonist_quality(antagonist):
                            self._log_debug(f"Generated antagonist: {antagonist.name}")
                            return antagonist
                        
            except Exception as e:
                self._log_debug(f"Antagonist generation attempt {attempt + 1} failed: {e}")
        
        # Fallback antagonist if generation fails
        return self._create_fallback_antagonist()
    
    def _parse_antagonist_response_robust(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse antagonist generation response with 5-strategy defensive handling"""
        
        # Strategy 1: Direct JSON parsing
        try:
            data = json.loads(response.strip())
            if self._validate_antagonist_data(data):
                return self._inject_antagonist_defaults(data)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Substring extraction
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                if self._validate_antagonist_data(data):
                    return self._inject_antagonist_defaults(data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 3: Field extraction via regex
        try:
            import re
            extracted_data = {}
            
            # Extract name
            name_match = re.search(r'"?name"?\s*:\s*"([^"]+)"', response)
            if name_match:
                extracted_data["name"] = name_match.group(1)
            
            # Extract motivation
            motivation_match = re.search(r'"?motivation"?\s*:\s*"([^"]+)"', response)
            if motivation_match:
                extracted_data["motivation"] = motivation_match.group(1)
            
            # Extract threat level
            threat_match = re.search(r'"?threat_level"?\s*:\s*"([^"]+)"', response)
            if threat_match:
                extracted_data["threat_level"] = threat_match.group(1)
            
            if "name" in extracted_data and "motivation" in extracted_data:
                return self._inject_antagonist_defaults(extracted_data)
        except:
            pass
        
        # Strategy 4: Keyword-based extraction
        try:
            lines = response.split('\n')
            extracted_data = {}
            
            for line in lines:
                line_lower = line.lower()
                if "name:" in line_lower:
                    extracted_data["name"] = line.split(':', 1)[1].strip().strip('"')
                elif "motivation:" in line_lower:
                    extracted_data["motivation"] = line.split(':', 1)[1].strip().strip('"')
                elif "background:" in line_lower:
                    extracted_data["background"] = line.split(':', 1)[1].strip().strip('"')
            
            if extracted_data:
                return self._inject_antagonist_defaults(extracted_data)
        except:
            pass
        
        # Strategy 5: Complete fallback
        return self._create_fallback_antagonist_data()
    
    def _validate_antagonist_data(self, data: Dict[str, Any]) -> bool:
        """Validate antagonist data has minimum required fields"""
        if not isinstance(data, dict):
            return False
        
        required_fields = ["name", "motivation"]
        return all(field in data for field in required_fields)
    
    def _inject_antagonist_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject missing antagonist fields with defaults"""
        defaults = {
            "name": "Unknown Adversary",
            "motivation": "opposes the player's goals",
            "commitment_level": "testing",
            "resources_available": ["cunning", "persistence"],
            "resources_lost": [],
            "personality_traits": ["determined", "strategic"],
            "background": "A mysterious figure with unclear origins",
            "threat_level": "moderate"
        }
        
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        return data
    
    def _create_fallback_antagonist_data(self) -> Dict[str, Any]:
        """Create complete fallback antagonist data"""
        return {
            "name": "The Opposition",
            "motivation": "seeks to challenge the player's progress",
            "commitment_level": "testing",
            "resources_available": ["adaptability", "local knowledge", "patience"],
            "resources_lost": [],
            "personality_traits": ["mysterious", "calculating", "persistent"],
            "background": "An enigmatic force that opposes those who disturb the established order",
            "threat_level": "moderate"
        }
    
    def _create_antagonist_from_data(self, data: Dict[str, Any]) -> Antagonist:
        """Create Antagonist object from parsed data"""
        threat_level_map = {"low": 0.3, "moderate": 0.5, "high": 0.7, "extreme": 0.9}
        threat_level = threat_level_map.get(data.get("threat_level", "moderate"), 0.5)
        
        antagonist = Antagonist(
            name=data.get("name", "Unknown Antagonist"),
            motivation=data.get("motivation", "opposes the player"),
            threat_level=threat_level,
            context="llm_generated"
        )
        
        antagonist.commitment_level = data.get("commitment_level", "testing")
        antagonist.resources_available = data.get("resources_available", [])
        antagonist.resources_lost = data.get("resources_lost", [])
        antagonist.personality_traits = data.get("personality_traits", [])
        antagonist.background = data.get("background", "")
        
        return antagonist
    
    def _create_fallback_antagonist(self) -> Antagonist:
        """Create fallback antagonist if LLM generation fails"""
        antagonist = Antagonist(
            name="The Shadow",
            motivation="seeks to disrupt the player's journey",
            threat_level=min(0.9, self.pressure_level + 0.1),
            context="fallback"
        )
        antagonist.commitment_level = "testing"
        antagonist.resources_available = ["stealth", "cunning", "local knowledge"]
        antagonist.personality_traits = ["mysterious", "patient", "observant"]
        antagonist.background = "A mysterious figure who opposes those who disturb the natural order"
        return antagonist
    
    def validate_antagonist_quality(self, antagonist: Optional[Antagonist]) -> bool:
        """Validate that an antagonist has sufficient detail and quality"""
        if not antagonist:
            return False
        
        # Check for meaningful name (not just defaults)
        if antagonist.name in ["Unknown Antagonist", "Unknown Adversary", "Shadow Entity"]:
            return False
        
        # Check for meaningful motivation
        if len(antagonist.motivation) < 10:
            return False
        
        # Check for personality traits
        if not antagonist.personality_traits:
            return False
        
        return True
    
    async def analyze_resource_loss(self, conversation_text: str, antagonist: Optional[Antagonist]) -> Dict[str, Any]:
        """Analyze recent conversation for antagonist resource losses"""
        if not antagonist:
            return {"events": [], "resources_lost": []}
        
        analysis_prompt = f"""Analyze this RPG conversation for events where the antagonist {antagonist.name} 
might have lost resources, suffered setbacks, or been exposed.

Antagonist: {antagonist.name} - {antagonist.motivation}
Available Resources: {', '.join(antagonist.resources_available)}

Recent Conversation:
{conversation_text[-2000:]}

Look for:
- Direct confrontations or defeats
- Exposure of plans or identity
- Loss of allies, resources, or territory
- Failed schemes or setbacks

Provide JSON response:
{{
    "events": ["event1", "event2"],
    "resources_lost": ["resource1", "resource2"]
}}"""
        
        try:
            response = await self._call_llm([{"role": "system", "content": analysis_prompt}])
            if response:
                return self._parse_resource_loss_response_robust(response)
        except Exception as e:
            self._log_debug(f"Resource loss analysis failed: {e}")
        
        return {"events": [], "resources_lost": []}
    
    def _parse_resource_loss_response_robust(self, response: str) -> Dict[str, Any]:
        """Parse resource loss analysis response with defensive handling"""
        
        # Strategy 1: Direct JSON parsing
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                return {
                    "events": data.get("events", []),
                    "resources_lost": data.get("resources_lost", [])
                }
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Substring extraction
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return {
                        "events": data.get("events", []),
                        "resources_lost": data.get("resources_lost", [])
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 3: Pattern extraction
        try:
            import re
            events = []
            resources = []
            
            # Look for list patterns
            events_match = re.search(r'"?events"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
            if events_match:
                events_str = events_match.group(1)
                events = [item.strip().strip('"') for item in events_str.split(',') if item.strip()]
            
            resources_match = re.search(r'"?resources_lost"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
            if resources_match:
                resources_str = resources_match.group(1)
                resources = [item.strip().strip('"') for item in resources_str.split(',') if item.strip()]
            
            return {"events": events, "resources_lost": resources}
        except:
            pass
        
        # Strategy 4: Keyword scanning
        try:
            response_lower = response.lower()
            events = []
            resources = []
            
            # Look for common event keywords
            if "defeat" in response_lower or "fail" in response_lower:
                events.append("tactical setback")
            if "exposed" in response_lower or "revealed" in response_lower:
                events.append("identity exposure")
            if "lost" in response_lower or "destroyed" in response_lower:
                events.append("resource loss")
            
            return {"events": events, "resources_lost": resources}
        except:
            pass
        
        # Strategy 5: Empty fallback
        return {"events": [], "resources_lost": []}
    
    def calculate_pressure_floor_ratchet(self, recent_events: List[str]) -> float:
        """Calculate the new pressure floor based on escalation events (ratcheting upward)"""
        # Increment escalation count if significant events occurred
        if recent_events:
            self.escalation_count += len(recent_events)
        
        # Calculate new floor (ratcheting upward)
        new_floor = min(0.3, self.base_pressure_floor + (self.escalation_count * 0.02))
        
        # Never decrease (ratchet mechanism)
        return max(self.base_pressure_floor, new_floor)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current momentum state for analysis"""
        return {
            "narrative_pressure": self.pressure_level,
            "pressure_source": self._get_pressure_source(),
            "manifestation_type": self._get_manifestation_type(),
            "escalation_count": self.escalation_count,
            "base_pressure_floor": self.base_pressure_floor,
            "last_analysis_count": self.last_analysis_count,
            "antagonist": self.current_antagonist.to_dict() if self.current_antagonist else None,
            "story_arc": self.story_arc.value
        }
    
    def _get_pressure_source(self) -> str:
        """Determine current pressure source"""
        if self.current_antagonist and self.current_antagonist.active:
            return "antagonist"
        elif self.pressure_level > 0.5:
            return "environment"
        elif len(self.user_input_buffer) > 0:
            last_input = self.user_input_buffer[-1].lower()
            if any(word in last_input for word in ["talk", "speak", "negotiate"]):
                return "social"
            elif any(word in last_input for word in ["examine", "search", "investigate"]):
                return "discovery"
        return "environment"
    
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
    
    def get_story_context(self) -> Dict[str, Any]:
        """Generate story context for MCP module integration"""
        context = {
            "pressure_level": round(self.pressure_level, 3),
            "story_arc": self.story_arc.value,
            "narrative_state": self._get_narrative_state_description(),
            "should_introduce_tension": self.pressure_level < 0.4,
            "climax_approaching": self.pressure_level > 0.7,
            "antagonist_present": self.current_antagonist is not None and self.current_antagonist.active,
            "pressure_floor": self.base_pressure_floor
        }
        
        if self.current_antagonist:
            context["antagonist"] = {
                "name": self.current_antagonist.name,
                "motivation": self.current_antagonist.motivation,
                "threat_level": self.current_antagonist.threat_level,
                "active": self.current_antagonist.active,
                "commitment_level": self.current_antagonist.commitment_level,
                "resources_lost": len(self.current_antagonist.resources_lost)
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
    
    def reset_story_state(self):
        """Reset story state for new session"""
        self.pressure_level = 0.0
        self.story_arc = StoryArc.SETUP
        self.current_antagonist = None
        self.pressure_history.clear()
        self.user_input_buffer.clear()
        self.last_analysis_time = 0.0
        self.last_analysis_count = 0
        self.escalation_count = 0
        self.base_pressure_floor = 0.0
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
            "current_arc": self.story_arc.value,
            "pressure_floor": self.base_pressure_floor,
            "escalation_count": self.escalation_count,
            "last_analysis_count": self.last_analysis_count
        }
        
        return stats
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of pressure values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        return sum(squared_diffs) / len(squared_diffs)
    
    def force_pressure_level(self, new_pressure: float):
        """Force specific pressure level (debug/testing use)"""
        self.pressure_level = max(self.base_pressure_floor, min(1.0, new_pressure))
        self._update_story_arc()
        self._log_debug(f"Pressure forced to {self.pressure_level}")
    
    def force_antagonist_introduction(self):
        """Force antagonist introduction (debug/testing use)"""
        if self.current_antagonist is None:
            self.current_antagonist = self._create_basic_antagonist()
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
            "last_analysis_count": self.last_analysis_count,
            "escalation_count": self.escalation_count,
            "base_pressure_floor": self.base_pressure_floor,
            "pressure_decay_rate": self.pressure_decay_rate,
            "antagonist_threshold": self.pressure_threshold_antagonist,
            "climax_threshold": self.pressure_threshold_climax,
            "analysis_cooldown": self.analysis_cooldown
        }
    
    # State persistence methods for EMM integration
    def save_state_to_dict(self) -> Dict[str, Any]:
        """Save SME state to dictionary for EMM storage"""
        return {
            "narrative_pressure": self.pressure_level,
            "pressure_source": self._get_pressure_source(),
            "manifestation_type": self._get_manifestation_type(),
            "escalation_count": self.escalation_count,
            "base_pressure_floor": self.base_pressure_floor,
            "last_analysis_count": self.last_analysis_count,
            "antagonist": self.current_antagonist.to_dict() if self.current_antagonist else None,
            "story_arc": self.story_arc.value,
            "pressure_history": self.pressure_history[-10:],  # Save last 10 pressure points
            "timestamp": time.time()
        }
    
    def load_state_from_dict(self, state_data: Dict[str, Any]) -> bool:
        """Load SME state from dictionary (from EMM)"""
        try:
            self.pressure_level = max(0.0, min(1.0, state_data.get("narrative_pressure", 0.0)))
            self.escalation_count = max(0, state_data.get("escalation_count", 0))
            self.base_pressure_floor = max(0.0, min(0.3, state_data.get("base_pressure_floor", 0.0)))
            self.last_analysis_count = max(0, state_data.get("last_analysis_count", 0))
            
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
            
            self._log_debug("SME state loaded from EMM")
            return True
            
        except Exception as e:
            self._log_debug(f"Failed to load SME state: {e}")
            return False

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
        "antagonist_present": sme.current_antagonist is not None,
        "pressure_floor": sme.base_pressure_floor
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
