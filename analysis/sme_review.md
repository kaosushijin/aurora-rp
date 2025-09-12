# Story Momentum Engine (SME) Review

## **High-Level Architecture**

The Story Momentum Engine serves as the **narrative progression controller** for the RPG client, managing story pacing, pressure dynamics, and antagonist generation. It operates as a **spoke module** coordinating with the orchestrator for LLM-powered narrative analysis.

---

## **Core Procedural Flow**

### **1. Input Processing Cycle**
```
process_user_input() → pressure_decay() → pattern_analysis() → pressure_calculation() → story_arc_update() → state_persistence()
```

**Process Details:**
- **Rate Limiting**: 30-second cooldown between analyses to prevent spam
- **Pressure Decay**: Gradual tension reduction over time (0.01 per minute)
- **Pattern Matching**: Keyword-based momentum detection from user input
- **State Updates**: Pressure, arc progression, and narrative time tracking

### **2. Pressure Management System**
```
Current Pressure → Apply Decay → Calculate Input Delta → Apply Floor Ratcheting → Update History
```

**Pressure Dynamics:**
- **Range**: 0.0 (no tension) to 1.0 (maximum tension)
- **Decay Rate**: -0.01 per minute to prevent infinite stagnation
- **Floor Ratcheting**: `base_pressure_floor` prevents regression below achieved tension
- **Input Sensitivity**: User actions modify pressure based on detected patterns

### **3. Story Arc Progression**
```
Pressure Thresholds → Arc Transition → Antagonist Management → Context Updates
```

**Arc Transition Logic:**
- **SETUP** (0.0-0.3): Initial exploration and world-building
- **RISING** (0.3-0.7): Tension building and conflict development  
- **CLIMAX** (0.7-1.0): Peak conflict and critical moments
- **RESOLUTION** (decay from climax): Conclusion and aftermath

### **4. Narrative Time Tracking**
```
detect_duration_from_text() → add_exchange() → update_cumulative_time() → sequence_history()
```

**Time Calculation:**
- **Pattern Detection**: Regex-based time reference extraction
- **Activity Estimation**: Default durations for common actions
- **Cumulative Tracking**: Total narrative time across conversation
- **Sequence History**: Last 50 exchanges with duration metadata

---

## **Pattern-Based Analysis System**

### **Current Momentum Patterns**
```python
self.momentum_patterns = {
    "action": ["attack", "fight", "run", "chase", "grab", "push", "pull", "strike"],
    "tension": ["danger", "threat", "fear", "worry", "concern", "suspicious", "uneasy"],
    "mystery": ["strange", "odd", "unusual", "mysterious", "hidden", "secret"],
    "discovery": ["find", "discover", "uncover", "reveal", "notice", "observe"],
    "social": ["talk", "speak", "conversation", "discuss", "negotiate", "persuade"]
}
```

### **Pressure Calculation Logic**
```python
def _calculate_pressure_change(self, input_text: str) -> float:
    pressure_delta = 0.0
    text_lower = input_text.lower()
    
    # Pattern-based pressure increases
    for pattern_type, keywords in self.momentum_patterns.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if pattern_type == "action":
            pressure_delta += matches * 0.05  # Combat increases tension
        elif pattern_type == "tension": 
            pressure_delta += matches * 0.04  # Anxiety/fear builds pressure
        # [additional pattern weights...]
    
    # Exclamation intensity and caps detection
    pressure_delta += min(input_text.count("!") * 0.02, 0.08)
    
    return min(pressure_delta, 0.3)  # Cap maximum single increase
```

---

## **Antagonist Management**

### **Current Implementation**
The current SME has **simplified antagonist management** with basic threshold detection:

```python
def _manage_antagonist_threshold(self) -> bool:
    """Basic threshold check for antagonist introduction"""
    if not self.current_antagonist and self.pressure_level > 0.5:
        # Create basic antagonist when pressure is high
        self.current_antagonist = Antagonist(
            name="Unknown Threat",
            threat_level=self.pressure_level,
            active=True,
            last_mention=""
        )
        return True
    return False
```

### **Antagonist Data Structure**
```python
@dataclass
class Antagonist:
    name: str = ""
    threat_level: float = 0.0
    active: bool = False
    last_mention: str = ""
```

---

## **Narrative Time System**

### **Duration Detection Patterns**
```python
time_patterns = {
    r'\b(\d+)\s*hours?\b': lambda m: float(m.group(1)) * 3600,
    r'\b(\d+)\s*minutes?\b': lambda m: float(m.group(1)) * 60,
    r'\bhalf\s+hour\b': lambda m: 1800,
    r'\ba\s+while\b': lambda m: 300,
    r'\bquickly\b': lambda m: 5,
}

activity_durations = {
    r'\b(examine|search|investigate)\b': 120,  # 2 minutes
    r'\b(travel|walk|journey)\b': 600,         # 10 minutes
    r'\b(rest|sleep|wait)\b': 1800,            # 30 minutes
    r'\b(fight|battle|combat)\b': 180,         # 3 minutes
}
```

### **Time Tracking Process**
1. **Pattern Matching**: Regex extraction of explicit time references
2. **Activity Analysis**: Default durations for detected activities
3. **Cumulative Addition**: Running total of narrative time
4. **History Maintenance**: Last 50 exchanges with metadata

---

## **Integration with Semantic Analysis (sem.py)**

### **Current LLM Integration Status**
The current SME implementation uses **minimal LLM integration**. Most analysis is pattern-based, with LLM requests coordinated through the orchestrator for:
- **Complex narrative analysis** (when pattern-based detection is insufficient)
- **Antagonist generation** (when threshold conditions are met)
- **Story context enhancement** (for context window building)

### **Semantic Request Types**
The SME can request these analysis types through the orchestrator:

1. **"momentum"**: Overall story pacing analysis
2. **"antagonist"**: Opposition element generation
3. **"narrative_context"**: Story state summarization

---

## **Exact Semantic Prompting from sem.py**

### **Momentum Analysis Prompt** (Used for SME requests)
```
Analyze this RPG conversation for story momentum and narrative pressure:

Conversation Context: {conversation_history}
Current Pressure: {current_pressure}
Story Arc: {current_arc}

Determine:
1. Overall narrative tension level (0.0-1.0)
2. Pressure change recommendations
3. Story arc progression assessment
4. Antagonist presence/need evaluation

Provide JSON response with momentum analysis.
```

### **Antagonist Generation Prompt**
```
Generate an appropriate antagonist for this RPG scenario:

Current Story Context: {story_context}
Pressure Level: {pressure_level}
Story Arc: {story_arc}

Create an antagonist with:
- Name and basic description
- Motivation and goals
- Threat level (0.0-1.0)
- Introduction method

Return JSON format with antagonist details.
```

### **Story Context Building**
```
Summarize the current story state for context:

Recent Messages: {recent_messages}
Pressure Level: {pressure}
Story Arc: {arc}
Narrative Time: {time_elapsed}

Provide concise story state summary focusing on:
- Current situation and location
- Active conflicts or tensions
- Character states and relationships
- Recent developments
```

---

## **State Management and Persistence**

### **Core State Structure**
```python
def get_story_context(self) -> Dict[str, Any]:
    return {
        "pressure_level": self.pressure_level,
        "story_arc": self.story_arc.value,
        "narrative_seconds": self.narrative_time.total_narrative_seconds,
        "exchange_count": self.narrative_time.exchange_count,
        "pressure_floor": self.base_pressure_floor,
        "manifestation_type": self._get_current_manifestation(),
        "narrative_state": self._get_narrative_state_description(),
        "antagonist_status": self._get_antagonist_status()
    }
```

### **EMM Integration**
The SME state is persisted through EMM using special message types:
- **Storage**: State serialized as JSON in `MessageType.MOMENTUM_STATE`
- **Retrieval**: Latest momentum state loaded from EMM on startup
- **Updates**: State changes trigger EMM persistence automatically

---

## **Performance and Analysis Features**

### **Rate Limiting and Efficiency**
- **30-second cooldown**: Prevents excessive LLM requests
- **Pattern-first approach**: Most analysis done locally
- **Background processing**: State updates don't block UI
- **Graceful degradation**: Continues operation without LLM access

### **Pressure Statistics Tracking**
```python
def get_pressure_stats(self) -> Dict[str, Any]:
    return {
        "current_pressure": self.pressure_level,
        "pressure_floor": self.base_pressure_floor,
        "story_arc": self.story_arc.value,
        "total_exchanges": len(self.user_input_buffer),
        "narrative_hours": self.narrative_time.total_narrative_seconds / 3600,
        "pressure_history_points": len(self.pressure_history)
    }
```

### **Story Context for LLM Integration**
The SME provides structured context for LLM requests through the orchestrator:
- **Current pressure and arc state**
- **Recent user input patterns**
- **Narrative time progression**
- **Antagonist status and history**

---

## **Current API Interface**

### **Primary Methods**
- `process_user_input(text, sequence_num)`: Main analysis entry point
- `get_story_context()`: Current state for LLM context building
- `get_pressure_stats()`: Statistical overview
- `set_orchestrator_callback()`: Hub communication setup

### **State Management**
- Automatic persistence through EMM integration
- Thread-safe state updates
- Graceful handling of missing components

---

## **Key Strengths and Limitations**

### **Strengths**
- **Real-time responsiveness**: Pattern-based analysis provides immediate feedback
- **Hub-spoke compliance**: Clean orchestrator communication
- **Persistent state**: Maintains story context across sessions
- **Graceful fallbacks**: Continues operation without LLM access

### **Current Limitations**
- **Basic antagonist system**: Limited complexity in opposition generation
- **Pattern-dependent**: Relies heavily on keyword matching
- **Limited LLM integration**: Could benefit from more sophisticated analysis
- **Simple time model**: Basic duration estimation

---

*Status: Complete analysis of SME architecture and semantic integration*