# Semantic Scoring Checkpoint Plan

## Executive Summary
This plan focuses on implementing and verifying semantic analysis functionality in the DevName RPG Client. The goal is to ensure all messages are properly categorized and scored, with results visible for debugging and verification before implementing condensation logic.

## Current State Analysis

### Working Components
- **sem.py**: Semantic analysis engine with categorization logic exists
- **emm.py**: Has `update_message_category()` method and `content_category` field in Message class
- **orch.py**: LLM queue system implemented but semantic results not processed
- **memory.json**: Stores messages but no semantic metadata persisted

### Problem Areas
1. Semantic analysis is queued but results never update messages
2. No visibility into semantic analysis results
3. `importance_score` field missing from Message class
4. Pattern-based fallback not implemented as primary method

## Priority 0: Add Semantic Tracking/Visibility

### Goal
Make semantic analysis results visible for debugging and verification

### Implementation Options

#### Option A: Enhanced memory.json (RECOMMENDED)
Add semantic fields to persisted message data:

```python
# In emm.py Message class:
def __init__(self, content: str, message_type: MessageType, timestamp: Optional[str] = None):
    self.content = content
    self.message_type = message_type
    self.timestamp = timestamp or datetime.now().isoformat()
    self.token_estimate = self._estimate_tokens(content)
    self.id = str(uuid4())
    self.content_category = "standard"
    self.importance_score = 0.4  # ADD THIS
    self.semantic_metadata = {}   # ADD THIS
    self.condensed = False

def to_dict(self) -> Dict[str, Any]:
    return {
        "id": self.id,
        "content": self.content,
        "type": self.message_type.value,
        "timestamp": self.timestamp,
        "tokens": self.token_estimate,
        "content_category": self.content_category,
        "importance_score": self.importance_score,  # ADD THIS
        "semantic_metadata": self.semantic_metadata,  # ADD THIS
        "condensed": self.condensed
    }

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'Message':
    message = cls(
        content=data["content"],
        message_type=MessageType(data["type"]),
        timestamp=data.get("timestamp")
    )
    message.id = data.get("id", str(uuid4()))
    message.token_estimate = data.get("tokens", message.token_estimate)
    message.content_category = data.get("content_category", "standard")
    message.importance_score = data.get("importance_score", 0.4)  # ADD THIS
    message.semantic_metadata = data.get("semantic_metadata", {})  # ADD THIS
    message.condensed = data.get("condensed", False)
    return message
```

#### Option B: Dedicated semantic_analysis.log
Create separate tracking file:

```python
# In orch.py after semantic analysis:
def _log_semantic_analysis(self, message_id: str, result: Dict[str, Any]):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "message_id": message_id,
        "category": result.get("category"),
        "score": result.get("importance_score"),
        "analysis_type": result.get("type"),  # pattern or llm
        "patterns_matched": result.get("patterns", [])
    }
    with open("semantic_analysis.log", "a") as f:
        f.write(json.dumps(entry) + "\n")
```

## Phase 1: Fix Semantic Analysis Flow

### Goal
Ensure semantic analysis executes immediately after each message using pattern matching

### Step 1.1: Implement Synchronous Pattern Analysis
Create immediate pattern-based analysis in orchestrator:

```python
# In orch.py add new method:
def _perform_immediate_semantic_analysis(self, message: Message) -> None:
    """Perform immediate pattern-based semantic analysis"""
    try:
        content_lower = message.content.lower()
        
        # Story critical patterns
        if any(phrase in content_lower for phrase in [
            "main quest", "primary objective", "critical mission",
            "fate of", "destiny", "prophecy", "chosen one",
            "world ending", "save the kingdom", "final battle"
        ]):
            message.content_category = "story_critical"
            message.importance_score = 0.9
            
        # Character focused patterns
        elif any(phrase in content_lower for phrase in [
            "my name is", "i am called", "backstory", "my past",
            "character development", "personality", "my motivation",
            "why i", "i believe", "my goal"
        ]):
            message.content_category = "character_focused"
            message.importance_score = 0.8
            
        # Relationship dynamics patterns
        elif any(phrase in content_lower for phrase in [
            "trust", "betray", "alliance", "friendship", "romance",
            "rival", "enemy turned", "companion", "loyalty",
            "working together", "conflict between"
        ]):
            message.content_category = "relationship_dynamics"
            message.importance_score = 0.7
            
        # Emotional significance patterns
        elif any(phrase in content_lower for phrase in [
            "i feel", "makes me", "emotional", "heart",
            "fear", "love", "hate", "joy", "sorrow",
            "triumph", "devastating", "overwhelming"
        ]):
            message.content_category = "emotional_significance"
            message.importance_score = 0.6
            
        # World building patterns
        elif any(phrase in content_lower for phrase in [
            "history of", "legend says", "long ago", "ancient",
            "culture", "tradition", "geography", "politics",
            "economy", "religion", "magic system", "technology"
        ]):
            message.content_category = "world_building"
            message.importance_score = 0.5
            
        # Default standard
        else:
            message.content_category = "standard"
            message.importance_score = 0.4
        
        # Add metadata
        message.semantic_metadata = {
            "analysis_type": "pattern",
            "analyzed_at": time.time(),
            "content_length": len(message.content),
            "token_count": message.token_estimate
        }
        
        # Log the analysis
        self._log_debug(f"Semantic analysis: {message.id[:8]} -> {message.content_category} ({message.importance_score})")
        
        # Force save to persist semantic data
        if self.memory_manager and hasattr(self.memory_manager, '_auto_save'):
            self.memory_manager._auto_save()
            
    except Exception as e:
        self._log_error(f"Semantic analysis failed: {e}")
```

### Step 1.2: Hook Into Message Processing
Modify message processing to include immediate analysis:

```python
# In orch.py _process_user_response_request(), after storing user message:
if self.memory_manager:
    self.memory_manager.add_message(user_input, MessageType.USER)
    user_message = self.memory_manager.messages[-1]
    self._perform_immediate_semantic_analysis(user_message)
    
# After storing assistant response:
if self.memory_manager and response_text:
    self.memory_manager.add_message(response_text, MessageType.ASSISTANT)
    assistant_message = self.memory_manager.messages[-1]
    self._perform_immediate_semantic_analysis(assistant_message)
```

## Phase 2: Message Splitting for Multi-Topic Content

### Goal
Handle messages that contain multiple semantic categories

### Step 2.1: Detect Multi-Topic Messages
```python
def _detect_multiple_topics(self, content: str) -> List[Dict[str, Any]]:
    """Detect if message contains multiple semantic topics"""
    topics_found = []
    
    # Check each category independently
    for category, patterns in CATEGORY_PATTERNS.items():
        if self._matches_patterns(content, patterns):
            topics_found.append({
                "category": category,
                "score": SEMANTIC_CATEGORIES[category]["preservation_ratio"]
            })
    
    return topics_found if len(topics_found) > 1 else []
```

### Step 2.2: Split Complex Messages
```python
def _split_message_by_sentences(self, message: Message) -> List[Message]:
    """Split message into sentence-level fragments for granular categorization"""
    sentences = re.split(r'[.!?]+', message.content)
    fragments = []
    
    for sentence in sentences:
        if sentence.strip():
            fragment = Message(
                content=sentence.strip(),
                message_type=message.message_type,
                timestamp=message.timestamp
            )
            fragment.semantic_metadata = {
                "parent_message_id": message.id,
                "is_fragment": True
            }
            fragments.append(fragment)
    
    return fragments
```

## Phase 3: Testing and Verification

### Test Case 1: Single Category Messages
```python
test_messages = [
    ("The prophecy speaks of a chosen one who will save the realm.", "story_critical", 0.9),
    ("I grew up in a small village, learning herbalism from my grandmother.", "character_focused", 0.8),
    ("I trust you with my life, old friend.", "relationship_dynamics", 0.7),
    ("My heart races with fear as the shadow approaches.", "emotional_significance", 0.6),
    ("The kingdom uses a lunar calendar with thirteen months.", "world_building", 0.5),
    ("I walk down the corridor.", "standard", 0.4)
]
```

### Test Case 2: Multi-Category Messages
```python
complex_message = """
The ancient prophecy my grandmother told me speaks of a chosen one. 
I feel overwhelming fear but trust my companion to stand beside me.
Together we must save the kingdom from the shadow plague.
"""
# Should identify: story_critical, character_focused, emotional_significance, relationship_dynamics
```

### Verification Commands
Add debug command to ncui.py:
```python
elif command == "/semantic":
    # Show semantic analysis for last 10 messages
    if self.callback_handler:
        self.callback_handler("show_semantic_analysis", {"limit": 10})
```

## Phase 4: LLM Enhancement (Optional)

### Goal
Queue LLM analysis for uncertain categorizations

### Step 4.1: Confidence Scoring
```python
def _calculate_pattern_confidence(self, content: str, category: str) -> float:
    """Calculate confidence in pattern-based categorization"""
    strong_matches = 0
    weak_matches = 0
    
    for pattern in STRONG_PATTERNS[category]:
        if pattern in content.lower():
            strong_matches += 1
    
    for pattern in WEAK_PATTERNS[category]:
        if pattern in content.lower():
            weak_matches += 1
    
    confidence = (strong_matches * 1.0 + weak_matches * 0.5) / 10.0
    return min(1.0, confidence)
```

### Step 4.2: Queue for LLM When Uncertain
```python
if confidence < 0.5:
    # Queue for LLM enhancement
    self._queue_semantic_enhancement(message.id, message.content)
```

## Success Criteria

1. **Visibility**: Can see semantic categories and scores in memory.json
2. **Accuracy**: Messages correctly categorized based on content
3. **Persistence**: Semantic data survives restart
4. **Performance**: Analysis completes within 100ms per message
5. **Coverage**: 100% of messages have semantic analysis

## Debugging Checklist

- [ ] Check memory.json has importance_score field
- [ ] Check memory.json has semantic_metadata field
- [ ] Verify categories match SEMANTIC_CATEGORIES enum
- [ ] Confirm scores align with preservation ratios
- [ ] Test with `/semantic` command
- [ ] Verify debug.log shows analysis results
- [ ] Check pattern matching for each category
- [ ] Test multi-topic message handling
- [ ] Verify persistence after restart
- [ ] Monitor analysis performance

## Next Steps
Once semantic scoring is working correctly, proceed to "Enhanced Memory Checkpoint Plan" for testing condensation based on these categories and scores.