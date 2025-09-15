# Enhanced Memory Checkpoint Plan

## Executive Summary
This plan focuses on testing and validating the Enhanced Memory Manager's condensation functionality using semantic categorization. The goal is to verify that low-importance messages are properly condensed while high-importance messages are preserved verbatim.

## Prerequisites
- Semantic Scoring Checkpoint Plan completed
- All messages have `content_category` and `importance_score` fields
- Semantic analysis running for every message

## Current State Analysis

### Working Components
- **emm.py**: Has condensation infrastructure
  - `check_condensation_needed()`
  - `get_condensation_candidates()`
  - `replace_messages_with_condensed()`
- **Semantic Categories**: Defined preservation ratios
  - story_critical: 90% preservation
  - character_focused: 80% preservation
  - relationship_dynamics: 70% preservation
  - emotional_significance: 60% preservation
  - world_building: 50% preservation
  - standard: 40% preservation

### Problem Areas
1. Condensation never triggers (25,000 token limit too high)
2. No actual summarization logic implemented
3. Preservation ratios not applied
4. No testing framework for condensation

## Phase 1: Reduce Token Budget for Testing

### Goal
Force condensation to trigger frequently for testing

### Step 1.1: Temporary Test Configuration
```python
# In emm.py __init__():
# For testing, reduce from 25000 to 500 tokens
self.max_memory_tokens = 500  # TEST VALUE - revert to 25000 for production
self.condensation_test_mode = True  # ADD flag for test mode
```

### Step 1.2: Add Condensation Statistics
```python
# In emm.py add tracking:
self.condensation_stats = {
    "total_condensations": 0,
    "messages_condensed": 0,
    "tokens_saved": 0,
    "categories_condensed": {},
    "last_condensation": None
}

def get_condensation_stats(self) -> Dict[str, Any]:
    """Get condensation statistics for testing"""
    return self.condensation_stats.copy()
```

## Phase 2: Implement Category-Based Condensation

### Goal
Implement proper condensation logic respecting semantic categories

### Step 2.1: Group Messages by Category
```python
# In emm.py add method:
def _group_messages_by_category(self, message_ids: List[str]) -> Dict[str, List[Message]]:
    """Group candidate messages by their semantic category"""
    groups = {
        "story_critical": [],
        "character_focused": [],
        "relationship_dynamics": [],
        "emotional_significance": [],
        "world_building": [],
        "standard": []
    }
    
    for msg_id in message_ids:
        for msg in self.messages:
            if msg.id == msg_id:
                category = msg.content_category
                if category in groups:
                    groups[category].append(msg)
                break
    
    return groups
```

### Step 2.2: Apply Preservation Ratios
```python
# In emm.py add method:
def _select_messages_for_condensation(self, candidates: List[str]) -> Dict[str, List[str]]:
    """Select which messages to condense based on preservation ratios"""
    grouped = self._group_messages_by_category(candidates)
    to_condense = {}
    to_preserve = []
    
    for category, messages in grouped.items():
        if not messages:
            continue
            
        preservation_ratio = SEMANTIC_CATEGORIES[category]["preservation_ratio"]
        total_count = len(messages)
        preserve_count = int(total_count * preservation_ratio)
        condense_count = total_count - preserve_count
        
        if condense_count > 0:
            # Sort by importance_score to preserve highest scored
            sorted_msgs = sorted(messages, key=lambda m: m.importance_score, reverse=True)
            
            # Preserve the most important
            for msg in sorted_msgs[:preserve_count]:
                to_preserve.append(msg.id)
            
            # Condense the rest
            condensable = [msg.id for msg in sorted_msgs[preserve_count:]]
            if condensable:
                to_condense[category] = condensable
    
    return {"condense": to_condense, "preserve": to_preserve}
```

### Step 2.3: Create Category-Aware Summaries
```python
# In emm.py add method:
def _create_condensed_summary(self, message_ids: List[str], category: str) -> str:
    """Create a condensed summary for messages in a category"""
    messages = [msg for msg in self.messages if msg.id in message_ids]
    
    if category == "standard":
        return f"[{len(messages)} standard interactions about routine activities]"
    
    elif category == "world_building":
        topics = self._extract_topics(messages)
        return f"[{len(messages)} world details discussed: {', '.join(topics[:3])}...]"
    
    elif category == "emotional_significance":
        emotions = self._extract_emotions(messages)
        return f"[{len(messages)} emotional moments: {', '.join(emotions[:3])}...]"
    
    elif category == "relationship_dynamics":
        characters = self._extract_characters(messages)
        return f"[{len(messages)} relationship developments with: {', '.join(characters[:3])}...]"
    
    elif category == "character_focused":
        aspects = self._extract_character_aspects(messages)
        return f"[{len(messages)} character details revealed: {', '.join(aspects[:3])}...]"
    
    elif category == "story_critical":
        # These should rarely be condensed
        return f"[CRITICAL: {len(messages)} key story events - see preserved messages]"
    
    return f"[{len(messages)} {category} messages condensed]"
```

## Phase 3: Implement Extraction Helpers

### Goal
Extract key information from messages being condensed

### Step 3.1: Topic Extraction
```python
def _extract_topics(self, messages: List[Message]) -> List[str]:
    """Extract main topics from messages"""
    topics = set()
    topic_patterns = {
        "magic": r'\b(magic|spell|enchant|wizard|sorcery)\b',
        "combat": r'\b(fight|battle|sword|weapon|attack)\b',
        "exploration": r'\b(explore|discover|journey|travel|quest)\b',
        "dialogue": r'\b(speak|talk|discuss|converse|ask)\b',
        "inventory": r'\b(item|equip|carry|inventory|gear)\b'
    }
    
    for msg in messages:
        content_lower = msg.content.lower()
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, content_lower):
                topics.add(topic)
    
    return list(topics)
```

### Step 3.2: Emotion Extraction
```python
def _extract_emotions(self, messages: List[Message]) -> List[str]:
    """Extract emotional states from messages"""
    emotions = set()
    emotion_words = {
        "fear": ["afraid", "scared", "frightened", "terrified"],
        "joy": ["happy", "joyful", "excited", "thrilled"],
        "anger": ["angry", "furious", "mad", "rage"],
        "sadness": ["sad", "sorrowful", "depressed", "melancholy"],
        "surprise": ["surprised", "shocked", "amazed", "astonished"]
    }
    
    for msg in messages:
        content_lower = msg.content.lower()
        for emotion, words in emotion_words.items():
            if any(word in content_lower for word in words):
                emotions.add(emotion)
    
    return list(emotions)
```

## Phase 4: Testing Framework

### Goal
Create comprehensive tests for condensation

### Step 4.1: Test Data Generator
```python
# Create test_condensation.py:
def generate_test_conversation():
    """Generate a conversation that will trigger condensation"""
    messages = [
        # Standard messages (should condense 60%)
        ("I walk down the hallway.", "standard", 0.4),
        ("I open the door.", "standard", 0.4),
        ("I look around.", "standard", 0.4),
        ("I pick up the torch.", "standard", 0.4),
        ("I continue forward.", "standard", 0.4),
        
        # World building (should condense 50%)
        ("The castle was built 500 years ago.", "world_building", 0.5),
        ("It uses ancient dwarven architecture.", "world_building", 0.5),
        ("The stones are imported from the northern mountains.", "world_building", 0.5),
        ("Local legends speak of hidden passages.", "world_building", 0.5),
        
        # Character focused (should preserve 80%)
        ("My name is Aldric, son of Theron.", "character_focused", 0.8),
        ("I trained as a paladin in the Silver Order.", "character_focused", 0.8),
        ("My sacred oath binds me to protect the innocent.", "character_focused", 0.8),
        ("I carry my father's blessed sword.", "character_focused", 0.8),
        ("My greatest fear is failing those who depend on me.", "character_focused", 0.8),
        
        # Story critical (should preserve 90%)
        ("The Dark Lord has returned to threaten the realm.", "story_critical", 0.9),
        ("Only the Sword of Dawn can defeat him.", "story_critical", 0.9),
        ("You are the chosen hero of the prophecy.", "story_critical", 0.9),
    ]
    
    return messages
```

### Step 4.2: Condensation Test Runner
```python
def test_condensation_cycle():
    """Test a full condensation cycle"""
    emm = EnhancedMemoryManager(auto_save_enabled=False)
    emm.max_memory_tokens = 200  # Very low for testing
    
    # Add test messages
    test_messages = generate_test_conversation()
    for content, category, score in test_messages:
        msg = Message(content, MessageType.USER)
        msg.content_category = category
        msg.importance_score = score
        emm.messages.append(msg)
    
    print(f"Starting with {len(emm.messages)} messages")
    print(f"Token count: {sum(m.token_estimate for m in emm.messages)}")
    
    # Check if condensation needed
    if emm.check_condensation_needed():
        print("Condensation triggered!")
        
        # Get candidates
        candidates = emm.get_condensation_candidates(preserve_recent=5)
        print(f"Candidates for condensation: {len(candidates)}")
        
        # Group and select
        selection = emm._select_messages_for_condensation(candidates)
        
        # Show what will be condensed
        for category, msg_ids in selection["condense"].items():
            print(f"  {category}: {len(msg_ids)} messages to condense")
        print(f"  Preserving: {len(selection['preserve'])} messages")
        
        # Perform condensation
        for category, msg_ids in selection["condense"].items():
            summary = emm._create_condensed_summary(msg_ids, category)
            emm.replace_messages_with_condensed(msg_ids, summary)
            print(f"  Condensed {category}: {summary}")
    
    print(f"After condensation: {len(emm.messages)} messages")
    print(f"Token count: {sum(m.token_estimate for m in emm.messages)}")
    print(f"Stats: {emm.get_condensation_stats()}")
```

## Phase 5: Integration Testing

### Goal
Test condensation in live conversation

### Step 5.1: Add Condensation Command
```python
# In ncui.py add command:
elif command == "/condense":
    # Force immediate condensation for testing
    if self.callback_handler:
        result = self.callback_handler("force_condensation", {})
        self.callback_handler("add_system_message", {
            "content": f"Condensation result: {result}",
            "message_type": "system"
        })
```

### Step 5.2: Add Orchestrator Handler
```python
# In orch.py _handle_ui_callback():
elif action == "force_condensation":
    return self._force_condensation_test()

def _force_condensation_test(self) -> Dict[str, Any]:
    """Force condensation for testing"""
    if not self.memory_manager:
        return {"success": False, "error": "No memory manager"}
    
    # Temporarily lower threshold
    original_max = self.memory_manager.max_memory_tokens
    self.memory_manager.max_memory_tokens = 100
    
    try:
        if self.memory_manager.check_condensation_needed():
            candidates = self.memory_manager.get_condensation_candidates()
            # ... perform condensation ...
            return {"success": True, "condensed": len(candidates)}
        else:
            return {"success": False, "error": "No condensation needed"}
    finally:
        self.memory_manager.max_memory_tokens = original_max
```

## Phase 6: Validation and Metrics

### Goal
Ensure condensation maintains narrative coherence

### Step 6.1: Coherence Checks
```python
def validate_condensation_coherence(before_messages: List[Message], after_messages: List[Message]):
    """Validate that condensation preserves narrative coherence"""
    checks = {
        "story_critical_preserved": True,
        "character_names_preserved": True,
        "timeline_preserved": True,
        "token_reduction": 0.0
    }
    
    # Check all story_critical messages preserved
    critical_before = [m for m in before_messages if m.content_category == "story_critical"]
    critical_after = [m for m in after_messages if m.content_category == "story_critical" and not m.condensed]
    checks["story_critical_preserved"] = len(critical_after) >= len(critical_before) * 0.9
    
    # Check character names preserved
    names_before = extract_character_names(before_messages)
    names_after = extract_character_names(after_messages)
    checks["character_names_preserved"] = names_before.issubset(names_after)
    
    # Calculate token reduction
    tokens_before = sum(m.token_estimate for m in before_messages)
    tokens_after = sum(m.token_estimate for m in after_messages)
    checks["token_reduction"] = (tokens_before - tokens_after) / tokens_before
    
    return checks
```

## Success Criteria

1. **Triggering**: Condensation triggers when tokens exceed limit
2. **Categorization**: Messages grouped correctly by category
3. **Preservation**: High-importance messages preserved per ratios
4. **Summarization**: Condensed summaries capture key information
5. **Coherence**: Narrative remains coherent after condensation
6. **Performance**: Condensation completes within 1 second
7. **Persistence**: Condensed state saves and loads correctly

## Testing Checklist

### Pre-Condensation
- [ ] Generate messages across all categories
- [ ] Verify semantic scores assigned
- [ ] Check token count approaching limit
- [ ] Confirm recent messages excluded

### During Condensation
- [ ] Verify correct candidates selected
- [ ] Check preservation ratios applied
- [ ] Monitor category grouping
- [ ] Validate summary generation

### Post-Condensation
- [ ] Verify token count reduced
- [ ] Check critical messages preserved
- [ ] Confirm summaries inserted correctly
- [ ] Test narrative coherence
- [ ] Validate memory.json structure

## Debug Commands

```python
# Commands to add for testing:
/memory stats     # Show memory statistics
/memory tokens    # Show current token usage
/condense test    # Run test condensation
/condense force   # Force immediate condensation
/semantic show    # Show semantic categories
/semantic stats   # Show category distribution
```

## Production Configuration

After testing, restore production values:

```python
# In emm.py:
self.max_memory_tokens = 25000  # Production value
self.condensation_test_mode = False

# Condensation should trigger when:
# - Token count > 25000
# - At least 20 messages exist
# - Recent 5 messages preserved
```

## Next Steps

1. Run test suite with low token limits
2. Validate preservation ratios working
3. Test with real conversation flow
4. Measure performance impact
5. Fine-tune summary generation
6. Restore production limits
7. Long-term conversation testing