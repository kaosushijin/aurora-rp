# Real Time vs Narrative Time Separation Project Plan (New Architecture)

## Problem Statement

The current Story Momentum Engine (SME) conflates real-world time passage with narrative time progression, causing unintended story effects:

- **Pressure Decay**: Story tension decreases based on real minutes elapsed, not narrative progression
- **Rate Limiting**: 2-second cooldowns can artificially constrain rapid narrative exchanges
- **Analysis Timing**: Some momentum calculations use wall-clock timestamps instead of story beats

This breaks narrative immersion - a player taking time to think shouldn't cause story pressure to dissipate, and rapid player responses shouldn't be throttled by real-time constraints.

## Impact Analysis

### Current Problems
- Player contemplation time incorrectly reduces story tension
- Fast-paced dramatic exchanges may be artificially slowed
- Narrative momentum becomes dependent on player typing speed
- Story analysis may be skewed by real-world interruptions

### Affected Components (NEW ARCHITECTURE)
- **`sme.py`**: `_apply_pressure_decay()`, `process_user_input()` cooldown
- **`sem.py`**: Shared semantic logic that may use timestamps
- **`emm.py`**: Timestamp-based operations (minimal impact)
- **`orch.py`**: Coordination of time-based operations
- Debug logging that references real-time for narrative decisions

## Proposed Solution

### Phase 1: Audit and Classification
**Duration**: 1-2 hours
**Goal**: Identify all time-dependent operations and classify them

#### Tasks:
1. **Code Audit**: Search for all `time.time()`, `datetime.now()`, and timestamp usage across:
   - `sme.py` (story momentum)
   - `sem.py` (semantic logic)
   - `emm.py` (memory manager)
   - `orch.py` (orchestrator)
   - `ui.py` (interface)

2. **Classification Matrix**:
   - **Real-Time Operations**: UI responsiveness, file I/O, network timeouts, thread management
   - **Narrative-Time Operations**: Pressure decay, story progression, momentum analysis
   - **Hybrid Operations**: Logging (needs both), session management

3. **Document Current Behavior**: Map which operations incorrectly use real-time for narrative purposes

### Phase 2: Narrative Time System Design
**Duration**: 2-3 hours
**Goal**: Design clean separation between real-time and narrative-time

#### Core Design Principles:
- **Narrative Time** = progression through story beats/message exchanges
- **Real Time** = system operations and user interface management
- **Clear APIs** = separate time systems with explicit naming

#### Proposed Implementation in `sem.py`:
```python
class NarrativeTimeTracker:
    """Track narrative progression independent of real time"""
    - message_count: int
    - story_beats: List[StoryBeat]
    - pressure_events: List[PressureEvent]
    - last_conflict: Optional[int]  # message index, not timestamp
    
class RealTimeTracker:
    """Track real-world time for system operations"""
    - session_start: datetime
    - last_save: datetime
    - ui_refresh_rate: float
```

### Phase 3: SME Refactoring with Semantic Logic Separation
**Duration**: 4-6 hours
**Goal**: Remove real-time dependencies from narrative systems

#### 3.1: Pressure Decay Redesign (in `sme.py` using `sem.py`)
**Current**: `decay = self.pressure_decay_rate * (time_delta / 60.0)`
**New**: Pressure decay based on narrative progression

**Implementation Strategy**:
- Move decay logic to `sem.py` as shared semantic logic
- Replace time-based decay with message-count-based decay
- Decay pressure after N non-escalating messages
- Maintain pressure during active story sequences

#### 3.2: Rate Limiting Redesign (in `sem.py`)
**Current**: 2-second cooldown prevents rapid analysis
**New**: Content-based spam prevention

**Implementation Strategy**:
- Create spam detection in `sem.py` as shared logic
- Replace time-based cooldown with duplicate content detection
- Allow rapid exchanges during dramatic moments
- Rate limit only identical or near-identical inputs

#### 3.3: Analysis Timing Correction (coordinated by `orch.py`)
**Current**: Some analysis uses real timestamps
**New**: All narrative analysis uses story progression markers

**Implementation**:
- `orch.py` manages when to trigger narrative analysis
- Based on message count, not elapsed time
- Coordinate between `sme.py` and `emm.py` using narrative markers

### Phase 4: Enhanced Narrative Time Tracking
**Duration**: 2-4 hours
**Goal**: Implement sophisticated narrative progression tracking in `sem.py`

#### 4.1: Story Beat Detection (in `sem.py`)
Implement as shared semantic logic:
- **Conflict Escalation**: Combat, tension, mystery discovery
- **Resolution Sequences**: Problem solving, social success, revelation
- **Transition Periods**: Travel, preparation, character development
- **Stagnation Detection**: Repeated actions, circular conversations

#### 4.2: Context-Aware Progression (in `sem.py`)
- **Active Sequences**: Maintain/escalate pressure during dramatic moments
- **Natural Breaks**: Allow pressure decay during genuine story lulls
- **Player Behavior**: Adapt to exploration vs action vs social patterns

### Phase 5: Implementation and Testing
**Duration**: 3-4 hours
**Goal**: Implement changes with thorough testing

#### 5.1: Core Changes
1. **`sem.py`**: Add NarrativeTimeTracker and story beat detection
2. **`sme.py`**: Refactor to use narrative time from `sem.py`
3. **`emm.py`**: Update to track narrative markers
4. **`orch.py`**: Coordinate narrative vs real time operations
5. **`ui.py`**: Ensure UI remains responsive with real-time

#### 5.2: Test Scenarios
**Narrative Time Tests**:
- Rapid player inputs don't get rate-limited
- Long thinking pauses don't reduce story pressure
- Pressure decay based on message exchanges
- Story beats detected correctly

**Real Time Tests**:
- UI remains responsive
- File saves work correctly
- Network timeouts function properly
- Debug logging includes both time types

### Phase 6: Orchestrator Integration
**Duration**: 2-3 hours
**Goal**: Update `orch.py` to manage dual time systems

#### 6.1: Orchestrator Responsibilities
- Initialize both time tracking systems
- Route time queries to appropriate tracker
- Coordinate narrative analysis triggers
- Manage debug logging with both timestamps

#### 6.2: Module Communication
- `orch.py` → `sem.py`: Query narrative time
- `orch.py` → `ui.py`: Provide real time for display
- `sme.py` → `sem.py`: Get narrative progression
- `emm.py` → `sem.py`: Store with narrative markers

## Module Responsibility Matrix (NEW ARCHITECTURE)

### `sem.py` (Semantic Logic - NEW)
- **NarrativeTimeTracker**: Track story progression
- **Story Beat Detection**: Identify narrative moments
- **Pressure Decay Logic**: Calculate based on narrative time
- **Spam Detection**: Content-based rate limiting

### `sme.py` (Story Momentum Engine)
- Use `sem.py` for all time-based calculations
- Store state with narrative markers
- Remove direct `time.time()` usage

### `emm.py` (Enhanced Memory Manager)
- Track messages with narrative indices
- Use real time only for file operations
- Store narrative markers with messages

### `orch.py` (Main Orchestrator)
- Manage both time systems
- Coordinate time-based triggers
- Route time queries appropriately

### `ui.py` (UI Controller)
- Use real time for display updates
- Show narrative progression indicators
- Maintain UI responsiveness

## Implementation Priority

### Critical Path
1. **Pressure Decay Fix** (Phase 3.1) - Most impactful for narrative quality
2. **Rate Limiting Fix** (Phase 3.2) - Improves rapid exchanges
3. **Story Beat Detection** (Phase 4.1) - Enhances narrative awareness

### Secondary Path
1. **Enhanced Tracking** (Phase 4.2) - Sophisticated progression
2. **Orchestrator Integration** (Phase 6) - Clean architecture

## Technical Considerations

### Backward Compatibility
- Saved games with timestamps need migration
- Debug logs should include both time types
- Configuration may need new narrative time settings

### Performance Impact
- Narrative tracking adds minimal overhead
- Story beat detection runs only on new messages
- No impact on UI responsiveness

## Risk Assessment

### Low Risk
- Adding narrative time tracking (new feature)
- Debug logging enhancements (additional info)

### Medium Risk
- Pressure decay changes (affects game balance)
- Save file migration (data compatibility)

### High Risk
- Complete time system separation (architectural change)

### Mitigation Strategies
- **Feature Flag**: Enable/disable narrative time system
- **Migration Tool**: Convert old saves to new format
- **Extensive Testing**: Validate narrative progression
- **Rollback Plan**: Keep time-based logic as fallback

## Success Criteria

### Functional Requirements
- [ ] Player thinking time doesn't reduce story pressure
- [ ] Rapid inputs aren't artificially rate-limited
- [ ] Pressure decay based on narrative progression
- [ ] Story beats detected and tracked correctly
- [ ] Real-time operations (saves, UI) still work

### Quality Requirements
- [ ] No performance regression
- [ ] Clean separation of time systems
- [ ] Clear debug logging with both time types
- [ ] Backward compatibility with saved games

### Narrative Improvements
- [ ] More consistent story tension
- [ ] Better pacing for dramatic moments
- [ ] Natural pressure progression
- [ ] No artificial timing constraints

## Dependencies on Remodularization

This plan assumes the remodularization is complete:
- `sem.py` exists for shared semantic logic
- `orch.py` coordinates between modules
- `sme.py` and `emm.py` use `sem.py` for shared logic

If remodularization is not complete:
- Implement changes directly in `sme.py`
- Add temporary time tracking to `sme.py`
- Migrate to `sem.py` during remodularization

## Estimated Total Effort
**14-21 hours** across 6 phases, with highest priority on fixing pressure decay and rate limiting to improve narrative quality immediately.
