#!/usr/bin/env python3
"""
Simple test to check if the LLM worker thread is actually running
and processing requests from the queue
"""

import sys
import time
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_queue_processing():
    """Test if LLM queue is actually working"""
    try:
        from orch import Orchestrator, LLMRequest, LLMRequestType
        
        print("Creating orchestrator...")
        config = {
            'mcp': {
                'server_url': 'http://localhost:3456/chat',
                'model': 'qwen3:8b-q4_K_M', 
                'timeout': 300
            }
        }
        
        # Create with minimal prompts
        orch = Orchestrator(config, {'critrules': 'Test system prompt'})
        
        print("Checking initial queue state...")
        print(f"Queue size: {orch.state.llm_queue.qsize()}")
        print(f"Worker thread exists: {orch.state.llm_worker_thread is not None}")
        
        if orch.state.llm_worker_thread:
            print(f"Worker thread alive: {orch.state.llm_worker_thread.is_alive()}")
        
        print(f"Shutdown event set: {orch.state.llm_worker_shutdown.is_set()}")
        
        # Create a test request
        print("\nCreating test request...")
        test_request = LLMRequest(
            request_type=LLMRequestType.USER_RESPONSE,
            user_input="test",
            context_data={},
            request_id="test_001",
            timestamp=time.time(),
            priority=1
        )
        
        print("Adding request to queue...")
        orch.state.llm_queue.put((test_request.priority, test_request))
        
        print(f"Queue size after adding: {orch.state.llm_queue.qsize()}")
        
        # Wait a moment and check if it was processed
        print("Waiting 3 seconds...")
        time.sleep(3)
        
        print(f"Queue size after wait: {orch.state.llm_queue.qsize()}")
        print(f"Completed requests: {len(orch.state.completed_requests)}")
        print(f"Failed requests: {len(orch.state.failed_requests)}")
        
        if orch.state.completed_requests:
            print("Completed request IDs:", list(orch.state.completed_requests.keys()))
        if orch.state.failed_requests:
            print("Failed request IDs:", list(orch.state.failed_requests.keys()))
            
        # Check if worker is still alive
        if orch.state.llm_worker_thread:
            print(f"Worker still alive: {orch.state.llm_worker_thread.is_alive()}")
            
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== LLM QUEUE TEST ===")
    test_queue_processing()
