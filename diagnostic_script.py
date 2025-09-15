#!/usr/bin/env python3
"""
Diagnostic script to check which methods are missing or incomplete in orch.py
Run this to identify exactly what's causing the timeout
"""

import sys
import inspect
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def check_orchestrator_methods():
    """Check if critical methods exist and are implemented"""
    try:
        from orch import Orchestrator
        
        # List of critical methods that MUST exist and be implemented
        critical_methods = [
            '_process_user_response_request',
            '_process_semantic_analysis_request', 
            '_process_condensation_request',
            '_record_completed_request',
            '_record_failed_request',
            '_check_resolution_guidance',
            '_make_llm_request',
            '_generate_request_id'
        ]
        
        print("=== ORCHESTRATOR METHOD DIAGNOSTIC ===")
        print(f"Checking {len(critical_methods)} critical methods...")
        
        missing_methods = []
        incomplete_methods = []
        
        for method_name in critical_methods:
            if hasattr(Orchestrator, method_name):
                method = getattr(Orchestrator, method_name)
                
                # Check if method is implemented (has more than just 'pass' or 'raise NotImplementedError')
                source = inspect.getsource(method)
                
                if ('pass' in source and len(source.strip().split('\n')) < 5) or 'NotImplementedError' in source:
                    incomplete_methods.append(method_name)
                    print(f"❌ {method_name}: EXISTS but INCOMPLETE")
                else:
                    print(f"✅ {method_name}: IMPLEMENTED")
            else:
                missing_methods.append(method_name)
                print(f"❌ {method_name}: MISSING")
        
        print(f"\n=== SUMMARY ===")
        print(f"Missing methods: {len(missing_methods)}")
        print(f"Incomplete methods: {len(incomplete_methods)}")
        
        if missing_methods:
            print(f"\nMissing: {', '.join(missing_methods)}")
        if incomplete_methods:
            print(f"\nIncomplete: {', '.join(incomplete_methods)}")
            
        return len(missing_methods) == 0 and len(incomplete_methods) == 0
        
    except ImportError as e:
        print(f"❌ Cannot import Orchestrator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking methods: {e}")
        return False

def check_interface_methods():
    """Check if interface methods exist in SME and EMM"""
    try:
        from sme import StoryMomentumEngine
        from emm import EnhancedMemoryManager
        
        print(f"\n=== MODULE INTERFACE DIAGNOSTIC ===")
        
        # Check SME interface
        sme_methods = ['get_stats', 'reset_state', 'check_resolution_trigger']
        print("SME methods:")
        for method in sme_methods:
            if hasattr(StoryMomentumEngine, method):
                print(f"✅ {method}: EXISTS")
            else:
                print(f"❌ {method}: MISSING")
        
        # Check EMM interface  
        emm_methods = ['get_stats', 'reset_state']
        print("EMM methods:")
        for method in emm_methods:
            if hasattr(EnhancedMemoryManager, method):
                print(f"✅ {method}: EXISTS") 
            else:
                print(f"❌ {method}: MISSING")
                
    except ImportError as e:
        print(f"❌ Cannot import modules: {e}")
    except Exception as e:
        print(f"❌ Error checking interfaces: {e}")

def check_queue_implementation():
    """Check if LLM queue and worker thread are properly implemented"""
    try:
        from orch import Orchestrator, OrchestrationState, LLMRequest, LLMRequestType
        
        print(f"\n=== QUEUE IMPLEMENTATION DIAGNOSTIC ===")
        
        # Check if LLM queue classes exist
        classes_to_check = ['OrchestrationState', 'LLMRequest', 'LLMRequestType']
        for class_name in classes_to_check:
            try:
                exec(f"from orch import {class_name}")
                print(f"✅ {class_name}: EXISTS")
            except ImportError:
                print(f"❌ {class_name}: MISSING")
        
        # Check if OrchestrationState has queue fields
        state = OrchestrationState()
        queue_fields = ['llm_queue', 'llm_worker_thread', 'llm_worker_shutdown']
        for field in queue_fields:
            if hasattr(state, field):
                print(f"✅ {field}: EXISTS")
            else:
                print(f"❌ {field}: MISSING")
                
    except Exception as e:
        print(f"❌ Error checking queue implementation: {e}")

def main():
    """Run all diagnostics"""
    print("DevName RPG Client - Timeout Diagnostic")
    print("=" * 50)
    
    methods_ok = check_orchestrator_methods()
    check_interface_methods()
    check_queue_implementation()
    
    print(f"\n=== DIAGNOSIS ===")
    if methods_ok:
        print("✅ All critical methods appear to be implemented")
        print("Timeout likely caused by other issues (network, infinite loops, deadlocks)")
    else:
        print("❌ CRITICAL METHODS MISSING OR INCOMPLETE")
        print("This is the likely cause of the timeout on second user input")
        print("\nThe LLM worker thread is probably hanging because it's trying to call")
        print("missing or incomplete methods like _process_semantic_analysis_request")
    
    print(f"\n=== RECOMMENDED ACTION ===")
    if not methods_ok:
        print("1. Add missing method implementations to orch.py")
        print("2. Use simplified Phase 1 implementations that don't hang")
        print("3. Test with a simple input after fixes")
    else:
        print("1. Check for infinite loops in existing method implementations")
        print("2. Add more debug logging to LLM worker thread")
        print("3. Check MCP client connection and timeouts")

if __name__ == "__main__":
    main()
