import os
import json
from config import config

# Test the exact path construction
STATE_FILE = "rolling_state_simple.json"
state_path = os.path.join(config.BASE_DIR, STATE_FILE)

print(f"BASE_DIR: {config.BASE_DIR}")
print(f"STATE_FILE: {STATE_FILE}")
print(f"state_path: {state_path}")
print(f"state_path length: {len(state_path)}")
print(f"state_path repr: {repr(state_path)}")
print(f"state_path bytes: {state_path.encode('utf-8')}")

# Test if path exists
print(f"\nPath exists: {os.path.exists(state_path)}")
print(f"Is file: {os.path.isfile(state_path)}")

# Try to write
try:
    test_data = {"test": "data"}
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    print("\n✓ Write test PASSED")
    
    # Try to read back
    with open(state_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Read test PASSED: {data}")
    
except Exception as e:
    print(f"\n✗ Test FAILED: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
