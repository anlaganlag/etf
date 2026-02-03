
from gm.api import *
import pandas as pd
import numpy as np
import os
import itertools
from dotenv import load_dotenv
from config import config

load_dotenv()

# Fixed Configuration
START_DATE = '2021-12-03 09:00:00'
END_DATE = '2026-01-23 16:00:00'
STRATEGY_ID = '0137c2ac-fd82-11f0-ae68-00ffda9d6e63'
STATE_FILE = "rolling_state_simple.json"
LIVE_DATA_UPDATE = False

# Optimization Search Space (Exploration)
# Focusing on Direction A (Trend) vs Direction B (High Freq)
params_grid = {
    'top_n': [4], # Core
    'T': [8, 10, 12], # Faster vs Normal
    'stop_loss': [0.10, 0.15], # Tight vs Normal
    'trailing_trigger': [0.15, 0.25], # Stable vs Aggressive
    'trailing_drop': [0.05], # Fixed for now
    'dynamic_position': [True]
}

def clean_state():
    state_path = os.path.join(config.BASE_DIR, STATE_FILE)
    if os.path.exists(state_path):
        try: os.remove(state_path)
        except: pass

def modify_config(top_n, t, sl, trig, drop, dynamic):
    # This function is a placeholder. 
    # In a real scenario, we would inject these params into the strategy process via env vars or args.
    # For this script, we will use os.environ to pass params to the child process.
    os.environ['OPT_TOP_N'] = str(top_n)
    os.environ['OPT_T'] = str(t)
    os.environ['OPT_SL'] = str(sl)
    os.environ['OPT_TRIG'] = str(trig)
    os.environ['OPT_DROP'] = str(drop)
    os.environ['OPT_DYNAMIC'] = str(dynamic)

def run_test():
    # Generate all combinations
    keys = params_grid.keys()
    values = params_grid.values()
    combinations = list(itertools.product(*values))
    
    results = []
    print(f"🚀 Starting Exploration: {len(combinations)} combinations...")
    
    # We will run the strategy by modifying gm_strategy_rolling0.py TO ACCEPT ENV VARS first.
    # But since we can't easily modify the running script on the fly without changing code permanently,
    # A better approach for this "Agent" context is to create a temporary strategy file that reads from Env.
    
    pass 

if __name__ == '__main__':
    # This script is just a plan. 
    # To run the exploration, we need to modify gm_strategy_rolling0.py to read params from Os.Environ if available.
    print("Please allow me to modify gm_strategy_rolling0.py to accept environment variables for parameters first.")
