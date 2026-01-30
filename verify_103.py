import subprocess
import os
import sys

script_path = 'gm_strategy_rolling0.py'
cwd = os.path.dirname(os.path.abspath(__file__))

env = os.environ.copy()
env['GM_TOP_N'] = '3'
env['GM_REBALANCE_T'] = '11'
env['GM_START_DATE'] = '2023-01-01 09:00:00'
env['GM_STOP_LOSS'] = '0.20'
env['GM_TRAILING_TRIGGER'] = '0.15'
env['GM_TRAILING_DROP'] = '0.08'

print("Running validation with explicit Optimal Params:")
print("N=3, T=11, SL=0.20, Trig=0.15, Drop=0.08")

cmd = [sys.executable, script_path]
result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)

print("Exit Code:", result.returncode)
print("Output snippet:")
lines = result.stdout.splitlines()
for line in lines[-20:]:
    print(line)
