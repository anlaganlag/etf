
import difflib

with open('gm_strategy_rolling0.py', 'r', encoding='utf-8') as f1, \
     open('gm_3766_return.py', 'r', encoding='utf-8') as f2:
    diff = difflib.unified_diff(
        f1.readlines(),
        f2.readlines(),
        fromfile='gm_strategy_rolling0.py',
        tofile='gm_3766_return.py',
    )
    print("".join(diff))
