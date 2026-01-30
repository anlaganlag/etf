from gm.api import *
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def init(context):
    print("--- Account Inspection ---")
    acc = context.account()
    print(f"Account type: {type(acc)}")
    methods = [m for m in dir(acc) if not m.startswith('_')]
    print(f"Account methods: {methods}")
    
    # Try to find something order related in acc
    order_methods = [m for m in methods if 'order' in m.lower()]
    print(f"Order-related methods: {order_methods}")
    
    try:
        print(f"Testing positions(): {acc.positions()[:1]}")
    except Exception as e:
        print(f"Testing positions() failed: {e}")
        
    try:
        print(f"Testing orders(): {acc.orders(status=OrderStatus_New)[:1]}")
    except Exception as e:
        print(f"Testing orders() failed: {e}")
        
    os._exit(0)

if __name__ == '__main__':
    run(strategy_id='d6d71d85-fb4c-11f0-99de-00ffda9d6e63', 
        filename='inspect_account.py', 
        mode=MODE_LIVE, 
        token=os.getenv('MY_QUANT_TGM_TOKEN'))
