
from gm.api import *
import os
from dotenv import load_dotenv
load_dotenv()
set_token(os.getenv('MY_QUANT_TGM_TOKEN'))
df = history(symbol='SHSE.000300', frequency='1d', start_time='2025-01-01 09:00:00', end_time='2025-01-20 16:00:00', df=True)
print(df)
