# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 07:56:54 2022
"""
#kis_api module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_api.py 가 있는 폴더를 추가해준다.
import sys
sys.path.append(r'd:\pysam')

import kis_api as ka
import pandas as pd

ka.auth()

import time



df = ka.get_acct_balance()


stock_code_list = df.index.to_list()
for x in stock_code_list:
    sdf = kae.get_stock_history(x, '20180101')
    print(x, df.loc[x]['종목명'], df.loc[x]['현재가'])
    sdf.Close.plot()
    plt.show()

time.sleep(.1)

print(ka.get_current_price('377990'))    

ka.do_sell('052300',  1,  1100)
time.sleep(1)
ka.do_buy('052300', 1, 950 ) 
time.sleep(1)

df_orders = ka.get_orders()
ka.do_cancel_all()

oll = df_orders.index.to_list()
ka.do_revise(oll[0], 1, 1050)

from datetime import datetime
sdt = datetime.now().strftime('%Y%m%d')
df_complete = ka.get_my_complete(sdt, sdt)

ar = ka.get_buyable_cash()
print(ar)

ocl = ka.get_stock_completed('052300')

hdf1 = ka.get_stock_history('052300')
time.sleep(.5)
hdf2 = ka.get_stock_history('052300', 'W')
time.sleep(.5)
hdf3 = ka.get_stock_history('052300', 'M')
time.sleep(.5)
hdfi = ka.get_stock_investor('052300')

