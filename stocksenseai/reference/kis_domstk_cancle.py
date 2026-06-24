주식주문(정정/취소)

[국내주식] 주문/계좌 > 주식주문(정정취소) API 를 활용하여 원하는 미체결 주문건을 취소 및 정정할 수 있습니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 주문/계좌 > 주식주문(정정취소) (한국거래소전송주문조직번호 5자리+원주문번호 10자리('0'을 채우지 않아도됨)+정정취소구분코드+정정및취소주문수량+잔량전부주문여부)
# 지정가 기준이며 시장가 옵션(주문구분코드)을 사용하는 경우 kis_domstk.py get_order_rvsecncl 수정요망!
rt_data = kb.get_order_rvsecncl(ord_orgno="06010", orgn_odno="0000224003", ord_dvsn="00", rvse_cncl_dvsn_cd="01", ord_qty=0, ord_unpr=64900, qty_all_ord_yn="Y")
print(rt_data.KRX_FWDG_ORD_ORGNO + "+" + rt_data.ODNO + "+" + rt_data.ORD_TMD) # 주문접수조직번호+주문접수번호+주문시각
