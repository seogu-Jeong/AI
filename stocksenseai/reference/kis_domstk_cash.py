주식현금주문

주식주문(현금) API 를 활용하여 원하는 종목을 매수/매도할 수 있습니다. 그럼 본격적으로 주식 매매를 시작해보겠습니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 주문/계좌 > 주식현금주문 (매수매도구분 buy,sell + 종목번호 6자리 + 주문수량 + 주문단가)
# 지정가 기준이며 시장가 옵션(주문구분코드)을 사용하는 경우 kis_domstk.py get_order_cash 수정요망!
rt_data = kb.get_order_cash(ord_dv="buy",itm_no="071050", qty=10, unpr=65000)
print(rt_data.KRX_FWDG_ORD_ORGNO + "+" + rt_data.ODNO + "+" + rt_data.ORD_TMD) # 주문접수조직번호+주문접수번호+주문시각