[국내주식] 기본시세 > 주식현재가 호가/예상체결 정보를 가져와보겠습니다.[국내주식] 기본시세 > 주식현재가 호가/예상체결 API는 조회하고 싶은 종목의 현재의 호가와 예상체결가 정보를 얻는데 사용됩니다.

자동 프로그램 개발시 원하는 종목의 1호가~10호가, 1호~10호가별 잔량 정보는 아래와 같이 API 호출하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 주식현재가 호가 (종목번호 6자리)
rt_data = kb.get_inquire_asking_price_exp_ccn(itm_no="071050")
print(rt_data)

또 종목의 예상체결가 정보는 아래와 같이 API 호출하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 주식현재가 예상체결 (출력구분="2" + 종목번호 6자리)
rt_data = kb.get_inquire_asking_price_exp_ccn(output_dv="2", itm_no="071050")
print(rt_data)

get_inquire_asking_price_exp_ccn ([국내주식] 기본시세 > 주식현재가 호가/예상체결)는 imort 샘플 파일 kis_domstk.py에 아래와 같이 정리되어 있으니 필요시 수정하여 사용하시면 됩니다.

##############################################################################################
# [국내주식] 기본시세 > 주식현재가 호가/예상체결
# 주식현재가 호가 예상체결 API입니다. 매수 매도 호가를 확인하실 수 있습니다. 실시간 데이터를 원하신다면 웹소켓 API를 활용하세요.
##############################################################################################
# 주식현재가 일자별 Object를 DataFrame 으로 반환
# Input: None (Option) 상세 Input값 변경이 필요한 경우 API문서 참조
# Output: DataFrame (Option) output
def get_inquire_asking_price_exp_ccn(output_dv='1', div_code="J", itm_no="", tr_cont="", FK100="", NK100="", dataframe=None):  # [국내주식] 기본시세 > 주식현재가 호가/예상체결
    url = '/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn'
    tr_id = "FHKST01010200"  # 주식현재가 호가 예상체결

    params = {
        "FID_COND_MRKT_DIV_CODE": div_code, # 시장 분류 코드  J : 주식/ETF/ETN, W: ELW
        "FID_INPUT_ISCD": itm_no           # 종목번호 (6자리) ETN의 경우, Q로 시작 (EX. Q500001)
    }
    res = kis._url_fetch(url, tr_id, tr_cont, params)

    # print(res.getBody())  # 오류 원인 확인 필요시 사용
    # Assuming 'output1' is a dictionary that you want to convert to a DataFrame
    if output_dv == "1":
        current_data = pd.DataFrame(res.getBody().output1, index=[0])  # 호가조회  * getBody() kis_auth.py 존재
    else:
        current_data = pd.DataFrame(res.getBody().output2, index=[0])  # 예상체결가조회

    dataframe = current_data

    return dataframe
