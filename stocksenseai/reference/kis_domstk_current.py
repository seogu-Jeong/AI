먼저, [국내주식] 기본시세 > 주식현재가 시세를 가져와보겠습니다. [국내주식] 기본시세 > 주식현재가 시세 API는 조회하고 싶은 종목의 현재 정보를 얻는데 사용됩니다. 그럼 시작해보겠습니다.

자동 프로그램 개발시 아래와 같이 API 호출하시고 Input 값은 종목코드(itm_no)만 입력하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 주식현재가 시세 (종목번호 6자리)
rt_data = kb.get_inquire_price(itm_no="071050")
print(rt_data.stck_prpr+ " " + rt_data.prdy_vrss)    # 현재가, 전일대비
get_inquire_price ([국내주식] 기본시세 > 주식현재가 시세 )은 imort 샘플 파일 kis_domstk.py에 아래와 같이 정리되어 있으니 필요시 수정하여 사용하시면 됩니다.

##############################################################################################
# [국내주식] 기본시세 > 주식현재가 시세
##############################################################################################
# 주식현재가 시세 Object를 DataFrame 으로 반환
# Input: None (Option) 상세 Input값 변경이 필요한 경우 API문서 참조
# Output: DataFrame (Option) output
def get_inquire_price(div_code="J", itm_no="", tr_cont="", FK100="", NK100="", dataframe=None):  # [국내주식] 기본시세 > 주식현재가 시세
    url = '/uapi/domestic-stock/v1/quotations/inquire-price'
    tr_id = "FHKST01010100" # 주식현재가 시세

    params = {
        "FID_COND_MRKT_DIV_CODE": div_code, # 시장 분류 코드  J : 주식/ETF/ETN, W: ELW
        "FID_INPUT_ISCD": itm_no            #   종목번호 (6자리) ETN의 경우, Q로 시작 (EX. Q500001)
    }
    res = kis._url_fetch(url, tr_id, tr_cont, params)

    # Assuming 'output' is a dictionary that you want to convert to a DataFrame
    current_data = pd.DataFrame(res.getBody().output, index=[0])  # getBody() kis_auth.py 존재

    dataframe = current_data

    return dataframe