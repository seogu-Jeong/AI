[국내주식] 기본시세 > 주식현재가 당일시간대별체결 정보를 가져와보겠습니다. 주식현재가 당일시간대별체결 API는 조회하고 싶은 종목의 당일 시간대별 체결 정보를 얻는데 사용됩니다.

자동 프로그램 개발시 원하는 종목의 [국내주식] 기본시세 > 주식현재가 당일체결 정보는 아래와 같이 API 호출하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 주식현재가 당일시간대별체결 (현재가 : 주식현재가, 전일대비, 전일대비율, 누적거래량,전일거래량, 대표시장한글명))
rt_data = kb.get_inquire_time_itemconclusion(itm_no="071050")
print(rt_data)

종목의 [국내주식] 기본시세 > 주식현재가 당일시간대별체결 정보는 아래와 같이 API 호출하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 주식현재가 당일시간대별체결 (시간대별체결내역)
rt_data = kb.get_inquire_time_itemconclusion(output_dv='2', itm_no="071050")  # 기준시각 미지정시 현재시각 이전 체결 내역이 30건 조회됨
#rt_data = kb.get_inquire_time_itemconclusion(output_dv='2', itm_no="071050", inqr_hour='100000') # 지정 기준시각 이전 체결 내역이 30건 조회됨
print(rt_data)

get_inquire_time_itemconclusion ([국내주식] 기본시세 > 주식현재가 당일시간대별체결)는 imort 샘플 파일 kis_domstk.py에 아래와 같이 정리되어 있으니 필요시 수정하여 사용하시면 됩니다.



####################################################################################
# [국내주식] 기본시세 > 주식현재가 당일시간대별체결
# 기준시각(HHMMSS) 이전 체결 내역 30건 조회됨 (시간 미지정시 현재시각 기준)
####################################################################################
# 주식현재가 당일시간대별체결 Object를 DataFrame 으로 반환
# Input: None (Option) 상세 Input값 변경이 필요한 경우 API문서 참조
# Output: DataFrame (Option) output
def get_inquire_time_itemconclusion(output_dv="1", div_code="J", itm_no="", inqr_hour=None, tr_cont="", FK100="", NK100="", dataframe=None):  # [국내주식] 기본시세 > 주식현재가 당일시간대별체결
    url = '/uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion'
    tr_id = "FHPST01060000"  # 주식현재가 당일시간대별체결

    if inqr_hour is None:
        now = datetime.now()

        # 시, 분, 초 추출
        hour = now.hour
        minute = now.minute
        second = now.second

        # HHMMSS 형식으로 조합
        inqr_hour  = f"{hour:02d}{minute:02d}{second:02d}" # 현재 시간 가져오기

    params = {
        "FID_COND_MRKT_DIV_CODE": div_code, # 시장 분류 코드  J : 주식/ETF/ETN, W: ELW
        "FID_INPUT_ISCD": itm_no,           # 종목번호 (6자리) ETN의 경우, Q로 시작 (EX. Q500001)
        "FID_INPUT_HOUR_1": inqr_hour       # 기준시간 (6자리; HH:MM:SS) ex) 155000 입력시 15시 50분 00초 기준 이전 체결 내역이 조회됨
    }
    res = kis._url_fetch(url, tr_id, tr_cont, params)

    # print(res.getBody())  # 오류 원인 확인 필요시 사용
    # Assuming 'output' is a dictionary that you want to convert to a DataFrame
    if output_dv == "1":
        current_data = pd.DataFrame(res.getBody().output1, index=[0])  # 호가조회  * getBody() kis_auth.py 존재
    else:
        current_data = pd.DataFrame(res.getBody().output2)  # 호가조회  * getBody() kis_auth.py 존재

    dataframe = current_data

    return dataframe