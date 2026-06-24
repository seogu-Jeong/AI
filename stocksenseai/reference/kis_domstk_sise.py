국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년) 정보를 가져와보겠습니다. 국내주식기간별시세(일/주/월/년) API는 조회하고 싶은 종목의 일자별 현재가, 거래량, 상한가, 하한가, 시/고/저가 등을 얻는데 사용됩니다.

자동 프로그램 개발시 원하는 종목의 국내주식기간별시세(현재) 거래량 정보는 아래와 같이 API 호출하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년) (종목번호 6자리)
rt_data = kb.get_inquire_asking_price_exp_ccn(itm_no="071050")
print(rt_data)

원하는 종목의 국내주식기간별시세의 현재 정보는 아래와 같이 API 호출하시면 됩니다.

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년) (기간별 데이터 Default는 일별이며 조회기간은 100일전(영업일수 아님)부터 금일까지)
#rt_data_obj = kb.get_inquire_daily_itemchartprice(output_dv="2", itm_no="071050")
print(rt_data)

get_inquire_daily_itemchartprice ([국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년))는 imort 샘플 파일 kis_domstk.py에 아래와 같이 정리되어 있으니 필요시 수정하여 사용하시면 됩니다.

####################################################################################
# [국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년)
# 국내주식기간별시세(일/주/월/년) API입니다.
# 실전계좌/모의계좌의 경우, 한 번의 호출에 최대 100건까지 확인 가능합니다.
####################################################################################
# 국내주식기간별시세(일/주/월/년) Object를 DataFrame 으로 반환
# Input: None (Option) 상세 Input값 변경이 필요한 경우 API문서 참조
# Output: DataFrame (Option) output
def get_inquire_daily_itemchartprice(div_code="J", itm_no="", tr_cont="", inqr_strt_dt=None, inqr_end_dt=None, period_code="D", adj_prc="1", FK100="", NK100="", dataframe=None):  # [국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년)
    url = '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
    tr_id = "FHKST03010100"  # 주식현재가 회원사

    if inqr_strt_dt is None:
        inqr_strt_dt = (datetime.now()-timedelta(days=14)).strftime("%Y%m%d")   # 시작일자 값이 없으면 현재일자
    if inqr_end_dt is None:
        inqr_end_dt  = datetime.today().strftime("%Y%m%d")   # 종료일자 값이 없으면 현재일자

    print(inqr_strt_dt)
    print(inqr_end_dt)
    params = {
        "FID_COND_MRKT_DIV_CODE": div_code, # 시장 분류 코드  J : 주식/ETF/ETN, W: ELW
        "FID_INPUT_ISCD": itm_no,           # 종목번호 (6자리) ETN의 경우, Q로 시작 (EX. Q500001)
        "FID_INPUT_DATE_1": inqr_strt_dt,   # 입력 날짜 (시작) 조회 시작일자 (ex. 20220501)
        "FID_INPUT_DATE_2": inqr_end_dt,    # 입력 날짜 (종료) 조회 종료일자 (ex. 20220530)
        "FID_PERIOD_DIV_CODE": period_code, # 기간분류코드 D:일봉, W:주봉, M:월봉, Y:년봉
        "FID_ORG_ADJ_PRC": adj_prc          # 수정주가 0:수정주가 1:원주가
    }
    res = kis._url_fetch(url, tr_id, tr_cont, params)

    # print(res.getBody())  # 오류 원인 확인 필요시 사용
    # Assuming 'output' is a dictionary that you want to convert to a DataFrame
    current_data = pd.DataFrame(res.getBody().output1, index=[0])  # 호가조회  * getBody() kis_auth.py 존재

    dataframe = current_data

    return dataframe
