주식일별주문체결(현황)조회

[국내주식] 주문/계좌 > 주식정정취소가능주문내역조회 API 를 활용하여 주문의 취소 및 정정 주문 정보를 확인

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 주문/계좌 > 주식일별주문체결(현황)조회
# dv="01"   01:3개월 이내 국내주식체결내역 (월단위 ex: 2024.04.25 이면 2024.01월~04월조회)
# dv="02"   02:3개월 이전 국내주식체결내역 (월단위 ex: 2024.04.25 이면 2024.01월이전)
rt_data = kb.get_inquire_daily_ccld_obj(dv="01")
print(rt_data)

# [국내주식] 주문/계좌 > 주식일별주문체결(내역)조회
# dv="01"   01:3개월 이내 국내주식체결내역 (월단위 ex: 2024.04.25 이면 2024.01월~04월조회)
# dv="02"   02:3개월 이전 국내주식체결내역 (월단위 ex: 2024.04.25 이면 2024.01월이전)
rt_data = kb.get_inquire_daily_ccld_lst(dv="01")
print(rt_data)

get_inquire_daily_ccld_obj ([국내주식] 주문/계좌 > 주식정정취소가능주문내역조회)는 imort 샘플 파일 kis_domstk.py에 아래와 같이 정리되어 있으니 필요시 수정하여 사용하시면 됩니다.

###################################################################################
# [국내주식] 주문/계좌 > 주식일별주문체결조회
###################################################################################

# 국내주식주문 > 주식일별주문체결조회 Object를 DataFrame 으로 반환
# Input: None (Option) 상세 Input값 변경이 필요한 경우 API문서 참조
#        dv 기간구분 - 01:3개월 이내(TTTC8001R),  02:3개월 이전(CTSC9115R)
# Output: DataFrame (Option) output2 API 문서 참조 등
def get_inquire_daily_ccld_obj(dv="01", inqr_strt_dt=None, inqr_end_dt=None, tr_cont="", FK100="", NK100="", dataframe=None):  # 국내주식주문 > 주식일별주문체결조회
    url = '/uapi/domestic-stock/v1/trading/inquire-daily-ccld'

    if dv == "01":
        tr_id = "TTTC8001R"  # 01:3개월 이내 국내주식체결내역 (월단위 ex: 2024.04.25 이면 2024.01월~04월조회)
    else:
        tr_id = "CTSC9115R"  # 02:3개월 이전 국내주식체결내역 (월단위 ex: 2024.04.25 이면 2024.01월이전)

    if inqr_strt_dt is None:
        inqr_strt_dt = datetime.today().strftime("%Y%m%d")   # 시작일자 값이 없으면 현재일자
    if inqr_end_dt is None:
        inqr_end_dt  = datetime.today().strftime("%Y%m%d")   # 종료일자 값이 없으면 현재일자

    params = {
        "CANO": kis.getTREnv().my_acct,         # 종합계좌번호 8자리
        "ACNT_PRDT_CD": kis.getTREnv().my_prod, # 계좌상품코드 2자리
        "INQR_STRT_DT": inqr_strt_dt,           # 조회시작일자
        "INQR_END_DT": inqr_end_dt,             # 조회종료일자
        "SLL_BUY_DVSN_CD": "00",                # 매도매수구분코드 00:전체 01:매도, 02:매수
        "INQR_DVSN": "01",                      # 조회구분(정렬순서)  00:역순, 01:정순
        "PDNO": "",                             # 종목번호(6자리)
        "CCLD_DVSN": "00",                      # 체결구분 00:전체, 01:체결, 02:미체결
        "ORD_GNO_BRNO": "",                     # 사용안함
        "ODNO": "",                             # 주문번호
        "INQR_DVSN_3": "00",                    # 조회구분3 00:전체, 01:현금, 02:융자, 03:대출, 04:대주
        "INQR_DVSN_1": "0",                     # 조회구분1 공란 : 전체, 1 : ELW, 2 : 프리보드
        "CTX_AREA_FK100": FK100,                # 공란 : 최초 조회시 이전 조회 Output CTX_AREA_FK100 값 : 다음페이지 조회시(2번째부터)
        "CTX_AREA_NK100": NK100                 # 공란 : 최초 조회시 이전 조회 Output CTX_AREA_NK100 값 : 다음페이지 조회시(2번째부터)
    }

    res = kis._url_fetch(url, tr_id, tr_cont, params)

    # Assuming 'output2' is a dictionary that you want to convert to a DataFrame
    current_data = pd.DataFrame(res.getBody().output2, index=[0])

    dataframe = current_data

    return dataframe