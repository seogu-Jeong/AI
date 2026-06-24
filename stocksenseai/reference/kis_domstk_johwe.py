주식정정취소가능주문내역조회

[국내주식] 주문/계좌 > 주식정정취소가능주문내역조회 API 를 활용하여 주문의 취소 및 정정 주문 정보를 확인

#kis_domstk module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_domstk.py 가 있는 폴더를 추가해준다.
import kis_auth as ka
import kis_domstk as kb

import pandas as pd

import sys

# 토큰 발급
ka.auth()

# [국내주식] 주문/계좌 > 주식정정취소가능주문내역조회
rt_data = kb.get_inquire_psbl_rvsecncl_lst()
print(rt_data)

get_inquire_psbl_rvsecncl_lst ([국내주식] 주문/계좌 > 주식정정취소가능주문내역조회)는 imort 샘플 파일 kis_domstk.py에 아래와 같이 정리되어 있으니 필요시 수정하여 사용하시면 됩니다.

####################################################################################
# [국내주식] 주문/계좌 > 주식정정취소가능주문조회[v1_국내주식-004]
####################################################################################

# 국내주식주문 > 주식정정취소가능주문조회 List를 DataFrame 으로 반환
# Input: None (Option) 상세 Input값 변경이 필요한 경우 API문서 참조
# Output: DataFrame (Option) output API 문서 참조 등
def get_inquire_psbl_rvsecncl_lst(tr_cont="", FK100="", NK100="", dataframe=None):  # 국내주식주문 > 주식정정취소가능주문조회
    url = '/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl'
    tr_id = "TTTC8036R"

    params = {
        "CANO": kis.getTREnv().my_acct,         # 종합계좌번호 8자리
        "ACNT_PRDT_CD": kis.getTREnv().my_prod, # 계좌상품코드 2자리
        "INQR_DVSN_1": "1",                     # 조회구분1(정렬순서)  0:조회순서, 1:주문순, 2:종목순
        "INQR_DVSN_2": "0",                     # 조회구분2 0:전체, 1:매도, 2:매수
        "CTX_AREA_FK100": FK100,                # 공란 : 최초 조회시 이전 조회 Output CTX_AREA_FK100 값 : 다음페이지 조회시(2번째부터)
        "CTX_AREA_NK100": NK100                 # 공란 : 최초 조회시 이전 조회 Output CTX_AREA_NK100 값 : 다음페이지 조회시(2번째부터)
    }

    res = kis._url_fetch(url, tr_id, tr_cont, params)

    # Assuming 'output2' is a dictionary that you want to convert to a DataFrame
    # current_data = res.getBody().output  # getBody() kis_auth.py 존재
    current_data = pd.DataFrame(res.getBody().output)

    # Append to the existing DataFrame if it exists
    if dataframe is not None:
        dataframe = pd.concat([dataframe, current_data], ignore_index=True)  #
    else:
        dataframe = current_data

    tr_cont, FK100, NK100 = res.getHeader().tr_cont, res.getBody().ctx_area_fk100, res.getBody().ctx_area_nk100 # 페이징 처리 getHeader(), getBody() kis_auth.py 존재
    # print(tr_cont, FK100, NK100)

    if tr_cont == "D" or tr_cont == "E": # 마지막 페이지
        print("The End")
        current_data = pd.DataFrame(dataframe)
        dataframe = current_data
        return dataframe
    elif tr_cont == "F" or tr_cont == "M": # 다음 페이지 존재하는 경우 자기 호출 처리
        print('Call Next')
        return get_inquire_psbl_rvsecncl_lst("N", FK100, NK100, dataframe)