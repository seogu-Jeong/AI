주문/정정/취소는 거래요청 과정에서 전문의 위변조 방지를 위하여 당사에서 제공하는 Hash알고리즘으로 암호화하여 처리 암호화 대상은 Input body 부분을 Hashkey생성하여 주문/정정/취소 API거래시 header에 적용

#주문 API에서 사용할 hash key값을 받아 header에 설정해 주는 함수
#Input: HTTP Header, HTTP post param
#Output: None
def set_order_hash_key(h, p):
    url = f"{getTREnv().my_url}/uapi/hashkey"

    res = requests.post(url, data=json.dumps(p), headers=h)
    rescode = res.status_code
    if rescode == 200:
        h['hashkey'] = _getResultObject(res.json()).HASH
    else:
        print("Error:", rescode)
