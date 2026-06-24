#end of initialize 토큰 유효기간(1일) 만료 재발급
def reAuth(svr='prod', product='01'):
    n2 = datetime.now()
    if (n2 - _last_auth_time).seconds >= 86400:
        auth(svr, product)
