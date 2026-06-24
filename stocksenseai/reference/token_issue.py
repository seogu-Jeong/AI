클라이언트 인증(2-legged) 방식 적용 동 절차는 OAuth 2.0의 Client Credentials Grant 절차를 준용하여 개발 토큰 유효기간 1일 이며 매번 API 신청시 발급된 Appkey와 Appsecret를 사용하여 발급(발급시 알림톡 발송)

제휴사는 3-legged 방식 적용 (Oauth 2.0의 Authorization Code Grant 절차에 준용하여 개발)


#prod 실전투자, product 투자(위탁)계좌
def auth(svr='prod', product='01'):
    p = {
        "grant_type": "client_credentials",
    }
    print(svr)
    #'prod':실전투자, 'vps':모의투자
    if svr == 'prod':
        ak1 = 'my_app'
        ak2 = 'my_sec'
    elif svr == 'vps':
        ak1 = 'paper_app'
        ak2 = 'paper_sec'

    p["appkey"] = _cfg[ak1]
    p["appsecret"] = _cfg[ak2]

    #아래 방식은 Qeury string 방식이며 URI 와 post 호출방식으로 /oauth2/tokenP로 대체하여 사용
    url = f'{_cfg[svr]}/oauth2/token'
    res = requests.post(url, params=p, headers=_getBaseHeader())
    #url = f'{_cfg[svr]}/oauth2/tokenP'
    #res = requests.post(url, data=json.dumps(p), headers=_getBaseHeader())

    rescode = res.status_code
    if rescode == 200:
        my_token = _getResultObject(res.json()).access_token
    else:
        print('Get Authentification token fail!\nYou have to restart your app!!!')
        return

    changeTREnv(f"Bearer {my_token}", svr, product)

    _base_headers["authorization"] = _TRENV.my_token
    _base_headers["appkey"] = _TRENV.my_app
    _base_headers["appsecret"] = _TRENV.my_sec

    global _last_auth_time
    _last_auth_time = datetime.now()

    if (_DEBUG):
        print(f'[{_last_auth_time}] => get AUTH Key completed!')