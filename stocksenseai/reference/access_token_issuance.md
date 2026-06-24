# 접근 코드 발급

APP_KEY = "홈페이지에서 발급받은 App Key"
APP_SECRET = "홈페이지에서 발급받은 App Secret"
URL_BASE = "https://openapivts.koreainvestment.com:29443" #모의투자서비스
먼저 발급받은 App Key와 App Secret을 변수에 저장합니다. 본 예제에서는 모의계좌를 활용할 예정이므로 모의투자서비스 주소를 변수에 저장합니다.

headers = {"content-type":"application/json"}
body = {"grant_type":"client_credentials",
        "appkey":APP_KEY, 
        "appsecret":APP_SECRET}
PATH = "oauth2/tokenP"
REST API는 크게 4가지로 이루어져 있습니다. ①권한 인증 등에 활용되는 header, ②위치를 나타내는 path, ③쿼리문을 활용한 query string, ④requsest 요청에 포함되는 body로 이루어져 있습니다. 일반적으로 ③query string은 GET 방식 ④body는 POST 방식에 활용됩니다.

우리는 POST 방식을 활용하여 보안인증키를 발급받을 예정이므로, headers와 data를 각각 dictionary 형태로 만들어줍니다. 마지막으로 호출할 API 의 위치를 PATH 변수에 저장합니다.

URL = f"{URL_BASE}/{PATH}"
print(URL)
>>> https://openapivts.koreainvestment.com:29443/oauth2/token
URL_BASE와 PATH를 합쳐 URL로 만들어줍니다. 쉽게 생각하면 파일 디렉토리와 같은데, URL_BASE인 상위 모의투자 파일에서 ouath2로 들어가 token이라는 파일로 들어간 것과 같습니다.

res = requests.post(URL, headers=headers, data=json.dumps(body))
res.text
>>> '{"access_token":"ACCESS_TOKEN","token_type":"Bearer","expires_in":86400}'
이제 request 요청을 해보겠습니다. 위와 같이 POST 요청을 진행하면 보안인증키(access_token)를 받을 수 있습니다.

ACCESS_TOKEN = res.json()["access_token"]
print(ACCESS_TOKEN)
>>> ACCESS_TOKEN
