import requests

def read_p(city_name):
    city_url = {'重庆': 'https://etax.chongqing.chinatax.gov.cn/Kaptcha.jpg?r=0.0405208157223127',
            '上海': 'https://etax.shanghai.chinatax.gov.cn/login-web/api/googleCaptcha?1598428694905'}
    if city_name=='重庆':
        url=city_url['重庆']
        headers = {'Accept': 'image/webp,*/*',
                   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"}
        try:
            req = requests.get(url, headers=headers,timeout=10)
            if requests.status_codes == 200:
                    pass
        except requests.exceptions.Timeout:
            print('由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。')
        return req.content
    elif city_name=='上海':
        url = city_url['上海']
        headers = {'Accept': 'image/webp,*/*',
                   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"}
        try:
            req = requests.get(url, headers=headers,timeout=10)
            if requests.status_codes == 200:
                    pass
        except requests.exceptions.Timeout:
            print('由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。')
        return req.content



