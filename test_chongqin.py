from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.action_chains import ActionChains
import requests
import numpy as np
import cv2
from PIL import Image

# driver = webdriver.Firefox()
# wait = ui.WebDriverWait(driver, 3)
# driver.get('https://etax.shanghai.chinatax.gov.cn/wszx-web/bszm/apps/views/beforeLogin/indexBefore/pageIndex.html#/')
# # img_pct=driver.find_element_by_xpath("//*[@id='image_qy']")
# img_pct = driver.find_element_by_xpath("//*[@class='ml10 fr']")
#
# img_url=img_pct.get_attribute('src')
# img_url='https://etax.shanghai.chinatax.gov.cn/login-web/api/googleCaptcha?1598428694905'
# headers={'Accept': 'image/webp,*/*',"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"}
# req=requests.post(img_url,headers=headers)
# print(req.text)
# with open('img/2.jpg','wb') as f:
#     f.write(req.content)

from http import cookiejar
import urllib.request
import re
import sys
'''模拟登录'''
# 防止中文报错
CaptchaUrl = "https://etax.shanghai.chinatax.gov.cn/login-web/api/googleCaptcha?1598428694905"
PostUrl = "http://202.115.80.153/default2.aspx"
# 验证码地址和post地址
cookie = cookiejar.CookieJar()
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
# 将cookies绑定到一个opener cookie由cookielib自动管理
username = 'username'
password = 'password123'
# 用户名和密码
picture = opener.open(CaptchaUrl).read()
# 用openr访问验证码地址,获取cookie
local = open('img/3.jpg', 'wb')
local.write(picture)
local.close()