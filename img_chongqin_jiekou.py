import base64
import json
import web
from chongqin import read_img_chongqin
from img_from import read_p


def read_chongqin(): #imagestring,path
    img_str = read_p('重庆')
    img_b64 = base64.b64encode(img_str)
    image_str = img_b64.decode('utf-8')#image_str是一个b64编码
    data={'image_str':image_str}
    # data.append(d_image)
    return data

# def read_label(paths):   #以二进制的方式读取图片，再进行b64编码
#     try:
#         with open(paths,'rb') as f:
#             label=f.read()
#             img_b64=base64.b64encode(label)
#         return img_b64.decode('utf-8')
#     except:
#         return ''

class Chongqin_OCR:  #接口
    def POST(self):
        data=web.data()
        data=json.loads(data)
        b=data['image_str']
        if type(b) is not str:
            print('The data type you entered is incorrect. STR is expected')
            pass
        elif b.count('=')>3 or len(b)%4!=0:
            print('Incorrect format of B64')
            pass
        else:
            result=read_img_chongqin(b)#识别验证码
            return result

class Img_chongqin:
    def GET(self):
        data = read_chongqin()
        if data != None:
            post = {'data': data, 'labelUrl': 'chongqin', 'ocrUrl': 'ocr2'}
            return render.chongqin(post)
        else:
            post = {'labelUrl': 'chongqin', 'ocrUrl': 'ocr2'}
            return render.chongqin(post)
    # def POST(self):
    #     data=web.data()
    #     data=json.loads(data)
render=web.template.render("templates")

URLS=('/chongqin','Img_chongqin',
      '/ocr2','Chongqin_OCR')

img_chongqin_jiekou=web.application(URLS,globals())

if __name__=='__main__':
    img_chongqin_jiekou.run()

