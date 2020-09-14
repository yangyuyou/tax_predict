import json
import web
from img_from import read_p
from shanghai import ocr_from_bsting
import base64

def read_shanghai(): #imagestring,path
    img_str=read_p('上海')
    img_b64 = base64.b64encode(img_str)
    image_str=img_b64.decode('utf-8')
    # iamge_str=read_label(paths)
    data={'image_str':image_str}
    # data.append(d_image)
    return data

# def read_label(paths):
#     try:
#         with open(paths,'rb') as f:
#             label=f.read()
#             img_b64=base64.b64encode(label)
#         return img_b64.decode('utf-8')
#     except:
#         return ''


class Shanghai_OCR:  #接口
    def POST(self):
        data=web.data()
        data=json.loads(data)
        a=data['image_str']
        if type(a) is not str:
            print('The data type you entered is incorrect. STR is expected')
            pass
        elif a.count('=')>3 or len(a)%4!=0:
            print('Incorrect format of B64')
            pass
        else:
            result=ocr_from_bsting(a) #识别验证码
            return result


class Img_shanghai:
    def GET(self):
        data=read_shanghai()
        if data != None:
            post={'data':data,'labelUrl':'shanghai','ocrUrl':'ocr1'}
            return render.shanghai(post)
        else:
            post={'labelUrl':'shanghai','ocrUrl':'ocr1'}
            return render.shanghai(post)


render=web.template.render("templates")

URLS=('/shanghai','Img_shanghai',
      '/ocr1','Shanghai_OCR',)

img_shanghai_jiekou=web.application(URLS,globals())

if __name__ == '__main__':
    img_shanghai_jiekou.run()
