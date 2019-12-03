#-*-coding:utf-8-*-
from flask import Flask,render_template,request,Response
import os,base64
from io import BytesIO
from forward import forward  # 导入前向网络模块
import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块
from PIL import Image  # 导入PIL模块
import random
import json

app = Flask(__name__)

def sm_stylize(img,label_list,sess,target,content,weight,alpha_list=(1,1,1,1)):
    if isinstance(label_list,int):
        style_num=1
    else:
        style_num=len(label_list)
    alpha_list=alpha_list[0:style_num]
    if style_num==1:
        alpha_list=(1)
    else:
        print(alpha_list)
        alpha_list=alpha_list/np.sum(alpha_list)
        alpha_list=tuple(alpha_list)
    input_weight=np.zeros([1,20])
    if style_num==1:
        weight_dict = dict([(label_list,alpha_list)])
    else:
        weight_dict = dict(zip(label_list, alpha_list))
    for k,v in weight_dict.items():
        input_weight[0,k]=v
    # 进行风格融合与迁移
    img = sess.run(target,
                        feed_dict={content: img[np.newaxis, :, :, :], weight: input_weight})
    # 保存单张图片
    # 直接str(tuple)会产生空格，js无法读取
    img=np.uint8(img[0, :, :, :])
    return img

def loading_model():
    #预定义模型变量，分配空间
    tf.reset_default_graph()
    content = tf.placeholder(tf.float32, [1, None, None, 3])  # 图片输入定义
    weight = tf.placeholder(tf.float32, [1, 20])  # 风格权重向量，用于存储用户选择的风格
    target = forward(content, weight)  # 定义将要生成的图片
    #从恢复点中载入模型
    model_id="m_style_model/model/"
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
#    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # 变量初始化
    ckpt = tf.train.get_checkpoint_state(model_id)  # 从模型存储路径中获取模型
    saver = tf.train.Saver()  # 定义模型saver
    if ckpt and ckpt.model_checkpoint_path:  # 从检查点中恢复模型
        saver.restore(sess, ckpt.model_checkpoint_path)
    return sess,target,content,weight

a=0
sess,target,content,weight=loading_model()

def image_to_base64(image):    
    img_buffer = BytesIO()    
    image.save(img_buffer, format='JPEG')    
    byte_data = img_buffer.getvalue()    
    base64_str = base64.b64encode(byte_data)    
    return base64_str  

@app.route('/up_a', methods=['GET', 'POST'])
def up_a():
        global a
        b=request.values.get('a1')
        a=int(b)-1
        print(b)
        return ""

@app.route('/up_w', methods=['GET', 'POST'])
def up_w():
        global w
        w=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ids=request.form.get('ids')
        w_data = json.loads(ids)
        for i in range(20):
            w[i]=int(w_data[i])
        print(w)
        return ""

@app.route('/up_file', methods=['GET', 'POST'])#接受并存储文件
def up_file():
    global a,sess,target,content,weight
    if request.method == "POST":
    	#接收图片
        img=request.files['inputfile']
        img.save('static/files/img1/pic.png')
        #photo.save(request.files['inputfile'], 'img1', 'pic.png')#保存图片
		#发送图片
        img = Image.open("static/files/img1/pic.png")
        img=np.array(img)
        img = img[:,:,:3]
        img=sm_stylize(img,a,sess,target,content,weight)
        img=Image.fromarray(img)
        data=image_to_base64(img)
        #data = base64.b64encode(img.read()).decode()#进行base64编码
        # html = '''<img src="data:image/png;base64,{}" style="width:100%;height:100%;"/>'''#html代码
        # htmlstr = html.format(data)#添加数据
        return data

@app.route('/m_up_file', methods=['GET', 'POST'])#接受并存储文件
def m_up_file():
    global w,sess,target,content,weight
    if request.method == "POST":
    	#接收图片
        img=request.files['inputfile']
        img.save('static/files/img1/pic.png')
        #photo.save(request.files['inputfile'], 'img1', 'pic.png')#保存图片
		#发送图片
        img = Image.open("static/files/img1/pic.png")
        img=np.array(img)
        img = img[:,:,:3]
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        img=sm_stylize(img,label_list,sess,target,content,weight,tuple(w))
        img=Image.fromarray(img)
        data=image_to_base64(img)
        #data = base64.b64encode(img.read()).decode()#进行base64编码
        # html = '''<img src="data:image/png;base64,{}" style="width:100%;height:100%;"/>'''#html代码
        # htmlstr = html.format(data)#添加数据
        return data

@app.route('/single_image_transfer', methods=['GET', 'POST'])
def single_image_transfer():
    return render_template('image.html')
    
@app.route('/real_time_transfer', methods=['GET', 'POST'])#接受并存储文件
def real_time_transfer():
    return  render_template('cap.html')

@app.route('/muilt_image_transfer', methods=['GET', 'POST'])
def muilt_image_transfer():
    return render_template('m_image.html')

@app.route('/video_transfer', methods=['GET', 'POST'])#接受并存储文件
def video_transfer():
    return  render_template('video.html')

@app.route("/",methods = ["GET","POST"])
def index():
    global sess,target,content,weight,a
    return  render_template('index.html')

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)