#-*-coding:utf-8-*-
from flask import Flask,render_template,request,Response
# from flask_uploads import configure_uploads,UploadSet
import os,base64
from io import BytesIO
#from m_style_model.s_style import stylize, loading_model
from forward import forward  # 导入前向网络模块
import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块
from PIL import Image  # 导入PIL模块
import random

app = Flask(__name__)

def stylize(img,label_list,sess,target,content,weight):
    #默认第一张权重为1，即100%
    alpha_list=1
    # 将风格列表及对应的权重放入字典中
    input_weight = np.zeros([1, 20])
    weight_dict = dict([(label_list,alpha_list)])
    for k, v in weight_dict.items():
        input_weight[0, k] = v
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
        img=stylize(img,a,sess,target,content,weight)
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

@app.route("/",methods = ["GET","POST"])
def index():
    global sess,target,content,weight,a
    a=0
    sess,target,content,weight=loading_model()
    return  render_template('index.html')

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)