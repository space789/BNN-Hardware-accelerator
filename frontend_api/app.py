import requests

from flask import *
from flask_cors import *
import platform

app = Flask(__name__, static_folder='ExportedData/assets/', template_folder='ExportedData/')


@app.route('/')
def hello_world():  # put application's code here
    return redirect('/index')


@app.route('/index')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/demo')
@app.route('/demo.html')
def demo():
    return render_template('demo.html')


@app.route('/result.html')
def result():
    return render_template('result.html')


@app.route('/intro')
@app.route('/intro.html')
def intro():
    return render_template('intro.html')


@app.route('/apis/submit', methods=['POST'])
@cross_origin()
def submit():
    form_data = request.form['gray_img']
    data = form_data.split('\r\n')
    ret = []
    for _ in range(len(data)):
        ret.append(list(map(int, data[_].split(' '))))

    # print(len(form_data))
    # if len(form_data) != 958:
    #     return "長度不符合標準", 400

    print(ret)
    if platform.platform() == 'Windows-10-10.0.22621-SP0':
        req = requests.post(url='http://127.0.0.1:8000/', json={'data': ret})
    else:
        req = requests.post(url='http://rasbpi.yinchian.com:8000', json={'data': ret})

    if req.status_code == 200:
        res = json.loads(req.text)
        print('result =', res['result'])

        # return render_template('result.html', result=res['result'],
        #                        # prob0=res['probability_arr'][0], prob1=res['probability_arr'][1],
        #                        # prob2=res['probability_arr'][2], prob3=res['probability_arr'][3],
        #                        # prob4=res['probability_arr'][4], prob5=res['probability_arr'][5],
        #                        # prob6=res['probability_arr'][6], prob7=res['probability_arr'][7],
        #                        # prob8=res['probability_arr'][8], prob9=res['probability_arr'][9],
        #                        prob=res['probability_arr'],
        #                        total_time=res['total_time'], data_prep_time=res['data_prep_time'],
        #                        conv_time=res['conv_time'], pool_time=res['pool_time']
        #                        ), 200

        return res, 200
    else:
        return 'PYNQ連線失敗', 500


if __name__ == '__main__':
    app.run()
