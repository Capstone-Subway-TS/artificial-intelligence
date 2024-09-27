from flask import Flask, request, jsonify, render_template
from ts_api import tsApi

app = Flask(__name__)

@app.route('/getPred', methods=['POST', 'GET'])
def get_route():
    data = request.get_json()
    # data = {
    #     'route' : ['선릉', '삼성', '종합운동장', '잠실새내', '잠실', '잠실나루', '강변', '구의', '건대입구', '어린이대공원(세종대)'],
    # }
    ts = tsApi(data)
    result = ts.divide_route()
    return result

if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)
