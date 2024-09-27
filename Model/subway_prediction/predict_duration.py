from flask import Flask, request, jsonify


# Flask 애플리케이션 생성
app = Flask(__name__)

# POST 요청을 처리하는 핸들러
@app.route('/receive_subway_arrive', methods=['POST', 'GET'])
def receive_subway_arrive():
    #Spring으로부터 JSON 객체를 전달받음
    data = request.get_json()

    # JSON 데이터에서 필요한 정보 추출
    # input_data = data.get('station_name')
    input_data = '성수'

    # TODO: AI 모델 호출 및 추론
    # 이 부분에서 AI 모델을 호출하여 입력 데이터에 대한 추론을 수행합니다.
    # 여기서는 예시로 임의의 결과를 생성합니다.
    # prediction_result = seoulAPI(input_data).staion_arrive_info()

    # JSON 형식으로 추론 결과 반환
    # return prediction_result

# Flask 애플리케이션 실행 - 0.0.0.0 으로 모든 IP에 대한 연결을 허용해놓고 포트는 8082로 설정
if __name__ == '__main__':
    app.run('0.0.0.0', port=8082, debug=True)