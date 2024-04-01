import requests
import xml.etree.ElementTree as ET
import json

def staion_arrive_info():
    xml_data = call_api()
    parsing_xml(xml_data)

def call_api():
    # API 키
    api_key = "47474654586769323634536e566661"

    # API 엔드포인트 URL
    api_url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/xml/realtimeStationArrival/0/5/"

    # 사용자 입력으로부터 역 이름 받기
    station_name = input("검색할 역 이름을 입력하세요: ")

    # API 호출을 위한 요청 URL 생성
    request_url = api_url + station_name

    try:
        # API 호출
        response = requests.get(request_url)

        # 응답 데이터 파싱
        if response.status_code == 200:
            # XML 형식의 응답을 파싱하여 필요한 정보 추출
            xml_data = response.text
            
            # 여기서부터는 XML 데이터를 파싱하여 원하는 정보를 추출하는 코드를 작성합니다.
            # 예를 들어, ElementTree를 사용하여 XML을 파싱하고 필요한 정보를 추출합니다.
            # XML 데이터 파싱 방법은 데이터의 구조와 요구사항에 따라 다를 수 있습니다.
            
            # 아래는 XML 데이터를 출력하는 예시입니다.
            return xml_data
        else:
            print("API 호출에 실패했습니다.")
    except Exception as e:
        print("오류가 발생했습니다:", e)

def parsing_xml(xml_data):
    # XML 파싱
    root = ET.fromstring(xml_data)

    # 파싱된 데이터를 저장할 리스트
    parsed_data = []

    # <row> 요소들을 순회하며 데이터 추출
    for row in root.findall('row'):
        data = {
            "rowNum": int(row.find('rowNum').text),
            "selectedCount": int(row.find('selectedCount').text),
            "totalCount": int(row.find('totalCount').text),
            "subwayId": int(row.find('subwayId').text),
            "updnLine": row.find('updnLine').text,
            "trainLineNm": row.find('trainLineNm').text,
            "statnFid": int(row.find('statnFid').text),
            "statnTid": int(row.find('statnTid').text),
            "statnId": int(row.find('statnId').text),
            "statnNm": row.find('statnNm').text,
            "trnsitCo": int(row.find('trnsitCo').text),
            "ordkey": row.find('ordkey').text,
            "subwayList": row.find('subwayList').text.split(','),
            "statnList": row.find('statnList').text.split(','),
            "btrainSttus": row.find('btrainSttus').text,
            "barvlDt": int(row.find('barvlDt').text),
            "btrainNo": row.find('btrainNo').text,
            "bstatnId": int(row.find('bstatnId').text),
            "bstatnNm": row.find('bstatnNm').text,
            "recptnDt": row.find('recptnDt').text,
            "arvlMsg2": row.find('arvlMsg2').text,
            "arvlMsg3": row.find('arvlMsg3').text,
            "arvlCd": int(row.find('arvlCd').text)
        }
        parsed_data.append(data)

    # 파싱된 데이터를 JSON 형식으로 변환하여 출력
    json_data = json.dumps(parsed_data, indent=4, ensure_ascii=False)
    print(json_data)


if __name__ == '__main__':
    staion_arrive_info()

# {
#         "rowNum": 1,
#         "selectedCount": 4,
#         "totalCount": 4,
#         "subwayId": 1007,
#         "updnLine": "상행",
#         "trainLineNm": "장암행 - 군자(능동)방면",
#         "statnFid": 1007000727,
#         "statnTid": 1007000725,
#         "statnId": 1007000726,
#         "statnNm": "어린이대공원(세종대)",
#         "trnsitCo": 1,
#         "ordkey": "01002장암0",
#         "subwayList": [
#             "1007"
#         ],
#         "statnList": [
#             "1007000726"
#         ],
#         "btrainSttus": "일반",
#         "barvlDt": 240,
#         "btrainNo": "7212",
#         "bstatnId": 1,
#         "bstatnNm": "장암",
#         "recptnDt": "2024-03-29 15:22:44",
#         "arvlMsg2": "4분 후 (뚝섬유원지)",
#         "arvlMsg3": "뚝섬유원지",
#         "arvlCd": 99
#     },