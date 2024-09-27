import requests
import xml.etree.ElementTree as ET
import json
import time
import collections
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import os
import sys
import schedule

class subwayApi:
    def __init__(self, api_key, date, start_time) -> None:
        # API 키
        self.api_key = api_key

        # API 엔드포인트 URL
        self.api_url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/xml/realtimeStationArrival/ALL"

        self.date = date
        self.start_time = start_time

        subway = ["apiCalltime", "rowNum", "selectedCount", "totalCount", "subwayId", "updnLine", "trainLineNm", "statnFid", "statnTid",
                    "statnId", "statnNm", "trnsitCo", "ordkey", "subwayList", "statnList", "btrainSttus", "barvlDt", "btrainNo",
                    "bstatnId", "bstatnNm", "recptnDt", "arvlMsg2", "arvlMsg3", "arvlCd"]
        self.data = {sub:[] for sub in subway}

    def staion_arrive_info(self):
        apiCalltime = datetime.now()
        xml_data = self.call_api()
        df = pd.DataFrame(self.parsing_xml(xml_data, apiCalltime.strftime('%H:%M:%S')))
        df.to_csv(f'/home/ubuntu/Capstone_TS/Model/web_scraping/data/api/{self.date}/train_arrival_info_{self.start_time}.csv', index=False, encoding='utf-8-sig')
        # df.to_csv(f'/Users/vvoo/Capstone_TS/Model/web_scraping/data/api/{self.date}/train_arrival_info_{self.start_time}.csv', index=False, encoding='utf-8-sig')
        print("[Api] Data fetched successfully. api 호출 시간 : {}".format(apiCalltime.strftime('%Y-%m-%d %H:%M:%S')))
        

    def call_api(self):
        try:
            # API 호출
            response = requests.get(self.api_url)

            # 응답 데이터 파싱
            if response.status_code == 200:
                # XML 형식의 응답을 파싱하여 필요한 정보 추출
                return response.text
            else:
                print("API 호출에 실패했습니다.")
        except Exception as e:
            print("오류가 발생했습니다:", e)

    def parsing_xml(self, xml_data, apiCalltime):
        # XML 파싱
        root = ET.fromstring(xml_data)
        
        for row in root.findall('row'):
            self.data["apiCalltime"].append(apiCalltime)
            self.data["rowNum"].append(int(row.find('rowNum').text))
            self.data["selectedCount"].append(int(row.find('selectedCount').text))
            self.data["totalCount"].append(int(row.find('totalCount').text))
            self.data["subwayId"].append(int(row.find('subwayId').text))             # 지하철 호선 ID
            self.data["updnLine"].append(row.find('updnLine').text)                  # 상하행선 구분 0:상행/내선, 1:하행/외선
            self.data["trainLineNm"].append(row.find('trainLineNm').text)            # 도착지 방면
            self.data["statnFid"].append(int(row.find('statnFid').text))             # 이전 지하철역 ID
            self.data["statnTid"].append(int(row.find('statnTid').text))             # 이전 지하철역 ID
            self.data["statnId"].append(int(row.find('statnId').text))               # 지하철역 ID
            self.data["statnNm"].append(row.find('statnNm').text)                    # 지하철역명
            self.data["trnsitCo"].append(int(row.find('trnsitCo').text))             # 환승노선수
            self.data["ordkey"].append(row.find('ordkey').text)                      # 도착예정열차순번
            self.data["subwayList"].append(row.find('subwayList').text.split(','))   # 연계호선ID
            self.data["statnList"].append(row.find('statnList').text.split(','))     # 연계지하철역ID
            self.data["btrainSttus"].append(row.find('btrainSttus').text)            # 열차종류
            self.data["barvlDt"].append(int(row.find('barvlDt').text))               # 열차도착예정시간
            self.data["btrainNo"].append(row.find('btrainNo').text)                  # 열차번호
            self.data["bstatnId"].append(int(row.find('bstatnId').text))             # 종착지하철역ID
            self.data["bstatnNm"].append(row.find('bstatnNm').text)                  # 종착지하철역명
            self.data["recptnDt"].append(row.find('recptnDt').text)                  # 열차도착정보를 생성한 시각
            self.data["arvlMsg2"].append(row.find('arvlMsg2').text)                  # 첫번째 도착메세지 - 도착, 출발, 진입 등
            self.data["arvlMsg3"].append(row.find('arvlMsg3').text)                  # 두번째 도착메세지 - (종합운동장 도착, 12분 후 (광명사거리) 등)
            self.data["arvlCd"].append(int(row.find('arvlCd').text))                 # 도착 코드 - (0:진입, 1:도착, 2:출발, 3:전역출발, 4:전역진입, 5:전역도착, 99:운행중)

        return self.data

class subwayScraper:
    def __init__(self, date, start_time) -> None:
        self.url = 'https://smss.seoulmetro.co.kr/traininfo/traininfoUserMap.do'
        self.date = date
        self.start_time = start_time
        # 각 라인의 도착 정보 추출
        self.train_info = {
                            'Time' : [],
                            'Line' : [],
                            'SubwayCode' : [],
                            'StationCode': [],
                            'StationName' : [],
                            'ArvlCd' : [],
                            'UpdnLine' : [],
                            # 'Decription' : []
                            }    

    def fetch_subway_data(self):
        now = datetime.now()
        response = requests.get(self.url)
        if response.status_code == 200:
            print("[Request] Data fetched successfully. request 호출 시간 : {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
            # 데이터 처리 로직을 여기에 추가하세요.
        else:
            print("Failed to fetch data:", response.status_code)
        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(response.content, 'html.parser')

        for div in soup.find_all('div', class_="tip"):
            line_name = div.parent['class'][0]  # 부모 div에서 클래스 이름 추출
            title = div.get('title')
            # print(title.split(' '))
            self.train_info['Time'].append(now.strftime('%Y-%m-%d %H:%M:%S'))
            self.train_info['Line'].append(line_name)
            self.train_info['SubwayCode'].append(title.split(' ')[0])
            self.train_info['StationCode'].append(div.get('data-statntcd'))
            self.train_info['StationName'].append(title.split(' ')[2])
            self.train_info['ArvlCd'].append(title.split(' ')[3])
            self.train_info['UpdnLine'].append(title.split(' ')[4])
            # self.train_info['Description'].append(title)

        # DataFrame 생성 및 CSV 파일로 저장
        df = pd.DataFrame(self.train_info)
        df.to_csv(f'/home/ubuntu/Capstone_TS/Model/web_scraping/data/request/{self.date}/train_arrival_info_{self.start_time}.csv', index=False, encoding='utf-8-sig')
        # df.to_csv(f'/Users/vvoo/Capstone_TS/Model/web_scraping/data/request/{self.date}/train_arrival_info_{self.start_time}.csv', index=False, encoding='utf-8-sig')

def job(text):
    date = datetime.now().strftime('%Y-%m-%d')
    print('{}:{}'.format(text, date))
    start_time = datetime.now().strftime('%H:%M:%S')
    try:
        os.mkdir(f'/home/ubuntu/Capstone_TS/Model/web_scraping/data/api/{date}/')
        os.mkdir(f'/home/ubuntu/Capstone_TS/Model/web_scraping/data/request/{date}/')
        # os.mkdir(f'/Users/vvoo/Capstone_TS/Model/web_scraping/data/request/{date}')
        # os.mkdir(f'/Users/vvoo/Capstone_TS/Model/web_scraping/data/api/{date}')
    except:
        pass
    scraper = subwayApi('517748504867693235366d695a6e77', date, start_time)
    subway = subwayScraper(date, start_time)
    schedule.every(80).seconds.do(scraper.staion_arrive_info)
    schedule.every(80).seconds.do(subway.fetch_subway_data)

# 매일 05:00에 첫 호출 시작
text = "API 시작"
job(text)

while True:
    # 현재 시간이 05:30 이후이고, 10:30 이전인지 확인
    current_time = datetime.now()
    end_time = current_time.replace(hour=10, minute=35, second=0, microsecond=0)
    
    if current_time >= current_time.replace(hour=5, minute=25, second=0, microsecond=0) and current_time <= end_time:
        # 현재 시간이 스케줄 시간 내인 경우에만 run_pending 실행
        schedule.run_pending()
        time.sleep(1)
    else:
        # 현재 시간이 스케줄 시간 외인 경우에는 호출 종료
        print("function exit")
        sys.exit()

