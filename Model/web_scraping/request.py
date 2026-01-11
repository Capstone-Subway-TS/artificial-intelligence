import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import os
import schedule
import sys

class subwayScraper:
    def __init__(self, date) -> None:
        self.url = 'https://smss.seoulmetro.co.kr/traininfo/traininfoUserMap.do'
        self.date = date
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
            print("Data fetched successfully. request 호출 시간 : {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
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
        df.to_csv(f'/Users/vvoo/Capstone_TS/Model/web_scraping/data/request/{self.date}/train_arrival_info.csv', index=False, encoding='utf-8-sig')

def job(text):
    date = datetime.now().strftime('%Y-%m-%d')
    start_time = datetime.now().strftime('%H:%M:%S')    
    print("{}:{}".format(text, start_time))
    try:
        os.mkdir(f'/Users/vvoo/Capstone_TS/Model/web_scraping/data/request/{date}')
    except:
        pass   
    subway = subwayScraper(date)
    schedule.every(1).seconds.do(subway.fetch_subway_data)

# 매일 05:30에 첫 호출 시작
text = "API 시작"
job(text)

while True:
    # 현재 시간이 05:30 이후이고, 25:00 (익일 01:00) 이전인지 확인
    current_time = datetime.now()
    end_time = (datetime.now() + timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
    
    if current_time >= current_time.replace(hour=0, minute=30, second=0, microsecond=0) and current_time <= end_time:
        # 현재 시간이 스케줄 시간 내인 경우에만 run_pending 실행
        schedule.run_pending()
        time.sleep(1)
    else:
        # 현재 시간이 스케줄 시간 외인 경우에는 호출 종료
        print("function exit")
        sys.exit()
