import pandas as pd
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import random
import time
class tsApi():
    def __init__(self, data) -> None:
        self.route = data['route']
        self.start_stat = data['route'][0]
        self.now = datetime.now()
        
    def check_updn(self):
        stat, prev = self.route[1], self.route[0]
        # df = pd.read_csv('/Users/vvoo/Capstone_TS/Model/data_processing/data/상하행.csv', encoding='cp949')
        df = pd.read_csv('상하행.csv', encoding='cp949')

        self.updn = df[(df['역명']==stat)&(df['이전역']==prev)]['상하행'].values[0]
        self.start_line = df[(df['역명']==stat)&(df['이전역']==prev)]['호선'].values[0]

    def call_subway_time(self):
        # 경로 시작역, 호선, 상하행 초기화
        self.check_updn()

        # 브라우저 꺼짐 방지
        chrome_options = Options()
        chrome_options.add_experimental_option("detach", True)

        # 불필요 메세지 없애기
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

        # 드라이버 생성
        driver = webdriver.Chrome(options=chrome_options)

        # 네이버 지도 지하철 페이지로 이동
        url = f'https://m.search.naver.com/search.naver?where=m&mra=QkRS^TVBBX1NCVw==&query={self.start_stat}%20{self.start_line}호선'

        # 페이지 요청
        response = driver.get(url)
        time.sleep(random.uniform(7,10))

        # 페이지 소스 가져오기
        page_source = driver.page_source

        # BeautifulSoup을 사용하여 HTML 파싱
        soup = bs(page_source, 'html.parser')
        
        # 실시간 열차 도착 정보를 포함하는 요소 찾기
        arrival_info, updn = {}, 0
        div_chuncks = soup.select('.subway_content_wrap.cs_mo_subway > div > div > div:nth-child(5) > div:nth-child(2) > ul > li')
        
        # 상하행 열차별로 가장 빠른 도착시간 저장
        for chunck in div_chuncks:
            if updn == 0:
                arrival_info['상행'] = chunck.select('em')[0].text
                arrival_info['내선'] = chunck.select('em')[0].text
            else:
                arrival_info['하행'] = chunck.select('em')[0].text
                arrival_info['외선'] = chunck.select('em')[0].text
            updn += 1

        # 드라이버 종료
        driver.quit()

        # 현재 시간을 시간 형식에 맞춰 datetime 객체로 변환
        time_obj = datetime.strptime(self.now.strftime('%H:%M:%S'), '%H:%M:%S')
        # 열차 도착까지 남은 시간을 더해 출발 시간 반환
        print('남은시간 : {}'.format(arrival_info[self.updn]))
        if '도착' in arrival_info[self.updn]:
            time_remain = random.randint(1,10)
            new_time_obj = time_obj + timedelta(seconds=time_remain)
            return time_obj.strftime('%H:%M:%S')
        else:
            time_remain = int(arrival_info[self.updn].replace('분', ''))
            new_time_obj = time_obj + timedelta(minutes=time_remain)
            return new_time_obj.strftime('%H:%M:%S')
                
    def divide_route(self):
        
        # subway_start = self.call_subway_time()
        subway_start = datetime.strptime(self.now.strftime('%H:%M:%S'), '%H:%M:%S')
        
        # Step 1: 각 역 사이의 평균 소요 시간 데이터 준비 (분 단위로 가정)
        data = {
            '역명': [stat for stat in self.route[1:]],
            # ['오류동', '개봉', '구일', '구로', '신도림', '영등포', '신길', '대방', '노량진', '용산', '남영', '서울역', '시청', '종각', '종로3'],
            '이전역': self.route[:-1],
            # ['성공회대입구', '오류동', '개봉', '구일', '구로', '신도림', '영등포', '신길', '대방', '노량진', '용산', '남영', '서울역', '시청', '종각']
            '소요시간': []
        }
        
        # 예측 시간 불러오기
        for stat, prev in zip(data['역명'], data['이전역']):
            # tmp = pd.read_csv(f'/Users/vvoo/Capstone_TS/data/subway/pred/{stat}_{prev}.csv', index_col=0)
            tmp = pd.read_csv(f'/subway/pred/{stat}_{prev}.csv', index_col=0)
            # 경로 호출 시간
            original_time = self.now.strftime('%H:00:00')
            print(stat, prev, original_time)

            # 분과 초를 0으로 설정
            # modified_time = original_time.replace(minute=0, second=0)

            # 결과 출력
            # modified_time_str = modified_time.strftime('%Y-%m-%d %H:%M:%S')

            # 해당 시간대의 소요시간 출력
            data['소요시간'].append(tmp.loc[tmp.index.str.contains(f'{str(original_time)}')]['yhat'].values[0]+30)
        
        return [subway_start, data['소요시간']]

if __name__ == '__main__':
    data = {
        'route' : ['선릉', '삼성', '종합운동장', '잠실새내', '잠실', '잠실나루', '강변', '구의', '건대입구', '어린이대공원(세종대)'],
    }
    ts = tsApi(data)

    # ts.call_subway_time()
    print(ts.divide_route())
    # start_time = datetime.strptime(data['now'], '%Y-%m-%d %H:%M:%S')
    
    # for station, time in arrival_times:
    #     print(f"{station} : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            

