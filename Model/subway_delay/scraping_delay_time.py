import time
from cachetools import cached, TTLCache
import requests
import schedule
import pandas as pd

# API 엔드포인트와 매개변수 설정
api_key = "47474654586769323634536e566661"
API_ENDPOINT = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/xml/realtimeStationArrival/0/5/"
STATIONS = ["건대입구", "성수", "사가정"]  # 수집할 역 목록

# 캐시 설정 (TTLCache: Time-To-Live 캐시, 유효 기간 설정)
cache = TTLCache(maxsize=100, ttl=60)  # 최대 100개의 항목을 저장하며, 각 항목은 60초 동안 유효함

# 역별 도착 정보를 API로부터 가져오는 함수
@cached(cache)
def get_subway_data(station):
    response = requests.get(f"{API_ENDPOINT}{station}")
    data = response.json()
    return data

def fetch_subway_data():
    all_data = []
    for station in STATIONS:
        subway_data = get_subway_data(station)
        all_data.extend(subway_data)
    save_to_csv(all_data)

def save_to_csv(data):
    df = pd.DataFrame(data)
    df.to_csv("subway_arrival_data.csv", index=False, encoding="utf-8-sig")

def main():
    # 매 분마다 fetch_subway_data 함수 실행
    schedule.every(1).minutes.do(fetch_subway_data)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
