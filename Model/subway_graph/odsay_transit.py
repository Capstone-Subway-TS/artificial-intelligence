"""
ODsay 대중교통 API 연동

ODsay API를 사용하여 지하철 시간표, 배차간격, 실시간 도착정보를 가져옵니다.

API 키 발급:
1. https://lab.odsay.com/ 회원가입
2. "API 키 발급" 메뉴에서 키 생성
3. 환경변수 설정: export ODSAY_API_KEY=your_key

주요 API:
- 지하철역 정보: /v1/api/subwayStationInfo
- 지하철 시간표: /v1/api/subwayTimeTable
- 지하철 경로검색: /v1/api/searchPubTransPathT
- 실시간 도착정보: /v1/api/subwayArrival
"""

import os
import json
import requests
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SubwayArrival:
    """실시간 도착 정보"""
    station: str
    line: str
    direction: str  # 상행/하행
    destination: str  # 행선지
    arrival_time: int  # 도착까지 남은 시간 (초)
    train_status: str  # 열차 상태


@dataclass
class SubwayTimetable:
    """지하철 시간표"""
    station: str
    line: int
    direction: str
    day_type: str  # 평일/토요일/휴일
    schedules: List[Dict]  # [{time: "05:30", destination: "성수"}, ...]


@dataclass
class TransitPath:
    """대중교통 경로"""
    total_time: int  # 총 소요시간 (분)
    transfer_count: int  # 환승 횟수
    walk_time: int  # 도보 시간 (분)
    fare: int  # 요금
    first_start_station: str  # 첫 승차역
    last_end_station: str  # 최종 하차역
    segments: List[Dict] = field(default_factory=list)  # 구간별 정보


class ODsayAPI:
    """ODsay 대중교통 API 클라이언트"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ODSAY_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ODSAY_API_KEY가 필요합니다.\n"
                "발급: https://lab.odsay.com/"
            )

        self.base_url = "https://api.odsay.com/v1/api"

        # 캐시
        self._station_cache = {}
        self._timetable_cache = {}

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """API 요청"""
        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}
        params['apiKey'] = self.api_key

        # localhost 도메인 검증용 헤더
        headers = {
            'Origin': 'http://localhost',
            'Referer': 'http://localhost/'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()

            # 에러 체크
            if 'error' in data:
                error = data['error']
                print(f"API 오류: {error.get('message', error)}")
                return {}

            return data.get('result', data)

        except requests.exceptions.Timeout:
            print(f"API 타임아웃: {endpoint}")
            return {}
        except Exception as e:
            print(f"API 요청 실패: {e}")
            return {}

    # ========== 지하철역 정보 ==========

    def get_subway_station_info(self, station_name: str, line: int = None) -> List[Dict]:
        """
        지하철역 정보 조회

        Args:
            station_name: 역명 (예: "강남")
            line: 호선 번호 (옵션)

        Returns:
            역 정보 목록
        """
        cache_key = f"{station_name}_{line}"
        if cache_key in self._station_cache:
            return self._station_cache[cache_key]

        # searchStation API 사용 (stationClass=2: 지하철)
        params = {"stationName": station_name, "stationClass": 2}

        data = self._request("searchStation", params)

        if not data:
            return []

        stations = data.get('station', [])
        if not isinstance(stations, list):
            stations = [stations] if stations else []

        # 정확히 일치하는 역만 필터
        stations = [s for s in stations if s.get('stationName') == station_name]

        # 호선 필터
        if line:
            line_name = f"{line}호선"
            stations = [s for s in stations if line_name in s.get('laneName', '')]

        self._station_cache[cache_key] = stations
        return stations

    def get_station_id(self, station_name: str, line: int) -> Optional[int]:
        """역 ID 조회"""
        stations = self.get_subway_station_info(station_name, line)
        if stations:
            return int(stations[0].get('stationID'))
        return None

    # ========== 지하철 시간표 ==========

    def get_subway_timetable(
        self,
        station_name: str,
        line: int,
        direction: int = 1,  # 1: 상행, 2: 하행
        day_type: int = 1    # 1: 평일, 2: 토요일, 3: 휴일
    ) -> SubwayTimetable:
        """
        지하철 시간표 조회

        Args:
            station_name: 역명
            line: 호선 (1-9)
            direction: 1=상행/내선, 2=하행/외선
            day_type: 1=평일, 2=토요일, 3=휴일

        Returns:
            SubwayTimetable
        """
        cache_key = f"{station_name}_{line}_{direction}_{day_type}"
        if cache_key in self._timetable_cache:
            return self._timetable_cache[cache_key]

        # 역 ID 조회
        station_id = self.get_station_id(station_name, line)
        if not station_id:
            print(f"역을 찾을 수 없습니다: {station_name} {line}호선")
            return SubwayTimetable(station_name, line, "", "", [])

        params = {
            "stationID": station_id,
            "wayCode": direction
        }

        data = self._request("subwayTimeTable", params)

        if not data:
            return SubwayTimetable(station_name, line, "", "", [])

        # 요일별 데이터 선택
        day_key_map = {1: 'OrdList', 2: 'SatList', 3: 'SunList'}
        day_key = day_key_map.get(day_type, 'OrdList')

        # 방향 선택
        dir_key = 'up' if direction == 1 else 'down'

        # 시간표 파싱
        schedules = []
        day_data = data.get(day_key, {})
        dir_data = day_data.get(dir_key, {})
        time_list = dir_data.get('time', [])

        if not isinstance(time_list, list):
            time_list = [time_list] if time_list else []

        import re
        for item in time_list:
            hour = item.get('Idx', 0)
            time_str = item.get('list', '')

            # "MM(행선지) MM(행선지)" 형식 파싱
            pattern = r'(\d+)\(([^)]+)\)'
            matches = re.findall(pattern, time_str)

            for minute, destination in matches:
                schedules.append({
                    'time': f"{hour:02d}:{minute}",
                    'hour': hour,
                    'minute': int(minute),
                    'destination': destination
                })

        day_names = {1: '평일', 2: '토요일', 3: '휴일'}
        dir_names = {1: '상행', 2: '하행'}

        result = SubwayTimetable(
            station=station_name,
            line=line,
            direction=dir_names.get(direction, ''),
            day_type=day_names.get(day_type, ''),
            schedules=schedules
        )

        self._timetable_cache[cache_key] = result
        return result

    def get_headway_from_timetable(
        self,
        station_name: str,
        line: int,
        hour: int,
        direction: int = 1,
        day_type: int = 1
    ) -> int:
        """
        시간표에서 배차간격 계산

        Args:
            station_name: 역명
            line: 호선
            hour: 시간 (0-23)
            direction: 방향
            day_type: 요일 유형

        Returns:
            배차간격 (초)
        """
        timetable = self.get_subway_timetable(station_name, line, direction, day_type)

        if not timetable.schedules:
            return 300  # 기본값 5분

        # 해당 시간대 열차 필터
        trains_in_hour = [
            s for s in timetable.schedules
            if s.get('hour') == hour
        ]

        if len(trains_in_hour) < 2:
            return 300  # 기본값

        # 분 단위로 정렬
        minutes = sorted([s.get('minute', 0) for s in trains_in_hour])

        # 평균 간격 계산
        intervals = [minutes[i+1] - minutes[i] for i in range(len(minutes)-1)]

        if not intervals:
            return 300

        avg_interval = sum(intervals) / len(intervals)

        return int(avg_interval * 60)  # 분 → 초

    # ========== 실시간 도착정보 ==========

    def get_realtime_arrival(self, station_name: str, line: int = None) -> List[SubwayArrival]:
        """
        실시간 지하철 도착정보

        Args:
            station_name: 역명
            line: 호선 (옵션)

        Returns:
            도착 정보 목록
        """
        # 역 ID 필요
        stations = self.get_subway_station_info(station_name, line)
        if not stations:
            return []

        station_id = stations[0].get('stationID')

        params = {"stationID": station_id}
        data = self._request("subwayArrival", params)

        if not data:
            return []

        arrivals = []
        arrival_list = data.get('arrival', [])
        if not isinstance(arrival_list, list):
            arrival_list = [arrival_list] if arrival_list else []

        for arr in arrival_list:
            arrivals.append(SubwayArrival(
                station=station_name,
                line=arr.get('lineId', ''),
                direction=arr.get('direction', ''),
                destination=arr.get('heading', ''),
                arrival_time=int(arr.get('arrivalSec', 0)),
                train_status=arr.get('trainStatus', '')
            ))

        return arrivals

    # ========== 경로 검색 ==========

    def search_transit_path(
        self,
        start_station: str,
        end_station: str,
        start_line: int = None,
        end_line: int = None,
        search_type: int = 0  # 0: 최단시간
    ) -> List[TransitPath]:
        """
        대중교통 경로 검색

        Args:
            start_station: 출발역
            end_station: 도착역
            start_line: 출발 호선 (옵션)
            end_line: 도착 호선 (옵션)
            search_type: 0=최단시간, 1=최소환승

        Returns:
            경로 목록
        """
        # 역 좌표 조회
        start_info = self.get_subway_station_info(start_station, start_line)
        end_info = self.get_subway_station_info(end_station, end_line)

        if not start_info or not end_info:
            print(f"역을 찾을 수 없습니다: {start_station} 또는 {end_station}")
            return []

        sx, sy = start_info[0].get('x'), start_info[0].get('y')
        ex, ey = end_info[0].get('x'), end_info[0].get('y')

        params = {
            "SX": sx, "SY": sy,
            "EX": ex, "EY": ey,
            "SearchType": search_type,
            "SearchPathType": 1  # 지하철만
        }

        data = self._request("searchPubTransPathT", params)

        if not data:
            return []

        paths = []
        path_list = data.get('path', [])
        if not isinstance(path_list, list):
            path_list = [path_list] if path_list else []

        for path in path_list:
            info = path.get('info', {})

            # 구간 정보 파싱
            segments = []
            sub_path_list = path.get('subPath', [])
            if not isinstance(sub_path_list, list):
                sub_path_list = [sub_path_list] if sub_path_list else []

            for sub in sub_path_list:
                traffic_type = sub.get('trafficType')  # 1=지하철, 2=버스, 3=도보

                if traffic_type == 1:  # 지하철
                    segments.append({
                        'type': 'subway',
                        'line': sub.get('lane', [{}])[0].get('name', ''),
                        'start_station': sub.get('startName', ''),
                        'end_station': sub.get('endName', ''),
                        'station_count': sub.get('stationCount', 0),
                        'time': sub.get('sectionTime', 0)
                    })
                elif traffic_type == 3:  # 도보
                    segments.append({
                        'type': 'walk',
                        'time': sub.get('sectionTime', 0),
                        'distance': sub.get('distance', 0)
                    })

            paths.append(TransitPath(
                total_time=info.get('totalTime', 0),
                transfer_count=info.get('subwayTransitCount', 0),
                walk_time=info.get('totalWalk', 0),
                fare=info.get('payment', 0),
                first_start_station=info.get('firstStartStation', ''),
                last_end_station=info.get('lastEndStation', ''),
                segments=segments
            ))

        return paths


def collect_all_headways(api: ODsayAPI, output_path: str = None):
    """
    모든 환승역의 시간대별 배차간격 수집

    Returns:
        {역명: {호선: {시간대: 배차간격(초)}}}
    """
    base_dir = os.path.dirname(__file__)

    # 환승역 목록 로드
    transfer_path = os.path.join(base_dir, 'data', 'transfer_stations.json')
    with open(transfer_path, 'r', encoding='utf-8') as f:
        transfer_stations = json.load(f)

    # 역-호선 매핑 로드
    station_lines_path = os.path.join(base_dir, 'data', 'station_lines.json')
    with open(station_lines_path, 'r', encoding='utf-8') as f:
        station_lines = json.load(f)

    print(f"환승역 {len(transfer_stations)}개의 배차간격 수집 중...")

    headways = {}
    time_periods = {
        'rush_morning': [7, 8, 9],
        'rush_evening': [18, 19, 20],
        'normal': [10, 11, 12, 13, 14, 15, 16, 17],
        'late_night': [21, 22, 23, 0]
    }

    for i, station in enumerate(transfer_stations[:10]):  # 테스트용 10개
        lines = station_lines.get(station, [])
        headways[station] = {}

        print(f"\n[{i+1}/{len(transfer_stations)}] {station}역 ({lines})")

        for line in lines:
            headways[station][line] = {}

            for period_name, hours in time_periods.items():
                intervals = []

                for hour in hours[:1]:  # 대표 시간만
                    try:
                        interval = api.get_headway_from_timetable(
                            station, line, hour, direction=1, day_type=1
                        )
                        intervals.append(interval)
                    except Exception as e:
                        print(f"    {line}호선 {hour}시: 오류 - {e}")

                    time.sleep(0.2)  # API 제한

                if intervals:
                    avg = sum(intervals) // len(intervals)
                    headways[station][line][period_name] = avg
                    print(f"    {line}호선 {period_name}: {avg}초")

    # 저장
    if output_path is None:
        output_path = os.path.join(base_dir, 'data', 'odsay_headways.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(headways, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path}")
    return headways


def demo():
    """데모 실행"""
    api_key = os.environ.get('ODSAY_API_KEY')

    if not api_key:
        print("="*60)
        print("ODsay API 키가 필요합니다")
        print("="*60)
        print("""
1. https://lab.odsay.com/ 회원가입
2. 로그인 후 "API 키 발급" 클릭
3. 서비스 용도 선택 후 키 생성
4. 환경변수 설정:
   export ODSAY_API_KEY=your_api_key
5. 다시 실행:
   python odsay_transit.py
        """)
        return None

    print("="*60)
    print("ODsay API 테스트")
    print("="*60)

    api = ODsayAPI(api_key)

    # 1. 역 정보 조회
    print("\n[1] 역 정보 조회")
    stations = api.get_subway_station_info("강남", line=2)
    if stations:
        print(f"  강남역 2호선: ID={stations[0].get('stationID')}")

    # 2. 시간표 조회
    print("\n[2] 시간표 조회 (강남역 2호선 평일 상행)")
    timetable = api.get_subway_timetable("강남", 2, direction=1, day_type=1)
    if timetable.schedules:
        print(f"  총 {len(timetable.schedules)}개 열차")
        print(f"  첫차: {timetable.schedules[0]}")
        print(f"  막차: {timetable.schedules[-1]}")

    # 3. 배차간격 계산
    print("\n[3] 시간대별 배차간격")
    for hour in [8, 12, 18, 22]:
        headway = api.get_headway_from_timetable("강남", 2, hour)
        print(f"  {hour}시: {headway}초 ({headway//60}분)")

    # 4. 경로 검색
    print("\n[4] 경로 검색 (강남 → 홍대입구)")
    paths = api.search_transit_path("강남", "홍대입구")
    if paths:
        path = paths[0]
        print(f"  소요시간: {path.total_time}분")
        print(f"  환승: {path.transfer_count}회")
        print(f"  요금: {path.fare}원")

    # 5. 실시간 도착정보
    print("\n[5] 실시간 도착정보 (강남역)")
    arrivals = api.get_realtime_arrival("강남", 2)
    if arrivals:
        for arr in arrivals[:3]:
            print(f"  {arr.destination}행: {arr.arrival_time}초 후 도착")
    else:
        print("  (실시간 정보 없음 또는 운행 종료)")

    return api


if __name__ == '__main__':
    api = demo()
