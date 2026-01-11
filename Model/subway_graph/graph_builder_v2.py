"""
서울 지하철 그래프 구축 V2

개선사항:
- 환승역을 호선별 별도 노드로 분리
- 환승 시 도보 시간을 엣지 가중치로 반영
- 예: "교대_2호선" ↔ "교대_3호선" (가중치: 180초)
"""

import pandas as pd
import json
import heapq
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import os


class SubwayGraphV2:
    """환승 시간이 반영된 지하철 그래프"""

    def __init__(self):
        # 그래프: {노드: {인접노드: 소요시간(초)}}
        # 노드 형식: "역명_호선" (예: "교대_2호선", "교대_3호선")
        self.graph: Dict[str, Dict[str, int]] = {}

        # 역 정보
        self.station_lines: Dict[str, Set[int]] = {}  # {역명: {호선들}}
        self.line_stations: Dict[int, List[str]] = {}  # {호선: [역 순서]}
        self.transfer_stations: Set[str] = set()  # 환승역 목록

        # 환승 시간 설정 (초)
        self.default_transfer_time = 180  # 3분
        self.transfer_times = {
            '종로3가': 240,      # 4분 (1-3-5호선)
            '동대문역사문화공원': 300,  # 5분 (2-4-5호선)
            '왕십리': 240,       # 4분
            '고속터미널': 240,   # 4분
            '신도림': 180,       # 3분
            '사당': 180,
            '교대': 180,
            '잠실': 180,
            '시청': 180,
            '을지로3가': 180,
            '충무로': 180,
        }

    def parse_time_to_seconds(self, time_str: str) -> int:
        """시간 문자열을 초로 변환"""
        if pd.isna(time_str) or time_str == '0:00':
            return 0
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0

    def make_node_id(self, station: str, line: int) -> str:
        """노드 ID 생성: 역명_호선"""
        return f"{station}_{line}호선"

    def parse_node_id(self, node_id: str) -> Tuple[str, int]:
        """노드 ID 파싱: (역명, 호선)"""
        parts = node_id.rsplit('_', 1)
        station = parts[0]
        line = int(parts[1].replace('호선', ''))
        return station, line

    def build_graph(self, data_path: str):
        """그래프 구축"""
        print("=" * 60)
        print("지하철 그래프 구축 V2 (환승 시간 반영)")
        print("=" * 60)

        # 데이터 로드
        df = pd.read_csv(data_path, encoding='cp949')
        print(f"\n데이터 로드: {len(df)}개 레코드")

        # 1단계: 호선별 역 연결
        print("\n[1단계] 호선별 역 연결...")
        self._build_line_connections(df)

        # 2단계: 환승역 식별
        print("\n[2단계] 환승역 식별...")
        self._identify_transfers()

        # 3단계: 환승 연결 추가
        print("\n[3단계] 환승 연결 추가...")
        self._add_transfer_connections()

        # 통계 출력
        self._print_stats()

    def _build_line_connections(self, df: pd.DataFrame):
        """호선별 역 연결"""
        for line in sorted(df['호선'].unique()):
            line_df = df[df['호선'] == line].reset_index(drop=True)
            stations = line_df['역명'].tolist()
            times = line_df['시간(분)'].tolist()

            self.line_stations[int(line)] = stations

            for i, station in enumerate(stations):
                # 역-호선 매핑
                if station not in self.station_lines:
                    self.station_lines[station] = set()
                self.station_lines[station].add(int(line))

                # 노드 ID
                node_id = self.make_node_id(station, line)

                if node_id not in self.graph:
                    self.graph[node_id] = {}

                # 다음 역과 연결
                if i < len(stations) - 1:
                    next_station = stations[i + 1]
                    next_node_id = self.make_node_id(next_station, line)
                    travel_time = self.parse_time_to_seconds(times[i + 1])

                    if travel_time > 0:
                        # 양방향 연결
                        self.graph[node_id][next_node_id] = travel_time

                        if next_node_id not in self.graph:
                            self.graph[next_node_id] = {}
                        self.graph[next_node_id][node_id] = travel_time

            print(f"  {line}호선: {len(stations)}개 역")

    def _identify_transfers(self):
        """환승역 식별"""
        for station, lines in self.station_lines.items():
            if len(lines) >= 2:
                self.transfer_stations.add(station)

        print(f"  환승역 수: {len(self.transfer_stations)}개")

    def _add_transfer_connections(self):
        """환승역에 호선 간 연결 추가"""
        transfer_count = 0

        for station in self.transfer_stations:
            lines = sorted(self.station_lines[station])
            transfer_time = self.transfer_times.get(station, self.default_transfer_time)

            # 모든 호선 쌍에 대해 환승 연결
            for i, line1 in enumerate(lines):
                for line2 in lines[i + 1:]:
                    node1 = self.make_node_id(station, line1)
                    node2 = self.make_node_id(station, line2)

                    # 양방향 환승 연결
                    if node1 in self.graph and node2 in self.graph:
                        self.graph[node1][node2] = transfer_time
                        self.graph[node2][node1] = transfer_time
                        transfer_count += 1

        print(f"  환승 연결 추가: {transfer_count}개")

    def _print_stats(self):
        """통계 출력"""
        print("\n" + "=" * 60)
        print("그래프 통계")
        print("=" * 60)

        total_nodes = len(self.graph)
        total_edges = sum(len(neighbors) for neighbors in self.graph.values())

        print(f"총 노드 수: {total_nodes}개")
        print(f"총 엣지 수: {total_edges}개")
        print(f"환승역 수: {len(self.transfer_stations)}개")
        print(f"기본 환승 시간: {self.default_transfer_time}초 ({self.default_transfer_time // 60}분)")

    def dijkstra(self, start_station: str, end_station: str,
                 start_line: Optional[int] = None,
                 end_line: Optional[int] = None) -> Tuple[List[str], int, List[dict]]:
        """
        최단 경로 탐색

        Args:
            start_station: 출발역 이름
            end_station: 도착역 이름
            start_line: 출발 호선 (없으면 모든 호선에서 탐색)
            end_line: 도착 호선 (없으면 모든 호선에서 탐색)

        Returns:
            (경로 노드 리스트, 총 소요시간, 상세 경로 정보)
        """
        # 시작/도착 노드 결정
        if start_line:
            start_nodes = [self.make_node_id(start_station, start_line)]
        else:
            start_nodes = [self.make_node_id(start_station, line)
                          for line in self.station_lines.get(start_station, [])]

        if end_line:
            end_nodes = [self.make_node_id(end_station, end_line)]
        else:
            end_nodes = [self.make_node_id(end_station, line)
                        for line in self.station_lines.get(end_station, [])]

        if not start_nodes or not end_nodes:
            return [], float('inf'), []

        # 다익스트라
        distances = {node: float('inf') for node in self.graph}
        previous = {node: None for node in self.graph}

        # 모든 시작 노드에서 시작
        queue = []
        for start_node in start_nodes:
            if start_node in distances:
                distances[start_node] = 0
                heapq.heappush(queue, (0, start_node))

        while queue:
            current_dist, current = heapq.heappop(queue)

            if current_dist > distances[current]:
                continue

            for neighbor, weight in self.graph[current].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(queue, (distance, neighbor))

        # 최적 도착 노드 찾기
        best_end = None
        best_dist = float('inf')
        for end_node in end_nodes:
            if end_node in distances and distances[end_node] < best_dist:
                best_dist = distances[end_node]
                best_end = end_node

        if best_end is None or best_dist == float('inf'):
            return [], float('inf'), []

        # 경로 복원
        path = []
        current = best_end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        # 상세 경로 정보 생성
        route_details = self._generate_route_details(path)

        return path, best_dist, route_details

    def _generate_route_details(self, path: List[str]) -> List[dict]:
        """경로 상세 정보 생성"""
        details = []
        current_line = None

        for i, node in enumerate(path):
            station, line = self.parse_node_id(node)

            is_transfer = (current_line is not None and line != current_line)

            detail = {
                'station': station,
                'line': line,
                'is_transfer': is_transfer,
            }

            if i < len(path) - 1:
                next_node = path[i + 1]
                next_station, next_line = self.parse_node_id(next_node)
                travel_time = self.graph[node].get(next_node, 0)

                detail['next_station'] = next_station
                detail['travel_time'] = travel_time
                detail['is_line_change'] = (line != next_line)

            details.append(detail)
            current_line = line

        return details

    def find_route(self, start: str, end: str) -> dict:
        """
        경로 탐색 및 결과 출력

        Args:
            start: 출발역
            end: 도착역

        Returns:
            경로 정보 딕셔너리
        """
        path, total_time, details = self.dijkstra(start, end)

        if not path:
            print(f"\n{start} → {end}: 경로를 찾을 수 없습니다.")
            return {}

        print(f"\n{'='*60}")
        print(f"경로: {start} → {end}")
        print(f"{'='*60}")

        # 경로 요약
        transfers = sum(1 for d in details if d.get('is_line_change', False))
        stations = len(path)

        print(f"\n📊 요약")
        print(f"   총 소요시간: {total_time}초 ({total_time // 60}분 {total_time % 60}초)")
        print(f"   정거장 수: {stations}개")
        print(f"   환승 횟수: {transfers}회")

        # 상세 경로
        print(f"\n📍 상세 경로")
        current_line = None

        for i, detail in enumerate(details):
            station = detail['station']
            line = detail['line']

            if detail.get('is_transfer'):
                print(f"   {'─'*40}")
                print(f"   🔄 환승: {current_line}호선 → {line}호선")
                print(f"   {'─'*40}")

            travel_time = detail.get('travel_time', 0)
            if i < len(details) - 1:
                next_station = detail.get('next_station', '')
                print(f"   {station} ({line}호선) → {next_station} [{travel_time}초]")
            else:
                print(f"   {station} ({line}호선) [도착]")

            current_line = line

        return {
            'path': path,
            'total_time': total_time,
            'transfers': transfers,
            'stations': stations,
            'details': details
        }

    def save(self, output_dir: str):
        """그래프 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # 그래프 저장
        graph_path = os.path.join(output_dir, 'subway_graph_v2.json')
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

        # 메타데이터 저장
        meta = {
            'station_lines': {k: list(v) for k, v in self.station_lines.items()},
            'transfer_stations': list(self.transfer_stations),
            'transfer_times': self.transfer_times,
            'default_transfer_time': self.default_transfer_time
        }
        meta_path = os.path.join(output_dir, 'subway_meta_v2.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료: {output_dir}")


def main():
    """메인 실행"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data_processing/data/서울교통공사 역간 거리 및 소요시간 정보.csv')
    output_dir = os.path.join(base_dir, 'subway_graph/data')

    # 그래프 생성
    graph = SubwayGraphV2()
    graph.build_graph(data_path)

    # 저장
    graph.save(output_dir)

    # 테스트
    print("\n" + "=" * 60)
    print("경로 탐색 테스트")
    print("=" * 60)

    # 테스트 케이스
    test_cases = [
        ("강남", "홍대입구"),
        ("서울역", "잠실"),
        ("신도림", "왕십리"),
    ]

    for start, end in test_cases:
        graph.find_route(start, end)

    return graph


if __name__ == '__main__':
    graph = main()
