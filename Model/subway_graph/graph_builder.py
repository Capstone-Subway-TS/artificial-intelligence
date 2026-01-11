"""
서울 지하철 그래프 구축

Step 1: 완전한 지하철 네트워크 그래프 생성
- 1~8호선 전체 역 포함
- 실제 소요시간 가중치 적용
- 양방향 연결
"""

import pandas as pd
import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import os


@dataclass
class Station:
    """역 정보"""
    name: str
    line: int

    def __hash__(self):
        return hash((self.name, self.line))

    def __eq__(self, other):
        return self.name == other.name and self.line == other.line


class SubwayGraphBuilder:
    """지하철 그래프 빌더"""

    def __init__(self):
        self.graph: Dict[str, Dict[str, float]] = {}  # {역명: {인접역: 소요시간(초)}}
        self.station_lines: Dict[str, Set[int]] = {}  # {역명: {호선들}}
        self.line_stations: Dict[int, List[str]] = {}  # {호선: [역 순서]}
        self.transfer_stations: Set[str] = set()  # 환승역 목록

    def parse_time_to_seconds(self, time_str: str) -> int:
        """시간 문자열을 초로 변환 (예: '2:30' -> 150초)"""
        if pd.isna(time_str) or time_str == '0:00':
            return 0

        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        return 0

    def load_station_data(self, filepath: str, encoding: str = 'cp949'):
        """역간 소요시간 데이터 로드"""
        df = pd.read_csv(filepath, encoding=encoding)

        print("=" * 60)
        print("Step 1-1: 데이터 로드")
        print("=" * 60)
        print(f"총 레코드: {len(df)}개")

        return df

    def build_line_graph(self, df: pd.DataFrame):
        """호선별 그래프 구축 (같은 호선 내 역 연결)"""

        print("\n" + "=" * 60)
        print("Step 1-2: 호선별 그래프 구축")
        print("=" * 60)

        for line in sorted(df['호선'].unique()):
            line_df = df[df['호선'] == line].reset_index(drop=True)
            stations = line_df['역명'].tolist()
            times = line_df['시간(분)'].tolist()

            self.line_stations[line] = stations

            print(f"\n{line}호선: {len(stations)}개 역")

            for i, station in enumerate(stations):
                # 역-호선 매핑
                if station not in self.station_lines:
                    self.station_lines[station] = set()
                self.station_lines[station].add(line)

                # 그래프 초기화
                if station not in self.graph:
                    self.graph[station] = {}

                # 다음 역과 연결 (양방향)
                if i < len(stations) - 1:
                    next_station = stations[i + 1]
                    next_time = self.parse_time_to_seconds(times[i + 1])

                    if next_time > 0:
                        # 현재 → 다음
                        self.graph[station][next_station] = next_time

                        # 다음 → 현재 (역방향)
                        if next_station not in self.graph:
                            self.graph[next_station] = {}
                        self.graph[next_station][station] = next_time

            # 2호선 순환선 처리 (시청 - 시청 연결)
            if line == 2:
                # 순환선: 까치산은 지선이므로 실제로는 시청-시청 순환
                # 데이터에서 순환 여부 확인 필요
                pass

        print(f"\n총 역 수: {len(self.graph)}개")

    def identify_transfer_stations(self):
        """환승역 식별 (2개 이상 호선이 지나는 역)"""

        print("\n" + "=" * 60)
        print("Step 1-3: 환승역 식별")
        print("=" * 60)

        for station, lines in self.station_lines.items():
            if len(lines) >= 2:
                self.transfer_stations.add(station)

        print(f"환승역 수: {len(self.transfer_stations)}개")
        print("\n환승역 목록:")

        # 호선 수로 정렬
        sorted_transfers = sorted(
            self.transfer_stations,
            key=lambda x: len(self.station_lines[x]),
            reverse=True
        )

        for station in sorted_transfers[:20]:  # 상위 20개만 출력
            lines = sorted(self.station_lines[station])
            lines_str = ", ".join([f"{l}호선" for l in lines])
            print(f"  {station}: {lines_str}")

        if len(sorted_transfers) > 20:
            print(f"  ... 외 {len(sorted_transfers) - 20}개")

    def add_transfer_connections(self, default_transfer_time: int = 180):
        """
        환승역에 호선 간 연결 추가

        환승 시 같은 역이름이지만 다른 호선으로 이동하는 것을 표현
        예: 시청(1호선) ←→ 시청(2호선)

        현재는 단순화를 위해 같은 역명은 하나의 노드로 처리하므로,
        이미 연결되어 있음. 하지만 환승 도보 시간을 별도로 추적해야 함.

        Args:
            default_transfer_time: 기본 환승 도보 시간 (초), 기본값 3분
        """
        print("\n" + "=" * 60)
        print("Step 2: 환승 연결 추가")
        print("=" * 60)

        # 환승역별 환승 시간 (실제 데이터 기반, 없으면 기본값 사용)
        # 일부 환승역은 환승 거리가 길어서 시간이 더 걸림
        transfer_times = {
            # 3호선 이상 환승역 (환승 시간 긺)
            '종로3가': 240,      # 4분 (1-3-5호선)
            '동대문역사문화공원': 300,  # 5분 (2-4-5호선)

            # 일반 환승역
            '시청': 180,         # 3분
            '신도림': 180,       # 3분
            '왕십리': 240,       # 4분
            '고속터미널': 240,   # 4분
            '사당': 180,         # 3분
            '교대': 180,         # 3분
            '잠실': 180,         # 3분
            '강남': 180,         # 3분 (신분당선 환승 시 더 김, 현재는 미포함)
        }

        # 환승 정보 저장용 딕셔너리
        self.transfer_times = {}

        for station in self.transfer_stations:
            # 해당 역에서 환승 가능한 호선들
            lines = self.station_lines[station]

            if len(lines) >= 2:
                # 환승 시간 결정
                transfer_time = transfer_times.get(station, default_transfer_time)
                self.transfer_times[station] = transfer_time

        print(f"환승 시간 설정 완료: {len(self.transfer_times)}개 역")
        print(f"기본 환승 시간: {default_transfer_time}초 ({default_transfer_time//60}분)")
        print("\n특별 환승 시간 적용 역:")
        for station, time in sorted(transfer_times.items(), key=lambda x: x[1], reverse=True):
            if station in self.transfer_stations:
                print(f"  {station}: {time}초 ({time//60}분)")

        return self.transfer_times

    def get_graph_stats(self):
        """그래프 통계"""

        print("\n" + "=" * 60)
        print("Step 1-4: 그래프 통계")
        print("=" * 60)

        total_edges = sum(len(neighbors) for neighbors in self.graph.values())
        avg_time = 0
        edge_count = 0

        for station, neighbors in self.graph.items():
            for neighbor, time in neighbors.items():
                avg_time += time
                edge_count += 1

        if edge_count > 0:
            avg_time /= edge_count

        print(f"총 역 수: {len(self.graph)}개")
        print(f"총 엣지 수: {total_edges}개 (양방향)")
        print(f"환승역 수: {len(self.transfer_stations)}개")
        print(f"평균 역간 소요시간: {avg_time:.0f}초 ({avg_time/60:.1f}분)")

        # 호선별 통계
        print("\n호선별 역 수:")
        for line in sorted(self.line_stations.keys()):
            print(f"  {line}호선: {len(self.line_stations[line])}개")

    def save_graph(self, output_dir: str):
        """그래프 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 그래프 저장 (JSON)
        graph_path = os.path.join(output_dir, 'subway_graph.json')
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

        # 2. 역-호선 매핑 저장 (int64 -> int 변환)
        station_lines_path = os.path.join(output_dir, 'station_lines.json')
        station_lines_serializable = {k: [int(x) for x in v] for k, v in self.station_lines.items()}
        with open(station_lines_path, 'w', encoding='utf-8') as f:
            json.dump(station_lines_serializable, f, ensure_ascii=False, indent=2)

        # 3. 환승역 저장
        transfer_path = os.path.join(output_dir, 'transfer_stations.json')
        with open(transfer_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.transfer_stations), f, ensure_ascii=False, indent=2)

        # 4. 호선별 역 순서 저장 (int64 키 -> str 변환)
        line_stations_path = os.path.join(output_dir, 'line_stations.json')
        line_stations_serializable = {str(k): v for k, v in self.line_stations.items()}
        with open(line_stations_path, 'w', encoding='utf-8') as f:
            json.dump(line_stations_serializable, f, ensure_ascii=False, indent=2)

        print(f"\n그래프 저장 완료: {output_dir}")
        print(f"  - subway_graph.json: 역간 연결 및 소요시간")
        print(f"  - station_lines.json: 역별 호선 정보")
        print(f"  - transfer_stations.json: 환승역 목록")
        print(f"  - line_stations.json: 호선별 역 순서")

        return graph_path

    def test_path_finding(self):
        """경로 탐색 테스트"""
        import heapq

        print("\n" + "=" * 60)
        print("Step 1-5: 경로 탐색 테스트")
        print("=" * 60)

        def dijkstra(start: str, end: str) -> Tuple[List[str], float]:
            """기본 Dijkstra 알고리즘"""
            if start not in self.graph or end not in self.graph:
                return [], float('inf')

            distances = {node: float('inf') for node in self.graph}
            distances[start] = 0
            previous = {node: None for node in self.graph}
            queue = [(0, start)]

            while queue:
                current_dist, current = heapq.heappop(queue)

                if current == end:
                    break

                if current_dist > distances[current]:
                    continue

                for neighbor, weight in self.graph[current].items():
                    distance = current_dist + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(queue, (distance, neighbor))

            # 경로 복원
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()

            return path, distances[end]

        # 테스트 케이스
        test_cases = [
            ("시청", "동대문역사문화공원"),  # 같은 호선
            ("강남", "홍대입구"),  # 다른 호선 (환승 필요)
            ("서울역", "잠실"),  # 장거리
        ]

        for start, end in test_cases:
            if start in self.graph and end in self.graph:
                path, total_time = dijkstra(start, end)
                if path:
                    print(f"\n{start} → {end}")
                    print(f"  경로: {' → '.join(path)}")
                    print(f"  소요시간: {total_time:.0f}초 ({total_time/60:.1f}분)")
                    print(f"  정거장 수: {len(path)}개")
                else:
                    print(f"\n{start} → {end}: 경로 없음")
            else:
                missing = []
                if start not in self.graph:
                    missing.append(start)
                if end not in self.graph:
                    missing.append(end)
                print(f"\n{start} → {end}: 역 없음 ({missing})")


def main():
    """메인 실행"""
    # 현재 스크립트 위치 기준 경로
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data_processing/data/서울교통공사 역간 거리 및 소요시간 정보.csv')
    output_dir = os.path.join(base_dir, 'subway_graph/data')

    # 그래프 빌더 생성
    builder = SubwayGraphBuilder()

    # 1. 데이터 로드
    df = builder.load_station_data(data_path)

    # 2. 그래프 구축
    builder.build_line_graph(df)

    # 3. 환승역 식별
    builder.identify_transfer_stations()

    # 4. 통계 출력
    builder.get_graph_stats()

    # 5. 저장
    builder.save_graph(output_dir)

    # 6. 테스트
    builder.test_path_finding()

    return builder


if __name__ == '__main__':
    builder = main()
