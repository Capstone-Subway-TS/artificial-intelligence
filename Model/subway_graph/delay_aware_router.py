"""
지연 인식 경로 탐색기 (Delay-Aware Router)

핵심 기능:
- 구간별 예측 지연시간 반영
- 환승 연결 실패 확률 계산
- 환승 실패 시 추가 대기시간 반영

환승 연결 실패 로직:
1. 환승역 도착 시 예상 지연시간 계산
2. 지연시간이 환승 여유시간을 초과하면 → 환승 실패 가능성
3. 환승 실패 시 → 다음 열차까지 배차 간격만큼 대기
4. 기대 비용 = P(실패) × 배차간격 추가
"""

import json
import heapq
import os
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RouteSegment:
    """경로 구간 정보"""
    from_station: str
    to_station: str
    line: int
    base_time: int  # 기본 소요시간 (초)
    predicted_delay: int = 0  # 예측 지연시간 (초)
    is_transfer: bool = False  # 환승 여부
    transfer_wait: int = 0  # 환승 대기시간 (초)


@dataclass
class RouteResult:
    """경로 탐색 결과"""
    path: List[str]  # 노드 경로
    total_time: int  # 총 예상 소요시간
    base_time: int  # 기본 소요시간 (지연 없이)
    total_delay: int  # 총 예상 지연시간
    transfer_count: int  # 환승 횟수
    transfer_failure_risk: float  # 환승 실패 위험도 (0~1)
    segments: List[RouteSegment] = field(default_factory=list)


class DelayAwareRouter:
    """지연을 고려한 경로 탐색기"""

    def __init__(self, graph_dir: str = None):
        if graph_dir is None:
            graph_dir = os.path.join(os.path.dirname(__file__), 'data')

        self.graph_dir = graph_dir

        # 그래프 로드
        self.graph: Dict[str, Dict[str, int]] = {}
        self.station_lines: Dict[str, List[int]] = {}
        self.transfer_stations: Set[str] = set()
        self.transfer_times: Dict[str, int] = {}

        # 배차 간격 데이터
        self.headway_data: Dict = {}

        # 구간별 예측 지연 (ML 모델에서 가져옴)
        # 형식: {("역1", "역2", 호선): 예측지연(초)}
        self.predicted_delays: Dict[Tuple[str, str, int], int] = {}

        # 설정
        self.transfer_margin = 60  # 환승 여유시간 (초)
        self.default_headway = 300  # 기본 배차간격 5분

        # 데이터 로드
        self._load_data()

    def _load_data(self):
        """데이터 로드"""
        # 그래프 (V2)
        graph_path = os.path.join(self.graph_dir, 'subway_graph_v2.json')
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.graph = json.load(f)
        else:
            # V1 그래프 폴백
            graph_path = os.path.join(self.graph_dir, 'subway_graph.json')
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.graph = json.load(f)

        # 메타데이터
        meta_path = os.path.join(self.graph_dir, 'subway_meta_v2.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                self.station_lines = meta.get('station_lines', {})
                self.transfer_stations = set(meta.get('transfer_stations', []))
                self.transfer_times = meta.get('transfer_times', {})

        # 배차 간격
        headway_path = os.path.join(self.graph_dir, 'headway_intervals.json')
        if os.path.exists(headway_path):
            with open(headway_path, 'r', encoding='utf-8') as f:
                self.headway_data = json.load(f)

        print(f"데이터 로드 완료:")
        print(f"  - 노드 수: {len(self.graph)}")
        print(f"  - 환승역 수: {len(self.transfer_stations)}")

    def make_node_id(self, station: str, line: int) -> str:
        """노드 ID 생성"""
        return f"{station}_{line}호선"

    def parse_node_id(self, node_id: str) -> Tuple[str, int]:
        """노드 ID 파싱"""
        parts = node_id.rsplit('_', 1)
        station = parts[0]
        line = int(parts[1].replace('호선', ''))
        return station, line

    def is_transfer_edge(self, from_node: str, to_node: str) -> bool:
        """환승 엣지인지 확인"""
        from_station, from_line = self.parse_node_id(from_node)
        to_station, to_line = self.parse_node_id(to_node)
        return from_station == to_station and from_line != to_line

    def get_headway(self, line: int, time_category: str = 'normal') -> int:
        """
        호선별 배차 간격 조회 (초)

        Args:
            line: 호선 번호
            time_category: 'rush_hour', 'normal', 'late_night', 'weekend'

        Returns:
            배차 간격 (초)
        """
        if not self.headway_data:
            return self.default_headway

        line_str = str(line)

        # 주중
        if time_category in ['rush_hour', 'normal', 'late_night']:
            weekday = self.headway_data.get('weekday', {})
            interval = weekday.get(time_category, {}).get('lines', {}).get(line_str)
            if interval:
                return int(interval * 60)  # 분 → 초

        # 주말
        elif time_category == 'weekend':
            weekend = self.headway_data.get('weekend', {})
            interval = weekend.get('lines', {}).get(line_str)
            if interval:
                return int(interval * 60)

        return self.default_headway

    def get_time_category(self, hour: int = None, is_weekend: bool = False) -> str:
        """시간대 카테고리 반환"""
        if is_weekend:
            return 'weekend'

        if hour is None:
            hour = datetime.now().hour

        if 7 <= hour < 9 or 18 <= hour < 20:
            return 'rush_hour'
        elif 22 <= hour or hour < 5:
            return 'late_night'
        else:
            return 'normal'

    def set_predicted_delays(self, delays: Dict[Tuple[str, str, int], int]):
        """
        ML 모델의 예측 지연시간 설정

        Args:
            delays: {(출발역, 도착역, 호선): 예측지연(초)}
        """
        self.predicted_delays = delays

    def get_segment_delay(self, from_station: str, to_station: str, line: int) -> int:
        """구간의 예측 지연시간 조회"""
        return self.predicted_delays.get((from_station, to_station, line), 0)

    def calculate_transfer_failure_cost(
        self,
        accumulated_delay: int,
        next_line: int,
        time_category: str = 'normal'
    ) -> Tuple[float, int]:
        """
        환승 연결 실패 비용 계산

        Args:
            accumulated_delay: 누적 지연시간 (초)
            next_line: 환승할 호선
            time_category: 시간대

        Returns:
            (실패 확률, 기대 추가 대기시간)

        로직:
        - 지연이 환승 여유시간을 초과하면 실패 확률 증가
        - 실패 확률 = min(1, (지연 - 여유시간) / 배차간격)
        - 기대 추가 대기 = 실패확률 × 배차간격
        """
        if accumulated_delay <= self.transfer_margin:
            # 여유시간 내면 실패 없음
            return 0.0, 0

        headway = self.get_headway(next_line, time_category)

        # 지연이 여유시간을 초과한 양
        excess_delay = accumulated_delay - self.transfer_margin

        # 실패 확률 (선형 모델)
        # 배차간격만큼 늦으면 100% 실패
        failure_prob = min(1.0, excess_delay / headway)

        # 기대 추가 대기시간
        expected_wait = int(failure_prob * headway)

        return failure_prob, expected_wait

    def dijkstra_with_delays(
        self,
        start_station: str,
        end_station: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        time_category: str = 'normal'
    ) -> RouteResult:
        """
        지연을 고려한 최단 경로 탐색

        Args:
            start_station: 출발역
            end_station: 도착역
            start_line: 출발 호선 (옵션)
            end_line: 도착 호선 (옵션)
            time_category: 시간대

        Returns:
            RouteResult
        """
        # 시작/종료 노드 결정
        if start_line:
            start_nodes = [self.make_node_id(start_station, start_line)]
        else:
            lines = self.station_lines.get(start_station, [])
            start_nodes = [self.make_node_id(start_station, line) for line in lines]

        if end_line:
            end_nodes = [self.make_node_id(end_station, end_line)]
        else:
            lines = self.station_lines.get(end_station, [])
            end_nodes = [self.make_node_id(end_station, line) for line in lines]

        if not start_nodes or not end_nodes:
            return RouteResult([], float('inf'), 0, 0, 0, 0.0)

        # 다익스트라 (확장 상태: (노드, 누적지연))
        # 비용 = 기본시간 + 지연 + 환승실패비용
        INF = float('inf')

        # distances[node] = (total_cost, base_time, accumulated_delay, transfer_failures)
        distances = {node: (INF, 0, 0, 0.0) for node in self.graph}
        previous = {node: None for node in self.graph}

        # (total_cost, base_time, acc_delay, transfer_failures, node)
        queue = []
        for start_node in start_nodes:
            if start_node in distances:
                distances[start_node] = (0, 0, 0, 0.0)
                heapq.heappush(queue, (0, 0, 0, 0.0, start_node))

        while queue:
            current_cost, base_time, acc_delay, transfer_fail, current = heapq.heappop(queue)

            if current_cost > distances[current][0]:
                continue

            for neighbor, edge_weight in self.graph[current].items():
                from_station, from_line = self.parse_node_id(current)
                to_station, to_line = self.parse_node_id(neighbor)

                # 기본 이동 시간
                new_base_time = base_time + edge_weight

                # 환승 엣지인지 확인
                is_transfer = self.is_transfer_edge(current, neighbor)

                if is_transfer:
                    # 환승: 누적 지연으로 인한 연결 실패 비용 계산
                    fail_prob, fail_cost = self.calculate_transfer_failure_cost(
                        acc_delay, to_line, time_category
                    )

                    new_acc_delay = 0  # 환승 후 지연 리셋 (새 열차)
                    new_transfer_fail = transfer_fail + fail_prob
                    new_cost = current_cost + edge_weight + fail_cost

                else:
                    # 일반 이동: 구간 지연 추가
                    segment_delay = self.get_segment_delay(from_station, to_station, from_line)
                    new_acc_delay = acc_delay + segment_delay
                    new_transfer_fail = transfer_fail
                    new_cost = current_cost + edge_weight + segment_delay

                if new_cost < distances[neighbor][0]:
                    distances[neighbor] = (new_cost, new_base_time, new_acc_delay, new_transfer_fail)
                    previous[neighbor] = current
                    heapq.heappush(queue, (new_cost, new_base_time, new_acc_delay, new_transfer_fail, neighbor))

        # 최적 종료 노드 찾기
        best_end = None
        best_info = (INF, 0, 0, 0.0)

        for end_node in end_nodes:
            if end_node in distances and distances[end_node][0] < best_info[0]:
                best_info = distances[end_node]
                best_end = end_node

        if best_end is None or best_info[0] == INF:
            return RouteResult([], float('inf'), 0, 0, 0, 0.0)

        # 경로 복원
        path = []
        current = best_end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        # 환승 횟수 계산
        transfer_count = 0
        for i in range(len(path) - 1):
            if self.is_transfer_edge(path[i], path[i + 1]):
                transfer_count += 1

        total_cost, base_time, _, transfer_fail = best_info
        total_delay = int(total_cost - base_time)

        return RouteResult(
            path=path,
            total_time=int(total_cost),
            base_time=base_time,
            total_delay=total_delay,
            transfer_count=transfer_count,
            transfer_failure_risk=transfer_fail
        )

    def find_route(
        self,
        start: str,
        end: str,
        hour: int = None,
        is_weekend: bool = False
    ) -> RouteResult:
        """
        경로 탐색 및 출력

        Args:
            start: 출발역
            end: 도착역
            hour: 시간 (0-23)
            is_weekend: 주말 여부
        """
        time_category = self.get_time_category(hour, is_weekend)
        result = self.dijkstra_with_delays(start, end, time_category=time_category)

        if not result.path:
            print(f"\n{start} → {end}: 경로를 찾을 수 없습니다.")
            return result

        print(f"\n{'='*60}")
        print(f"경로: {start} → {end}")
        print(f"시간대: {time_category}")
        print(f"{'='*60}")

        # 요약
        print(f"\n[요약]")
        print(f"  총 예상 소요시간: {result.total_time}초 ({result.total_time // 60}분 {result.total_time % 60}초)")
        print(f"  기본 소요시간: {result.base_time}초 ({result.base_time // 60}분)")
        print(f"  예상 지연: {result.total_delay}초")
        print(f"  환승 횟수: {result.transfer_count}회")
        print(f"  환승 실패 위험도: {result.transfer_failure_risk:.1%}")

        # 상세 경로
        print(f"\n[상세 경로]")
        current_line = None

        for i, node in enumerate(result.path):
            station, line = self.parse_node_id(node)

            # 환승 표시
            if current_line is not None and line != current_line:
                headway = self.get_headway(line, time_category)
                print(f"  {'─'*40}")
                print(f"  [환승] {current_line}호선 → {line}호선")
                print(f"         (배차간격: {headway // 60}분)")
                print(f"  {'─'*40}")

            # 다음 역
            if i < len(result.path) - 1:
                next_node = result.path[i + 1]
                travel_time = self.graph[node].get(next_node, 0)
                next_station, next_line = self.parse_node_id(next_node)

                # 구간 지연
                if line == next_line:
                    delay = self.get_segment_delay(station, next_station, line)
                    delay_str = f" (+{delay}초 지연)" if delay > 0 else ""
                else:
                    delay_str = ""

                print(f"  {station} ({line}호선) → {next_station} [{travel_time}초{delay_str}]")
            else:
                print(f"  {station} ({line}호선) [도착]")

            current_line = line

        return result

    def compare_routes(
        self,
        start: str,
        end: str,
        hour: int = None,
        is_weekend: bool = False
    ):
        """
        지연 고려/미고려 경로 비교

        지연 정보가 설정된 경우, 지연을 고려한 경로와
        고려하지 않은 경로를 비교합니다.
        """
        time_category = self.get_time_category(hour, is_weekend)

        # 지연 고려 경로
        result_with_delay = self.dijkstra_with_delays(start, end, time_category=time_category)

        # 지연 미고려 경로 (예측 지연을 0으로 설정)
        saved_delays = self.predicted_delays.copy()
        self.predicted_delays = {}
        result_no_delay = self.dijkstra_with_delays(start, end, time_category=time_category)
        self.predicted_delays = saved_delays

        print(f"\n{'='*60}")
        print(f"경로 비교: {start} → {end}")
        print(f"{'='*60}")

        print(f"\n[지연 미고려 경로]")
        print(f"  경로: {' → '.join([self.parse_node_id(n)[0] for n in result_no_delay.path])}")
        print(f"  소요시간: {result_no_delay.total_time}초 ({result_no_delay.total_time // 60}분)")
        print(f"  환승: {result_no_delay.transfer_count}회")

        print(f"\n[지연 고려 경로]")
        print(f"  경로: {' → '.join([self.parse_node_id(n)[0] for n in result_with_delay.path])}")
        print(f"  예상 소요시간: {result_with_delay.total_time}초 ({result_with_delay.total_time // 60}분)")
        print(f"  환승: {result_with_delay.transfer_count}회")
        print(f"  환승 실패 위험: {result_with_delay.transfer_failure_risk:.1%}")

        # 경로가 다른 경우
        if result_no_delay.path != result_with_delay.path:
            time_diff = result_no_delay.total_time - result_with_delay.total_time
            print(f"\n[분석]")
            print(f"  지연 고려 시 다른 경로 추천!")
            print(f"  예상 시간 절감: {abs(time_diff)}초")

        return result_no_delay, result_with_delay


def demo():
    """데모 실행"""
    print("="*60)
    print("지연 인식 경로 탐색기 (Delay-Aware Router) 데모")
    print("="*60)

    # 라우터 생성
    router = DelayAwareRouter()

    # 예측 지연 설정 (가상 데이터)
    # 실제로는 ML 모델에서 예측한 값을 사용
    sample_delays = {
        # 2호선 강남-역삼 구간이 혼잡한 상황 가정
        ("강남", "역삼", 2): 60,  # 1분 지연
        ("역삼", "선릉", 2): 45,  # 45초 지연
        ("선릉", "삼성", 2): 30,  # 30초 지연

        # 2호선 신도림 방면도 약간 지연
        ("교대", "서초", 2): 20,
        ("서초", "방배", 2): 15,
    }

    router.set_predicted_delays(sample_delays)

    print("\n[가상 지연 데이터 설정]")
    for (fr, to, line), delay in sample_delays.items():
        print(f"  {fr} → {to} ({line}호선): {delay}초 지연")

    # 테스트 1: 일반 경로 탐색
    print("\n" + "="*60)
    print("테스트 1: 기본 경로 탐색")
    print("="*60)
    router.find_route("강남", "홍대입구", hour=9)

    # 테스트 2: 환승이 있는 경로
    print("\n" + "="*60)
    print("테스트 2: 환승이 있는 경로")
    print("="*60)
    router.find_route("서울역", "잠실", hour=18)  # 퇴근 시간

    # 테스트 3: 경로 비교
    print("\n" + "="*60)
    print("테스트 3: 지연 고려/미고려 경로 비교")
    print("="*60)
    router.compare_routes("강남", "왕십리", hour=8)  # 출근 시간

    return router


def demo_transfer_failure():
    """환승 연결 실패 시나리오 데모"""
    print("="*60)
    print("환승 연결 실패 시나리오 데모")
    print("="*60)

    router = DelayAwareRouter()

    # 시나리오: 4호선이 심하게 지연되어 사당역에서 2호선 환승이 위험한 상황
    # 서울역(4호선) → 사당(4→2호선 환승) → 강남(2호선)
    heavy_delays = {
        # 4호선 지연 (서울역 방면에서 사당까지 지연 누적)
        ("서울역", "회현", 4): 30,
        ("회현", "명동", 4): 45,
        ("명동", "충무로", 4): 40,
        ("충무로", "동대문역사문화공원", 4): 35,
        # ... 실제로는 모든 구간에 지연 설정

        # 간단히 사당 직전 구간에 큰 지연 설정
        ("총신대입구", "사당", 4): 120,  # 2분 지연 (환승 실패 가능)
    }

    router.set_predicted_delays(heavy_delays)

    print("\n[시나리오]")
    print("  4호선이 심하게 지연되는 상황")
    print("  총신대입구 → 사당: 120초 지연")
    print("  사당역에서 2호선으로 환승 시 열차를 놓칠 수 있음")

    print("\n" + "-"*60)
    print("환승 실패 비용 계산 예시")
    print("-"*60)

    # 환승 실패 비용 직접 계산
    for delay in [0, 30, 60, 90, 120, 180]:
        prob, cost = router.calculate_transfer_failure_cost(
            accumulated_delay=delay,
            next_line=2,
            time_category='rush_hour'
        )
        headway = router.get_headway(2, 'rush_hour')
        print(f"  누적지연 {delay:3d}초 → 실패확률 {prob:5.1%}, 추가대기 {cost:3d}초 (배차간격: {headway}초)")

    print("\n" + "-"*60)
    print("경로 탐색: 서울역 → 강남 (4호선 지연 상황)")
    print("-"*60)

    # 러시아워 (배차간격 짧음)
    print("\n[러시아워]")
    router.find_route("서울역", "강남", hour=8)

    # 심야 (배차간격 김)
    print("\n[심야 시간대]")
    router.find_route("서울역", "강남", hour=23)

    return router


if __name__ == '__main__':
    router = demo()
    print("\n\n")
    router2 = demo_transfer_failure()


if __name__ == '__main__':
    router = demo()
