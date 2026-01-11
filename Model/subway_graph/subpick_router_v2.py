"""
SUBPICK 통합 라우터 V2 - ODsay API 연동

ODsay API를 통한 실제 시간표/배차간격 데이터 사용
환승 연결 실패 확률을 더 정확하게 계산

사용법:
    export ODSAY_API_KEY=your_key
    python subpick_router_v2.py
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from delay_aware_router import DelayAwareRouter, RouteResult

# ODsay API 사용 가능 여부
try:
    from odsay_transit import ODsayAPI
    ODSAY_AVAILABLE = True
except ImportError:
    ODSAY_AVAILABLE = False


@dataclass
class SubpickRecommendation:
    """SUBPICK 경로 추천 결과"""
    primary_route: RouteResult
    alternative_routes: List[RouteResult]
    predicted_delays: Dict[str, int]
    risk_assessment: Dict[str, Any]
    recommendation_reason: str
    headway_info: Dict[str, int]  # 환승역별 배차간격


class SubpickRouterV2:
    """SUBPICK 통합 라우터 V2 (ODsay 연동)"""

    def __init__(
        self,
        model_path: str = None,
        graph_dir: str = None,
        odsay_api_key: str = None
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if graph_dir is None:
            graph_dir = os.path.join(base_dir, 'data')

        # 라우터 초기화
        self.router = DelayAwareRouter(graph_dir)

        # ODsay API
        self.odsay = None
        self._init_odsay(odsay_api_key)

        # 캐시된 배차간격
        self.headway_cache = self._load_headway_cache()

        # 구간 정보
        self.segment_info = self._build_segment_info()

    def _init_odsay(self, api_key: str = None):
        """ODsay API 초기화"""
        if not ODSAY_AVAILABLE:
            print("ODsay 모듈 없음 - 기본 배차간격 사용")
            return

        key = api_key or os.environ.get('ODSAY_API_KEY')
        if not key:
            print("ODSAY_API_KEY 없음 - 기본 배차간격 사용")
            return

        try:
            self.odsay = ODsayAPI(key)
            print("ODsay API 연결 성공")
        except Exception as e:
            print(f"ODsay API 초기화 실패: {e}")

    def _load_headway_cache(self) -> Dict:
        """캐시된 배차간격 로드"""
        cache_path = os.path.join(
            os.path.dirname(__file__), 'data', 'odsay_headways.json'
        )

        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return {}

    def _build_segment_info(self) -> Dict:
        """구간 정보 구축"""
        segments = {}
        for node, neighbors in self.router.graph.items():
            from_station, from_line = self.router.parse_node_id(node)
            for neighbor, travel_time in neighbors.items():
                to_station, to_line = self.router.parse_node_id(neighbor)
                if from_line == to_line and from_station != to_station:
                    segments[(from_station, to_station, from_line)] = {
                        'base_time': travel_time,
                        'line': from_line
                    }
        return segments

    def get_real_headway(
        self,
        station: str,
        line: int,
        hour: int,
        is_weekend: bool = False
    ) -> int:
        """
        실제 배차간격 조회 (ODsay API 또는 캐시)

        Args:
            station: 역명
            line: 호선
            hour: 시간 (0-23)
            is_weekend: 주말 여부

        Returns:
            배차간격 (초)
        """
        # 1. ODsay API로 실시간 조회
        if self.odsay:
            try:
                day_type = 3 if is_weekend else 1  # 1=평일, 3=휴일
                headway = self.odsay.get_headway_from_timetable(
                    station, line, hour, direction=1, day_type=day_type
                )
                if headway > 0:
                    return headway
            except Exception as e:
                pass  # 실패시 캐시/기본값 사용

        # 2. 캐시에서 조회
        if station in self.headway_cache:
            line_data = self.headway_cache[station].get(str(line), {})

            # 시간대별 매핑
            if 7 <= hour <= 9:
                period = 'rush_morning'
            elif 18 <= hour <= 20:
                period = 'rush_evening'
            elif 21 <= hour or hour <= 5:
                period = 'late_night'
            else:
                period = 'normal'

            if period in line_data:
                return line_data[period]

        # 3. 기본 배차간격 (호선별)
        return self.router.get_headway(line, self.router.get_time_category(hour, is_weekend))

    def calculate_transfer_risk(
        self,
        accumulated_delay: int,
        station: str,
        next_line: int,
        hour: int,
        is_weekend: bool = False
    ) -> Tuple[float, int, int]:
        """
        환승 실패 위험 계산 (실제 배차간격 기반)

        Returns:
            (실패확률, 추가대기시간, 실제배차간격)
        """
        headway = self.get_real_headway(station, next_line, hour, is_weekend)

        # 환승 여유시간 (60초)
        margin = 60

        if accumulated_delay <= margin:
            return 0.0, 0, headway

        excess = accumulated_delay - margin
        failure_prob = min(1.0, excess / headway)
        expected_wait = int(failure_prob * headway)

        return failure_prob, expected_wait, headway

    def recommend_route(
        self,
        start: str,
        end: str,
        departure_time: str = None,
        max_alternatives: int = 2
    ) -> SubpickRecommendation:
        """
        최적 경로 추천 (ODsay 배차간격 반영)
        """
        # 시간 파싱
        if departure_time:
            if isinstance(departure_time, str):
                dt = datetime.strptime(departure_time, "%Y-%m-%d %H:%M")
            else:
                dt = departure_time
        else:
            dt = datetime.now()

        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        time_category = self.router.get_time_category(hour, is_weekend)

        print(f"\n{'='*60}")
        print(f"SUBPICK V2 경로 추천 (ODsay 연동)")
        print(f"{'='*60}")
        print(f"출발: {start} → 도착: {end}")
        print(f"시간: {dt.strftime('%Y-%m-%d %H:%M')} ({time_category})")

        if self.odsay:
            print(f"배차간격: ODsay API 실시간 조회")
        else:
            print(f"배차간격: 기본값 사용")

        # 지연 예측 (간단한 시뮬레이션)
        predicted_delays = self._simulate_delays(time_category)
        self.router.set_predicted_delays(predicted_delays)

        # 경로 탐색
        primary = self.router.dijkstra_with_delays(
            start, end, time_category=time_category
        )

        # 환승역별 실제 배차간격 수집
        headway_info = self._collect_transfer_headways(primary, hour, is_weekend)

        # 대안 경로
        alternatives = self._find_alternatives(start, end, primary, time_category, max_alternatives)

        # 위험 평가
        risk = self._assess_risk_v2(primary, headway_info, time_category)

        # 추천 이유
        reason = self._generate_reason(primary, alternatives, risk, headway_info)

        # 출력
        self._print_result(primary, alternatives, risk, reason, headway_info)

        return SubpickRecommendation(
            primary_route=primary,
            alternative_routes=alternatives,
            predicted_delays={f"{k[0]}-{k[1]}": v for k, v in predicted_delays.items() if v > 0},
            risk_assessment=risk,
            recommendation_reason=reason,
            headway_info=headway_info
        )

    def _simulate_delays(self, time_category: str) -> Dict:
        """시간대별 지연 시뮬레이션"""
        delays = {}

        # 시간대별 기본 지연
        base_delay = {
            'rush_hour': 25,
            'normal': 10,
            'late_night': 5,
            'weekend': 8
        }.get(time_category, 10)

        # 혼잡 구간 (2호선 강남~삼성)
        congested = [
            ('강남', '역삼', 2),
            ('역삼', '선릉', 2),
            ('선릉', '삼성', 2),
            ('신도림', '대림', 2),
        ]

        for segment in self.segment_info.keys():
            if segment in congested:
                delays[segment] = base_delay * 2
            else:
                delays[segment] = base_delay

        return delays

    def _collect_transfer_headways(
        self,
        route: RouteResult,
        hour: int,
        is_weekend: bool
    ) -> Dict[str, int]:
        """경로의 환승역별 배차간격 수집"""
        headways = {}

        for i in range(len(route.path) - 1):
            current = route.path[i]
            next_node = route.path[i + 1]

            if self.router.is_transfer_edge(current, next_node):
                station, _ = self.router.parse_node_id(current)
                _, next_line = self.router.parse_node_id(next_node)

                headway = self.get_real_headway(station, next_line, hour, is_weekend)
                headways[f"{station}_{next_line}호선"] = headway

        return headways

    def _find_alternatives(
        self,
        start: str,
        end: str,
        primary: RouteResult,
        time_category: str,
        max_count: int
    ) -> List[RouteResult]:
        """대안 경로 탐색"""
        alternatives = []
        start_lines = self.router.station_lines.get(start, [])
        end_lines = self.router.station_lines.get(end, [])

        for sl in start_lines:
            for el in end_lines:
                route = self.router.dijkstra_with_delays(
                    start, end, start_line=sl, end_line=el, time_category=time_category
                )
                if route.path and route.path != primary.path:
                    alternatives.append(route)

        alternatives.sort(key=lambda x: (x.transfer_count, x.total_time))

        seen = set()
        unique = []
        for r in alternatives:
            key = tuple(r.path)
            if key not in seen:
                seen.add(key)
                unique.append(r)
                if len(unique) >= max_count:
                    break

        return unique

    def _assess_risk_v2(
        self,
        route: RouteResult,
        headway_info: Dict[str, int],
        time_category: str
    ) -> Dict[str, Any]:
        """위험 평가 V2"""
        risk_level = "LOW"
        factors = []

        # 환승 실패 위험
        if route.transfer_failure_risk > 0.5:
            risk_level = "HIGH"
            factors.append(f"환승 실패 위험 {route.transfer_failure_risk:.0%}")
        elif route.transfer_failure_risk > 0.2:
            risk_level = "MEDIUM"
            factors.append(f"환승 실패 가능성 {route.transfer_failure_risk:.0%}")

        # 배차간격 체크
        for station_line, headway in headway_info.items():
            if headway > 480:  # 8분 이상
                factors.append(f"{station_line} 배차간격 {headway//60}분")

        # 시간대 경고
        if time_category == 'rush_hour' and route.transfer_count > 0:
            factors.append("러시아워 환승 혼잡")
        elif time_category == 'late_night':
            factors.append("심야 배차간격 증가")

        # 지연 체크
        if route.total_delay > 120:
            if risk_level == "LOW":
                risk_level = "MEDIUM"
            factors.append(f"예상 지연 {route.total_delay}초")

        return {
            'level': risk_level,
            'factors': factors,
            'transfer_failure_risk': route.transfer_failure_risk,
            'headway_info': headway_info
        }

    def _generate_reason(
        self,
        primary: RouteResult,
        alternatives: List[RouteResult],
        risk: Dict,
        headway_info: Dict
    ) -> str:
        """추천 이유 생성"""
        reasons = []

        if primary.transfer_count == 0:
            reasons.append("환승 없는 직행 경로")
        else:
            reasons.append(f"{primary.transfer_count}회 환승")

            # 배차간격 정보
            if headway_info:
                avg_headway = sum(headway_info.values()) // len(headway_info)
                reasons.append(f"평균 배차 {avg_headway//60}분")

        if alternatives:
            diff = alternatives[0].total_time - primary.total_time
            if diff > 0:
                reasons.append(f"대안보다 {diff}초 빠름")

        if risk['level'] == "LOW":
            reasons.append("안정적")
        elif risk['level'] == "HIGH":
            reasons.append("주의 필요")

        return " | ".join(reasons)

    def _print_result(
        self,
        primary: RouteResult,
        alternatives: List[RouteResult],
        risk: Dict,
        reason: str,
        headway_info: Dict
    ):
        """결과 출력"""
        print(f"\n[추천 경로]")
        print(f"  총 소요: {primary.total_time}초 ({primary.total_time//60}분 {primary.total_time%60}초)")
        print(f"  환승: {primary.transfer_count}회")
        print(f"  환승 실패 위험: {primary.transfer_failure_risk:.1%}")

        # 경로
        stations = []
        for node in primary.path:
            s, _ = self.router.parse_node_id(node)
            if not stations or stations[-1] != s:
                stations.append(s)
        print(f"  경로: {' → '.join(stations[:6])}{'...' if len(stations) > 6 else ''}")

        # 배차간격
        if headway_info:
            print(f"\n[환승역 배차간격]")
            for station_line, headway in headway_info.items():
                print(f"  {station_line}: {headway//60}분 {headway%60}초")

        # 위험 평가
        emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}
        print(f"\n[위험 평가] {emoji.get(risk['level'], '')} {risk['level']}")
        for f in risk['factors']:
            print(f"  - {f}")

        # 추천 이유
        print(f"\n[추천 이유] {reason}")

        # 대안
        if alternatives:
            print(f"\n[대안 경로]")
            for i, alt in enumerate(alternatives, 1):
                alt_stations = []
                for node in alt.path:
                    s, _ = self.router.parse_node_id(node)
                    if not alt_stations or alt_stations[-1] != s:
                        alt_stations.append(s)

                diff = alt.total_time - primary.total_time
                print(f"  {i}. {' → '.join(alt_stations[:4])}...")
                print(f"     {alt.total_time//60}분 (+{diff}초), 환승 {alt.transfer_count}회")


def demo():
    """데모 실행"""
    print("="*60)
    print("SUBPICK V2 데모 (ODsay API 연동)")
    print("="*60)

    router = SubpickRouterV2()

    test_cases = [
        ("강남", "홍대입구", "2024-01-15 08:30"),   # 출근
        ("서울역", "잠실", "2024-01-15 18:30"),     # 퇴근
        ("신도림", "왕십리", "2024-01-14 14:00"),   # 주말
    ]

    for start, end, time_str in test_cases:
        router.recommend_route(start, end, time_str)
        print("\n")


if __name__ == '__main__':
    demo()
