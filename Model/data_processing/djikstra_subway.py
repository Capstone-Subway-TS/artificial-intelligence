import heapq

def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    queue = []
    heapq.heappush(queue, [distances[start], start])

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == end:
            break

        if distances[current_node] < current_distance:
            continue

        for adjacent, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[adjacent]:
                distances[adjacent] = distance
                previous_nodes[adjacent] = current_node
                heapq.heappush(queue, [distance, adjacent])

    path = []
    step = end
    while step is not None:
        path.append(step)
        step = previous_nodes[step]
    path.reverse()

    return path

# 지하철 노선도 그래프
subway_graph = {
    '서울역': {'시청': 1, '남영': 1},
    '시청': {'서울역': 1, '종각': 1},
    '종각': {'시청': 1, '종로3가': 1},
    '종로3가': {'종각': 1, '을지로3가': 1, '안국': 1},
    '을지로입구': {'을지로3가': 1, '시청': 1},
    '을지로3가': {'을지로4가': 1, '을지로입구': 1, '종로3가': 1},
    '을지로4가': {'동대문역사문화공원': 1, '을지로3가': 1},
    '동대문역사문화공원': {'동대문': 1, '을지로4가': 1, '충무로': 1},
    '동대문': {'동대문역사문화공원': 1, '신설동': 1},
    '신설동': {'동대문': 1, '제기동': 1},
    '제기동': {'신설동': 1, '청량리': 1},
    '청량리': {'제기동': 1},
    '안국': {'종로3가': 1},
    '충무로': {'동대문역사문화공원': 1, '명동': 1},
    '명동': {'충무로': 1, '회현': 1},
    '회현': {'명동': 1, '서울역': 1},
    '남영': {'서울역': 1, '용산': 1},
    '용산': {'남영': 1},
    # 추가적으로 2~8호선의 주요 역들도 이런 식으로 포함될 수 있습니다.
}

# '시청'에서 '동대문역사문화공원'까지의 최단 경로
path = dijkstra(subway_graph, '시청', '동대문역사문화공원')
print("최단 경로:", path)
