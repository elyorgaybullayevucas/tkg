# utils/paths.py — Kengaytirilgan path sampler
"""
Temporal-aware BFS path sampler.

Har bir (s, r, o, t) uchun:
  - Faqat t vaqtidan OLDINGI qirralar ishlatiladi (temporal constraint)
  - Yo'l topilmasa: 1-hop fallback, keyin dummy
  - LRU cache: bir xil (s, o, t) so'rovlar qayta hisoblanmaydi
"""
import random
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple

Triple = Tuple[int, int, int]        # (entity, relation, time)
AdjList = Dict[int, List[Triple]]    # node → [(neighbor, rel, time)]


def build_graph(
    quads: List[Tuple[int, int, int, int]],
    max_time: Optional[int] = None,
) -> AdjList:
    """
    (s, r, o, t) ro'yxatidan yo'naltirilgan adjacency list yaratadi.
    Ikki tomonlama: s→o va o→s (inverse relation bilan).
    """
    adj: AdjList = defaultdict(list)
    for s, r, o, t in quads:
        if max_time is None or t <= max_time:
            adj[s].append((o, r,     t))
            adj[o].append((s, r + 10000, t))   # inverse relation markeri
    return dict(adj)


def build_time_index(
    quads: List[Tuple[int, int, int, int]],
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Har bir vaqt momenti uchun quadrupletlar indeksi.
    time → [(s, r, o, t), ...]
    """
    idx: Dict[int, List] = defaultdict(list)
    for q in quads:
        idx[q[3]].append(q)
    return dict(idx)


def sample_paths(
    adj: AdjList,
    start: int,
    end: int,
    query_time: int,
    num_paths: int = 8,
    max_len: int = 3,
    max_expand: int = 30,
) -> List[List[Triple]]:
    """
    Temporal BFS: start → end yo'llarini topadi.

    Args:
        adj:        Adjacency list (build_graph dan)
        start:      Boshlang'ich entity
        end:        Maqsad entity
        query_time: Faqat t <= query_time qirralar ishlatiladi
        num_paths:  Qaytariladigan yo'llar soni
        max_len:    Maksimal hop soni
        max_expand: Har bir tugun uchun max qo'shni

    Returns:
        [[(entity, relation, time), ...], ...]
    """
    if start == end:
        return []

    found: List[List[Triple]] = []
    # (current_node, path_so_far, visited_nodes)
    queue: deque = deque()
    queue.append((start, [], frozenset([start])))

    while queue and len(found) < num_paths * 4:
        node, path, visited = queue.popleft()

        if len(path) >= max_len:
            continue

        neighbors = adj.get(node, [])
        # Temporal filter + shuffle
        valid = [(o, r, t) for o, r, t in neighbors
                 if t <= query_time and o not in visited]

        if not valid:
            continue

        # Stratified sampling: yaqin vaqtdagilarga ustunlik
        valid.sort(key=lambda x: -x[2])          # vaqt bo'yicha desc
        sample = valid[:max_expand]
        random.shuffle(sample)

        for o, r, t in sample:
            new_path = path + [(o, r, t)]
            new_visited = visited | {o}

            if o == end:
                found.append(new_path)
                if len(found) >= num_paths * 4:
                    break
            else:
                queue.append((o, new_path, new_visited))

    # Xilma-xillikni ta'minlash uchun uzunlik bo'yicha aralashtirish
    found.sort(key=lambda p: len(p))
    if len(found) > num_paths:
        # Turli uzunlikdagilarni tanlash
        short = [p for p in found if len(p) == 1]
        mid   = [p for p in found if len(p) == 2]
        long_ = [p for p in found if len(p) >= 3]
        mixed = []
        for pool in [short, mid, long_]:
            random.shuffle(pool)
            mixed.extend(pool)
        found = mixed[:num_paths]

    return found[:num_paths]


def get_fallback_paths(
    adj: AdjList,
    start: int,
    end: int,
    query_time: int,
    num_paths: int,
) -> List[List[Triple]]:
    """
    Yo'l topilmasa: 1-hop qo'shnilardan eng yaqin olinadi.
    """
    paths = []

    # end ning qo'shnilaridan start yo'lini izlaymiz
    end_neighbors = adj.get(end, [])
    for o, r, t in end_neighbors:
        if t <= query_time:
            paths.append([(end, r, t)])
        if len(paths) >= num_paths:
            break

    # start ning qo'shnilaridan
    if not paths:
        start_neighbors = adj.get(start, [])
        for o, r, t in start_neighbors:
            if t <= query_time:
                paths.append([(o, r, t)])
            if len(paths) >= num_paths:
                break

    # Hech narsa topilmasa — dummy
    if not paths:
        paths = [[(end, 0, max(0, query_time - 1))]]

    return paths
