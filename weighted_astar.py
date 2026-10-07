"""Weighted A* time-dependent, tránh ngập theo loại xe.
f(n) = g(n) + w * h(n)   (w >= 1; w = 1 là A* thường)
"""
import heapq, math, itertools
from dataclasses import dataclass, field

INF = float("inf")

VEHICLES = {                      # H_max (mm) - số liệu mô phỏng
    "motorbike": {"name": "Xe máy", "H_max": 20},
    "vf3":       {"name": "VF3",    "H_max": 30},
    "car":       {"name": "Ô tô",   "H_max": 50},
}
SPEED_KMH = 20.0                  # tốc độ nền (giả định)
ALPHA = 3.0                       # hệ số phạt ngập
K = 2                             # số mũ phạt


# ---------------------------------------------------------------- Graph
@dataclass
class Graph:
    coords: dict                                   # node -> (x, y) mét
    adj: dict = field(default_factory=dict)        # node -> {nbr: distance}
    flood: dict = field(default_factory=dict)      # (u,v) -> [(t_phút, H_mm), ...]

    def add_edge(self, u, v, dist=None):
        if dist is None:
            (x1, y1), (x2, y2) = self.coords[u], self.coords[v]
            dist = math.hypot(x1 - x2, y1 - y2)
        self.adj.setdefault(u, {})[v] = dist
        self.adj.setdefault(v, {})[u] = dist

    def set_flood(self, u, v, profile):
        """profile = [(t0,H0),(t1,H1),...] hằng số từng đoạn; 2 chiều."""
        p = sorted(profile)
        self.flood[(u, v)] = p
        self.flood[(v, u)] = p

    def water_level(self, u, v, t):
        h = 0.0
        for t0, h0 in self.flood.get((u, v), []):
            if t >= t0:
                h = h0
        return h


def make_grid(n=5, spacing=500.0):
    g = Graph(coords={(r, c): (c * spacing, r * spacing)
                      for r in range(n) for c in range(n)})
    for r in range(n):
        for c in range(n):
            if c + 1 < n: g.add_edge((r, c), (r, c + 1))
            if r + 1 < n: g.add_edge((r, c), (r + 1, c))
    return g


# ---------------------------------------------------------------- Cost
def flood_penalty(H, H_max):
    """1 + alpha * r^k ; ngập vượt H_max -> INF."""
    if H > H_max:
        return INF
    return 1 + ALPHA * (H / H_max) ** K


def edge_cost(dist, H, vehicle, mode="tradeoff", H_safe=0):
    if mode == "safe" and H > H_safe:
        return INF
    return dist * flood_penalty(H, vehicle["H_max"])


def travel_minutes(dist, H, H_max):
    """Đi chậm lại khi nước sâu: v = v0 * (1 - 0.5 r)."""
    r = min(H / H_max, 1.0)
    v = SPEED_KMH * 1000 / 60 * (1 - 0.5 * r)      # m/phút
    return dist / v


def heuristic(g, n, goal):
    (x1, y1), (x2, y2) = g.coords[n], g.coords[goal]
    return math.hypot(x1 - x2, y1 - y2)            # admissible vì penalty >= 1


# ---------------------------------------------------------------- Search
def weighted_a_star(g, start, goal, vehicle="motorbike", w=1.0,
                    start_time=0.0, mode="tradeoff", H_safe=0,
                    record_steps=True, top_k=4):
    veh = VEHICLES[vehicle]
    tie = itertools.count()
    G = {start: 0.0}
    T = {start: start_time}
    parent = {start: None}
    h0 = heuristic(g, start, goal)
    open_heap = [(w * h0, h0, next(tie), start)]
    closed, steps, expanded = set(), [], 0

    def frontier_view():
        best = {}
        for f, h, _, n in open_heap:
            if n not in closed and (n not in best or f < best[n][0]):
                best[n] = (f, h)
        rows = sorted(((f, h, n) for n, (f, h) in best.items()), key=lambda x: x[0])
        return [{"node": str(n), "g": round(G[n], 1), "h": round(h, 1),
                 "f": round(f, 1)} for f, h, n in rows]

    while open_heap:
        f, h, _, cur = heapq.heappop(open_heap)
        if cur in closed or f > G[cur] + w * heuristic(g, cur, goal) + 1e-9:
            continue                                # bản ghi cũ
        fr = frontier_view()                        # các ứng viên còn lại
        closed.add(cur)
        expanded += 1
        if record_steps:
            others = fr[:top_k]
            reason = (f"Chọn {cur}: f={f:.1f} nhỏ nhất frontier."
                      if not others else
                      f"Chọn {cur}: f={f:.1f} nhỏ hơn ứng viên tốt nhì "
                      f"{others[0]['node']} (f={others[0]['f']}) "
                      f"(chênh {others[0]['f'] - f:.1f}).")
            steps.append({"step": expanded, "node": str(cur),
                          "g": round(G[cur], 1), "h": round(h, 1),
                          "w": w, "f": round(f, 1),
                          "t_arrive_min": round(T[cur], 2),
                          "frontier": others, "reason": reason,
                          "blocked": []})
        if cur == goal:
            path, n = [], cur
            while n is not None:
                path.append(n); n = parent[n]
            path.reverse()
            return _result(True, path, G[goal], T[goal], expanded, steps, w, vehicle, mode)

        for nb, dist in g.adj[cur].items():
            if nb in closed:
                continue
            Hn = g.water_level(cur, nb, T[cur])     # mức nước TẠI thời điểm tới cur
            c = edge_cost(dist, Hn, veh, mode, H_safe)
            if c == INF:
                if record_steps:
                    steps[-1]["blocked"].append(
                        {"to": str(nb), "H": Hn,
                         "why": f"H={Hn}mm > {'H_safe' if mode=='safe' else 'H_max'}"
                                f"={H_safe if mode=='safe' else veh['H_max']}mm"})
                continue
            ng = G[cur] + c
            if ng < G.get(nb, INF):
                G[nb] = ng
                T[nb] = T[cur] + travel_minutes(dist, Hn, veh["H_max"])
                parent[nb] = cur
                hn = heuristic(g, nb, goal)
                heapq.heappush(open_heap, (ng + w * hn, hn, next(tie), nb))
    return _result(False, [], INF, None, expanded, steps, w, vehicle, mode)


def _result(found, path, cost, t_end, expanded, steps, w, vehicle, mode):
    return {"found": found, "path": [str(p) for p in path], "path_raw": path,
            "cost": None if not found else round(cost, 2),
            "arrival_min": None if t_end is None else round(t_end, 2),
            "nodes_expanded": expanded, "w": w, "vehicle": vehicle,
            "mode": mode, "steps": steps}


def compare(g, start, goal, vehicle, w_fast=2.0, **kw):
    a = weighted_a_star(g, start, goal, vehicle, w=1.0, **kw)
    b = weighted_a_star(g, start, goal, vehicle, w=w_fast, **kw)
    saved = a["nodes_expanded"] - b["nodes_expanded"]
    return {"astar": a, "weighted": b, "nodes_saved": saved,
            "nodes_saved_pct": round(100 * saved / max(a["nodes_expanded"], 1), 1),
            "cost_increase_pct": None if not (a["found"] and b["found"]) else
                round(100 * (b["cost"] - a["cost"]) / a["cost"], 1)}


# ---------------------------------------------------------------- Test data
def build_test_graph():
    """Lưới 5x5, 500 m. Bức tường ngập giữa cột 1|2, hàng 1..4; chỉ hở hàng 0."""
    g = make_grid(5)
    for r in range(1, 5):
        g.set_flood((r, 1), (r, 2), [(0, 30)])
    return g
