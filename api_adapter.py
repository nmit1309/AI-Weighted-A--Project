"""Lớp mỏng giữa backend (JSON request) và thuật toán. Sửa tên field cho khớp API thật."""
from weighted_astar import weighted_a_star, compare, build_test_graph

def solve_request(payload, graph=None):
    g = graph or build_test_graph()
    kw = dict(vehicle=payload["vehicle"], w=payload.get("w", 1.0),
              start_time=payload.get("start_time", 0.0),
              mode=payload.get("mode", "tradeoff"))
    s, e = tuple(payload["start"]), tuple(payload["goal"])
    if payload.get("compare"):
        r = compare(g, s, e, payload["vehicle"], payload.get("w", 2.0),
                    start_time=kw["start_time"], mode=kw["mode"])
        for k in ("astar", "weighted"):
            r[k].pop("path_raw", None)
        return r
    r = weighted_a_star(g, s, e, **kw)
    r.pop("path_raw", None)
    return r
