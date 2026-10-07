"""PASS toàn bộ mới được push.  Chạy: python integration_test.py   (hoặc pytest -q)"""
import json, sys
from weighted_astar import *
from api_adapter import solve_request

S, G_ = (4, 0), (4, 4)
WALL_EDGE = ((4, 1), (4, 2))

def edges(path):
    return list(zip(path, path[1:]))

def uses_edge(path, e):
    return e in edges(path) or (e[1], e[0]) in edges(path)

def test_TC1_no_flood_equals_shortest():
    g = make_grid(5)
    r = weighted_a_star(g, S, G_, "motorbike", w=1)
    assert r["found"] and r["cost"] == 2000.0            # 4 cạnh x 500 m
    assert r["path_raw"] == [(4,0),(4,1),(4,2),(4,3),(4,4)]

def test_TC2_flood_beyond_hmax_forces_detour():
    g = build_test_graph()                               # H=30 > 20 (xe máy)
    r = weighted_a_star(g, S, G_, "motorbike")
    assert r["found"] and r["cost"] == 6000.0            # 12 cạnh: lên hàng 0 rồi xuống
    assert not any(uses_edge(r["path_raw"], (( rr,1),(rr,2))) for rr in range(1,5))
    assert any(b["H"] == 30 for st in r["steps"] for b in st["blocked"])

def test_TC3_vehicle_and_mode_tradeoff():
    g = build_test_graph()
    car = weighted_a_star(g, S, G_, "car")               # 30 <= 50 -> đi thẳng
    assert car["cost"] == 2540.0 and uses_edge(car["path_raw"], WALL_EDGE)
    g2 = make_grid(5)                                    # ngập nhẹ 15 mm mọi hàng
    for rr in range(1, 5): g2.set_flood((rr,1),(rr,2),[(0,15)])
    trade = weighted_a_star(g2, S, G_, "motorbike", mode="tradeoff")
    safe  = weighted_a_star(g2, S, G_, "motorbike", mode="safe")
    assert uses_edge(trade["path_raw"], WALL_EDGE) and trade["cost"] < 6000   # chấp nhận ngập nhẹ
    assert not uses_edge(safe["path_raw"], WALL_EDGE) and safe["cost"] == 6000.0

def test_TC4_no_route():
    g = make_grid(5)
    for rr in range(5): g.set_flood((rr,1),(rr,2),[(0,30)])
    assert weighted_a_star(g, S, G_, "motorbike")["found"] is False
    assert weighted_a_star(g, S, G_, "car")["found"] is True

def test_TC5_time_dependent():
    g = make_grid(5)
    g.set_flood(*WALL_EDGE, [(0, 0), (5, 30), (60, 0)])  # nước dâng phút 5, rút phút 60
    early = weighted_a_star(g, S, G_, "motorbike", start_time=0)
    mid   = weighted_a_star(g, S, G_, "motorbike", start_time=10)
    late  = weighted_a_star(g, S, G_, "motorbike", start_time=70)
    assert uses_edge(early["path_raw"], WALL_EDGE) and early["cost"] == 2000.0
    assert not uses_edge(mid["path_raw"], WALL_EDGE) and mid["cost"] == 3000.0
    assert uses_edge(late["path_raw"], WALL_EDGE) and late["cost"] == 2000.0

def test_TC6_weighted_faster_but_bounded():
    g = build_test_graph()
    c = compare(g, S, G_, "motorbike", w_fast=2.0)
    a, b = c["astar"], c["weighted"]
    assert b["nodes_expanded"] <= a["nodes_expanded"]
    assert b["cost"] <= 2.0 * a["cost"] + 1e-6           # chặn dưới-tối-ưu w
    big = make_grid(10)                                  # lưới rộng: khác biệt rõ
    ca = weighted_a_star(big, (0,0), (9,9), "car", w=1)
    cb = weighted_a_star(big, (0,0), (9,9), "car", w=2)
    assert cb["nodes_expanded"] < ca["nodes_expanded"] and cb["cost"] <= 2 * ca["cost"]

def test_response_schema_has_steps():
    r = solve_request({"start": S, "goal": G_, "vehicle": "motorbike", "w": 1.5})
    json.dumps(r)                                        # phải serialize được
    st = r["steps"][0]
    for k in ("step","node","g","h","f","w","frontier","reason","blocked"):
        assert k in st, k
    assert abs(st["f"] - (st["g"] + st["w"] * st["h"])) < 0.2

if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    fails = 0
    for n, f in tests:
        try: f(); print("PASS", n)
        except Exception as e: fails += 1; print("FAIL", n, "->", repr(e))
    print(f"\n{len(tests)-fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
