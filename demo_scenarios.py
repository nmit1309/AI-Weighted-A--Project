"""2 kịch bản demo cố định. Chạy: python demo_scenarios.py"""
from weighted_astar import *

def show(title, r):
    print(f"  {title:<22} found={r['found']} cost={r['cost']} nodes={r['nodes_expanded']} path_len={len(r['path'])}")

print("KỊCH BẢN 1 - Xe máy gặp tường ngập, ô tô đi thẳng")
g = build_test_graph()
show("Xe máy (20mm)", weighted_a_star(g,(4,0),(4,4),"motorbike"))
show("Ô tô (50mm)",   weighted_a_star(g,(4,0),(4,4),"car"))

print("KỊCH BẢN 2 - Kéo slider thời gian (nước dâng phút 5, rút phút 60)")
g = make_grid(5); g.set_flood((4,1),(4,2),[(0,0),(5,30),(60,0)])
for t in (0,10,70):
    show(f"Xuất phát t={t}", weighted_a_star(g,(4,0),(4,4),"motorbike",start_time=t))

print("SO SÁNH A* (w=1) vs Weighted A*")
for name,gr,s,e in (("Tường ngập 5x5",build_test_graph(),(4,0),(4,4)),
                    ("Lưới 10x10 khô",make_grid(10),(0,0),(9,9))):
    for w in (1,1.5,2,3):
        r = weighted_a_star(gr,s,e,"car" if "khô" in name else "motorbike",w=w)
        print(f"  {name:<16} w={w:<4} nodes={r['nodes_expanded']:<3} cost={r['cost']}")
