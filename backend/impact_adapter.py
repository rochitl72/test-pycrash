"""
Geometric collision detection + pycrash Impact (IMPC) setup.

pycrash's SingleMotion simulates each vehicle alone, so two bodies can drive
straight through each other. We use the body polygons it already produces
(p_gx / p_gy) to find the first frame where the two bodies overlap, work out
which vehicle is striking, where it touches (impact point on the striking
vehicle, impact edge on the struck vehicle) and the contact-plane angle, and
then hand all of that to pycrash's Impact model so momentum is actually
exchanged.

pycrash conventions (from pycrash/model_calcs/collision_plane.py):
  vehicle frame: x forward, y = +90 deg from x (same rotation sense as theta)
  impact_points  : [(x_ft, y_ft, normal_plane_angle_deg)] on the striking vehicle
  edgeimpact     : [0 front | 1 right (y=+w/2) | 2 rear | 3 left (y=-w/2)] on the struck
  edgeimpact_points: (x1, y1, x2, y2) of that edge in the struck vehicle frame
"""
from __future__ import annotations

import math
from typing import Optional

BODY_COLS = ["b_lfc", "b_rfc", "b_rrc", "b_lrc"]
EDGE_NAMES = {0: "front", 1: "right side", 2: "rear", 3: "left side"}


def _poly_at(veh, i):
    """Body polygon [(x, y) x4] of a simulated vehicle at frame i (global ft)."""
    return [(float(veh.p_gx.iloc[i][c]), float(veh.p_gy.iloc[i][c])) for c in BODY_COLS]


def _axes(poly):
    out = []
    for k in range(len(poly)):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % len(poly)]
        out.append((-(y2 - y1), x2 - x1))
    return out


def _project(poly, axis):
    d = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
    return min(d), max(d)


def polygons_overlap(a, b) -> bool:
    """Separating-axis test for two convex polygons."""
    for axis in _axes(a) + _axes(b):
        amin, amax = _project(a, axis)
        bmin, bmax = _project(b, axis)
        if amax < bmin or bmax < amin:
            return False
    return True


def _point_in_convex(pt, poly) -> bool:
    sign = 0
    for k in range(len(poly)):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % len(poly)]
        cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        if cross != 0:
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def _to_local(veh, i, gx, gy):
    """Global point -> vehicle frame at frame i (x forward, y +90 deg from x)."""
    cx = float(veh.model.Dx.iloc[i])
    cy = float(veh.model.Dy.iloc[i])
    th = float(veh.model.theta_rad.iloc[i])
    dx, dy = gx - cx, gy - cy
    return dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)


def _extent(pi):
    """front_x, rear_x, half_width in the vehicle frame (ft)."""
    front = pi["lcgf"] + pi["f_hang"]
    rear = -(pi["lcgr"] + pi["r_hang"])
    return front, rear, pi["width"] / 2.0


def _edge_points(pi, edge):
    front, rear, hw = _extent(pi)
    return {
        0: (front, -hw, front, hw),
        1: (front, hw, rear, hw),
        2: (rear, hw, rear, -hw),
        3: (rear, -hw, front, -hw),
    }[edge]


def _norm_angle(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a


def find_first_contact(sims, pis) -> Optional[dict]:
    """
    sims: [Vehicle, Vehicle] after SingleMotion; pis: their pycrash input dicts.
    Returns None if the bodies never overlap, else a dict describing the contact.
    """
    a, b = sims
    n = min(len(a.p_gx), len(b.p_gx), len(a.model), len(b.model))
    hit = None
    for i in range(n):
        try:
            if polygons_overlap(_poly_at(a, i), _poly_at(b, i)):
                hit = i
                break
        except Exception:
            continue
    if hit is None:
        # closest approach, for a useful message
        return None
    i = hit
    pa, pb = _poly_at(a, i), _poly_at(b, i)
    inside_a_in_b = [p for p in pa if _point_in_convex(p, pb)]  # corners of A inside B
    inside_b_in_a = [p for p in pb if _point_in_convex(p, pa)]

    # Striking vehicle = the one whose own velocity points toward the other's CG
    # (a car rear-ending a slow truck is striking even if the truck's corner
    # penetrates the car's front edge first, geometrically).
    def toward_score(v, other):
        ux = float(other.model.Dx.iloc[i]) - float(v.model.Dx.iloc[i])
        uy = float(other.model.Dy.iloc[i]) - float(v.model.Dy.iloc[i])
        n = math.hypot(ux, uy) or 1.0
        return (float(v.model.Vx.iloc[i]) * ux + float(v.model.Vy.iloc[i]) * uy) / n

    s_idx = 0 if toward_score(a, b) >= toward_score(b, a) else 1
    t_idx = 1 - s_idx
    veh_s, veh_t = (a, b)[s_idx], (a, b)[t_idx]
    pi_s, pi_t = pis[s_idx], pis[t_idx]
    poly_s, poly_t = (pa, pb)[s_idx], (pa, pb)[t_idx]
    corners = (inside_a_in_b, inside_b_in_a)[s_idx]
    if not corners:
        # edge-on-edge overlap: use the striking corner nearest the struck CG
        cgx, cgy = float(veh_t.model.Dx.iloc[i]), float(veh_t.model.Dy.iloc[i])
        corners = [min(poly_s, key=lambda p: (p[0] - cgx) ** 2 + (p[1] - cgy) ** 2)]

    # deepest corner of the striking body inside the struck body (struck frame)
    f_t, r_t, hw_t = _extent(pi_t)

    def depth(p):
        lx, ly = _to_local(veh_t, i, p[0], p[1])
        return min(f_t - lx, lx - r_t, hw_t - ly, ly + hw_t)

    corner = max(corners, key=depth)
    lx_t, ly_t = _to_local(veh_t, i, corner[0], corner[1])
    # entry edge = the struck edge the point is closest to
    d = {0: f_t - lx_t, 1: hw_t - ly_t, 2: lx_t - r_t, 3: ly_t + hw_t}
    edge = min(d, key=lambda k: abs(d[k]))

    px, py = _to_local(veh_s, i, corner[0], corner[1])
    th_s = math.degrees(float(veh_s.model.theta_rad.iloc[i]))
    th_t = math.degrees(float(veh_t.model.theta_rad.iloc[i]))
    rel = th_t - th_s
    # normal of the contact plane = struck vehicle's x axis (front/rear edge)
    # or y axis (side edge), expressed in the striking vehicle's frame.
    angle = _norm_angle(-rel if edge in (0, 2) else 90.0 - rel)

    t = float(a.model.t.iloc[i]) if "t" in a.model.columns else i * 0.01
    return {
        "frame": i,
        "t": round(t, 3),
        "striking_index": s_idx,
        "struck_index": t_idx,
        "impact_point": (round(px, 4), round(py, 4), round(angle, 3)),
        "edge": edge,
        "edge_name": EDGE_NAMES[edge],
        "edge_points": tuple(round(v, 4) for v in _edge_points(pi_t, edge)),
        "global_point": (round(corner[0], 3), round(corner[1], 3)),
        "heading_striking_deg": round(th_s, 2),
        "heading_struck_deg": round(th_t, 2),
    }


def extend_time_inputs(ti: dict, end_time: float) -> dict:
    """Pad driver inputs with their last value so both vehicles share one end time."""
    out = {k: list(v) for k, v in ti.items()}
    if out["time"][-1] < end_time:
        out["time"].append(end_time)
        for k in ("throttle", "brake", "steer"):
            out[k].append(out[k][-1])
    return out


def run_impact(pis, tis, contact, end_time, cor=0.1, mu=0.45):
    """Build fresh Vehicles, attach impact data and run pycrash Impact (IMPC)."""
    from pycrash import Vehicle, Impact

    s_idx, t_idx = contact["striking_index"], contact["struck_index"]
    vehicles = []
    for k, (pi, ti) in enumerate(zip(pis, tis)):
        p = dict(pi)
        p["striking"] = 1 if k == s_idx else 0
        v = Vehicle(f"{p.get('vehmake', 'V')}{k + 1}", p)
        ti2 = extend_time_inputs(ti, end_time)
        v.time_inputs(ti2["time"], ti2["throttle"], ti2["brake"], ti2["steer"], False)
        if k == s_idx:
            v.impact_points = [contact["impact_point"]]
        else:
            v.edgeimpact = [contact["edge"]]
            v.edgeimpact_points = [contact["edge_points"]]
        vehicles.append(v)

    imp = Impact(
        "crashrecon_impact", end_time, "IMPC", vehicles,
        impact_order=[[s_idx, t_idx]],
        impc_inputs={0: {"vehicle_mu": mu, "cor": cor}},
    )
    imp.simulate(show_results=False)
    return imp
