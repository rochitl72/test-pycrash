"""
Indian vehicle database for CrashRecon.

Specs are published manufacturer figures (dimensions, kerb weight, wheelbase);
CG height, axle split and yaw inertia are standard reconstruction estimates:
  izz ~= m/12 * (L^2 + W^2)   (rectangular slab approximation)
  lcgf = (1 - front_axle_weight_fraction) * wb
All stored metric; to_pycrash_input() converts to pycrash's US-customary units.
"""
from __future__ import annotations

M_TO_FT = 3.28084
KG_TO_LB = 2.20462
KGM2_TO_LBFTS2 = 0.737562
KMH_TO_MPH = 0.621371

# category -> rendering hints the frontend uses (2D + 3D)
BODY_STYLES = ["hatchback", "sedan", "suv", "truck", "mini_truck", "auto", "bike", "bus"]

INDIAN_VEHICLES: dict[str, dict] = {
    "maruti_swift": {
        "label": "Maruti Suzuki Swift", "make": "Maruti Suzuki", "model": "Swift", "year": 2022,
        "body": "hatchback", "length_m": 3.845, "width_m": 1.735, "wb_m": 2.450,
        "track_m": 1.530, "f_hang_m": 0.800, "mass_kg": 905, "hcg_m": 0.54,
        "front_axle_frac": 0.61, "tire_d_m": 0.62, "tire_w_m": 0.185,
        "drive": "fwd", "steer_ratio": 16.0, "color": "#e11d48",
    },
    "maruti_alto": {
        "label": "Maruti Suzuki Alto K10", "make": "Maruti Suzuki", "model": "Alto K10", "year": 2023,
        "body": "hatchback", "length_m": 3.530, "width_m": 1.490, "wb_m": 2.380,
        "track_m": 1.295, "f_hang_m": 0.690, "mass_kg": 755, "hcg_m": 0.52,
        "front_axle_frac": 0.62, "tire_d_m": 0.56, "tire_w_m": 0.145,
        "drive": "fwd", "steer_ratio": 16.0, "color": "#f59e0b",
    },
    "honda_city": {
        "label": "Honda City 1.5", "make": "Honda", "model": "City 1.5", "year": 2015,
        "body": "sedan", "length_m": 4.440, "width_m": 1.695, "wb_m": 2.600,
        "track_m": 1.495, "f_hang_m": 0.945, "mass_kg": 1107, "hcg_m": 0.55,
        "front_axle_frac": 0.60, "tire_d_m": 0.62, "tire_w_m": 0.175,
        "drive": "fwd", "steer_ratio": 16.0, "color": "#3b82f6",
    },
    "mahindra_scorpio": {
        "label": "Mahindra Scorpio-N", "make": "Mahindra", "model": "Scorpio-N", "year": 2023,
        "body": "suv", "length_m": 4.662, "width_m": 1.917, "wb_m": 2.750,
        "track_m": 1.640, "f_hang_m": 0.880, "mass_kg": 1850, "hcg_m": 0.72,
        "front_axle_frac": 0.52, "tire_d_m": 0.74, "tire_w_m": 0.255,
        "drive": "rwd", "steer_ratio": 17.0, "color": "#334155",
    },
    "tata_nexon": {
        "label": "Tata Nexon", "make": "Tata", "model": "Nexon", "year": 2023,
        "body": "suv", "length_m": 3.995, "width_m": 1.804, "wb_m": 2.498,
        "track_m": 1.540, "f_hang_m": 0.800, "mass_kg": 1250, "hcg_m": 0.62,
        "front_axle_frac": 0.59, "tire_d_m": 0.66, "tire_w_m": 0.215,
        "drive": "fwd", "steer_ratio": 16.5, "color": "#0ea5e9",
    },
    "tata_ace": {
        "label": "Tata Ace (mini truck)", "make": "Tata", "model": "Ace", "year": 2022,
        "body": "mini_truck", "length_m": 3.800, "width_m": 1.500, "wb_m": 2.100,
        "track_m": 1.270, "f_hang_m": 0.720, "mass_kg": 890, "hcg_m": 0.75,
        "front_axle_frac": 0.44, "tire_d_m": 0.58, "tire_w_m": 0.155,
        "drive": "rwd", "steer_ratio": 18.0, "color": "#eab308",
        "load_kg_typical": 750,
    },
    "tata_407": {
        "label": "Tata 407 (light truck)", "make": "Tata", "model": "407", "year": 2020,
        "body": "truck", "length_m": 4.998, "width_m": 1.860, "wb_m": 2.900,
        "track_m": 1.560, "f_hang_m": 1.000, "mass_kg": 2180, "hcg_m": 1.00,
        "front_axle_frac": 0.42, "tire_d_m": 0.76, "tire_w_m": 0.215,
        "drive": "rwd", "steer_ratio": 20.0, "color": "#16a34a",
        "load_kg_typical": 2250,
    },
    "eicher_1095": {
        "label": "Eicher Pro 1095XP (truck)", "make": "Eicher", "model": "1095XP", "year": 2017,
        "body": "truck", "length_m": 7.100, "width_m": 2.200, "wb_m": 4.760,
        "track_m": 1.840, "f_hang_m": 1.160, "mass_kg": 3990, "hcg_m": 1.15,
        "front_axle_frac": 0.40, "tire_d_m": 0.90, "tire_w_m": 0.255,
        "drive": "rwd", "steer_ratio": 20.0, "color": "#ef4444",
        "load_kg_typical": 6500,
    },
    "ashok_leyland_bus": {
        "label": "Ashok Leyland bus (state transport)", "make": "Ashok Leyland", "model": "Viking", "year": 2019,
        "body": "bus", "length_m": 10.300, "width_m": 2.440, "wb_m": 5.335,
        "track_m": 1.950, "f_hang_m": 1.900, "mass_kg": 8300, "hcg_m": 1.20,
        "front_axle_frac": 0.38, "tire_d_m": 0.96, "tire_w_m": 0.275,
        "drive": "rwd", "steer_ratio": 22.0, "color": "#dc2626",
        "load_kg_typical": 3500,
    },
    "bajaj_re_auto": {
        "label": "Bajaj RE auto-rickshaw", "make": "Bajaj", "model": "RE", "year": 2021,
        "body": "auto", "length_m": 2.635, "width_m": 1.300, "wb_m": 2.000,
        "track_m": 1.100, "f_hang_m": 0.350, "mass_kg": 395, "hcg_m": 0.55,
        "front_axle_frac": 0.35, "tire_d_m": 0.40, "tire_w_m": 0.100,
        "drive": "rwd", "steer_ratio": 10.0, "color": "#facc15",
        "load_kg_typical": 200,
        "note": "3-wheeler approximated as a narrow 4-wheel track in the planar model",
    },
    "hero_splendor": {
        "label": "Hero Splendor (motorcycle)", "make": "Hero", "model": "Splendor Plus", "year": 2022,
        "body": "bike", "length_m": 2.000, "width_m": 0.720, "wb_m": 1.236,
        "track_m": 0.300, "f_hang_m": 0.380, "mass_kg": 112, "hcg_m": 0.55,
        "front_axle_frac": 0.48, "tire_d_m": 0.58, "tire_w_m": 0.090,
        "drive": "rwd", "steer_ratio": 1.0, "color": "#111827",
        "load_kg_typical": 75,
        "note": "2-wheeler approximated as a narrow 4-wheel track; no lean dynamics",
    },
}


def vehicle_catalog() -> list[dict]:
    """Compact list for the frontend picker."""
    out = []
    for vid, v in INDIAN_VEHICLES.items():
        out.append({
            "id": vid, "label": v["label"], "body": v["body"], "color": v["color"],
            "length_m": v["length_m"], "width_m": v["width_m"], "mass_kg": v["mass_kg"],
            "load_kg_typical": v.get("load_kg_typical", 0), "note": v.get("note", ""),
        })
    return out


def to_pycrash_input(vehicle_id: str, *, speed_kmh: float = 0.0, heading_deg: float = 0.0,
                     x_m: float = 0.0, y_m: float = 0.0, loaded: bool = False,
                     omega_deg_s: float = 0.0) -> dict:
    """Build the full pycrash Vehicle input dict (US customary) for an Indian vehicle."""
    v = INDIAN_VEHICLES[vehicle_id]
    mass = v["mass_kg"] + (v.get("load_kg_typical", 0) if loaded else 0)
    L, W, wb = v["length_m"], v["width_m"], v["wb_m"]
    lcgf_m = (1.0 - v["front_axle_frac"]) * wb
    lcgr_m = wb - lcgf_m
    r_hang_m = L - v["f_hang_m"] - wb
    izz_kgm2 = mass / 12.0 * (L * L + W * W)
    drive = v["drive"]
    return {
        "year": v["year"], "vehmake": v["make"], "vehmodel": v["model"], "vin": "", "notes": v.get("note", ""),
        "weight": round(mass * KG_TO_LB, 1),
        "length": round(L * M_TO_FT, 3), "width": round(W * M_TO_FT, 3),
        "wb": round(wb * M_TO_FT, 3), "track": round(v["track_m"] * M_TO_FT, 3),
        "f_hang": round(v["f_hang_m"] * M_TO_FT, 3), "r_hang": round(max(r_hang_m, 0.1) * M_TO_FT, 3),
        "lcgf": round(lcgf_m * M_TO_FT, 3), "lcgr": round(lcgr_m * M_TO_FT, 3),
        "hcg": round(v["hcg_m"] * M_TO_FT, 3),
        "tire_d": round(v["tire_d_m"] * M_TO_FT, 3), "tire_w": round(v["tire_w_m"] * M_TO_FT, 3),
        "izz": round(izz_kgm2 * KGM2_TO_LBFTS2, 1),
        "steer_ratio": v["steer_ratio"], "brake": 0,
        "fwd": 1 if drive == "fwd" else 0, "rwd": 1 if drive == "rwd" else 0,
        "awd": 1 if drive == "awd" else 0,
        "A": 0, "B": 0, "k": 0, "L": 0, "c": 0, "gvwr": 0, "striking": 0, "isTrailer": False,
        "vx_initial": round(speed_kmh * KMH_TO_MPH, 2), "vy_initial": 0.0,
        "omega_z": omega_deg_s, "head_angle": heading_deg,
        "init_x_pos": round(x_m * M_TO_FT, 3), "init_y_pos": round(y_m * M_TO_FT, 3),
    }


# ---- Preset Indian accident scenes (PoC) ----
# Coordinates: metres, x east, y north. Headings: deg CCW from +x.
SCENES: dict[str, dict] = {
    "nh_rear_end": {
        "label": "NH-44 rear-end: car into slow truck",
        "environment": "highway",
        "description": ("Divided national highway, dry asphalt (mu 0.80). A Honda City doing 85 km/h "
                        "comes up on an Eicher truck crawling at 10 km/h in the left lane and brakes "
                        "0.9 s before contact."),
        "mu_ground": 0.8, "cor": 0.2, "vehicle_mu": 0.45,
        "vehicles": [
            {"vehicle_id": "honda_city", "speed_kmh": 85, "heading_deg": 0, "x_m": 0, "y_m": 0,
             "brake": {"t_start": 1.2, "t_full": 1.5, "level": 0.85}},
            {"vehicle_id": "eicher_1095", "speed_kmh": 10, "heading_deg": 0, "x_m": 38, "y_m": 0.3,
             "loaded": True},
        ],
    },
    "rural_t_bone": {
        "label": "Rural T-bone: SUV crosses in front of car",
        "environment": "rural_intersection",
        "description": ("Unsignalised rural crossroads, dry but dusty asphalt (mu 0.70). A Maruti Swift "
                        "at 55 km/h reaches the junction as a Mahindra Scorpio crosses from its left at "
                        "25 km/h; the Swift brakes 0.6 s before impact."),
        "mu_ground": 0.7, "cor": 0.15, "vehicle_mu": 0.5,
        "vehicles": [
            {"vehicle_id": "maruti_swift", "speed_kmh": 55, "heading_deg": 0, "x_m": 0, "y_m": 0,
             "brake": {"t_start": 1.0, "t_full": 1.25, "level": 0.9}},
            {"vehicle_id": "mahindra_scorpio", "speed_kmh": 25, "heading_deg": 90, "x_m": 16.5, "y_m": -9},
        ],
    },
    "auto_sideswipe": {
        "label": "Urban: car clips auto-rickshaw",
        "environment": "urban",
        "description": ("Four-lane city road (mu 0.75). A Tata Nexon at 45 km/h clips a Bajaj RE "
                        "auto-rickshaw pulling out at 15 km/h."),
        "mu_ground": 0.75, "cor": 0.2, "vehicle_mu": 0.5,
        "vehicles": [
            {"vehicle_id": "tata_nexon", "speed_kmh": 45, "heading_deg": 0, "x_m": 0, "y_m": 0},
            {"vehicle_id": "bajaj_re_auto", "speed_kmh": 15, "heading_deg": 25, "x_m": 14, "y_m": -1.8},
        ],
    },
}


def build_time_inputs_for(spec: dict, end_time: float = 5.0) -> dict:
    """Driver inputs for a scenario vehicle: optional ramped braking, no steer."""
    br = spec.get("brake")
    if not br:
        return {"time": [0.0, end_time], "throttle": [0, 0], "brake": [0, 0], "steer": [0, 0]}
    t0, t1, lvl = br["t_start"], br["t_full"], br["level"]
    t = [0.0, round(t0, 3), round(t1, 3), end_time]
    return {"time": t, "throttle": [0, 0, 0, 0], "brake": [0, 0, lvl, lvl], "steer": [0, 0, 0, 0]}
