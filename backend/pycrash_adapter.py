"""
Converts parsed PC-Crash scene (metric) → pycrash Vehicle input dicts (US customary).
Identifies gaps and provides defaults with source tracking.

pycrash units: ft, lb, mph, ft/s, lb-ft-s², lb/in, in, deg, deg/s
PC-Crash units: m, kg, km/h, m/s, kgm², N/m, deg, rad/s
"""
from typing import Any
import math

# ─── Conversion factors ───
M_TO_FT = 3.28084
KG_TO_LB = 2.20462
KMH_TO_MPH = 0.621371
KGM2_TO_LBFTS2 = 0.737562  # kg·m² → lb·ft·s²
RADS_TO_DEGS = 57.2958
M_TO_IN = 39.3701

# ─── Default values for fields not in PC-Crash export ───
DEFAULTS = {
    "tire_d": {"value": 2.13, "unit": "ft", "source": "default", "note": "~26 in, typical sedan"},
    "tire_w": {"value": 0.59, "unit": "ft", "source": "default", "note": "~185mm, typical sedan"},
    "steer_ratio": {"value": 15.0, "unit": "ratio", "source": "default", "note": "typical power steering"},
    "brake": {"value": 0.0, "unit": "fraction", "source": "default", "note": "no static braking"},
    "gvwr": {"value": 0.0, "unit": "lb", "source": "default", "note": "not required for simulation"},
    "vin": {"value": "", "unit": "text", "source": "default", "note": "not in PC-Crash export"},
    "A": {"value": 0.0, "unit": "lb/in", "source": "not_available", "note": "crush stiffness — not in file, SDOF model cannot run"},
    "B": {"value": 0.0, "unit": "lb/in/in", "source": "not_available", "note": "crush slope — not in file"},
    "k": {"value": 0.0, "unit": "lb/in", "source": "not_available", "note": "effective spring stiffness — not in file"},
    "L": {"value": 0.0, "unit": "in", "source": "not_available", "note": "damage length — not in file"},
    "c": {"value": 0.0, "unit": "in", "source": "not_available", "note": "crush depth — not in file"},
    "fwd": {"value": True, "unit": "bool", "source": "default", "note": "assumed FWD, user should verify"},
    "rwd": {"value": False, "unit": "bool", "source": "default"},
    "awd": {"value": False, "unit": "bool", "source": "default"},
    "striking": {"value": True, "unit": "bool", "source": "user_required", "note": "user must select which vehicle struck"},
    "isTrailer": {"value": False, "unit": "bool", "source": "default"},
    "notes": {"value": "", "unit": "text", "source": "default"},
}

# Truck-specific overrides
TRUCK_DEFAULTS = {
    "tire_d": {"value": 3.28, "unit": "ft", "source": "default", "note": "~1m, typical truck tire"},
    "tire_w": {"value": 0.82, "unit": "ft", "source": "default", "note": "~250mm, truck tire"},
    "steer_ratio": {"value": 20.0, "unit": "ratio", "source": "default", "note": "typical truck"},
    "fwd": {"value": False, "unit": "bool", "source": "default", "note": "trucks are typically RWD"},
    "rwd": {"value": True, "unit": "bool", "source": "default"},
}

# ─── User-selectable options for dropdowns ───
DROPDOWN_OPTIONS = {
    "striking": {
        "label": "Which vehicle struck the other?",
        "options": ["Vehicle 1 (striking)", "Vehicle 2 (striking)", "Both (head-on)"],
        "maps_to": "striking",
    },
    "drive_type": {
        "label": "Drive type",
        "options": ["FWD", "RWD", "AWD"],
        "maps_to": ["fwd", "rwd", "awd"],
    },
    "collision_type": {
        "label": "Collision type",
        "options": [
            "Rear-end",
            "Head-on",
            "Side-impact (T-bone)",
            "Sideswipe (same direction)",
            "Sideswipe (opposite direction)",
            "Single vehicle (barrier)",
            "Other",
        ],
        "maps_to": "impact_config",
    },
    "impact_location_v1": {
        "label": "Impact location on Vehicle 1",
        "options": ["Front center", "Front left", "Front right", "Left side", "Right side", "Rear center", "Rear left", "Rear right"],
        "maps_to": "impact_points",
    },
    "impact_location_v2": {
        "label": "Impact location on Vehicle 2",
        "options": ["Front center", "Front left", "Front right", "Left side", "Right side", "Rear center", "Rear left", "Rear right"],
        "maps_to": "edgeimpact",
    },
    "cor": {
        "label": "Coefficient of restitution",
        "options": ["0.0 (perfectly plastic — vehicles stick)", "0.1 (typical)", "0.2 (moderate bounce)", "0.3 (high restitution)"],
        "maps_to": "cor",
    },
    "vehicle_mu": {
        "label": "Inter-vehicle friction during impact",
        "options": ["0.3 (smooth metal)", "0.45 (typical)", "0.6 (rough surface)", "0.8 (high friction)"],
        "maps_to": "vehicle_mu",
    },
}


def convert_vehicle(parsed_veh: dict, vehicle_index: int = 0) -> dict:
    """
    Convert a single parsed PC-Crash vehicle to pycrash input format.

    Returns: {
        "pycrash_input": {field: value},         # ready for Vehicle(name, input_dict)
        "field_sources": {field: {value, unit, source, note?, original?}},
        "gaps": [list of fields needing user input],
        "dropdowns": {field: dropdown_config},
        "time_inputs": {time, throttle, brake, steer},  # for Vehicle.time_inputs()
        "warnings": [list of conversion warnings],
    }
    """
    sv = parsed_veh.get("start_values", {})
    iv = parsed_veh.get("input_values", {})
    seq = parsed_veh.get("sequences", {})
    name = parsed_veh.get("name", f"Vehicle_{vehicle_index + 1}")

    is_truck = _is_heavy_vehicle(iv, name)
    defaults = {**DEFAULTS, **(TRUCK_DEFAULTS if is_truck else {})}

    pycrash_input: dict[str, Any] = {}
    field_sources: dict[str, dict] = {}
    gaps: list[str] = []
    warnings: list[str] = []
    dropdowns: dict[str, dict] = {}

    def _set(key: str, value: Any, unit: str, source: str, note: str = "", original: Any = None):
        pycrash_input[key] = value
        entry: dict[str, Any] = {"value": value, "unit": unit, "source": source}
        if note:
            entry["note"] = note
        if original is not None:
            entry["original"] = original
        field_sources[key] = entry

    # ─── Text fields ───
    year_match = _extract_year(name)
    _set("year", year_match or 2020, "year", "file" if year_match else "default", original=name)

    make, model = _extract_make_model(name)
    _set("vehmake", make, "text", "file", original=name)
    _set("vehmodel", model, "text", "file", original=name)

    # ─── Mass → weight ───
    total_weight_kg = iv.get("total_weight", [None])[0] if isinstance(iv.get("total_weight"), list) else iv.get("total_weight")
    if total_weight_kg is None:
        total_weight_kg = sv.get("total_weight")
    if total_weight_kg is not None:
        _set("weight", round(total_weight_kg * KG_TO_LB, 2), "lb", "file",
             f"{total_weight_kg} kg × {KG_TO_LB}", original=f"{total_weight_kg} kg")
    else:
        _set("weight", defaults.get("weight", {}).get("value", 3000), "lb", "default")
        gaps.append("weight")

    # ─── Dimensions ───
    dim_map = {
        "length": "length",
        "width": "width",
        "wheelbase": "wb",
        "front_track_width": "track",
        "front_overhang": "f_hang",
    }
    for pc_key, py_key in dim_map.items():
        val = _get_input_val(iv, pc_key)
        if val is not None:
            ft_val = round(val * M_TO_FT, 4)
            _set(py_key, ft_val, "ft", "file", f"{val} m × {M_TO_FT}", original=f"{val} m")
        else:
            gaps.append(py_key)
            warnings.append(f"Missing {py_key} ({pc_key})")

    # Rear overhang: derived from length - front_overhang - wheelbase
    length_m = _get_input_val(iv, "length")
    f_hang_m = _get_input_val(iv, "front_overhang")
    wb_m = _get_input_val(iv, "wheelbase")
    if all(v is not None for v in [length_m, f_hang_m, wb_m]):
        r_hang_m = length_m - f_hang_m - wb_m
        _set("r_hang", round(r_hang_m * M_TO_FT, 4), "ft", "derived",
             f"length({length_m}) - f_hang({f_hang_m}) - wb({wb_m}) = {r_hang_m} m",
             original=f"{r_hang_m} m")
    else:
        _set("r_hang", 0.0, "ft", "default")
        gaps.append("r_hang")

    # ─── CG position ───
    lcgf_m = _get_input_val(iv, "distance_c_g_front_axle")
    if lcgf_m is not None:
        _set("lcgf", round(lcgf_m * M_TO_FT, 4), "ft", "file",
             f"{lcgf_m} m × {M_TO_FT}", original=f"{lcgf_m} m")
        if wb_m is not None:
            lcgr_m = wb_m - lcgf_m
            _set("lcgr", round(lcgr_m * M_TO_FT, 4), "ft", "derived",
                 f"wb({wb_m}) - lcgf({lcgf_m}) = {lcgr_m} m", original=f"{lcgr_m} m")
    else:
        gaps.append("lcgf")
        gaps.append("lcgr")

    # CG height
    hcg_m = _get_input_val(iv, "c_g_height_above_ground")
    if hcg_m is not None:
        _set("hcg", round(hcg_m * M_TO_FT, 4), "ft", "file",
             f"{hcg_m} m × {M_TO_FT}", original=f"{hcg_m} m")
    else:
        _set("hcg", 1.64, "ft", "default", "~0.5m typical sedan CG height")

    # ─── Yaw moment of inertia ───
    izz_kgm2 = _get_input_val(iv, "yaw_moment_of_inertia")
    if izz_kgm2 is not None:
        _set("izz", round(izz_kgm2 * KGM2_TO_LBFTS2, 4), "lb-ft-s²", "file",
             f"{izz_kgm2} kgm² × {KGM2_TO_LBFTS2}", original=f"{izz_kgm2} kgm²")
    else:
        gaps.append("izz")

    # ─── Initial position and heading ───
    cgx = sv.get("center_of_gravity_x")
    cgy = sv.get("center_of_gravity_y")
    if cgx is not None:
        _set("init_x_pos", round(cgx * M_TO_FT, 4), "ft", "file",
             f"{cgx} m × {M_TO_FT}", original=f"{cgx} m")
    else:
        _set("init_x_pos", 0.0, "ft", "default")
        gaps.append("init_x_pos")

    if cgy is not None:
        _set("init_y_pos", round(cgy * M_TO_FT, 4), "ft", "file",
             f"{cgy} m × {M_TO_FT}", original=f"{cgy} m")
    else:
        _set("init_y_pos", 0.0, "ft", "default")
        gaps.append("init_y_pos")

    heading = sv.get("heading_angle")
    if heading is not None:
        _set("head_angle", heading, "deg", "file",
             note="PC-Crash heading convention — verify against pycrash frame",
             original=f"{heading} deg")
        warnings.append("Heading angle convention may differ between PC-Crash and pycrash — verify with a calibration run")
    else:
        _set("head_angle", 0.0, "deg", "default")
        gaps.append("head_angle")

    # ─── Initial velocities ───
    speed_kmh = sv.get("velocity_magnitude")
    heading_deg = sv.get("heading_angle", 0)
    if speed_kmh is not None:
        mph = round(speed_kmh * KMH_TO_MPH, 2)
        _set("vx_initial", mph, "mph", "file",
             f"{speed_kmh} km/h × {KMH_TO_MPH}", original=f"{speed_kmh} km/h")
        _set("vy_initial", 0.0, "mph", "file", "lateral velocity assumed 0 at start")
    else:
        _set("vx_initial", 0.0, "mph", "default")
        _set("vy_initial", 0.0, "mph", "default")

    # Yaw rate
    yaw_rads = sv.get("yaw_velocity")
    if yaw_rads is not None:
        _set("omega_z", round(yaw_rads * RADS_TO_DEGS, 4), "deg/s", "file",
             f"{yaw_rads} rad/s × {RADS_TO_DEGS}", original=f"{yaw_rads} rad/s")
    else:
        _set("omega_z", 0.0, "deg/s", "default")

    # ─── lcgf / lcgr fallback (wb/2 approximation) ───
    if "lcgf" not in pycrash_input:
        wb_val = pycrash_input.get("wb")
        if wb_val:
            half_wb = round(wb_val / 2, 4)
            _set("lcgf", half_wb, "ft", "derived", f"wb/2 = {wb_val}/2 (CG at wheelbase center)")
            _set("lcgr", half_wb, "ft", "derived", f"wb/2 = {wb_val}/2 (CG at wheelbase center)")
        else:
            _set("lcgf", 4.185, "ft", "default", "~1.276m typical sedan CG-to-front-axle")
            _set("lcgr", 4.185, "ft", "default", "~1.276m typical sedan CG-to-rear-axle")
            gaps.append("lcgf")
            gaps.append("lcgr")

    # ─── Defaults for missing fields ───
    for key, default_info in defaults.items():
        if key not in pycrash_input:
            _set(key, default_info["value"], default_info.get("unit", ""),
                 default_info.get("source", "default"), default_info.get("note", ""))
            if default_info.get("source") in ("user_required", "not_available"):
                gaps.append(key)

    # ─── Time inputs from sequences ───
    time_inputs = _build_time_inputs(seq)

    # ─── Dropdowns for user selection ───
    dropdowns = {}
    for key, config in DROPDOWN_OPTIONS.items():
        # Only include dropdowns relevant to this vehicle
        if key.endswith("_v1") and vehicle_index != 0:
            continue
        if key.endswith("_v2") and vehicle_index != 1:
            continue
        dropdowns[key] = config

    return {
        "pycrash_input": pycrash_input,
        "field_sources": field_sources,
        "gaps": gaps,
        "dropdowns": dropdowns,
        "time_inputs": time_inputs,
        "warnings": warnings,
    }


def build_impact_config(
    vehicles: list[dict],
    user_selections: dict,
) -> dict:
    """
    Build the Impact() constructor arguments from converted vehicles + user dropdown selections.
    Returns: {name, dt, model, vehicles, pairs, impact_params, impact_points_config}
    """
    # Extract user selections
    cor = float(user_selections.get("cor", "0.1").split("(")[0].strip())
    vehicle_mu = float(user_selections.get("vehicle_mu", "0.45").split("(")[0].strip())
    collision_type = user_selections.get("collision_type", "Rear-end")

    impact_config = {
        "name": "Scenario_from_PCCrash",
        "dt": 3,  # impact duration in seconds
        "model": "IMPC",  # impulse-momentum (doesn't need crush stiffness)
        "pairs": [[0, 1]],  # vehicle indices that interact
        "impact_params": {
            0: {"vehicle_mu": vehicle_mu, "cor": cor}
        },
    }

    # Impact points from user dropdown selections
    impact_points_config = {
        "v1_location": user_selections.get("impact_location_v1", "Front center"),
        "v2_location": user_selections.get("impact_location_v2", "Left side"),
    }

    return {**impact_config, "impact_points_config": impact_points_config}


def compute_impact_points(vehicle_data: dict, location: str) -> list[tuple]:
    """
    Compute pycrash impact_points from a location description and vehicle dimensions.
    Returns [(x, y, angle)] in vehicle-local coordinates.
    """
    lcgf = vehicle_data.get("lcgf", 5.0)
    lcgr = vehicle_data.get("lcgr", 4.0)
    f_hang = vehicle_data.get("f_hang", 2.5)
    r_hang = vehicle_data.get("r_hang", 1.5)
    width = vehicle_data.get("width", 5.5)
    half_w = width / 2

    front_x = lcgf + f_hang
    rear_x = -(lcgr + r_hang)

    location_map = {
        "Front center": [(front_x, 0, 0)],
        "Front left": [(front_x, half_w, -15)],
        "Front right": [(front_x, -half_w, 15)],
        "Left side": [(0, half_w, -90)],
        "Right side": [(0, -half_w, 90)],
        "Rear center": [(rear_x, 0, 180)],
        "Rear left": [(rear_x, half_w, -165)],
        "Rear right": [(rear_x, -half_w, 165)],
    }

    return location_map.get(location, [(front_x, 0, 0)])


def compute_edge_impact(vehicle_data: dict, location: str) -> tuple[list[int], list[tuple]]:
    """
    Compute pycrash edgeimpact and edgeimpact_points from location description.
    Edge indices: 1=front, 2=right, 3=left, 4=rear
    """
    lcgf = vehicle_data.get("lcgf", 5.0)
    lcgr = vehicle_data.get("lcgr", 4.0)
    f_hang = vehicle_data.get("f_hang", 2.5)
    r_hang = vehicle_data.get("r_hang", 1.5)
    width = vehicle_data.get("width", 5.5)
    half_w = width / 2

    front_x = lcgf + f_hang
    rear_x = -(lcgr + r_hang)

    edge_map = {
        "Front center": (1, (rear_x, -half_w, front_x, -half_w)),
        "Front left": (1, (rear_x, -half_w, front_x, -half_w)),
        "Front right": (1, (rear_x, half_w, front_x, half_w)),
        "Left side": (3, (rear_x, half_w, front_x, half_w)),
        "Right side": (2, (rear_x, -half_w, front_x, -half_w)),
        "Rear center": (4, (rear_x, -half_w, rear_x, half_w)),
        "Rear left": (4, (rear_x, half_w, rear_x, -half_w)),
        "Rear right": (4, (rear_x, -half_w, rear_x, half_w)),
    }

    edge_idx, edge_pts = edge_map.get(location, (3, (rear_x, -half_w, front_x, -half_w)))
    return [edge_idx], [edge_pts]


# ─── Helper functions ───

def _get_input_val(iv: dict, key: str) -> float | None:
    """Get the first value from an input_values field, handling list-or-scalar."""
    val = iv.get(key)
    if val is None:
        return None
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _is_heavy_vehicle(iv: dict, name: str) -> bool:
    """Heuristic: is this a truck/bus?"""
    name_lower = name.lower()
    heavy_keywords = ["eicher", "tata", "ashok", "truck", "bus", "lorry", "tipper", "trailer"]
    if any(kw in name_lower for kw in heavy_keywords):
        return True
    mass = _get_input_val(iv, "total_weight")
    if mass and mass > 2500:
        return True
    return False


def _extract_year(name: str) -> int | None:
    """Extract a 4-digit year from vehicle name."""
    import re
    m = re.search(r"\((\d{4})\)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(19|20)\d{2}\b", name)
    if m:
        return int(m.group(0))
    return None


def _extract_make_model(name: str) -> tuple[str, str]:
    """Split 'Honda City 1.5 SVMT (2015)' into ('Honda', 'City 1.5 SVMT')."""
    import re
    clean = re.sub(r"\s*\(\d{4}\)\s*", "", name).strip()
    parts = clean.split(None, 1)
    if len(parts) >= 2:
        return parts[0], parts[1]
    return clean, ""


def _build_time_inputs(seq: dict) -> dict:
    """
    Convert PC-Crash BRAKE/STEERING sequences into pycrash time_inputs arrays.
    Returns {time: [...], throttle: [...], brake: [...], steer: [...]}
    """
    if not seq or "actions" not in seq:
        # No sequence data — return default 6-second no-input
        return {
            "time": [0, 1, 2, 3, 4, 5],
            "throttle": [0, 0, 0, 0, 0, 0],
            "brake": [0, 0, 0, 0, 0, 0],
            "steer": [0, 0, 0, 0, 0, 0],
        }

    actions = seq["actions"]
    t = 0.0
    time_points = [0.0]
    throttle_points = [0.0]
    brake_points = [0.0]
    steer_points = [0.0]

    current_brake = 0.0
    current_steer = 0.0

    for action in actions:
        atype = action.get("type", "")

        if atype == "start":
            continue

        if atype == "brake":
            duration = action.get("maximum_stopping_time", 1.0)
            axle1_left = action.get("axle_1_left", 0)
            axle1_right = action.get("axle_1_right", 0)
            axle2_left = action.get("axle_2_left", 0)
            axle2_right = action.get("axle_2_right", 0)

            bf = action.get("brake_force", {})
            if isinstance(bf, dict):
                axle1_left = bf.get("axle_1_left", axle1_left)
                axle1_right = bf.get("axle_1_right", axle1_right)
                axle2_left = bf.get("axle_2_left", axle2_left)
                axle2_right = bf.get("axle_2_right", axle2_right)

            # Convert percentage to fraction (pycrash uses 0-1 range)
            avg_brake = (axle1_left + axle1_right + axle2_left + axle2_right) / 400.0

            t += duration
            time_points.append(round(t, 3))
            throttle_points.append(0.0)
            brake_points.append(avg_brake)
            steer_points.append(current_steer)
            current_brake = avg_brake

        elif atype == "steering":
            duration = action.get("steering_time", 0.5)
            new_angle = action.get("new_steering_angle_axle_1", 0)

            t += duration
            time_points.append(round(t, 3))
            throttle_points.append(0.0)
            brake_points.append(current_brake)
            steer_points.append(new_angle)
            current_steer = new_angle

    # Ensure we have at least 2 time points
    if len(time_points) < 2:
        time_points.append(5.0)
        throttle_points.append(0.0)
        brake_points.append(0.0)
        steer_points.append(0.0)

    return {
        "time": time_points,
        "throttle": throttle_points,
        "brake": brake_points,
        "steer": steer_points,
    }
