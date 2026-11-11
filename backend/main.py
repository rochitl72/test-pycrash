"""
CrashRecon API — FastAPI backend for Indian accident reconstruction.
Endpoints: upload PC-Crash file, parse, convert, describe, simulate with REAL pycrash.
"""
import asyncio
import json
import math
import os
import traceback
import warnings
from pathlib import Path

# Suppress matplotlib GUI warnings on headless server
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=UserWarning)
# pycrash (third-party) uses pandas chained assignment; harmless on pandas<3 (keep pandas pinned <3).
warnings.filterwarnings('ignore', category=FutureWarning, module=r'pycrash(\..*)?')

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pccrash_parser import parse_pccrash_file
from pycrash_adapter import (
    DROPDOWN_OPTIONS,
    build_impact_config,
    compute_edge_impact,
    compute_impact_points,
    convert_vehicle,
)
from scene_describer import generate_scene_description
import indian_vehicles as iv

load_dotenv()

app = FastAPI(
    title="CrashRecon",
    description="Open-source Indian accident reconstruction from PC-Crash data",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
SAMPLE_DIR = PROJECT_DIR / "sample_data"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# In-memory session store (single-user prototype)
sessions: dict[str, dict] = {}


class UserSelections(BaseModel):
    session_id: str
    selections: dict  # dropdown field -> chosen option string


class SimulationRequest(BaseModel):
    session_id: str


class ScenarioVehicle(BaseModel):
    vehicle_id: str
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    loaded: bool = False
    brake: dict | None = None  # {t_start, t_full, level}


class ScenarioRequest(BaseModel):
    scene_id: str | None = None          # use a preset...
    vehicles: list[ScenarioVehicle] | None = None  # ...or custom vehicles
    mu_ground: float = 0.8
    cor: float = 0.15
    vehicle_mu: float = 0.45


@app.get("/api/vehicles")
async def list_indian_vehicles():
    """Catalog of Indian vehicles available for scenario building."""
    return {"vehicles": iv.vehicle_catalog(), "scenes": [
        {"id": k, "label": v["label"], "description": v["description"],
         "environment": v["environment"]} for k, v in iv.SCENES.items()]}


@app.post("/api/scenario")
async def build_scenario(req: ScenarioRequest):
    """
    Build a crash scenario from explicit crash variables (Indian vehicle models,
    impact speeds, positions, friction, restitution) - no PC-Crash file needed.
    Returns the same payload shape as /api/upload so the UI flow is identical.
    """
    if req.scene_id:
        scene = iv.SCENES.get(req.scene_id)
        if not scene:
            raise HTTPException(404, f"Unknown scene '{req.scene_id}'")
        specs = scene["vehicles"]
        cor, vehicle_mu = scene["cor"], scene["vehicle_mu"]
        description = scene["label"] + " - " + scene["description"]
        environment = scene["environment"]
    else:
        if not req.vehicles or not (1 <= len(req.vehicles) <= 2):
            raise HTTPException(400, "Provide scene_id or 1-2 vehicles")
        specs = [v.model_dump() for v in req.vehicles]
        cor, vehicle_mu = req.cor, req.vehicle_mu
        names = ", ".join(iv.INDIAN_VEHICLES[v["vehicle_id"]]["label"]
                          for v in specs if v["vehicle_id"] in iv.INDIAN_VEHICLES)
        description = f"Custom scenario: {names}. Restitution {cor}, inter-vehicle friction {vehicle_mu}."
        environment = "highway"

    converted = []
    for spec in specs:
        vid = spec["vehicle_id"]
        if vid not in iv.INDIAN_VEHICLES:
            raise HTTPException(404, f"Unknown vehicle '{vid}'")
        pi = iv.to_pycrash_input(
            vid, speed_kmh=spec.get("speed_kmh", 0.0), heading_deg=spec.get("heading_deg", 0.0),
            x_m=spec.get("x_m", 0.0), y_m=spec.get("y_m", 0.0), loaded=spec.get("loaded", False))
        converted.append({
            "pycrash_input": pi,
            "time_inputs": iv.build_time_inputs_for(spec, SIM_END),
            "field_sources": {k: {"value": v, "source": "indian_vehicle_db"} for k, v in pi.items()},
            "gaps": [], "warnings": [], "dropdowns": {},
            "render": {"body": iv.INDIAN_VEHICLES[vid]["body"], "color": iv.INDIAN_VEHICLES[vid]["color"]},
        })

    import uuid
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "parsed": {"vehicles": [], "source": "scenario"},
        "converted": converted,
        "barricade_converted": [],
        "scene_description": description,
        "user_selections": {"cor": str(cor), "vehicle_mu": str(vehicle_mu)},
    }
    return {
        "session_id": session_id,
        "source": "scenario",
        "environment": environment,
        "scene_description": description,
        "converted_vehicles": [
            {"vehicle_name": (cv["pycrash_input"].get("vehmake", "") + " " +
                              cv["pycrash_input"].get("vehmodel", "")).strip(),
             "pycrash_input": cv["pycrash_input"], "field_sources": cv["field_sources"],
             "gaps": [], "warnings": [], "time_inputs": cv["time_inputs"],
             "render": cv["render"]}
            for cv in converted],
        "dropdowns": {},
    }


@app.get("/api/health")
async def health():
    pycrash_ok = False
    try:
        from pycrash import Vehicle, SingleMotion
        pycrash_ok = True
    except ImportError:
        pass
    return {
        "status": "ok",
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "pycrash_available": pycrash_ok,
    }


@app.get("/api/dropdowns")
async def get_dropdowns():
    """Return all available dropdown options for the frontend."""
    return DROPDOWN_OPTIONS


@app.post("/api/upload")
async def upload_pccrash_file(file: UploadFile = File(...)):
    """
    Upload and parse a PC-Crash values export file.
    Returns parsed scene, converted pycrash inputs, AI description, gaps, and dropdowns.
    """
    try:
        raw = await file.read()
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(400, "Could not decode file — try saving as UTF-8")

        # 1. Parse
        parsed = parse_pccrash_file(text)

        # 2. Convert each vehicle
        converted = []
        for i, veh in enumerate(parsed["vehicles"]):
            cv = convert_vehicle(veh, i)
            converted.append(cv)

        # Also convert barricades
        barricade_converted = []
        for barr in parsed.get("barricades", []):
            bc = convert_vehicle(barr, len(parsed["vehicles"]))
            barricade_converted.append(bc)

        # 3. AI scene description
        scene_desc = await generate_scene_description(parsed, converted)

        # 4. Store session
        import uuid
        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = {
            "parsed": parsed,
            "converted": converted,
            "barricade_converted": barricade_converted,
            "scene_description": scene_desc,
            "user_selections": {},
        }

        return {
            "session_id": session_id,
            "parsed_scene": parsed,
            "converted_vehicles": [
                {
                    "vehicle_name": cv["pycrash_input"].get("vehmake", "") + " " + cv["pycrash_input"].get("vehmodel", ""),
                    "pycrash_input": cv["pycrash_input"],
                    "field_sources": cv["field_sources"],
                    "gaps": cv["gaps"],
                    "warnings": cv["warnings"],
                    "time_inputs": cv["time_inputs"],
                }
                for cv in converted
            ],
            "barricades": [
                {
                    "name": bc["pycrash_input"].get("vehmake", "Barricade"),
                    "pycrash_input": bc["pycrash_input"],
                    "field_sources": bc["field_sources"],
                }
                for bc in barricade_converted
            ],
            "scene_description": scene_desc,
            "dropdowns": DROPDOWN_OPTIONS,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Parse error: {str(e)}")


@app.post("/api/selections")
async def update_selections(req: UserSelections):
    """Store user's dropdown selections for a session."""
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sessions[req.session_id]["user_selections"] = req.selections
    return {"status": "ok", "selections": req.selections}


SIM_DEFAULTS = {"dt_motion": 0.01, "mu_max": 0.76, "alpha_max": 0.174533}
SIM_END = 5.0  # seconds simulated


def _selection_float(selections: dict, key: str, default: float) -> float:
    """Dropdown values look like '0.1 (typical)' - take the leading number."""
    try:
        return float(str(selections.get(key, default)).split("(")[0].strip())
    except (TypeError, ValueError):
        return default


def _vehicle_payload(i, pi, trajectory, geometry, model, error=None, render=None) -> dict:
    out = {
        "vehicle_id": f"v{i+1}",
        "vehicle_name": (pi.get("vehmake", "") + " " + pi.get("vehmodel", "")).strip(),
        "trajectory": trajectory,
        "geometry": geometry,
        "dimensions": {
            "length_ft": pi.get("length", 14),
            "width_ft": pi.get("width", 6),
            "height_ft": pi.get("hcg", 1.6) * 3,
            "wb_ft": pi.get("wb", 8.5),
            "track_ft": pi.get("track", 5.0),
            "f_hang_ft": pi.get("f_hang", 3.0),
            "r_hang_ft": pi.get("r_hang", 3.0),
            "tire_d_ft": pi.get("tire_d", 2.1),
            "lcgf_ft": pi.get("lcgf", 4.2),
        },
        "model": model,
    }
    if render:
        out["render"] = render
    if error:
        out["error"] = error
    return out


def _speed_mph(model_df, idx) -> float:
    idx = max(0, min(idx, len(model_df) - 1))
    row = model_df.iloc[idx]
    return math.hypot(float(row["Vx"]), float(row["Vy"])) * 3600 / 5280


def _impact_summary(contact, imp, pis, cor, mu) -> dict:
    """Plain-language summary of the collision for the UI / report."""
    s, t = contact["striking_index"], contact["struck_index"]
    i = contact["frame"]
    speeds = []
    for k, veh in enumerate(imp.vehicles):
        before = _speed_mph(veh.model, i - 2)
        after = _speed_mph(veh.model, i + 15)
        speeds.append({
            "vehicle_id": f"v{k+1}",
            "role": "striking" if k == s else "struck",
            "before_mph": round(before, 1),
            "after_mph": round(after, 1),
            "delta_v_mph": round(abs(after - before), 1),
        })
    names = [(p.get("vehmake", "") + " " + p.get("vehmodel", "")).strip() for p in pis]
    return {
        "detected": True,
        "model": "pycrash Impact (IMPC, Carpenter-Welcher)",
        "t": contact["t"],
        "striking": f"v{s+1}",
        "struck": f"v{t+1}",
        "striking_name": names[s],
        "struck_name": names[t],
        "struck_edge": contact["edge_name"],
        "point": {"x": contact["global_point"][0], "y": contact["global_point"][1]},
        "impact_point_local": list(contact["impact_point"]),
        "restitution": cor,
        "vehicle_friction": mu,
        "speeds": speeds,
    }


def _run_simulation_sync(converted, selections, session_id):
    """
    Run in a worker thread (pycrash is CPU heavy and synchronous).
    1. Simulate every vehicle alone with pycrash SingleMotion.
    2. If two vehicles' bodies overlap, derive the contact (striking vehicle, impact
       point, struck edge, plane angle) from the geometry and re-run both vehicles
       through pycrash's Impact (IMPC) so momentum is exchanged at the collision.
    """
    cor = _selection_float(selections, "cor", 0.1)
    mu = _selection_float(selections, "vehicle_mu", 0.45)
    n = len(converted)
    pis = [cv["pycrash_input"] for cv in converted]
    tis = [cv["time_inputs"] for cv in converted]
    payloads = [None] * n
    impact_info = None
    model_used = "pycrash_SingleMotion"

    try:
        from pycrash import Vehicle, SingleMotion
        import impact_adapter as ia
    except ImportError:
        Vehicle = SingleMotion = ia = None

    sims = [None] * n
    if SingleMotion is not None:
        for i in range(n):
            pi = pis[i]
            try:
                veh = Vehicle(f"{pi.get('vehmake', 'V')}{i+1}", dict(pi))
                ti = ia.extend_time_inputs(tis[i], SIM_END)
                veh.time_inputs(ti["time"], ti["throttle"], ti["brake"], ti["steer"], False)
                sims[i] = SingleMotion(f"sim_{i+1}", veh, dict(SIM_DEFAULTS)).veh
            except Exception as veh_err:
                print(f"[pycrash] Vehicle {i+1} error: {veh_err}")
                traceback.print_exc()
                payloads[i] = _vehicle_payload(
                    i, pi, _compute_kinematic_trajectory(converted[i]), [],
                    "kinematic_fallback", str(veh_err), converted[i].get("render"))

    # ---- collision between the first two vehicles ----
    if n >= 2 and sims[0] is not None and sims[1] is not None:
        try:
            contact = ia.find_first_contact(sims[:2], pis[:2])
        except Exception as e:
            traceback.print_exc()
            contact, impact_info = None, {"detected": False, "error": str(e)}
        if contact:
            try:
                imp = ia.run_impact(pis[:2], tis[:2], contact, SIM_END, cor=cor, mu=mu)
                impact_info = _impact_summary(contact, imp, pis[:2], cor, mu)
                model_used = "pycrash_Impact_IMPC"
                for k, veh in enumerate(imp.vehicles):
                    payloads[k] = _vehicle_payload(
                        k, pis[k], _extract_pycrash_trajectory(veh, pis[k]),
                        _extract_vehicle_geometry(veh), "pycrash_Impact_IMPC",
                        render=converted[k].get("render"))
            except Exception as e:
                print(f"[pycrash] Impact error: {e}")
                traceback.print_exc()
                impact_info = {"detected": True, "t": contact["t"], "error": str(e),
                               "struck_edge": contact["edge_name"]}
        elif impact_info is None:
            impact_info = {"detected": False,
                           "message": "The vehicle bodies never overlap in the simulated time."}

    # ---- anything not covered by Impact: use the single-vehicle result ----
    for i in range(n):
        if payloads[i] is not None:
            continue
        if sims[i] is not None:
            payloads[i] = _vehicle_payload(
                i, pis[i], _extract_pycrash_trajectory(sims[i], pis[i]),
                _extract_vehicle_geometry(sims[i]), "pycrash_SingleMotion",
                render=converted[i].get("render"))
        else:
            payloads[i] = _vehicle_payload(
                i, pis[i], _compute_kinematic_trajectory(converted[i]), [],
                "kinematic_fallback", render=converted[i].get("render"))
    if SingleMotion is None:
        model_used = "kinematic_fallback"

    return {
        "session_id": session_id,
        "impact": impact_info,
        "trajectories": payloads,
        "model_used": model_used,
    }


@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    """
    Run pycrash SingleMotion simulation in a thread pool to avoid blocking.
    """
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[req.session_id]
    converted = session["converted"]
    selections = session.get("user_selections", {})

    if len(converted) < 1:
        raise HTTPException(400, "Need at least 1 vehicle for simulation")

    result = await asyncio.to_thread(
        _run_simulation_sync, converted, selections, req.session_id
    )
    return result


def _extract_pycrash_trajectory(sim_veh, pi: dict) -> list[dict]:
    """
    Extract trajectory from a pycrash Vehicle after SingleMotion has run.
    sim_veh.model is a DataFrame with columns including:
      t, Vx, Vy, Vr, theta_deg, theta_rad, Dx, Dy, oz_deg, ...
    Dx, Dy = CG global position (ft)
    Vx, Vy = CG velocity (ft/s)
    theta_deg = heading angle (degrees)
    """
    trajectory = []
    try:
        df = sim_veh.model
        if df is None or len(df) == 0:
            return []
        df = df.dropna(subset=["t", "Dx", "Dy", "Vx", "Vy", "theta_deg"])
        if len(df) == 0:
            return []

        # Sample every Nth row (~200 points max for API response)
        step = max(1, len(df) // 200)

        for idx in range(0, len(df), step):
            row = df.iloc[idx]
            t = float(row["t"])
            x = float(row["Dx"])
            y = float(row["Dy"])
            vx = float(row["Vx"])
            vy = float(row["Vy"])
            heading = float(row["theta_deg"])
            speed_fps = math.sqrt(vx**2 + vy**2)
            speed_mph = speed_fps * 3600 / 5280

            trajectory.append({
                "t": round(t, 3),
                "x": round(x, 4),
                "y": round(y, 4),
                "heading": round(heading, 2),
                "vx": round(vx, 4),
                "vy": round(vy, 4),
                "speed_mph": round(speed_mph, 2),
            })

        # Always include last point
        if len(df) > 1:
            last = df.iloc[-1]
            vx_l = float(last["Vx"])
            vy_l = float(last["Vy"])
            speed_l = math.sqrt(vx_l**2 + vy_l**2) * 3600 / 5280
            trajectory.append({
                "t": round(float(last["t"]), 3),
                "x": round(float(last["Dx"]), 4),
                "y": round(float(last["Dy"]), 4),
                "heading": round(float(last["theta_deg"]), 2),
                "vx": round(vx_l, 4),
                "vy": round(vy_l, 4),
                "speed_mph": round(speed_l, 2),
            })

    except Exception as e:
        print(f"[pycrash] Trajectory extraction error: {e}")
        traceback.print_exc()

    return trajectory


def _extract_vehicle_geometry(sim_veh) -> list[dict]:
    """
    Extract detailed vehicle geometry at sampled timesteps from pycrash's
    position DataFrames (p_gx, p_gy). Each frame contains:
      - body: 4 corners (b_lfc, b_rfc, b_rrc, b_lrc)
      - wheels: 4 wheels × 4 corners each (lfw_a/b/c/d, rfw_a/b/c/d, etc.)
      - cg: center of gravity position
      - velocity vector endpoint
    """
    geometry = []
    try:
        pgx = sim_veh.p_gx
        pgy = sim_veh.p_gy
        model_df = sim_veh.model

        if pgx is None or pgy is None or len(pgx) == 0:
            return []

        # Body corner columns
        body_cols = ["b_lfc", "b_rfc", "b_rrc", "b_lrc"]
        # Wheel corner columns (4 wheels × 4 corners)
        wheel_groups = {
            "lf": ["lfw_a", "lfw_b", "lfw_c", "lfw_d"],  # left front
            "rf": ["rfw_a", "rfw_b", "rfw_c", "rfw_d"],  # right front
            "rr": ["rrw_a", "rrw_b", "rrw_c", "rrw_d"],  # right rear
            "lr": ["lrw_a", "lrw_b", "lrw_c", "lrw_d"],  # left rear
        }

        # Check which columns exist
        avail_body = [c for c in body_cols if c in pgx.columns and c in pgy.columns]
        avail_wheels = {}
        for wname, wcols in wheel_groups.items():
            avail = [c for c in wcols if c in pgx.columns and c in pgy.columns]
            if len(avail) == 4:
                avail_wheels[wname] = avail

        if not avail_body:
            return []

        # Sample ~150 frames for API response
        total = len(pgx)
        step = max(1, total // 150)
        indices = list(range(0, total, step))
        if total - 1 not in indices:
            indices.append(total - 1)

        for idx in indices:
            row_x = pgx.iloc[idx]
            row_y = pgy.iloc[idx]
            if any(not math.isfinite(float(row_x[c])) or not math.isfinite(float(row_y[c])) for c in avail_body):
                continue
            frame = {"i": idx}

            # Time from model DataFrame
            if model_df is not None and idx < len(model_df):
                frame["t"] = round(float(model_df.iloc[idx]["t"]), 3)

            # Body corners [lfc, rfc, rrc, lrc] — closed polygon
            body_x = [round(float(pgx.iloc[idx][c]), 3) for c in avail_body]
            body_y = [round(float(pgy.iloc[idx][c]), 3) for c in avail_body]
            frame["body"] = {"x": body_x, "y": body_y}

            # Wheels
            wheels = {}
            for wname, wcols in avail_wheels.items():
                wx = [round(float(pgx.iloc[idx][c]), 3) for c in wcols]
                wy = [round(float(pgy.iloc[idx][c]), 3) for c in wcols]
                wheels[wname] = {"x": wx, "y": wy}
            frame["wheels"] = wheels

            # CG position
            if "cg" in pgx.columns and "cg" in pgy.columns:
                frame["cg"] = {
                    "x": round(float(pgx.iloc[idx]["cg"]), 3),
                    "y": round(float(pgy.iloc[idx]["cg"]), 3),
                }

            geometry.append(frame)

    except Exception as e:
        print(f"[pycrash] Geometry extraction error: {e}")
        traceback.print_exc()

    return geometry


def _compute_kinematic_trajectory(cv: dict) -> list[dict]:
    """
    Kinematic fallback: constant deceleration with steering (bicycle model).
    Returns [{t, x, y, heading, vx, vy, speed_mph}] in feet and degrees.
    """
    pi = cv["pycrash_input"]
    ti = cv["time_inputs"]

    x = pi.get("init_x_pos", 0)
    y = pi.get("init_y_pos", 0)
    heading_deg = pi.get("head_angle", 0)
    speed_mph = pi.get("vx_initial", 0)

    vx_fps = speed_mph * 5280 / 3600
    heading_rad = math.radians(heading_deg)

    dt = 0.05
    total_time = max(ti["time"]) if ti["time"] else 5.0
    total_time = max(total_time, 0.1)

    trajectory = []
    t = 0.0

    times = ti["time"]
    brakes = ti["brake"]
    steers = ti["steer"]

    wb_ft = pi.get("wb", 8.5)
    mu = 0.8
    g = 32.174

    while t <= total_time + dt:
        brake_frac = _interp(t, times, brakes)
        steer_deg = _interp(t, times, steers)

        trajectory.append({
            "t": round(t, 3),
            "x": round(x, 4),
            "y": round(y, 4),
            "heading": round(math.degrees(heading_rad), 2),
            "vx": round(vx_fps * math.cos(heading_rad), 4),
            "vy": round(vx_fps * math.sin(heading_rad), 4),
            "speed_mph": round(vx_fps * 3600 / 5280, 2),
        })

        decel = brake_frac * mu * g
        vx_fps = max(0, vx_fps - decel * dt)

        if abs(steer_deg) > 0.01 and vx_fps > 0.1:
            steer_rad = math.radians(steer_deg)
            turn_radius = wb_ft / math.tan(steer_rad) if abs(steer_rad) > 0.001 else 1e6
            yaw_rate = vx_fps / turn_radius
            heading_rad += yaw_rate * dt

        x += vx_fps * math.cos(heading_rad) * dt
        y += vx_fps * math.sin(heading_rad) * dt
        t += dt

    return trajectory


def _interp(t: float, times: list[float], values: list[float]) -> float:
    """Linear interpolation with clamping."""
    if not times or not values:
        return 0.0
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            frac = (t - times[i]) / (times[i + 1] - times[i]) if times[i + 1] != times[i] else 0
            return values[i] + frac * (values[i + 1] - values[i])
    return values[-1]


@app.get("/api/sample")
async def get_sample_file():
    """Return the bundled sample PC-Crash file."""
    sample = SAMPLE_DIR / "sample_1.txt"
    if not sample.exists():
        raise HTTPException(404, "No sample file bundled")
    return FileResponse(sample, media_type="text/plain", filename="sample_1.txt")


@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "Frontend not found. Place index.html in ../frontend/"}, 404)
    return FileResponse(index, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
