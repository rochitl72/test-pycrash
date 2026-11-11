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

# In-memory session store (single-user prototype)
sessions: dict[str, dict] = {}


class UserSelections(BaseModel):
    session_id: str
    selections: dict  # dropdown field -> chosen option string


class SimulationRequest(BaseModel):
    session_id: str


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


def _run_simulation_sync(converted, selections, session_id):
    """Synchronous simulation — run in a thread to avoid blocking the event loop."""
    impact_cfg = build_impact_config(converted, selections) if len(converted) >= 2 else {}
    model_used = "kinematic_fallback"
    trajectories = []

    try:
        from pycrash import Vehicle, SingleMotion
        model_used = "pycrash_SingleMotion"

        for i, cv in enumerate(converted):
            pi = cv["pycrash_input"]
            ti = cv["time_inputs"]
            name = f"{pi.get('vehmake', 'V')}{i+1}"

            try:
                veh = Vehicle(name, pi)
                veh.time_inputs(
                    ti["time"], ti["throttle"], ti["brake"], ti["steer"], False
                )
                sim = SingleMotion(f"sim_{name}", veh)
                traj = _extract_pycrash_trajectory(sim.veh, pi)
                geom = _extract_vehicle_geometry(sim.veh)
                trajectories.append({
                    "vehicle_id": f"v{i+1}",
                    "vehicle_name": pi.get("vehmake", "") + " " + pi.get("vehmodel", ""),
                    "trajectory": traj,
                    "geometry": geom,
                    "dimensions": {
                        "length_ft": pi.get("length", 14),
                        "width_ft": pi.get("width", 6),
                        "height_ft": pi.get("hcg", 1.6) * 3,
                    },
                    "model": "pycrash_SingleMotion",
                })
            except Exception as veh_err:
                print(f"[pycrash] Vehicle {i+1} error: {veh_err}")
                traceback.print_exc()
                traj = _compute_kinematic_trajectory(cv)
                trajectories.append({
                    "vehicle_id": f"v{i+1}",
                    "vehicle_name": pi.get("vehmake", "") + " " + pi.get("vehmodel", ""),
                    "trajectory": traj,
                    "dimensions": {
                        "length_ft": pi.get("length", 14),
                        "width_ft": pi.get("width", 6),
                        "height_ft": pi.get("hcg", 1.6) * 3,
                    },
                    "model": "kinematic_fallback",
                    "error": str(veh_err),
                })

    except ImportError:
        model_used = "kinematic_fallback"
        for i, cv in enumerate(converted):
            traj = _compute_kinematic_trajectory(cv)
            pi = cv["pycrash_input"]
            trajectories.append({
                "vehicle_id": f"v{i+1}",
                "vehicle_name": pi.get("vehmake", "") + " " + pi.get("vehmodel", ""),
                "trajectory": traj,
                "dimensions": {
                    "length_ft": pi.get("length", 14),
                    "width_ft": pi.get("width", 6),
                    "height_ft": pi.get("hcg", 1.6) * 3,
                },
                "model": "kinematic_fallback",
            })

    return {
        "session_id": session_id,
        "impact_config": impact_cfg,
        "trajectories": trajectories,
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
