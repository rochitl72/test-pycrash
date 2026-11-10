"""
Groq-based AI scene description generator.
Takes parsed + converted scene data and produces a human-readable accident description.
The LLM NEVER invents physics values — it only narrates what the parser extracted.
"""
import json
import os
import httpx
from typing import Any

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # open-weight, fast on Groq

SYSTEM_PROMPT = """You are an accident reconstruction analyst assistant. Your job is to take structured accident scene data (parsed from a PC-Crash values export file) and produce:

1. A clear, human-readable description of the accident scene that a traffic officer can verify.
2. A list of assumptions made (fields that were defaulted or derived).
3. A list of missing information that needs officer input.

RULES:
- NEVER invent or guess any physics values (speeds, positions, angles, masses, forces).
- Only describe what is in the data provided.
- Use plain language, not technical jargon where possible.
- Distances in meters, speeds in km/h (these are Indian accidents).
- Flag anything that looks inconsistent or unusual.
- Structure your response as JSON with keys: "description", "assumptions", "missing_items", "warnings"."""

USER_PROMPT_TEMPLATE = """Here is the parsed accident scene data from a PC-Crash values export file.
Generate a human-readable accident scene description for officer verification.

SCENE DATA:
{scene_json}

CONVERTED PYCRASH DATA:
Vehicle count: {vehicle_count}
{vehicle_summaries}

GAPS DETECTED:
{gaps_summary}

Return your response as valid JSON with these keys:
- "description": A 2-3 paragraph plain-language description of the accident scene
- "assumptions": Array of strings listing each assumption made
- "missing_items": Array of strings listing what the officer needs to provide
- "warnings": Array of strings listing any inconsistencies or concerns"""


async def generate_scene_description(
    parsed_scene: dict,
    converted_vehicles: list[dict],
) -> dict:
    """
    Call Groq API to generate a human-readable scene description.
    Falls back to a template-based description if Groq is unavailable.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")

    # Build summaries for the prompt
    vehicle_summaries = []
    all_gaps: list[str] = []

    for i, cv in enumerate(converted_vehicles):
        pi = cv["pycrash_input"]
        sources = cv["field_sources"]
        gaps = cv["gaps"]
        all_gaps.extend([f"Vehicle {i+1}: {g}" for g in gaps])

        summary = f"""Vehicle {i+1}: {pi.get('vehmake', '?')} {pi.get('vehmodel', '?')} ({pi.get('year', '?')})
  - Weight: {sources.get('weight', {}).get('original', '?')} -> {pi.get('weight', '?')} lb
  - Speed: {sources.get('vx_initial', {}).get('original', '?')} -> {pi.get('vx_initial', '?')} mph
  - Position: ({pi.get('init_x_pos', '?')}, {pi.get('init_y_pos', '?')}) ft
  - Heading: {pi.get('head_angle', '?')} deg
  - Dimensions: {sources.get('length', {}).get('original', '?')} L x {sources.get('width', {}).get('original', '?')} W"""
        vehicle_summaries.append(summary)

    gaps_summary = "\n".join(f"- {g}" for g in all_gaps) if all_gaps else "No critical gaps detected."

    user_prompt = USER_PROMPT_TEMPLATE.format(
        scene_json=json.dumps(parsed_scene, indent=2, default=str)[:4000],
        vehicle_count=len(converted_vehicles),
        vehicle_summaries="\n".join(vehicle_summaries),
        gaps_summary=gaps_summary,
    )

    if not api_key:
        # No API key — use template fallback
        return _template_description(parsed_scene, converted_vehicles)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        # Fallback to template
        result = _template_description(parsed_scene, converted_vehicles)
        result["warnings"].append(f"Groq API unavailable ({type(e).__name__}), using template description")
        return result


def _template_description(parsed_scene: dict, converted_vehicles: list[dict]) -> dict:
    """Template-based fallback when Groq is not available."""
    vehicles = parsed_scene.get("vehicles", [])
    barricades = parsed_scene.get("barricades", [])

    desc_parts = []
    assumptions = []
    missing = []
    warnings = []

    desc_parts.append("ACCIDENT SCENE DESCRIPTION (auto-generated from PC-Crash export)\n")

    for i, veh in enumerate(vehicles):
        sv = veh.get("start_values", {})
        iv = veh.get("input_values", {})
        name = veh.get("name", f"Vehicle {i+1}")

        speed = sv.get("velocity_magnitude", 0)
        heading = sv.get("heading_angle", 0)
        cgx = sv.get("center_of_gravity_x", 0)
        cgy = sv.get("center_of_gravity_y", 0)

        mass = iv.get("total_weight", [None])
        if isinstance(mass, list):
            mass = mass[0] if mass else None

        length = iv.get("length", [None])
        if isinstance(length, list):
            length = length[0] if length else None

        desc_parts.append(
            f"Vehicle {i+1}: {name}\n"
            f"  Starting speed: {speed} km/h\n"
            f"  Heading: {heading} deg\n"
            f"  Position (CG): ({cgx}, {cgy}) meters\n"
            f"  Total mass: {mass} kg\n"
            f"  Length: {length} m\n"
        )

        # Sequence summary
        seq = veh.get("sequences", {})
        if seq and "actions" in seq:
            actions = seq["actions"]
            action_strs = []
            for act in actions:
                if act["type"] == "brake":
                    dur = act.get("maximum_stopping_time", "?")
                    accel = act.get("specified_brake_acceleration", "?")
                    action_strs.append(f"brakes for {dur}s (decel: {accel} m/s2)")
                elif act["type"] == "steering":
                    dur = act.get("steering_time", "?")
                    angle = act.get("new_steering_angle_axle_1", "?")
                    action_strs.append(f"steers to {angle} deg over {dur}s")
            if action_strs:
                desc_parts.append(f"  Driver actions: {', then '.join(action_strs)}\n")

    if barricades:
        for barr in barricades:
            biv = barr.get("input_values", {})
            bsv = barr.get("start_values", {})
            desc_parts.append(
                f"\nBarricade: {barr.get('name', 'Unknown')}\n"
                f"  Position: ({bsv.get('center_of_gravity_x', '?')}, {bsv.get('center_of_gravity_y', '?')}) m\n"
                f"  Mass: {_first(biv.get('total_weight'))} kg\n"
            )

    # Gaps from converted data
    for i, cv in enumerate(converted_vehicles):
        for gap in cv.get("gaps", []):
            missing.append(f"Vehicle {i+1}: {gap}")

    # Standard assumptions
    for i, cv in enumerate(converted_vehicles):
        for field, info in cv.get("field_sources", {}).items():
            if info.get("source") in ("default", "derived"):
                assumptions.append(f"Vehicle {i+1} {field}: {info.get('note', info.get('value'))}")

    if not warnings:
        warnings.append("START VALUES = END VALUES in this file — no simulation results present, only initial setup")

    return {
        "description": "\n".join(desc_parts),
        "assumptions": assumptions,
        "missing_items": missing,
        "warnings": warnings,
    }


def _first(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val
