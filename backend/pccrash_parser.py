"""
Deterministic parser for PC-Crash values export files.
No LLM involved — pure string parsing with latin-1/cp1252 encoding support.
Outputs a canonical scene dict in SI units (metric).
"""
import re
from typing import Any


def parse_pccrash_file(text: str) -> dict:
    """Parse a PC-Crash values output file into a structured scene dict."""
    # Normalise encoding artifacts: replace corrupted degree/squared symbols
    text = text.replace("�", "°")  # replacement char → degree
    text = text.replace("m/s°", "m/s²")  # fix broken m/s²

    lines = text.split("\n")
    scene: dict[str, Any] = {"vehicles": [], "barricades": [], "raw_sections": {}}

    # --- Pass 1: split into major sections ---
    sections = _split_sections(lines)
    scene["raw_sections"] = {k: len(v) for k, v in sections.items()}

    # --- Pass 2: parse vehicle header blocks (first block before START VALUES) ---
    vehicle_names, vehicle_count = _parse_vehicle_headers(sections.get("header", []))

    # --- Pass 3: parse START VALUES / END VALUES ---
    start_vals = _parse_value_block(sections.get("start_values_1", []), vehicle_count)
    end_vals = _parse_value_block(sections.get("end_values_1", []), vehicle_count)

    # Barricade block (second START/END VALUES pair if present)
    barricade_start = _parse_value_block(sections.get("start_values_2", []), 1)
    barricade_end = _parse_value_block(sections.get("end_values_2", []), 1)

    # --- Pass 4: parse SEQUENCES ---
    sequences = _parse_sequences(sections.get("sequences", []))

    # --- Pass 5: parse INPUT VALUES ---
    input_vals = _parse_input_values(sections.get("input_values_1", []), vehicle_count)
    barricade_inputs = _parse_input_values(sections.get("input_values_2", []), 1)

    # --- Assemble vehicles ---
    for i in range(vehicle_count):
        name = vehicle_names[i] if i < len(vehicle_names) else f"Vehicle_{i+1}"
        is_barricade = "barricade" in name.lower()

        veh: dict[str, Any] = {
            "id": f"v{i+1}",
            "name": name,
            "is_barricade": is_barricade,
            "start_values": {k: v[i] if i < len(v) else None for k, v in start_vals.items()},
            "end_values": {k: v[i] if i < len(v) else None for k, v in end_vals.items()},
            "input_values": {k: v[i] if i < len(v) else None for k, v in input_vals.items()},
            "sequences": sequences.get(i, {}),
        }
        scene["vehicles"].append(veh)

    # Handle standalone barricade block
    if barricade_start or barricade_inputs:
        barr_name = "Barricade"
        # Try to find barricade name from sections
        for line in sections.get("barricade_header", []):
            m = re.match(r"Vehicle:\s+(.+)", line)
            if m:
                barr_name = m.group(1).strip()
        barr: dict[str, Any] = {
            "id": "barricade",
            "name": barr_name,
            "is_barricade": True,
            "start_values": {k: v[0] if v else None for k, v in barricade_start.items()},
            "end_values": {k: v[0] if v else None for k, v in barricade_end.items()},
            "input_values": {k: v[0] if v else None for k, v in barricade_inputs.items()},
            "sequences": sequences.get(len(vehicle_names), {}),
        }
        scene["barricades"].append(barr)

    return scene


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Split the file into named sections based on header lines."""
    sections: dict[str, list[str]] = {}
    current = "header"
    sections[current] = []

    start_count = 0
    end_count = 0
    input_count = 0
    in_sequences = False

    for line in lines:
        stripped = line.strip()

        if stripped == "SEQUENCES":
            current = "sequences"
            sections[current] = []
            in_sequences = True
            continue
        elif stripped == "INPUT VALUES":
            input_count += 1
            current = f"input_values_{input_count}"
            sections[current] = []
            in_sequences = False
            continue
        elif stripped == "START VALUES" and not in_sequences:
            start_count += 1
            current = f"start_values_{start_count}"
            sections[current] = []
            continue
        elif stripped == "END VALUES" and not in_sequences:
            end_count += 1
            current = f"end_values_{end_count}"
            sections[current] = []
            continue

        # Detect barricade header between end_values and start_values
        if current.startswith("end_values") and stripped.startswith("Vehicle:"):
            current = "barricade_header"
            sections[current] = []

        sections.setdefault(current, []).append(line)

    return sections


def _parse_vehicle_headers(lines: list[str]) -> tuple[list[str], int]:
    """Extract vehicle names from the header block."""
    names: list[str] = []
    for line in lines:
        if line.strip().startswith("Vehicle:"):
            parts = line.split("\t")
            for p in parts[1:]:
                p = p.strip()
                if p:
                    names.append(p)
    count = len(names) if names else 2  # default assumption
    return names, count


def _parse_value_block(lines: list[str], expected_cols: int) -> dict[str, list[float | None]]:
    """Parse a START VALUES or END VALUES block into {field_name: [val_per_vehicle]}."""
    result: dict[str, list[float | None]] = {}
    for line in lines:
        if not line.strip() or line.strip().startswith("Vehicle:") or line.strip().startswith("Database:") or line.strip().startswith("RecordID:"):
            continue
        # Pattern: "Label [unit]: \t val1 \t val2"
        m = re.match(r"^(.+?):\s*\t(.+)$", line)
        if not m:
            continue
        label = m.group(1).strip()
        vals_str = m.group(2).split("\t")
        vals: list[float | None] = []
        for v in vals_str:
            v = v.strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(None)
        # Normalise label
        key = _normalise_key(label)
        result[key] = vals
    return result


def _parse_sequences(lines: list[str]) -> dict[int, dict]:
    """Parse SEQUENCES section into per-vehicle sequence data."""
    sequences: dict[int, dict] = {}
    current_veh_idx = -1
    current_veh: dict[str, Any] = {}
    current_block: str | None = None
    current_block_data: dict[str, Any] = {}

    def _save_block():
        nonlocal current_block, current_block_data
        if current_block and current_block_data:
            current_veh.setdefault("actions", []).append(
                {"type": current_block, **current_block_data}
            )
        current_block = None
        current_block_data = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Vehicle header: "1  Honda City:" (possibly with trailing tab)
        m = re.match(r"^(\d+)\s+(.+?):\s*\t?\s*$", stripped)
        if m:
            # Save previous vehicle
            if current_veh_idx >= 0:
                _save_block()
                sequences[current_veh_idx] = current_veh

            current_veh_idx = int(m.group(1)) - 1
            current_veh = {"name": m.group(2).strip(), "actions": []}
            current_block = None
            current_block_data = {}
            continue

        # START VALUES inside sequence
        if stripped == "START VALUES":
            _save_block()
            current_block = "start"
            current_block_data = {}
            continue

        # BRAKE / STEERING blocks
        if stripped == "BRAKE":
            _save_block()
            current_block = "brake"
            current_block_data = {}
            continue

        if stripped == "STEERING":
            _save_block()
            current_block = "steering"
            current_block_data = {}
            continue

        # Skip header-only lines like "Brake force [%]" (no tab-value)
        if stripped.endswith("]") or stripped.endswith("[%]"):
            continue

        # Parse key-value pairs (may be indented like "  Axle 1, left: \t0.00")
        m = re.match(r"^(.+?):\s*\t\s*(.+)$", stripped)
        if m:
            key = m.group(1).strip()
            val_str = m.group(2).strip()
            try:
                val: Any = float(val_str)
            except ValueError:
                val = val_str
            norm_key = _normalise_key(key)
            current_block_data[norm_key] = val
            continue

        # Lines with colon but no tab (e.g. "Brake force [%]")
        # Already handled above, but catch any remainders
        if ":" in stripped:
            pass  # skip

    # Save last vehicle
    if current_veh_idx >= 0:
        _save_block()
        sequences[current_veh_idx] = current_veh

    return sequences


def _parse_input_values(lines: list[str], expected_cols: int) -> dict[str, list[Any]]:
    """Parse INPUT VALUES section with tab-separated columns per vehicle."""
    result: dict[str, list[Any]] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip sub-value lines (indented lines with parenthetical values like "(1.32)")
        if stripped.startswith("("):
            continue

        # Pattern: "Label [unit]: \t val1 \t val2"
        m = re.match(r"^(.+?):\s*\t(.+)$", line)
        if not m:
            continue

        label = m.group(1).strip()
        vals_str = m.group(2).split("\t")

        # Skip metadata rows
        if label in ("Vehicle", "Database", "RecordID", "Characters"):
            continue

        vals: list[Any] = []
        for v in vals_str:
            v = v.strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(v)  # e.g. "No" for ABS
            # skip empty strings

        key = _normalise_key(label)
        result[key] = vals

    return result


def _normalise_key(label: str) -> str:
    """Normalise a PC-Crash field label to a snake_case key."""
    # Remove unit brackets
    label = re.sub(r"\s*\[.*?\]", "", label)
    # Remove parenthetical notes
    label = re.sub(r"\s*\(.*?\)", "", label)
    # Clean up
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = label.strip("_")
    return label
