import json
import os
import subprocess
import sys
import time
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

from flask import Flask, jsonify, render_template_string, request, send_from_directory, session

from agent import choose_best_victim, compute_path_risk, find_nearest_hospital
from csp import csp_assign
from environment import Environment
from fuzzy import fuzzy_inference
from ml_model import get_model_metrics, predict_priority
from search import bfs, dfs, a_star, a_star_risk, greedy_best_first, hill_climbing


PROJECT_DIR = Path(__file__).resolve().parent
SIMULATION_SCRIPT = PROJECT_DIR / "main.py"

app = Flask(__name__)
app.secret_key = os.environ.get("AIDRA_SECRET_KEY", "dev-only-change-me")

ACTIVE_SCENARIO_SESSION_KEY = "active_scenario"


def get_active_scenario():
    """Last successfully built scenario (e.g. after Replan); used for tab pages."""
    return session.get(ACTIVE_SCENARIO_SESSION_KEY, DEFAULT_SCENARIO)


def set_active_scenario(scenario):
    session[ACTIVE_SCENARIO_SESSION_KEY] = normalize_scenario(scenario)


def clear_active_scenario():
    session.pop(ACTIVE_SCENARIO_SESSION_KEY, None)


def dual_pickup_requested():
    """Client wants CSP-aligned dual-victim routing when capacity allows (≤2 per ambulance)."""
    if request.args.get("dual_pickup") in ("1", "true", "yes"):
        return True
    if request.form.get("dual_pickup") in ("on", "1", "true", "yes"):
        return True
    body = request.get_json(silent=True)
    if isinstance(body, dict) and body.get("dual_pickup") in (True, "1", 1, "true", "yes"):
        return True
    return False


def scenario_from_dashboard_json():
    """Parse POST /api/dashboard JSON body; strip UI-only flags from scenario payload."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return None
    return {k: v for k, v in body.items() if k != "dual_pickup"}


DEFAULT_SCENARIO = {
    "size": 10,
    "base": [0, 0],
    "hospitals": [[9, 9], [6, 8]],
    # Four victims so CSP fills both ambulances (2 + 2). V1–V2: same severity, corridor y=2
    # for a clear dual leg (base → V1 → V2 → hospital). V3–V4: other ambulance; single pickup
    # still targets the best-scoring critical (typically nearest: V1).
    "victims": [
        {"id": 1, "x": 2, "y": 2, "severity": "critical"},
        {"id": 2, "x": 6, "y": 2, "severity": "critical"},
        {"id": 3, "x": 2, "y": 7, "severity": "moderate"},
        {"id": 4, "x": 8, "y": 6, "severity": "minor"},
    ],
    "risk_zones": [
        {"x": 1, "y": 1, "level": 2},
        {"x": 5, "y": 3, "level": 2},
        {"x": 4, "y": 7, "level": 3},
        {"x": 7, "y": 8, "level": 2},
        {"x": 0, "y": 4, "level": 2},
        {"x": 1, "y": 4, "level": 2},
    ],
    "blocked_cells": [[4, 4], [5, 4], [3, 5], [5, 6]],
}

MAX_FORM_VICTIMS = 12
MAX_FORM_RISK_ZONES = 12
MAX_FORM_BLOCKED_CELLS = 12


def quiet_call(func, *args, **kwargs):
    """Call existing print-heavy functions without sending logs to the web server."""
    with redirect_stdout(StringIO()):
        return func(*args, **kwargs)


def default_scenario_json():
    return json.dumps(DEFAULT_SCENARIO, indent=2)


def parse_int(value, field):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.")


def parse_optional_position(form, x_key, y_key, field):
    x_value = form.get(x_key, "").strip()
    y_value = form.get(y_key, "").strip()
    if not x_value and not y_value:
        return None
    if not x_value or not y_value:
        raise ValueError(f"{field} needs both x and y values.")
    return [parse_int(x_value, f"{field} x"), parse_int(y_value, f"{field} y")]


def parse_position(value, field, size):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be [x, y].")

    x, y = value
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError(f"{field} coordinates must be integers.")

    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"{field} position {value} is outside the {size}x{size} grid.")

    return x, y


def normalize_scenario(scenario=None):
    scenario = scenario or DEFAULT_SCENARIO
    if not isinstance(scenario, dict):
        raise ValueError("Scenario must be a JSON object.")

    size = scenario.get("size", 10)
    if not isinstance(size, int) or size < 3 or size > 20:
        raise ValueError("size must be an integer between 3 and 20.")

    base = parse_position(scenario.get("base", [0, 0]), "base", size)

    hospitals = [parse_position(item, "hospital", size) for item in scenario.get("hospitals", [])]
    if not hospitals:
        raise ValueError("At least one hospital is required.")

    victims = []
    for index, item in enumerate(scenario.get("victims", []), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"victims[{index}] must be an object.")
        severity = item.get("severity", "minor")
        if severity not in {"critical", "moderate", "minor"}:
            raise ValueError(f"victims[{index}].severity must be critical, moderate, or minor.")
        x, y = parse_position([item.get("x"), item.get("y")], f"victims[{index}]", size)
        victims.append({"id": int(item.get("id", index)), "x": x, "y": y, "severity": severity})
    if not victims:
        raise ValueError("At least one victim is required.")
    victim_cells = {(item["x"], item["y"]): item["id"] for item in victims}
    if base in victim_cells:
        raise ValueError(f"base overlaps victim V{victim_cells[base]} at ({base[0]}, {base[1]}).")
    for index, (hx, hy) in enumerate(hospitals, start=1):
        if (hx, hy) in victim_cells:
            raise ValueError(
                f"hospital[{index}] overlaps victim V{victim_cells[(hx, hy)]} at ({hx}, {hy})."
            )

    risk_zones = []
    for index, item in enumerate(scenario.get("risk_zones", []), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"risk_zones[{index}] must be an object.")
        x, y = parse_position([item.get("x"), item.get("y")], f"risk_zones[{index}]", size)
        if (x, y) in victim_cells:
            raise ValueError(f"risk_zones[{index}] overlaps victim V{victim_cells[(x, y)]} at ({x}, {y}).")
        level = int(item.get("level", 1))
        if level < 0 or level > 3:
            raise ValueError(f"risk_zones[{index}].level must be between 0 and 3.")
        risk_zones.append({"x": x, "y": y, "level": level})

    blocked_cells = [
        parse_position(item, f"blocked_cells[{index}]", size)
        for index, item in enumerate(scenario.get("blocked_cells", []), start=1)
    ]
    for index, (x, y) in enumerate(blocked_cells, start=1):
        if (x, y) in victim_cells:
            raise ValueError(f"blocked_cells[{index}] overlaps victim V{victim_cells[(x, y)]} at ({x}, {y}).")

    important_cells = {base, *hospitals, *((v["x"], v["y"]) for v in victims)}
    blocked_cells = [cell for cell in blocked_cells if cell not in important_cells]

    return {
        "size": size,
        "base": list(base),
        "hospitals": [list(item) for item in hospitals],
        "victims": victims,
        "risk_zones": risk_zones,
        "blocked_cells": [list(item) for item in blocked_cells],
    }


def scenario_from_form(form):
    size = parse_int(form.get("size", 10), "Grid size")
    base = [
        parse_int(form.get("base_x", 0), "Base x"),
        parse_int(form.get("base_y", 0), "Base y"),
    ]

    hospitals = []
    for index in range(1, 7):
        position = parse_optional_position(form, f"hospital_x_{index}", f"hospital_y_{index}", f"Hospital {index}")
        if position:
            hospitals.append(position)

    victims = []
    for index in range(1, MAX_FORM_VICTIMS + 1):
        position = parse_optional_position(form, f"victim_x_{index}", f"victim_y_{index}", f"Victim {index}")
        if position:
            victims.append({
                "id": parse_int(form.get(f"victim_id_{index}", index), f"Victim {index} id"),
                "x": position[0],
                "y": position[1],
                "severity": form.get(f"victim_severity_{index}", "minor"),
            })

    risk_zones = []
    for index in range(1, MAX_FORM_RISK_ZONES + 1):
        position = parse_optional_position(form, f"risk_x_{index}", f"risk_y_{index}", f"Risk zone {index}")
        if position:
            risk_zones.append({
                "x": position[0],
                "y": position[1],
                "level": parse_int(form.get(f"risk_level_{index}", 1), f"Risk zone {index} level"),
            })

    blocked_cells = []
    for index in range(1, MAX_FORM_BLOCKED_CELLS + 1):
        position = parse_optional_position(form, f"blocked_x_{index}", f"blocked_y_{index}", f"Blocked cell {index}")
        if position:
            blocked_cells.append(position)

    return normalize_scenario({
        "size": size,
        "base": base,
        "hospitals": hospitals,
        "victims": victims,
        "risk_zones": risk_zones,
        "blocked_cells": blocked_cells,
    })


def pad_rows(rows, total, empty_factory):
    padded = list(rows[:total])
    while len(padded) < total:
        padded.append(empty_factory(len(padded) + 1))
    return padded


def scenario_form_data(scenario):
    scenario = normalize_scenario(scenario)
    return {
        "size": scenario["size"],
        "base_x": scenario["base"][0],
        "base_y": scenario["base"][1],
        "hospitals": pad_rows(
            [{"x": x, "y": y} for x, y in scenario["hospitals"]],
            6,
            lambda _: {"x": "", "y": ""},
        ),
        "victims": pad_rows(
            scenario["victims"],
            MAX_FORM_VICTIMS,
            lambda index: {"id": index, "x": "", "y": "", "severity": "minor"},
        ),
        "risk_zones": pad_rows(
            scenario["risk_zones"],
            MAX_FORM_RISK_ZONES,
            lambda _: {"x": "", "y": "", "level": 1},
        ),
        "blocked_cells": pad_rows(
            [{"x": x, "y": y} for x, y in scenario["blocked_cells"]],
            MAX_FORM_BLOCKED_CELLS,
            lambda _: {"x": "", "y": ""},
        ),
    }


def create_environment(scenario=None):
    scenario = normalize_scenario(scenario)
    env = Environment(size=scenario["size"])
    env.set_base(*scenario["base"])

    for hospital in scenario["hospitals"]:
        env.add_hospital(*hospital)

    for victim in scenario["victims"]:
        env.add_victim(victim["id"], victim["x"], victim["y"], victim["severity"])

    for risk_zone in scenario["risk_zones"]:
        env.add_risk_zone(risk_zone["x"], risk_zone["y"], risk_zone["level"])

    for blocked_cell in scenario["blocked_cells"]:
        env.block_cell(*blocked_cell)

    return env


def serialize_grid(env, rescue_path=None):
    rescue_path = set(rescue_path or [])
    cells = []

    for x in range(env.size):
        row = []
        for y in range(env.size):
            cell = env.grid[x][y]
            terrain_type = "empty"
            terrain_label = ""

            if (x, y) == env.base:
                terrain_type = "base"
                terrain_label = "B"
            elif cell.is_hospital:
                terrain_type = "hospital"
                terrain_label = "H"
            elif cell.is_blocked:
                terrain_type = "blocked"
                terrain_label = "X"
            elif cell.has_victim:
                terrain_type = "victim"
                terrain_label = "V"
            elif cell.risk_level > 0:
                terrain_type = "risk"
                terrain_label = f"R{cell.risk_level}"

            on_path = (x, y) in rescue_path

            row.append({
                "x": x,
                "y": y,
                "type": "path" if on_path else terrain_type,
                "label": "*" if on_path else terrain_label,
                "terrain_type": terrain_type,
                "terrain_label": terrain_label,
                "on_path": on_path,
                "risk": cell.risk_level,
            })
        cells.append(row)

    return cells


def find_open_cell(env):
    for x in range(env.size):
        for y in range(env.size):
            cell = env.grid[x][y]
            if not cell.is_blocked and not cell.has_victim and not cell.is_hospital and (x, y) != env.base:
                return x, y
    return None


def _temporary_block_other_victims(env, allowed_victim_ids):
    """Block victim cells except those in allowed_victim_ids (so routes do not cross other pickups)."""
    allowed = set(allowed_victim_ids)
    blocked = []
    for v in env.victims:
        if v.id in allowed or v.rescued:
            continue
        cell = env.grid[v.x][v.y]
        if not cell.is_blocked:
            cell.is_blocked = True
            blocked.append((v.x, v.y))
    return blocked


def _restore_temporary_blocks(env, blocked_coords):
    for x, y in blocked_coords:
        env.grid[x][y].is_blocked = False


def _worse_severity(severity_a, severity_b):
    order = {"critical": 3, "moderate": 2, "minor": 1}
    return severity_a if order.get(severity_a, 0) >= order.get(severity_b, 0) else severity_b


def _csp_dual_pair(env):
    """Return two victims assigned to the same ambulance (CSP), if any, else (None, None)."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        assignment, _ = csp_assign(env.victims)
    by_amb = {}
    for v in env.victims:
        amb = assignment.get(v.id)
        if amb is not None:
            by_amb.setdefault(amb, []).append(v)
    sev_key = {"critical": 3, "moderate": 2, "minor": 1}
    for amb in ("Ambulance 1", "Ambulance 2"):
        grp = by_amb.get(amb, [])
        if len(grp) >= 2:
            grp.sort(key=lambda x: (-sev_key.get(x.severity, 0), x.id))
            return grp[0], grp[1]
    return None, None


def _try_dual_rescue_paths(env, start, va, vb):
    """Base -> victim A -> victim B -> nearest hospital; avoid other victim cells."""
    allowed = {va.id, vb.id}
    lifted = _temporary_block_other_victims(env, allowed)
    try:
        g1, g2 = (va.x, va.y), (vb.x, vb.y)
        p1 = quiet_call(a_star, env, start, g1)
        p2 = quiet_call(a_star, env, g1, g2)
        hospital, p3 = quiet_call(find_nearest_hospital, env, g2)
        if not p1 or not p2 or not p3:
            return None
        leg = p1 + p2[1:]
        full_path = leg + p3[1:]
        return {
            "hospital": hospital,
            "hospital_path": p3,
            "leg": leg,
            "full_path": full_path,
            "p1": p1,
            "p2": p2,
        }
    finally:
        _restore_temporary_blocks(env, lifted)


def build_dashboard_data(scenario=None, dual_pickup=False):
    scenario = normalize_scenario(scenario)
    env = create_environment(scenario)
    start = env.base
    ml_metrics = get_model_metrics()

    victim_b = None
    dual_pack = None
    va = vb = None
    if dual_pickup:
        va, vb = _csp_dual_pair(env)
        if va is not None and vb is not None:
            dual_pack = _try_dual_rescue_paths(env, start, va, vb)

    use_dual = dual_pack is not None
    sev_ml = None
    base_to_victim_path = None
    dual_mid_leg = None

    if use_dual:
        victim = va
        victim_b = vb
        hospital = dual_pack["hospital"]
        hospital_path = dual_pack["hospital_path"]
        goal = (victim_b.x, victim_b.y)
        lifted = _temporary_block_other_victims(env, {victim.id, victim_b.id})
        try:
            leg0 = dual_pack["leg"]
            distance = len(leg0)
            risk = compute_path_risk(env, leg0)
            sev_ml = _worse_severity(victim.severity, victim_b.severity)

            knn_pred, nb_pred, dt_pred = predict_priority(distance, risk, sev_ml)
            labels = ["LOW", "MEDIUM", "HIGH"]
            ml_votes = [int(knn_pred), int(nb_pred), int(dt_pred)]
            final_ml = max(set(ml_votes), key=ml_votes.count)

            blockage_probability = 0.6
            fuzzy_result = quiet_call(fuzzy_inference, distance, risk, sev_ml, blockage_probability)
            fuzzy_log = (
                "--- FUZZY LOGIC INPUT ---\n"
                f"Distance: {distance}\n"
                f"Risk: {risk}\n"
                f"Severity: {sev_ml}\n"
                f"Blockage Probability: {blockage_probability} -> possible\n\n"
                "--- FUZZY OUTPUT ---\n"
                f"Decision: {fuzzy_result}"
            )
            selected_path = leg0
            if fuzzy_result == "HIGH":
                risk_path = a_star_risk(env, start, goal, risk_weight=3)
                if risk_path is not None:
                    selected_path = risk_path
            hospital, hospital_path = quiet_call(find_nearest_hospital, env, goal)
            full_path = selected_path + hospital_path[1:]

            algorithm_paths = {
                "BFS": bfs(env, start, goal),
                "DFS": dfs(env, start, goal),
                "Greedy": greedy_best_first(env, start, goal),
                "Hill Climbing": hill_climbing(env, start, goal),
                "A*": a_star(env, start, goal),
                "Risk-Aware A*": a_star_risk(env, start, goal, risk_weight=3),
            }
            algorithm_comparison = []
            algorithm_profiles = {
                "BFS": {"type": "Optimal", "speed": "Slow", "optimality": "Optimal", "safety": "Ignores risk", "use_case": "Guaranteed shortest path"},
                "DFS": {"type": "Non-Optimal", "speed": "Fast", "optimality": "Non-optimal", "safety": "Ignores risk", "use_case": "Quick exploration"},
                "Greedy": {"type": "Approximate", "speed": "Very fast", "optimality": "Approximate", "safety": "Ignores risk", "use_case": "Speed-critical situations"},
                "Hill Climbing": {"type": "Local Search", "speed": "Fast", "optimality": "Local minima risk", "safety": "Ignores risk", "use_case": "Simple terrains only"},
                "A*": {"type": "Optimal", "speed": "Balanced", "optimality": "Optimal", "safety": "Partial", "use_case": "Best general purpose"},
                "Risk-Aware A*": {"type": "Safe-Optimal", "speed": "Balanced", "optimality": "Near-optimal", "safety": "Risk-aware", "use_case": "Safety-critical rescue"},
            }
            analytics_rows = []
            path_visualizations = []
            for name, algo_path in algorithm_paths.items():
                profile = algorithm_profiles.get(name, {})
                risk_score = compute_path_risk(env, algo_path) if algo_path else 0
                path_len = len(algo_path) if algo_path else 0
                algorithm_comparison.append({
                    "name": name,
                    "length": path_len,
                    "risk": risk_score,
                    "reached": bool(algo_path and algo_path[-1] == goal),
                    "type": profile.get("type", "-"),
                })
                analytics_rows.append({
                    "name": name,
                    "length": path_len,
                    "risk": risk_score,
                    "speed": profile.get("speed", "-"),
                    "optimality": profile.get("optimality", "-"),
                    "safety": profile.get("safety", "-"),
                    "use_case": profile.get("use_case", "-"),
                })
                preview = " -> ".join([f"({x},{y})" for x, y in (algo_path or [])[:12]])
                if algo_path and len(algo_path) > 12:
                    preview = f"{preview} -> ..."
                if not preview:
                    preview = "No reachable route"
                path_visualizations.append({"name": name, "preview": preview})
        finally:
            _restore_temporary_blocks(env, lifted)

        victim.rescued = True
        victim_b.rescued = True
        base_to_victim_path = dual_pack["p1"]
        dual_mid_leg = dual_pack["p2"]
    else:
        victim, path = quiet_call(choose_best_victim, env, start)
        if victim is None or path is None:
            raise ValueError("No reachable victim found for this scenario.")

        goal = (victim.x, victim.y)

        lifted = _temporary_block_other_victims(env, {victim.id})
        try:
            path_respecting = quiet_call(a_star, env, start, goal)
            if path_respecting is not None:
                path = path_respecting

            distance = len(path)
            risk = compute_path_risk(env, path)
            sev_ml = victim.severity

            knn_pred, nb_pred, dt_pred = predict_priority(distance, risk, victim.severity)
            labels = ["LOW", "MEDIUM", "HIGH"]
            ml_votes = [int(knn_pred), int(nb_pred), int(dt_pred)]
            final_ml = max(set(ml_votes), key=ml_votes.count)

            blockage_probability = 0.6
            fuzzy_result = quiet_call(fuzzy_inference, distance, risk, victim.severity, blockage_probability)
            fuzzy_log = (
                "--- FUZZY LOGIC INPUT ---\n"
                f"Distance: {distance}\n"
                f"Risk: {risk}\n"
                f"Severity: {victim.severity}\n"
                f"Blockage Probability: {blockage_probability} -> possible\n\n"
                "--- FUZZY OUTPUT ---\n"
                f"Decision: {fuzzy_result}"
            )
            selected_path = path
            if fuzzy_result == "HIGH":
                risk_path = a_star_risk(env, start, goal, risk_weight=3)
                if risk_path is not None:
                    selected_path = risk_path

            hospital, hospital_path = quiet_call(find_nearest_hospital, env, goal)
            if hospital is None or hospital_path is None:
                raise ValueError("No reachable hospital found from the selected victim.")

            full_path = selected_path + hospital_path[1:]

            algorithm_paths = {
                "BFS": bfs(env, start, goal),
                "DFS": dfs(env, start, goal),
                "Greedy": greedy_best_first(env, start, goal),
                "Hill Climbing": hill_climbing(env, start, goal),
                "A*": a_star(env, start, goal),
                "Risk-Aware A*": a_star_risk(env, start, goal, risk_weight=3),
            }
            algorithm_comparison = []
            algorithm_profiles = {
                "BFS": {"type": "Optimal", "speed": "Slow", "optimality": "Optimal", "safety": "Ignores risk", "use_case": "Guaranteed shortest path"},
                "DFS": {"type": "Non-Optimal", "speed": "Fast", "optimality": "Non-optimal", "safety": "Ignores risk", "use_case": "Quick exploration"},
                "Greedy": {"type": "Approximate", "speed": "Very fast", "optimality": "Approximate", "safety": "Ignores risk", "use_case": "Speed-critical situations"},
                "Hill Climbing": {"type": "Local Search", "speed": "Fast", "optimality": "Local minima risk", "safety": "Ignores risk", "use_case": "Simple terrains only"},
                "A*": {"type": "Optimal", "speed": "Balanced", "optimality": "Optimal", "safety": "Partial", "use_case": "Best general purpose"},
                "Risk-Aware A*": {"type": "Safe-Optimal", "speed": "Balanced", "optimality": "Near-optimal", "safety": "Risk-aware", "use_case": "Safety-critical rescue"},
            }
            analytics_rows = []
            path_visualizations = []
            for name, algo_path in algorithm_paths.items():
                profile = algorithm_profiles.get(name, {})
                risk_score = compute_path_risk(env, algo_path) if algo_path else 0
                path_len = len(algo_path) if algo_path else 0
                algorithm_comparison.append({
                    "name": name,
                    "length": path_len,
                    "risk": risk_score,
                    "reached": bool(algo_path and algo_path[-1] == goal),
                    "type": profile.get("type", "-"),
                })
                analytics_rows.append({
                    "name": name,
                    "length": path_len,
                    "risk": risk_score,
                    "speed": profile.get("speed", "-"),
                    "optimality": profile.get("optimality", "-"),
                    "safety": profile.get("safety", "-"),
                    "use_case": profile.get("use_case", "-"),
                })
                preview = " -> ".join([f"({x},{y})" for x, y in (algo_path or [])[:12]])
                if algo_path and len(algo_path) > 12:
                    preview = f"{preview} -> ..."
                if not preview:
                    preview = "No reachable route"
                path_visualizations.append({"name": name, "preview": preview})
        finally:
            _restore_temporary_blocks(env, lifted)

        victim.rescued = True
        base_to_victim_path = selected_path
        dual_mid_leg = None

    selected_ids = {victim.id}
    if victim_b is not None:
        selected_ids.add(victim_b.id)

    scenario_env = create_environment(scenario)
    current_position = full_path[len(full_path) // 2]
    blockage_cell = (current_position[0] + 1, current_position[1])
    if scenario_env.is_valid(*blockage_cell):
        scenario_env.block_cell(*blockage_cell)
    risk_cell = (current_position[0], current_position[1] + 1)
    if scenario_env.is_valid(*risk_cell):
        scenario_env.add_risk_zone(*risk_cell, 3)
    new_victim_id = max(item.id for item in env.victims) + 1
    route_cells = set(full_path)
    new_victim_cell = None
    for x in range(env.size):
        for y in range(env.size):
            cell = env.grid[x][y]
            if (
                not cell.is_blocked
                and not cell.has_victim
                and not cell.is_hospital
                and (x, y) != env.base
                and (x, y) not in route_cells
            ):
                new_victim_cell = (x, y)
                break
        if new_victim_cell is not None:
            break
    if new_victim_cell:
        scenario_env.add_victim(new_victim_id, *new_victim_cell, "critical")
        env.add_victim(new_victim_id, *new_victim_cell, "critical")
        new_selected_victim, new_victim_path = quiet_call(choose_best_victim, env, start)
        new_victim_event = f"Live update: critical victim added at {new_victim_cell}"
        new_victim_impact = (
            "Live mission update: "
            f"added a new critical victim at {new_victim_cell}; "
            f"the planner now targets V{new_selected_victim.id} "
            f"(path length {len(new_victim_path)})."
        )
    else:
        new_victim_cell = None
        new_victim_event = "No open cell available for adding a new victim."
        new_victim_impact = "No open cell was available for adding a new victim."

    mission_grid = serialize_grid(env)
    mission_victims = [
        {
            "id": item.id,
            "x": item.x,
            "y": item.y,
            "severity": item.severity,
            "rescued": item.rescued,
            "selected": item.id in selected_ids,
        }
        for item in env.victims
    ]

    mission_hospitals = [{"x": x, "y": y, "selected": (x, y) == hospital} for x, y in env.hospitals]
    csp_trace_buffer = StringIO()
    with redirect_stdout(csp_trace_buffer):
        assignment, backtrack_count = csp_assign(env.victims)
    csp_trace_text = csp_trace_buffer.getvalue().strip()
    csp_trace_lines = [line for line in csp_trace_text.splitlines() if line.strip()]
    assigned_ids = set(assignment.keys())
    assignment_rows = []
    for item in mission_victims:
        ambulance = assignment.get(item["id"])
        assignment_rows.append({
            "victim": item["id"],
            "severity": item["severity"],
            "location": f'({item["x"]}, {item["y"]})',
            "ambulance": ambulance if ambulance is not None else "-",
            "assigned": ambulance is not None,
        })

    ambulance_loads = {}
    for ambulance in assignment.values():
        ambulance_loads[ambulance] = ambulance_loads.get(ambulance, 0) + 1
    ambulance_load_rows = [{"ambulance": key, "count": value} for key, value in sorted(ambulance_loads.items())]
    return {
        "scenario": scenario,
        "grid": mission_grid,
        "size": env.size,
        "base": env.base,
        "victims": mission_victims,
        "hospitals": mission_hospitals,
        "decision": {
            "selected_victim": victim.id,
            "selected_victims": [victim.id, victim_b.id] if victim_b is not None else [victim.id],
            "dual_pickup": victim_b is not None,
            "selected_severity": sev_ml,
            "selected_hospital": hospital,
            "path_length": len(full_path),
            "total_risk": compute_path_risk(env, full_path),
            "saved": 2 if victim_b is not None else 1,
            "utilization": min(1.0, len(selected_ids) / 2.0),
        },
        "ml": {
            "knn": labels[knn_pred],
            "naive_bayes": labels[nb_pred],
            "decision_tree": labels[dt_pred],
            "final": labels[final_ml],
            "final_priority": f"{labels[final_ml]} PRIORITY",
            "metrics": ml_metrics,
        },
        "fuzzy": {
            "blockage_probability": blockage_probability,
            "decision": fuzzy_result,
            "distance": distance,
            "risk": risk,
            "severity": sev_ml,
            "log": fuzzy_log,
        },
        "paths": {
            "base_to_victim": base_to_victim_path,
            "victim_to_hospital": hospital_path,
            "full": full_path,
            "second_leg": dual_mid_leg,
            "dual_pickup": victim_b is not None,
        },
        "algorithms": algorithm_comparison,
        "analytics": {
            "comparison": analytics_rows,
            "paths": path_visualizations,
        },
        "scenarios": [
            {
                "name": "Road Blockage",
                "event": f"Blocked road at {blockage_cell}",
                "impact": "Agent recomputes a feasible path.",
            },
            {
                "name": "Risk Change",
                "event": f"Risk increased at {risk_cell}",
                "impact": "Risk-aware A* prefers safer movement.",
            },
            {
                "name": "New Victim",
                "event": new_victim_event,
                "impact": new_victim_impact,
            },
            {
                "name": "Resource Depletion",
                "event": "Only two ambulances with max two victims each.",
                "impact": f"{len(assignment)} victims assigned, backtracks: {backtrack_count}.",
            },
        ],
        "assignments": assignment_rows,
        "csp": {
            "assigned_count": len(assignment),
            "unassigned_count": len(mission_victims) - len(assignment),
            "total_victims": len(mission_victims),
            "backtrack_count": backtrack_count,
            "ambulance_count_used": len(ambulance_load_rows),
            "ambulance_loads": ambulance_load_rows,
            "all_assigned": len(assigned_ids) == len(mission_victims),
            "trace_text": csp_trace_text,
            "trace_lines": csp_trace_lines,
        },
    }


def run_simulation(timeout_seconds=60):
    """Run the existing console simulation and return output for API/HTML clients."""
    started_at = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        completed = subprocess.run(
            [sys.executable, str(SIMULATION_SCRIPT)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - started_at, 3)
        output = (exc.stdout or "") + (exc.stderr or "")
        return {
            "ok": False,
            "exit_code": None,
            "duration_seconds": duration,
            "output": output,
            "lines": output.splitlines(),
            "error": f"Simulation timed out after {timeout_seconds} seconds.",
        }

    duration = round(time.perf_counter() - started_at, 3)
    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n--- STDERR ---\n{completed.stderr}"

    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "output": output,
        "lines": output.splitlines(),
        "error": None if completed.returncode == 0 else "Simulation exited with an error.",
    }


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIDRA Command Center</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      --bg: #f3f6fb;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --line: #dbe6f3;
      --blue: #3b82f6;
      --green: #22c55e;
      --orange: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
      --text: #1f2937;
      --muted: #64748b;
      --cell-size: 28px;
      --cell-gap: 2px;
      --grid-step: calc(var(--cell-size) + var(--cell-gap));
      --grid-pad: 4px;
    }
    * { box-sizing: border-box; }
    body { background: var(--bg); color: var(--text); margin: 0; overflow-x: hidden; }
    a, button { border: 0; color: inherit; cursor: pointer; font: inherit; text-decoration: none; }
    .app { min-height: 100vh; }
    .top-nav {
      background: #ffffff;
      border-right: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 20px;
      height: 100vh;
      left: 0;
      overflow: hidden;
      padding: 20px 14px;
      position: fixed;
      top: 0;
      width: 240px;
      z-index: 20;
    }
    .brand { align-items: center; display: flex; gap: 12px; }
    .logo { align-items: center; background: transparent; border: 0; border-radius: 5px; display: grid; height: 2.9rem; overflow: hidden; place-items: center; width: 2.9rem; }
    .logo img { height: 100%; object-fit: contain; width: 100%; }
    .brand h1 { font-size: 12px; line-height: 1; margin: 0; }
    .brand span, .muted { color: var(--muted); font-size: 10px; }
    .tabs { display: flex; flex-direction: column; gap: 8px; }
    .tab { border-radius: 10px; color: #64748b; font-size: 14px; font-weight: 700; padding: 11px 12px; }
    .tab.active { background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; }
    .workspace { display: grid; gap: 12px; grid-template-columns: minmax(500px, .9fr) minmax(520px, 1.1fr); margin-left: 240px; padding: 16px; width: calc(100% - 240px); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; }
    .panel-head { align-items: center; border-bottom: 1px solid #e5edf7; color: #475569; display: flex; font-size: 12px; font-weight: 800; justify-content: space-between; letter-spacing: .08em; padding: 10px 12px; text-transform: uppercase; }
    .pill { border-radius: 999px; display: inline-block; font-size: 10px; font-weight: 900; padding: 4px 8px; text-transform: uppercase; }
    .open-config { align-items: center; background: #2563eb; border-radius: 10px; color: #fff; display: inline-flex; font-size: 18px; font-weight: 900; height: 44px; justify-content: center; width: 44px; }
    .config-form { display: grid; gap: 12px; }
    .form-section { background: var(--panel-2); border: 1px solid #dbe6f3; border-radius: 10px; padding: 10px; }
    .form-section h3 { color: #1d4ed8; font-size: 12px; letter-spacing: .06em; margin: 0 0 8px; text-transform: uppercase; }
    .row { display: grid; gap: 6px; grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 6px; }
    .row.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    label { color: #475569; display: grid; font-size: 10px; gap: 3px; }
    input, select { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 7px; color: #0f172a; min-width: 0; padding: 7px 6px; width: 100%; }
    .replan { background: linear-gradient(135deg, #3b82f6, #6366f1); border-radius: 9px; color: white; font-weight: 900; padding: 12px; width: 100%; }
    .error { background: rgba(220,38,38,.18); border: 1px solid rgba(248,113,113,.35); border-radius: 9px; color: #fecaca; font-size: 12px; padding: 10px; }
    .main { display: grid; gap: 10px; grid-template-rows: auto 1fr; min-width: 0; }
    .runbar { align-items: center; display: flex; gap: 10px; padding: 10px; }
    .runbar .dual-opt { align-items: center; color: var(--muted); display: flex; font-size: 12px; font-weight: 600; gap: 6px; margin-right: 4px; user-select: none; white-space: nowrap; }
    .runbar .dual-opt input { accent-color: var(--blue); height: 14px; width: 14px; }
    .run.icon-btn { border-radius: 12px; color: #fff; font-size: 18px; font-weight: 900; height: 44px; width: 44px; }
    .start { background: #16a34a; }
    .pause { background: #f59e0b; }
    .reset { background: #475569; }
    .map-panel { min-height: 640px; padding: 12px; }
    .objective { align-items: center; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 9px; color: #c2410c; display: flex; font-size: 12px; font-weight: 900; justify-content: space-between; margin-bottom: 10px; padding: 9px 10px; }
    .map-title { align-items: center; display: flex; justify-content: space-between; margin: 4px 0 9px; }
    .map-title strong { color: #2563eb; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; }
    .grid-wrap { background: #ffffff; border: 1px solid #dbe6f3; border-radius: 8px; overflow: auto; padding: 12px; }
    .grid {
      align-content: start;
      display: grid;
      grid-auto-rows: var(--cell-size);
      column-gap: var(--cell-gap);
      row-gap: 1px;
      grid-template-columns: repeat(10, var(--cell-size));
      justify-content: center;
      min-height: 440px;
      min-width: 360px;
      padding: var(--grid-pad);
    }
    .cell {
      align-items: center;
      border: 1px solid #cbd5e1;
      border-radius: 5px;
      color: #0f172a;
      display: flex;
      font-size: 10px;
      font-weight: 900;
      height: var(--cell-size);
      justify-content: center;
      line-height: 1;
      overflow: hidden;
      position: relative;
      width: var(--cell-size);
    }
    .cell small { display: none; }
    .empty { background: #ffffff; }
    .base { background: #2563eb; }
    .hospital { background: #166534; }
    .victim { background: #dc2626; border-radius: 50%; }
    .blocked { background: #334155; color: #e2e8f0; }
    .risk { background: rgba(127,29,29,.9); }
    .path { background: #f59e0b; border-radius: 50%; color: #111827; }
    .route-marker { outline: 3px solid #f59e0b; outline-offset: -3px; }
    .legend { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 10px; }
    .legend span { color: #475569; font-size: 11px; }
    .dot { border-radius: 50%; display: inline-block; height: 9px; margin-right: 5px; width: 9px; }
    .right { display: grid; gap: 10px; grid-template-columns: 1.2fr 1fr; }
    .stack { display: flex; flex-direction: column; gap: 10px; }
    .ai-card { padding: 13px; }
    .ai-card h2 { color: #7c3aed; font-size: 14px; letter-spacing: .08em; margin: 0 0 10px; text-transform: uppercase; }
    .formula { color: #6d28d9; font-family: Consolas, monospace; font-size: 12px; line-height: 1.45; }
    .metric-list { display: grid; gap: 10px; }
    .metric { align-items: center; display: flex; justify-content: space-between; }
    .metric span { color: #64748b; font-size: 12px; }
    .metric strong { color: #111827; font-size: 13px; }
    table { border-collapse: collapse; width: 100%; }
    th { color: #2563eb; font-size: 11px; letter-spacing: .08em; padding: 8px 6px; text-align: left; text-transform: uppercase; }
    td { border-top: 1px solid #e2e8f0; color: #1f2937; font-size: 12px; padding: 8px 6px; }
    .sev-critical { color: #f87171; font-weight: 900; }
    .sev-moderate { color: #fbbf24; font-weight: 900; }
    .sev-minor { color: #4ade80; font-weight: 900; }
    .waiting { background: rgba(245,158,11,.16); color: #fbbf24; }
    .saved { background: rgba(34,197,94,.16); color: #86efac; }
    .surv { background: #dbeafe; border-radius: 999px; height: 7px; overflow: hidden; width: 58px; }
    .surv i { background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e); display: block; height: 100%; }
    .priority { padding: 14px; }
    .priority h2 { color: #0284c7; font-size: 14px; letter-spacing: .08em; margin: 0 0 10px; text-transform: uppercase; }
    .priority p { color: #334155; font-size: 12px; line-height: 1.55; margin: 0 0 10px; }
    .log { background: #f8fafc; border: 1px solid #dbe6f3; border-radius: 8px; color: #475569; font-family: Consolas, monospace; height: 82px; padding: 10px; }
    .loading {
      align-items: center;
      backdrop-filter: blur(8px);
      background: rgba(241, 245, 249, .75);
      display: none;
      inset: 0;
      justify-content: center;
      position: fixed;
      z-index: 50;
    }
    .loading.active { display: flex; }
    .modal {
      align-items: center;
      backdrop-filter: blur(9px);
      background: rgba(241, 245, 249, .8);
      display: none;
      inset: 0;
      justify-content: center;
      padding: 22px;
      position: fixed;
      z-index: 45;
    }
    .modal.active { display: flex; }
    .modal-card {
      background: #ffffff;
      border: 1px solid #bfdbfe;
      border-radius: 18px;
      max-height: 88vh;
      max-width: 1120px;
      overflow: auto;
      padding: 18px;
      width: min(1120px, 96vw);
    }
    .modal-head { align-items: center; display: flex; gap: 12px; justify-content: space-between; margin-bottom: 14px; }
    .modal-head h2 { color: #1d4ed8; font-size: 22px; margin: 0; }
    .modal-head p { color: #64748b; margin: 4px 0 0; }
    .close-modal { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; color: #334155; font-weight: 900; padding: 9px 12px; }
    .modal .config-form { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .modal .replan { grid-column: 1 / -1; }
    .modal #formError { grid-column: 1 / -1; }
    .loader-box {
      background: #ffffff;
      border: 1px solid #bfdbfe;
      border-radius: 18px;
      min-width: 280px;
      padding: 22px;
      text-align: center;
    }
    .spinner {
      animation: spin 1s linear infinite;
      border: 4px solid rgba(96,165,250,.18);
      border-top-color: #60a5fa;
      border-radius: 50%;
      height: 48px;
      margin: 0 auto 14px;
      width: 48px;
    }
    .loader-box strong { display: block; font-size: 18px; margin-bottom: 6px; }
    .loader-box span { color: #64748b; font-size: 13px; }
    button:disabled { cursor: wait; filter: grayscale(.25); opacity: .65; }
    .grid.running .path { animation: pulseRoute .8s ease-in-out infinite alternate; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulseRoute { from { transform: scale(.9); } to { transform: scale(1.08); } }
    @media (max-width: 1180px) { .workspace { grid-template-columns: 1fr; margin-left: 0; width: 100%; } .right { grid-template-columns: 1fr; } .top-nav { height: auto; position: static; width: auto; } .tabs { flex-direction: row; flex-wrap: wrap; } }
  </style>
</head>
<body>
  <main class="app">
    <header class="top-nav">
      <div class="brand">
        <div class="logo"><img src="/logo.png" alt="AIDRA logo"></div>
        <div><h1>AIDRA</h1><span>Adaptive Intelligent Disaster Response Agent</span></div>
      </div>
      <nav class="tabs">
        <a class="tab active" href="/">Live Simulation</a>
        <a class="tab" href="/search-trace">Search Trace</a>
        <a class="tab" href="/csp-solver">CSP Solver</a>
        <a class="tab" href="/ml-studio">ML Studio</a>
        <a class="tab" href="/analytics">Analytics</a>
      </nav>
    </header>

    <section class="workspace">
      <section class="main">
        <section class="panel runbar">
          <label class="dual-opt" title="When CSP puts two victims on the same ambulance, route base → first → second → hospital (max 2 per trip).">
            <input type="checkbox" id="dualPickupOpt" name="dual_pickup">
            Dual pickup
          </label>
          <button class="run icon-btn start" id="startBtn" type="button" title="Start" aria-label="Start">▶</button>
          <button class="run icon-btn pause" id="pauseBtn" type="button" title="Pause" aria-label="Pause">⏸</button>
          <button class="run icon-btn reset" id="resetBtn" type="button" title="Reset" aria-label="Reset">↺</button>
          <button class="open-config" id="openConfigBtn" type="button" title="Layout Editor" aria-label="Layout editor">⚙</button>
        </section>

        <section class="panel map-panel">
          <div class="objective">
            <span>Objective: Minimize risk</span>
            <span>h = risk x distance</span>
          </div>
          <div class="map-title">
            <strong>Grid Map</strong>
            <span class="muted">Hover cells for coordinates</span>
          </div>
          <div class="grid-wrap">
            <div class="grid" id="gridMap" style="grid-template-columns: repeat({{ data.size }}, var(--cell-size));">
              {% for row in data.grid %}
                {% for cell in row %}
                  <div class="cell {{ cell.terrain_type }}" title="({{ cell.x }}, {{ cell.y }}) risk {{ cell.risk }}">
                    {{ cell.terrain_label }}
                    <small>{{ cell.x }},{{ cell.y }}</small>
                  </div>
                {% endfor %}
              {% endfor %}
            </div>
          </div>
          <div class="legend">
            <span><i class="dot base"></i>Road/Base</span>
            <span><i class="dot risk"></i>Fire Zone</span>
            <span><i class="dot blocked"></i>Blocked</span>
            <span><i class="dot hospital"></i>Safe Area</span>
            <span><i class="dot victim"></i>Critical/Moderate/Minor</span>
            <span><i class="dot path"></i>Route</span>
          </div>
        </section>
      </section>

      <aside class="right">
        <div class="stack">
          <section class="panel ai-card">
            <h2>Fuzzy On</h2>
            <div class="formula" id="fuzzyFormula">Not started. Click Start or Replan to run fuzzy inference.</div>
          </section>
          <section class="panel ai-card">
            <h2>ML On</h2>
            <div class="formula" id="mlFormula">Not started. Click Start or Replan to run ML evaluation.</div>
            <div class="metric-list" id="mlMetrics">
              <div class="metric"><span>kNN</span><strong>--</strong></div>
              <div class="metric"><span>Naive Bayes</span><strong>--</strong></div>
              <div class="metric"><span>Decision Tree</span><strong>--</strong></div>
              <div class="metric"><span>Final Priority</span><strong>--</strong></div>
            </div>
          </section>
          <section class="panel ai-card">
            <h2>KPIs</h2>
            <div class="metric-list" id="kpiMetrics">
              <div class="metric"><span>Victims Saved</span><strong>--</strong></div>
              <div class="metric"><span>Avg Rescue Time</span><strong>--</strong></div>
              <div class="metric"><span>Risk Exposure</span><strong>--</strong></div>
              <div class="metric"><span>Resource Util</span><strong>--</strong></div>
            </div>
          </section>
        </div>

        <div class="stack">
          <section class="panel">
            <div class="panel-head">Victims</div>
            <table>
              <thead><tr><th>ID</th><th>SEV</th><th>Status</th><th>Surv</th><th>ML</th><th>ETA</th></tr></thead>
              <tbody id="victimRows">
                <tr><td colspan="6" class="muted">Not started. Click Start to evaluate victims.</td></tr>
              </tbody>
            </table>
          </section>

          <section class="panel priority" id="priorityPanel">
            <h2>Priority</h2>
            <p class="muted">No priority decision yet. Click Start or Replan.</p>
          </section>

          <section class="panel priority" id="algorithmPanel">
            <h2>Search Algorithms</h2>
            <p class="muted">No algorithm comparison yet.</p>
          </section>

          <section class="panel priority">
            <h2>Log</h2>
            <div class="log" id="eventLog">Simulation ready. Use Layout Editor to customize the map.</div>
          </section>
        </div>
      </aside>
    </section>
  </main>
  <div class="modal" id="configModal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="configTitle">
      <div class="modal-head">
        <div>
          <h2 id="configTitle">AI Configuration</h2>
          <p>Edit the disaster layout, then click Replan to update the simulation without refreshing.</p>
        </div>
        <button class="close-modal" id="closeConfigBtn" type="button">Close</button>
      </div>
      <form class="config-form" id="scenarioForm" method="post" action="/dashboard">
        <div class="form-section">
          <h3>Map</h3>
          <div class="row">
            <label>Size<input name="size" type="number" min="3" max="20" value="{{ form.size }}"></label>
            <label>Base X<input name="base_x" type="number" min="0" value="{{ form.base_x }}"></label>
            <label>Base Y<input name="base_y" type="number" min="0" value="{{ form.base_y }}"></label>
          </div>
        </div>
        <div class="form-section">
          <h3>Hospitals</h3>
          {% for hospital in form.hospitals %}
            <div class="row two">
              <label>H{{ loop.index }} X<input name="hospital_x_{{ loop.index }}" type="number" min="0" value="{{ hospital.x }}"></label>
              <label>H{{ loop.index }} Y<input name="hospital_y_{{ loop.index }}" type="number" min="0" value="{{ hospital.y }}"></label>
            </div>
          {% endfor %}
        </div>
        <div class="form-section">
          <h3>Victims</h3>
          {% for victim in form.victims %}
            <div class="row">
              <label>ID<input name="victim_id_{{ loop.index }}" type="number" min="1" value="{{ victim.id }}"></label>
              <label>X<input name="victim_x_{{ loop.index }}" type="number" min="0" value="{{ victim.x }}"></label>
              <label>Y<input name="victim_y_{{ loop.index }}" type="number" min="0" value="{{ victim.y }}"></label>
              <label>SEV<select name="victim_severity_{{ loop.index }}">
                <option value="critical" {% if victim.severity == "critical" %}selected{% endif %}>Crit</option>
                <option value="moderate" {% if victim.severity == "moderate" %}selected{% endif %}>Mod</option>
                <option value="minor" {% if victim.severity == "minor" %}selected{% endif %}>Min</option>
              </select></label>
            </div>
          {% endfor %}
        </div>
        <div class="form-section">
          <h3>Risk Zones</h3>
          {% for risk in form.risk_zones %}
            <div class="row">
              <label>X<input name="risk_x_{{ loop.index }}" type="number" min="0" value="{{ risk.x }}"></label>
              <label>Y<input name="risk_y_{{ loop.index }}" type="number" min="0" value="{{ risk.y }}"></label>
              <label>LVL<select name="risk_level_{{ loop.index }}">
                <option value="1" {% if risk.level == 1 %}selected{% endif %}>1</option>
                <option value="2" {% if risk.level == 2 %}selected{% endif %}>2</option>
                <option value="3" {% if risk.level == 3 %}selected{% endif %}>3</option>
              </select></label>
            </div>
          {% endfor %}
        </div>
        <div class="form-section">
          <h3>Blocked</h3>
          {% for blocked in form.blocked_cells %}
            <div class="row two">
              <label>X<input name="blocked_x_{{ loop.index }}" type="number" min="0" value="{{ blocked.x }}"></label>
              <label>Y<input name="blocked_y_{{ loop.index }}" type="number" min="0" value="{{ blocked.y }}"></label>
            </div>
          {% endfor %}
        </div>
        <div id="formError">{% if error %}<div class="error">{{ error }}</div>{% endif %}</div>
        <button class="replan" type="submit">Replan</button>
      </form>
    </div>
  </div>
  <div class="loading" id="loadingOverlay" aria-live="polite">
    <div class="loader-box">
      <div class="spinner"></div>
      <strong id="loadingTitle">Running Simulation</strong>
      <span id="loadingMessage">Planning route and updating AI panels...</span>
    </div>
  </div>
  <script>
    const initialDashboardData = {{ data|tojson }};
    const form = document.getElementById("scenarioForm");
    const formError = document.getElementById("formError");
    const eventLog = document.getElementById("eventLog");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const loadingTitle = document.getElementById("loadingTitle");
    const loadingMessage = document.getElementById("loadingMessage");
    const configModal = document.getElementById("configModal");
    const openConfigBtn = document.getElementById("openConfigBtn");
    const closeConfigBtn = document.getElementById("closeConfigBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const dualPickupOpt = document.getElementById("dualPickupOpt");
    const actionButtons = [
      document.getElementById("startBtn"),
      pauseBtn,
      document.getElementById("resetBtn"),
      form.querySelector("button[type='submit']"),
    ];
    let currentDashboardData = initialDashboardData;
    let animationTimer = null;
    let routePath = [];
    let routeIndex = 0;
    let isPaused = false;

    function severityClass(severity) {
      if (severity === "critical") return "sev-critical";
      if (severity === "moderate") return "sev-moderate";
      return "sev-minor";
    }

    function log(message) {
      eventLog.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    }

    function setConfigModal(open) {
      configModal.classList.toggle("active", open);
      configModal.setAttribute("aria-hidden", open ? "false" : "true");
    }

    openConfigBtn.addEventListener("click", () => setConfigModal(true));
    closeConfigBtn.addEventListener("click", () => setConfigModal(false));
    configModal.addEventListener("click", (event) => {
      if (event.target === configModal) {
        setConfigModal(false);
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        setConfigModal(false);
      }
    });

    function delay(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function setLoading(active, title = "Running Simulation", message = "Planning route and updating AI panels...") {
      loadingTitle.textContent = title;
      loadingMessage.textContent = message;
      loadingOverlay.classList.toggle("active", active);
      document.getElementById("gridMap").classList.toggle("running", active);
      actionButtons.forEach((button) => {
        if (button) button.disabled = active;
      });
    }

    async function runWithLoader(title, message, callback) {
      setLoading(true, title, message);
      const startedAt = Date.now();
      try {
        return await callback();
      } finally {
        const remaining = Math.max(0, 650 - (Date.now() - startedAt));
        await delay(remaining);
        setLoading(false);
      }
    }

    function terrainFor(cell, routeSet) {
      const key = `${cell.x},${cell.y}`;
      if (routeSet.has(key)) {
        if (["base", "victim", "hospital"].includes(cell.terrain_type)) {
          return { type: `${cell.terrain_type} route-marker`, label: cell.terrain_label || cell.label || "" };
        }
        return { type: "path", label: "*" };
      }
      return { type: cell.terrain_type || cell.type, label: cell.terrain_label || cell.label || "" };
    }

    function updateGrid(data, activeRoute = []) {
      const grid = document.getElementById("gridMap");
      const routeSet = new Set(activeRoute.map((point) => `${point[0]},${point[1]}`));
      grid.style.gridTemplateColumns = `repeat(${data.size}, var(--cell-size))`;
      grid.innerHTML = data.grid.flat().map((cell) => {
        const terrain = terrainFor(cell, routeSet);
        return `
        <div class="cell ${terrain.type}" title="(${cell.x}, ${cell.y}) risk ${cell.risk}">
          ${terrain.label}
          <small>${cell.x},${cell.y}</small>
        </div>
      `;
      }).join("");
    }

    function updateAiPanels(data) {
      document.getElementById("fuzzyFormula").innerHTML =
        `h = urgency + risk<br>IF hazard HIGH or urgency HIGH -> raise risk penalties on edges<br>s=${data.fuzzy.blockage_probability} m=1.00 csp=0.800`;
      document.getElementById("mlFormula").innerHTML =
        `ML vote = mode(kNN, NB, DT)<br>kNN=${data.ml.knn}, NB=${data.ml.naive_bayes}, DT=${data.ml.decision_tree}<br>Final priority=${data.ml.final}`;

      document.getElementById("mlMetrics").innerHTML = `
        <div class="metric"><span>KNN Acc</span><strong>${data.ml.knn}</strong></div>
        <div class="metric"><span>NB Acc</span><strong>${data.ml.naive_bayes}</strong></div>
        <div class="metric"><span>DT Acc</span><strong>${data.ml.decision_tree}</strong></div>
        <div class="metric"><span>Final Priority</span><strong>${data.ml.final}</strong></div>
      `;

      document.getElementById("kpiMetrics").innerHTML = `
        <div class="metric"><span>Victims Saved</span><strong>${data.decision.saved}/${data.victims.length}</strong></div>
        <div class="metric"><span>Avg Rescue Time</span><strong>${data.decision.path_length}</strong></div>
        <div class="metric"><span>Risk Exposure</span><strong>${data.decision.total_risk} pts</strong></div>
        <div class="metric"><span>Resource Util</span><strong>${Math.round(data.decision.utilization * 100)}%</strong></div>
      `;
    }

    function updateVictims(data) {
      document.getElementById("victimRows").innerHTML = data.victims.map((victim, index) => {
        const survival = Math.max(20, 95 - ((index + 1) * 8));
        return `
          <tr>
            <td>V${victim.id}</td>
            <td class="${severityClass(victim.severity)}">${victim.severity.slice(0, 4)}</td>
            <td><span class="pill ${victim.rescued ? "saved" : "waiting"}">${victim.rescued ? "Saved" : "Waiting"}</span></td>
            <td><div class="surv"><i style="width: ${survival}%"></i></div></td>
            <td>${victim.selected ? data.ml.final : "-"}</td>
            <td>${victim.selected ? data.decision.path_length : "-"}</td>
          </tr>
        `;
      }).join("");
    }

    function updatePriority(data) {
      const scenarioLines = data.scenarios.slice(0, 3).map((scenario) =>
        `<p><strong>${scenario.name}:</strong> ${scenario.impact}</p>`
      ).join("");

      const pickupLine = data.decision.dual_pickup && data.decision.selected_victims && data.decision.selected_victims.length >= 2
        ? `Pickup route: ${data.decision.selected_victims.map((id) => `V${id}`).join(" → ")} (same ambulance leg, capacity 2)`
        : `<strong>V${data.decision.selected_victim} first</strong>`;

      document.getElementById("priorityPanel").innerHTML = `
        <h2>Priority</h2>
        <p>${pickupLine} — ${data.ml.final} priority, fuzzy decision ${data.fuzzy.decision}.</p>
        <p>Selected hospital: ${data.decision.selected_hospital}. Path length ${data.decision.path_length}, risk exposure ${data.decision.total_risk}.</p>
        <hr style="border:0;border-top:1px solid rgba(148,163,184,.14)">
        ${scenarioLines}
      `;

      document.getElementById("algorithmPanel").innerHTML = `
        <h2>Search Algorithms</h2>
        ${data.algorithms.map((algo) => `<p><strong>${algo.name}</strong> - ${algo.length} steps, ${algo.risk} risk</p>`).join("")}
      `;
    }

    function renderDashboard(data, activeRoute = []) {
      currentDashboardData = data;
      updateGrid(data, activeRoute);
      updateAiPanels(data);
      updateVictims(data);
      updatePriority(data);
    }

    function clearDashboardOutputs() {
      document.getElementById("fuzzyFormula").innerHTML = "Not started. Click Start or Replan to run fuzzy inference.";
      document.getElementById("mlFormula").innerHTML = "Not started. Click Start or Replan to run ML evaluation.";
      document.getElementById("mlMetrics").innerHTML = `
        <div class="metric"><span>kNN</span><strong>--</strong></div>
        <div class="metric"><span>Naive Bayes</span><strong>--</strong></div>
        <div class="metric"><span>Decision Tree</span><strong>--</strong></div>
        <div class="metric"><span>Final Priority</span><strong>--</strong></div>
      `;
      document.getElementById("kpiMetrics").innerHTML = `
        <div class="metric"><span>Victims Saved</span><strong>--</strong></div>
        <div class="metric"><span>Avg Rescue Time</span><strong>--</strong></div>
        <div class="metric"><span>Risk Exposure</span><strong>--</strong></div>
        <div class="metric"><span>Resource Util</span><strong>--</strong></div>
      `;
      document.getElementById("victimRows").innerHTML =
        `<tr><td colspan="6" class="muted">Not started. Click Start to evaluate victims.</td></tr>`;
      document.getElementById("priorityPanel").innerHTML =
        `<h2>Priority</h2><p class="muted">No priority decision yet. Click Start or Replan.</p>`;
      document.getElementById("algorithmPanel").innerHTML =
        `<h2>Search Algorithms</h2><p class="muted">No algorithm comparison yet.</p>`;
    }

    function stopRouteAnimation(clearRoute = false) {
      if (animationTimer) {
        clearTimeout(animationTimer);
        animationTimer = null;
      }
      isPaused = false;
      pauseBtn.textContent = "⏸";
      pauseBtn.title = "Pause";
      pauseBtn.setAttribute("aria-label", "Pause");
      document.getElementById("gridMap").classList.remove("running");
      if (clearRoute) {
        routePath = [];
        routeIndex = 0;
        updateGrid(currentDashboardData, []);
      }
    }

    function animateNextRouteStep() {
      if (isPaused) {
        return;
      }

      routeIndex += 1;
      updateGrid(currentDashboardData, routePath.slice(0, routeIndex));

      if (routeIndex < routePath.length) {
        animationTimer = setTimeout(animateNextRouteStep, 180);
      } else {
        animationTimer = null;
        document.getElementById("gridMap").classList.remove("running");
        log("Rescue route animation completed.");
      }
    }

    function startRouteAnimation(data) {
      stopRouteAnimation(true);
      routePath = data.paths.full || [];
      routeIndex = 0;
      document.getElementById("gridMap").classList.add("running");
      log("Animating rescue route step by step...");
      animateNextRouteStep();
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || "Request failed.");
      }
      return data;
    }

    document.getElementById("startBtn").addEventListener("click", async () => {
      try {
        await runWithLoader("Starting Simulation", "Running path planning, ML, fuzzy logic, and CSP...", async () => {
          log("Running live simulation...");
          const params = new URLSearchParams();
          if (dualPickupOpt.checked) {
            params.set("dual_pickup", "1");
          }
          const query = params.toString();
          const data = await fetchJson(`/api/dashboard${query ? `?${query}` : ""}`);
          renderDashboard(data, []);
          formError.innerHTML = "";
          currentDashboardData = data;
        });
        startRouteAnimation(currentDashboardData);
      } catch (error) {
        setLoading(false);
        log(error.message);
      }
    });

    document.getElementById("pauseBtn").addEventListener("click", () => {
      if (!routePath.length || routeIndex >= routePath.length) {
        log("Nothing is currently running. Click Start to animate the route.");
        return;
      }

      isPaused = !isPaused;
      pauseBtn.textContent = isPaused ? "▶" : "⏸";
      pauseBtn.title = isPaused ? "Resume" : "Pause";
      pauseBtn.setAttribute("aria-label", isPaused ? "Resume" : "Pause");
      log(isPaused ? "Route animation paused." : "Route animation resumed.");
      if (!isPaused) {
        animateNextRouteStep();
      }
    });

    document.getElementById("resetBtn").addEventListener("click", async () => {
      try {
        await runWithLoader("Resetting Scenario", "Restoring the default disaster map...", async () => {
          log("Resetting to default scenario...");
          const params = new URLSearchParams({ reset: "1" });
          if (dualPickupOpt.checked) {
            params.set("dual_pickup", "1");
          }
          const data = await fetchJson(`/api/dashboard?${params.toString()}`);
          currentDashboardData = data;
          updateGrid(data, []);
          clearDashboardOutputs();
          formError.innerHTML = "";
        });
        stopRouteAnimation(true);
        log("Default scenario restored. Route is cleared until you click Start.");
      } catch (error) {
        setLoading(false);
        log(error.message);
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        await runWithLoader("Replanning Scenario", "Applying your form inputs and recalculating decisions...", async () => {
          log("Submitting custom scenario...");
          const fd = new FormData(form);
          if (dualPickupOpt.checked) {
            fd.append("dual_pickup", "1");
          }
          const data = await fetchJson("/api/dashboard/form", {
            method: "POST",
            body: fd,
          });
          renderDashboard(data, []);
          formError.innerHTML = "";
          currentDashboardData = data;
        });
        setConfigModal(false);
        startRouteAnimation(currentDashboardData);
      } catch (error) {
        setLoading(false);
        formError.innerHTML = `<div class="error">${error.message}</div>`;
        log(error.message);
      }
    });

    currentDashboardData = initialDashboardData;
    updateGrid(initialDashboardData, []);
    clearDashboardOutputs();
  </script>
</body>
</html>
"""


TAB_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} | AIDRA</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; --bg:#f3f6fb; --panel:#ffffff; --line:#dbe6f3; --blue:#3b82f6; --text:#1f2937; --muted:#64748b; }
    * { box-sizing: border-box; }
    body { background: var(--bg); color: var(--text); margin: 0; }
    a { color: inherit; text-decoration: none; }
    .app { min-height:100vh; }
    .top-nav { background:#ffffff; border-right:1px solid var(--line); display:flex; flex-direction:column; gap:20px; height:100vh; left:0; overflow:hidden; padding:20px 14px; position:fixed; top:0; width:240px; z-index:20; }
    .brand { align-items:center; display:flex; gap:12px; }
    .logo { align-items:center; background:transparent; border:0; border-radius:5px; display:grid; height:2.9rem; overflow:hidden; place-items:center; width:2.9rem; }
    .logo img { height:100%; object-fit:contain; width:100%; }
    h1, h2, h3 { margin: 0; }
    .brand h1 { font-size:12px; line-height:1; }
    .brand span, .muted { color:var(--muted); font-size:10px; }
    .tabs { display:flex; flex-direction:column; gap:8px; }
    .tab { border-radius:10px; color:#64748b; font-size:14px; font-weight:700; padding:11px 12px; }
    .tab.active { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; }
    .page { display:grid; gap:14px; margin-left:240px; padding:16px; width:calc(100% - 240px); }
    .hero { background:#ffffff; border:1px solid var(--line); border-radius:16px; padding:18px; }
    .hero h2 { color:#1d4ed8; font-size:26px; margin-bottom:6px; }
    .grid { display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .panel { background:#ffffff; border:1px solid var(--line); border-radius:16px; padding:16px; }
    .panel h3 { color:#0284c7; font-size:14px; letter-spacing:.08em; margin-bottom:12px; text-transform:uppercase; }
    table { border-collapse:collapse; width:100%; }
    th { color:#2563eb; font-size:11px; letter-spacing:.08em; padding:9px 6px; text-align:left; text-transform:uppercase; }
    td { border-top:1px solid #e2e8f0; color:#1f2937; font-size:13px; padding:10px 6px; }
    .pill { border-radius:999px; display:inline-block; font-size:11px; font-weight:900; padding:5px 9px; text-transform:uppercase; }
    .ok { background:#dcfce7; color:#166534; }
    .warn { background:#fef3c7; color:#92400e; }
    .bad { background:#fee2e2; color:#991b1b; }
    .metric { align-items:center; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; padding:11px 0; }
    .metric:first-of-type { border-top:0; }
    .metric span { color:#64748b; }
    .metric strong { color:#111827; }
    .trace { background:#f8fafc; border:1px solid #dbe6f3; border-radius:10px; color:#334155; font-family:Consolas, monospace; line-height:1.6; padding:14px; white-space:pre-wrap; }
    .chart-grid { display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .chart-card { background:#ffffff; border:1px solid var(--line); border-radius:16px; padding:12px; }
    .chart-card h3 { margin:0 0 8px; }
    .chart-card canvas { height:260px !important; width:100% !important; }
    @media (max-width: 1180px) { .top-nav { height:auto; position:static; width:auto; } .page { margin-left:0; width:100%; } .tabs { flex-direction:row; flex-wrap:wrap; } }
  </style>
</head>
<body>
  <div class="app">
    <header class="top-nav">
      <div class="brand">
        <div class="logo"><img src="/logo.png" alt="AIDRA logo"></div>
        <div><h1>AIDRA</h1><span>Adaptive Intelligent Disaster Response Agent</span></div>
      </div>
      <nav class="tabs">
        <a class="tab {{ 'active' if active == 'live' else '' }}" href="/">Live Simulation</a>
        <a class="tab {{ 'active' if active == 'search' else '' }}" href="/search-trace">Search Trace</a>
        <a class="tab {{ 'active' if active == 'csp' else '' }}" href="/csp-solver">CSP Solver</a>
        <a class="tab {{ 'active' if active == 'ml' else '' }}" href="/ml-studio">ML Studio</a>
        <a class="tab {{ 'active' if active == 'analytics' else '' }}" href="/analytics">Analytics</a>
      </nav>
    </header>

    <main class="page">
    <section class="hero">
      <h2>{{ title }}</h2>
      <p class="muted">{{ subtitle }}</p>
    </section>

    {% if active == "search" %}
      <section class="grid">
        <div class="panel">
          <h3>Selected Victim</h3>
          <div class="metric"><span>Victim ID</span><strong>V{{ data.decision.selected_victim }}</strong></div>
          <div class="metric"><span>Severity</span><strong>{{ data.decision.selected_severity|upper }}</strong></div>
          <div class="metric"><span>Path Length</span><strong>{{ data.decision.path_length }}</strong></div>
        </div>
        <div class="panel">
          <h3>Hospital Selection</h3>
          <div class="metric"><span>Nearest Hospital</span><strong>{{ data.decision.selected_hospital }}</strong></div>
          <div class="metric"><span>Hospital Path Length</span><strong>{{ data.paths.victim_to_hospital|length }}</strong></div>
          <div class="metric"><span>Total Risk</span><strong>{{ data.decision.total_risk }}</strong></div>
        </div>
      </section>
      <section class="panel">
        <h3>Required Map Trace</h3>
        <div class="trace">Base to victim:
{{ data.paths.base_to_victim }}

Victim to hospital:
{{ data.paths.victim_to_hospital }}

Full route:
{{ data.paths.full }}</div>
      </section>
    {% elif active == "csp" %}
      <section class="grid">
        <div class="panel">
          <h3>Ambulance Assignments</h3>
          <table>
            <thead><tr><th>Victim</th><th>Severity</th><th>Location</th><th>Assigned Resource</th><th>State</th></tr></thead>
            <tbody>
              {% for item in data.assignments %}
                <tr>
                  <td>V{{ item.victim }}</td>
                  <td>{{ item.severity|upper }}</td>
                  <td>{{ item.location }}</td>
                  <td>{{ item.ambulance }}</td>
                  <td><span class="pill {{ 'ok' if item.assigned else 'warn' }}">{{ "Assigned" if item.assigned else "Pending" }}</span></td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h3>CSP Summary</h3>
          <div class="metric"><span>Rule</span><strong>Max 2 victims per ambulance</strong></div>
          <div class="metric"><span>Assigned</span><strong>{{ data.csp.assigned_count }}</strong></div>
          <div class="metric"><span>Unassigned</span><strong>{{ data.csp.unassigned_count }}</strong></div>
          <div class="metric"><span>Total Victims</span><strong>{{ data.csp.total_victims }}</strong></div>
          <div class="metric"><span>Backtracking Steps</span><strong>{{ data.csp.backtrack_count }}</strong></div>
          <div class="metric"><span>Ambulances Used</span><strong>{{ data.csp.ambulance_count_used }}</strong></div>
          <div class="metric"><span>Feasible Assignment</span><strong>{{ "Yes" if data.csp.all_assigned else "Partial" }}</strong></div>
          <div class="metric"><span>Solver</span><strong>Backtracking + MRV</strong></div>
          {% for load in data.csp.ambulance_loads %}
            <div class="metric"><span>{{ load.ambulance }}</span><strong>{{ load.count }} victims</strong></div>
          {% endfor %}
        </div>
      </section>
      <section class="panel">
        <h3>Backtracking Trace</h3>
        <div class="trace">{% if data.csp.trace_text %}{{ data.csp.trace_text }}{% else %}No CSP trace output available.{% endif %}</div>
      </section>
    {% elif active == "ml" %}
      <section class="grid">
        <div class="panel">
          <h3>Model Predictions</h3>
          <div class="metric"><span>kNN</span><strong>{{ data.ml.knn }}</strong></div>
          <div class="metric"><span>Naive Bayes</span><strong>{{ data.ml.naive_bayes }}</strong></div>
          <div class="metric"><span>Decision Tree</span><strong>{{ data.ml.decision_tree }}</strong></div>
          <div class="metric"><span>Majority Vote Decision</span><strong>{{ data.ml.final_priority }}</strong></div>
        </div>
        <div class="panel">
          <h3>Fuzzy Logic Uncertainty</h3>
          <div class="metric"><span>Fuzzy Decision (blockage_prob={{ data.fuzzy.blockage_probability }})</span><strong>{{ data.fuzzy.decision }}</strong></div>
          <div class="metric"><span>Distance</span><strong>{{ data.fuzzy.distance }}</strong></div>
          <div class="metric"><span>Risk</span><strong>{{ data.fuzzy.risk }}</strong></div>
          <div class="metric"><span>Severity</span><strong>{{ data.fuzzy.severity|upper }}</strong></div>
        </div>
      </section>
      <section class="panel">
        <h3>Fuzzy Logic Log</h3>
        <div class="trace">{{ data.fuzzy.log }}</div>
      </section>
      <section class="panel">
        <h3>ML Metrics</h3>
        <table>
          <thead>
            <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>kNN</td>
              <td>{{ data.ml.metrics.knn.accuracy }}</td>
              <td>{{ data.ml.metrics.knn.precision }}</td>
              <td>{{ data.ml.metrics.knn.recall }}</td>
              <td>{{ data.ml.metrics.knn.f1 }}</td>
            </tr>
            <tr>
              <td>Naive Bayes</td>
              <td>{{ data.ml.metrics.naive_bayes.accuracy }}</td>
              <td>{{ data.ml.metrics.naive_bayes.precision }}</td>
              <td>{{ data.ml.metrics.naive_bayes.recall }}</td>
              <td>{{ data.ml.metrics.naive_bayes.f1 }}</td>
            </tr>
            <tr>
              <td>Decision Tree</td>
              <td>{{ data.ml.metrics.decision_tree.accuracy }}</td>
              <td>{{ data.ml.metrics.decision_tree.precision }}</td>
              <td>{{ data.ml.metrics.decision_tree.recall }}</td>
              <td>{{ data.ml.metrics.decision_tree.f1 }}</td>
            </tr>
          </tbody>
        </table>
      </section>
      <section class="grid">
        <div class="panel">
          <h3>Victim Features</h3>
          <table>
            <thead><tr><th>ID</th><th>Severity</th><th>Location</th><th>Selected</th></tr></thead>
            <tbody>
              {% for victim in data.victims %}
                <tr><td>V{{ victim.id }}</td><td>{{ victim.severity }}</td><td>({{ victim.x }}, {{ victim.y }})</td><td><span class="pill {{ 'ok' if victim.selected else 'warn' }}">{{ "Yes" if victim.selected else "No" }}</span></td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h3>Model Notes</h3>
          <div class="metric"><span>Priority Label</span><strong>{{ data.ml.final_priority }}</strong></div>
          <div class="metric"><span>Selected Victim</span><strong>V{{ data.decision.selected_victim }}</strong></div>
          <div class="metric"><span>Selected Severity</span><strong>{{ data.decision.selected_severity|upper }}</strong></div>
        </div>
      </section>
    {% else %}
      <section class="chart-grid">
        <div class="chart-card">
          <h3>Path Length by Algorithm</h3>
          <canvas id="lengthChart" aria-label="Path length by algorithm" role="img"></canvas>
        </div>
        <div class="chart-card">
          <h3>Risk Exposure by Algorithm</h3>
          <canvas id="riskChart" aria-label="Risk exposure by algorithm" role="img"></canvas>
        </div>
      </section>
      <section class="grid">
        <div class="panel">
          <h3>Search Algorithm Comparison</h3>
          <table>
            <thead><tr><th>Algorithm</th><th>Path Length</th><th>Risk Exposure</th><th>Type</th></tr></thead>
            <tbody>
              {% for algo in data.analytics.comparison %}
                <tr><td>{{ algo.name }}</td><td>{{ algo.length }}</td><td>{{ algo.risk }}</td><td>{{ algo.safety }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h3>Path Visualizations (Text Preview)</h3>
          {% for route in data.analytics.paths %}
            <div class="metric"><span>{{ route.name }}</span><strong>{{ route.preview }}</strong></div>
          {% endfor %}
        </div>
      </section>
      <section class="panel">
        <h3>Trade-off Analysis</h3>
        <table>
          <thead><tr><th>Algorithm</th><th>Speed</th><th>Optimality</th><th>Safety</th><th>Best Use Case</th></tr></thead>
          <tbody>
            {% for algo in data.analytics.comparison %}
              <tr>
                <td>{{ algo.name }}</td>
                <td>{{ algo.speed }}</td>
                <td>{{ algo.optimality }}</td>
                <td>{{ algo.safety }}</td>
                <td>{{ algo.use_case }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </section>
    {% endif %}
    </main>
    {% if active == "analytics" %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      const analyticsRows = {{ data.analytics.comparison | tojson }};
      const labels = analyticsRows.map((item) => item.name);
      const lengths = analyticsRows.map((item) => item.length);
      const risks = analyticsRows.map((item) => item.risk);

      const palette = ["#60a5fa", "#818cf8", "#34d399", "#f59e0b", "#f87171", "#fb923c"];

      new Chart(document.getElementById("lengthChart"), {
        type: "bar",
        data: {
          labels,
          datasets: [{ label: "Path Length", data: lengths, backgroundColor: palette }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { beginAtZero: true } },
          plugins: { legend: { display: false } }
        }
      });

      new Chart(document.getElementById("riskChart"), {
        type: "bar",
        data: {
          labels,
          datasets: [{ label: "Risk Exposure", data: risks, backgroundColor: palette }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { beginAtZero: true } },
          plugins: { legend: { display: false } }
        }
      });
    </script>
    {% endif %}
</body>
</html>
"""


SIMULATION_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simulation Output</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; --bg:#f3f6fb; --panel:#ffffff; --line:#dbe6f3; --text:#1f2937; --muted:#64748b; }
    * { box-sizing: border-box; }
    body { background: var(--bg); color: var(--text); margin: 0; }
    a { color: inherit; text-decoration: none; }
    .app { min-height:100vh; }
    .top-nav { background:#ffffff; border-right:1px solid var(--line); display:flex; flex-direction:column; gap:20px; height:100vh; left:0; overflow:hidden; padding:20px 14px; position:fixed; top:0; width:240px; z-index:20; }
    .brand { align-items:center; display:flex; gap:12px; }
    .logo { align-items:center; background:transparent; border:0; border-radius:5px; display:grid; height:2.9rem; overflow:hidden; place-items:center; width:2.9rem; }
    .logo img { height:100%; object-fit:contain; width:100%; }
    .brand h1 { font-size:12px; line-height:1; margin:0; }
    .brand span, .muted { color:var(--muted); font-size:10px; }
    .tabs { display:flex; flex-direction:column; gap:8px; }
    .tab { border-radius:10px; color:#64748b; font-size:14px; font-weight:700; padding:11px 12px; }
    .tab.active { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; }
    .page { margin-left:240px; padding:16px; width:calc(100% - 240px); }
    .panel { background:#ffffff; border:1px solid var(--line); border-radius:16px; padding:16px; }
    .panel-head { align-items:center; display:flex; justify-content:space-between; margin-bottom:12px; }
    .panel-head h2 { margin:0; }
    .back-btn { background:#2563eb; border-radius:10px; color:#fff; font-weight:700; padding:10px 14px; }
    .meta { color:#475569; margin-bottom:14px; }
    pre { background:#ffffff; border:1px solid #dbe6f3; border-radius:12px; color:#334155; font-family:Consolas, Monaco, monospace; overflow:auto; padding:16px; white-space:pre-wrap; }
    @media (max-width: 1180px) { .top-nav { height:auto; position:static; width:auto; } .page { margin-left:0; width:100%; } .tabs { flex-direction:row; flex-wrap:wrap; } }
  </style>
</head>
<body>
  <div class="app">
    <header class="top-nav">
      <div class="brand">
        <div class="logo"><img src="/logo.png" alt="AIDRA logo"></div>
        <div><h1>AIDRA</h1><span>Adaptive Intelligent Disaster Response Agent</span></div>
      </div>
      <nav class="tabs">
        <a class="tab" href="/">Live Simulation</a>
        <a class="tab" href="/search-trace">Search Trace</a>
        <a class="tab" href="/csp-solver">CSP Solver</a>
        <a class="tab" href="/ml-studio">ML Studio</a>
        <a class="tab" href="/analytics">Analytics</a>
        <a class="tab active" href="/simulation-log">Simulation Output</a>
      </nav>
    </header>
    <main class="page">
      <section class="panel">
        <div class="panel-head">
          <h2>AIDRA Simulation Output</h2>
          <a class="back-btn" href="/">Back to Dashboard</a>
        </div>
        <div class="meta">
          Status: {{ "success" if result.ok else "failed" }} |
          Exit code: {{ result.exit_code }} |
          Duration: {{ result.duration_seconds }}s
        </div>
        <pre>{{ result.output }}</pre>
      </section>
    </main>
  </div>
</div>
</body>
</html>
"""


def render_dashboard(scenario=None, error=None, status_code=200, dual_pickup=False):
    scenario = scenario or DEFAULT_SCENARIO
    try:
        form = scenario_form_data(scenario)
        data = build_dashboard_data(scenario, dual_pickup=dual_pickup)
        if status_code == 200:
            set_active_scenario(scenario)
    except ValueError as exc:
        clear_active_scenario()
        form = scenario_form_data(DEFAULT_SCENARIO)
        data = build_dashboard_data(DEFAULT_SCENARIO, dual_pickup=False)
        error = str(exc)
        status_code = 400

    return render_template_string(
        INDEX_HTML,
        data=data,
        form=form,
        error=error,
    ), status_code


def render_tab_page(title, subtitle, active):
    return render_template_string(
        TAB_PAGE_HTML,
        title=title,
        subtitle=subtitle,
        active=active,
        data=build_dashboard_data(get_active_scenario(), dual_pickup=False),
    )


@app.get("/")
def index():
    return render_dashboard(
        get_active_scenario(),
        dual_pickup=dual_pickup_requested(),
    )


@app.get("/logo.png")
def logo_file():
    return send_from_directory(PROJECT_DIR, "logo.png")


@app.get("/search-trace")
def search_trace_page():
    return render_tab_page(
        "Search Trace",
        "Compare BFS, DFS, Greedy, Hill Climbing, A*, and Risk-Aware A* route behavior.",
        "search",
    )


@app.get("/csp-solver")
def csp_solver_page():
    return render_tab_page(
        "CSP Solver",
        "Inspect ambulance allocation under resource and capacity constraints.",
        "csp",
    )


@app.get("/ml-studio")
def ml_studio_page():
    return render_tab_page(
        "ML Studio",
        "Review model predictions from kNN, Naive Bayes, Decision Tree, and majority voting.",
        "ml",
    )


@app.get("/analytics")
def analytics_page():
    return render_tab_page(
        "Analytics",
        "Review mission KPIs, route quality, risk exposure, and dynamic scenario outcomes.",
        "analytics",
    )


@app.post("/dashboard")
def custom_dashboard():
    try:
        scenario = scenario_from_form(request.form)
    except ValueError as exc:
        return render_dashboard(DEFAULT_SCENARIO, str(exc), 400)

    return render_dashboard(
        scenario,
        dual_pickup=dual_pickup_requested(),
    )


@app.get("/simulation")
def simulation_page():
    result = run_simulation()
    return render_template_string(SIMULATION_HTML, result=result), 200 if result["ok"] else 500


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "AIDRA Flask API"})


@app.route("/api/dashboard", methods=["GET", "POST"])
def dashboard_api():
    try:
        dual = dual_pickup_requested()
        if request.method == "GET" and request.args.get("reset"):
            clear_active_scenario()
            return jsonify(build_dashboard_data(DEFAULT_SCENARIO, dual_pickup=dual))
        if request.method == "GET":
            scenario = get_active_scenario()
        else:
            scenario = scenario_from_dashboard_json()
            if scenario is None or scenario == {}:
                scenario = DEFAULT_SCENARIO
        payload = build_dashboard_data(scenario, dual_pickup=dual)
        if request.method == "POST":
            set_active_scenario(scenario)
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/dashboard/form")
def dashboard_form_api():
    try:
        scenario = scenario_from_form(request.form)
        payload = build_dashboard_data(scenario, dual_pickup=dual_pickup_requested())
        set_active_scenario(scenario)
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/simulation")
def simulation_api():
    result = run_simulation()
    return jsonify(result), 200 if result["ok"] else 500


if __name__ == "__main__":
    app.run(debug=True)
