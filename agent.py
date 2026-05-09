from search import a_star

# SEVERITY SCORING

def get_severity_score(severity):
    if severity == "critical":
        return 100
    elif severity == "moderate":
        return 50
    else:
        return 10


# PATH RISK CALCULATION

def compute_path_risk(env, path):
    return sum(env.grid[x][y].risk_level for x, y in path)


# DECISION EXPLANATION (NEW)

def explain_decision(victim, distance, risk):
    print("\n--- DECISION EXPLANATION ---")

    if victim.severity == "critical":
        print("High priority: Victim is CRITICAL")

    if risk > distance:
        print("Decision influenced more by risk than distance")

    if distance < 10:
        print("Victim is relatively close -> faster rescue possible")

    print("Final decision balances severity, distance, and risk")


# MAIN AGENT FUNCTIONS

def choose_best_victim(env, start):
    best_victim = None
    best_score = -float('inf')
    best_path = None

    print("\n--- VICTIM SELECTION PROCESS ---")

    for victim in env.victims:
        if victim.rescued:
            continue

        goal = (victim.x, victim.y)

        path = a_star(env, start, goal)
        if path is None:
            continue

        distance = len(path)
        risk = compute_path_risk(env, path)
        severity_score = get_severity_score(victim.severity)

        # scoring function
        score = severity_score - (distance + risk)

        print(f"\nVictim {victim.id}:")
        print(f"  Severity: {victim.severity}")
        print(f"  Distance: {distance}")
        print(f"  Risk: {risk}")
        print(f"  Score: {score}")

        # NEW NEW: explanation per victim
        explain_decision(victim, distance, risk)

        if score > best_score:
            best_score = score
            best_victim = victim
            best_path = path

    print("\nOK Selected Victim:", best_victim.id)
    print("Reason: Highest combined priority (severity, distance, risk)")

    return best_victim, best_path

def find_nearest_hospital(env, victim_pos):
    best_hospital = None
    best_path = None
    best_distance = float('inf')

    print("\n--- HOSPITAL SELECTION ---")

    for hospital in env.hospitals:
        path = a_star(env, victim_pos, hospital)

        if path is None:
            continue

        distance = len(path)

        print(f"Hospital {hospital} -> Distance: {distance}")

        if distance < best_distance:
            best_distance = distance
            best_hospital = hospital
            best_path = path

    print(f"\nOK Selected Hospital: {best_hospital}")
    print("Reason: Closest hospital for fastest rescue")

    return best_hospital, best_path