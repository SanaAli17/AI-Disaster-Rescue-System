# CSP: RESOURCE ALLOCATION

backtrack_count = 0   # GLOBAL COUNTER


def is_valid(assignment, ambulance):
    if list(assignment.values()).count(ambulance) >= 2:
        print(f"Constraint violated: {ambulance} already has 2 victims")
        return False
    return True


# MRV: SELECT NEXT VARIABLE

def select_unassigned_variable(assignment, victims):
    # Pick first unassigned victim (MRV simplified)
    for v in victims:
        if v.id not in assignment:
            return v
    return None


# BACKTRACKING SEARCH

def backtrack(assignment, victims, ambulances):
    global backtrack_count   # OK USE GLOBAL

    if len(assignment) == 4:
        return assignment

    victim = select_unassigned_variable(assignment, victims)

    if victim is None:
        return assignment

    for amb in ambulances:

        print(f"Trying: Victim {victim.id} -> {amb}")

        if is_valid(assignment, amb):

            assignment[victim.id] = amb

            result = backtrack(assignment, victims, ambulances)

            if result is not None:
                return result

            # BACKTRACK STEP
            backtrack_count += 1   
            print(f"Backtracking: Victim {victim.id} from {amb}")

            del assignment[victim.id]

    return assignment


# MAIN CSP FUNCTION

def csp_assign(victims):
    global backtrack_count
    backtrack_count = 0   

    print("\n--- CSP (Backtracking + MRV) ---")

    ambulances = ["Ambulance 1", "Ambulance 2"]

    # Sort victims by severity (heuristic improvement)
    victims_sorted = sorted(
        victims,
        key=lambda v: {"critical": 3, "moderate": 2, "minor": 1}[v.severity],
        reverse=True
    )

    assignment = {}

    result = backtrack(assignment, victims_sorted, ambulances)


    # OUTPUT RESULTS

    if result and len(result) > 0:
        for v in result:
            print(f"Victim {v} -> {result[v]}")
    else:
        print("WARNING: No assignment possible")

    # Show unassigned victims
    assigned_ids = set(result.keys()) if result else set()
    for v in victims:
        if v.id not in assigned_ids:
            print(f"WARNING: Victim {v.id} could not be assigned (capacity exceeded)")


    # RETURN BOTH RESULT + METRIC
   
    return result, backtrack_count   