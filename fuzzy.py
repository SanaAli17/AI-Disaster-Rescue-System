# FUZZIFICATION FUNCTIONS

def fuzzify_distance(d):
    if d <= 6:
        return "near"
    elif d <= 12:
        return "medium"
    else:
        return "far"


def fuzzify_risk(r):
    if r <= 2:
        return "low"
    elif r <= 5:
        return "medium"
    else:
        return "high"


def fuzzify_blockage(prob):
    if prob < 0.3:
        return "unlikely"
    elif prob < 0.7:
        return "possible"
    else:
        return "likely"

# FUZZY INFERENCE SYSTEM

def fuzzy_inference(distance, risk, severity, blockage_prob):
    d = fuzzify_distance(distance)
    r = fuzzify_risk(risk)
    b = fuzzify_blockage(blockage_prob)

    print("\n--- FUZZY LOGIC INPUT ---")
    print(f"Distance: {distance} -> {d}")
    print(f"Risk: {risk} -> {r}")
    print(f"Severity: {severity}")
    print(f"Blockage Probability: {blockage_prob} -> {b}")

    # RULE BASE
    
    if severity == "critical":
        if b == "likely":
            decision = "HIGH (urgent but risky)"
        else:
            decision = "HIGH"

    elif severity == "moderate":
        if r == "high" or b == "likely":
            decision = "MEDIUM"
        elif d == "near":
            decision = "HIGH"
        else:
            decision = "MEDIUM"

    else:  # minor
        if r == "low" and b == "unlikely":
            decision = "LOW"
        else:
            decision = "MEDIUM"

    return decision



# IMPACT ANALYSIS FUNCTION

def explain_uncertainty_impact(fuzzy_result):
    print("\n--- UNCERTAINTY IMPACT ANALYSIS ---")

    if "risky" in fuzzy_result:
        print("Uncertainty Impact: High blockage probability -> route considered unsafe")
        print("Agent Decision: Prefer safer route even if longer")

    elif fuzzy_result.startswith("HIGH"):
        print("Uncertainty Impact: Severity dominates under uncertainty")
        print("Agent Decision: Immediate rescue prioritized")

    elif fuzzy_result.startswith("MEDIUM"):
        print("Uncertainty Impact: Balanced decision under uncertain conditions")
        print("Agent Decision: Moderate priority assigned")

    else:
        print("Uncertainty Impact: Low urgency with low risk")
        print("Agent Decision: Rescue can be delayed")