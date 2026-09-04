#!/usr/bin/env python3
"""
Verification Script: test_scoring_matrix.py
Stress-tests and independently verifies the mathematical accuracy of the scoring matrix
in Section 8.2 of COMPARATIVE_OPPORTUNITY_REPORT.md.
"""

import sys

def verify_scoring_matrix():
    weights = {
        'viral_hook': 0.25,
        'latent_demand': 0.25,
        'competition_vacuum': 0.20,
        'feasibility_mvp': 0.15,
        'defensibility_moat': 0.15
    }
    
    # Verify weights sum to 1.0
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-9, f"Weights do not sum to 1.0: {weight_sum}"

    candidates = [
        {
            "name": "WorldEngine.cpp (PlayWorld)",
            "viral_hook": 10.0,
            "latent_demand": 9.8,
            "competition_vacuum": 10.0,
            "feasibility_mvp": 8.8,
            "defensibility_moat": 9.5,
            "reported_score": 9.695,
            "reported_rank": 1
        },
        {
            "name": "AegisBox / BranchBox",
            "viral_hook": 9.7,
            "latent_demand": 9.8,
            "competition_vacuum": 9.5,
            "feasibility_mvp": 8.8,
            "defensibility_moat": 9.3,
            "reported_score": 9.490,
            "reported_rank": 2
        },
        {
            "name": "ChronoAgent",
            "viral_hook": 9.6,
            "latent_demand": 9.5,
            "competition_vacuum": 9.2,
            "feasibility_mvp": 9.0,
            "defensibility_moat": 8.7,
            "reported_score": 9.270,
            "reported_rank": 3
        },
        {
            "name": "AgentMux",
            "viral_hook": 9.4,
            "latent_demand": 9.0,
            "competition_vacuum": 8.8,
            "feasibility_mvp": 8.2,
            "defensibility_moat": 8.6,
            "reported_score": 8.880,
            "reported_rank": 4
        },
        {
            "name": "AgentDeck",
            "viral_hook": 9.3,
            "latent_demand": 9.0,
            "competition_vacuum": 8.5,
            "feasibility_mvp": 8.9,
            "defensibility_moat": 7.9,
            "reported_score": 8.795,
            "reported_rank": 5
        },
        {
            "name": "MeshQL",
            "viral_hook": 8.8,
            "latent_demand": 8.5,
            "competition_vacuum": 9.0,
            "feasibility_mvp": 7.2,
            "defensibility_moat": 8.4,
            "reported_score": 8.465,
            "reported_rank": 6
        },
        {
            "name": "OpenAtlas",
            "viral_hook": 8.8,
            "latent_demand": 8.5,
            "competition_vacuum": 8.2,
            "feasibility_mvp": 6.0,
            "defensibility_moat": 8.0,
            "reported_score": 8.065,
            "reported_rank": 7
        },
        {
            "name": "InfiniteWorld-Studio",
            "viral_hook": 8.2,
            "latent_demand": 8.0,
            "competition_vacuum": 8.5,
            "feasibility_mvp": 7.8,
            "defensibility_moat": 7.5,
            "reported_score": 8.045,
            "reported_rank": 8
        },
        {
            "name": "SWE-Bench-Local",
            "viral_hook": 7.5,
            "latent_demand": 7.8,
            "competition_vacuum": 8.2,
            "feasibility_mvp": 8.8,
            "defensibility_moat": 7.5,
            "reported_score": 7.910,
            "reported_rank": 9
        },
        {
            "name": "OmniPhysics-WM",
            "viral_hook": 6.5,
            "latent_demand": 7.2,
            "competition_vacuum": 8.5,
            "feasibility_mvp": 6.2,
            "defensibility_moat": 7.8,
            "reported_score": 7.225,
            "reported_rank": 10
        }
    ]

    all_passed = True
    calculated_entries = []

    print(f"{'Candidate':<26} | {'Reported':<8} | {'Calculated':<10} | {'Diff':<8} | {'Rank Check'}")
    print("-" * 75)

    for c in candidates:
        calc_score = (
            c['viral_hook'] * weights['viral_hook'] +
            c['latent_demand'] * weights['latent_demand'] +
            c['competition_vacuum'] * weights['competition_vacuum'] +
            c['feasibility_mvp'] * weights['feasibility_mvp'] +
            c['defensibility_moat'] * weights['defensibility_moat']
        )
        diff = abs(calc_score - c['reported_score'])
        match = diff < 1e-4
        if not match:
            all_passed = False
        calculated_entries.append((c['name'], calc_score, c['reported_rank'], match, diff))
        print(f"{c['name']:<26} | {c['reported_score']:<8.3f} | {calc_score:<10.4f} | {diff:<8.5f} | {'PASS' if match else 'FAIL'}")

    # Verify ranking consistency (strictly monotonic descending)
    print("\nVerifying monotonicity of ranks:")
    for i in range(len(calculated_entries) - 1):
        c1 = calculated_entries[i]
        c2 = calculated_entries[i+1]
        if c1[1] <= c2[1]:
            print(f"Rank inversion detected between {c1[0]} ({c1[1]:.4f}) and {c2[0]} ({c2[1]:.4f})!")
            all_passed = False
        else:
            print(f"  Rank #{c1[2]} ({c1[0]}: {c1[1]:.3f}) > Rank #{c2[2]} ({c2[0]}: {c2[1]:.3f}) [OK]")

    if all_passed:
        print("\n>>> ALL SCORING MATRIX CALCULATIONS EMPIRICALLY VERIFIED (100% MATCH).")
        return 0
    else:
        print("\n>>> SCORING MATRIX CONTAINS DISCREPANCIES.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_scoring_matrix())
