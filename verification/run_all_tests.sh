#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "RUNNING ALL EMPIRICAL CHALLENGER VERIFICATION HARNESSES"
echo "================================================================================"

echo ""
echo ">>> [1/4] Running Scoring Matrix Verification..."
python3 "/Users/nabils/antigravity/open source/verification/test_scoring_matrix.py"

echo ""
echo ">>> [2/4] Running Compute & Memory Bandwidth Verification..."
python3 "/Users/nabils/antigravity/open source/verification/test_compute_and_bandwidth.py"

echo ""
echo ">>> [3/4] Running Frustum Voxel Memory & Morton Hashing Verification..."
python3 "/Users/nabils/antigravity/open source/verification/test_voxel_memory.py"

echo ""
echo ">>> [4/4] Running Input-to-Photon Latency Budget Verification..."
python3 "/Users/nabils/antigravity/open source/verification/test_latency_budget.py"

echo ""
echo "================================================================================"
echo "ALL EMPIRICAL VERIFICATION HARNESSES COMPLETED"
echo "================================================================================"
