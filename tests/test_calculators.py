import pytest
from src.calculators import (
    calculate_cpu_scheduling,
    simulate_page_replacement,
    compare_page_replacements,
    run_bankers_safety_algorithm,
    run_bankers_resource_request
)

def test_fcfs_scheduling():
    procs = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 24},
        {"pid": "P2", "arrival_time": 0, "burst_time": 3},
        {"pid": "P3", "arrival_time": 0, "burst_time": 3},
    ]
    res = calculate_cpu_scheduling(procs, "FCFS")
    assert "error" not in res
    assert res["metrics"]["average_waiting_time"] == 17.0
    assert res["metrics"]["average_turnaround_time"] == 27.0
    assert len(res["gantt_chart"]) == 3

def test_sjf_non_preemptive():
    procs = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 6},
        {"pid": "P2", "arrival_time": 1, "burst_time": 8},
        {"pid": "P3", "arrival_time": 2, "burst_time": 7},
        {"pid": "P4", "arrival_time": 3, "burst_time": 3},
    ]
    res = calculate_cpu_scheduling(procs, "SJF")
    assert "error" not in res
    p1 = next(p for p in res["processes"] if p["pid"] == "P1")
    p4 = next(p for p in res["processes"] if p["pid"] == "P4")
    # P1 finishes at 6, then P4 (BT=3) is chosen next!
    assert p1["completion_time"] == 6
    assert p4["completion_time"] == 9

def test_round_robin():
    procs = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 5},
        {"pid": "P2", "arrival_time": 1, "burst_time": 3},
        {"pid": "P3", "arrival_time": 2, "burst_time": 1},
    ]
    res = calculate_cpu_scheduling(procs, "RR", time_quantum=2)
    assert "error" not in res
    assert res["metrics"]["total_execution_time"] == 9

def test_page_replacement_silberschatz_baseline():
    refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    fifo = simulate_page_replacement(refs, 3, "FIFO")
    assert fifo["summary"]["page_faults"] == 15

    lru = simulate_page_replacement(refs, 3, "LRU")
    assert lru["summary"]["page_faults"] == 12

    opt = simulate_page_replacement(refs, 3, "Optimal")
    assert opt["summary"]["page_faults"] == 9

def test_bankers_safety_silberschatz():
    processes = ["P0", "P1", "P2", "P3", "P4"]
    resources = ["A", "B", "C"]
    allocation = [
        [0, 1, 0],
        [2, 0, 0],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2],
    ]
    max_mat = [
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3],
    ]
    avail = [3, 3, 2]

    res = run_bankers_safety_algorithm(processes, resources, allocation, max_mat, avail)
    assert res["is_safe"] is True
    assert len(res["safe_sequence"]) == 5
    # P1 can finish first: Need [1, 2, 2] <= [3, 3, 2]
    assert res["safe_sequence"][0] == "P1"

    # Test Resource Request for P1 requesting [1, 0, 2]
    req_res = run_bankers_resource_request(
        "P1", [1, 0, 2], processes, resources, allocation, max_mat, avail
    )
    assert req_res["granted"] is True

    # Test invalid request exceeding max claim
    req_invalid = run_bankers_resource_request(
        "P0", [8, 0, 0], processes, resources, allocation, max_mat, avail
    )
    assert req_invalid["granted"] is False
    assert "exceeds" in req_invalid["error"].lower()
