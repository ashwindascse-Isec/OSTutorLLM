"""
Deterministic Operating Systems Laboratory Calculators and Simulators.
Implements:
1. CPU Scheduling: FCFS, SJF (non-preemptive & preemptive), Priority (non-preemptive & preemptive), Round Robin.
2. Page Replacement: FIFO, LRU, Optimal.
3. Deadlock Safety & Resource Request (Banker's Algorithm).
"""

from typing import List, Dict, Any, Tuple, Optional
import copy


# ============================================================================
# 1. CPU SCHEDULING ALGORITHMS
# ============================================================================

def calculate_cpu_scheduling(
    processes: List[Dict[str, Any]],
    algorithm: str = "FCFS",
    time_quantum: int = 2,
    priority_convention: str = "lower_is_higher" # "lower_is_higher" or "higher_is_higher"
) -> Dict[str, Any]:
    """
    Executes CPU scheduling simulation.
    processes: list of dicts with keys:
        - pid (str): Process ID (e.g., 'P1')
        - arrival_time (int)
        - burst_time (int)
        - priority (int, optional, default 0)
    """
    if not processes:
        return {"error": "Process list is empty"}

    # Validate inputs
    procs = []
    for p in processes:
        try:
            pid = str(p.get("pid", "")).strip()
            at = int(p.get("arrival_time", 0))
            bt = int(p.get("burst_time", 1))
            pr = int(p.get("priority", 0))
            if bt <= 0:
                return {"error": f"Process {pid} burst time must be greater than 0"}
            if at < 0:
                return {"error": f"Process {pid} arrival time cannot be negative"}
            procs.append({
                "pid": pid,
                "arrival_time": at,
                "burst_time": bt,
                "remaining_time": bt,
                "priority": pr,
                "first_start_time": None,
                "completion_time": 0
            })
        except (ValueError, TypeError):
            return {"error": f"Invalid numerical values in process input: {p}"}

    algo = algorithm.upper().strip()
    gantt: List[Dict[str, Any]] = []
    trace_log: List[str] = []
    current_time = 0
    completed = 0
    n = len(procs)

    def is_higher_priority(p1_pr: int, p2_pr: int) -> bool:
        if priority_convention == "lower_is_higher":
            return p1_pr < p2_pr
        return p1_pr > p2_pr

    if algo == "FCFS":
        # Sort primarily by arrival time, then by PID for stable tie-breaking
        procs.sort(key=lambda x: (x["arrival_time"], x["pid"]))
        for p in procs:
            if current_time < p["arrival_time"]:
                idle_dur = p["arrival_time"] - current_time
                gantt.append({"pid": "IDLE", "start": current_time, "end": p["arrival_time"], "duration": idle_dur})
                trace_log.append(f"Time {current_time} -> {p['arrival_time']}: CPU is IDLE waiting for {p['pid']}.")
                current_time = p["arrival_time"]

            p["first_start_time"] = current_time
            start = current_time
            current_time += p["burst_time"]
            p["completion_time"] = current_time
            gantt.append({"pid": p["pid"], "start": start, "end": current_time, "duration": p["burst_time"]})
            trace_log.append(f"Time {start} -> {current_time}: Process {p['pid']} executes to completion (Burst={p['burst_time']}).")

    elif algo in ["SJF", "SJF (NON-PREEMPTIVE)"]:
        ready_queue = []
        unstarted = list(procs)
        while completed < n:
            # Check arrivals
            for p in list(unstarted):
                if p["arrival_time"] <= current_time:
                    ready_queue.append(p)
                    unstarted.remove(p)

            if not ready_queue:
                next_arr = min(p["arrival_time"] for p in unstarted)
                gantt.append({"pid": "IDLE", "start": current_time, "end": next_arr, "duration": next_arr - current_time})
                trace_log.append(f"Time {current_time} -> {next_arr}: CPU is IDLE until next arrival at {next_arr}.")
                current_time = next_arr
                continue

            # Pick process with shortest burst time (tie break by arrival time, then PID)
            ready_queue.sort(key=lambda x: (x["burst_time"], x["arrival_time"], x["pid"]))
            chosen = ready_queue.pop(0)

            chosen["first_start_time"] = current_time
            start = current_time
            current_time += chosen["burst_time"]
            chosen["completion_time"] = current_time
            completed += 1

            gantt.append({"pid": chosen["pid"], "start": start, "end": current_time, "duration": chosen["burst_time"]})
            trace_log.append(f"Time {start} -> {current_time}: Process {chosen['pid']} selected with shortest burst ({chosen['burst_time']}), finished.")

    elif algo in ["SRTF", "SJF (PREEMPTIVE)", "SHORTEST REMAINING TIME FIRST"]:
        # Preemptive SJF
        timeline = []
        max_time = 10000 # safety guard
        while completed < n and current_time < max_time:
            # Candidates in ready queue
            ready = [p for p in procs if p["arrival_time"] <= current_time and p["remaining_time"] > 0]
            if not ready:
                unstarted = [p for p in procs if p["remaining_time"] > 0]
                next_arr = min(p["arrival_time"] for p in unstarted)
                timeline.append((current_time, next_arr, "IDLE"))
                current_time = next_arr
                continue

            # Shortest remaining time (tie break by arrival time, then PID)
            ready.sort(key=lambda x: (x["remaining_time"], x["arrival_time"], x["pid"]))
            chosen = ready[0]

            if chosen["first_start_time"] is None:
                chosen["first_start_time"] = current_time

            # Run 1 unit of time
            timeline.append((current_time, current_time + 1, chosen["pid"]))
            chosen["remaining_time"] -= 1
            current_time += 1

            if chosen["remaining_time"] == 0:
                chosen["completion_time"] = current_time
                completed += 1
                trace_log.append(f"Time {current_time}: Process {chosen['pid']} completed execution.")

        # Condense timeline into gantt events
        gantt = _condense_timeline(timeline)

    elif algo in ["PRIORITY", "PRIORITY (NON-PREEMPTIVE)"]:
        ready_queue = []
        unstarted = list(procs)
        while completed < n:
            for p in list(unstarted):
                if p["arrival_time"] <= current_time:
                    ready_queue.append(p)
                    unstarted.remove(p)

            if not ready_queue:
                next_arr = min(p["arrival_time"] for p in unstarted)
                gantt.append({"pid": "IDLE", "start": current_time, "end": next_arr, "duration": next_arr - current_time})
                trace_log.append(f"Time {current_time} -> {next_arr}: CPU is IDLE.")
                current_time = next_arr
                continue

            # Sort by priority
            if priority_convention == "lower_is_higher":
                ready_queue.sort(key=lambda x: (x["priority"], x["arrival_time"], x["pid"]))
            else:
                ready_queue.sort(key=lambda x: (-x["priority"], x["arrival_time"], x["pid"]))

            chosen = ready_queue.pop(0)
            chosen["first_start_time"] = current_time
            start = current_time
            current_time += chosen["burst_time"]
            chosen["completion_time"] = current_time
            completed += 1

            gantt.append({"pid": chosen["pid"], "start": start, "end": current_time, "duration": chosen["burst_time"]})
            trace_log.append(f"Time {start} -> {current_time}: Process {chosen['pid']} (Priority={chosen['priority']}) finished.")

    elif algo in ["PRIORITY (PREEMPTIVE)"]:
        timeline = []
        max_time = 10000
        while completed < n and current_time < max_time:
            ready = [p for p in procs if p["arrival_time"] <= current_time and p["remaining_time"] > 0]
            if not ready:
                unstarted = [p for p in procs if p["remaining_time"] > 0]
                next_arr = min(p["arrival_time"] for p in unstarted)
                timeline.append((current_time, next_arr, "IDLE"))
                current_time = next_arr
                continue

            if priority_convention == "lower_is_higher":
                ready.sort(key=lambda x: (x["priority"], x["arrival_time"], x["pid"]))
            else:
                ready.sort(key=lambda x: (-x["priority"], x["arrival_time"], x["pid"]))

            chosen = ready[0]
            if chosen["first_start_time"] is None:
                chosen["first_start_time"] = current_time

            timeline.append((current_time, current_time + 1, chosen["pid"]))
            chosen["remaining_time"] -= 1
            current_time += 1

            if chosen["remaining_time"] == 0:
                chosen["completion_time"] = current_time
                completed += 1
                trace_log.append(f"Time {current_time}: Process {chosen['pid']} (Priority={chosen['priority']}) finished.")

        gantt = _condense_timeline(timeline)

    elif algo in ["ROUND ROBIN", "RR"]:
        time_q = max(1, int(time_quantum))
        procs.sort(key=lambda x: (x["arrival_time"], x["pid"]))
        ready_queue: List[Dict[str, Any]] = []
        in_queue = set()
        unstarted = list(procs)

        # Initial arrival
        for p in list(unstarted):
            if p["arrival_time"] <= current_time:
                ready_queue.append(p)
                in_queue.add(p["pid"])
                unstarted.remove(p)

        while completed < n:
            if not ready_queue:
                if unstarted:
                    next_arr = min(p["arrival_time"] for p in unstarted)
                    gantt.append({"pid": "IDLE", "start": current_time, "end": next_arr, "duration": next_arr - current_time})
                    trace_log.append(f"Time {current_time} -> {next_arr}: CPU is IDLE.")
                    current_time = next_arr
                    for p in list(unstarted):
                        if p["arrival_time"] <= current_time:
                            ready_queue.append(p)
                            in_queue.add(p["pid"])
                            unstarted.remove(p)
                else:
                    break
                continue

            current_proc = ready_queue.pop(0)
            in_queue.remove(current_proc["pid"])

            if current_proc["first_start_time"] is None:
                current_proc["first_start_time"] = current_time

            exec_time = min(time_q, current_proc["remaining_time"])
            start_t = current_time
            current_time += exec_time
            current_proc["remaining_time"] -= exec_time

            gantt.append({"pid": current_proc["pid"], "start": start_t, "end": current_time, "duration": exec_time})

            # Check new arrivals during this execution window
            for p in list(unstarted):
                if p["arrival_time"] <= current_time:
                    ready_queue.append(p)
                    in_queue.add(p["pid"])
                    unstarted.remove(p)

            if current_proc["remaining_time"] == 0:
                current_proc["completion_time"] = current_time
                completed += 1
                trace_log.append(f"Time {start_t} -> {current_time}: Process {current_proc['pid']} executed {exec_time} units and finished.")
            else:
                ready_queue.append(current_proc)
                in_queue.add(current_proc["pid"])
                trace_log.append(f"Time {start_t} -> {current_time}: Process {current_proc['pid']} executed quantum {exec_time} units (remaining={current_proc['remaining_time']}), returned to ready queue.")
    else:
        return {"error": f"Unsupported CPU scheduling algorithm: {algorithm}"}

    # Compute per-process metrics according to textbook formulas:
    # Turnaround Time = Completion Time - Arrival Time
    # Waiting Time = Turnaround Time - Burst Time
    # Response Time = First Start Time - Arrival Time
    results_table = []
    total_tat = 0
    total_wt = 0
    total_rt = 0

    # Sort results table by PID for consistent presentation
    for p in sorted(procs, key=lambda x: x["pid"]):
        ct = p["completion_time"]
        at = p["arrival_time"]
        bt = p["burst_time"]
        fst = p["first_start_time"] if p["first_start_time"] is not None else at
        tat = ct - at
        wt = tat - bt
        rt = fst - at

        total_tat += tat
        total_wt += wt
        total_rt += rt

        results_table.append({
            "pid": p["pid"],
            "arrival_time": at,
            "burst_time": bt,
            "priority": p["priority"],
            "first_start_time": fst,
            "completion_time": ct,
            "turnaround_time": tat,
            "waiting_time": wt,
            "response_time": rt
        })

    avg_tat = round(total_tat / n, 2)
    avg_wt = round(total_wt / n, 2)
    avg_rt = round(total_rt / n, 2)

    total_time = current_time
    busy_time = sum(p["burst_time"] for p in procs)
    cpu_utilization = round((busy_time / total_time * 100), 2) if total_time > 0 else 0.0
    throughput = round(n / total_time, 4) if total_time > 0 else 0.0

    return {
        "algorithm": algo,
        "processes": results_table,
        "gantt_chart": gantt,
        "trace_log": trace_log,
        "metrics": {
            "average_turnaround_time": avg_tat,
            "average_waiting_time": avg_wt,
            "average_response_time": avg_rt,
            "total_execution_time": total_time,
            "cpu_utilization_percent": cpu_utilization,
            "throughput": throughput
        }
    }


def _condense_timeline(timeline: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    """Merges adjacent unit time intervals of the same process into single Gantt blocks."""
    if not timeline:
        return []
    condensed: List[Dict[str, Any]] = []
    cur_start, cur_end, cur_pid = timeline[0]
    for s, e, pid in timeline[1:]:
        if pid == cur_pid and s == cur_end:
            cur_end = e
        else:
            condensed.append({"pid": cur_pid, "start": cur_start, "end": cur_end, "duration": cur_end - cur_start})
            cur_start, cur_end, cur_pid = s, e, pid
    condensed.append({"pid": cur_pid, "start": cur_start, "end": cur_end, "duration": cur_end - cur_start})
    return condensed


# ============================================================================
# 2. PAGE REPLACEMENT ALGORITHMS
# ============================================================================

def simulate_page_replacement(
    reference_string: List[Any],
    num_frames: int,
    algorithm: str = "FIFO"
) -> Dict[str, Any]:
    """
    Simulates page replacement for a reference string.
    Supported algorithms: FIFO, LRU, Optimal.
    Returns step-by-step frame matrix, hit/fault indicators, victim pages, and totals.
    """
    if not reference_string:
        return {"error": "Reference string is empty"}
    if num_frames <= 0:
        return {"error": "Number of frames must be at least 1"}

    algo = algorithm.upper().strip()
    refs = [str(x).strip() for x in reference_string if str(x).strip()]
    if not refs:
        return {"error": "No valid pages in reference string"}

    frames: List[Optional[str]] = [None] * num_frames
    steps: List[Dict[str, Any]] = []
    page_faults = 0
    page_hits = 0

    # Data structures for replacement policies
    fifo_queue: List[str] = [] # tracks insertion order
    lru_history: Dict[str, int] = {} # tracks last reference step index

    for step_idx, page in enumerate(refs):
        is_hit = page in frames
        victim = None
        action_desc = ""

        if is_hit:
            page_hits += 1
            action_desc = f"Hit: Page {page} is already in frame."
            if algo == "LRU":
                lru_history[page] = step_idx
        else:
            page_faults += 1
            # Check if empty frame available
            if None in frames:
                empty_idx = frames.index(None)
                frames[empty_idx] = page
                fifo_queue.append(page)
                lru_history[page] = step_idx
                action_desc = f"Page Fault: Loaded Page {page} into empty Frame {empty_idx}."
            else:
                # Replacement needed
                if algo == "FIFO":
                    victim = fifo_queue.pop(0)
                    rep_idx = frames.index(victim)
                    frames[rep_idx] = page
                    fifo_queue.append(page)
                    action_desc = f"Page Fault: Replaced Page {victim} (oldest in memory) with Page {page}."

                elif algo == "LRU":
                    # Find page with minimum last access step index
                    victim = min(frames, key=lambda p: lru_history.get(p, -1))
                    rep_idx = frames.index(victim)
                    frames[rep_idx] = page
                    lru_history[page] = step_idx
                    action_desc = f"Page Fault: Replaced Page {victim} (least recently used) with Page {page}."

                elif algo in ["OPTIMAL", "OPT"]:
                    # Look ahead in refs[step_idx + 1:]
                    future_refs = refs[step_idx + 1:]
                    victim_candidate = None
                    farthest_next_use = -1

                    for f_page in frames:
                        if f_page not in future_refs:
                            victim_candidate = f_page
                            break
                        else:
                            next_use = future_refs.index(f_page)
                            if next_use > farthest_next_use:
                                farthest_next_use = next_use
                                victim_candidate = f_page

                    victim = victim_candidate
                    rep_idx = frames.index(victim)
                    frames[rep_idx] = page
                    action_desc = f"Page Fault: Replaced Page {victim} (not used for longest time in future) with Page {page}."
                else:
                    return {"error": f"Unsupported page replacement algorithm: {algorithm}"}

        steps.append({
            "step": step_idx + 1,
            "page": page,
            "frames": list(frames),
            "is_hit": is_hit,
            "status": "Hit" if is_hit else "Fault",
            "victim": victim,
            "description": action_desc
        })

    total_refs = len(refs)
    fault_ratio = round(page_faults / total_refs, 4) if total_refs > 0 else 0.0
    hit_ratio = round(page_hits / total_refs, 4) if total_refs > 0 else 0.0

    return {
        "algorithm": algo,
        "num_frames": num_frames,
        "reference_string": refs,
        "steps": steps,
        "summary": {
            "total_references": total_refs,
            "page_faults": page_faults,
            "page_hits": page_hits,
            "fault_ratio": fault_ratio,
            "hit_ratio": hit_ratio,
            "fault_percentage": round(fault_ratio * 100, 2),
            "hit_percentage": round(hit_ratio * 100, 2)
        }
    }


def compare_page_replacements(
    reference_string: List[Any],
    num_frames: int
) -> Dict[str, Any]:
    """Runs FIFO, LRU, and Optimal on the exact same input for side-by-side comparison."""
    fifo_res = simulate_page_replacement(reference_string, num_frames, "FIFO")
    lru_res = simulate_page_replacement(reference_string, num_frames, "LRU")
    opt_res = simulate_page_replacement(reference_string, num_frames, "Optimal")
    return {
        "reference_string": reference_string,
        "num_frames": num_frames,
        "FIFO": fifo_res,
        "LRU": lru_res,
        "Optimal": opt_res
    }


# ============================================================================
# 3. DEADLOCK SAFETY & BANKER'S ALGORITHM
# ============================================================================

def run_bankers_safety_algorithm(
    process_names: List[str],
    resource_names: List[str],
    allocation_matrix: List[List[int]],
    max_matrix: List[List[int]],
    available_vector: List[int]
) -> Dict[str, Any]:
    """
    Executes Dijkstra's Banker's Safety Algorithm deterministically.
    Calculates Need = Max - Allocation.
    Finds a safe sequence if one exists, tracking intermediate Work and Finish states.
    """
    n = len(process_names)
    m = len(resource_names)

    # Validations
    if n == 0 or m == 0:
        return {"error": "Processes and resources cannot be empty"}
    if len(allocation_matrix) != n or len(max_matrix) != n:
        return {"error": "Allocation and Max matrices must match the number of processes"}
    if len(available_vector) != m:
        return {"error": "Available vector length must match the number of resource types"}

    # Compute Need Matrix and validate bounds
    need_matrix: List[List[int]] = []
    for i in range(n):
        if len(allocation_matrix[i]) != m or len(max_matrix[i]) != m:
            return {"error": f"Row {i} dimension mismatch with resource count {m}"}
        row = []
        for j in range(m):
            alloc = int(allocation_matrix[i][j])
            mx = int(max_matrix[i][j])
            if alloc < 0 or mx < 0:
                return {"error": f"Resource counts cannot be negative at P{i}, R{j}"}
            if alloc > mx:
                return {"error": f"Allocation ({alloc}) cannot exceed Max ({mx}) for process {process_names[i]}, resource {resource_names[j]}"}
            row.append(mx - alloc)
        need_matrix.append(row)

    work = [int(x) for x in available_vector]
    finish = [False] * n
    safe_sequence: List[str] = []
    step_trace: List[Dict[str, Any]] = []

    step_num = 1
    while len(safe_sequence) < n:
        found_process = False
        for i in range(n):
            if not finish[i]:
                # Check if Need[i] <= Work
                can_allocate = all(need_matrix[i][j] <= work[j] for j in range(m))
                if can_allocate:
                    prev_work = list(work)
                    # Work = Work + Allocation[i]
                    for j in range(m):
                        work[j] += allocation_matrix[i][j]
                    finish[i] = True
                    safe_sequence.append(process_names[i])
                    found_process = True

                    step_trace.append({
                        "step": step_num,
                        "process": process_names[i],
                        "need": list(need_matrix[i]),
                        "work_before": prev_work,
                        "allocation": list(allocation_matrix[i]),
                        "work_after": list(work),
                        "safe_sequence_so_far": list(safe_sequence),
                        "explanation": f"Process {process_names[i]} has Need {need_matrix[i]} <= Work {prev_work}. It finishes and releases its Allocation {allocation_matrix[i]}, making new Work = {work}."
                    })
                    step_num += 1
                    break

        if not found_process:
            # No process could be safely allocated
            break

    is_safe = len(safe_sequence) == n
    uncompleted = [process_names[i] for i in range(n) if not finish[i]]

    return {
        "is_safe": is_safe,
        "safe_sequence": safe_sequence if is_safe else [],
        "need_matrix": need_matrix,
        "allocation_matrix": allocation_matrix,
        "max_matrix": max_matrix,
        "available_vector": available_vector,
        "step_trace": step_trace,
        "uncompleted_processes": uncompleted,
        "status_message": (
            f"System is in a SAFE STATE. A safe execution sequence exists: <{' -> '.join(safe_sequence)}>."
            if is_safe else
            f"System is in an UNSAFE STATE! No safe sequence could complete. Blocked processes: {', '.join(uncompleted)}."
        )
    }


def run_bankers_resource_request(
    requesting_process: str,
    request_vector: List[int],
    process_names: List[str],
    resource_names: List[str],
    allocation_matrix: List[List[int]],
    max_matrix: List[List[int]],
    available_vector: List[int]
) -> Dict[str, Any]:
    """
    Simulates Resource-Request Algorithm for process Pi:
    1. If Request_i <= Need_i, go to step 2. Otherwise raise error condition (claim exceeded).
    2. If Request_i <= Available, go to step 3. Otherwise Pi must wait (resources not available).
    3. Pretend to allocate requested resources:
       Available = Available - Request_i
       Allocation_i = Allocation_i + Request_i
       Need_i = Need_i - Request_i
    4. Run safety algorithm. If safe -> Grant request. If unsafe -> Must wait.
    """
    if requesting_process not in process_names:
        return {"granted": False, "error": f"Process '{requesting_process}' not found"}

    p_idx = process_names.index(requesting_process)
    m = len(resource_names)

    if len(request_vector) != m:
        return {"granted": False, "error": f"Request vector dimension {len(request_vector)} != resource count {m}"}

    # First compute current Need
    safety_init = run_bankers_safety_algorithm(process_names, resource_names, allocation_matrix, max_matrix, available_vector)
    if "error" in safety_init:
        return {"granted": False, "error": safety_init["error"]}

    need_i = safety_init["need_matrix"][p_idx]

    # Step 1: Check Request <= Need
    for j in range(m):
        if request_vector[j] > need_i[j]:
            return {
                "granted": False,
                "error": f"Error: Process {requesting_process} requested {request_vector[j]} of {resource_names[j]}, which exceeds its declared maximum need ({need_i[j]})."
            }

    # Step 2: Check Request <= Available
    for j in range(m):
        if request_vector[j] > available_vector[j]:
            return {
                "granted": False,
                "can_grant": False,
                "must_wait": True,
                "reason": f"Resources not available: Process {requesting_process} requested {request_vector[j]} of {resource_names[j]}, but only {available_vector[j]} are currently available. Process must wait."
            }

    # Step 3: Pretend allocation
    sim_avail = [available_vector[j] - request_vector[j] for j in range(m)]
    sim_alloc = copy.deepcopy(allocation_matrix)
    for j in range(m):
        sim_alloc[p_idx][j] += request_vector[j]

    sim_safety = run_bankers_safety_algorithm(
        process_names, resource_names, sim_alloc, max_matrix, sim_avail
    )

    if sim_safety.get("is_safe"):
        return {
            "granted": True,
            "can_grant": True,
            "must_wait": False,
            "safe_sequence": sim_safety["safe_sequence"],
            "simulated_available": sim_avail,
            "simulated_allocation": sim_alloc,
            "simulated_need": sim_safety["need_matrix"],
            "step_trace": sim_safety["step_trace"],
            "reason": f"Request can be GRANTED IMMEDIATELY! The resulting state is safe with sequence: <{' -> '.join(sim_safety['safe_sequence'])}>."
        }
    else:
        return {
            "granted": False,
            "can_grant": False,
            "must_wait": True,
            "simulated_available": sim_avail,
            "reason": f"Request CANNOT be granted immediately because it leads to an UNSAFE STATE. Process {requesting_process} must wait."
        }
