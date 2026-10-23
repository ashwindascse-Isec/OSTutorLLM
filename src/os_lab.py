"""
Operating Systems Laboratory Interactive Simulation UI.
Pure deterministic calculators with Gantt charts, memory frame grids,
and Banker's algorithm safety traces.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from src.calculators import (
    calculate_cpu_scheduling,
    simulate_page_replacement,
    compare_page_replacements,
    run_bankers_safety_algorithm,
    run_bankers_resource_request
)


def render_os_lab():
    st.markdown("### Operating Systems Deterministic Laboratory")
    st.caption("Deterministic, reproducible simulations of core OS algorithms. Calculations follow exact textbook rules without model guesswork.")

    lab_tab1, lab_tab2, lab_tab3 = st.tabs([
        "CPU Scheduling Simulator",
        "Virtual Memory Page Replacement",
        "Deadlock Safety (Banker's Algorithm)"
    ])


    # ========================================================================
    # 1. CPU SCHEDULING
    # ========================================================================
    with lab_tab1:
        st.markdown("#### CPU Scheduling Algorithms")
        st.caption("Simulates execution order, computes exact Completion, Turnaround, Waiting, and Response times, and plots the execution Gantt chart.")

        col_c1, col_c2 = st.columns([1, 2])

        with col_c1:
            st.markdown("##### Configuration")
            algo_choice = st.selectbox(
                "Algorithm",
                ["FCFS", "SJF (Non-preemptive)", "SRTF (Preemptive SJF)", "Priority (Non-preemptive)", "Priority (Preemptive)", "Round Robin"],
                key="cpu_algo"
            )

            time_q = 2
            if "Round Robin" in algo_choice:
                time_q = st.number_input("Time Quantum", min_value=1, max_value=20, value=2, step=1)

            p_convention = "lower_is_higher"
            if "Priority" in algo_choice:
                p_convention = st.radio("Priority Convention", ["Lower number = Higher priority", "Higher number = Higher priority"], index=0)
                p_convention = "lower_is_higher" if "Lower" in p_convention else "higher_is_higher"

            preset_choice = st.selectbox(
                "Load Process Preset",
                ["Silberschatz FCFS Example", "Silberschatz SJF Example", "Round Robin Demo", "Preemptive Mix"],
                key="cpu_preset"
            )

            if preset_choice == "Silberschatz FCFS Example":
                default_procs = [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 24, "priority": 3},
                    {"pid": "P2", "arrival_time": 0, "burst_time": 3, "priority": 1},
                    {"pid": "P3", "arrival_time": 0, "burst_time": 3, "priority": 2},
                ]
            elif preset_choice == "Silberschatz SJF Example":
                default_procs = [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 6, "priority": 2},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 8, "priority": 1},
                    {"pid": "P3", "arrival_time": 2, "burst_time": 7, "priority": 3},
                    {"pid": "P4", "arrival_time": 3, "burst_time": 3, "priority": 4},
                ]
            elif preset_choice == "Round Robin Demo":
                default_procs = [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 5, "priority": 1},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 3, "priority": 2},
                    {"pid": "P3", "arrival_time": 2, "burst_time": 8, "priority": 3},
                    {"pid": "P4", "arrival_time": 3, "burst_time": 6, "priority": 4},
                ]
            else:
                default_procs = [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 7, "priority": 3},
                    {"pid": "P2", "arrival_time": 2, "burst_time": 4, "priority": 1},
                    {"pid": "P3", "arrival_time": 4, "burst_time": 1, "priority": 4},
                    {"pid": "P4", "arrival_time": 5, "burst_time": 4, "priority": 2},
                ]

        with col_c2:
            st.markdown("##### Process Table (Editable)")
            edited_df = st.data_editor(
                pd.DataFrame(default_procs),
                num_rows="dynamic",
                use_container_width=True,
                key="cpu_data_editor"
            )

        # Run CPU simulation
        procs_input = edited_df.to_dict(orient="records")
        res = calculate_cpu_scheduling(
            procs_input,
            algorithm=algo_choice,
            time_quantum=time_q,
            priority_convention=p_convention
        )

        if "error" in res:
            st.error(res["error"])
        else:
            st.markdown("---")
            st.markdown("##### Gantt Chart Execution Timeline")
            _render_gantt_chart(res["gantt_chart"])

            st.markdown("##### Process Results & Metrics")
            m = res["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Avg Turnaround Time", f"{m['average_turnaround_time']} ms")
            mc2.metric("Avg Waiting Time", f"{m['average_waiting_time']} ms")
            mc3.metric("Avg Response Time", f"{m['average_response_time']} ms")
            mc4.metric("CPU Utilization", f"{m['cpu_utilization_percent']}%")
            mc5.metric("Makespan / Total", f"{m['total_execution_time']} ms")

            # Table of process metrics
            res_df = pd.DataFrame(res["processes"])
            res_df.columns = ["PID", "Arrival Time", "Burst Time", "Priority", "First Start", "Completion (CT)", "Turnaround (TAT)", "Waiting (WT)", "Response (RT)"]
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            with st.expander("View Step-by-Step Transition Log"):
                for line in res["trace_log"]:
                    st.markdown(f"- {line}")

    # ========================================================================
    # 2. PAGE REPLACEMENT
    # ========================================================================
    with lab_tab2:
        st.markdown("#### Virtual Memory Page Replacement")
        st.caption("Demonstrates page faults, memory frame updates, and victim selection across FIFO, LRU, and Optimal policies.")

        pr_col1, pr_col2 = st.columns([1, 2])
        with pr_col1:
            st.markdown("##### Input Settings")
            pr_algo = st.selectbox("Policy", ["FIFO", "LRU", "Optimal", "Compare All 3 Policies Side-by-Side"], index=3)

            num_f = st.slider("Number of Frames", min_value=2, max_value=7, value=3)

            ref_preset = st.selectbox(
                "Reference String Preset",
                [
                    "Silberschatz Textbook (7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1)",
                    "Belady's Anomaly Demonstration (1,2,3,4,1,2,5,1,2,3,4,5)",
                    "Locality of Reference Loop (1,2,3,2,1,4,5,1,2,3,4,5)",
                    "Custom Input"
                ]
            )

            if "Silberschatz" in ref_preset:
                def_ref_str = "7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1"
            elif "Belady" in ref_preset:
                def_ref_str = "1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5"
            elif "Locality" in ref_preset:
                def_ref_str = "1, 2, 3, 2, 1, 4, 5, 1, 2, 3, 4, 5"
            else:
                def_ref_str = "7, 0, 1, 2, 0, 3, 0, 4, 2, 3"

        with pr_col2:
            st.markdown("##### Reference String")
            ref_input = st.text_area("Page Reference Stream (comma or space separated)", value=def_ref_str, height=100)

        raw_pages = [x.strip() for x in ref_input.replace(",", " ").split() if x.strip()]

        if not raw_pages:
            st.warning("Please enter a valid page reference string.")
        else:
            if "Compare" in pr_algo:
                st.markdown("---")
                st.markdown("##### Side-by-Side Comparison: FIFO vs LRU vs Optimal")
                comp_data = compare_page_replacements(raw_pages, num_f)

                c1, c2, c3 = st.columns(3)
                with c1:
                    f_sum = comp_data["FIFO"]["summary"]
                    st.markdown("##### FIFO")
                    st.metric("Page Faults", f"{f_sum['page_faults']} / {f_sum['total_references']}", delta=f"{f_sum['fault_percentage']}% Fault Rate", delta_color="inverse")
                    st.metric("Page Hits", f"{f_sum['page_hits']} ({f_sum['hit_percentage']}%)")

                with c2:
                    l_sum = comp_data["LRU"]["summary"]
                    st.markdown("##### LRU")
                    st.metric("Page Faults", f"{l_sum['page_faults']} / {l_sum['total_references']}", delta=f"{l_sum['fault_percentage']}% Fault Rate", delta_color="inverse")
                    st.metric("Page Hits", f"{l_sum['page_hits']} ({l_sum['hit_percentage']}%)")

                with c3:
                    o_sum = comp_data["Optimal"]["summary"]
                    st.markdown("##### Optimal (OPT)")
                    st.metric("Page Faults", f"{o_sum['page_faults']} / {o_sum['total_references']}", delta=f"{o_sum['fault_percentage']}% Fault Rate", delta_color="inverse")
                    st.metric("Page Hits", f"{o_sum['page_hits']} ({o_sum['hit_percentage']}%)")

                st.markdown("##### Frame by Frame Comparison")
                # Show tabs for each step trace
                t_fifo, t_lru, t_opt = st.tabs(["FIFO Step Trace", "LRU Step Trace", "Optimal Step Trace"])
                with t_fifo:
                    _render_page_trace_table(comp_data["FIFO"]["steps"], num_f)
                with t_lru:
                    _render_page_trace_table(comp_data["LRU"]["steps"], num_f)
                with t_opt:
                    _render_page_trace_table(comp_data["Optimal"]["steps"], num_f)

            else:
                sim_res = simulate_page_replacement(raw_pages, num_f, pr_algo)
                st.markdown("---")
                st.markdown(f"##### {pr_algo} Trace ({num_f} Frames)")
                s = sim_res["summary"]
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Total References", s["total_references"])
                sc2.metric("Page Faults", s["page_faults"], delta=f"{s['fault_percentage']}% Rate", delta_color="inverse")
                sc3.metric("Page Hits", s["page_hits"], delta=f"{s['hit_percentage']}% Rate")
                sc4.metric("Hit / Fault Ratio", f"{s['hit_ratio']} / {s['fault_ratio']}")

                _render_page_trace_table(sim_res["steps"], num_f)

    # ========================================================================
    # 3. DEADLOCK SAFETY & BANKER'S ALGORITHM
    # ========================================================================
    with lab_tab3:
        st.markdown("#### Deadlock Avoidance: Dijkstra's Banker's Algorithm")
        st.caption("Verifies system safety, derives the Need Matrix (Need = Max - Allocation), discovers safe execution sequences, and tests resource requests.")

        b_preset = st.selectbox(
            "Load Matrix Preset",
            ["Silberschatz 5 Processes & 3 Resources (Safe State)", "Unsafe State Example (Potential Deadlock)"],
            key="banker_preset"
        )

        if "Safe" in b_preset:
            p_names = ["P0", "P1", "P2", "P3", "P4"]
            r_names = ["A", "B", "C"]
            alloc_mat = [
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
            avail_vec = [3, 3, 2]
        else:
            p_names = ["P0", "P1", "P2", "P3", "P4"]
            r_names = ["A", "B", "C"]
            alloc_mat = [
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
            avail_vec = [0, 0, 0] # Unsafe!

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("##### Allocation Matrix")
            alloc_df = pd.DataFrame(alloc_mat, index=p_names, columns=r_names)
            st.dataframe(alloc_df, use_container_width=True)

        with col_m2:
            st.markdown("##### Max Demand Matrix")
            max_df = pd.DataFrame(max_mat, index=p_names, columns=r_names)
            st.dataframe(max_df, use_container_width=True)

        with col_m3:
            st.markdown("##### Available Vector")
            avail_df = pd.DataFrame([avail_vec], columns=r_names, index=["Available"])
            st.dataframe(avail_df, use_container_width=True)

        # Run Safety Algorithm
        safety_res = run_bankers_safety_algorithm(p_names, r_names, alloc_mat, max_mat, avail_vec)

        if "error" in safety_res:
            st.error(safety_res["error"])
        else:
            st.markdown("---")
            if safety_res["is_safe"]:
                seq_str = " -> ".join(safety_res["safe_sequence"])
                st.success(f"### SAFE STATE CONFIRMED\n**Safe Execution Sequence:** `< {seq_str} >`")
            else:
                st.error(f"### SYSTEM IS IN AN UNSAFE STATE\n{safety_res['status_message']}")

            # Display Computed Need Matrix
            st.markdown("##### Derived Need Matrix (Need = Max - Allocation)")
            need_df = pd.DataFrame(safety_res["need_matrix"], index=p_names, columns=r_names)
            st.dataframe(need_df, use_container_width=True)

            # Step trace
            if safety_res["step_trace"]:
                st.markdown("##### Step-by-Step Safety Algorithm Trace")
                trace_rows = []
                for s in safety_res["step_trace"]:
                    trace_rows.append({
                        "Step": s["step"],
                        "Process Selected": s["process"],
                        "Need": str(s["need"]),
                        "Work (Before)": str(s["work_before"]),
                        "Allocation Released": str(s["allocation"]),
                        "Work (After)": str(s["work_after"]),
                        "Safe Sequence": " -> ".join(s["safe_sequence_so_far"])
                    })
                st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)

            # Resource-Request simulator
            st.markdown("---")
            st.markdown("##### Simulate Additional Resource Request (Pi Request Algorithm)")

            rq_c1, rq_c2, rq_c3 = st.columns([1, 2, 1])
            with rq_c1:
                req_proc = st.selectbox("Requesting Process", p_names, index=1)
            with rq_c2:
                req_inputs = []
                r_cols = st.columns(len(r_names))
                for j, r_name in enumerate(r_names):
                    with r_cols[j]:
                        val = st.number_input(f"Req {r_name}", min_value=0, max_value=20, value=1 if j == 0 else (2 if j == 2 else 0))
                        req_inputs.append(int(val))
            with rq_c3:
                st.write("")
                st.write("")
                sim_btn = st.button("Simulate Request", type="primary")

            if sim_btn:
                req_eval = run_bankers_resource_request(
                    requesting_process=req_proc,
                    request_vector=req_inputs,
                    process_names=p_names,
                    resource_names=r_names,
                    allocation_matrix=alloc_mat,
                    max_matrix=max_mat,
                    available_vector=avail_vec
                )
                if req_eval.get("granted"):
                    st.success(f"**{req_eval['reason']}**")
                    if req_eval.get("safe_sequence"):
                        st.info(f"New Safe Sequence: `< {' -> '.join(req_eval['safe_sequence'])} >`")
                else:
                    if "error" in req_eval:
                        st.error(req_eval["error"])
                    else:
                        st.warning(f"**{req_eval.get('reason')}**")


def _render_gantt_chart(gantt_events: List[Dict[str, Any]]):
    """Draws a CSS-styled horizontal Gantt chart timeline."""
    if not gantt_events:
        st.write("No events to render.")
        return

    total_duration = gantt_events[-1]["end"] if gantt_events else 1
    if total_duration == 0:
        total_duration = 1

    colors = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#3b82f6"]
    proc_color_map = {}
    c_idx = 0

    blocks_html = []
    time_markers = [0]

    for event in gantt_events:
        pid = event["pid"]
        dur = event["duration"]
        pct = max(2.5, (dur / total_duration) * 100)
        time_markers.append(event["end"])

        if pid == "IDLE":
            bg = "#334155"
            txt_color = "#94a3b8"
        else:
            if pid not in proc_color_map:
                proc_color_map[pid] = colors[c_idx % len(colors)]
                c_idx += 1
            bg = proc_color_map[pid]
            txt_color = "#ffffff"

        blocks_html.append(
            f'<div style="flex: {dur}; min-width: 45px; background-color: {bg}; color: {txt_color}; height: 50px; '
            f'display: flex; flex-direction: column; align-items: center; justify-content: center; '
            f'border-right: 2px solid #0f172a; font-weight: 700; font-size: 13px; border-radius: 4px;">'
            f'<span>{pid}</span>'
            f'<span style="font-size: 10px; opacity: 0.85;">({event["start"]}-{event["end"]})</span>'
            f'</div>'
        )

    gantt_html = (
        f'<div style="display: flex; width: 100%; border-radius: 8px; overflow-x: auto; background-color: #0f172a; '
        f'padding: 8px; border: 1px solid #1e293b; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">'
        + "".join(blocks_html) +
        f'</div>'
    )
    st.markdown(gantt_html, unsafe_allow_html=True)


def _render_page_trace_table(steps: List[Dict[str, Any]], num_frames: int):
    """Renders page replacement step-by-step memory frame grid."""
    headers = ["Step", "Page Ref"] + [f"Frame {i}" for i in range(num_frames)] + ["Status", "Victim Page", "Action Details"]
    rows = []
    for s in steps:
        f_row = [s["frames"][i] if i < len(s["frames"]) and s["frames"][i] is not None else "-" for i in range(num_frames)]
        stat_badge = "HIT" if s["is_hit"] else "FAULT"
        vic = s["victim"] if s["victim"] is not None else "-"
        rows.append([s["step"], s["page"]] + f_row + [stat_badge, vic, s["description"]])

    df = pd.DataFrame(rows, columns=headers)

    st.dataframe(df, use_container_width=True, hide_index=True)
