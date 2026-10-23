"""
Document Library Management, Persistence, and Deduplication.
Manages library manifest and document passage stores.
Seeds default course materials on Operating Systems.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import LIBRARY_DIR, MANIFEST_FILE, DATA_DIR
from src.ingestion import extract_text_from_pdf, extract_text_from_txt, chunk_text, compute_file_hash


class DocumentLibrary:
    def __init__(self):
        self.manifest_path = MANIFEST_FILE
        self.library_dir = LIBRARY_DIR
        self._ensure_manifest()
        self.seed_default_curriculum_if_empty()

    def _ensure_manifest(self):
        if not self.manifest_path.exists():
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump({"documents": {}}, f, indent=2)

    def load_manifest(self) -> Dict[str, Any]:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"documents": {}}

    def save_manifest(self, manifest: Dict[str, Any]):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def list_documents(self) -> List[Dict[str, Any]]:
        manifest = self.load_manifest()
        return list(manifest.get("documents", {}).values())

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        manifest = self.load_manifest()
        return manifest.get("documents", {}).get(doc_id)

    def get_passages_for_docs(self, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        manifest = self.load_manifest()
        all_docs = manifest.get("documents", {})
        target_ids = doc_ids if doc_ids is not None else list(all_docs.keys())

        combined_passages = []
        for d_id in target_ids:
            passage_file = self.library_dir / f"{d_id}.json"
            if passage_file.exists():
                try:
                    with open(passage_file, "r", encoding="utf-8") as f:
                        passages = json.load(f)
                        combined_passages.extend(passages)
                except Exception:
                    pass
        return combined_passages

    def add_document(self, title: str, filename: str, content_bytes_or_str: Any, file_type: str = "txt") -> Dict[str, Any]:
        """
        Ingests and indexes a new document into the library.
        Prevents duplicate ingestion via SHA-256 hash.
        """
        file_hash = compute_file_hash(content_bytes_or_str)
        manifest = self.load_manifest()
        docs = manifest.get("documents", {})

        # Check for duplicates
        for doc_id, doc in docs.items():
            if doc.get("hash") == file_hash:
                return {
                    "success": False,
                    "is_duplicate": True,
                    "doc_id": doc_id,
                    "message": f"Document '{doc['title']}' with matching content already exists in the library."
                }

        # Extract pages
        if file_type.lower() == "pdf":
            pages = extract_text_from_pdf(content_bytes_or_str)
        else:
            if isinstance(content_bytes_or_str, bytes):
                text_str = content_bytes_or_str.decode("utf-8", errors="replace")
            else:
                text_str = str(content_bytes_or_str)
            pages = extract_text_from_txt(text_str)

        if not pages:
            return {
                "success": False,
                "error": "No readable text could be extracted from the document."
            }

        # Generate unique doc_id
        safe_name = "".join(c for c in title if c.isalnum() or c in ("-", "_")).lower()[:20]
        doc_id = f"doc_{safe_name}_{file_hash[:8]}"

        # Chunk passages
        passages = chunk_text(pages, doc_id=doc_id, doc_name=title)
        if not passages:
            return {
                "success": False,
                "error": "Document text was too short to construct valid retrievable passages."
            }

        # Save passages
        passage_file = self.library_dir / f"{doc_id}.json"
        with open(passage_file, "w", encoding="utf-8") as f:
            json.dump(passages, f, indent=2)

        # Update manifest
        doc_metadata = {
            "doc_id": doc_id,
            "title": title,
            "filename": filename,
            "hash": file_hash,
            "file_type": file_type.lower(),
            "page_count": len(pages),
            "chunk_count": len(passages),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        docs[doc_id] = doc_metadata
        manifest["documents"] = docs
        self.save_manifest(manifest)

        return {
            "success": True,
            "doc_id": doc_id,
            "chunk_count": len(passages),
            "page_count": len(pages),
            "message": f"Successfully ingested '{title}' ({len(pages)} pages, {len(passages)} passages)."
        }

    def delete_document(self, doc_id: str) -> bool:
        manifest = self.load_manifest()
        docs = manifest.get("documents", {})
        if doc_id in docs:
            del docs[doc_id]
            self.save_manifest(manifest)
            passage_file = self.library_dir / f"{doc_id}.json"
            if passage_file.exists():
                passage_file.unlink()
            return True
        return False

    def seed_default_curriculum_if_empty(self):
        """Seeds default comprehensive Operating Systems course material if library has 0 docs."""
        manifest = self.load_manifest()
        if manifest.get("documents"):
            return

        defaults_dir = DATA_DIR / "default_materials"
        defaults_dir.mkdir(parents=True, exist_ok=True)

        materials = _get_default_os_textbook_chapters()
        for mat in materials:
            file_path = defaults_dir / mat["filename"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(mat["content"])
            self.add_document(
                title=mat["title"],
                filename=mat["filename"],
                content_bytes_or_str=mat["content"],
                file_type="txt"
            )


def _get_default_os_textbook_chapters() -> List[Dict[str, str]]:
    """Authoritative textbook passages for Operating Systems (Silberschatz / Galvin)."""
    return [
        {
            "title": "OS Concepts: CPU Scheduling & Multiprogramming",
            "filename": "os_cpu_scheduling.txt",
            "content": """--- PAGE 1 ---
Chapter 5: CPU Scheduling and Multiprogramming Concepts.
CPU scheduling is the basis of multiprogrammed operating systems. By switching the CPU among processes, the operating system can make the computer more productive.
In a uniprocessor system, only one process can run at a time; any others must wait until the CPU is free and can be rescheduled. The objective of multiprogramming is to have some process running at all times, to maximize CPU utilization.

A process alternates between CPU execution (CPU burst) and I/O wait (I/O burst). Process execution begins with a CPU burst, followed by an I/O burst, which is followed by another CPU burst, and so on. An I/O-bound program typically has many short CPU bursts, whereas a CPU-bound program might have a few long CPU bursts.

--- PAGE 2 ---
CPU Scheduler and Preemption Criteria:
Whenever the CPU becomes idle, the operating system must select one of the processes in the ready queue to be executed. The selection process is carried out by the CPU scheduler (also called the short-term scheduler). The scheduler selects a process from the processes in memory that are ready to execute and allocates the CPU to that process.

CPU scheduling decisions may take place under four circumstances:
1. When a process switches from the running state to the waiting state (for example, as a result of an I/O request or an invocation of wait()).
2. When a process switches from the running state to the ready state (for example, when an interrupt occurs or when a timer expires).
3. When a process switches from the waiting state to the ready state (for example, at completion of I/O).
4. When a process terminates.

Under circumstances 1 and 4, there is no choice in terms of scheduling. A new process must be selected. When scheduling takes place only under circumstances 1 and 4, we say that the scheduling scheme is non-preemptive or cooperative; otherwise, it is preemptive.

--- PAGE 3 ---
Scheduling Criteria and Performance Metrics:
Different CPU-scheduling algorithms have different properties, and the choice of a particular algorithm may favor one class of processes over another. The criteria include:
1. CPU utilization: We want to keep the CPU as busy as possible. Conceptually, CPU utilization can range from 0 to 100 percent. In a real system, it should range from 40 percent (for a lightly loaded system) to 90 percent (for a heavily loaded system).
2. Throughput: If the CPU is busy executing processes, then work is being done. One measure of work is the number of processes that are completed per time unit, called throughput.
3. Turnaround time: From the point of view of a particular process, the important criterion is how long it takes to execute that process. The interval from the time of submission of a process to the time of completion is the turnaround time. Turnaround time is the sum of the periods spent waiting in the ready queue, executing on the CPU, and doing I/O:
Turnaround Time = Completion Time - Arrival Time.
4. Waiting time: The CPU-scheduling algorithm does not affect the amount of time during which a process executes or does I/O; it affects only the amount of time that a process spends waiting in the ready queue. Waiting time is the sum of the periods spent waiting in the ready queue:
Waiting Time = Turnaround Time - Burst Time.
5. Response time: In an interactive system, turnaround time may not be the best criterion. Often, a process can produce some output fairly early and can continue computing new results while previous results are being output to the user. Another measure is the time from the submission of a request until the first response is produced:
Response Time = First Start Time - Arrival Time.

--- PAGE 4 ---
Scheduling Algorithms:
1. First-Come, First-Served (FCFS) Scheduling:
By far the simplest CPU-scheduling algorithm is the first-come, first-served (FCFS) algorithm. With this scheme, the process that requests the CPU first is allocated the CPU first. The implementation of the FCFS policy is easily managed with a FIFO queue.
A major disadvantage is the convoy effect: all other processes wait for the one big process to get off the CPU. This results in lower CPU and device utilization than might be possible if shorter processes were allowed to go first.

2. Shortest-Job-First (SJF) Scheduling:
This algorithm associates with each process the length of the process's next CPU burst. When the CPU is available, it is assigned to the process that has the smallest next CPU burst. If the next CPU bursts of two processes are the same, FCFS scheduling is used to break the tie.
The SJF scheduling algorithm is provably optimal, in that it gives the minimum average waiting time for a given set of processes. The preemptive version of SJF is known as Shortest-Remaining-Time-First (SRTF).

3. Round-Robin (RR) Scheduling:
The round-robin (RR) scheduling algorithm is designed especially for timesharing systems. It is similar to FCFS scheduling, but preemption is added to enable the system to switch between processes. A small unit of time, called a time quantum or time slice, is defined. A time quantum is generally from 10 to 100 milliseconds. The ready queue is treated as a circular queue. The CPU scheduler goes around the ready queue, allocating the CPU to each process for a time interval of up to 1 time quantum. If the time quantum is extremely large, RR behaves like FCFS. If it is extremely small, context-switch overhead dominates.
"""
        },
        {
            "title": "OS Concepts: Deadlocks, Prevention, and Avoidance",
            "filename": "os_deadlocks.txt",
            "content": """--- PAGE 1 ---
Chapter 7: Deadlocks in Operating Systems.
In a multiprogramming environment, several processes may compete for a finite number of resources. A process requests resources; if the resources are not available at that time, the process enters a waiting state. Sometimes, a waiting process is never again able to change state, because the resources it has requested are held by other waiting processes. This situation is called a deadlock.

The Four Necessary Conditions for Deadlock (Coffman Conditions):
A deadlock situation can arise if and only if the following four conditions hold simultaneously in a system:
1. Mutual Exclusion: At least one resource must be held in a non-shareable mode; that is, only one process at a time can use the resource. If another process requests that resource, the requesting process must be delayed until the resource has been released.
2. Hold and Wait: A process must be currently holding at least one resource and waiting to acquire additional resources that are currently being held by other processes.
3. No Preemption: Resources cannot be preempted; that is, a resource can be released only voluntarily by the process holding it, after that process has completed its task.
4. Circular Wait: A closed chain of processes exists such that each process holds at least one resource needed by the next process in the chain. That is, P0 is waiting for a resource held by P1, P1 is waiting for P2, ..., and Pn is waiting for P0.

--- PAGE 2 ---
Methods for Handling Deadlocks:
Generally speaking, we can deal with the deadlock problem in one of three ways:
1. We can use a protocol to prevent or avoid deadlocks, ensuring that the system will never enter a deadlocked state.
2. We can allow the system to enter a deadlocked state, detect it, and recover.
3. We can ignore the problem altogether and pretend that deadlocks never occur in the system. This solution is the one used by most operating systems, including Linux and Windows; it is then up to the application developer to write programs that handle deadlocks.

Deadlock Prevention:
Deadlock prevention provides a set of methods for ensuring that at least one of the four necessary conditions cannot hold:
- Mutual Exclusion: Read-only files do not require mutual exclusion. However, some resources are intrinsically non-shareable (such as a mutex lock or printer).
- Hold and Wait: To prevent hold-and-wait, we must guarantee that whenever a process requests a resource, it does not hold any other resources. One protocol requires each process to request and be allocated all its resources before execution begins.
- No Preemption: If a process that is holding some resources requests another resource that cannot be immediately allocated to it, then all resources currently being held are preempted.
- Circular Wait: Impose a total ordering of all resource types, and require that each process requests resources in an increasing order of enumeration.

--- PAGE 3 ---
Deadlock Avoidance and Banker's Algorithm:
Deadlock avoidance requires that the operating system be given additional information in advance concerning which resources a process will request and use during its lifetime.
A state is safe if the system can allocate resources to each process (up to its maximum) in some order and still avoid a deadlock. More formally, a system is in a safe state only if there exists a safe sequence of processes <P1, P2, ..., Pn>.
If a system is in a safe state, there can be no deadlock. Conversely, an unsafe state is not necessarily a deadlock, but an unsafe state may lead to a deadlock.

Banker's Algorithm:
Banker's algorithm is a deadlock avoidance algorithm applicable to a system with multiple instances of each resource type.
Data Structures:
- Available: A vector of length m indicates the number of available resources of each type.
- Max: An n x m matrix defines the maximum demand of each process.
- Allocation: An n x m matrix defines the number of resources of each type currently allocated to each process.
- Need: An n x m matrix indicates the remaining resource need of each process:
Need[i][j] = Max[i][j] - Allocation[i][j].

Safety Algorithm:
1. Let Work and Finish be vectors of length m and n, respectively. Initialize Work = Available and Finish[i] = False for i = 0, 1, ..., n-1.
2. Find an index i such that both:
   a. Finish[i] == False
   b. Need[i] <= Work
   If no such i exists, go to step 4.
3. Work = Work + Allocation[i], Finish[i] = True. Go to step 2.
4. If Finish[i] == True for all i, then the system is in a safe state.
"""
        },
        {
            "title": "OS Concepts: Virtual Memory, Paging, and Page Replacement",
            "filename": "os_virtual_memory.txt",
            "content": """--- PAGE 1 ---
Chapter 9: Virtual Memory and Paging.
Virtual memory is a technique that allows the execution of processes that are not completely in memory. One major advantage of this scheme is that programs can be larger than physical memory. Further, virtual memory abstracts main memory into an extremely large, uniform array of storage, separating logical memory as viewed by the programmer from physical memory.

Paging:
Paging is a memory-management scheme that permits the physical address space of a process to be noncontiguous. Paging avoids external fragmentation and the need for compaction.
Physical memory is broken into fixed-sized blocks called frames. Logical memory is also broken into blocks of the same size called pages. When a process is to be executed, its pages are loaded into any available memory frames from backing store.
Every address generated by the CPU is divided into two parts: a page number (p) and a page offset (d). The page number is used as an index into a page table. The page table contains the base address of each page in physical memory.

--- PAGE 2 ---
Demand Paging and Page Faults:
With demand paging, pages are only loaded into memory when they are demanded during program execution; pages that are never accessed are thus never loaded into physical memory.
When a process tries to access a page that was not brought into memory, the page table entry indicates that the page is invalid. Access to a page marked invalid causes a page fault. The paging hardware, in translating the address through the page table, will notice that the invalid bit is set, causing a trap to the operating system.

Handling a Page Fault (Steps):
1. Check an internal table for this process to determine whether the reference was a valid or an invalid memory access.
2. If the reference was invalid, terminate the process. If it was valid, but page has not been brought in, page it in.
3. Find a free frame (by taking one from the free-frame list or selecting a victim frame via page replacement).
4. Schedule a secondary storage operation to read the desired page into the newly allocated frame.
5. When the storage read is complete, modify the internal table and page table to indicate that the page is now in memory.
6. Restart the instruction that was interrupted by the trap.

--- PAGE 3 ---
Page Replacement Algorithms:
When a page fault occurs and no free frame is available on the free-frame list, the operating system must replace one of the resident pages.
1. FIFO Page Replacement:
The simplest page-replacement algorithm is a first-in, first-out (FIFO) algorithm. When a page must be replaced, the oldest page is chosen. A FIFO queue can be used to track the ages of all pages in memory.
Disadvantage: FIFO suffers from Belady's anomaly: for some page-replacement algorithms, the page-fault rate may increase as the number of allocated frames increases.

2. Least Recently Used (LRU) Page Replacement:
LRU replacement associates with each page the time of that page's last use. When a page must be replaced, LRU chooses that page that has not been used for the longest period of time. LRU is considered a good approximation of the optimal algorithm and does not suffer from Belady's anomaly.

3. Optimal Page Replacement (OPT):
An optimal page-replacement algorithm has the lowest page-fault rate of all algorithms. It replaces the page that will not be used for the longest period of time in the future.
Because the optimal algorithm requires future knowledge of the reference string, it is impossible to implement in a general-purpose operating system. It is used mainly for comparison and benchmark studies.
"""
        },
        {
            "title": "OS Concepts: Process Synchronization and Deadlock Interplay",
            "filename": "os_process_synchronization.txt",
            "content": """--- PAGE 1 ---
Chapter 6: Process Synchronization.
Cooperant processes can either directly share a logical address space (that is, both code and data) or be allowed to share data only through files or messages. Concurrent access to shared data may result in data inconsistency.
A race condition occurs when several processes access and manipulate the same data concurrently and the outcome of the execution depends on the particular order in which the access takes place.

The Critical-Section Problem:
A critical section is a piece of code that accesses shared resources (such as shared variables, files, or hardware devices). When one process is executing in its critical section, no other process is allowed to execute in its critical section.
A solution to the critical-section problem must satisfy the following three requirements:
1. Mutual Exclusion: If process Pi is executing in its critical section, then no other processes can be executing in their critical sections.
2. Progress: If no process is executing in its critical section and some processes wish to enter their critical sections, then only those processes that are not executing in their remainder sections can participate in deciding which will enter its critical section next, and this selection cannot be postponed indefinitely.
3. Bounded Waiting: There must be a bound, or limit, on the number of times that other processes are allowed to enter their critical sections after a process has made a request to enter its critical section and before that request is granted.

--- PAGE 2 ---
Synchronization Tools:
1. Mutex Locks:
A mutex (mutual exclusion) lock is the simplest tool for synchronization. A process must acquire the lock before entering a critical section; it releases the lock when it exits the critical section. A mutex lock has a boolean variable whose value indicates if the lock is available or not.

2. Semaphores:
A semaphore S is an integer variable that, apart from initialization, is accessed only through two standard atomic operations: wait() and signal() (historically termed P and V).
- wait(S):
  while (S <= 0)
      ; // busy wait
  S--;
- signal(S):
  S++;

Counting semaphores have an unrestricted domain; binary semaphores can range only between 0 and 1 (acting essentially as mutex locks).
Classical synchronization problems include the Bounded-Buffer (Producer-Consumer) Problem, the Readers-Writers Problem, and the Dining-Philosophers Problem.
"""
        },
        {
            "title": "OS Concepts: File Systems and Disk Scheduling",
            "filename": "os_file_systems.txt",
            "content": """--- PAGE 1 ---
Chapter 10 & 11: File Systems and Mass Storage Structure.
The file system provides the mechanism for on-line storage and access to both data and programs of the operating system and all the users of the computer system.
A file is an abstract data type defined by its attributes: name, identifier, type, location, size, protection, and time/date/user identification.

File Allocation Methods:
1. Contiguous Allocation: Each file occupies a set of contiguous blocks on the disk. Provides excellent read performance, but suffers from external fragmentation and difficulty in growing files.
2. Linked Allocation: Each file is a linked list of disk blocks; the blocks may be scattered anywhere on the disk. Solves external fragmentation, but does not support efficient direct/random access.
3. Indexed Allocation: Brings all the pointers together into one location: the index block. Supports direct access without external fragmentation, but has index block overhead for small files.

--- PAGE 2 ---
Disk Scheduling Algorithms:
Because disk access involves physical movement of the disk head across tracks (seek time), disk scheduling algorithms attempt to minimize seek distance.
1. FCFS (First-Come, First-Served): Service requests in arrival order. Simple, but large head movements can result.
2. SSTF (Shortest Seek Time First): Selects the request with the minimum seek time from the current head position. May cause starvation of distant requests.
3. SCAN (Elevator Algorithm): The disk arm starts at one end of the disk and moves toward the other end, servicing requests along the way until it reaches the end, where the direction of head movement is reversed.
4. C-SCAN (Circular SCAN): Moves the head from one end of the disk to the other, servicing requests along the way. When it reaches the other end, however, it immediately returns to the beginning of the disk without servicing any requests on the return trip.
"""
        }
    ]
