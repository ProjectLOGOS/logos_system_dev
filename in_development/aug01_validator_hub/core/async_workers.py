"""
async_workers.py

Defines background worker processes for Telos embedding, planner rollouts,
and graph writes to enable low-latency immediate responses and
async deep computations.
"""
import threading
import queue
import time

# Task queue for heavy modules
task_queue = queue.Queue()

# Worker function template
def worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        func, args = task
        try:
            func(*args)
        except Exception as e:
            print(f"Async worker error: {e}")
        finally:
            task_queue.task_done()

# Start a pool of workers
_workers = []
for i in range(4):
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    _workers.append(t)

# API to submit tasks
def submit_async(func, *args):
    """Schedule a function to run asynchronously"""
    task_queue.put((func, args))

# Graceful shutdown
def shutdown_workers():
    for _ in _workers:
        task_queue.put(None)
    for t in _workers:
        t.join()
