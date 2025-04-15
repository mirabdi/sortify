import os
import time
import random
import numpy as np
import pandas as pd
from typing import List, Callable, Tuple, Optional, Dict, Any
from tqdm import tqdm
from sorting import SortingAlgorithms
import signal
from contextlib import contextmanager
import threading
import tracemalloc
import math

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    timer = threading.Timer(seconds, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()

class SortingBenchmark:
    def __init__(self):
        self.quadratic_algorithms = {
            'Bubble Sort': SortingAlgorithms.bubble_sort,
            'Selection Sort': SortingAlgorithms.selection_sort,
            'Insertion Sort': SortingAlgorithms.insertion_sort,
            'Cocktail Sort': SortingAlgorithms.cocktail_sort,
        }
        
        self.linearithmic_algorithms = {
            'Merge Sort': SortingAlgorithms.merge_sort,
            'Heap Sort': SortingAlgorithms.heap_sort,
            'Quick Sort': SortingAlgorithms.quick_sort,
            'Timsort': SortingAlgorithms.timsort,
            'Introsort': SortingAlgorithms.introsort,
            'Library Sort': SortingAlgorithms.library_sort,
            'Tournament Sort': SortingAlgorithms.tournament_sort,
            'Comb Sort': SortingAlgorithms.comb_sort
        }
        
        self.algorithms = {**self.quadratic_algorithms, **self.linearithmic_algorithms}
        
        self.stability_results: Dict[str, bool] = {} 
        self.checked_stability: set[str] = set()

    def _check_stability(self, algo_name: str, algo_func: Callable) -> bool:
        if algo_name in self.checked_stability:
            return self.stability_results[algo_name]

        print(f"Checking stability for {algo_name}...")
        n = 100
        data_to_sort: List[Tuple[int, int]] = []
        random.seed(42)
        for i in range(n):
            key = random.randint(1, n // 5) 
            data_to_sort.append((key, i))

        sorted_data = algo_func(data_to_sort.copy())

        is_stable = True
        for i in range(len(sorted_data) - 1):
            if sorted_data[i][0] == sorted_data[i+1][0]:
                if sorted_data[i][1] > sorted_data[i+1][1]:
                    is_stable = False
                    break
        
        print(f"{algo_name} is Stable: {is_stable}")
        self.stability_results[algo_name] = is_stable
        self.checked_stability.add(algo_name)
        return is_stable

    def generate_data(self, size: int, data_type: str = 'random') -> List[int]:
        if data_type == 'random':
            return [random.randint(1, size) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(1, size + 1))
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'nearly_sorted':
            arr = list(range(1, size + 1))
            swaps = max(1, size // 20)
            for _ in range(swaps):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def measure_performance(self, func: Callable, arr: List[Any], timeout: float = 20.0) -> Tuple[Optional[float], Optional[float]]:
        arr_copy = arr.copy()
        execution_time = None
        peak_memory_bytes = None
        
        try:
            tracemalloc.start()
            tracemalloc.clear_traces()
            
            with time_limit(timeout):
                start_time = time.time()
                func(arr_copy)
                end_time = time.time()
                execution_time = end_time - start_time
            
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        except (TimeoutException, KeyboardInterrupt):
            print(f"Algorithm timed out after {timeout} seconds or was interrupted.")
            tracemalloc.stop()
            return None, None
        except Exception as e:
            print(f"An error occurred during measurement: {e}")
            tracemalloc.stop()
            return None, None

        return execution_time, peak_memory_bytes

    def run_benchmark(self, input_sizes: List[int], data_types: List[str], trials: int = 10, timeout: float = 10.0) -> pd.DataFrame:
        results = []
        trial_logs = []  
        timed_out_algorithms_by_type = {data_type: set() for data_type in data_types}
        
        
            
        for size in tqdm(input_sizes, desc="Processing array input_sizes"):
            log2_size = int(math.log2(size))
            
            for data_type in tqdm(data_types, desc=f"Testing data types (size {size}, log2={log2_size})", leave=False):
                print(f"\nTesting {data_type} arrays of size {size} (log2={log2_size})")
                size_data_type_results = []

                for algo_name, algo_func in self.algorithms.items():
                    if algo_name not in self.checked_stability:
                         self._check_stability(algo_name, algo_func)
                    is_stable = self.stability_results.get(algo_name, False) 

                    if algo_name in timed_out_algorithms_by_type[data_type]:
                        print(f"Skipping {algo_name} for size {size}, data type {data_type} (previously timed out)")
                        continue

                    times = []
                    peak_mems_bytes = []
                    timed_out_this_run = False
                    
                    for trial in range(trials):
                        data = self.generate_data(size, data_type)
                        execution_time, peak_memory = self.measure_performance(algo_func, data, timeout)
                        
                        if execution_time is None or peak_memory is None:
                            timed_out_this_run = True
                            print(f"{algo_name}: Timed out or errored for size {size}, data type {data_type}")
                            timed_out_algorithms_by_type[data_type].add(algo_name)
                            break 
                        
                        times.append(execution_time)
                        peak_mems_bytes.append(peak_memory)
                        
                        trial_logs.append({
                            'Algorithm': algo_name,
                            'Complexity': 'Quadratic' if algo_name in self.quadratic_algorithms else 'Linearithmic',
                            'Stable': is_stable,
                            'Size': size,
                            'Log2_Size': log2_size,
                            'Data Type': data_type,
                            'Trial': trial + 1,
                            'Time (s)': execution_time,
                            'Peak Memory (MiB)': peak_memory / (1024 * 1024)
                        })
                        
                        print(f"{algo_name}: Trial {trial + 1}/{trials} - Time: {execution_time:.4f}s, Peak Mem: {peak_memory / (1024*1024):.4f} MiB")
                    
                    if not timed_out_this_run and times:
                        avg_time = np.mean(times)
                        std_time = np.std(times)
                        avg_peak_mem_bytes = np.mean(peak_mems_bytes)
                        std_peak_mem_bytes = np.std(peak_mems_bytes)
                        
                        complexity = 'Quadratic' if algo_name in self.quadratic_algorithms else 'Linearithmic'
                        size_data_type_results.append({
                            'Algorithm': algo_name,
                            'Complexity': complexity,
                            'Stable': is_stable,
                            'Size': size,
                            'Log2_Size': log2_size,
                            'Data Type': data_type,
                            'Time (s)': avg_time,
                            'Std Dev Time': std_time,
                            'Peak Memory (MiB)': avg_peak_mem_bytes / (1024 * 1024),
                            'Std Dev Memory (MiB)': std_peak_mem_bytes / (1024 * 1024)
                        })
                
                results.extend(size_data_type_results)
            
        results_df = pd.DataFrame(results)
        trial_logs_df = pd.DataFrame(trial_logs)
        
        results_df.to_csv('data/benchmark_results.csv', index=False)
        trial_logs_df.to_csv('data/benchmark_trials.csv', index=False)
        
        return results_df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    benchmark = SortingBenchmark()
    
    input_sizes = []
    start_size = 64
    max_size = 1000000
    while start_size < max_size:
        start_size *= 2
        input_sizes.append(start_size)

    data_types = ['random', 'sorted', 'reverse', 'nearly_sorted']

    results = benchmark.run_benchmark(input_sizes, data_types, timeout=10.0, trials=10)
    
    print("\nResults saved to:")
    print("- data/benchmark_results.csv (averages)")
    print("- data/benchmark_trials.csv (individual trials)")