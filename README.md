# Sortify - Sorting Algorithm Implementation and Benchmarking

This project implements various sorting algorithms and provides comprehensive benchmarking tools to analyze their performance.

## Algorithms Implemented

### Classical Sorting Algorithms
- Bubble Sort
- Selection Sort
- Insertion Sort
- Merge Sort
- Quick Sort
- Heap Sort

### Contemporary Sorting Algorithms
- Shell Sort (optimization of insertion sort)
- Timsort (Python's built-in sorting algorithm)
- Introsort (C++ STL's sorting algorithm)

## Features

- Codes for both classical and modern sorting algorithms
-  Testing with various test cases
- Time and memory benchmarking with different input types:
  - Random data
  - Already sorted data
  - Reverse sorted data
  - Nearly sorted data (95% sorted)
- Stability testing for stable sorting algorithms
- Performance visualization with matplotlib
- CSV export of benchmark results

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest test_sorting.py
```

## Running Benchmarks

```bash
python run_benchmarks.py
```

The banchmark script will generate stats

## Algorithm Characteristics

### Stable Sorting Algorithms
- Merge Sort
- Insertion Sort
- Timsort

### Best Time Complexity
- Merge Sort: O(n log n)
- Quick Sort: O(n log n) average case
- Heap Sort: O(n log n)
- Timsort: O(n log n)
- Introsort: O(n log n)

### Space Complexity
- Merge Sort: O(n)
- Quick Sort: O(log n) average case
- Heap Sort: O(1)
- Timsort: O(n)
- Introsort: O(log n)

