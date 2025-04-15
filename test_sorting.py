import pytest
import random
from sorting import SortingAlgorithms

class TestSortingAlgorithms:
    @pytest.fixture
    def sample_arrays(self):
        return {
            'empty': [],
            'single': [1],
            'sorted': [1, 2, 3, 4, 5],
            'reverse': [5, 4, 3, 2, 1],
            'random': [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            'duplicates': [3, 1, 3, 1, 3, 2, 2, 1],
            'nearly_sorted': [1, 2, 4, 3, 5, 6],
            'all_same': [4, 4, 4, 4, 4]
        }

    def verify_sorting(self, original, sorted_array):
        # Check if arrays have the same length
        assert len(original) == len(sorted_array)
        
        # Check if array is sorted
        assert all(sorted_array[i] <= sorted_array[i+1] for i in range(len(sorted_array)-1))
        
        # Check if arrays have the same elements
        assert sorted(original) == sorted_array

    @pytest.mark.parametrize("algo_name, sort_func", [
        ("Bubble Sort", SortingAlgorithms.bubble_sort),
        ("Selection Sort", SortingAlgorithms.selection_sort),
        ("Insertion Sort", SortingAlgorithms.insertion_sort),
        ("Shell Sort", SortingAlgorithms.shell_sort),
        ("Merge Sort", SortingAlgorithms.merge_sort),
        ("Quick Sort", SortingAlgorithms.quick_sort),
        ("Heap Sort", SortingAlgorithms.heap_sort),
        ("Timsort", SortingAlgorithms.timsort),
        ("Introsort", SortingAlgorithms.introsort)
    ])
    def test_sorting_algorithms(self, algo_name, sort_func, sample_arrays):
        for array_type, arr in sample_arrays.items():
            original = arr.copy()
            sorted_array = sort_func(arr.copy())
            self.verify_sorting(original, sorted_array)

    def test_large_random_array(self):
        # Test with a large random array
        large_array = [random.randint(1, 1000) for _ in range(1000)]
        for sort_func in [
            SortingAlgorithms.bubble_sort,
            SortingAlgorithms.selection_sort,
            SortingAlgorithms.insertion_sort,
            SortingAlgorithms.shell_sort,
            SortingAlgorithms.merge_sort,
            SortingAlgorithms.quick_sort,
            SortingAlgorithms.heap_sort,
            SortingAlgorithms.timsort,
            SortingAlgorithms.introsort
        ]:
            original = large_array.copy()
            sorted_array = sort_func(large_array.copy())
            self.verify_sorting(original, sorted_array)

    def test_stability(self):
        # Test stability of stable sorting algorithms
        class Item:
            def __init__(self, value, order):
                self.value = value
                self.order = order
            
            def __lt__(self, other):
                return self.value < other.value
            
            def __le__(self, other):
                return self.value <= other.value
            
            def __eq__(self, other):
                return self.value == other.value
        
        # Create array with duplicate values but different order
        test_array = [
            Item(1, 1), Item(2, 1), Item(1, 2), Item(2, 2),
            Item(1, 3), Item(2, 3)
        ]
        
        stable_algorithms = [
            ("Merge Sort", SortingAlgorithms.merge_sort),
            ("Insertion Sort", SortingAlgorithms.insertion_sort),
            ("Timsort", SortingAlgorithms.timsort)
        ]
        
        for algo_name, sort_func in stable_algorithms:
            arr_copy = test_array.copy()
            sorted_arr = sort_func(arr_copy)
            
            # Check if elements with same values maintain their relative order
            for i in range(1, len(sorted_arr)):
                if sorted_arr[i].value == sorted_arr[i-1].value:
                    assert sorted_arr[i].order > sorted_arr[i-1].order, \
                        f"{algo_name} failed stability test" 