from typing import List

class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr):
        """O(n²) bubble sort. Repeatedly swaps adjacent elements if out of order."""
        result = arr.copy()
        for i in range(len(result)):
            for j in range(len(result) - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result

    @staticmethod
    def selection_sort(arr):
        """O(n²) selection sort. Finds min element and moves it to sorted region."""
        result = arr.copy()
        for i in range(len(result)):
            min = i
            for j in range(i + 1, len(result)):
                if result[j] < result[min]:
                    min = j
            temp = result[i]
            result[i] = result[min]
            result[min] = temp
        return result

    @staticmethod
    def insertion_sort(arr):
        """O(n²) insertion sort. Builds sorted array incrementally."""
        result = arr.copy()
        for i in range(1, len(result)):
            current = result[i]
            j = i - 1
            while j >= 0 and current < result[j]:
                result[j + 1] = result[j]
                j = j - 1
            result[j + 1] = current
        return result

    @staticmethod
    def merge_sort(arr):
        """O(n log n) merge sort. Divides, recursively sorts, then merges."""
        if len(arr) <= 1:
            return arr
            
        m = len(arr) // 2
        l = SortingAlgorithms.merge_sort(arr[:m])
        r = SortingAlgorithms.merge_sort(arr[m:])
        
        result = []
        i = 0
        j = 0
        while i < len(l) and j < len(r):
            if l[i] <= r[j]:
                result.append(l[i])
                i = i + 1
            else:
                result.append(r[j])
                j = j + 1
        
        while i < len(l):
            result.append(l[i])
            i = i + 1
            
        while j < len(r):
            result.append(r[j])
            j = j + 1
            
        return result

    @staticmethod
    def quick_sort(arr):
        """O(n log n) avg. quick sort. Partitions array around pivot value."""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        
        left_list = []
        equal_list = []
        right_list = []
        
        for x in arr:
            if x < pivot:
                left_list.append(x)
            elif x == pivot:
                equal_list.append(x)
            else:
                right_list.append(x)
                
        return SortingAlgorithms.quick_sort(left_list) + equal_list + SortingAlgorithms.quick_sort(right_list)

    @staticmethod
    def heap_sort(arr):
        """O(n log n) heap sort. Builds max heap, extracts elements in order."""
        result = arr.copy()
        
        def make_heap(arr, n, i):
            """Heapify subtree rooted at index i."""
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2

            if l < n and arr[l] > arr[largest]:
                largest = l

            if r < n and arr[r] > arr[largest]:
                largest = r

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                make_heap(arr, n, largest)

        n = len(result)

        for i in range(n // 2 - 1, -1, -1):
            make_heap(result, n, i)

        for i in range(n - 1, 0, -1):
            result[0], result[i] = result[i], result[0]
            make_heap(result, i, 0)

        return result

    @staticmethod
    def shell_sort(arr):
        """Shell sort. Enhanced insertion sort with gap sequences."""
        result = arr.copy()
        n = len(result)
        
        gap = n // 2
        
        while gap > 0:
            for i in range(gap, n):
                temp = result[i]
                j = i
                while j >= gap and result[j - gap] > temp:
                    result[j] = result[j - gap]
                    j -= gap
                result[j] = temp
            gap = gap // 2
            
        return result

    @staticmethod 
    def timsort(arr):
        """O(n log n) TimSort. Hybrid of merge and insertion sort."""
        result = arr.copy()
        
        # Handle empty arrays
        if len(result) <= 1:
            return result
            
        # Define the minimum run size
        MIN_RUN = 32
        
        def calc_min_run(n):
            """Calculate minimum run length."""
            r = 0
            while n >= MIN_RUN:
                r |= n & 1
                n >>= 1
            return max(n + r, 1)  # Ensure min_run is at least 1
        
        def insertion_sort_limited(arr, left, right):
            """Sort array slice using insertion sort."""
            for i in range(left + 1, right + 1):
                temp = arr[i]
                j = i - 1
                while j >= left and arr[j] > temp:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = temp
        
        def merge(arr, l, m, r):
            """Merge sorted subarrays arr[l...m] and arr[m+1...r]."""
            len1, len2 = m - l + 1, r - m
            left, right = [], []
            
            for i in range(0, len1):
                left.append(arr[l + i])
            for i in range(0, len2):
                right.append(arr[m + 1 + i])
            
            i, j, k = 0, 0, l
            
            while i < len1 and j < len2:
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            
            while i < len1:
                arr[k] = left[i]
                k += 1
                i += 1
            
            while j < len2:
                arr[k] = right[j]
                k += 1
                j += 1
        
        # Main TimSort function
        n = len(result)
        
        # Sort individual subarrays of size MIN_RUN or less
        min_run = calc_min_run(n)
        
        for start in range(0, n, min_run):
            end = min(start + min_run - 1, n - 1)
            insertion_sort_limited(result, start, end)
        
        # Start merging from size min_run
        size = min_run
        while size < n:
            # Pick starting points of merges
            for left in range(0, n, 2 * size):
                # Find ending points
                mid = min(n - 1, left + size - 1)
                right = min(left + 2 * size - 1, n - 1)
                
                # Merge subarrays arr[left...mid] & arr[mid+1...right]
                if mid < right:
                    merge(result, left, mid, right)
            
            size = 2 * size
            
        return result
        
    @staticmethod
    def cocktail_sort(arr):
        """Bidirectional bubble sort. Traverses array in both directions."""
        result = arr.copy()
        has_swapped = True
        start_index = 0
        end_index = len(result) - 1
        
        while has_swapped:
            has_swapped = False
            
            for i in range(start_index, end_index):
                if result[i] > result[i + 1]:
                    result[i], result[i + 1] = result[i + 1], result[i]
                    has_swapped = True
            
            if not has_swapped:
                break
                
            has_swapped = False
            
            end_index = end_index - 1
            
            for i in range(end_index - 1, start_index - 1, -1):
                if result[i] > result[i + 1]:
                    result[i], result[i + 1] = result[i + 1], result[i]
                    has_swapped = True
            
            start_index = start_index + 1
            
        return result
        
    @staticmethod
    def introsort(arr):
        """O(n log n) introsort. Hybrid of quicksort, heapsort, and insertion sort."""
        result = arr.copy()
        
        # Handle empty arrays
        if len(result) <= 1:
            return result
            
        depth_limit = 2 * int(math.log2(len(result))) if len(result) > 1 else 0
        
        def sort_part(arr, start, end, depth):
            """Recursive introsort helper."""
            # Size check for small arrays
            if end - start <= 16:
                for i in range(start + 1, end):
                    temp = arr[i]
                    j = i - 1
                    while j >= start and arr[j] > temp:
                        arr[j + 1] = arr[j]
                        j = j - 1
                    arr[j + 1] = temp
                return
            
            # Switch to heapsort if recursion depth exceeds limit
            if depth <= 0:
                # Extract slice for heapsort if needed
                if start == 0 and end == len(arr):
                    arr[:] = SortingAlgorithms.heap_sort(arr)
                else:
                    sub_arr = arr[start:end]
                    sub_arr = SortingAlgorithms.heap_sort(sub_arr)
                    arr[start:end] = sub_arr
                return
            
            # Quicksort partition
            pivot_idx = start + (end - start) // 2
            pivot = arr[pivot_idx]
            
            # Move pivot to end
            arr[pivot_idx], arr[end-1] = arr[end-1], arr[pivot_idx]
            pivot_idx = end - 1
            
            # Partition around pivot
            i = start
            for j in range(start, end-1):
                if arr[j] <= pivot:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
            
            # Move pivot to its final position
            arr[i], arr[pivot_idx] = arr[pivot_idx], arr[i]
            pivot_idx = i
            
            # Recursive calls
            if start < pivot_idx:
                sort_part(arr, start, pivot_idx, depth - 1)
            if pivot_idx + 1 < end:
                sort_part(arr, pivot_idx + 1, end, depth - 1)
        
        # Start the recursion
        sort_part(result, 0, len(result), depth_limit)
        return result