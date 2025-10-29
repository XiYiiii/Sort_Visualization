# sort.py
'''
存储所有的排序算法。
'''
import math
import random
from tracked_array import TrackedArray

def QuickSort(arr, low, high):
    '''
    快速排序。
    原理为分区排序：将每次的arr[low: high+1]变成两个分区，取某个数为基准，大于这个基准的数放到基准右端，小于这个基准的数放到基准左端。
    每次分区结束之后，在对子分区进行同样的处理。
    '''
    def partition(arr, low, high):
        i = (low - 1)
        pivot = arr[high]
        for j in range(low, high):
            if arr[j] <= pivot:
                i = i + 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return (i + 1)

    if low < high:
        pi = partition(arr, low, high)
        QuickSort(arr, low, pi - 1)
        QuickSort(arr, pi + 1, high)


def BubbleSort(arr):
    '''
    冒泡排序。
    原理为：比较两个相邻的数，将大的移到右侧。
    '''
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def ImprovedBubbleSort(arr):
    '''
    改进的冒泡排序。
    原理为：比较两个相邻的数，将大的移到右侧。
    当一轮结束后没有交换时，结束排序。
    '''
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def CombSort(arr):
    '''
    梳排序 (Comb Sort)。
    这是冒泡排序的一种改进算法，引入了一个动态变化的gap。算法开始时使用一个较大的间隙来比较和交换相距较远的元素，从而快速地将乱序的元素移动到大致正确的位置。然后，这个gap按照一个固定的值（这里取1.3）不断缩小，直到最终间隙变为1。当间隙为1时，算法就退化为一次或几次常规的冒泡排序，对基本有序的数组进行最后的微调。
    '''
    n = len(arr)
    gap = n
    shrink = 1.3
    swapped = True
    while gap > 1 or swapped:
        gap = int(gap / shrink)
        if gap < 1:
            gap = 1
        swapped = False
        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
    return arr


def CocktailShakerSort(arr):
    '''
    鸡尾酒排序（Cocktail Shaker Sort），也叫双向冒泡排序。
    原理为：在序列中从左到右和从右到左交替地进行冒泡排序。
    第一轮从左到右，将最大的元素移动到最右侧；
    下一轮从右到左，将最小的元素移动到最左侧。
    如此往复，直到没有元素交换时排序完成。
    '''
    n = len(arr)
    left = 0
    right = n - 1
    swapped = True
    while swapped:
        swapped = False
        for i in range(left, right):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                swapped = True
        if not swapped:
            break
        right -= 1
        swapped = False
        for i in range(right, left, -1):
            if arr[i] < arr[i-1]:
                arr[i], arr[i-1] = arr[i-1], arr[i]
                swapped = True
        left += 1
    return arr


def SelectionSort(arr):
    '''
    选择排序。
    原理为：找到当前序列的最小数字，放到序列首，然后序列往后伸缩一位。
    '''
    for i in range(len(arr) - 1):
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr


def InsertionSort(arr):
    '''
    插入排序。
    原理为：类似扑克牌一般，对每个找到的数，将其放到现有已经整理好的列表中合适的位置。
    '''
    for i in range(len(arr)):
        preIndex = i-1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:
            arr[preIndex+1] = arr[preIndex]
            preIndex-=1
        arr[preIndex+1] = current
    return arr


def ShellSort(arr):
    '''
    希尔排序。
    插入排序的改进版，将整体分解成数个子列表，每个子列表进行插入排序。
    '''
    gap=1
    while(gap < len(arr)/3):
        gap = gap*3+1
    while gap > 0:
        for i in range(gap,len(arr)):
            temp = arr[i]
            j = i-gap
            while j >=0 and arr[j] > temp:
                arr[j+gap]=arr[j]
                j-=gap
            arr[j+gap] = temp
        gap = math.floor(gap/3)
    return arr


def LibrarySort(arr):
    '''
    图书馆排序。
    插入排序的改进(?)，一个模拟类型的算法，会在列表的每个数字之间留出空隙，在数字想要插入进来时便可直接插入。
    间隙是可以调整的。间隙越大则空间消耗越大，但时间会略微减少。
    '''
    n = len(arr)
    if n <= 1:
        return arr
    gap_factor = 1.0
    lib_size = int(n * (1 + gap_factor))
    library = TrackedArray([None] * lib_size)
    if n > 0:
        library[0] = arr[0]
    filled_count = 1
    for i in range(1, n):
        element_to_insert = arr[i]
        insert_pos = 0
        while insert_pos < lib_size and library[insert_pos] is not None and library[insert_pos] < element_to_insert:
            insert_pos += 1
        if library[insert_pos] is not None:
            empty_slot = insert_pos
            while empty_slot < lib_size and library[empty_slot] is not None:
                empty_slot += 1
            if empty_slot == lib_size:
                empty_slot = lib_size - 1
            for j in range(empty_slot, insert_pos, -1):
                library[j] = library[j - 1]
        library[insert_pos] = element_to_insert
        filled_count += 1
    Counts = 0
    for x in library:
        if x is not None:
            arr[Counts] = x
            Counts += 1
    return arr


def MonkeySort(arr, seeds = None):
    '''
    猴子排序。
    ？
    '''
    def is_sorted(arr):
        for i in range(1, len(arr)):
            if arr[i - 1] > arr[i]:
                return False
        return True

    while not is_sorted(arr):
        if seeds:
            random.seed(seeds)
            seeds += 1
        random.shuffle(arr)
    return arr


def LazyMonkeySort(arr, seeds = None):
    '''
    懒惰的猴子排序。
    猴子一次只会移动一组数。
    '''
    def is_sorted(arr):
        for i in range(1, len(arr)):
            if arr[i - 1] > arr[i]:
                return False
        return True
    
    def get_random(cur, seeds = None):
        if seeds:
            random.seed(seeds)
        return random.randint(cur, len(arr) - 1), seeds

    cur = 0
    while not is_sorted(arr):
        i, newSeed = get_random(cur, seeds)
        if newSeed:
            seeds += 1
        j, newSeed = get_random(cur, seeds)
        if newSeed:
            seeds += 1
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def CleverMonkeySort(arr, seeds = None):
    '''
    在人类指导下的猴子排序。
    ？
    '''
    def where_cur(arr):
        for i in range(cur, len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[i]:
                    return i
        return -999
    
    def get_random(cur, seeds = None):
        if seeds:
            random.seed(seeds)
            seeds += 1
        return random.randint(cur, len(arr) - 1), seeds
    
    cur = 0
    while (cur := where_cur(arr)) != -999:
        i, newSeed = get_random(cur, seeds)
        if newSeed:
            seeds += 1
        j, newSeed = get_random(cur, seeds)
        if newSeed:
            seeds += 1
        while j == i:
            j, newSeed = get_random(cur, seeds)
            if newSeed:
                seeds += 1
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def MergeSort(arr):
    '''
    归并排序。
    原理为：将整个区域通过不断地中部划分划分成数个小区域，然后将小区域中已经有序的数组拿出来再进行有序排列。
    '''
    def merge(arr, left, mid, right):
        temp = TrackedArray([])
        i = left
        j = mid + 1
        while i <= mid and j <= right:
            if arr[i] < arr[j]:
                temp.append(arr[i])
                i += 1
            else:
                temp.append(arr[j])
                j += 1
        while i <= mid:
            temp.append(arr[i])
            i += 1
        while j <= right:
            temp.append(arr[j])
            j += 1
        for k in range(len(temp)):
            arr[left + k] = temp[k]

    def sort_range(arr, left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        sort_range(arr, left, mid)
        sort_range(arr, mid + 1, right)
        merge(arr, left, mid, right)

    if not arr:
        return arr
    sort_range(arr, 0, len(arr) - 1)
    return arr


def HeapSort(arr):
    '''
    堆排序。
    原理为：将数组原地做一个堆，然后不断将最大的值放到堆顶，再将这个值与堆底交换，移出堆。重复直到排序完毕。
    '''
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr


def CountingSort(arr):
    '''
    计数排序。
    空间占用量大的算法，会先列出一个列表，空间大小为max(arr)+1，然后将所有数组中的数字分配给对应位数，以此进行重排序。
    '''
    max_val = max(arr)
    count = TrackedArray([0] * (max_val + 1))
    for num in arr:
        count[num] += 1
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    output = TrackedArray([0] * len(arr))
    for num in reversed(arr):
        output[count[num] - 1] = num
        count[num] -= 1
    for i in range(len(arr)):
        arr[i] = output[i]
    return arr


def PigeonHoleSort(arr):
    '''
    鸽巢排序。
    本质上是优化的计数排序，将新列出的列表的空间大小缩小到max(arr)-min(arr)+2。
    '''
    max_val = max(arr)
    min_val = min(arr)
    count = TrackedArray([0] * (max_val - min_val + 2))
    for num in arr:
        count[num - min_val] += 1
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    output = TrackedArray([0] * len(arr))
    for num in reversed(arr):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1
    for i in range(len(arr)):
        arr[i] = output[i]
    return arr


def BucketSort(arr):
    '''
    桶排序。
    类似计数排序，但列出的列表空间大小为(max(arr) - min(arr)) // bucketSize + 1，即将值相近的数字分进同一个桶中进行小排序（这里需要使用别的函数），然后再将所有桶放到一起。
    这里使用了快速排序作为辅助排序。
    '''
    if len(arr) == 0:
        return arr
    min_val = min(arr)
    max_val = max(arr)
    bucket_size = 10
    bucket_count = int((max_val - min_val) // bucket_size) + 1
    buckets = TrackedArray([[] for _ in range(bucket_count)])
    for num in arr:
        index = int((num - min_val) // bucket_size)
        print(num, min_val, index, int((num - min_val) // bucket_size))
        buckets[index].append(num)
    counter = 0
    for bucket in buckets:
        low = counter
        for i in range(len(bucket)):
            arr[counter] = bucket[i]
            counter += 1
        QuickSort(arr, low, counter - 1)
    return arr


def RadixSort(arr):
    '''
    基数排序。
    基于位数的排序算法，先根据个位数排，再根据十位数排……以此类推。
    '''
    def radix_counting(arr, exp):
        n = len(arr)
        output = TrackedArray([0] * n)
        count = TrackedArray([0] * 10)
        for num in arr:
            index = (num // exp) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]
            
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        radix_counting(arr, exp)
        exp *= 10
    return arr


def TimSort(arr):
    '''
    Tim排序。
    基于归并排序和插入排序的算法，会对大规模数组使用归并排序，小规模数组使用插入排序。
    '''
    def insertion_sort(arr, left, right):
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            
    def merge(arr, l, m, r):
        len1, len2 = m - l + 1, r - m
        left = arr[l : l + len1]
        right = arr[m + 1 : r + 1]
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

    min_merge = 32
    n = len(arr)
    if n <= min_merge:
        insertion_sort(arr, 0, n - 1)
        return arr
    for start in range(0, n, min_merge):
        end = min(start + min_merge - 1, n - 1)
        insertion_sort(arr, start, end)
    size = min_merge
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge(arr, left, mid, right)
        size = 2 * size
    return arr


def BeadSort(arr):
    '''
    珠排序。
    一种非常低效的排序算法，原理为创建一个二维数组，然后模拟现实中数珠子的模样进行排序。
    '''
    num_items = len(arr)
    max_val = max(arr)
    grid = TrackedArray([TrackedArray([0] * max_val) for _ in range(num_items)])
    for i, num in enumerate(arr):
        for j in range(num):
            grid[i][j] = 1
    for j in range(max_val):
        bead_count_in_col = sum(grid[i][j] for i in range(num_items))
        for i in range(num_items):
            grid[i][j] = 0
        for i in range(num_items - bead_count_in_col, num_items):
            grid[i][j] = 1
    sorted_arr = [sum(row) for row in grid]
    for i in range(len(sorted_arr)):
        arr[i] = sorted_arr[i]
    return sorted_arr


def PDQSort(arr):
    '''
    PDQ排序。
    类似于TimSort，但对于大规模数组使用的是快速排序。
    '''
    def insertion_sort(arr, left, right):
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort(arr, low, high):
        if low < high:
            if high - low <= 24:
                insertion_sort(arr, low, high)
            else:
                pi = partition(arr, low, high)
                quicksort(arr, low, pi - 1)
                quicksort(arr, pi + 1, high)

    quicksort(arr, 0, len(arr) - 1)


def PancakeSort(arr):
    '''
    煎饼排序。
    其核心思想是通过反复“翻转”数组的前缀部分来进行排序。在每一轮迭代中，算法会找到当前未排序部分中的最大元素，先通过一次翻转将其移动到数组的最顶端（索引0），然后再通过一次翻转将这个最大元素移动到它最终应该在的有序位置。这个过程不断重复，直到整个数组有序。
    '''
    def flip(sub_arr, k):
        left = 0
        while left < k - 1:
            sub_arr[left], sub_arr[k - 1] = sub_arr[k - 1], sub_arr[left]
            left += 1
            k -= 1

    n = len(arr)
    for current_size in range(n, 1, -1):
        max_index = 0
        for i in range(current_size):
            if arr[i] > arr[max_index]:
                max_index = i
        if max_index != current_size - 1:
            if max_index != 0:
                flip(arr, max_index + 1)
            flip(arr, current_size)
    return arr


def StoogeSort(arr):
    '''
    臭皮匠排序 (Stooge Sort)。
    这是一个极其低效的递归分治排序算法。其原理是将待排序的数组（或子数组）分为三个重叠的部分：前2/3和后2/3。算法首先确保首尾元素是正确的相对顺序（小的在前，大的在后），然后递归地：(1) 对前2/3进行排序；(2) 对后2/3进行排序；(3) 再次对前2/3进行排序，以确保在第二步中可能被移动到前段的元素被正确归位。这个过程会一直递归下去，直到子数组的长度小于3。
    '''
    def _stooge_sort_recursive(sub_arr, low, high):
        if low >= high:
            return
        if sub_arr[low] > sub_arr[high]:
            sub_arr[low], sub_arr[high] = sub_arr[high], sub_arr[low]
        if high - low + 1 > 2:
            t = (high - low + 1) // 3
            _stooge_sort_recursive(sub_arr, low, high - t)
            _stooge_sort_recursive(sub_arr, low + t, high)
            _stooge_sort_recursive(sub_arr, low, high - t)

    if not arr:
        return arr
    _stooge_sort_recursive(arr, 0, len(arr) - 1)
    return arr

def GnomeSort(arr):
    '''
    侏儒排序。
    一种只有一个指针的算法，指针会从数列左往数列右移动，如果当前位大于前一位则继续往后，否则交换两位并回退。
    '''
    cur = 0
    while cur <= len(arr) - 1:
        if cur == 0 or arr[cur] > arr[cur-1]:
            cur += 1
        else:
            arr[cur], arr[cur-1] = arr[cur-1], arr[cur]
            cur -= 1
    return arr




