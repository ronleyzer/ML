"""Implement quick sort in Python.
Input a list.
Output a sorted list."""


def pivot_loop(i, array, pivot_num):
    '''
    :param i: index in the array
    :param array: array of numbers
    :param pivot_num: number in the array
    :return: by thhe end of this function the element in index i is <= pivot_num.
    '''

    print("i", i)
    print("pivot:", pivot_num)
    while pivot_num < array[i]:
        bigger = array[i]
        temp = array[array.index(pivot_num) - 1]

        array[i] = temp
        array[array.index(pivot_num) - 1] = pivot_num
        array[array.index(pivot_num) + 1] = bigger
        print(array)


def quicksort(array):
    '''
    :param array: array of numbers
    :return: sorted array
    '''
    print(array)
    i = 0
    if len(array) > 1:
        pivot_num = array[-1]
        while i < array.index(pivot_num):
            pivot_loop(i, array, pivot_num)
            i += 1
        # split the array with the pivot and recurse
        return quicksort(array[:array.index(pivot_num)]) + quicksort(array[array.index(pivot_num):])
    return array


test = [8, 3, 1, 7, 0, 10, 2]
print(quicksort(test))



