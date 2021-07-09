"""You're going to write a binary search function.
You should use an iterative approach - meaning
using loops.
Your function should take two inputs:
a Python list to search through, and the value
you're searching for.
Assume the list only has distinct elements,
meaning there are no repeated values, and
elements are in a strictly increasing order.
Return the index of value, or -1 if the value
doesn't exist in the list."""


# def binary_search(input_array, value):
#     """Your code goes here."""
#     # origin_input_array = input_array.copy()
#     length = len(input_array)
#     middle_length = (length)//2
#     temp_index = 0
#
#     while length:
#
#         if value == input_array[middle_length]:
#             return middle_length + temp_index
#             break
#
#         if value > input_array[middle_length]:
#             temp_index += middle_length + 1
#             input_array = input_array[middle_length+1:]
#             length = len(input_array)
#             middle_length = (length) // 2
#
#         elif value < input_array[middle_length]:
#             input_array = input_array[:middle_length]
#             length = len(input_array)
#             middle_length = (length) // 2
#
#     return -1
#
#
# test_list = [1, 3, 9, 11, 15, 19]
# test_val1 = 1
# test_val2 = 31
# print(binary_search(test_list, test_val1))
# print(binary_search(test_list, test_val2))


# answer option
def binary_search(input_array, value):
    low = 0
    high = len(input_array) - 1
    while low <= high:
        mid = (low + high)//2
        if input_array[mid] == value:
            return mid
        elif input_array[mid] < value:
            low = mid + 1
        else:
            high = mid - 1
    else:
        -1


test_list = [1, 3, 9, 11, 15, 19, 31]
test_val1 = 1
test_val2 = 31
print(binary_search(test_list, test_val1))
print(binary_search(test_list, test_val2))