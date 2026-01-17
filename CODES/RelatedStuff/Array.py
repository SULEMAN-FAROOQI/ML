# Introduction:

'''

NumPy is used for working with array. It is short for "Numerical Python". It also has functions for working in domain of linear algebra, 
fourier transform, and matrices. NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and 
manipulate them very efficiently. This behavior is called locality of reference in computer science.


'''

'''

Example 1 (Creating Numpy array): 

import numpy

arr = numpy.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

'''

'''

Example 2 (Multidimensional arrays):

import numpy as np

arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)

'''

'''

Example 3 (Checking Dimensions):

import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

say = print
say(a)
say(b)
say(c)
say(d)
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

'''

'''

Example 4 (Creating an array containing 250 random floats between 0 and 5):

import numpy

x = numpy.random.uniform(0.0, 5.0, 250)
print(x)

'''

# Array indexing:

'''

Array indexing is the same as accessing an array element. You can access an array element by referring to its index number.
The indexes in NumPy arrays start with 0, meaning that the first element has index 0, and the second has index 1 etc.

Example 1:

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[1])

Example 2 (Adding two elements):

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[2] + arr[3])

'''

# Accessing 2D Arrays:

'''

To access elements from 2-D arrays we can use comma separated integers representing the dimension and the index of the element.
Think of 2-D arrays like a table with rows and columns, where the dimension represents the row and the index represents the column.

variable[row, column], always remember computer starts computing from 0th index.

Example:

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('5th element on 2nd row: ', arr[1, 4])

'''

# Accessing 3D Arrays:

'''

To access elements from 3-D arrays we can use comma separated integers representing the dimensions and the index of the element.

Example of 3d array:
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
d[nth array, xth array in nth array, element of xth array in nth array]

Example:

import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2])

'''

# Negative Indexing:

'''

Its used to access an array from the end.

Example:

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print(''Last element from 2nd dim: '', arr[1, -1])

'''

# Slicing arrays:

'''

Slicing in python means taking elements from one given index to another given index.

We slice an array like this: [start:end].
We can also define the step, like this: [start:end:step].
If we don't pass start its considered 0.
If we don't pass end its considered length of array in that dimension.
If we don't pass step its considered 1.

Example 1:

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5])

Example 2:

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5:2])

'''

# Negative Slicing:

'''

Use the minus operator to refer to an index from the end:

Example:

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])

'''

# Slicing 2-D Arrays:

'''

Example:

import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])

'''

# Data Type:

'''

Changing data type from float to integer by using int as parameter value.

Example

import numpy as np:

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype(int)

print(newarr)
print(newarr.dtype)

'''

# Reshaping:

'''

Example 1:

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)

Example 2:

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2)
print(newarr)

'''

# Iterating Arrays:

'''

Example:

import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)

'''

# Sorting Arrays:

'''

Sorting means putting elements in an ordered sequence. Ordered sequence is any sequence that has an order corresponding to elements, 
like numeric or alphabetical, ascending or descending. The NumPy ndarray object has a function called sort(), that will sort a specified array.
We can sort any data type using sort() functon.

Example:

import numpy as np

arr = np.array([3, 2, 0, 1])
print(np.sort(arr))

'''
