# Standard Deviation and Variance:

'''

Standard deviation and Variance is a number that describes how spread out the values are.
A low standard deviation means that most of the numbers are close to the mean (average) value.
A high standard deviation means that the values are spread out over a wider range.
variance is the sum of standard deviation

Standard Deviation example:

Use the NumPy std() method to find the standard deviation:

import numpy

speed = [86,87,88,86,87,85,86]
x = numpy.std(speed)
print(x)

Variance Example:

Use the NumPy var() method to find the variance:

import numpy

speed = [32,111,138,28,59,77,97]
x = numpy.var(speed)
print(x)

'''

import numpy

speed1 = [86,87,88,86,87,85,86]
x = numpy.std(speed1)
print(x)

speed2 = [32,111,138,28,59,77,97]
x = numpy.var(speed2)
print(x)