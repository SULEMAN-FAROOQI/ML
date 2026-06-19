# Data Distribution:

# Normal Data Distribution using Histogram:

'''

The Python Program used to create an array where the values are concentrated around a given value will be:

import numpy
import matplotlib.pyplot as plt 

variable = numpy.random.normal(loc, scale, size)
hist(variable, n)
plt.show()

loc : (Mean) where the peak of the bell exists.
scale : (Standard Deviation) how flat the graph distribution should be.
size : The dimension of the returned array.
n = Number of Bars

A normal distribution graph is also known as the bell curve because of it's characteristic shape of a bell.
It shows most values in a dataset cluster around a central point with values decreasing symmetrically on either side.

Example:

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

'''

# Uniform Data distribution using Histogram:

'''

The Python Program used to create an array where the values are concentrated around a given value will be:

import numpy
import matplotlib.pyplot as plt 

variable = numpy.random.normal(S, x-intervals, y-intervals)
hist(variable, n)
plt.show()

low : This represents the point from where the value on x axis starts.
high : This represents the point from where the value on x axis ends.
size : The dimension of the returned array
n = Number of Bars

It shows how many values are between x-interval and y-interval.

Example:

import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()

'''

# Random Data Distributions:

'''

In Machine Learning the data sets can contain thousands, or even millions, of values. You might not have real 
world data when you are testing an algorithm, you might have to use randomly generated values. For that we will
create two arrays that are both filled with random numbers from a normal data distribution.

It tells us how many values are concentrated around one point in graph.

Example:

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()

'''