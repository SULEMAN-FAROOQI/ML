# Graphing:

'''

We can plot garphs in python using matplotlib libraray. Matplotlib is a low level graph plotting library in python
that serves as a visualization utility. Most of the Matplotlib utilities lies under the pyplot submodule, 
and are usually imported as plt in short. 

The plot() function is used to draw points (markers) in a diagram. By default, the plot() function draws a line 
from point to point. The function takes parameters for specifying points in the diagram. 
Parameter 1 is an array containing the points on the x-axis. Parameter 2 is an array containing the points on the y-axis.

Arguments in plot() function:

1. marker = Gives a symbol to represent a marker (like *, x, o, D etc).
2. mfc = Gives color to marker.
3. ms = Gives size to marker.
4. mec = Gives color to the edge of marker.
5. linestyle = Gives a certain style to the line. example like ("--" , "..").
6. color = Gives color to the line.
7. linewidth =  Gives width to the line.

1. The title() function gives a title to our graph.
2. The xlabel() function gives a label to x-axis.
3. The ylabel() function gives a label to y-axis.
4. The show() function shows the graph in the new window.
5. The grid() function is used to show the grid on the graph.
6. The subplot() function makes multiple garphs in a single window. subplot(num of rows, num of columns, graph num).
7. The colorbar() function gives a colorbar to the respected colors used to plot the values.

The bar() function is used to make a bar graph. The argument of x-plane can be a string as most of the examples will have strings on x-plane.

Arguments in bar() function:

1. color = Gives color to the bar graph.
2. width = Gives width to the bar graph.
3. height = Gives height to the bar graph to barh() function. barh() is horizontal bar graph.

'''

'''

Example 1(Drawing a line in graph from position (0,0) to position (6,250):

import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints, marker = 'x')
plt.show()

'''

'''

Example 2(Ploting with different markers, marker size, markercolor):

import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints , marker = 'o', ms = 20, mec = "green" , mfc = 'g', linestyle = "--")
plt.show()

'''

'''

Example 3(Ploting Multiple lines):

import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.show()

'''

'''

Example 4 (Enhancing the looks of garph):

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.grid()
plt.show()

'''

'''

Example 5 (Multiple graphs in one window):

import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 1)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 2)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 3)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 4)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 5)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 6)
plt.plot(x,y)

plt.show()

'''

# Scattered Graph:

'''

The scatter() function is used to plot scattered points in the graph.

Arguments in scatter() function:

1. color = The color argument can be used to give color to the scattered plotted values. 
2. cmap = The cmap argument can be used within scatter function to give a scale of different colors to different scattered plotted values.
3. alpha = It is used to adjust the transparency of the scattered plotted values.

Example (Scattered Graph):

import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y)

x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y)

plt.show()

Example (Color mapping of scattered garph):

import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(100))
y = np.random.randint(100, size=(100))
colors = np.random.randint(100, size=(100))
sizes = 10 * np.random.randint(100, size=(100))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

plt.show()

'''

# Bar Graph:

'''

The bar() function is used to make a bar graph. The argument of x-plane can be a string as most of the examples will have strings on x-plane.

Arguments in bar() function:

1. color = Gives color to the bar graph.
2. width = Gives width to the bar graph.
3. height = Gives height to the bar graph to barh() function. barh() is horizontal bar graph.

Example (Bar Graph):

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x, y, color = "red", width = 0.5)
plt.show()

Example (Horizontal bar graph):

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x, y, color = "green", height = 1.5)
plt.show()

'''

# Histogram:

'''

A histogram is a graph showing frequency distributions and the number of observations within each given interval.
In Matplotlib, we use the hist() function to create histograms. The hist() function will use an array of numbers to create
a histogram, the array is sent into the function as an argument.

Example (Uniform data distribution):

import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()

Example (Normal data distribution):

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

'''

# Pie Charts:

'''

With Pyplot, we can use the pie() function to draw pie chart.

Arguments in Pie Charts:

1. labels = Can be a list of strings or integers to provide label the chart. The value of label is in correspondence with the index of array.
2. startangle = Gives angle to the starting point of the pie chart.
3. shadow = Gives shadow to the label.
4. explode = Explodes out a part of pie chart to show a specific part of pie chart.
5. colors = Gives color to the pie chart.

Legend() function is used to add a list of explanation for each wedge. It has an argument called title which gives title to the legend.

Example:

import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
my = [0.2, 0 , 0,0]

plt.pie(y, labels = mylabels, shadow=True, explode= my, startangle=90)
plt.legend(title="Fruits")
plt.show()

'''

# Customization of Graphs:

'''

The graphs created by matplotlib are used to show relationship between two variables. By seaborn module, we can create
graphs between different categories in a dataset containing more than one categories.

The seaborn.load_datasets() function provides quick access to a small number of example datasets. Use seaborn.get_dataset_names to see 
a list of available datasets

'''

# Common plot types:

# Visualizing statistical relationships:

'''

The relplot() function is function for showing the relationship between two variables, using scatterplot() or lineplot().

Example(Scatter Plot):

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.relplot(data= datasets, x="total_bill", y = "tip", kind="scatter")
plt.show()

Example (Line plot):

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.relplot(data= datasets, x="total_bill", y = "tip", kind="line")
plt.show()

'''

# Visualizing distributions:

'''

Seaborn offers several ways to visualize the distribution of a single variable. 

1. Histogram: The histplot() function shows the frequency of observations within a specific bin. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.histplot(data= datasets, x="total_bill")
plt.show()

2. Kernel Density Estimate (KDE) plot: A KDE plot shows the probability density of a variable using a continuous curve. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.kdeplot(data= datasets, x="total_bill")
plt.show()

3. Combined distribution plot: The displot() function provides a figure-level interface for plotting distribution plots, 
such as a histogram with a KDE. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.displot(data=datasets, x='total_bill', kde=True)
plt.show()

'''

# Visualizing categorical data:

'''

The catplot() function is a figure-level interface for plotting relationships between numerical and categorical variables. 

1. Box plot: Box plots are useful for comparing the distribution of a quantitative variable across different categorical levels. 

Example: 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.catplot(x='day', y='total_bill', kind='box', data=datasets)
plt.show()

2. Violin plot: A violin plot is an alternative to a box plot that shows the distribution of the data more clearly. 

Example:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.catplot(x='day', y='total_bill', kind='violin', data=datasets)
plt.show()

'''

# Plotting Parameters:

'''

1. x = Variable Distribution on x-axis.
2. y = Variable Distribution on y-axis.
3. kind = kind of graph
4. style: It uses different marker shapes on the same plot to show different categories within your data. 
5. hue: It uses different colors on the same graph to show different categories within your data.
6. row and col: They split your data into different subplots helping to compare different categories side-by-side

Example:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = sns.load_dataset("tips")
print(datasets.head())

sns.relplot(data= datasets, x="total_bill", y = "tip", 
            hue = "smoker", style="smoker", kind="scatter"
            col="time")
plt.show()

'''