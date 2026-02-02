
# Pandas is a Python library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data.

# Series:

'''

A Pandas Series is like a column in a table. It is a one-dimensional array holding data of any type. We can also give it index which complies
to values in the column and using that index we can access that column. Index is like giving serial number to each row in your dataset.
We can also plot our dataset by manuplating our data with panda module.

Example:

import pandas as pd
import numpy as np

x = [3,4,5,6]

y = pd.Series(x, index=["X","Y","Z","O"])

print(y)

Example (Key/Value object in series):

import pandas as pd
import numpy as np

x = {
    "day1" : 900,
    "day2" : 1000,
    "day3" : 1100
}

y = pd.Series(x)

print(y)

'''

# DataFrame:

'''

Some Datasets contains text and data in the form of arrays. To extract this data, we use DataFrame.
A python dataframe is a 2D datasetof a 2D array or a 2D matrix. If we want to access a specific row, we can use loc() function.
If a dataframe is created using a dictionary, then the keys will be the column names.
We can use multiple arguments with our dataset:

1. head() : Prints first 5 rows of the dataset when used with print function.
2. tail() : Prints last 5 rows of the dataset when used with print function.
3. info():  Gives information about our dataset.
4. corr(): Calculates relation between each column in the datasets.
5. iloc[] : It is used to select data from DataFrames. This means you access rows and columns by their numerical position, 
   starting from 0 for the first element. It contains two arg rows and columns [r, c]. The Symbol ':' is used to 
   specify the distance between those rows and columns like from row 0 to 1 or column 0 to 3.
6. loc[] : It selects data by using row and column labels, which are the names assigned to them. It Requires specific 
   row and column labels as input. 
7. sample(n) : will print n random columns.

'''

# Eliminating and Replacing Wrong Data:

'''

1. The variable.dropna(inplace = True) function will remove all the rows with null values.

2. The variable.fillna(number, inplace = True) will replace empty cells with a value. It can also replace values for specified columns with a 
specified number.

3. A common way to replace empty cells is to calculate mean, median and mode of the columns.

4. variable.replace(new_val, old_val, inplace = TRUE) : This will change the value at the location, it was given. We can also set conditions to
manupilate our data.

5. The variable.drop_duplicates(inplace = True) function will remove all the duplicates from the dataset.

6. the variable.isna().mean()*100 function will tell you the percentage of null values in dataset.

7. If there are multiple arrays in a dataset, we can use the variable.astype() function to convert them into a single datatype.

8. If there are multiple arrays in a dataset having different keys (columns), we can use 'columns' parameter to set the columns we want 
in our dataframe.

9. We can use the variable.shape function to get the number of rows and columns in our dataframe made from our dataset.

10. In some datasets, the data is defined in a different array and the labels are defined in a different array.
Then we use the 'data' and 'columns' parameters to set the data and columns of our dataframe.

dataframe = pd.DataFrame(data = dataset.x, columns=dataset.y)

where,
x is the array containing data
y is the array containing labels (Name of columns)

11. If we want to add a column to our dataset we can add it by using:

dataframe["Name_of_new_column"] = dataset.column_name[dataset.data_for_that_column]

12. data.drop("name_of_col", axis, inplace = true) is used to drop or remove a whole column.

here,
1. axis specifies that the operation should be performed on columns (whereas axis=0 refers to rows while axis = 1 specifies columns).
2. inplace=True (optional) modifies the DataFrame directly instead of returning a new one

13. data.use_cols["col-1" to "col-n"] is used to print only the required column from the dataset.

14. np.where(x,y,z): Use to change values of a specific column.

x = Condition
y = Command when condition is true
z = Command when condition is false

15. df["column name"].apply(func) : Used to apply function on a column 

Example:

import pandas as pd
import numpy as np

data = {
    "Calories" : [500,600,800],
    "Protein" : [32,45,52]
}

dataframe = pd.DataFrame(data, index = [1,2,3])

print(dataframe)

Example (Dictionary in DataFrame):

import pandas as pd

data = {
    "Duration" : {
       "0":60,
       "1":50,
       "2":40
    },
    "Pulse": {
        "0":110,
        "1":120,
        "2":130
    }
}

frame = pd.DataFrame(data)
print(frame)

'''

# Comma Seprated Files (CSV) and JSON Files:

'''

A simple way to load big data is to use csv files. It is a plain text format and can be used by pandas.

1. We can use names function in pd.read_csv and define a list of columns, if our columns are not given or are abrupt.
2. If there is a faulty column in the dataset and you see that the real column is in the first row, then you will use header  = 1, this will
automatically drop the faulty column and replace it with the real column.
3. pd.read_csv(data, err_bad_lines = False) will skip unbalanced lines.
4. pd.read_csv(data, parse_dates = ["name of column"]) will change the data type of dates.
5. pd.read_csv(data, usecols["Name of the columns"]) will only print the column you want.

Example (CSV):

import pandas as pd
import numpy as np

x = pd.read_csv("test.csv")
print(x)

Example (JSON):

import pandas as pd
import numpy as np

x = pd.read_csv("test.json")
print(x)

'''

# Opening a csv file from an URL:

'''

import requests
from io import StringIO
import pandas as pd

req = requests.get(url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
                   headers = {"User-Agent": "Mozilla/5.0"})

data = StringIO(req.text)
print(pd.read_csv(data))


Always add raw.githubusercontent to access the csv file.

'''

# Fetching Data from an API:

'''

import pandas as pd
import requests

df = pd.DataFrame()
for i in range(1,429):
    response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page={}'.format(i))
    temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
    df = df.append(temp_df,ignore_index=True)

df.to_csv('movies.csv')

Above is an example of a database which was fetched from a server using an API. here the variable key in above holds the API. The web server had 
428 pages, we want all so we formated it with respect to i. 

'''

# Making a report using Pandas Profiling:

'''

from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("C:\\Users\\Suleman\\Desktop\\Workspace\\import\\DOCUMENTS\\Codes\\ML\\Datasets\\titanic.csv")
print(df.head())

prof = ProfileReport(df)
prof.to_file(output_file="data.html")

'''