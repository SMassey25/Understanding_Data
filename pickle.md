Title: Pickling in Python!
Date: 2019-07-29 9:33
Tags: python
Slug: blog-3-part1

Once you have been throught EDA and built a satisfactory model with tuned hyperparameters, you are ready to predict using new values. However, you do not want to train your model every time you want to predict on new values. You just want to use the previously trained model and run new values through it. This is where pickle, a Python's module for serializing and de-serialzing comes in.

## What is pickling

As said before, pickle can be used to serialize and de-serialize the object. So what do I mean by serialization and de-serialization of the objects? Serialization is a process of converting an object to a byte stream, storing the byte stream in memory for later use. De-serialization then takes the byte stream from the memory and retrieves the original object. Therefore, pikling an object converts it to a byte stream(serializzation), making sure that the character stream contains all the infomation to retrieve the original object(de-serialization). You can pickle data stored in list, tuples, dictionaries. You can even pickle your classes, functions and tuned models. 

In this post, I will pickle a dataset that can be used in other jupyter notebook. However using the same steps you can pickle classes, functions and even models with their hypertuned parameters.

###### Serialization

![blog3_pickle.jpg](attachment:blog3_pickle.jpg)

###### De - serialization

![blog3_pickle2.jpg](attachment:blog3_pickle2.jpg)

### Pickling you Data?

But why? Once you are done with cleaning, you might want to use the cleaned dataset in a different model or a different jupyter notebook. Therefore having a pickled dataset, would allow you to *dump* you data in a file and then be *open* in a differnt dataset. Let's spin up some data and then store pickle it.


```python
import numpy as np
import pandas as pd

random_dict = {
     'volts': np.random.random(10),
     'current': np.random.random(10)
 }
random_data = pd.DataFrame(random_dict)
random_data.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volts</th>
      <th>current</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>0.355205</td>
      <td>0.055690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.697468</td>
      <td>0.668440</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.102049</td>
      <td>0.284304</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pickle

file_name = 'random_data_pick.pkl'

output = open(file_name, 'wb') 
pickle.dump(random_data, output)
output.close()
```

****Few things to remember****: Open the file and don't forget wb(write byte). Then use pickle.dump(), that uses two aruguments: object you want to pickle and the file to which the object has to be saved. Also, don't forget to close the file. Lastly, this would create a file on the same file level as your notebook. 

Now that you have pickled the data, you can use the data in a without going over the EDA steps in a differnt notebook. Let us *load* in the data. 


```python
input_ = open('random_data_pick.pkl', 'rb')
pickled_dataframe = pickle.load(input_)
input_.close
```




    <function BufferedReader.close>



Now let us just make sure that we unpickled the file correct. 


```python
pickled = pd.DataFrame(pickled_dataframe)
pickled.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volts</th>
      <th>current</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0.115438</td>
      <td>0.134317</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.413739</td>
      <td>0.321276</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.145731</td>
      <td>0.728365</td>
    </tr>
  </tbody>
</table>
</div>



Personally, I pickle data and model that can be used in a web application. See my github page for my hackathon project - pokemon and world happiness repo. to learn more about linking your model to a web application. 
