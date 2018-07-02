

```python
import pandas as pd
```


```python
import numpy as np
```


```python
from scipy import interpolate
```


```python
# 数据结构
```


```python
# Series
```


```python
s1 = pd.Series([100, 78, 59, 63])
```


```python
s1
```




    0    100
    1     78
    2     59
    3     63
    dtype: int64




```python
s1.values
```




    array([100,  78,  59,  63], dtype=int64)




```python
s1.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
s1.index = ['No.1', 'No.2', 'No.3', 'No.4']
```


```python
s1
```




    No.1    100
    No.2     78
    No.3     59
    No.4     63
    dtype: int64




```python
s2 = pd.Series([100, 78, 59, 63], index=['Maths', 'English', 'Literature', 'History'])
```


```python
s2
```




    Maths         100
    English        78
    Literature     59
    History        63
    dtype: int64




```python
s2[['English', 'History']]
```




    English    78
    History    63
    dtype: int64




```python
# 有字典创建
d3 = {'Name': 'Zhang San', 'Gender': 'Male', 'Age': 19, 'Height': 178, 'Weight': 66}
```


```python
s3 = pd.Series(d3)
```


```python
s3
```




    Name      Zhang San
    Gender         Male
    Age              19
    Height          178
    Weight           66
    dtype: object




```python
student_attrib = ['ID', 'Name', 'Gender', 'Age', 'Grade', 'Height', 'Weight']
```


```python
s4 = pd.Series(d3, index=student_attrib)
```


```python
s4
```




    ID              NaN
    Name      Zhang San
    Gender         Male
    Age              19
    Grade           NaN
    Height          178
    Weight           66
    dtype: object




```python
pd.isnull(s4)
```




    ID         True
    Name      False
    Gender    False
    Age       False
    Grade      True
    Height    False
    Weight    False
    dtype: bool




```python
s3 + s4
```




    Age                       38
    Gender              MaleMale
    Grade                    NaN
    Height                   356
    ID                       NaN
    Name      Zhang SanZhang San
    Weight                   132
    dtype: object




```python
s4.name = 'Student\'s profie'
```


```python
s4.index.name = 'Attrubute'
```


```python
s4
```




    Attrubute
    ID              NaN
    Name      Zhang San
    Gender         Male
    Age              19
    Grade           NaN
    Height          178
    Weight           66
    Name: Student's profie, dtype: object




```python
# 按照指定顺序实现重新索引(返回视图)
s4.reindex(index=['Name', 'ID', 'Age', 'Gender', 'Height', 'Weight', 'Grade'])
```




    Attrubute
    Name      Zhang San
    ID              NaN
    Age              19
    Gender         Male
    Height          178
    Weight           66
    Grade           NaN
    Name: Student's profie, dtype: object




```python
# index必须是单调
s4.index = ['b', 'g', 'a', 'c', 'e', 'f', 'd']
```


```python
s4
```




    b          NaN
    g    Zhang San
    a         Male
    c           19
    e          NaN
    f          178
    d           66
    Name: Student's profie, dtype: object




```python
# 重新索引 增加'h' 并填充 0
s4.reindex(index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], fill_value=0)
```




    a         Male
    b          NaN
    c           19
    d           66
    e          NaN
    f          178
    g    Zhang San
    h            0
    Name: Student's profie, dtype: object




```python
s4.index = [0, 2, 3, 6, 8, 9, 11]
```


```python
s4.reindex(range(10), method='ffill')
```




    0          NaN
    1          NaN
    2    Zhang San
    3         Male
    4         Male
    5         Male
    6           19
    7           19
    8          NaN
    9          178
    Name: Student's profie, dtype: object




```python
# DataFrame
```


```python
# 创建DataFrame
```


```python
# 使用字典创建
```


```python
dfdata = {'Name': ['Zhang San', 'Li Si', 'Wang Laowu', 'Zhao Liu', 'Qian Qi', 'Sun Ba'], 'Subject': ['Literature', 'History', 'English', 'Maths', 'Physics', 'Chemics'], 'Score': [98, 76, 84, 70, 93, 83]}
```


```python
scoresheet = pd.DataFrame(dfdata)
```


```python
scoresheet.index = ['No1', 'No2', 'No3', 'No4', 'No5', 'No6']
```


```python
scoresheet
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查看前几行 默认5
```


```python
scoresheet.head()
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.columns
```




    Index(['Name', 'Subject', 'Score'], dtype='object')




```python
scoresheet.values
```




    array([['Zhang San', 'Literature', 98],
           ['Li Si', 'History', 76],
           ['Wang Laowu', 'English', 84],
           ['Zhao Liu', 'Maths', 70],
           ['Qian Qi', 'Physics', 93],
           ['Sun Ba', 'Chemics', 83]], dtype=object)




```python
dfdata2 = {'Name': {101: 'Zhang San', 102: 'Li Si', 103: 'Wang Laowu', 104: 'Zhao Liu', 105: 'Qian Qi', 106: 'Sun Ba'},
           'Subject': {101: 'Literature', 102: 'History', 103: 'English', 104: 'Maths', 105: 'Physics', 106: 'Chemics'},
           'Score': {101: 98, 102: 76, 103: 84, 104: 70, 105: 93, 106: 83}}
```


```python
scoresheet2 = pd.DataFrame(dfdata2)
```


```python
scoresheet2
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
# numpy创建DataFrame
```


```python
numframe = np.random.randn(10, 5)
```


```python
framenum = pd.DataFrame(numframe)
```


```python
framenum.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.444496</td>
      <td>-0.749858</td>
      <td>-0.601702</td>
      <td>0.503988</td>
      <td>-0.352682</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.089145</td>
      <td>1.791063</td>
      <td>0.312342</td>
      <td>1.199606</td>
      <td>0.888155</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.183990</td>
      <td>0.624378</td>
      <td>-0.972992</td>
      <td>0.513611</td>
      <td>-0.575874</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.145332</td>
      <td>0.426415</td>
      <td>-0.076375</td>
      <td>-0.043228</td>
      <td>1.684722</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.194206</td>
      <td>0.336490</td>
      <td>-0.371831</td>
      <td>-0.711729</td>
      <td>0.439429</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 打印数据框属性信息
```


```python
framenum.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 5 columns):
    0    10 non-null float64
    1    10 non-null float64
    2    10 non-null float64
    3    10 non-null float64
    4    10 non-null float64
    dtypes: float64(5)
    memory usage: 480.0 bytes
    


```python
# 打印每列属性
```


```python
framenum.dtypes
```




    0    float64
    1    float64
    2    float64
    3    float64
    4    float64
    dtype: object




```python
stock = np.dtype([('name', np.str_, 4), ('time', np.str_, 10), ('opening_price', np.float64), ('closing_price', np.float64), ('lowest_price', np.float64), ('highest_price', np.float64), ('volume', np.int32)])
```


```python
jd_stoct = np.loadtxt('data.csv', delimiter=',', dtype=stock)
```


```python
jd = pd.DataFrame(jd_stoct)
```


```python
jd.head()
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JD</td>
      <td>3-Jan-17</td>
      <td>25.95</td>
      <td>25.82</td>
      <td>25.64</td>
      <td>26.11</td>
      <td>8275300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JD</td>
      <td>4-Jan-17</td>
      <td>26.05</td>
      <td>25.85</td>
      <td>25.58</td>
      <td>26.08</td>
      <td>7862800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JD</td>
      <td>5-Jan-17</td>
      <td>26.15</td>
      <td>26.30</td>
      <td>26.05</td>
      <td>26.80</td>
      <td>10205600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JD</td>
      <td>6-Jan-17</td>
      <td>26.30</td>
      <td>26.27</td>
      <td>25.92</td>
      <td>26.41</td>
      <td>6234300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JD</td>
      <td>9-Jan-17</td>
      <td>26.64</td>
      <td>26.26</td>
      <td>26.14</td>
      <td>26.95</td>
      <td>8071500</td>
    </tr>
  </tbody>
</table>
</div>




```python
jd.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 71 entries, 0 to 70
    Data columns (total 7 columns):
    name             71 non-null object
    time             71 non-null object
    opening_price    71 non-null float64
    closing_price    71 non-null float64
    lowest_price     71 non-null float64
    highest_price    71 non-null float64
    volume           71 non-null int32
    dtypes: float64(4), int32(1), object(2)
    memory usage: 3.7+ KB
    


```python
# 直接读入csv文件构造DataFrame
jddf = pd.read_csv('data.csv', header=None, names=['name', 'time', 'opening_price', 'closing_price', 'lowest_price', 'highest_price', 'volume'])
```


```python
# header=None 不自动把数据的第一行 第一列设置成行,列索引
```


```python
# name 指定索引
```


```python
jddf.head()
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JD</td>
      <td>3-Jan-17</td>
      <td>25.95</td>
      <td>25.82</td>
      <td>25.64</td>
      <td>26.11</td>
      <td>8275300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JD</td>
      <td>4-Jan-17</td>
      <td>26.05</td>
      <td>25.85</td>
      <td>25.58</td>
      <td>26.08</td>
      <td>7862800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JD</td>
      <td>5-Jan-17</td>
      <td>26.15</td>
      <td>26.30</td>
      <td>26.05</td>
      <td>26.80</td>
      <td>10205600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JD</td>
      <td>6-Jan-17</td>
      <td>26.30</td>
      <td>26.27</td>
      <td>25.92</td>
      <td>26.41</td>
      <td>6234300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JD</td>
      <td>9-Jan-17</td>
      <td>26.64</td>
      <td>26.26</td>
      <td>26.14</td>
      <td>26.95</td>
      <td>8071500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 直接读入excel文件构造DataFrame
```


```python
jddf = pd.read_excel('data.xlsx', header=None, names=['name', 'time', 'opening_price', 'closing_price', 'lowest_price', 'highest_price', 'volume'])
```


```python
jddf.head()
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JD</td>
      <td>2017-04-13</td>
      <td>32.74</td>
      <td>32.47</td>
      <td>32.45</td>
      <td>32.87</td>
      <td>3013600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JD</td>
      <td>2017-04-12</td>
      <td>32.31</td>
      <td>32.71</td>
      <td>32.31</td>
      <td>32.88</td>
      <td>6818000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JD</td>
      <td>2017-04-11</td>
      <td>32.70</td>
      <td>32.30</td>
      <td>32.22</td>
      <td>33.28</td>
      <td>8054200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JD</td>
      <td>2017-04-10</td>
      <td>32.16</td>
      <td>32.67</td>
      <td>32.15</td>
      <td>32.92</td>
      <td>8303800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JD</td>
      <td>2017-04-07</td>
      <td>32.20</td>
      <td>32.01</td>
      <td>31.57</td>
      <td>32.25</td>
      <td>5651000</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
其他数据源构造DataFrame
  * read_table: 读入具有分隔符的文件
  * read_sql: 输入SQL数据库文件
  * read_sas: 读入SAS的xpt或sas7bdat格式的数据集
  * read_stata: 读入STATA数据集
  * read_json: 读入json数据
  * read_html: 读入网页中的表
  * read_clipboard: 读入剪贴板中的数据内容
  * read_fwf: 读入固定宽度格式化数据
  * read_hdf: 读入分布式存储系统中的文件
'''
```




    '\n其他数据源构造DataFrame\n  * read_table: 读入具有分隔符的文件\n  * read_sql: 输入SQL数据库文件\n  * read_sas: 读入SAS的xpt或sas7bdat格式的数据集\n  * read_stata: 读入STATA数据集\n  * read_json: 读入json数据\n  * read_html: 读入网页中的表\n  * read_clipboard: 读入剪贴板中的数据内容\n  * read_fwf: 读入固定宽度格式化数据\n  * read_hdf: 读入分布式存储系统中的文件\n'




```python
jddf = pd.read_excel('data.xlsx', sep=',', header=None, names=['name', 'time', 'opening_price', 'closing_price', 'lowest_price', 'highest_price', 'volume'])
```


```python
jddfsetindex = jddf.set_index(jddf['time'])
```


```python
jddfsetindex.head()
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-04-13</th>
      <td>JD</td>
      <td>2017-04-13</td>
      <td>32.74</td>
      <td>32.47</td>
      <td>32.45</td>
      <td>32.87</td>
      <td>3013600</td>
    </tr>
    <tr>
      <th>2017-04-12</th>
      <td>JD</td>
      <td>2017-04-12</td>
      <td>32.31</td>
      <td>32.71</td>
      <td>32.31</td>
      <td>32.88</td>
      <td>6818000</td>
    </tr>
    <tr>
      <th>2017-04-11</th>
      <td>JD</td>
      <td>2017-04-11</td>
      <td>32.70</td>
      <td>32.30</td>
      <td>32.22</td>
      <td>33.28</td>
      <td>8054200</td>
    </tr>
    <tr>
      <th>2017-04-10</th>
      <td>JD</td>
      <td>2017-04-10</td>
      <td>32.16</td>
      <td>32.67</td>
      <td>32.15</td>
      <td>32.92</td>
      <td>8303800</td>
    </tr>
    <tr>
      <th>2017-04-07</th>
      <td>JD</td>
      <td>2017-04-07</td>
      <td>32.20</td>
      <td>32.01</td>
      <td>31.57</td>
      <td>32.25</td>
      <td>5651000</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(jddfsetindex.index)
```




    pandas.core.indexes.datetimes.DatetimeIndex




```python
# 数据导出
```


```python
# jddf.to_csv('jdstockdata.csv')
```


```python
# jddf.to_excel('jdstockdata.xlsx')
```


```python
# 索引和切片
```


```python
scoresheet.Subject
```




    No1    Literature
    No2       History
    No3       English
    No4         Maths
    No5       Physics
    No6       Chemics
    Name: Subject, dtype: object




```python
scoresheet[['Name', 'Score']]
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
      <th>Name</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>Qian Qi</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet[:'No4']
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.ix[['No1', 'No3', 'No6']]
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.
    




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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.ix[3:6, ['Name', 'Score']]
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.
    




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
      <th>Name</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>Qian Qi</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet['Subject']
```




    No1    Literature
    No2       History
    No3       English
    No4         Maths
    No5       Physics
    No6       Chemics
    Name: Subject, dtype: object




```python
# print(scoresheet.ix[3:6, [1, 3]])
```


```python
# print(scoresheet.iloc[[1, 4, 5], [0, 3]])
```


```python
scoresheet.loc[['No1', 'No5'], ['ID', 'Score']]
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
      """Entry point for launching an IPython kernel.
    




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
      <th>ID</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>NaN</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>NaN</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet[(scoresheet.Score > 80) & (scoresheet.Score <= 90)]
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet[['Name', 'Score']][(scoresheet.Score > 80) & (scoresheet.Score <= 90)]
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
      <th>Name</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 行列操作
```


```python
scoresheet = pd.DataFrame(dfdata, columns=['ID', 'Name', 'Subject', 'Score'], index=['No1', 'No2', 'No3', 'No4', 'No5', 'No6'])
```


```python
scoresheet
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
      <th>ID</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>NaN</td>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>NaN</td>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>NaN</td>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>NaN</td>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>NaN</td>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>NaN</td>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.reindex(columns=['Name', 'Subject', 'ID', 'Score'])
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
      <th>Name</th>
      <th>Subject</th>
      <th>ID</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>NaN</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>History</td>
      <td>NaN</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>NaN</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>NaN</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>NaN</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>NaN</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改行/列数据
```


```python
# 新增列
```


```python
scoresheet['Homeword'] = 90
```


```python
scoresheet
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
      <th>ID</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
      <th>Homeword</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>NaN</td>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>NaN</td>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>NaN</td>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>NaN</td>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>NaN</td>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>NaN</td>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改列名
```


```python
scoresheet.rename(columns={'Homeword': 'Homework'}, inplace=True)
```


```python
scoresheet
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
      <th>ID</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
      <th>Homework</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>NaN</td>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>NaN</td>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>NaN</td>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>NaN</td>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>NaN</td>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>NaN</td>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet['ID'] = np.arange(6)
```


```python
scoresheet
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
      <th>ID</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
      <th>Homework</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>0</td>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>1</td>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>2</td>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>3</td>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>4</td>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
      <td>90</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>5</td>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 替代精确匹配的索引的值
```


```python
fixed = pd.Series([97, 76, 83], index=['No1', 'No3', 'No6'])
```


```python
scoresheet['Homework'] = fixed
```


```python
scoresheet
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
      <th>ID</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
      <th>Homework</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>0</td>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>1</td>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>2</td>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>3</td>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>4</td>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>5</td>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
      <td>83.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 删除行/列数据
```


```python
# 删除列
```


```python
del scoresheet['Homework']
```


```python
scoresheet
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
      <th>ID</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>0</td>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>1</td>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>2</td>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>3</td>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>4</td>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>5</td>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 删除行或列 (inplace=True直接修改元数据内存值 inplace=False返回修改后的值)
```


```python
scoresheet.drop('ID', axis=1, inplace=True)  # axis=1 列 axis=0 行
```


```python
scoresheet
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No1</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>No5</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>No6</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.drop(['No1', 'No5', 'No6'], axis=0, inplace=True)
```


```python
scoresheet
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No2</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>No3</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>No4</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 排序
```


```python
ssort = pd.Series(range(5), index=['b', 'a', 'd', 'e', 'c'])
```


```python
ssort.sort_index()
```




    a    1
    b    0
    c    4
    d    2
    e    3
    dtype: int64




```python
ssort.sort_index(ascending=False)
```




    e    3
    d    2
    c    4
    b    0
    a    1
    dtype: int64




```python
scoresheet2.index = [102, 101, 106, 104, 103, 105]
```


```python
scoresheet2
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet2.sort_index()
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet2.sort_index(axis=0, ascending=False)
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet2.sort_index(axis=1, ascending=False)
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
      <th>Subject</th>
      <th>Score</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102</th>
      <td>Literature</td>
      <td>98</td>
      <td>Zhang San</td>
    </tr>
    <tr>
      <th>101</th>
      <td>History</td>
      <td>76</td>
      <td>Li Si</td>
    </tr>
    <tr>
      <th>106</th>
      <td>English</td>
      <td>84</td>
      <td>Wang Laowu</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Maths</td>
      <td>70</td>
      <td>Zhao Liu</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Physics</td>
      <td>93</td>
      <td>Qian Qi</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Chemics</td>
      <td>83</td>
      <td>Sun Ba</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet2.sort_index(by='Score', ascending=False)
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: by argument to sort_index is deprecated, please use .sort_values(by=...)
      """Entry point for launching an IPython kernel.
    




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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet2.sort_values(by='Score', ascending=False)
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 排名
```


```python
rrank = pd.Series([10, 12, 9, 9, 14, 4, 2, 4, 9, 1])
```


```python
rrank.rank()
```




    0     8.0
    1     9.0
    2     6.0
    3     6.0
    4    10.0
    5     3.5
    6     2.0
    7     3.5
    8     6.0
    9     1.0
    dtype: float64




```python
rrank.rank(ascending=False)
```




    0     3.0
    1     2.0
    2     5.0
    3     5.0
    4     1.0
    5     7.5
    6     9.0
    7     7.5
    8     5.0
    9    10.0
    dtype: float64




```python
'''
average  平均分配排名
min  最小排名
max  最大排名
first  按出现顺序排名
'''
```




    '\naverage  平均分配排名\nmin  最小排名\nmax  最大排名\nfirst  按出现顺序排名\n'




```python
rrank.rank(method='first')
```




    0     8.0
    1     9.0
    2     5.0
    3     6.0
    4    10.0
    5     3.0
    6     2.0
    7     4.0
    8     7.0
    9     1.0
    dtype: float64




```python
rrank.rank(method='max')
```




    0     8.0
    1     9.0
    2     7.0
    3     7.0
    4    10.0
    5     4.0
    6     2.0
    7     4.0
    8     7.0
    9     1.0
    dtype: float64




```python
scoresheet2.rank()
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>6.0</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2.0</td>
      <td>6.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 运算
```


```python
cs1 = pd.Series([1.5, 2.5, 3, 5, 1], index=['a', 'c', 'd', 'b', 'e'])
```


```python
cs2 = pd.Series([10, 20, 30, 50, 10, 100, 20], index=['c', 'a', 'e', 'b', 'f', 'g', 'd'])
```


```python
cs1 + cs2
```




    a    21.5
    b    55.0
    c    12.5
    d    23.0
    e    31.0
    f     NaN
    g     NaN
    dtype: float64




```python
cdf1 = pd.DataFrame(np.arange(10).reshape((2, 5)), columns=list('bcaed'))
```


```python
cdf2 = pd.DataFrame(np.arange(12).reshape((3, 4)), columns=list('abcd'))
```


```python
cdf1 + cdf2
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
cdf1.add(cdf2, fill_value=0)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 函数应用与映射
```


```python
reversef = lambda x: -x
```


```python
reversef(cs2)
```




    c    -10
    a    -20
    e    -30
    b    -50
    f    -10
    g   -100
    d    -20
    dtype: int64




```python
rangef = lambda x: x.max() - x.min()
```


```python
rangef(cs2)
```




    90




```python
rangef(cdf1.add(cdf2, fill_value=0))
```




    a    9.0
    b    9.0
    c    9.0
    d    9.0
    e    5.0
    dtype: float64




```python
(cdf1.add(cdf2, fill_value=0)).apply(rangef, axis=0)
```




    a    9.0
    b    9.0
    c    9.0
    d    9.0
    e    5.0
    dtype: float64




```python
(cdf1.add(cdf2, fill_value=0)).apply(rangef, axis=1)
```




    0    6.0
    1    8.0
    2    3.0
    dtype: float64




```python
def statistics(x):
    return pd.Series([x.min(), x.max(), x.max() - x.min(), x.mean(), x.count()], index=['Min', 'Max', 'Range', 'Mean', 'N'])
```


```python
outformat = lambda x: '%.2f' % x
```


```python
# 格式化
```


```python
((cdf1.add(cdf2, fill_value=0)).apply(statistics)).applymap(outformat)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Min</th>
      <td>2.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>7.00</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>Max</th>
      <td>11.00</td>
      <td>10.00</td>
      <td>12.00</td>
      <td>16.00</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>Range</th>
      <td>9.00</td>
      <td>9.00</td>
      <td>9.00</td>
      <td>9.00</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>7.00</td>
      <td>6.67</td>
      <td>8.33</td>
      <td>11.33</td>
      <td>5.50</td>
    </tr>
    <tr>
      <th>N</th>
      <td>3.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>2.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# apply 的操作对象是DataFrame的一列或一行数据
# applymap是元素级的 只支持一个函数 作用于每个DataFrame的每个数据
# map也是元素级的, 对Series中的每个数据调用一次函数
```


```python
# 分组 (python2 map返回list python3 map返回map对象)
```


```python
jddf['Market'] = list(map(lambda x: 'Good' if x > 0 else ('Bad' if x < 0 else 'OK'), jddf['closing_price'] - jddf['opening_price']))
```


```python
jddf.head()
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
      <th>Market</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JD</td>
      <td>2017-04-13</td>
      <td>32.74</td>
      <td>32.47</td>
      <td>32.45</td>
      <td>32.87</td>
      <td>3013600</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JD</td>
      <td>2017-04-12</td>
      <td>32.31</td>
      <td>32.71</td>
      <td>32.31</td>
      <td>32.88</td>
      <td>6818000</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JD</td>
      <td>2017-04-11</td>
      <td>32.70</td>
      <td>32.30</td>
      <td>32.22</td>
      <td>33.28</td>
      <td>8054200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JD</td>
      <td>2017-04-10</td>
      <td>32.16</td>
      <td>32.67</td>
      <td>32.15</td>
      <td>32.92</td>
      <td>8303800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JD</td>
      <td>2017-04-07</td>
      <td>32.20</td>
      <td>32.01</td>
      <td>31.57</td>
      <td>32.25</td>
      <td>5651000</td>
      <td>Bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
jddfgrouped = jddf.groupby(jddf['Market'])
```


```python
jddfgrouped.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">opening_price</th>
      <th colspan="2" halign="left">closing_price</th>
      <th>...</th>
      <th colspan="2" halign="left">highest_price</th>
      <th colspan="8" halign="left">volume</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Market</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bad</th>
      <td>32.0</td>
      <td>30.028750</td>
      <td>2.102453</td>
      <td>25.95</td>
      <td>28.6275</td>
      <td>30.900</td>
      <td>31.62</td>
      <td>32.74</td>
      <td>32.0</td>
      <td>29.724062</td>
      <td>...</td>
      <td>31.8525</td>
      <td>33.28</td>
      <td>32.0</td>
      <td>6970318.75</td>
      <td>3.306978e+06</td>
      <td>3013600.0</td>
      <td>5258175.0</td>
      <td>6300900.0</td>
      <td>8058525.0</td>
      <td>22176000.0</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>38.0</td>
      <td>29.542105</td>
      <td>1.795692</td>
      <td>26.15</td>
      <td>28.2900</td>
      <td>29.785</td>
      <td>31.24</td>
      <td>32.31</td>
      <td>38.0</td>
      <td>29.827105</td>
      <td>...</td>
      <td>31.6175</td>
      <td>32.92</td>
      <td>38.0</td>
      <td>6686600.00</td>
      <td>3.252937e+06</td>
      <td>3107000.0</td>
      <td>4425325.0</td>
      <td>5938000.0</td>
      <td>7880550.0</td>
      <td>20417400.0</td>
    </tr>
    <tr>
      <th>OK</th>
      <td>1.0</td>
      <td>29.520000</td>
      <td>NaN</td>
      <td>29.52</td>
      <td>29.5200</td>
      <td>29.520</td>
      <td>29.52</td>
      <td>29.52</td>
      <td>1.0</td>
      <td>29.520000</td>
      <td>...</td>
      <td>29.6500</td>
      <td>29.65</td>
      <td>1.0</td>
      <td>6084500.00</td>
      <td>NaN</td>
      <td>6084500.0</td>
      <td>6084500.0</td>
      <td>6084500.0</td>
      <td>6084500.0</td>
      <td>6084500.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 40 columns</p>
</div>




```python
# 合并
c1 = pd.DataFrame({'Name': {101: 'Zhang San', 102: 'Li Si', 103: 'Wang Laowu', 104: 'Zhao Liu', 105: 'Qian Qi', 106: 'Sun Ba'},
                   'Subject': {101: 'Literature', 102: 'History', 103: 'English', 104: 'Maths', 105: 'Physics', 106: 'Chemics'},
                   'Score': {101: 98, 102: 76, 103: 84, 104: 70, 105: 93, 106: 83}})
```


```python
c1
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
c2 = pd.DataFrame({'Gender': {101: 'Male', 102: 'Male', 103: 'Male', 104: 'Female', 105: 'Female', 106: 'Male'}})
```


```python
c2
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
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Male</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Female</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
c = pd.concat([c1, c2], axis=1)
```


```python
c
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
c1.append(c2)
```

    c:\program files\python36\lib\site-packages\pandas\core\frame.py:6201: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=True'.
    
    To retain the current behavior and silence the warning, pass sort=False
    
      sort=sort)
    




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
      <th>Gender</th>
      <th>Name</th>
      <th>Score</th>
      <th>Subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>NaN</td>
      <td>Zhang San</td>
      <td>98.0</td>
      <td>Literature</td>
    </tr>
    <tr>
      <th>102</th>
      <td>NaN</td>
      <td>Li Si</td>
      <td>76.0</td>
      <td>History</td>
    </tr>
    <tr>
      <th>103</th>
      <td>NaN</td>
      <td>Wang Laowu</td>
      <td>84.0</td>
      <td>English</td>
    </tr>
    <tr>
      <th>104</th>
      <td>NaN</td>
      <td>Zhao Liu</td>
      <td>70.0</td>
      <td>Maths</td>
    </tr>
    <tr>
      <th>105</th>
      <td>NaN</td>
      <td>Qian Qi</td>
      <td>93.0</td>
      <td>Physics</td>
    </tr>
    <tr>
      <th>106</th>
      <td>NaN</td>
      <td>Sun Ba</td>
      <td>83.0</td>
      <td>Chemics</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([c1, c2], axis=0)
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=True'.
    
    To retain the current behavior and silence the warning, pass sort=False
    
      """Entry point for launching an IPython kernel.
    




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
      <th>Gender</th>
      <th>Name</th>
      <th>Score</th>
      <th>Subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>NaN</td>
      <td>Zhang San</td>
      <td>98.0</td>
      <td>Literature</td>
    </tr>
    <tr>
      <th>102</th>
      <td>NaN</td>
      <td>Li Si</td>
      <td>76.0</td>
      <td>History</td>
    </tr>
    <tr>
      <th>103</th>
      <td>NaN</td>
      <td>Wang Laowu</td>
      <td>84.0</td>
      <td>English</td>
    </tr>
    <tr>
      <th>104</th>
      <td>NaN</td>
      <td>Zhao Liu</td>
      <td>70.0</td>
      <td>Maths</td>
    </tr>
    <tr>
      <th>105</th>
      <td>NaN</td>
      <td>Qian Qi</td>
      <td>93.0</td>
      <td>Physics</td>
    </tr>
    <tr>
      <th>106</th>
      <td>NaN</td>
      <td>Sun Ba</td>
      <td>83.0</td>
      <td>Chemics</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 按照指定关键字合并
```


```python
c3 = pd.DataFrame({'Name': {101: 'Zhang San', 102: 'Li Si', 103: 'Wang Laowu', 104: 'Zhao Liu', 105: 'Qian Qi', 106: 'Sun Ba'},
                   'Gender': {101: 'Male', 102: 'Male', 103: 'Male', 104: 'Female', 105: 'Female', 106: 'Male'}})
```


```python
c3
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
      <th>Name</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>Zhang San</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Li Si</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Wang Laowu</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Zhao Liu</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Qian Qi</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Sun Ba</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 按照相同列合并(类SQL JOIN)
pd.merge(c1, c3, on='Name')
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
      <th>Name</th>
      <th>Subject</th>
      <th>Score</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zhang San</td>
      <td>Literature</td>
      <td>98</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Li Si</td>
      <td>History</td>
      <td>76</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wang Laowu</td>
      <td>English</td>
      <td>84</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Zhao Liu</td>
      <td>Maths</td>
      <td>70</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Qian Qi</td>
      <td>Physics</td>
      <td>93</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun Ba</td>
      <td>Chemics</td>
      <td>83</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 分类数据
```


```python
student_profile = pd.DataFrame({'Name': ['Morgan Wang', 'Jackie Li', 'Tom Ding', 'Erricson John', 'Juan Saint', 'Sui Mike', 'Li Rose'],
                                'Gender': [1, 0, 0, 1, 0, 1, 2],
                                'Blood': ['A', 'AB', 'O', 'AB', 'B', 'O', 'A'],
                                'Grade': [1, 2, 3, 2, 3, 1, 2],
                                'Height': [175, 180, 168, 170, 158, 183, 173]})
```


```python
student_profile
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
      <th>Name</th>
      <th>Gender</th>
      <th>Blood</th>
      <th>Grade</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Morgan Wang</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
      <td>175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jackie Li</td>
      <td>0</td>
      <td>AB</td>
      <td>2</td>
      <td>180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tom Ding</td>
      <td>0</td>
      <td>O</td>
      <td>3</td>
      <td>168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Erricson John</td>
      <td>1</td>
      <td>AB</td>
      <td>2</td>
      <td>170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Juan Saint</td>
      <td>0</td>
      <td>B</td>
      <td>3</td>
      <td>158</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sui Mike</td>
      <td>1</td>
      <td>O</td>
      <td>1</td>
      <td>183</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Li Rose</td>
      <td>2</td>
      <td>A</td>
      <td>2</td>
      <td>173</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 为Male列添加标签
```


```python
student_profile['Gender_Value'] = student_profile['Gender'].astype('category')
```


```python
student_profile['Gender_Value'].cat.categories = ['Female', 'Male', 'Unconfirmed']
```


```python
student_profile
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
      <th>Name</th>
      <th>Gender</th>
      <th>Blood</th>
      <th>Grade</th>
      <th>Height</th>
      <th>Gender_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Morgan Wang</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
      <td>175</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jackie Li</td>
      <td>0</td>
      <td>AB</td>
      <td>2</td>
      <td>180</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tom Ding</td>
      <td>0</td>
      <td>O</td>
      <td>3</td>
      <td>168</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Erricson John</td>
      <td>1</td>
      <td>AB</td>
      <td>2</td>
      <td>170</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Juan Saint</td>
      <td>0</td>
      <td>B</td>
      <td>3</td>
      <td>158</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sui Mike</td>
      <td>1</td>
      <td>O</td>
      <td>1</td>
      <td>183</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Li Rose</td>
      <td>2</td>
      <td>A</td>
      <td>2</td>
      <td>173</td>
      <td>Unconfirmed</td>
    </tr>
  </tbody>
</table>
</div>




```python
student_profile['Gender_Value'].cat.categories = ['male', 'FeMale', 'Unconfirmed']
```


```python
student_profile
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
      <th>Name</th>
      <th>Gender</th>
      <th>Blood</th>
      <th>Grade</th>
      <th>Height</th>
      <th>Gender_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Morgan Wang</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
      <td>175</td>
      <td>FeMale</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jackie Li</td>
      <td>0</td>
      <td>AB</td>
      <td>2</td>
      <td>180</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tom Ding</td>
      <td>0</td>
      <td>O</td>
      <td>3</td>
      <td>168</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Erricson John</td>
      <td>1</td>
      <td>AB</td>
      <td>2</td>
      <td>170</td>
      <td>FeMale</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Juan Saint</td>
      <td>0</td>
      <td>B</td>
      <td>3</td>
      <td>158</td>
      <td>male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sui Mike</td>
      <td>1</td>
      <td>O</td>
      <td>1</td>
      <td>183</td>
      <td>FeMale</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Li Rose</td>
      <td>2</td>
      <td>A</td>
      <td>2</td>
      <td>173</td>
      <td>Unconfirmed</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对数值型数据分段标签
```


```python
labels = ["{0}-{1}".format(i, i + 10) for i in range(160, 200, 10)]
```


```python
student_profile['Height_Group'] = pd.cut(student_profile.Height, range(160, 205, 10), right=False, labels=labels)
```


```python
student_profile
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
      <th>Name</th>
      <th>Gender</th>
      <th>Blood</th>
      <th>Grade</th>
      <th>Height</th>
      <th>Gender_Value</th>
      <th>Height_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Morgan Wang</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
      <td>175</td>
      <td>FeMale</td>
      <td>170-180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jackie Li</td>
      <td>0</td>
      <td>AB</td>
      <td>2</td>
      <td>180</td>
      <td>male</td>
      <td>180-190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tom Ding</td>
      <td>0</td>
      <td>O</td>
      <td>3</td>
      <td>168</td>
      <td>male</td>
      <td>160-170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Erricson John</td>
      <td>1</td>
      <td>AB</td>
      <td>2</td>
      <td>170</td>
      <td>FeMale</td>
      <td>170-180</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Juan Saint</td>
      <td>0</td>
      <td>B</td>
      <td>3</td>
      <td>158</td>
      <td>male</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sui Mike</td>
      <td>1</td>
      <td>O</td>
      <td>1</td>
      <td>183</td>
      <td>FeMale</td>
      <td>180-190</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Li Rose</td>
      <td>2</td>
      <td>A</td>
      <td>2</td>
      <td>173</td>
      <td>Unconfirmed</td>
      <td>170-180</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 时间序列
```


```python
# 创建时间序列
```


```python
# 将当前时间转化为时间戳
```


```python
pd.Timestamp('now')
```




    Timestamp('2018-07-02 18:47:06.768946')




```python
# 利用时间戳创建时间序列
```


```python
dates = [pd.Timestamp('2017-07-05'), pd.Timestamp('2017-07-06'), pd.Timestamp('2017-07-07')]
```


```python
ts = pd.Series(np.random.randn(3), dates)
```


```python
ts
```




    2017-07-05    1.018477
    2017-07-06   -1.383504
    2017-07-07   -1.474000
    dtype: float64




```python
ts.index
```




    DatetimeIndex(['2017-07-05', '2017-07-06', '2017-07-07'], dtype='datetime64[ns]', freq=None)




```python
type(ts.index)
```




    pandas.core.indexes.datetimes.DatetimeIndex




```python
dates = pd.date_range('2017-07-05', '2017-07-07')
```


```python
tsdr = pd.Series(np.random.randn(3), dates)
```


```python
tsdr
```




    2017-07-05   -2.522755
    2017-07-06   -1.765218
    2017-07-07    0.216270
    Freq: D, dtype: float64




```python
type(tsdr.index)
```




    pandas.core.indexes.datetimes.DatetimeIndex




```python
dates = [pd.Period('2017-07-05'), pd.Period('2017-07-06'), pd.Period('2017-07-07')]
```


```python
tsp = pd.Series(np.random.randn(3), dates)
```


```python
tsp
```




    2017-07-05    1.499539
    2017-07-06    0.061571
    2017-07-07    1.898084
    Freq: D, dtype: float64




```python
type(tsdr.index)
```




    pandas.core.indexes.datetimes.DatetimeIndex




```python
jd_ts = jddf.set_index(pd.to_datetime(jddf['time']))
```


```python
type(jd_ts.index)
```




    pandas.core.indexes.datetimes.DatetimeIndex




```python
jd_ts.head()
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
      <th>Market</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-04-13</th>
      <td>JD</td>
      <td>2017-04-13</td>
      <td>32.74</td>
      <td>32.47</td>
      <td>32.45</td>
      <td>32.87</td>
      <td>3013600</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-12</th>
      <td>JD</td>
      <td>2017-04-12</td>
      <td>32.31</td>
      <td>32.71</td>
      <td>32.31</td>
      <td>32.88</td>
      <td>6818000</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-04-11</th>
      <td>JD</td>
      <td>2017-04-11</td>
      <td>32.70</td>
      <td>32.30</td>
      <td>32.22</td>
      <td>33.28</td>
      <td>8054200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-10</th>
      <td>JD</td>
      <td>2017-04-10</td>
      <td>32.16</td>
      <td>32.67</td>
      <td>32.15</td>
      <td>32.92</td>
      <td>8303800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-04-07</th>
      <td>JD</td>
      <td>2017-04-07</td>
      <td>32.20</td>
      <td>32.01</td>
      <td>31.57</td>
      <td>32.25</td>
      <td>5651000</td>
      <td>Bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 索引与切片
```


```python
jd_ts['2017-02']
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
      <th>Market</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-02-28</th>
      <td>JD</td>
      <td>2017-02-28</td>
      <td>30.97</td>
      <td>30.57</td>
      <td>30.36</td>
      <td>31.16</td>
      <td>8639500</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-27</th>
      <td>JD</td>
      <td>2017-02-27</td>
      <td>30.30</td>
      <td>30.80</td>
      <td>30.25</td>
      <td>30.91</td>
      <td>5946300</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-24</th>
      <td>JD</td>
      <td>2017-02-24</td>
      <td>30.50</td>
      <td>30.27</td>
      <td>30.03</td>
      <td>30.52</td>
      <td>4641300</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-23</th>
      <td>JD</td>
      <td>2017-02-23</td>
      <td>30.75</td>
      <td>30.61</td>
      <td>30.23</td>
      <td>30.88</td>
      <td>9920800</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-22</th>
      <td>JD</td>
      <td>2017-02-22</td>
      <td>30.27</td>
      <td>30.47</td>
      <td>30.10</td>
      <td>30.67</td>
      <td>6599800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-21</th>
      <td>JD</td>
      <td>2017-02-21</td>
      <td>30.00</td>
      <td>30.23</td>
      <td>29.81</td>
      <td>30.28</td>
      <td>5131400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-17</th>
      <td>JD</td>
      <td>2017-02-17</td>
      <td>29.57</td>
      <td>29.85</td>
      <td>29.51</td>
      <td>30.27</td>
      <td>7079600</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-16</th>
      <td>JD</td>
      <td>2017-02-16</td>
      <td>30.32</td>
      <td>30.23</td>
      <td>30.03</td>
      <td>30.57</td>
      <td>7706200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-15</th>
      <td>JD</td>
      <td>2017-02-15</td>
      <td>29.50</td>
      <td>30.14</td>
      <td>29.40</td>
      <td>30.25</td>
      <td>8001200</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-14</th>
      <td>JD</td>
      <td>2017-02-14</td>
      <td>29.48</td>
      <td>29.43</td>
      <td>29.28</td>
      <td>29.74</td>
      <td>3330800</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-13</th>
      <td>JD</td>
      <td>2017-02-13</td>
      <td>29.52</td>
      <td>29.52</td>
      <td>29.10</td>
      <td>29.65</td>
      <td>6084500</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>2017-02-10</th>
      <td>JD</td>
      <td>2017-02-10</td>
      <td>29.21</td>
      <td>29.38</td>
      <td>29.01</td>
      <td>29.52</td>
      <td>4491400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-09</th>
      <td>JD</td>
      <td>2017-02-09</td>
      <td>28.96</td>
      <td>29.01</td>
      <td>28.76</td>
      <td>29.20</td>
      <td>6234700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-08</th>
      <td>JD</td>
      <td>2017-02-08</td>
      <td>28.81</td>
      <td>28.98</td>
      <td>28.63</td>
      <td>29.06</td>
      <td>3769400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-07</th>
      <td>JD</td>
      <td>2017-02-07</td>
      <td>28.66</td>
      <td>28.81</td>
      <td>28.46</td>
      <td>28.94</td>
      <td>4834600</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-06</th>
      <td>JD</td>
      <td>2017-02-06</td>
      <td>28.64</td>
      <td>28.54</td>
      <td>28.35</td>
      <td>28.80</td>
      <td>3919200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-03</th>
      <td>JD</td>
      <td>2017-02-03</td>
      <td>28.28</td>
      <td>28.32</td>
      <td>28.08</td>
      <td>28.40</td>
      <td>3671500</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-02</th>
      <td>JD</td>
      <td>2017-02-02</td>
      <td>28.00</td>
      <td>28.17</td>
      <td>27.88</td>
      <td>28.21</td>
      <td>4403300</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-01</th>
      <td>JD</td>
      <td>2017-02-01</td>
      <td>28.59</td>
      <td>28.13</td>
      <td>28.02</td>
      <td>28.59</td>
      <td>4109200</td>
      <td>Bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
jd_ts['2017-02':'2017-03']  # Empty
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
      <th>Market</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
jd_ts.truncate(after='2017-01-06')
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
      <th>name</th>
      <th>time</th>
      <th>opening_price</th>
      <th>closing_price</th>
      <th>lowest_price</th>
      <th>highest_price</th>
      <th>volume</th>
      <th>Market</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-04-13</th>
      <td>JD</td>
      <td>2017-04-13</td>
      <td>32.74</td>
      <td>32.47</td>
      <td>32.45</td>
      <td>32.87</td>
      <td>3013600</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-12</th>
      <td>JD</td>
      <td>2017-04-12</td>
      <td>32.31</td>
      <td>32.71</td>
      <td>32.31</td>
      <td>32.88</td>
      <td>6818000</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-04-11</th>
      <td>JD</td>
      <td>2017-04-11</td>
      <td>32.70</td>
      <td>32.30</td>
      <td>32.22</td>
      <td>33.28</td>
      <td>8054200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-10</th>
      <td>JD</td>
      <td>2017-04-10</td>
      <td>32.16</td>
      <td>32.67</td>
      <td>32.15</td>
      <td>32.92</td>
      <td>8303800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-04-07</th>
      <td>JD</td>
      <td>2017-04-07</td>
      <td>32.20</td>
      <td>32.01</td>
      <td>31.57</td>
      <td>32.25</td>
      <td>5651000</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-06</th>
      <td>JD</td>
      <td>2017-04-06</td>
      <td>31.69</td>
      <td>32.23</td>
      <td>31.49</td>
      <td>32.26</td>
      <td>5840700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-04-05</th>
      <td>JD</td>
      <td>2017-04-05</td>
      <td>31.58</td>
      <td>31.53</td>
      <td>31.44</td>
      <td>32.06</td>
      <td>5368400</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-04</th>
      <td>JD</td>
      <td>2017-04-04</td>
      <td>31.80</td>
      <td>31.44</td>
      <td>31.23</td>
      <td>31.89</td>
      <td>4927500</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-04-03</th>
      <td>JD</td>
      <td>2017-04-03</td>
      <td>31.43</td>
      <td>31.92</td>
      <td>31.27</td>
      <td>32.10</td>
      <td>7934900</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-31</th>
      <td>JD</td>
      <td>2017-03-31</td>
      <td>31.40</td>
      <td>31.11</td>
      <td>31.08</td>
      <td>31.56</td>
      <td>5975200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-30</th>
      <td>JD</td>
      <td>2017-03-30</td>
      <td>31.50</td>
      <td>31.56</td>
      <td>31.47</td>
      <td>31.89</td>
      <td>4907700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-29</th>
      <td>JD</td>
      <td>2017-03-29</td>
      <td>31.50</td>
      <td>31.58</td>
      <td>31.42</td>
      <td>31.67</td>
      <td>3152800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-28</th>
      <td>JD</td>
      <td>2017-03-28</td>
      <td>31.33</td>
      <td>31.46</td>
      <td>31.21</td>
      <td>32.00</td>
      <td>5862300</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-27</th>
      <td>JD</td>
      <td>2017-03-27</td>
      <td>31.00</td>
      <td>31.38</td>
      <td>30.76</td>
      <td>31.55</td>
      <td>4055400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-24</th>
      <td>JD</td>
      <td>2017-03-24</td>
      <td>31.42</td>
      <td>31.26</td>
      <td>31.10</td>
      <td>31.84</td>
      <td>5826800</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-23</th>
      <td>JD</td>
      <td>2017-03-23</td>
      <td>30.85</td>
      <td>31.29</td>
      <td>30.61</td>
      <td>31.35</td>
      <td>8940500</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-22</th>
      <td>JD</td>
      <td>2017-03-22</td>
      <td>30.18</td>
      <td>30.58</td>
      <td>29.88</td>
      <td>30.61</td>
      <td>6790000</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-21</th>
      <td>JD</td>
      <td>2017-03-21</td>
      <td>31.88</td>
      <td>30.32</td>
      <td>30.14</td>
      <td>31.93</td>
      <td>9364400</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-20</th>
      <td>JD</td>
      <td>2017-03-20</td>
      <td>31.57</td>
      <td>31.72</td>
      <td>31.54</td>
      <td>31.87</td>
      <td>4270800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-17</th>
      <td>JD</td>
      <td>2017-03-17</td>
      <td>31.77</td>
      <td>31.56</td>
      <td>31.34</td>
      <td>31.77</td>
      <td>6464100</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-16</th>
      <td>JD</td>
      <td>2017-03-16</td>
      <td>31.87</td>
      <td>31.54</td>
      <td>31.44</td>
      <td>32.21</td>
      <td>6367500</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-15</th>
      <td>JD</td>
      <td>2017-03-15</td>
      <td>31.35</td>
      <td>31.38</td>
      <td>31.08</td>
      <td>31.53</td>
      <td>4368800</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-14</th>
      <td>JD</td>
      <td>2017-03-14</td>
      <td>31.46</td>
      <td>31.37</td>
      <td>31.11</td>
      <td>31.59</td>
      <td>4364700</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-13</th>
      <td>JD</td>
      <td>2017-03-13</td>
      <td>31.27</td>
      <td>31.69</td>
      <td>31.15</td>
      <td>31.82</td>
      <td>6473500</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-10</th>
      <td>JD</td>
      <td>2017-03-10</td>
      <td>31.35</td>
      <td>31.06</td>
      <td>30.94</td>
      <td>31.55</td>
      <td>6038500</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-09</th>
      <td>JD</td>
      <td>2017-03-09</td>
      <td>31.15</td>
      <td>31.20</td>
      <td>30.96</td>
      <td>31.51</td>
      <td>5546200</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-08</th>
      <td>JD</td>
      <td>2017-03-08</td>
      <td>30.61</td>
      <td>31.34</td>
      <td>30.60</td>
      <td>31.64</td>
      <td>8226600</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-07</th>
      <td>JD</td>
      <td>2017-03-07</td>
      <td>30.20</td>
      <td>30.75</td>
      <td>29.96</td>
      <td>30.94</td>
      <td>7488300</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-03-06</th>
      <td>JD</td>
      <td>2017-03-06</td>
      <td>30.97</td>
      <td>30.41</td>
      <td>30.24</td>
      <td>30.98</td>
      <td>8109800</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-03-03</th>
      <td>JD</td>
      <td>2017-03-03</td>
      <td>31.32</td>
      <td>30.93</td>
      <td>30.87</td>
      <td>31.58</td>
      <td>9199300</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-02-17</th>
      <td>JD</td>
      <td>2017-02-17</td>
      <td>29.57</td>
      <td>29.85</td>
      <td>29.51</td>
      <td>30.27</td>
      <td>7079600</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-16</th>
      <td>JD</td>
      <td>2017-02-16</td>
      <td>30.32</td>
      <td>30.23</td>
      <td>30.03</td>
      <td>30.57</td>
      <td>7706200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-15</th>
      <td>JD</td>
      <td>2017-02-15</td>
      <td>29.50</td>
      <td>30.14</td>
      <td>29.40</td>
      <td>30.25</td>
      <td>8001200</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-14</th>
      <td>JD</td>
      <td>2017-02-14</td>
      <td>29.48</td>
      <td>29.43</td>
      <td>29.28</td>
      <td>29.74</td>
      <td>3330800</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-13</th>
      <td>JD</td>
      <td>2017-02-13</td>
      <td>29.52</td>
      <td>29.52</td>
      <td>29.10</td>
      <td>29.65</td>
      <td>6084500</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>2017-02-10</th>
      <td>JD</td>
      <td>2017-02-10</td>
      <td>29.21</td>
      <td>29.38</td>
      <td>29.01</td>
      <td>29.52</td>
      <td>4491400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-09</th>
      <td>JD</td>
      <td>2017-02-09</td>
      <td>28.96</td>
      <td>29.01</td>
      <td>28.76</td>
      <td>29.20</td>
      <td>6234700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-08</th>
      <td>JD</td>
      <td>2017-02-08</td>
      <td>28.81</td>
      <td>28.98</td>
      <td>28.63</td>
      <td>29.06</td>
      <td>3769400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-07</th>
      <td>JD</td>
      <td>2017-02-07</td>
      <td>28.66</td>
      <td>28.81</td>
      <td>28.46</td>
      <td>28.94</td>
      <td>4834600</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-06</th>
      <td>JD</td>
      <td>2017-02-06</td>
      <td>28.64</td>
      <td>28.54</td>
      <td>28.35</td>
      <td>28.80</td>
      <td>3919200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-02-03</th>
      <td>JD</td>
      <td>2017-02-03</td>
      <td>28.28</td>
      <td>28.32</td>
      <td>28.08</td>
      <td>28.40</td>
      <td>3671500</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-02</th>
      <td>JD</td>
      <td>2017-02-02</td>
      <td>28.00</td>
      <td>28.17</td>
      <td>27.88</td>
      <td>28.21</td>
      <td>4403300</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-02-01</th>
      <td>JD</td>
      <td>2017-02-01</td>
      <td>28.59</td>
      <td>28.13</td>
      <td>28.02</td>
      <td>28.59</td>
      <td>4109200</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-31</th>
      <td>JD</td>
      <td>2017-01-31</td>
      <td>28.32</td>
      <td>28.40</td>
      <td>28.01</td>
      <td>28.46</td>
      <td>4381700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-30</th>
      <td>JD</td>
      <td>2017-01-30</td>
      <td>28.17</td>
      <td>28.56</td>
      <td>28.06</td>
      <td>28.64</td>
      <td>3595000</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-27</th>
      <td>JD</td>
      <td>2017-01-27</td>
      <td>28.35</td>
      <td>28.36</td>
      <td>28.12</td>
      <td>28.50</td>
      <td>3107000</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-26</th>
      <td>JD</td>
      <td>2017-01-26</td>
      <td>28.80</td>
      <td>28.32</td>
      <td>28.18</td>
      <td>28.80</td>
      <td>6052100</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-25</th>
      <td>JD</td>
      <td>2017-01-25</td>
      <td>28.77</td>
      <td>28.59</td>
      <td>28.35</td>
      <td>28.84</td>
      <td>7459500</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-24</th>
      <td>JD</td>
      <td>2017-01-24</td>
      <td>28.50</td>
      <td>28.54</td>
      <td>28.22</td>
      <td>28.91</td>
      <td>13843200</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-23</th>
      <td>JD</td>
      <td>2017-01-23</td>
      <td>27.50</td>
      <td>28.18</td>
      <td>27.46</td>
      <td>28.27</td>
      <td>9972400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-20</th>
      <td>JD</td>
      <td>2017-01-20</td>
      <td>27.96</td>
      <td>27.60</td>
      <td>27.54</td>
      <td>28.14</td>
      <td>7142000</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-19</th>
      <td>JD</td>
      <td>2017-01-19</td>
      <td>27.30</td>
      <td>27.75</td>
      <td>27.13</td>
      <td>27.84</td>
      <td>10279700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-18</th>
      <td>JD</td>
      <td>2017-01-18</td>
      <td>27.34</td>
      <td>27.16</td>
      <td>26.86</td>
      <td>27.50</td>
      <td>5974400</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-17</th>
      <td>JD</td>
      <td>2017-01-17</td>
      <td>26.82</td>
      <td>27.21</td>
      <td>26.71</td>
      <td>27.65</td>
      <td>7717500</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-13</th>
      <td>JD</td>
      <td>2017-01-13</td>
      <td>26.77</td>
      <td>26.84</td>
      <td>26.52</td>
      <td>26.90</td>
      <td>5929700</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-12</th>
      <td>JD</td>
      <td>2017-01-12</td>
      <td>26.83</td>
      <td>26.61</td>
      <td>26.48</td>
      <td>26.83</td>
      <td>4867700</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-11</th>
      <td>JD</td>
      <td>2017-01-11</td>
      <td>26.76</td>
      <td>26.77</td>
      <td>26.44</td>
      <td>27.18</td>
      <td>5498100</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-10</th>
      <td>JD</td>
      <td>2017-01-10</td>
      <td>26.30</td>
      <td>26.90</td>
      <td>26.25</td>
      <td>27.10</td>
      <td>20417400</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2017-01-09</th>
      <td>JD</td>
      <td>2017-01-09</td>
      <td>26.64</td>
      <td>26.26</td>
      <td>26.14</td>
      <td>26.95</td>
      <td>8071500</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2017-01-06</th>
      <td>JD</td>
      <td>2017-01-06</td>
      <td>26.30</td>
      <td>26.27</td>
      <td>25.92</td>
      <td>26.41</td>
      <td>6234300</td>
      <td>Bad</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 8 columns</p>
</div>




```python
jd_ts[['opening_price', 'closing_price']].truncate(after='2017-01-20', before='2017-01-13')  # Empty
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 范围和偏移量
```


```python
'''
pd.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None)
start:时间日期字符串指定起始时间日期
end:时间日期字符串指定终止时间日期
periods:时间日期的个数
freq:时间日期的频率
tz:时区
normalize:生成日期范围之前 将开始/结束日期标准化为午夜
name:命名时间日期索引
closed:生成的时间日期索引是/否包含start和end
'''
```




    "\npd.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None)\nstart:时间日期字符串指定起始时间日期\nend:时间日期字符串指定终止时间日期\nperiods:时间日期的个数\nfreq:时间日期的频率\ntz:时区\nnormalize:生成日期范围之前 将开始/结束日期标准化为午夜\nname:命名时间日期索引\nclosed:生成的时间日期索引是/否包含start和end\n"




```python
pd.date_range(start='2017/07/07', periods=3, freq='M')
```




    DatetimeIndex(['2017-07-31', '2017-08-31', '2017-09-30'], dtype='datetime64[ns]', freq='M')




```python
pd.date_range('2017/07/07', '2018/07/07', freq='BMS')
```




    DatetimeIndex(['2017-08-01', '2017-09-01', '2017-10-02', '2017-11-01',
                   '2017-12-01', '2018-01-01', '2018-02-01', '2018-03-01',
                   '2018-04-02', '2018-05-01', '2018-06-01', '2018-07-02'],
                  dtype='datetime64[ns]', freq='BMS')




```python
'''
B       工作日                  Q       季度末
C       自定义工作日            QS      季度初
D       日历日                  BQ      季度末工作日
W       周                      BQS     季度初工作日
M       月末                    A       年末
SM      半月及月末              BA      年末工作日
BM      月末工作日              AS      年初
CBM     自定义月末工作日        BAS     年初工作日
MS      月初                    BH      工作小时
SMS     月初及月中              H       小时
BMS     月初工作日              T,min   分钟
CBMS    自定义月初工作日        S       秒
                                L,ms    毫秒
                                U,us    微秒
                                N       纳秒
'''
```




    '\nB       工作日                  Q       季度末\nC       自定义工作日            QS      季度初\nD       日历日                  BQ      季度末工作日\nW       周                      BQS     季度初工作日\nM       月末                    A       年末\nSM      半月及月末              BA      年末工作日\nBM      月末工作日              AS      年初\nCBM     自定义月末工作日        BAS     年初工作日\nMS      月初                    BH      工作小时\nSMS     月初及月中              H       小时\nBMS     月初工作日              T,min   分钟\nCBMS    自定义月初工作日        S       秒\n                                L,ms    毫秒\n                                U,us    微秒\n                                N       纳秒\n'




```python
pd.date_range('2017/07/07', periods=10, freq='1D2h20min')
```




    DatetimeIndex(['2017-07-07 00:00:00', '2017-07-08 02:20:00',
                   '2017-07-09 04:40:00', '2017-07-10 07:00:00',
                   '2017-07-11 09:20:00', '2017-07-12 11:40:00',
                   '2017-07-13 14:00:00', '2017-07-14 16:20:00',
                   '2017-07-15 18:40:00', '2017-07-16 21:00:00'],
                  dtype='datetime64[ns]', freq='1580T')




```python
pd.date_range('2017/07/07', '2018/01/22', freq='W-WED')
```




    DatetimeIndex(['2017-07-12', '2017-07-19', '2017-07-26', '2017-08-02',
                   '2017-08-09', '2017-08-16', '2017-08-23', '2017-08-30',
                   '2017-09-06', '2017-09-13', '2017-09-20', '2017-09-27',
                   '2017-10-04', '2017-10-11', '2017-10-18', '2017-10-25',
                   '2017-11-01', '2017-11-08', '2017-11-15', '2017-11-22',
                   '2017-11-29', '2017-12-06', '2017-12-13', '2017-12-20',
                   '2017-12-27', '2018-01-03', '2018-01-10', '2018-01-17'],
                  dtype='datetime64[ns]', freq='W-WED')




```python
ts_offset = pd.tseries.offsets.Week(1) + pd.tseries.offsets.Hour(8)
```


```python
ts_offset
```




    Timedelta('7 days 08:00:00')




```python
pd.date_range('2017/07/07', periods=10, freq=ts_offset)
```




    DatetimeIndex(['2017-07-07 00:00:00', '2017-07-14 08:00:00',
                   '2017-07-21 16:00:00', '2017-07-29 00:00:00',
                   '2017-08-05 08:00:00', '2017-08-12 16:00:00',
                   '2017-08-20 00:00:00', '2017-08-27 08:00:00',
                   '2017-09-03 16:00:00', '2017-09-11 00:00:00'],
                  dtype='datetime64[ns]', freq='176H')




```python
# 时间移动及运算
```


```python
sample = jd_ts['2017-01-01': '2017-01-10'][['opening_price', 'closing_price']]
```


```python
sample
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 将时序数据向后移2期
```


```python
sample.shift(2)
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 将时序数据按天向前移2天
```


```python
sample.shift(-2, freq='1D')
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 时间序列运算
```


```python
date = pd.date_range('2017/01/01', '2017/01/08', freq='D')
```


```python
s1 = pd.DataFrame({'opening_price': np.random.randn(8), 'closing_price': np.random.randn(8)}, index=date)
```


```python
s1
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-01</th>
      <td>-1.119615</td>
      <td>-1.120856</td>
    </tr>
    <tr>
      <th>2017-01-02</th>
      <td>0.411136</td>
      <td>-1.210855</td>
    </tr>
    <tr>
      <th>2017-01-03</th>
      <td>0.136178</td>
      <td>-2.179800</td>
    </tr>
    <tr>
      <th>2017-01-04</th>
      <td>0.279866</td>
      <td>-0.508876</td>
    </tr>
    <tr>
      <th>2017-01-05</th>
      <td>-0.068631</td>
      <td>-0.593860</td>
    </tr>
    <tr>
      <th>2017-01-06</th>
      <td>-0.512813</td>
      <td>-0.526601</td>
    </tr>
    <tr>
      <th>2017-01-07</th>
      <td>-0.935569</td>
      <td>0.020359</td>
    </tr>
    <tr>
      <th>2017-01-08</th>
      <td>0.881377</td>
      <td>-0.580553</td>
    </tr>
  </tbody>
</table>
</div>




```python
s1 + sample
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-03</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-04</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-06</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-07</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-01-08</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 频率转换及重采样
```


```python
sample.asfreq(freq='D')
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 重采样
```


```python
# 按照12小时频率进行上采样,并指定缺失值按当日最后一个有效观测值来填充
```


```python
sample.resample('12H').ffill()
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 按照4天频率进行下采样 ohlc表示时序初始值,最大值,最小值,时序终止数据
```


```python
sample.resample('4D').ohlc()
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 提取股票交易周均开,收盘价信息
```


```python
sample.groupby(lambda x: x.week).mean()
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 提取股票交易月均开,收盘价信息
```


```python
jd_ts[['opening_price', 'closing_price']].groupby(lambda x: x.month).mean()
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
      <th>opening_price</th>
      <th>closing_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>27.279000</td>
      <td>27.314500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.491053</td>
      <td>29.550526</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31.240000</td>
      <td>31.177826</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32.067778</td>
      <td>32.142222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 缺失值处理
```


```python
# 缺失数据的形式
```


```python
scoresheet = pd.DataFrame({'Name': ['Christoph', 'Morgan', 'Mickel', 'Jones'],
                           'Economics': [89, 97, 56, 82],
                           'Statistics': [98, 93, 76, 85]})
```


```python
scoresheet
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Morgan</td>
      <td>97</td>
      <td>93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jones</td>
      <td>82</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet['Datamining'] = [79, np.nan, None, 89]
```


```python
scoresheet.loc[[1, 3], ['Name']] = [np.nan, None]
```


```python
scoresheet
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>97</td>
      <td>93</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 缺失值在默认情况下不参与运算
```


```python
scoresheet['Datamining'].mean()
```




    84.0




```python
scoresheet['Datamining'].mean() == (79 + 89) / 2
```




    True




```python
# 时间戳的datetime64[ns]数据格式, 默认缺失值为'NaT'
```


```python
scoresheet['Exam_Date'] = pd.date_range('20170707', periods=4)
```


```python
scoresheet['Exam_Date']
```




    0   2017-07-07
    1   2017-07-08
    2   2017-07-09
    3   2017-07-10
    Name: Exam_Date, dtype: datetime64[ns]




```python
scoresheet.loc[[2, 3], ['Exam_Date']] = np.nan
```


```python
scoresheet
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>97</td>
      <td>93</td>
      <td>NaN</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 缺失数据填充与清洗
```


```python
'''
对缺失值进行填充
value:填充缺失值的标量或字典对象
methond:指定填充方法:'backfill'/'bfill'向后填充, 'pad'/'ffill'向前填充, 'ffill'默认值, None
axis: 指定待填充的轴: 0, 1或'index', 'columns', 默认axis=0
inplace: 指定是否修改对象上的任何其他视图
limit: 指定'ffill'和'backfill'填充可连续填充的最大数量
'''
```




    "\n对缺失值进行填充\nvalue:填充缺失值的标量或字典对象\nmethond:指定填充方法:'backfill'/'bfill'向后填充, 'pad'/'ffill'向前填充, 'ffill'默认值, None\naxis: 指定待填充的轴: 0, 1或'index', 'columns', 默认axis=0\ninplace: 指定是否修改对象上的任何其他视图\nlimit: 指定'ffill'和'backfill'填充可连续填充的最大数量\n"




```python
scoresheet.fillna(0)
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>97</td>
      <td>93</td>
      <td>0.0</td>
      <td>2017-07-08 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet['Name'].fillna('missing')
```




    0    Christoph
    1      missing
    2       Mickel
    3      missing
    Name: Name, dtype: object




```python
scoresheet.fillna(method='pad')
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Christoph</td>
      <td>97</td>
      <td>93</td>
      <td>79.0</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>79.0</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mickel</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>2017-07-08</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.fillna(method='bfill')
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mickel</td>
      <td>97</td>
      <td>93</td>
      <td>89.0</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.bfill()
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mickel</td>
      <td>97</td>
      <td>93</td>
      <td>89.0</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.ffill(limit=1)
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Christoph</td>
      <td>97</td>
      <td>93</td>
      <td>79.0</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>NaN</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mickel</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet['Datamining'].fillna(scoresheet['Datamining'].mean())
```




    0    79.0
    1    84.0
    2    84.0
    3    89.0
    Name: Datamining, dtype: float64




```python
scoresheet.dropna(axis=0)
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>2017-07-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.dropna(how='any', axis=1)
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
      <th>Economics</th>
      <th>Statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>89</td>
      <td>98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>97</td>
      <td>93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>82</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.loc[[0], ['Exam_Date']] = np.nan
```


```python
scoresheet
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>97</td>
      <td>93</td>
      <td>NaN</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.dropna(how='any', thresh=2, axis=1)
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>97</td>
      <td>93</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 缺失数据差值
```


```python
# 线性插值
```


```python
scoresheet.interpolate(method='linear')
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.000000</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>97</td>
      <td>93</td>
      <td>82.333333</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>85.666667</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.000000</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
scoresheet.interpolate(method='polynomial', order=1)
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
      <th>Name</th>
      <th>Economics</th>
      <th>Statistics</th>
      <th>Datamining</th>
      <th>Exam_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Christoph</td>
      <td>89</td>
      <td>98</td>
      <td>79.000000</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>97</td>
      <td>93</td>
      <td>82.333333</td>
      <td>2017-07-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mickel</td>
      <td>56</td>
      <td>76</td>
      <td>85.666667</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>82</td>
      <td>85</td>
      <td>89.000000</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>


