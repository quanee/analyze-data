

```python
import numpy as np
```


```python
v = np.arange(10)
```


```python
v
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
v.dtype
```




    dtype('int32')




```python
v.shape
```




    (10,)




```python
# 返回列表
vstep = np.arange(0, 10, 0.5)
```


```python
vstep
```




    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
           6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])




```python
vstep * 10
```




    array([ 0.,  5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.,
           65., 70., 75., 80., 85., 90., 95.])




```python
# 等差数列 初始值1 终止值19 元素个数为10
```


```python
np.linspace(1, 19, 10)
```




    array([ 1.,  3.,  5.,  7.,  9., 11., 13., 15., 17., 19.])




```python
# 等比数列
```


```python
from math import e
```


```python
np.logspace(1, 20, 10, endpoint=False, base=e)
```




    array([2.71828183e+00, 1.81741454e+01, 1.21510418e+02, 8.12405825e+02,
           5.43165959e+03, 3.63155027e+04, 2.42801617e+05, 1.62334599e+06,
           1.08535199e+07, 7.25654884e+07])




```python
# 创建全0整型向量
```


```python
np.zeros(10, np.int)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
np.empty(10, np.int)
```




    array([-952650496,        606, 1661424176, 1988385690, 1324770695,
                12290,          0,    3211382,    3145774,    5046364])




```python
# 创建随机数向量
```


```python
np.random.randn(10)
```




    array([ 1.6076276 , -0.33808963,  1.48854085,  0.40731709,  1.45744737,
            0.12829737,  2.63979199, -0.16847692, -0.55002947, -0.87905684])




```python
s = 'Hello, Python!'
```


```python
np.fromstring(s, dtype=np.int8)
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
      """Entry point for launching an IPython kernel.
    




    array([ 72, 101, 108, 108, 111,  44,  32,  80, 121, 116, 104, 111, 110,
            33], dtype=int8)




```python
def multiply99(i, j):
    return (i + 1) * (j + 1)
```


```python
np.fromfunction(multiply99, (9, 9))
```




    array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
           [ 2.,  4.,  6.,  8., 10., 12., 14., 16., 18.],
           [ 3.,  6.,  9., 12., 15., 18., 21., 24., 27.],
           [ 4.,  8., 12., 16., 20., 24., 28., 32., 36.],
           [ 5., 10., 15., 20., 25., 30., 35., 40., 45.],
           [ 6., 12., 18., 24., 30., 36., 42., 48., 54.],
           [ 7., 14., 21., 28., 35., 42., 49., 56., 63.],
           [ 8., 16., 24., 32., 40., 48., 56., 64., 72.],
           [ 9., 18., 27., 36., 45., 54., 63., 72., 81.]])




```python
# 数组
```


```python
a = np.array([np.arange(3), np.arange(3)])
```


```python
a
```




    array([[0, 1, 2],
           [0, 1, 2]])




```python
a.shape  # 形状
```




    (2, 3)




```python
a.ndim  # 维度
```




    2




```python
# 单位矩阵
```


```python
np.identity(9).astype(np.int8)
```




    array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=int8)




```python
a.tolist()
```




    [[0, 1, 2], [0, 1, 2]]




```python
type(a.tolist())
```




    list




```python
# 查看ndarray数据类型
set(np.typeDict.values())
```




    {numpy.bool_,
     numpy.bytes_,
     numpy.complex128,
     numpy.complex128,
     numpy.complex64,
     numpy.datetime64,
     numpy.float16,
     numpy.float32,
     numpy.float64,
     numpy.float64,
     numpy.int16,
     numpy.int32,
     numpy.int32,
     numpy.int64,
     numpy.int8,
     numpy.object_,
     numpy.str_,
     numpy.timedelta64,
     numpy.uint16,
     numpy.uint32,
     numpy.uint32,
     numpy.uint64,
     numpy.uint8,
     numpy.void}




```python
# 结构数组
goodslist = np.dtype([('name', np.str_, 50), ('location', np.str_, 30), ('price', np.float16), ('volume', np.int32)])
```


```python
goods = np.array([('Gree Airconditioner', 'JD.com', 6245, 1),
                  ('Sony Blueray Player', 'Amazon,com', 3210, 2),
                  ('Apple Mackbook Pro 13', 'Tmall.com', 12388, 5),
                  ('iPhoneSE', 'JD.com', 4588, 2)], dtype=goodslist)
```


```python
goods
```




    array([('Gree Airconditioner', 'JD.com',  6244., 1),
           ('Sony Blueray Player', 'Amazon,com',  3210., 2),
           ('Apple Mackbook Pro 13', 'Tmall.com', 12380., 5),
           ('iPhoneSE', 'JD.com',  4588., 2)],
          dtype=[('name', '<U50'), ('location', '<U30'), ('price', '<f2'), ('volume', '<i4')])




```python
# 使用字典定义结果数组
goodsdict = np.dtype({'names': ['name', 'location', 'price', 'volume'], 'formats': ['S50', 'S30', 'f', 'i']})
```


```python
goods_new = np.array([('Gree Airconditioner', 'JD.com', 6245, 1),
                      ('Sony Blueray Player', 'Amazon,com', 3210, 2),
                      ('Apple Mackbook Pro 13', 'Tmall.com', 12388, 5),
                      ('iPhoneSE', 'JD.com', 4588, 2)], dtype=goodsdict)
```


```python
goods_new
```




    array([(b'Gree Airconditioner', b'JD.com',  6245., 1),
           (b'Sony Blueray Player', b'Amazon,com',  3210., 2),
           (b'Apple Mackbook Pro 13', b'Tmall.com', 12388., 5),
           (b'iPhoneSE', b'JD.com',  4588., 2)],
          dtype=[('name', 'S50'), ('location', 'S30'), ('price', '<f4'), ('volume', '<i4')])




```python
# 索引与切片
```


```python
# ":"分割起止位置与间隔 ","表示不同维度 "..."遍历剩余维度
```


```python
a = np.arange(1, 20, 2)
```


```python
a
```




    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])




```python
a[3]
```




    7




```python
a[1:4]
```




    array([3, 5, 7])




```python
a[:2]
```




    array([1, 3])




```python
a[-2]
```




    17




```python
a[::-1]
```




    array([19, 17, 15, 13, 11,  9,  7,  5,  3,  1])




```python
b = np.arange(24).reshape(2, 3, 4)
```


```python
b
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
b.shape
```




    (2, 3, 4)




```python
b[1, 1, 2]
```




    18




```python
b[0, 2, :]
```




    array([ 8,  9, 10, 11])




```python
b[0, 2]
```




    array([ 8,  9, 10, 11])




```python
b[0, ...]
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
b[0]
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
b[:, 1]
```




    array([[ 4,  5,  6,  7],
           [16, 17, 18, 19]])




```python
b[:, :, 1]
```




    array([[ 1,  5,  9],
           [13, 17, 21]])




```python
b[..., 1]
```




    array([[ 1,  5,  9],
           [13, 17, 21]])




```python
b[0, ::2, -2]
```




    array([ 2, 10])




```python
goods['name']
```




    array(['Gree Airconditioner', 'Sony Blueray Player',
           'Apple Mackbook Pro 13', 'iPhoneSE'], dtype='<U50')




```python
goods[3]
```




    ('iPhoneSE', 'JD.com', 4588., 2)




```python
goods[3]['name']
```




    'iPhoneSE'




```python
sum(goods['volume'])
```




    10




```python
# 逻辑索引(布尔索引 条件索引)
```


```python
b[b >= 15]
```




    array([15, 16, 17, 18, 19, 20, 21, 22, 23])




```python
b[~(b >= 15)]
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
# 逻辑运算符and, or, 在布尔数组中无效
```


```python
b[(b >= 5) & (b <= 15)]
```




    array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
b_bool1 = np.array([False, True], dtype=bool)
```


```python
b[b_bool1]
```




    array([[[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
b_bool2 = np.array([False, True, True], dtype=bool)
```


```python
b_bool3 = np.array([False, True, True, False], dtype=bool)
```


```python
b[b_bool1, b_bool2]
```




    array([[16, 17, 18, 19],
           [20, 21, 22, 23]])




```python
b[b_bool1, b_bool2, b_bool3]
```




    array([17, 22])




```python
# 花式索引
```


```python
b[[[0], [1, 2], [2, 3]]]
```




    array([ 6, 11])




```python
# ix_函数将若干一维整数数组转换为一个用于选取矩形区域的索引器
```


```python
b[np.ix_([1, 0])]
```




    array([[[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]],
    
           [[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])




```python
b[np.ix_([1, 0], [2, 1])]
```




    array([[[20, 21, 22, 23],
            [16, 17, 18, 19]],
    
           [[ 8,  9, 10, 11],
            [ 4,  5,  6,  7]]])




```python
b[np.ix_([1, 0], [2, 1], [0, 3, 2])]
```




    array([[[20, 23, 22],
            [16, 19, 18]],
    
           [[ 8, 11, 10],
            [ 4,  7,  6]]])




```python
'''
数组切片是原始数组的视图, 与原始数组共享同一块数据存储空间
'''
```




    '\n数组切片是原始数组的视图, 与原始数组共享同一块数据存储空间\n'




```python
b_slice = b[0, 1, 1:3]
```


```python
b_copy = b[0, 1, 1:3].copy()
```


```python
b_slice
```




    array([5, 6])




```python
b_copy
```




    array([5, 6])




```python
b_slice[1] = 666
```


```python
b_slice
```




    array([  5, 666])




```python
b
```




    array([[[  0,   1,   2,   3],
            [  4,   5, 666,   7],
            [  8,   9,  10,  11]],
    
           [[ 12,  13,  14,  15],
            [ 16,  17,  18,  19],
            [ 20,  21,  22,  23]]])




```python
b_copy[1] = 999
```


```python
b_copy
```




    array([  5, 999])




```python
b
```




    array([[[  0,   1,   2,   3],
            [  4,   5, 666,   7],
            [  8,   9,  10,  11]],
    
           [[ 12,  13,  14,  15],
            [ 16,  17,  18,  19],
            [ 20,  21,  22,  23]]])




```python
# 数组属性
```


```python
ac = np.arange(12)
```


```python
ac.shape = (2, 2, 3)
```


```python
ac
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])




```python
# 数组形状
```


```python
ac.shape
```




    (2, 2, 3)




```python
# 数组各元素类型
ac.dtype
```




    dtype('int32')




```python
# 数组维数
ac.ndim
```




    3




```python
# 数组元素总个数
ac.size
```




    12




```python
# 数组元素在内存中所占字节数
ac.itemsize
```




    4




```python
# 数组元素所占存储空间(size与itemsize)
ac.nbytes
```




    48




```python
# 转置数组
ac.T
```




    array([[[ 0,  6],
            [ 3,  9]],
    
           [[ 1,  7],
            [ 4, 10]],
    
           [[ 2,  8],
            [ 5, 11]]])




```python
# 数组扁平迭代器(像遍历一维数组一样去遍历任意多维数组)
ac.flat
```




    <numpy.flatiter at 0x25ecb29fe10>




```python
# 数组排序
s = np.array([1, 2, 4, 3, 1, 2, 2, 4, 6, 7, 2, 4, 8, 4, 5])
```


```python
# 返回排序后的数组
np.sort(s)
```




    array([1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8])




```python
# 返回数组排序后的下标
np.argsort(s)
```




    array([ 0,  4,  1,  5,  6, 10,  3,  2,  7, 11, 13, 14,  8,  9, 12],
          dtype=int64)




```python
# 降序排序
s[np.argsort(-s)]
```




    array([8, 7, 6, 5, 4, 4, 4, 4, 3, 2, 2, 2, 2, 1, 1])




```python
# 就地排序
s.sort()
s
```




    array([1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8])




```python
# 多维数组排序
s_r = np.array([3, 23, 52, 34, 52, 3, 6, 645, 34, 7, 85, 23]).reshape(6, 2)
```


```python
s_r
```




    array([[  3,  23],
           [ 52,  34],
           [ 52,   3],
           [  6, 645],
           [ 34,   7],
           [ 85,  23]])




```python
s_r.sort(axis=1)
```


```python
s_r
```




    array([[  3,  23],
           [ 34,  52],
           [  3,  52],
           [  6, 645],
           [  7,  34],
           [ 23,  85]])




```python
s_r.sort(axis=0)
```


```python
s_r
```




    array([[  3,  23],
           [  3,  34],
           [  6,  52],
           [  7,  52],
           [ 23,  85],
           [ 34, 645]])




```python
s_r.sort(axis=-1)
```


```python
s_r
```




    array([[  3,  23],
           [  3,  34],
           [  6,  52],
           [  7,  52],
           [ 23,  85],
           [ 34, 645]])




```python
# 指定排序顺序
a = [1, 5, 1, 4, 3, 4, 4]
```


```python
b = [9, 4, 0, 4, 0, 2, 1]
```


```python
# 先按a排序 再按b排序
ind = np.lexsort((b, a))
```


```python
[(a[i], b[i]) for i in ind]
```




    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]




```python
# 数组维度
# 展平
b = np.arange(24).reshape(2, 3, 4)
```


```python
b
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
b.ndim
```




    3




```python
# 展平为一维数组(返回视图)
br = np.ravel(b)
```


```python
br
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23])




```python
br.ndim
```




    1




```python
# reshape函数也可以达到相同效果, 但维度不变
brsh = b.reshape(1, 1, 24)
```


```python
brsh
```




    array([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23]]])




```python
brsh.ndim
```




    3




```python
# 展平为一维数组(返回新对象, 分配内存)
b.flatten()
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23])




```python
# 维度改变
# 返回视图
bd = b.reshape(4, 6)
```


```python
bd
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]])




```python
b.shape = (1, 1, 24)
```


```python
b
```




    array([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23]]])




```python
# 直接修改数组
b.resize(1, 1, 24)
```


```python
b
```




    array([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23]]])




```python
# 转置
b.shape = (3, 4, 2)
```


```python
b
```




    array([[[ 0,  1],
            [ 2,  3],
            [ 4,  5],
            [ 6,  7]],
    
           [[ 8,  9],
            [10, 11],
            [12, 13],
            [14, 15]],
    
           [[16, 17],
            [18, 19],
            [20, 21],
            [22, 23]]])




```python
# 等价于 b.T
np.transpose(b)
```




    array([[[ 0,  8, 16],
            [ 2, 10, 18],
            [ 4, 12, 20],
            [ 6, 14, 22]],
    
           [[ 1,  9, 17],
            [ 3, 11, 19],
            [ 5, 13, 21],
            [ 7, 15, 23]]])




```python
# 数组组合
# 水平组合
a = np.arange(9).reshape(3, 3)
```


```python
a
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
b = np.array([[0, 11, 22, 33], [44, 55, 66, 77], [88, 00, 00, 11]])
```


```python
b
```




    array([[ 0, 11, 22, 33],
           [44, 55, 66, 77],
           [88,  0,  0, 11]])




```python
np.hstack((a, b))
```




    array([[ 0,  1,  2,  0, 11, 22, 33],
           [ 3,  4,  5, 44, 55, 66, 77],
           [ 6,  7,  8, 88,  0,  0, 11]])




```python
np.concatenate((a, b), axis=1)
```




    array([[ 0,  1,  2,  0, 11, 22, 33],
           [ 3,  4,  5, 44, 55, 66, 77],
           [ 6,  7,  8, 88,  0,  0, 11]])




```python
c = np.array([[0, 11, 22], [44, 55, 66], [88, 99, 00], [22, 33, 44]])
```


```python
c
# print(np.hstack((a, c)))  # ValueError: all the input array dimensions except for the concatenation axis must match exactly
```




    array([[ 0, 11, 22],
           [44, 55, 66],
           [88, 99,  0],
           [22, 33, 44]])




```python
# 垂直组合
np.vstack((a, c))
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 0, 11, 22],
           [44, 55, 66],
           [88, 99,  0],
           [22, 33, 44]])




```python
np.concatenate((a, c), axis=0)
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 0, 11, 22],
           [44, 55, 66],
           [88, 99,  0],
           [22, 33, 44]])




```python
# 深度组合
d = np.delete(b, 3, axis=1)  # 删除数组中指定数据, axis=1表示列, axis=0表示行
```


```python
d
```




    array([[ 0, 11, 22],
           [44, 55, 66],
           [88,  0,  0]])




```python
np.dstack((a, d))
```




    array([[[ 0,  0],
            [ 1, 11],
            [ 2, 22]],
    
           [[ 3, 44],
            [ 4, 55],
            [ 5, 66]],
    
           [[ 6, 88],
            [ 7,  0],
            [ 8,  0]]])




```python
# 列组合
a1 = np.arange(4)
```


```python
a2 = np.arange(4) * 2
```


```python
np.column_stack((a1, a2))
```




    array([[0, 0],
           [1, 2],
           [2, 4],
           [3, 6]])




```python
# 行组合
np.row_stack((a1, a2))
```




    array([[0, 1, 2, 3],
           [0, 2, 4, 6]])




```python
# 数组分拆
```


```python
# 水平分拆
a = np.arange(9).reshape(3, 3)
```


```python
a
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
ahs = np.hsplit(a, 3)
```


```python
ahs
```




    [array([[0],
            [3],
            [6]]), array([[1],
            [4],
            [7]]), array([[2],
            [5],
            [8]])]




```python
# 列表
type(ahs)
```




    list




```python
# 元素是numpy数组
type(ahs[1])
```




    numpy.ndarray




```python
np.split(a, 3, axis=1)
```




    [array([[0],
            [3],
            [6]]), array([[1],
            [4],
            [7]]), array([[2],
            [5],
            [8]])]




```python
# 垂直分拆
np.vsplit(a, 3)
```




    [array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]])]




```python
# 深度分拆
ads = np.arange(12)
```


```python
ads.shape = (2, 2, 3)
```


```python
ads
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])




```python
np.dsplit(ads, 3)
```




    [array([[[0],
             [3]],
     
            [[6],
             [9]]]), array([[[ 1],
             [ 4]],
     
            [[ 7],
             [10]]]), array([[[ 2],
             [ 5]],
     
            [[ 8],
             [11]]])]




```python
# ufunc运算
# 函数运算 比较运算 布尔运算
a1 = np.arange(0, 10)
```


```python
a1
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
a2 = np.arange(0, 20, 2)
```


```python
a2
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
# ufunc运算
# 函数运算 比较运算 布尔运算
```


```python
a1 = np.arange(0, 10)
```


```python
a1
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
a2 = np.arange(0, 20, 2)
```


```python
a2
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
# out后面的数组必须事先定义 可以省略out
a3 = np.add(a1, a2, out=a1)
```


```python
a3, a1
```




    (array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27]),
     array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27]))




```python
id(a3) == id(a1)
```




    True




```python
a1 > a2
```




    array([False,  True,  True,  True,  True,  True,  True,  True,  True,
            True])




```python
any(a1 > a2)
```




    True




```python
all(a1 > a2)
```




    False




```python
# ufunc函数比Python内置函数快
# def mathcal(n):
#     s = []
#     for i in range(n + 1):
#         s.append(100 + i)
#     return
```


```python
# def ufunccal(n):
```


```python
#     s = np.array(range(n + 1)) + 100
#     return
```


```python
# import timeit

# print(timeit.timeit('mathcal(1000000)', globals=globals()))
# print(timeit.timeit('ufunccal(1000000)', globals=globals()))
```


```python
# 自定义ufunc函数
def liftscore(n):
    n_new = np.sqrt((n ^ 2) * 100)
    return n_new
```


```python
score = np.array([87, 77, 56, 100, 60])
```


```python
# 定义ufunc函数
score_1 = np.frompyfunc(liftscore, 1, 1)(score)
```


```python
score_1
```




    array([92.19544457292888, 88.88194417315589, 76.15773105863909,
           100.99504938362078, 78.74007874011811], dtype=object)




```python
# ufunc函数返回数组类型为object 转换为float
score_1 = score_1.astype(float)
```


```python
score_1.dtype
```




    dtype('float64')




```python
score_2 = np.vectorize(liftscore, otypes=[float])(score)
```


```python
any(score_1 == score_2)
```




    True




```python
# 广播
'''
  * 所有输入输入数组向维数最多的数组看齐 shape属性
  * 输出数组的shape属性是输入数组的shape属性在各个轴上的最大值
  * 当输入数组的某个轴长度为1时 沿此轴运算时都用此轴上的第一组值
'''
```




    '\n  * 所有输入输入数组向维数最多的数组看齐 shape属性\n  * 输出数组的shape属性是输入数组的shape属性在各个轴上的最大值\n  * 当输入数组的某个轴长度为1时 沿此轴运算时都用此轴上的第一组值\n'




```python
a = np.arange(0, 10).reshape(5, 2)
```


```python
a
```




    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])




```python
b = np.arange(0, 1, 0.2).reshape(5, 1)
```


```python
b
```




    array([[0. ],
           [0.2],
           [0.4],
           [0.6],
           [0.8]])




```python
c = a + b
```


```python
c
```




    array([[0. , 1. ],
           [2.2, 3.2],
           [4.4, 5.4],
           [6.6, 7.6],
           [8.8, 9.8]])




```python
c.shape
```




    (5, 2)




```python
# 返回用来广播计算的数组
x, y = np.ogrid[:5, :5]
```


```python
x
```




    array([[0],
           [1],
           [2],
           [3],
           [4]])




```python
y
```




    array([[0, 1, 2, 3, 4]])




```python
x1 = np.ogrid[:5, : 5]
```


```python
x1
```




    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]




```python
# 返回广播后的数组
x2 = np.mgrid[:5, :5]
```


```python
x2
```




    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
    
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])




```python
# ufunc方法
np.add.reduce(np.arange(5))
```




    10




```python
np.add.reduce([[1, 2, 3, 4], [5, 6, 7, 8]], axis=1)
```




    array([10, 26])




```python
np.add.reduce([[1, 2, 3, 4], [5, 6, 7, 8]], axis=0)
```




    array([ 6,  8, 10, 12])




```python
np.add.accumulate(np.arange(5))
```




    array([ 0,  1,  3,  6, 10], dtype=int32)




```python
np.add.accumulate([[1, 2, 3, 4], [5, 6, 7, 8]], axis=1)
```




    array([[ 1,  3,  6, 10],
           [ 5, 11, 18, 26]], dtype=int32)




```python
np.add.accumulate([[1, 2, 3, 4], [5, 6, 7, 8]], axis=0)
```




    array([[ 1,  2,  3,  4],
           [ 6,  8, 10, 12]], dtype=int32)




```python
ara = np.arange(8)
```


```python
ara
```




    array([0, 1, 2, 3, 4, 5, 6, 7])




```python
np.add.reduceat(ara, indices=[0, 4, 1, 5, 2, 6, 3, 7])
```




    array([ 6,  4, 10,  5, 14,  6, 18,  7], dtype=int32)




```python
np.add.reduceat(ara, [0, 4, 1, 5, 2, 6, 3, 7])[::2]
```




    array([ 6, 10, 14, 18], dtype=int32)




```python
np.add.outer([1, 2, 3, 4], [5, 6, 7, 8])
```




    array([[ 6,  7,  8,  9],
           [ 7,  8,  9, 10],
           [ 8,  9, 10, 11],
           [ 9, 10, 11, 12]])




```python
np.multiply.outer([1, 2, 3], [5, 6, 7, 8])
```




    array([[ 5,  6,  7,  8],
           [10, 12, 14, 16],
           [15, 18, 21, 24]])




```python
# 矩阵
m1 = np.mat([[1, 2, 3], [4, 5, 6]])
```


```python
m1
```




    matrix([[1, 2, 3],
            [4, 5, 6]])




```python
m1 * 8
```




    matrix([[ 8, 16, 24],
            [32, 40, 48]])




```python
m2 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```


```python
m1 * m2
# print(m2.I)  # 逆矩阵
```




    matrix([[30, 36, 42],
            [66, 81, 96]])




```python
# 文件读写
np.savetxt('m2.txt', m2)
```


```python
m2_reload = np.loadtxt('m2.txt', delimiter=' ')
```


```python
m2_reload
```




    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])




```python
stock = np.dtype([('name', np.str_, 4), ('time', np.str_, 10), ('opening_price', np.float64), ('closing_price', np.float64), ('lowest_price', np.float64), ('highest_price', np.float64), ('volume', np.int32)])
jd_stoct = np.loadtxt('data.csv', delimiter=',', dtype=stock)
```


```python
jd_stoct
```




    array([('JD', '3-Jan-17', 25.95, 25.82, 25.64, 26.11,  8275300),
           ('JD', '4-Jan-17', 26.05, 25.85, 25.58, 26.08,  7862800),
           ('JD', '5-Jan-17', 26.15, 26.3 , 26.05, 26.8 , 10205600),
           ('JD', '6-Jan-17', 26.3 , 26.27, 25.92, 26.41,  6234300),
           ('JD', '9-Jan-17', 26.64, 26.26, 26.14, 26.95,  8071500),
           ('JD', '10-Jan-17', 26.3 , 26.9 , 26.25, 27.1 , 20417400),
           ('JD', '11-Jan-17', 26.76, 26.77, 26.44, 27.18,  5498100),
           ('JD', '12-Jan-17', 26.83, 26.61, 26.48, 26.83,  4867700),
           ('JD', '13-Jan-17', 26.77, 26.84, 26.52, 26.9 ,  5929700),
           ('JD', '17-Jan-17', 26.82, 27.21, 26.71, 27.65,  7717500),
           ('JD', '18-Jan-17', 27.34, 27.16, 26.86, 27.5 ,  5974400),
           ('JD', '19-Jan-17', 27.3 , 27.75, 27.13, 27.84, 10279700),
           ('JD', '20-Jan-17', 27.96, 27.6 , 27.54, 28.14,  7142000),
           ('JD', '23-Jan-17', 27.5 , 28.18, 27.46, 28.27,  9972400),
           ('JD', '24-Jan-17', 28.5 , 28.54, 28.22, 28.91, 13843200),
           ('JD', '25-Jan-17', 28.77, 28.59, 28.35, 28.84,  7459500),
           ('JD', '26-Jan-17', 28.8 , 28.32, 28.18, 28.8 ,  6052100),
           ('JD', '27-Jan-17', 28.35, 28.36, 28.12, 28.5 ,  3107000),
           ('JD', '30-Jan-17', 28.17, 28.56, 28.06, 28.64,  3595000),
           ('JD', '31-Jan-17', 28.32, 28.4 , 28.01, 28.46,  4381700),
           ('JD', '1-Feb-17', 28.59, 28.13, 28.02, 28.59,  4109200),
           ('JD', '2-Feb-17', 28.  , 28.17, 27.88, 28.21,  4403300),
           ('JD', '3-Feb-17', 28.28, 28.32, 28.08, 28.4 ,  3671500),
           ('JD', '6-Feb-17', 28.64, 28.54, 28.35, 28.8 ,  3919200),
           ('JD', '7-Feb-17', 28.66, 28.81, 28.46, 28.94,  4834600),
           ('JD', '8-Feb-17', 28.81, 28.98, 28.63, 29.06,  3769400),
           ('JD', '9-Feb-17', 28.96, 29.01, 28.76, 29.2 ,  6234700),
           ('JD', '10-Feb-17', 29.21, 29.38, 29.01, 29.52,  4491400),
           ('JD', '13-Feb-17', 29.52, 29.52, 29.1 , 29.65,  6084500),
           ('JD', '14-Feb-17', 29.48, 29.43, 29.28, 29.74,  3330800),
           ('JD', '15-Feb-17', 29.5 , 30.14, 29.4 , 30.25,  8001200),
           ('JD', '16-Feb-17', 30.32, 30.23, 30.03, 30.57,  7706200),
           ('JD', '17-Feb-17', 29.57, 29.85, 29.51, 30.27,  7079600),
           ('JD', '21-Feb-17', 30.  , 30.23, 29.81, 30.28,  5131400),
           ('JD', '22-Feb-17', 30.27, 30.47, 30.1 , 30.67,  6599800),
           ('JD', '23-Feb-17', 30.75, 30.61, 30.23, 30.88,  9920800),
           ('JD', '24-Feb-17', 30.5 , 30.27, 30.03, 30.52,  4641300),
           ('JD', '27-Feb-17', 30.3 , 30.8 , 30.25, 30.91,  5946300),
           ('JD', '28-Feb-17', 30.97, 30.57, 30.36, 31.16,  8639500),
           ('JD', '1-Mar-17', 30.83, 30.67, 30.56, 30.99,  7942600),
           ('JD', '2-Mar-17', 31.74, 30.93, 30.48, 32.47, 22176000),
           ('JD', '3-Mar-17', 31.32, 30.93, 30.87, 31.58,  9199300),
           ('JD', '6-Mar-17', 30.97, 30.41, 30.24, 30.98,  8109800),
           ('JD', '7-Mar-17', 30.2 , 30.75, 29.96, 30.94,  7488300),
           ('JD', '8-Mar-17', 30.61, 31.34, 30.6 , 31.64,  8226600),
           ('JD', '9-Mar-17', 31.15, 31.2 , 30.96, 31.51,  5546200),
           ('JD', '10-Mar-17', 31.35, 31.06, 30.94, 31.55,  6038500),
           ('JD', '13-Mar-17', 31.27, 31.69, 31.15, 31.82,  6473500),
           ('JD', '14-Mar-17', 31.46, 31.37, 31.11, 31.59,  4364700),
           ('JD', '15-Mar-17', 31.35, 31.38, 31.08, 31.53,  4368800),
           ('JD', '16-Mar-17', 31.87, 31.54, 31.44, 32.21,  6367500),
           ('JD', '17-Mar-17', 31.77, 31.56, 31.34, 31.77,  6464100),
           ('JD', '20-Mar-17', 31.57, 31.72, 31.54, 31.87,  4270800),
           ('JD', '21-Mar-17', 31.88, 30.32, 30.14, 31.93,  9364400),
           ('JD', '22-Mar-17', 30.18, 30.58, 29.88, 30.61,  6790000),
           ('JD', '23-Mar-17', 30.85, 31.29, 30.61, 31.35,  8940500),
           ('JD', '24-Mar-17', 31.42, 31.26, 31.1 , 31.84,  5826800),
           ('JD', '27-Mar-17', 31.  , 31.38, 30.76, 31.55,  4055400),
           ('JD', '28-Mar-17', 31.33, 31.46, 31.21, 32.  ,  5862300),
           ('JD', '29-Mar-17', 31.5 , 31.58, 31.42, 31.67,  3152800),
           ('JD', '30-Mar-17', 31.5 , 31.56, 31.47, 31.89,  4907700),
           ('JD', '31-Mar-17', 31.4 , 31.11, 31.08, 31.56,  5975200),
           ('JD', '3-Apr-17', 31.43, 31.92, 31.27, 32.1 ,  7934900),
           ('JD', '4-Apr-17', 31.8 , 31.44, 31.23, 31.89,  4927500),
           ('JD', '5-Apr-17', 31.58, 31.53, 31.44, 32.06,  5368400),
           ('JD', '6-Apr-17', 31.69, 32.23, 31.49, 32.26,  5840700),
           ('JD', '7-Apr-17', 32.2 , 32.01, 31.57, 32.25,  5651000),
           ('JD', '10-Apr-17', 32.16, 32.67, 32.15, 32.92,  8303800),
           ('JD', '11-Apr-17', 32.7 , 32.3 , 32.22, 33.28,  8054200),
           ('JD', '12-Apr-17', 32.31, 32.71, 32.31, 32.88,  6818000),
           ('JD', '13-Apr-17', 32.74, 32.47, 32.45, 32.87,  3013600)],
          dtype=[('name', '<U4'), ('time', '<U10'), ('opening_price', '<f8'), ('closing_price', '<f8'), ('lowest_price', '<f8'), ('highest_price', '<f8'), ('volume', '<i4')])
