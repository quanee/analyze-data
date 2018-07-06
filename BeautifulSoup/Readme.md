

```python
from bs4 import BeautifulSoup
```


```python
from bs4 import element
```


```python
import re
```


```python
html = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title" name="dromouse"><b><a>The Dormouse's story</a></b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
```


```python
soup = BeautifulSoup(html, 'lxml')
print(soup.prettify())
```

    <html>
     <head>
      <title>
       The Dormouse's story
      </title>
     </head>
     <body>
      <p class="title" name="dromouse">
       <b>
        <a>
         The Dormouse's story
        </a>
       </b>
      </p>
      <p class="story">
       Once upon a time there were three little sisters; and their names were
       <a class="sister" href="http://example.com/elsie" id="link1">
        <!-- Elsie -->
       </a>
       ,
       <a class="sister" href="http://example.com/lacie" id="link2">
        Lacie
       </a>
       and
       <a class="sister" href="http://example.com/tillie" id="link3">
        Tillie
       </a>
       ;
    and they lived at the bottom of a well.
      </p>
      <p class="story">
       ...
      </p>
     </body>
    </html>
    


```python
# tag
print(soup.title)
```

    <title>The Dormouse's story</title>
    


```python
print(soup.head)
```

    <head><title>The Dormouse's story</title></head>
    


```python
print(soup.a)
```

    <a>The Dormouse's story</a>
    


```python
# name
print(soup.name)
```

    [document]
    


```python
print(soup.head.name)
```

    head
    


```python
# attrs
print(soup.p.attrs)
```

    {'class': ['title'], 'name': 'dromouse'}
    


```python
print(soup.p['class'])
```

    ['title']
    


```python
print(soup.p.get('class'))
```

    ['title']
    


```python
soup.p['class'] = 'newClass'
```


```python
print(soup.p)
```

    <p class="newClass" name="dromouse"><b><a>The Dormouse's story</a></b></p>
    


```python
del soup.p['class']
```


```python
print(soup.p)
```

    <p name="dromouse"><b><a>The Dormouse's story</a></b></p>
    


```python
# NavigableString  获取标签内部文字
```


```python
print(soup.p.string)  # 不能包含多个子结点
```

    The Dormouse's story
    


```python
print(type(soup.p.string))
```

    <class 'bs4.element.NavigableString'>
    


```python
# BeautifulSoup
```


```python
print(type(soup.name))
```

    <class 'str'>
    


```python
print(soup.name)
```

    [document]
    


```python
print(soup.attrs)
```

    {}
    


```python
# Comment
```


```python
print(soup.a)
```

    <a>The Dormouse's story</a>
    


```python
print(soup.a.string)
```

    The Dormouse's story
    


```python
if type(soup.a.string)==element.Comment:
    print(soup.a.string)
```


```python
# 遍历文档书
```


```python
# 直接子节点 .contents .children
```


```python
print(soup.head.contents)  # 返回列表
```

    [<title>The Dormouse's story</title>]
    


```python
print(soup.head.contents[0])
```

    <title>The Dormouse's story</title>
    


```python
print(soup.head.children)  # list生成器
```

    <list_iterator object at 0x000001F039001080>
    


```python
for child in soup.body.children:
    print(child)
```

    
    
    <p name="dromouse"><b><a>The Dormouse's story</a></b></p>
    
    
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    
    
    <p class="story">...</p>
    
    
    


```python
# 所有子孙节点  .descendants
```


```python
for child in soup.descendants:
    print(child)
```

    <html><head><title>The Dormouse's story</title></head>
    <body>
    <p name="dromouse"><b><a>The Dormouse's story</a></b></p>
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    <p class="story">...</p>
    </body></html>
    <head><title>The Dormouse's story</title></head>
    <title>The Dormouse's story</title>
    The Dormouse's story
    
    
    <body>
    <p name="dromouse"><b><a>The Dormouse's story</a></b></p>
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    <p class="story">...</p>
    </body>
    
    
    <p name="dromouse"><b><a>The Dormouse's story</a></b></p>
    <b><a>The Dormouse's story</a></b>
    <a>The Dormouse's story</a>
    The Dormouse's story
    
    
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    Once upon a time there were three little sisters; and their names were
    
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>
     Elsie 
    ,
    
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
    Lacie
     and
    
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
    Tillie
    ;
    and they lived at the bottom of a well.
    
    
    <p class="story">...</p>
    ...
    
    
    


```python
# 节点内容 string
```


```python
print(soup.head.string)
```

    The Dormouse's story
    


```python
print(soup.title.string)
```

    The Dormouse's story
    


```python
print(soup.html.string)  # 不能包含多个子结点
```

    None
    


```python
# 多个内容  strings stripped_strings
```


```python
for string in soup.strings:
    print(repr(string))
```

    "The Dormouse's story"
    '\n'
    '\n'
    "The Dormouse's story"
    '\n'
    'Once upon a time there were three little sisters; and their names were\n'
    ',\n'
    'Lacie'
    ' and\n'
    'Tillie'
    ';\nand they lived at the bottom of a well.'
    '\n'
    '...'
    '\n'
    


```python
for string in soup.stripped_strings:
    print(repr(string))
```

    "The Dormouse's story"
    "The Dormouse's story"
    'Once upon a time there were three little sisters; and their names were'
    ','
    'Lacie'
    'and'
    'Tillie'
    ';\nand they lived at the bottom of a well.'
    '...'
    


```python
# 父节点  parent
```


```python
p = soup.p
```


```python
print(p.parent.name)
```

    body
    


```python
content = soup.head.title.string
```


```python
print(content.parent.name)
```

    title
    


```python
# 全部父节点 parents
```


```python
content = soup.head.title.string
```


```python
for parent in content.parents:
    print(parent.name)
```

    title
    head
    html
    [document]
    


```python
# 兄弟节点 next_sibling previous_sibling
```


```python
print(soup.p.next_sibling)
```

    
    
    


```python
print(soup.p.prev_sibling)
```

    None
    


```python
print(soup.p.next_sibling.next_sibling)
```

    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    


```python
# 全部兄弟节点 next_siblings previous_siblings
```


```python
for sibling in soup.a.next_siblings:
    print(repr(sibling))
```


```python
# 前后节点 next_element previous_element
```


```python
print(soup.head.next_element)
```

    <title>The Dormouse's story</title>
    


```python
# 所有前后节点 next_elements previous_elements
```


```python
for element in soup.a.next_elements:
    print(repr(element))
```

    "The Dormouse's story"
    '\n'
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    'Once upon a time there were three little sisters; and their names were\n'
    <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>
    ' Elsie '
    ',\n'
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
    'Lacie'
    ' and\n'
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
    'Tillie'
    ';\nand they lived at the bottom of a well.'
    '\n'
    <p class="story">...</p>
    '...'
    '\n'
    


```python
# 搜索文档树 find_all( name, attrs, recursive, text, **kwargs)
```


```python
# name 属性
```


```python
# 传入字符串
soup.find_all('b')
```




    [<b><a>The Dormouse's story</a></b>]




```python
# 传入正则表达式
for tag in soup.find_all(re.compile('^b')):
    print(tag.name)
```

    body
    b
    


```python
# 传入列表
soup.find_all(['a', 'b'])
```




    [<b><a>The Dormouse's story</a></b>,
     <a>The Dormouse's story</a>,
     <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]




```python
# 传True
for tag in soup.find_all(True):
    print(tag.name)
```

    html
    head
    title
    body
    p
    b
    a
    p
    a
    a
    a
    p
    


```python
# 传方法
def has_class_but_no_id(tag):
    return tag.has_attr('class') and not tag.has_attr('id')
soup.find_all(has_class_but_no_id)
```




    [<p class="story">Once upon a time there were three little sisters; and their names were
     <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
     and they lived at the bottom of a well.</p>, <p class="story">...</p>]




```python
# keyword参数
soup.find_all(id='link2')
```




    [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]




```python
soup.find_all(href=re.compile('elsie'))
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
soup.find_all(href=re.compile('elsie'), id='link1')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
soup.find_all('a', class_='sister')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]




```python
data_soup = BeautifulSoup('<div data-foo="value">foo!</div>', 'lxml')
data_soup.find_all(attrs={"data-foo": "value"})
```




    [<div data-foo="value">foo!</div>]




```python
# text 参数
```


```python
soup.find_all(text='Elsie')
```




    []




```python
soup.find_all(text=["Tillie", "Elsie", "Lacie"])
```




    ['Lacie', 'Tillie']




```python
soup.find_all(text=re.compile("Dormouse"))
```




    ["The Dormouse's story", "The Dormouse's story"]




```python
# limit 参数  限制返回数量
```


```python
soup.find_all("a", limit=2)
```




    [<a>The Dormouse's story</a>,
     <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
# recursive 参数  是否递归检索
```


```python
soup.html.find_all("title")
```




    [<title>The Dormouse's story</title>]




```python
soup.html.find_all("title", recursive=False)
```




    []




```python
# css选择器
```


```python
# 通过标签名查找
soup.select('title') 
```




    [<title>The Dormouse's story</title>]




```python
soup.select('a')
```




    [<a>The Dormouse's story</a>,
     <a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]




```python
soup.select('b')
```




    [<b><a>The Dormouse's story</a></b>]




```python
# 通过类名查找
soup.select('.sister')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]




```python
# 通过 id 名查找
soup.select('#link1')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
# 组合查找
soup.select('p #link1')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
soup.select("head > title")
```




    [<title>The Dormouse's story</title>]




```python
# 属性查找
soup.select('a[class="sister"]')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]




```python
soup.select('a[href="http://example.com/elsie"]')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
soup.select('p a[href="http://example.com/elsie"]')
```




    [<a class="sister" href="http://example.com/elsie" id="link1"><!-- Elsie --></a>]




```python
soup = BeautifulSoup(html, 'lxml')
```


```python
type(soup.select('title'))
```




    list




```python
soup.select('title')[0].get_text()
```




    "The Dormouse's story"




```python
for title in soup.select('title'):
    print(title.get_text())
```

    The Dormouse's story
    
