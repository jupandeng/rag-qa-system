# Python 性能优化技巧

## 字符串拼接的正确姿势

在 Python 中，字符串是不可变对象。每次使用 `+=` 拼接字符串时，Python 会创建一个全新的字符串对象并复制原有内容。如果在一个循环中反复拼接，时间复杂度会达到 O(n²)。

正确的做法是使用列表收集所有片段，最后用 `join` 一次性拼接：

```python
parts = []
for item in items:
    parts.append(str(item))
result = "\n".join(parts)
```

## 列表推导式 vs 普通循环

列表推导式不仅代码更简洁，而且执行效率通常比等价的 for 循环快 10-20%。原因是列表推导式在 C 层面执行，避免了 Python 层面的循环开销和属性查找。

```python
# 慢
result = []
for x in range(1000):
    result.append(x * 2)

# 快
result = [x * 2 for x in range(1000)]
```

## 用 dataclass 减少样板代码

Python 3.7 引入的 `dataclass` 装饰器可以自动生成 `__init__`、`__repr__`、`__eq__` 等方法，大幅减少样板代码。

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
```
