#


### get_init_fn
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/initialization.py/#L68)
```python
.get_init_fn(
   method: str = 'orthogonal'
)
```

---
Returns a network initialization function.


**Args**

* **method** (str) : Initialization method name.


**Returns**

Initialization function.

----


### _xavier_normal
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/initialization.py/#L57)
```python
._xavier_normal(
   m
)
```

---
Xavier normal initialization.

----


### _xavier_uniform
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/initialization.py/#L46)
```python
._xavier_uniform(
   m
)
```

---
Xavier uniform initialization.

----


### _orthogonal
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/initialization.py/#L34)
```python
._orthogonal(
   m
)
```

---
Orthogonal initialization.

----


### _identity
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/initialization.py/#L30)
```python
._identity(
   m
)
```

---
Identity initialization.
