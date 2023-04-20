#


## Logger
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L27)
```python 
Logger(
   log_dir: Path
)
```


---
The logger class.


**Args**

* **log_dir**  : The logging location.


**Returns**

Logger instance.


**Methods:**


### .parse_train_msg
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L57)
```python
.parse_train_msg(
   msg: Any
)
```


### .parse_test_msg
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L64)
```python
.parse_test_msg(
   msg: Any
)
```


### .time_stamp
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L72)
```python
.time_stamp()
```

---
Return the current time stamp.

### .info
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L80)
```python
.info(
   msg: str
)
```

---
Output msg with 'info' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.

### .debug
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L96)
```python
.debug(
   msg: str
)
```

---
Output msg with 'debug' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.

### .error
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L112)
```python
.error(
   msg: str
)
```

---
Output msg with 'error' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.

### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L128)
```python
.train(
   msg: str
)
```

---
Output msg with 'train' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/logger.py\#L147)
```python
.test(
   msg: str
)
```

---
Output msg with 'test' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.
