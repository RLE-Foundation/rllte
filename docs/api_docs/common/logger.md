#


## Logger
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L26)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L56)
```python
.parse_train_msg(
   msg: Any
)
```


### .parse_test_msg
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L63)
```python
.parse_test_msg(
   msg: Any
)
```


### .time_stamp
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L71)
```python
.time_stamp()
```

---
Return the current time stamp.

### .info
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L75)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L87)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L99)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L111)
```python
.train(
   msg: Dict
)
```

---
Output msg with 'train' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/logger.py/#L126)
```python
.test(
   msg: Dict
)
```

---
Output msg with 'test' level.


**Args**

* **msg** (str) : Message to be printed.


**Returns**

None.
