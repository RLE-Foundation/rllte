#


## Logger
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L59)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L99)
```python
.parse_train_msg(
   msg: Dict
)
```

---
Parse the training message.


**Args**

* **msg** (Dict) : The training message.


**Returns**

The formatted string.

### .parse_eval_msg
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L114)
```python
.parse_eval_msg(
   msg: Dict
)
```

---
Parse the evaluation message.


**Args**

* **msg** (Dict) : The evaluation message.


**Returns**

The formatted string.

### .time_stamp
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L130)
```python
.time_stamp()
```

---
Return the current time stamp.

### .info
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L134)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L145)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L156)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L167)
```python
.train(
   msg: Dict
)
```

---
Output msg with 'train' level.


**Args**

* **msg** (Dict) : Message to be printed.


**Returns**

None.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/logger.py/#L181)
```python
.eval(
   msg: Dict
)
```

---
Output msg with 'eval' level.


**Args**

* **msg** (Dict) : Message to be printed.


**Returns**

None.
