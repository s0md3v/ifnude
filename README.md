# ifnude

It is a nudity detection library that actually works, on both on humans and drawings.. It tells you exactly what NSFW parts of the body are visible. Optionally, you can censor the said parts.

<img src="https://i.imgur.com/0KPJbl9.jpg" width=600>

### Installation
```bash
pip install ifnude
```

### Example
```python
from ifnude import detect

# use mode="fast" for x3 speed with slightly lower accuracy
print(detect('/path/to/nsfw.png'))
```

#### Output
```
[{'box': [164, 188, 246, 271], 'score': 0.8253238201141357, 'label': 'EXPOSED_BREAST_F'}, {'box': [252, 190, 335, 270], 'score': 0.8235630989074707, 'label': 'EXPOSED_BREAST_F'}]
```

### Credits
This is fork of [NudeNet](https://pypi.org/project/NudeNet/) library which doesn't work anymore. I have taken the liberty to remove the video detection functionality as it was prone to crashes. It will be re-implemented in future.
