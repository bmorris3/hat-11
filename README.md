# hat-11

To view the HAT-P-11 light curve prepped for STSP, clone this repository, `cd` into the `data/` directory and run the following in a Python shell: 

```python
from datacleaner import LightCurve
hat11_for_stsp = LightCurve.from_dir('hat11_for_stsp')
hat11_for_stsp.plot(show=True)
```
