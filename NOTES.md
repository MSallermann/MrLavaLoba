# Notes

The `mr_lava_loba.py` script does the following things


## Input parsing
First the input data is parsed from two python scripts `input_data.py` and `input_data.py` like so
```python
from input_data import run_name
from input_data import source
from input_data import x_vent
# ...
from input_data_advanced import npoints
from input_data_advanced import n_init
from input_data_advanced import dist_fact
# ...
```

There seem to be two mandatory inputs which throw an exception if not used

```python
try:
    from input_data import x_vent_end
except ImportError:
    print("x_vent_end not used")

try:
    from input_data import y_vent_end
except ImportError:
    print("y_vent_end not used")

try:
    from input_data import fissure_probabilities
except ImportError:
    print("fissure_probabilities not used")
```