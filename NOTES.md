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
### Mandatory Inputs 
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

The variable `alloc_n_lobes` is set, either by using some user-defined input (`max_n_lobes`), or
by drawing from a beta function

```python
if (a_beta == 0) and (b_beta == 0):
    alloc_n_lobes = int(max_n_lobes)

else:

    x_beta = np.rint(range(0, n_flows)) / (n_flows - 1)
    beta_pdf = beta.pdf(x_beta, a_beta, b_beta)s
    alloc_n_lobes = int(
        np.rint(min_n_lobes + 0.5 * (max_n_lobes - min_n_lobes) * np.max(beta_pdf))
    )
```
The code also needs `volume_flag`,  `channel_file`, `alfa_channel`, `d1`, `d2`, `eps`

## Initialization 
Now several arrays are initialized: 

```python
# initialize the arrays for the lobes variables
angle = np.zeros(alloc_n_lobes)
x = np.zeros(alloc_n_lobes)
y = np.zeros(alloc_n_lobes)
x1 = np.zeros(alloc_n_lobes)
x2 = np.zeros(alloc_n_lobes)
h = np.zeros(alloc_n_lobes)

dist_int = np.zeros(alloc_n_lobes, dtype=int) - 1
descendents = np.zeros(alloc_n_lobes, dtype=int)
parent = np.zeros(alloc_n_lobes, dtype=int)
alfa_inertial = np.zeros(alloc_n_lobes)

# This is later, and uses the shape of the topology data Zc
filling_parameter = (1.0 - thickening_parameter) * np.ones_like(Zc)
```

## Reading topology 

The .asc file is read in and optionally cropped (using header information from the .asc file).
This file contains height data (Z) in a rectangular grid. Then a meshgrid is made from X and Y limits, after which Z data is assigned. `arr` is the topology array. 

```python
xcmin = np.min(xc)
xcmax = np.max(xc)

ycmin = np.min(yc)
ycmax = np.max(yc)

Xc, Yc = np.meshgrid(xc, yc)

Zc = np.zeros((ny, nx))
np.copyto(Zc, arr)
```