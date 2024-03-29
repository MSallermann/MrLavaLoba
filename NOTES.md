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

# Algorithm

## Pre-processing
`n_vents` is fixed and gives the number of vents.

Then we compute the cumulative fissure length, which is defined as is the cumulative distance between vents.
    [0, |x_1-x_0|, |x_2-x_1| + |x_3 - x_2| ...]
This is only needed when certain `vent_flags` are set.

Then the `run_file` is set up, which is some naive form of parallelism. If runs 0 to 10 are present, the current run file will be something like `<self.run_name>11.bak` etc. 

Then the lobe data is allocated:
First a variable `alloc_n_lobes` is determined, either from `input.max_n_lobes` or drawn from a beta distribution.

```python
# initialize the arrays for the lobes variables
self.angle = np.zeros(self.alloc_n_lobes)  # @NOTE: azimuthal angle of lobes
self.x = np.zeros(self.alloc_n_lobes)  # @NOTE: x position of lobe centers
self.y = np.zeros(self.alloc_n_lobes)  # @NOTE: y position of lobe centers
self.x1 = np.zeros(self.alloc_n_lobes) # @NOTE: semi major axis
self.x2 = np.zeros(self.alloc_n_lobes) # @NOTE: semi minor axis
self.dist_int = np.zeros(self.alloc_n_lobes, dtype=int) - 1 # @NOTE: distance from initial lobe in number of lobes
self.parent = np.zeros(self.alloc_n_lobes, dtype=int) # @NOTE: index of parent lobe
self.alfa_inertial = np.zeros(self.alloc_n_lobes) #????
self.h = np.zeros(self.alloc_n_lobes) #@NOTE: height of lobe??? seems to be unused
self.descendents = np.zeros(self.alloc_n_lobes, dtype=int) # @NOTE: number of descendant lobes (cumulative?)
```

Then the `.asc` file is read, which contains the topology
Most of the things here are redundant, but this is the data it contains:

```python
class AscFile:
    cols: int = 0 #@NOTE: number of cols
    rows: int = 0 #@NOTE: number of rows
    lx: float = 0 #@NOTE: coordinate of lower left corner
    ly: float = 0 #@NOTE: coordinate of lower left corner
    cell: float = 0 #@NOTE: side length of square cell
    nd: float = -9999 #@NOTE: number that indicates lack of data
    arr = np.array([], dtype=float) #@NOTE: array that contains height data
    crop_flag: bool = False
    nx: int = 0 # @NOTE: number of entries in x direction
    ny: int = 0 # @NOTE: number of entries in y direction
    Xc = np.array([], dtype=float) #@NOTE: meshgrid of X data
    Yc = np.array([], dtype=float) #@NOTE: meshgrid of Y data
    Zc = np.array([], dtype=float) #@NOTE: array that also contains height data lolol
    iW: int = 0 #@NOTE: indices used for the crop slice
    iE: int = 0 #@NOTE: indices used for the crop slice
    jS: int = 0 #@NOTE: indices used for the crop slice
    jN: int = 0 #@NOTE: indices used for the crop slice
    xcmin: float = 0 #@NOTE: np.min(x)
    xcmax: float = 0 #@NOTE: np.max(x)
    ycmin: float = 0 #@NOTE: np.min(y)
    ycmax: float = 0 #@NOTE: np.max(y)
```

Now we compute the lobe dimensions.

The dimensions to be computed are
    - avg_lobe_thickness
    - lobe_are
    - max_semiaxis
    - max_cells
    - thickness_min

If the volume_flag is set therea are two possibilites:
- if fixed_dimension_flag = 1  => the area of the lobes is assigned
```python
self.avg_lobe_thickness = input.total_volume / (
                    input.n_flows
                    * input.lobe_area
                    * 0.5
                    * (input.min_n_lobes + input.max_n_lobes)
                )
```

- if fixed_dimension_flag = 2  => the thickness of the lobes is assigend
```python
input.lobe_area = input.total_volume / (
                    input.n_flows
                    * self.avg_lobe_thickness
                    * 0.5
                    * (input.min_n_lobes + input.max_n_lobes)
                )
```

If the volume_flag is not set the avg_lobe_thickness and the lobe area are additional input parameters. We are not sure if this was ever properly implemented.

Further we compute the `max_semiaxis`, which is a result of restricting the aspect ratio, while keeping the area fixed.

```python
self.max_semiaxis = np.sqrt(input.lobe_area * input.max_aspect_ratio / np.pi)
```

Then there is this
```python
self.max_cells = np.ceil(2.0 * self.max_semiaxis / self.asc_file.cell) + 2
```
max_cells seems to be the maximum number of cells straddled by each lobe? It seems to be used in the stopping criterion and some other random place.

```python
self.thickness_min = (
    2.0
    * input.thickness_ratio
    / (input.thickness_ratio + 1.0)
    * self.avg_lobe_thickness
)
```
About thickness ratio the input file has to say this

```python
Ratio between the thickness of the first lobe of the flow and the thickness of the last lobe.
# thickness_ratio < 1   => the thickness increases with lobe "age"
# thickness_ratio = 1   => all the lobes have the same thickness
# thickness_ratio > 1   => the thickness decreases with lobe "age"
```

We dont understand this. The minimum thickness should either be the thickness of the first lobe or the last lobe?
Figure this out later.

We dont quite understand how the thickening parameter works
```python
    self.filling_parameter = (1.0 - input.thickening_parameter) * np.ones_like(
                self.asc_file.Zc
            )
```

Then the code checks the "channel_file", whatever that is.

Then we deal with the restart files.

Then we define some cryptic variables:
```python
Ztot = np.array(asc_file.Zc) # Ztot is the same as asc_file.Zc
Zflow = np.zeros((asc_file.ny, asc_file.nx))

jtop_array = np.zeros(self.alloc_n_lobes, dtype=int)
jbottom_array = np.zeros(self.alloc_n_lobes, dtype=int)

iright_array = np.zeros(self.alloc_n_lobes, dtype=int)
ileft_array = np.zeros(self.alloc_n_lobes, dtype=int)

Zhazard = np.zeros((asc_file.ny, asc_file.nx), dtype=int)
Zhazard_temp = np.zeros((asc_file.ny, asc_file.nx), dtype=int)

Zdist = np.zeros((asc_file.ny, asc_file.nx), dtype=int) + 9999
```

Notes:
- At this point it is not clear why we copy asc_file.Zc into Ztot.
- Zhazard_temp seems to never be used
- Zdist is initialized to 9999 for some reason???
- WTF are jtop_array, jbottom_array, iright_array, ileft_array?!

## Processing
There is a big outer loop over the flows
```python
    for flow in range(0, input.n_flows):
        Zflow_local_array = np.zeros(
            (self.alloc_n_lobes, self.max_cells, self.max_cells), dtype=int
        )
        descendents = np.zeros(self.alloc_n_lobes, dtype=int)
        # ...
```
Note that `descendents` seem to be counted per flow.

For each flow, the number of lobes in this flow `n_lobes` is calculated, this is *not* the same as `alloc_n_lobes`, however.
`n_lobes` is either drawn from a uniform distribution in the interval `[input.min_n_lobes, input.max_n_lobes]` or from beta distribution, which is also dependant on `min_n_lobes` and `max_n_lobes`.

This is just bizarre, why do this twice. What if `n_lobes > alloc_n_lobes`?! So many questions. What even is the point of this?

The total number of lobes are tallied up
```python
            n_lobes_tot += n_lobes
```
so that `n_lobes_tot` contains the number of lobes summed over all flows.

Then, we compute `delta_lobe_thickness`
```python
    delta_lobe_thickness = (
        2.0 * (self.avg_lobe_thickness - self.thickness_min) / (n_lobes - 1.0)
    )
```
This doesnt make a lot of sense to us at the moment, it seems to be the difference in thickness between subsequent lobes.

Now we produce `input.n_init` initial lobes, these are treated differently from the rest of the lobes. And their position depends on the vent flags etc.

For each initial lobe
1. The position is computed, dependent on the vent parameters
2. The slope of the terrain at the lobe center is computed as well as the angle of the direction of the maximum slope. With respect to some arbitrary origin ...
3. The lobe angle is computed by perturbin the initial angle according to some truncated gaussian.
4. The lobe major and minor axes are computed, using the calculated slope and `input.max_aspect_ratio` and `input.aspect_ratio_coeff`.
The aspect ratio linearly increases with the slope with a slope of `aspect_ratio_coeff` and is cutoff at a maximum of `max_aspect ratio`
5. The lobe is rasterized.

Then we move on to the rest of the lobes. These lobes bud from parents and not directly from the vent(s).

For each subsequet lobe:
1. Compute the index of the parent lobe. Generally speaking this is done in a probabilistic way. First, a random index is drawn in one of two ways. (1) If `input.force_max_length == True`, the parent lobe is chosen only among those with `dist_int` smaller than `input.max_length` with uniform probability. (2) Else, the parent lobe is chosen amongst all lobes up to the preceeding lobe. 
    If, however, `input.lobe_exponent <= 0` then the parent lobe is the preceeding lobe. I.e the flow is just a single "line".
2. Compute the preliminary budding point of the new lobe. Use equation from the paper. Summarily, we need to find the point of lowest elevation on the circumference of the parent lobe.
3. Apply the angle perturbation, which is the same as for the initial lobes.
4. Add the inertial contribution. See code below.
5. Get the center of the new lobe.
    First compute the budding point by finding the intersection of the direction of the new lobe with the boundary of the parent lobe.
    Then we find the slope between the budding point and the center of the parent via linear interpolation yadda yadda. Then we, again, apply the aspect ration (like we did for initial lobes). This gives us the semi-axes of the new lobe.
    Lastly, we compute the new center by displacing the ellipse along the direction we found before. There is an input parameter `input.dist_fact` that controls how much the new ellipse overlaps with its parent. `dist_fact = 1` means they touch in one point only.
5. Build the new lobe (aka store the data in our lovely arrays)
6. Rasterize the lobe
7. Calculate the thickness of the new lobe and update the topology.
    Our best guess is that the arrays mean the following
    - `Zflow_local` is the thickness added due to the current lobe
    - `Zflow` is the thickness added due to the current flow
    - `Ztot` is the total topology, i.e. the "base map" plus the thickness due to all flows
    - `Zdist` is the minimum lobe distance of the pixels from the vent
    - We dont know what `Zflow_local_int` and `Zflow_local_array` are
    Also the "boundary" box of the current lobe is saved.

8. The "hazard map" is updated, which to the best of our knowledge is the heat map of lava placement probability frequently plotted in the papers

9. The rest is just prints and output

### About the inertial contribution
```python
    # cos and sin of the angle of the parent lobe
    cos_angle1 = np.cos(self.angle[idx] * np.pi / 180.0)
    sin_angle1 = np.sin(self.angle[idx] * np.pi / 180.0)

    # cos and sin of the angle of maximum slope
    cos_angle2 = np.cos(new_angle * np.pi / 180.0)
    sin_angle2 = np.sin(new_angle * np.pi / 180.0)

    if input.inertial_exponent == 0:
        self.alfa_inertial[i] = 0.0

    else:
        self.alfa_inertial[i] = (
            1.0 - (2.0 * np.arctan(slope) / np.pi) ** input.inertial_exponent
        ) ** (1.0 / input.inertial_exponent)

    x_avg = (1.0 - self.alfa_inertial[i]) * cos_angle2 + self.alfa_inertial[
        i
    ] * cos_angle1
    y_avg = (1.0 - self.alfa_inertial[i]) * sin_angle2 + self.alfa_inertial[
        i
    ] * sin_angle1

    angle_avg = np.mod(180 * np.arctan2(y_avg, x_avg) / np.pi, 360)

    new_angle = angle_avg
    return new_angle
```
$$
\begin{aligned}
x_\text{avg} = (1 - \alpha) * \cos \phi + \alpha \cos \phi_\text{p}\\
y_\text{avg} = (1 - \alpha) * \sin \phi + \alpha \sin \phi_\text{p} 
\end{aligned}
$$
where 
$$
\alpha = \left[1 - \left(\frac{2\arctan(s)}{\pi}\right)^\eta\right]^{\frac{1}{\eta}}
$$
where $$\eta = \text{input.inertial\_exponent}$$
where $$s = \text{slope}$$

This is some wonky-ass function that ensures $\alpha \in [0,1)$. It doesnt seem to be defined for negative slopes however.