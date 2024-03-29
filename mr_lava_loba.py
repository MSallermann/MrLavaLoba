import numpy as np
from linecache import getline
#from scipy.stats import beta
from matplotlib.patches import Ellipse
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
import time
import os
import numpy.ma as ma
import sys
import shutil
import datetime
import rtnorm
from random import randrange
from os.path import exists
import gc
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Input:
    # from input
    run_name: str = ""
    source: str = ""
    x_vent: List[float] = field(default_factory=list)
    y_vent: List[float] = field(default_factory=list)
    hazard_flag: int = 0
    masking_threshold: float = 0
    n_flows: int = 0
    min_n_lobes: int = 0
    max_n_lobes: int = 0
    thickening_parameter: float = 0
    lobe_area: float = 0
    inertial_exponent: float = 0
    lobe_exponent: float = 0
    max_slope_prob: float = 0
    thickness_ratio: float = 0
    fixed_dimension_flag: int = 0
    vent_flag: int = 0
    fissure_probabilities: Optional[float] = None
    total_volume: Optional[float] = None
    volume_flag: Optional[int] = None
    east_to_vent: Optional[float] = None
    west_to_vent: Optional[float] = None
    south_to_vent: Optional[float] = None
    north_to_vent: Optional[float] = None
    channel_file: Optional[str] = None
    alfa_channel: Optional[float] = None
    d1: Optional[float] = None
    d2: Optional[float] = None
    eps: Optional[float] = None
    union_diff_file: Optional[str] = None

    # from input_advanced
    npoints: int = 0
    n_init: int = 0
    dist_fact: float = 0
    flag_threshold: int = 0
    a_beta: float = 0
    b_beta: float = 0
    max_aspect_ratio: float = 0
    saveraster_flag: int = 0
    aspect_ratio_coeff: float = 0
    start_from_dist_flag: int = 0
    force_max_length: int = 0
    max_length: float = 0
    n_check_loop: int = 0
    restart_files: Optional[List[str]] = None
    restart_filling_parameters: Optional[List[float]] = None


def parse_input() -> Input:
    import input_data
    import input_data_advanced

    input = Input()
    input.run_name = input_data.run_name
    input.source = input_data.source
    input.x_vent = input_data.x_vent
    input.y_vent = input_data.y_vent
    input.hazard_flag = input_data.hazard_flag
    input.masking_threshold = input_data.masking_threshold
    input.n_flows = input_data.n_flows
    input.min_n_lobes = input_data.min_n_lobes
    input.max_n_lobes = input_data.max_n_lobes
    input.thickening_parameter = input_data.thickening_parameter
    input.lobe_area = input_data.lobe_area
    input.inertial_exponent = input_data.inertial_exponent
    input.lobe_exponent = input_data.lobe_exponent
    input.max_slope_prob = input_data.max_slope_prob
    input.thickness_ratio = input_data.thickness_ratio
    input.fixed_dimension_flag = input_data.fixed_dimension_flag
    input.vent_flag = input_data.vent_flag

    input.npoints = input_data_advanced.npoints
    # @NOTE: Number of repetitions of the first lobe (useful for initial spreading)
    input.n_init = input_data_advanced.n_init
    input.dist_fact = input_data_advanced.dist_fact
    input.flag_threshold = input_data_advanced.flag_threshold
    input.a_beta = input_data_advanced.a_beta
    input.b_beta = input_data_advanced.b_beta
    input.max_aspect_ratio = input_data_advanced.max_aspect_ratio
    input.saveraster_flag = input_data_advanced.saveraster_flag
    input.aspect_ratio_coeff = input_data_advanced.aspect_ratio_coeff
    input.start_from_dist_flag = input_data_advanced.start_from_dist_flag
    input.force_max_length = input_data_advanced.force_max_length
    input.max_length = input_data_advanced.max_length
    input.n_check_loop = input_data_advanced.n_check_loop

    try:
        input.fissure_probabilities = input_data.fissure_probabilities
    except AttributeError:
        input.fissure_probabilities = None

    try:
        input.volume_flag = input_data.volume_flag
    except AttributeError:
        input.volume_flag = None

    try:
        input.total_volume = input_data.total_volume
    except AttributeError:
        input.total_volume = None

    try:
        input.west_to_vent = input_data.west_to_vent
    except AttributeError:
        input.west_to_vent = None

    try:
        input.east_to_vent = input_data.east_to_vent
    except AttributeError:
        input.east_to_vent = None

    try:
        input.south_to_vent = input_data.south_to_vent
    except AttributeError:
        input.south_to_vent = None

    try:
        input.north_to_vent = input_data.north_to_vent
    except AttributeError:
        input.north_to_vent = None

    try:
        input.channel_file = input_data.channel_file
        input.alfa_channel = input_data.alfa_channel
        input.d1 = input_data.d1
        input.d2 = input_data.d2
        input.eps = input_data.eps
    except AttributeError:
        input.channel_file = None
        input.alfa_channel = None
        input.d1 = None
        input.d2 = None
        input.eps = None

    try:
        input.restart_files = input_data_advanced.restart_files
        input.restart_filling_parameters = (
            input_data_advanced.restart_filling_parameters
        )
    except AttributeError:
        input.restart_files = None
        input.restart_filling_parameters = None

    try:
        input.union_diff_file = input_data.union_diff_file
    except AttributeError:
        input.union_diff_file = None

    return input


@dataclass
class AscFile:
    cols: int = 0
    rows: int = 0
    lx: float = 0
    ly: float = 0
    cell: float = 0
    nd: float = -9999
    arr = np.array([], dtype=float)
    crop_flag: bool = False
    nx: int = 0
    ny: int = 0
    Xc = np.array([], dtype=float)
    Yc = np.array([], dtype=float)
    Zc = np.array([], dtype=float)
    iW: int = 0
    iE: int = 0
    jS: int = 0
    jN: int = 0
    xcmin: float = 0
    xcmax: float = 0
    ycmin: float = 0
    ycmax: float = 0


def read_asc_file(input: Input):
    # Parse the header using a loop and
    # the built-in linecache module
    asc_file = AscFile()
    hdr = [getline(input.source, i) for i in range(1, 7)]

    values = [float(h.split(" ")[-1].strip()) for h in hdr]
    del hdr

    cols, rows, asc_file.lx, asc_file.ly, asc_file.cell, asc_file.nd = values
    # del values

    asc_file.cols = int(cols)
    asc_file.rows = int(rows)

    crop_flag = (
        (input.west_to_vent is None)
        and (input.east_to_vent is None)
        and (input.south_to_vent is None)
        and (input.north_to_vent is None)
    )

    print("west_to_vent", input.west_to_vent)
    print("input.x_vent", input.x_vent)
    print("Crop flag = ", crop_flag)

    if sys.version_info >= (3, 0):
        start = time.process_time()
    else:
        start = time.clock()

    source_npy = input.source.replace(".asc", ".npy")

    if os.path.isfile(source_npy):
        print(source_npy, " exists")
    else:
        print(source_npy, " does not exist")
        data = np.loadtxt(input.source, skiprows=6)
        np.save(source_npy, data)
        del data

    if crop_flag:
        # Load the dem into a numpy array
        arr_temp = np.flipud(np.load(input.source_npy))

        # the values are associated to the center of the pixels
        xc_temp = lx + asc_file.cell * (0.5 + np.arange(0, arr_temp.shape[1]))
        yc_temp = ly + asc_file.cell * (0.5 + np.arange(0, arr_temp.shape[0]))

        xW = np.min(input.x_vent) - input.west_to_vent
        xE = np.max(input.x_vent) + input.east_to_vent
        yS = np.min(input.y_vent) - input.south_to_vent
        yN = np.max(input.y_vent) + input.north_to_vent

        # crop the DEM to the desired domain
        asc_file.iW = np.maximum(0, (np.floor((xW - lx) / asc_file.cell)).astype(int))
        asc_file.iE = np.minimum(cols, (np.ceil((xE - lx) / asc_file.cell)).astype(int))
        asc_file.jS = np.maximum(0, (np.floor((yS - ly) / asc_file.cell)).astype(int))
        asc_file.jN = np.minimum(rows, (np.ceil((yN - ly) / asc_file.cell)).astype(int))

        print("Cropping of original DEM")
        print("xW,xE,yS,yN", xW, xE, yS, yN)
        print("iW,iE,jS,jN", asc_file.iW, asc_file.iE, asc_file.jS, asc_file.jN)
        print("")

        arr = arr_temp[asc_file.jS : asc_file.jN, asc_file.iW : asc_file.iE]
        xc = xc_temp[asc_file.iW : asc_file.iE]
        yc = yc_temp[asc_file.jS : asc_file.jN]

        lx = xc[0] - 0.5 * asc_file.cell
        ly = yc[0] - 0.5 * asc_file.cell

        asc_file.nx = arr.shape[1]
        asc_file.ny = arr.shape[0]

        header = "ncols     %s\n" % arr.shape[1]
        header += "nrows    %s\n" % arr.shape[0]
        header += "xllcorner " + str(lx) + "\n"
        header += "yllcorner " + str(ly) + "\n"
        header += "asc_file.cellsize " + str(asc_file.cell) + "\n"
        header += "NODATA_value " + str(asc_file.nd) + "\n"

        output_DEM = input.run_name + "_DEM.asc"

        np.savetxt(output_DEM, np.flipud(arr), header=header, fmt="%1.5f", comments="")

        del arr_temp
        del xc_temp
        del yc_temp
        gc.collect()

    else:
        # Load the dem into a numpy array
        arr = np.flipud(np.load(source_npy))

        asc_file.nx = arr.shape[1]
        asc_file.ny = arr.shape[0]

        # the values are associated to the center of the pixels
        xc = asc_file.lx + asc_file.cell * (0.5 + np.arange(0, asc_file.nx))
        yc = asc_file.ly + asc_file.cell * (0.5 + np.arange(0, asc_file.ny))

    gc.collect()

    if sys.version_info >= (3, 0):
        elapsed = time.process_time() - start
    else:
        elapsed = time.clock() - start

    print("Time to read DEM " + str(elapsed) + "s")

    asc_file.xcmin = np.min(xc)
    asc_file.xcmax = np.max(xc)

    asc_file.ycmin = np.min(yc)
    asc_file.ycmax = np.max(yc)

    asc_file.Xc, asc_file.Yc = np.meshgrid(xc, yc)

    asc_file.Zc = np.zeros((asc_file.ny, asc_file.nx))
    np.copyto(asc_file.Zc, arr)

    return asc_file


def interp2Dgrids(xin, yin, Zin, Xout, Yout):
    """
    Interpolation from a regular grid to a second regular grid

    @params:
        xin      - Required : original grid X values (1D Dble)
        yin      - Required : original grid Y values (1D Dble)
        Zin      - Required : original grid Z values (2D Dble)
        xout     - Required : new grid X values (2D Dble)
        yout     - Required : new grid Y values (2D Dble)
    """
    xinMin = np.min(xin)

    yinMin = np.min(yin)

    cellin = xin[1] - xin[0]

    if Xout.ndim == 2:
        xout = Xout[0, :]

    else:
        xout = Xout

    if Yout.ndim == 2:
        yout = Yout[:, 0]

    else:
        yout = Yout

    # Search for the cell containing the center of the parent lobe
    xi = (xout - xinMin) / cellin
    yi = (yout - yinMin) / cellin

    # Indexes of the lower-left corner of the cell containing the center of
    # the parent lobe
    ix = np.maximum(0, np.minimum(xin.shape[0] - 2, np.floor(xi).astype(int)))
    iy = np.maximum(0, np.minimum(yin.shape[0] - 2, np.floor(yi).astype(int)))

    # Indexes of the top-right corner of the cell containing the center of
    # the parent lobe
    ix1 = ix + 1
    iy1 = iy + 1

    # Relative coordinates of the center of the parent lobe in the cell

    xi_fract = np.maximum(0.0, np.minimum(1.0, (xi - ix).reshape(1, Xout.shape[1])))

    yi_fract = np.maximum(0.0, np.minimum(1.0, (yi - iy).reshape(Yout.shape[0], 1)))

    cos_angle = np.cos(angle*np.pi/180);
    sin_angle = np.sin(angle*np.pi/180);

    Zout = (
        xi_out_yi * Zin[np.ix_(iy1, ix1)]
        + (xi_fract - xi_out_yi) * Zin[np.ix_(iy, ix1)]
        + (yi_fract - xi_out_yi) * Zin[np.ix_(iy1, ix)]
        + (1.0 - xi_fract - yi_fract + xi_out_yi) * Zin[np.ix_(iy, ix)]
    )

    return Zout


class MrLavaLoba:
    def __init__(self, input: Input) -> None:
        self.input = input
        self.cum_fiss_length = np.array([], dtype=float)
        self.n_vents: int = 0
        self.angle = np.array([], dtype=float)  # @NOTE: azimuthal angle of ellipse
        self.x = np.array([], dtype=float)  # @NOTE: coordinates of ellipse centers
        self.y = np.array([], dtype=float)  # @NOTE: coordinates of ellipse centers
        self.x1 = np.array([], dtype=float)  # @NOTE: major semi-axis of ellipse
        self.x2 = np.array([], dtype=float)  # @NOTE: minor semi-axis of ellipse
        self.dist_int = np.array(
            [], dtype=int
        )  # @NOTE: the length of the shortest path from the initial lobe to the current lobe in number of ellipses
        self.parent = np.array([], dtype=int)
        self.alfa_inertial = np.array([], dtype=float)
        self.avg_lobe_thickness: float = 0

        self.X_circle = np.array([], dtype=float)
        self.Y_circle = np.array([], dtype=float)
        self.filling_parameter = np.array([], dtype=float)

        self.asc_file: AscFile = AscFile()
        self.vx = np.array([], dtype=float)
        self.vy = np.array([], dtype=float)
        self.distxy = np.array([], dtype=float)

    def ellipse(self, xc, yc, ax1, ax2, angle):
        cos_angle = np.cos(angle * np.pi / 180)
        sin_angle = np.sin(angle * np.pi / 180)

        # x1 = xc + ax1 * cos_angle
        # y1 = yc + ax1 * sin_angle

        # x2 = xc - ax2 * sin_angle
        # y2 = yc + ax2 * cos_angle

        X = ax1 * self.X_circle
        Y = ax2 * self.Y_circle

        xe = xc + X * cos_angle - Y * sin_angle
        ye = yc + X * sin_angle + Y * cos_angle

        return (xe, ye)

    def local_intersection(self, Xc_local, Yc_local, xc_e, yc_e, ax1, ax2, angle):
        # the accuracy of this procedure depends on the resolution of xv and yv
        # representing a grid of points [-0.5*cell;0.5*cell] X [-0.5*cell;0.5*cell]
        # built around the centers

        nx_cell = Xc_local.shape[0]
        ny_cell = Xc_local.shape[1]

        c = np.cos(angle * np.pi / 180)
        s = np.sin(angle * np.pi / 180)

        c1 = c / ax1
        s1 = s / ax1

        c2 = c / ax2
        s2 = s / ax2

        xv_local = self.xv - xc_e
        yv_local = self.yv - yc_e

        Xc_local_1d = Xc_local.ravel()
        Yc_local_1d = Yc_local.ravel()

        c1xv_p_s1yv = c1 * xv_local + s1 * yv_local
        c2yv_m_s2yv = c2 * yv_local - s2 * xv_local

        term1 = (c1**2 + s2**2) * Xc_local_1d**2
        term2 = (2 * c1 * s1 - 2 * c2 * s2) * Xc_local_1d * Yc_local_1d
        term3 = np.tensordot(
            Xc_local_1d, 2 * c1 * c1xv_p_s1yv - 2 * s2 * c2yv_m_s2yv, 0
        )
        term4 = (c2**2 + s1**2) * Yc_local_1d**2
        term5 = np.tensordot(
            Yc_local_1d, 2 * c2 * c2yv_m_s2yv + 2 * s1 * c1xv_p_s1yv, 0
        )
        term6 = c1xv_p_s1yv**2 + c2yv_m_s2yv**2

        term124 = term1 + term2 + term4
        term356 = term3 + term5 + term6

        term_tot = term124 + term356.transpose()

        inside = term_tot <= 1

        area_fract_1d = np.sum(inside.astype(float), axis=0)

        area_fract_1d /= self.nv2

        area_fract = area_fract_1d.reshape(nx_cell, ny_cell)

        return area_fract

    def compute_cumulative_fissure_length(self):
        self.cum_fiss_length = np.zeros(self.n_vents)
        first_j = 1
        for j in range(first_j, self.n_vents):
            delta_xvent = self.input.x_vent[j] - self.input.x_vent[j - 1]
            delta_yvent = self.input.y_vent[j] - self.input.y_vent[j - 1]
            self.cum_fiss_length[j] = self.cum_fiss_length[j - 1] + np.sqrt(
                delta_xvent**2 + delta_yvent**2
            )

        if self.n_vents > 1:
            self.cum_fiss_length = (
                self.cum_fiss_length.astype(float) / self.cum_fiss_length[-1]
            )

    def setup_run_file(self):
        # search if another run with the same base name already exists
        i = 0
        condition = True
        base_name = self.input.run_name

        while condition:
            self.input.run_name = base_name + "_{0:03}".format(i)
            backup_advanced_file = self.input.run_name + "_advanced_inp.bak"
            backup_file = self.input.run_name + "_inp.bak"
            condition = os.path.isfile(backup_file)
            i = i + 1

        # create a backup file of the input parameters
        shutil.copy2("input_data_advanced.py", backup_advanced_file)
        shutil.copy2("input_data.py", backup_file)

        print("Run name", self.input.run_name)

    def allocate_lobe_data(self):
        input = self.input

        if (input.a_beta == 0) and (input.b_beta == 0):
            self.alloc_n_lobes = int(input.max_n_lobes)
        else:
            x_beta = np.rint(range(0, input.n_flows)) / (input.n_flows - 1)
            beta_pdf = beta.pdf(x_beta, input.a_beta, input.b_beta)
            self.alloc_n_lobes = int(
                np.rint(
                    input.min_n_lobes
                    + 0.5 * (input.max_n_lobes - input.min_n_lobes) * np.max(beta_pdf)
                )
            )
            print("Flow with the maximum number of lobes", np.argmax(beta_pdf))
        print("Maximum number of lobes", self.alloc_n_lobes)

        # initialize the arrays for the lobes variables
        self.angle = np.zeros(self.alloc_n_lobes)  # @NOTE: azimuthal angle of lobes
        self.x = np.zeros(self.alloc_n_lobes)  # @NOTE: x position of lobe centers
        self.y = np.zeros(self.alloc_n_lobes)  # @NOTE: y position of lobe centers
        self.x1 = np.zeros(self.alloc_n_lobes)
        self.x2 = np.zeros(self.alloc_n_lobes)
        self.dist_int = np.zeros(self.alloc_n_lobes, dtype=int) - 1
        self.parent = np.zeros(self.alloc_n_lobes, dtype=int)
        self.alfa_inertial = np.zeros(self.alloc_n_lobes)
        h = np.zeros(self.alloc_n_lobes)
        descendents = np.zeros(self.alloc_n_lobes, dtype=int)

    def compute_lobe_dimensions(self):
        input = self.input

        if input.volume_flag == 1:
            if input.total_volume is None:
                raise Exception("Total volume flag not set")

            if input.fixed_dimension_flag == 1:
                self.avg_lobe_thickness = input.total_volume / (
                    input.n_flows
                    * input.lobe_area
                    * 0.5
                    * (input.min_n_lobes + input.max_n_lobes)
                )
                sys.stdout.write(
                    "Average Lobe thickness = %f m\n\n" % (self.avg_lobe_thickness)
                )

            elif input.fixed_dimension_flag == 2:
                input.lobe_area = input.total_volume / (
                    input.n_flows
                    * self.avg_lobe_thickness
                    * 0.5
                    * (input.min_n_lobes + input.max_n_lobes)
                )
                sys.stdout.write("Lobe area = %f m2\n\n" % (input.lobe_area))

        self.max_semiaxis = np.sqrt(input.lobe_area * input.max_aspect_ratio / np.pi)
        self.max_cells = np.ceil(2.0 * self.max_semiaxis / self.asc_file.cell) + 2
        self.max_cells = self.max_cells.astype(int)

        self.thickness_min = (
            2.0
            * input.thickness_ratio
            / (input.thickness_ratio + 1.0)
            * self.avg_lobe_thickness
        )

        print("max_semiaxis", self.max_semiaxis)
        print("max_cells", self.max_cells)

    def check_channel_file(self):
        import shapefile
        from shapely.geometry import Point, LineString, MultiPoint
        from shapely.ops import nearest_points

        input = self.input
        asc_file = self.asc_file

        # arrays of components of the direction vectors computer from channel
        self.vx = np.zeros_like(asc_file.Xc)
        self.vy = np.zeros_like(asc_file.Yc)
        # arrays of distances from channel
        self.distxy = np.zeros_like(asc_file.Yc)

        print("")

        print("Reading shapefile " + input.channel_file)

        sf = shapefile.Reader(input.channel_file)

        shapes = sf.shapes()
        shapeRecs = sf.shapeRecords()
        points = shapeRecs[0].shape.points[0:]
        ln = LineString(points)

        pnew_x = points[-1][0] + 200.0 * (points[-1][0] - points[-2][0])
        pnew_y = points[-1][1] + 200.0 * (points[-1][1] - points[-2][1])

        points.append([pnew_x, pnew_y])

        nlx = []
        nly = []
        for i in range(len(points) - 1):
            nnx = points[i + 1][0] - points[i][0]
            nny = points[i + 1][1] - points[i][1]
            nn = np.sqrt(nnx**2 + nny**2)
            nlx.append(nnx / nn)
            nly.append(nny / nn)

        minx, miny, maxx, maxy = ln.bounds

        print("Channel Bounding Box", minx, miny, maxx, maxy)

        dx = 3.0 * input.d2

        minx = minx - dx
        maxx = maxx + dx
        miny = miny - dx
        maxy = maxy + dx

        min_xe = minx
        max_xe = maxx

        min_ye = miny
        max_ye = maxy

        xi = (min_xe - asc_file.xcmin) / asc_file.cell
        ix = np.floor(xi)
        i_left = ix.astype(int)

        xi = (max_xe - asc_file.xcmin) / asc_file.cell
        ix = np.floor(xi)
        i_right = ix.astype(int) + 2

        yj = (min_ye - asc_file.ycmin) / asc_file.cell
        jy = np.floor(yj)
        j_bottom = jy.astype(int)

        yj = (max_ye - asc_file.ycmin) / asc_file.cell
        jy = np.floor(yj)
        j_top = jy.astype(int) + 2

        print("i_left,i_right", i_left, i_right)
        print("j_bottom,j_top", j_bottom, j_top)

        # define the subgrid of pixels to check for coverage
        Xgrid = asc_file.Xc[j_bottom:j_top, i_left:i_right]
        Ygrid = asc_file.Yc[j_bottom:j_top, i_left:i_right]

        xgrid = Xgrid[0, :]
        ygrid = Ygrid[:, 0]

        coords = np.vstack((Xgrid.ravel(), Ygrid.ravel())).T
        pts = MultiPoint(coords)
        """
        dist_pl = np.zeros_like(arr)

        for idx, valx in enumerate(xgrid):
            for idy, valy in enumerate(ygrid):

                pt = Point(valx,valy)
                dist = np.exp(-pt.distance(ln)**2 / ( 2.0*d1**2))
                dist_pl[j_bottom+idy,i_left+idx] = dist
                filling_parameter[j_bottom+idy,i_left+idx] *= ( 1.0 - dist)


        header = "ncols     %s\n" % arr.shape[1]
        header += "nrows    %s\n" % arr.shape[0]
        header += "xllcorner " + str(lx) +"\n"
        header += "yllcorner " + str(ly) +"\n"
        header += "cellsize " + str(cell) +"\n"
        header += "NODATA_value 0\n"

        print('dist_pl',np.shape(dist_pl))

        output_full = input.run_name + '_channel_distance.asc'
        Zc -= 10*dist_pl


        print('Zc',np.shape(Zc))
        alfa_channel = 0.0

        np.savetxt(output_full, np.flipud(dist_pl), header=header,
                fmt='%1.5f',comments='')
        """
        # print(ciao)

        for idx, valx in enumerate(xgrid):
            for idy, valy in enumerate(ygrid):
                pt = Point(valx, valy)

                p1, p2 = nearest_points(ln, pt)
                xx, yy = p1.coords.xy
                vx1 = xx - valx
                vy1 = yy - valy
                v1mod = np.sqrt(vx1**2 + vy1**2)
                vx1 = vx1 / v1mod
                vy1 = vy1 / v1mod

                dist = []
                for i in range(len(points) - 1):
                    dist.append(
                        np.maximum(
                            input.eps, pt.distance(LineString(points[i : i + 2]))
                        )
                    )

                dist = np.array(dist) ** 2
                vx2 = np.sum(np.array(nlx) / dist) / np.sum(1.0 / np.array(dist))
                vy2 = np.sum(np.array(nly) / dist) / np.sum(1.0 / np.array(dist))

                v2mod = np.sqrt(vx2**2 + vy2**2)
                vx2 = vx2 / v2mod
                vy2 = vy2 / v2mod

                dist_pl = np.exp(
                    -pt.distance(LineString(points[0:])) ** 2 / (2.0 * input.d1**2)
                )
                self.vx[j_bottom + idy, i_left + idx] = (
                    dist_pl * vx2 + (1.0 - dist_pl) * vx1
                )
                self.vy[j_bottom + idy, i_left + idx] = (
                    dist_pl * vy2 + (1.0 - dist_pl) * vy1
                )

                vmod = np.sqrt(
                    self.vx[j_bottom + idy, i_left + idx] ** 2
                    + self.vy[j_bottom + idy, i_left + idx] ** 2
                )

                if vmod > 0:
                    self.vx[j_bottom + idy, i_left + idx] = (
                        self.vx[j_bottom + idy, i_left + idx] / vmod
                    )
                    self.vy[j_bottom + idy, i_left + idx] = (
                        self.vy[j_bottom + idy, i_left + idx] / vmod
                    )

                dist_pl = np.exp(
                    -pt.distance(LineString(points[0:-1])) ** 2 / (2.0 * input.d2**2)
                )

                self.distxy[j_bottom + idy, i_left + idx] = dist_pl

        print("Channel map completed")
        print("")

    def load_restarts(self):
        input = self.input
        asc_file = self.asc_file

        if input.restart_files is not None:
            n_restarts = len(input.restart_files)
        else:
            n_restarts = 0

        for i_restart in range(0, n_restarts):
            print("Read restart file ", input.restart_files[i_restart])
            Zflow_old = np.zeros((asc_file.ny, asc_file.nx))

            input.source = input.restart_files[i_restart]
            file_exists = not input.source is None
            if not file_exists:
                print(input.source + " not found.")
                quit()

            hdr = [getline(input.source, i) for i in range(1, 7)]

            try:
                values_restart = [float(h.split(" ")[-1].strip()) for h in hdr]
            except ValueError:
                print("An problem occurred with header of file ", input.source)
                print(hdr)

            cols_re, rows_re, lx_re, ly_re, cell_re, nd_re = values_restart
            header_asc = [
                asc_file.cols,
                asc_file.rows,
                asc_file.lx,
                asc_file.ly,
                asc_file.cell,
            ]

            if header_asc != values_restart[0:5]:
                print("Check on restart size failed")
                print(header_asc)
                print(values_restart)
                sys.exit(0)

            else:
                print("Check on restart size OK")

            # Load the previous flow thickness into a numpy array
            if asc_file.crop_flag:
                specific_rows = list(np.arange(6 + rows_re - asc_file.jN)) + list(
                    np.arange(6 + rows_re - asc_file.jS, 6 + rows_re)
                )
                specific_columns = list(np.arange(asc_file.iW, asc_file.iE))
                arr_df = pd.read_csv(
                    input.source,
                    delimiter=" ",
                    header=None,
                    usecols=specific_columns,
                    skiprows=specific_rows,
                    skipinitialspace=True,
                )
                arr = arr_df.to_numpy()
                arr[arr == nd_re] = 0.0
                arr = np.flipud(arr)

            else:
                arr_df = pd.read_csv(
                    input.source,
                    delimiter=" ",
                    skiprows=6,
                    header=None,
                    skipinitialspace=True,
                )
                arr = arr_df.to_numpy()
                arr[arr == nd_re] = 0.0
                arr = np.flipud(arr)

            Zflow_old = arr

            # print(np.where(Zflow_old==np.amax(Zflow_old)))

            # Load the relevant filling_parameter (to account for "subsurface flows")
            filling_parameter_i = input.restart_filling_parameters[i_restart]

            Zc = Zc + (Zflow_old * filling_parameter_i)
            print("Restart file read")

    def print_flow_status(self, flow, est_rem_time):
        input = self.input

        if input.n_flows > 1 and not ("SLURM_JOB_NAME" in os.environ.keys()):
            # print on screen bar with percentage of flows computed
            last_percentage_5 = np.rint(flow * 20.0 / (input.n_flows)).astype(int)
            last_percentage = np.rint(flow * 100.0 / (input.n_flows))
            last_percentage = np.rint(flow * 100.0 / (input.n_flows))
            last_percentage = last_percentage.astype(int)
            sys.stdout.write("\r")
            sys.stdout.write(
                "[%-20s] %d%% %s"
                % ("=" * (last_percentage_5), last_percentage, est_rem_time)
            )
            sys.stdout.flush()

    def print_lobe_status(self, i, n_lobes):
        input = self.input

        if input.n_flows == 1 and "SLURM_JOB_NAME" not in os.environ.keys():
            # print on screen bar with percentage of flows computed
            last_percentage = np.rint(i * 20.0 / (n_lobes - 1)) * 5
            sys.stdout.write("\r")
            sys.stdout.write(
                "[%-20s] %d%%" % ("=" * (last_percentage / 5), last_percentage)
            )
            sys.stdout.flush()

    def compute_first_lobe_position(self, i, flow):
        input = self.input

        if self.n_vents == 1:
            self.x[i] = input.x_vent[0]
            self.y[i] = input.y_vent[0]
        else:
            if input.vent_flag == 0:
                # input.vent_flag = 0  => the initial lobes are on the vents
                #                   coordinates and the flows start initially
                #                   from the first vent, then from the second
                #                   and so on.

                i_vent = int(np.floor(flow * self.n_vents / input.n_flows))

                self.x[i] = input.x_vent[i_vent]
                self.y[i] = input.y_vent[i_vent]

            elif input.vent_flag == 1:
                # input.vent_flag = 1  => the initial lobes are chosen randomly from
                #                   the vents coordinates and each vent has the
                #                   same probability

                i_vent = np.random.randint(self.n_vents, size=1)

                self.x[i] = input.x_vent[int(i_vent)]
                self.y[i] = input.y_vent[int(i_vent)]

            elif (input.vent_flag == 2) or (input.vent_flag == 6):
                # input.vent_flag = 2  => the initial lobes are on the polyline
                #                   connecting the vents and all the point of
                #                   the polyline have the same probability

                # input.vent_flag = 6  => the initial lobes are on the polyline
                #                   connecting the vents and the probability of
                #                   each segment is fixed in the input file

                alfa_polyline = np.random.uniform(0, 1, size=1)

                input.x_vent = np.argmax(self.cum_fiss_length > alfa_polyline)

                num = alfa_polyline - self.cum_fiss_length[input.x_vent - 1]
                den = (
                    self.cum_fiss_length[input.x_vent]
                    - self.cum_fiss_length[input.x_vent - 1]
                )

                alfa_segment = num / den

                self.x[i] = (
                    alfa_segment * input.x_vent[input.x_vent]
                    + (1.0 - alfa_segment) * input.x_vent[input.x_vent - 1]
                )

                self.y[i] = (
                    alfa_segment * input.y_vent[input.x_vent]
                    + (1.0 - alfa_segment) * input.y_vent[input.x_vent - 1]
                )

            elif input.vent_flag == 3:
                # input.vent_flag = 3  => the initial lobes are on the polyline
                #                   connecting the vents and all the segments
                #                   of the polyline have the same probability

                i_segment = randrange(self.n_vents)

                alfa_segment = np.random.uniform(0, 1, size=1)

                self.x[i] = (
                    alfa_segment * input.x_vent[i_segment]
                    + (1.0 - alfa_segment) * input.x_vent[i_segment - 1]
                )

                self.y[i] = (
                    alfa_segment * input.y_vent[i_segment]
                    + (1.0 - alfa_segment) * input.y_vent[i_segment - 1]
                )

            elif (input.vent_flag == 4) or (input.vent_flag == 7):
                # input.vent_flag = 4  => the initial lobes are on multiple
                #                   fissures and all the point of the fissures
                #                   have the same probability

                # input.vent_flag = 7  => the initial lobes are on multiple
                #                   fissures and the probability of
                #                   each fissure is fixed in the input file

                alfa_polyline = np.random.uniform(0, 1, size=1)

                input.x_vent = np.argmax(self.cum_fiss_length > alfa_polyline)

                num = alfa_polyline - self.cum_fiss_length[input.x_vent - 1]
                den = (
                    self.cum_fiss_length[input.x_vent]
                    - self.cum_fiss_length[input.x_vent - 1]
                )

                alfa_segment = num / den
                print()
                print(input.x_vent - 1, alfa_segment)

                self.x[i] = (
                    alfa_segment * input.x_vent_end[input.x_vent - 1]
                    + (1.0 - alfa_segment) * input.x_vent[input.x_vent - 1]
                )

                self.y[i] = (
                    alfa_segment * input.y_vent_end[input.x_vent - 1]
                    + (1.0 - alfa_segment) * input.y_vent[input.x_vent - 1]
                )

            elif input.vent_flag == 5:
                # input.vent_flag = 5  => the initial lobes are on multiple
                #                   fissures and all the fissures
                #                   have the same probability

                i_segment = randrange(self.n_vents)

                alfa_segment = np.random.uniform(0, 1, size=1)

                self.x[i] = (
                    alfa_segment * input.x_vent_end[i_segment]
                    + (1.0 - alfa_segment) * input.x_vent[i_segment]
                )

                self.y[i] = (
                    alfa_segment * input.y_vent_end[i_segment]
                    + (1.0 - alfa_segment) * input.y_vent[i_segment]
                )

            elif input.vent_flag == 8:
                # input.vent_flag = 1  => the initial lobes are chosen randomly from
                #                   the vents coordinates and each vent has
                #                   the same probability

                alfa_vent = np.random.uniform(0, 1, size=1)
                i_vent = np.argmax(self.cum_fiss_length > alfa_vent)

                self.x[i] = input.x_vent[int(i_vent)]
                self.y[i] = input.y_vent[int(i_vent)]

    def get_slope(self, i, Ztot):
        # compute the gradient of the topography(+ eventually the flow)
        # here the centered grid is used (Z values saved at the centers of
        # the pixels)
        # xc[ix] < lobe_center_x < xc[ix1]
        # yc[iy] < lobe_center_y < yc[iy1]
        asc_file = self.asc_file

        xi = (self.x[i] - asc_file.xcmin) / asc_file.cell
        yi = (self.y[i] - asc_file.ycmin) / asc_file.cell
        ix = np.floor(xi)
        iy = np.floor(yi)
        ix = ix.astype(int)  # @NOTE: Integer offset indices?
        iy = iy.astype(int)  # @NOTE: Integer offset indices?

        # compute the baricentric coordinated of the lobe center in the pixel
        # 0 < xi_fract < 1
        # 0 < yi_fract < 1
        xi_fract = xi - ix
        yi_fract = yi - iy

        # interpolate the slopes at the edges of the pixel to find the slope
        # at the center of the lobe
        Fx_test = (
            yi_fract * (Ztot[iy + 1, ix + 1] - Ztot[iy + 1, ix])
            + (1.0 - yi_fract) * (Ztot[iy, ix + 1] - Ztot[iy, ix])
        ) / asc_file.cell

        Fy_test = (
            xi_fract * (Ztot[iy + 1, ix + 1] - Ztot[iy, ix + 1])
            + (1.0 - xi_fract) * (Ztot[iy + 1, ix] - Ztot[iy, ix])
        ) / asc_file.cell

        # major semi-axis direction
        max_slope_angle = np.mod(
            180.0 + (180.0 * np.arctan2(Fy_test, Fx_test) / np.pi), 360.0
        )

        # slope of the topography at (x[0],y[0])
        slope = np.sqrt(np.square(Fx_test) + np.square(Fy_test))

        return max_slope_angle, slope

    def get_slope2(self, i, idx, Ztot):
        asc_file = self.asc_file
        input = self.input

        xi = (self.x[idx] - asc_file.xcmin) / asc_file.cell
        yi = (self.y[idx] - asc_file.ycmin) / asc_file.cell
        ix = np.floor(xi)
        iy = np.floor(yi)
        ix = ix.astype(int)
        iy = iy.astype(int)

        # stopping condition (lobe close the domain boundary)
        if (
            (ix <= 0.5 * self.max_cells)
            or (ix + 1 >= (asc_file.nx - 0.5 * self.max_cells))
            or (iy <= 0.5 * self.max_cells)
            or (iy + 1 >= (asc_file.ny - 0.5 * self.max_cells))
            or (asc_file.Zc[iy, ix] == asc_file.nd)
            or (asc_file.Zc[iy + 1, ix + 1] == asc_file.nd)
            or (asc_file.Zc[iy, ix + 1] == asc_file.nd)
            or (asc_file.Zc[iy + 1, ix] == asc_file.nd)
        ):
            last_lobe = i - 1
            return False, 0, 0, 0

        # compute the baricentric coordinated of the lobe center in the pixel
        # 0 < xi_fract < 1
        # 0 < yi_fract < 1
        xi_fract = xi - ix
        yi_fract = yi - iy

        # interpolate the elevation at the corners of the pixel to find the
        # elevation at the center of the lobe
        zidx = xi_fract * (
            yi_fract * Ztot[iy + 1, ix + 1] + (1.0 - yi_fract) * Ztot[iy, ix + 1]
        ) + (1.0 - xi_fract) * (
            yi_fract * Ztot[iy + 1, ix] + (1.0 - yi_fract) * Ztot[iy, ix]
        )

        # compute the lobe (input.npoints on the ellipse)
        [xe, ye] = self.ellipse(
            self.x[idx],
            self.y[idx],
            self.x1[idx],
            self.x2[idx],
            self.angle[idx],
        )

        # For all the points of the ellipse compute the indexes of the pixel
        # containing the points. This is done with respect to the centered
        # grid. We want to interpolate from the centered values (elevation)
        # to the location of the points on the ellipse)
        xei = (xe - asc_file.xcmin) / asc_file.cell
        yei = (ye - asc_file.ycmin) / asc_file.cell

        ixe = np.floor(xei)
        iye = np.floor(yei)

        ixe = ixe.astype(int)
        iye = iye.astype(int)

        ixe1 = ixe + 1
        iye1 = iye + 1

        # compute the local coordinates of the points (0<x,y<1) within the
        # pixels containing them
        xei_fract = xei - ixe
        yei_fract = yei - iye

        # interpolate the grid values to find the elevation at the ellipse
        # points
        ze = xei_fract * (
            yei_fract * Ztot[iye1, ixe1] + (1.0 - yei_fract) * Ztot[iye, ixe1]
        ) + (1.0 - xei_fract) * (
            yei_fract * Ztot[iye1, ixe] + (1.0 - yei_fract) * Ztot[iye, ixe]
        )

        # find the point on the ellipse with minimum elevation
        idx_min = np.argmin(ze)

        # compute the vector from the center of the lobe to the point of
        # minimum z on the boundary
        Fx_lobe = self.x[idx] - xe[idx_min]
        Fy_lobe = self.y[idx] - ye[idx_min]

        # compute the slope and the angle
        slope = np.maximum(
            0.0,
            (zidx - ze[idx_min]) / (np.sqrt(np.square(Fx_lobe) + np.square(Fy_lobe))),
        )

        max_slope_angle = np.mod(
            180.0 + (180.0 * np.arctan2(Fy_lobe, Fx_lobe) / np.pi), 360.0
        )

        return True, slope, max_slope_angle, zidx

    def compute_lobe_angle(self, max_slope_angle, slope):
        input = self.input
        # PERTURBE THE MAXIMUM SLOPE ANGLE ACCORDING TO PROBABILITY LAW
        # this expression define a coefficient used for the direction of the
        # next slope
        input = self.input
        if input.max_slope_prob < 1:
            # angle defining the direction of the new slope. when slope=0, then
            # we have an uniform distribution for the possible angles for the
            # next lobe.
            slopedeg = 180.0 * np.arctan(slope) / np.pi
            if (slopedeg > 0.0) and (input.max_slope_prob > 0):
                sigma = (
                    (1.0 - input.max_slope_prob)
                    / input.max_slope_prob
                    * (90.0 - slopedeg)
                    / slopedeg
                )
                rand_angle_new = rtnorm.rtnorm(-180, 180, 0, sigma)
            else:
                rand = np.random.uniform(0, 1, size=1)
                rand_angle_new = 360.0 * np.abs(rand - 0.5)

            return max_slope_angle + rand_angle_new
        else:
            return max_slope_angle

    def compute_lobe_axes(self, i, slope):
        input = self.input
        # factor for the lobe eccentricity
        aspect_ratio = min(
            input.max_aspect_ratio, 1.0 + input.aspect_ratio_coeff * slope
        )
        # semi-axes of the lobe:
        # self.x1(i) is the major semi-axis of the lobe;
        # self.x2(i) is the minor semi-axis of the lobe.
        self.x1[i] = np.sqrt(input.lobe_area / np.pi) * np.sqrt(aspect_ratio)
        self.x2[i] = np.sqrt(input.lobe_area / np.pi) / np.sqrt(aspect_ratio)

    def compute_lobe_parent_idx(self, i, descendents):
        input = self.input
        if input.lobe_exponent > 0:
            idx0 = np.random.uniform(0, 1, size=1)
            idx1 = idx0**input.lobe_exponent

            if input.force_max_length:
                # the parent lobe is chosen only among those with
                # dist smaller than the maximum value fixed in input
                mask = self.dist_int[0:i] < input.max_length
                idx2 = sum(mask[0:i]) * idx1
                idx3 = np.floor(idx2)
                idx = int(idx3)
                sorted_dist = np.argsort(self.dist_int[0:i])
                idx = sorted_dist[idx]
            else:
                # the parent lobe is chosen among all the lobes
                idx2 = i * idx1
                idx3 = np.floor(idx2)
                idx = int(idx3)

            if input.start_from_dist_flag:
                # the probability law is associated to the distance
                # from the vent
                sorted_dist = np.argsort(self.dist_int[0:i])
                idx = sorted_dist[idx]
        else:
            idx = i - 1

        # save the index of the parent and the distance from first lobe of the
        # chain
        self.parent[i] = idx
        self.dist_int[i] = self.dist_int[idx] + 1

        # for all the "ancestors" increase by one the number of descendents
        last = i
        for j in range(0, self.dist_int[idx] + 1):
            previous = self.parent[last]
            descendents[previous] = descendents[previous] + 1
            last = previous

        return idx

    def rasterize_lobe(
        self,
        i,
        delta_lobe_thickness,
        Zflow,
        Ztot,
        Zdist,
        Zflow_local_array,
        jtop_array,
        jbottom_array,
        iright_array,
        ileft_array,
    ):
        input = self.input
        asc_file = self.asc_file

        if input.saveraster_flag == 1:
            # compute the points of the lobe
            [xe, ye] = self.ellipse(
                self.x[i], self.y[i], self.x1[i], self.x2[i], self.angle[i]
            )

            # bounding box for the lobe
            # (xc[i_left]<xe<xc[i_right];yc[j_bottom]<ye<yc[j_top])
            # the indexes are referred to the centers of the pixels
            min_xe = np.min(xe)
            max_xe = np.max(xe)

            min_ye = np.min(ye)
            max_ye = np.max(ye)

            xi = (min_xe - asc_file.xcmin) / asc_file.cell
            ix = np.floor(xi)
            i_left = ix.astype(int)

            xi = (max_xe - asc_file.xcmin) / asc_file.cell
            ix = np.floor(xi)
            i_right = ix.astype(int) + 2

            yj = (min_ye - asc_file.ycmin) / asc_file.cell
            jy = np.floor(yj)
            j_bottom = jy.astype(int)

            yj = (max_ye - asc_file.ycmin) / asc_file.cell
            jy = np.floor(yj)
            j_top = jy.astype(int) + 2

            # define the subgrid of pixels to check for coverage
            Xc_local = asc_file.Xc[j_bottom:j_top, i_left:i_right]
            Yc_local = asc_file.Yc[j_bottom:j_top, i_left:i_right]

            # compute the fraction of cells covered by the lobe (local index)
            # for each pixel a square [-0.5*cell;0.5*cell]X[-0.5*cell;0.5*cell]
            # is built around its center to compute the intersection with the
            # lobe the coverage values are associated to each pixel (at the
            # center)
            area_fract = self.local_intersection(
                Xc_local,
                Yc_local,
                self.x[i],
                self.y[i],
                self.x1[i],
                self.x2[i],
                self.angle[i],
            )
            Zflow_local = area_fract

            # compute also as integer (0=pixel non covereb by lobe;1=pixel
            # covered by lobe)
            Zflow_local_int = np.ceil(area_fract)
            Zflow_local_int = Zflow_local_int.astype(int)

            # compute the thickness of the lobe
            lobe_thickness = self.thickness_min + (i - 1) * delta_lobe_thickness

            # update the thickness of the flow with the new lobe
            Zflow[j_bottom:j_top, i_left:i_right] += lobe_thickness * Zflow_local

            # update the topography

            # change 2022/01/13
            # FROM HERE
            Ztot[j_bottom:j_top, i_left:i_right] += (
                self.filling_parameter[j_bottom:j_top, i_left:i_right]
                * lobe_thickness
                * Zflow_local
            )

            # TO HERE

            # compute the new minimum "lobe distance" of the pixels from the
            # vent
            Zdist_local = Zflow_local_int * self.dist_int[i] + 9999 * (Zflow_local == 0)

            Zdist[j_bottom:j_top, i_left:i_right] = np.minimum(
                Zdist[j_bottom:j_top, i_left:i_right], Zdist_local
            )

            # store the bounding box of the new lobe
            jtop_array[i] = j_top
            jbottom_array[i] = j_bottom
            iright_array[i] = i_right
            ileft_array[i] = i_left

            if input.hazard_flag:
                # store the local array of integer coverage in the global array
                Zflow_local_array[
                    i, 0 : j_top - j_bottom, 0 : i_right - i_left
                ] = Zflow_local_int

    def add_inertia(self, i, idx, slope, new_angle):
        input = self.input

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

        # if not input.alfa_channel is None and input.alfa_channel > 0.0:
        #     old_angle = new_angle

        #     # interpolate the vector at the corners of the pixel to find the
        #     # vector at the center of the lobe
        #     cos_angle_old = np.cos(np.radians(old_angle))
        #     sin_angle_old = np.sin(np.radians(old_angle))

        #     # print('cos_angle1,sin_angle1',cos_angle1,sin_angle1)

        #     x_avg2 = xi_fract * (
        #         yi_fract * self.vx[iy1, ix1] + (1.0 - yi_fract) * self.vx[iy, ix1]
        #     ) + (1.0 - xi_fract) * (
        #         yi_fract * self.vx[iy1, ix] + (1.0 - yi_fract) * self.vx[iy, ix]
        #     )
        #     y_avg2 = xi_fract * (
        #         yi_fract * self.vy[iy1, ix1] + (1.0 - yi_fract) * self.vy[iy, ix1]
        #     ) + (1.0 - xi_fract) * (
        #         yi_fract * self.vy[iy1, ix] + (1.0 - yi_fract) * self.vy[iy, ix]
        #     )

        #     if x_avg2**2 + y_avg2**2 > 0.0:
        #         cos_angle_new = x_avg2 / np.sqrt(x_avg2**2 + y_avg2**2)
        #         sin_angle_new = y_avg2 / np.sqrt(x_avg2**2 + y_avg2**2)

        #         # print('cos_angle2,sin_angle2',cos_angle2,sin_angle2)

        #         distxyidx = xi_fract * (
        #             yi_fract * self.distxy[iy1, ix1]
        #             + (1.0 - yi_fract) * self.distxy[iy, ix1]
        #         ) + (1.0 - xi_fract) * (
        #             yi_fract * self.distxy[iy1, ix]
        #             + (1.0 - yi_fract) * self.distxy[iy, ix]
        #         )

        #         x_avg = (
        #             1.0 - input.alfa_channel * distxyidx
        #         ) * cos_angle_old + input.alfa_channel * distxyidx * cos_angle_new
        #         y_avg = (
        #             1.0 - input.alfa_channel * distxyidx
        #         ) * sin_angle_old + input.alfa_channel * distxyidx * sin_angle_new

        #         angle_avg = np.mod(180.0 * np.arctan2(y_avg, x_avg) / np.pi, 360.0)

        #         new_angle = angle_avg

    def run(self):
        input = self.input

        print("\nMr Lava Loba by M.de' Michieli Vitturi and S.Tarquini\n")

        # read the run parameters from the file input_data.py
        self.n_vents = len(input.x_vent)

        self.compute_cumulative_fissure_length()

        self.setup_run_file()

        self.allocate_lobe_data()

        # Define variables needed to build the ellipses
        t = np.linspace(0.0, 2.0 * np.pi, input.npoints)
        self.X_circle = np.cos(t)
        self.Y_circle = np.sin(t)

        self.asc_file = read_asc_file(input)
        asc_file = self.asc_file

        self.compute_lobe_dimensions()

        self.filling_parameter = (1.0 - input.thickening_parameter) * np.ones_like(
            self.asc_file.Zc
        )

        if not self.input.channel_file is None:
            self.check_channel_file()

        # load restart files (if existing)
        self.load_restarts()

        # Define a small grid for lobe-cells intersection (for the local intersections)
        nv = 15
        self.xv, self.yv = np.meshgrid(
            np.linspace(-0.5 * asc_file.cell, 0.5 * asc_file.cell, nv),
            np.linspace(-0.5 * asc_file.cell, 0.5 * asc_file.cell, nv),
        )
        self.xv = np.reshape(self.xv, -1)
        self.yv = np.reshape(self.yv, -1)
        self.nv2 = nv**2

        # I think you'd rather do this
        Ztot = np.array(asc_file.Zc)
        Zflow = np.zeros((asc_file.ny, asc_file.nx))

        jtop_array = np.zeros(self.alloc_n_lobes, dtype=int)
        jbottom_array = np.zeros(self.alloc_n_lobes, dtype=int)

        iright_array = np.zeros(self.alloc_n_lobes, dtype=int)
        ileft_array = np.zeros(self.alloc_n_lobes, dtype=int)

        Zhazard = np.zeros((asc_file.ny, asc_file.nx), dtype=int)
        Zhazard_temp = np.zeros((asc_file.ny, asc_file.nx), dtype=int)

        Zdist = np.zeros((asc_file.ny, asc_file.nx), dtype=int) + 9999

        print("End pre-processing\n")

        if sys.version_info >= (3, 0):
            start = time.process_time()
        else:
            start = time.clock()

        est_rem_time = ""

        n_lobes_tot = 0

        for flow in range(0, input.n_flows):
            Zflow_local_array = np.zeros(
                (self.alloc_n_lobes, self.max_cells, self.max_cells), dtype=int
            )

            # @NOTE: number of decendents ???
            descendents = np.zeros(self.alloc_n_lobes, dtype=int)

            # @NOTE: This seems unused?
            i_first_check = input.n_check_loop

            if (input.a_beta == 0) and (input.b_beta == 0):
                # DEFINE THE NUMBER OF LOBES OF THE FLOW (RANDOM VALUE BETWEEN
                # MIN AND MAX)
                # @NOTE: What does n_lobes mean? Whats the difference to alloc_n_lobes
                n_lobes = int(
                    np.ceil(
                        np.random.uniform(input.min_n_lobes, input.max_n_lobes, size=1)
                    )
                )
            else:
                x_beta = (1.0 * flow) / (input.n_flows - 1)
                n_lobes = int(
                    np.rint(
                        input.min_n_lobes
                        + 0.5
                        * (input.max_n_lobes - input.min_n_lobes)
                        * beta.pdf(x_beta, input.a_beta, input.b_beta)
                    )
                )

            n_lobes_tot += n_lobes

            delta_lobe_thickness = (
                2.0 * (self.avg_lobe_thickness - self.thickness_min) / (n_lobes - 1.0)
            )

            self.print_flow_status(flow, est_rem_time)

            # @NOTE: this loops over the n_init initial lobes
            for i in range(0, input.n_init):
                self.print_lobe_status(i, n_lobes)

                # STEP 0: COMPUTE THE FIRST LOBES OF EACH FLOW
                self.compute_first_lobe_position(i, flow)

                # initialize distance from first lobe and number of descendents
                self.dist_int[i] = 0  # @NOTE: why is the distance an integer?
                descendents[i] = 0  # @NOTE: the total number of descendants per lobe?

                max_slope_angle, slope = self.get_slope(i, Ztot)

                self.angle[i] = self.compute_lobe_angle(max_slope_angle, slope)

                self.compute_lobe_axes(i, slope)

                self.rasterize_lobe(
                    i,
                    delta_lobe_thickness,
                    Zflow,
                    Ztot,
                    Zdist,
                    Zflow_local_array,
                    jtop_array,
                    jbottom_array,
                    iright_array,
                    ileft_array,
                )

            last_lobe = n_lobes

            # @NOTE: this loops over the rest of the lobes (skipping the initial ones)
            for i in range(input.n_init, n_lobes):
                # print('i',i)

                self.print_lobe_status(i, n_lobes)
                # STEP 0: DEFINE THE INDEX idx OF THE PARENT LOBE
                idx = self.compute_lobe_parent_idx(i, descendents)

                # local slope of the topography. The slope affects both the location of
                # the new lobe on the boundary of the previous one and its aspect
                # ratio:
                # if slope = 0 the lobe is a circle (x1=x2);
                # if slope > 1 the lobe is an ellipse.

                # STEP 1: COMPUTE THE SLOPE AND THE MAXIMUM SLOPE ANGLE
                # here the centered grid is used (Z values saved at the centers of the
                # pixels)
                # xc[ix] < lobe_center_x < xc[ix1]
                # yc[iy] < lobe_center_y < yc[iy1]
                status, slope, max_slope_angle, zidx = self.get_slope2(i, idx, Ztot)

                if not status:
                    break

                # STEP 2: PERTURBE THE MAXIMUM SLOPE ANGLE ACCORDING TO PROBABILITY LAW
                # this expression define a coefficient used for the direction of the
                # next slope

                if input.max_slope_prob < 1:
                    # angle defining the direction of the new slope. when slope=0, then
                    # we have an uniform distribution for the possible angles for the
                    # next lobe.
                    new_angle = self.compute_lobe_angle(max_slope_angle, slope)

                # STEP 3: ADD THE EFFECT OF INERTIA

                new_angle = self.add_inertia(i, idx, slope, new_angle)

                # STEP 4: DEFINE THE SEMI-AXIS OF THE NEW LOBE

                # a define the ang.coeff. of the line defining the location of the
                # center of the new lobe in a coordinate system defined by the
                # semi-axes of the existing lobe
                a = np.tan(np.pi / 180.0 * (new_angle - self.angle[idx]))

                # xt is the 1st-coordinate of the point of the boundary of the ellipse
                # definind the direction of the new lobe, in a coordinate system
                # defined by the semi-axes of the existing lobe
                if np.cos(np.pi / 180.0 * (new_angle - self.angle[idx])) > 0:
                    xt = np.sqrt(
                        self.x1[idx] ** 2
                        * self.x2[idx] ** 2
                        / (self.x2[idx] ** 2 + self.x1[idx] ** 2 * a**2)
                    )

                else:
                    xt = -np.sqrt(
                        self.x1[idx] ** 2
                        * self.x2[idx] ** 2
                        / (self.x2[idx] ** 2 + self.x1[idx] ** 2 * a**2)
                    )

                # yt is the 2nd-coordinate of the point of the boundary of the ellipse
                # definind the direction of the new lobe, in a coordinate system
                # defined by the semi-axes of the existing lobe
                yt = a * xt

                # (delta_x,delta_y) is obtained rotating the vector (xt,yt) by the
                # angle defined by the major semi-axis of the existing lobe. In this
                # way we obtain the location in a coordinate-system centered in the
                # center of the existing lobe, but this time with the axes parallel to
                # the original x and y axes.

                cos_angle1 = np.cos(self.angle[idx] * np.pi / 180.0)
                sin_angle1 = np.sin(self.angle[idx] * np.pi / 180.0)
                delta_x = xt * cos_angle1 - yt * sin_angle1
                delta_y = xt * sin_angle1 + yt * cos_angle1

                # the slope coefficient is evaluated at the point of the boundary of
                # the ellipse definind by the direction of the new lobe

                xi = (self.x[idx] + delta_x - asc_file.xcmin) / asc_file.cell
                yi = (self.y[idx] + delta_y - asc_file.ycmin) / asc_file.cell

                ix = np.floor(xi)
                iy = np.floor(yi)

                ix = ix.astype(int)
                iy = iy.astype(int)

                ix1 = ix + 1
                iy1 = iy + 1

                # stopping condition (lobe close the domain boundary)
                if (
                    (ix <= 0.5 * self.max_cells)
                    or (ix1 >= asc_file.nx - 0.5 * self.max_cells)
                    or (iy <= 0.5 * self.max_cells)
                    or (iy1 >= asc_file.ny - 0.5 * self.max_cells)
                ):
                    # print('ix',ix,'iy',iy)
                    last_lobe = i - 1
                    break

                xi_fract = xi - ix
                yi_fract = yi - iy

                # ztot at the new budding point
                ze = xi_fract * (
                    yi_fract * Ztot[iy1, ix1] + (1.0 - yi_fract) * Ztot[iy, ix1]
                ) + (1.0 - xi_fract) * (
                    yi_fract * Ztot[iy1, ix] + (1.0 - yi_fract) * Ztot[iy, ix]
                )

                slope = np.maximum(
                    0.0,
                    (zidx - ze) / (np.sqrt(np.square(delta_x) + np.square(delta_y))),
                )

                aspect_ratio = min(
                    input.max_aspect_ratio, 1.0 + input.aspect_ratio_coeff * slope
                )

                # (new_x1,new_x2) are the semi-axes of the new lobe. slope_coeff is
                # used to have an elongated lobe accoriding to the slope of the
                # topography. It is possible to modifiy these values in order to have
                # the same volume for all the lobes.
                new_x1 = np.sqrt(input.lobe_area / np.pi) * np.sqrt(aspect_ratio)
                new_x2 = np.sqrt(input.lobe_area / np.pi) / np.sqrt(aspect_ratio)

                # v1 is the distance of the new point found on the boundary of the lobe
                # from the center of the lobe
                v1 = np.sqrt(delta_x**2 + delta_y**2)

                # v2 is the distance between the centers of the two lobes when they
                # intersect in one point only
                v2 = v1 + new_x1

                # v is the distance between the centers of the two lobes, according to
                # the value of the parameter input.dist_fact
                v = (v1 * (1.0 - input.dist_fact) + v2 * input.dist_fact) / v1

                # STEP 5: BUILD THE NEW LOBE

                # (x_new,y_new) are the coordinates of the center of the new lobe
                x_new = self.x[idx] + v * delta_x
                y_new = self.y[idx] + v * delta_y

                # store the parameters of the new lobe in arrays
                self.angle[i] = new_angle
                self.x1[i] = new_x1
                self.x2[i] = new_x2
                self.x[i] = x_new
                self.y[i] = y_new

                # check the grid points covered by the lobe
                if input.saveraster_flag == 1:
                    # compute the new lobe
                    [xe, ye] = self.ellipse(
                        self.x[i], self.y[i], self.x1[i], self.x2[i], self.angle[i]
                    )

                    # bounding box for the new lobe
                    # the indexes are referred to the centers of the pixels
                    min_xe = np.min(xe)
                    max_xe = np.max(xe)

                    min_ye = np.min(ye)
                    max_ye = np.max(ye)

                    xi = (min_xe - asc_file.xcmin) / asc_file.cell
                    ix = np.floor(xi)
                    i_left = ix.astype(int)
                    i_left = np.maximum(0, np.minimum(asc_file.nx - 1, i_left))

                    xi = (max_xe - asc_file.xcmin) / asc_file.cell
                    ix = np.floor(xi)
                    i_right = ix.astype(int) + 2
                    i_right = np.maximum(0, np.minimum(asc_file.nx - 1, i_right))

                    yj = (min_ye - asc_file.ycmin) / asc_file.cell
                    jy = np.floor(yj)
                    j_bottom = jy.astype(int)
                    j_bottom = np.maximum(0, np.minimum(asc_file.ny - 1, j_bottom))

                    yj = (max_ye - asc_file.ycmin) / asc_file.cell
                    jy = np.floor(yj)
                    j_top = jy.astype(int) + 2
                    j_top = np.maximum(0, np.minimum(asc_file.ny - 1, j_top))

                    # the centers of the pixels are used to compute the intersection
                    # with the lobe
                    Xc_local = asc_file.Xc[j_bottom:j_top, i_left:i_right]
                    Yc_local = asc_file.Yc[j_bottom:j_top, i_left:i_right]

                    # compute the fraction of cells covered by the lobe (local index)
                    # for each pixel a square [-0.5*cell;0.5*cell]X[-0.5*cell;0.5*cell]
                    # is built around its center to compute the intersection with the
                    # lobe the coverage values are associated to each pixel (at the
                    # center)
                    area_fract = self.local_intersection(
                        Xc_local,
                        Yc_local,
                        self.x[i],
                        self.y[i],
                        self.x1[i],
                        self.x2[i],
                        self.angle[i],
                    )

                    Zflow_local = area_fract

                    # compute the local integer covering (0-not covered  1-covered)
                    Zflow_local_int = np.ceil(area_fract)
                    Zflow_local_int = Zflow_local_int.astype(int)

                    # print('Zflow_local_int')
                    # print(Zflow_local_int)

                    # define the distance (number of lobes) from the vent (local index)
                    Zdist_local = Zflow_local_int * self.dist_int[i] + 9999 * (
                        Zflow_local == 0
                    )

                    # update the minimum distance in the global indexing
                    Zdist[j_bottom:j_top, i_left:i_right] = np.minimum(
                        Zdist[j_bottom:j_top, i_left:i_right], Zdist_local
                    )

                    # compute the thickness of the lobe
                    lobe_thickness = self.thickness_min + (i - 1) * delta_lobe_thickness

                    # update the thickness for the grid points selected (global index)
                    Zflow[j_bottom:j_top, i_left:i_right] += (
                        lobe_thickness * Zflow_local
                    )

                    # change 2022/01/13

                    Ztot[j_bottom:j_top, i_left:i_right] += (
                        self.filling_parameter[j_bottom:j_top, i_left:i_right]
                        * lobe_thickness
                        * Zflow_local
                    )
                    # TO HERE

                    # save the bounding box of the i-th lobe
                    jtop_array[i] = j_top
                    jbottom_array[i] = j_bottom

                    iright_array[i] = i_right
                    ileft_array[i] = i_left

                    if input.hazard_flag:
                        # store the local arrays used later for the hazard map

                        if not (Zflow_local_int.shape[0] == (j_top - j_bottom)):
                            print(Zflow_local_int.shape[0], j_top, j_bottom)
                            print(Zflow_local_int.shape[1], i_right, i_left)
                            print("")

                        if not (Zflow_local_int.shape[1] == (i_right - i_left)):
                            print(Zflow_local_int.shape[0], j_top, j_bottom)
                            print(Zflow_local_int.shape[1], i_right, i_left)
                            print("")

                        if np.max(Zflow_local.shape) > Zflow_local_array.shape[1]:
                            print("check 3")
                            print(asc_file.cell, new_x1, new_x2, new_angle)
                            print(self.x[i], self.y[i], self.x1[i], self.x2[i])
                            np.set_printoptions(precision=1)
                            print(Zflow_local_int)

                        Zflow_local_array[
                            i, 0 : j_top - j_bottom, 0 : i_right - i_left
                        ] = Zflow_local_int

            if input.hazard_flag:
                # update the hazard map accounting for the number of descendents,
                # representative of the number of times a flow has passed over a cell

                for i in range(0, last_lobe):
                    j_top = jtop_array[i]
                    j_bottom = jbottom_array[i]

                    i_right = iright_array[i]
                    i_left = ileft_array[i]

                    if i > 0:
                        j_top_int = np.minimum(j_top, jtop_array[self.parent[i]])
                        j_bottom_int = np.maximum(
                            j_bottom, jbottom_array[self.parent[i]]
                        )
                        i_left_int = np.maximum(i_left, ileft_array[self.parent[i]])
                        i_right_int = np.minimum(i_right, iright_array[self.parent[i]])

                        Zlocal_new = np.zeros(
                            (self.max_cells, self.max_cells), dtype=int
                        )
                        Zlocal_parent = np.zeros(
                            (self.max_cells, self.max_cells), dtype=int
                        )

                        Zlocal_parent = Zflow_local_array[
                            self.parent[i],
                            np.maximum(
                                0, j_bottom_int - jbottom_array[self.parent[i]]
                            ) : np.minimum(
                                j_top_int - jbottom_array[self.parent[i]],
                                jtop_array[self.parent[i]]
                                - jbottom_array[self.parent[i]],
                            ),
                            np.maximum(
                                i_left_int - ileft_array[self.parent[i]], 0
                            ) : np.minimum(
                                i_right_int - ileft_array[self.parent[i]],
                                iright_array[self.parent[i]]
                                - ileft_array[self.parent[i]],
                            ),
                        ]

                        Zlocal_new = Zflow_local_array[
                            i, 0 : j_top - j_bottom, 0 : i_right - i_left
                        ]

                        if Zlocal_parent.shape[0] == 0 or Zlocal_parent.shape[1] == 0:
                            print("check")
                            print("idx", i)
                            print("j", j_bottom, j_top)
                            print("i", i_left, i_right)
                            print("idx parent", self.parent[i])
                            print(
                                "j",
                                jbottom_array[self.parent[i]],
                                jtop_array[self.parent[i]],
                            )
                            print(
                                "i",
                                ileft_array[self.parent[i]],
                                iright_array[self.parent[i]],
                            )
                            print(j_bottom_int, j_top_int, i_left_int, i_right_int)

                        Zlocal_new[
                            np.maximum(0, j_bottom_int - j_bottom) : np.minimum(
                                j_top_int - j_bottom, j_top - j_bottom
                            ),
                            np.maximum(i_left_int - i_left, 0) : np.minimum(
                                i_right_int - i_left, i_right - i_left
                            ),
                        ] *= (
                            1 - Zlocal_parent
                        )

                        Zhazard[j_bottom:j_top, i_left:i_right] += (
                            descendents[i]
                            * Zlocal_new[0 : j_top - j_bottom, 0 : i_right - i_left]
                        )

                    else:
                        Zhazard[j_bottom:j_top, i_left:i_right] += (
                            descendents[i]
                            * Zflow_local_array[
                                i, 0 : j_top - j_bottom, 0 : i_right - i_left
                            ]
                        )

            if sys.version_info >= (3, 0):
                elapsed = time.process_time() - start
            else:
                elapsed = time.clock() - start

            estimated = np.ceil(elapsed * input.n_flows / (flow + 1) - elapsed)
            est_rem_time = str(datetime.timedelta(seconds=estimated))

        if input.n_flows > 1 and "SLURM_JOB_NAME" not in os.environ.keys():
            # print on screen bar with percentage of flows computed
            last_percentage = 100
            sys.stdout.write("\r")
            sys.stdout.write("[%-20s] %d%%" % ("=" * 20, last_percentage))
            sys.stdout.flush()

        if sys.version_info >= (3, 0):
            elapsed = time.process_time() - start
        else:
            elapsed = time.clock() - start

        print("")
        print("")
        print(
            "Total number of lobes",
            n_lobes_tot,
            "Average number of lobes",
            int(1.0 * n_lobes_tot / input.n_flows),
        )
        print("")
        print(
            "Time elapsed "
            + str(elapsed)
            + " sec."
            + " / "
            + str(int(elapsed / 60))
            + " min."
        )
        print("")
        print("Saving files")

        if input.saveraster_flag == 1:
            # Save raster files

            header = "ncols     %s\n" % Zflow.shape[1]
            header += "nrows    %s\n" % Zflow.shape[0]
            header += "xllcorner " + str(asc_file.lx) + "\n"
            header += "yllcorner " + str(asc_file.ly) + "\n"
            header += "cellsize " + str(asc_file.cell) + "\n"
            header += "NODATA_value 0\n"

            output_full = input.run_name + "_thickness_full.asc"

            np.savetxt(
                output_full, np.flipud(Zflow), header=header, fmt="%1.5f", comments=""
            )

            print("")
            print(output_full + " saved")

            flag_union_diff = False

            if input.union_diff_file is not None:
                # Parse the header using a loop and
                # the built-in linecache module
                hdr = [getline(input.union_diff_file, i) for i in range(1, 7)]
                values = [float(h.split(" ")[-1].strip()) for h in hdr]
                Zs_temp = np.flipud(np.loadtxt(input.union_diff_file, skiprows=6))

                cols_ud, rows_ud, lx_ud, ly_ud, cell_ud, nd_ud = values
                if (
                    (cols_ud != Zflow.shape[1])
                    or (rows_ud != Zflow.shape[0])
                    or (lx_ud != asc_file.lx)
                    or (ly_ud != asc_file.ly)
                    or (cell_ud != asc_file.cell)
                ):
                    print("Union_diff_file", input.union_diff_file)
                    print("Different header: interpolating data")

                    xin = lx_ud + cell_ud * np.arange(cols_ud)
                    yin = ly_ud + cell_ud * np.arange(rows_ud)

                    xout = asc_file.lx + cell_ud * np.arange(asc_file.nx)
                    yout = asc_file.ly + cell_ud * np.arange(asc_file.ny)

                    Xout, Yout = np.meshgrid(xout, yout)

                    Zs1 = interp2Dgrids(xin, yin, Zs_temp, Xout, Yout)

                    flag_union_diff = True

                else:
                    flag_union_diff = True
                    Zs1 = Zs_temp

                Zs2 = Zflow

                Zs_union = np.maximum(Zs1, Zs2)

                Zs_union = Zs_union / np.maximum(Zs_union, 1)
                area_union = np.sum(Zs_union) * asc_file.cell**2

                Zs_inters = np.minimum(Zs1, Zs2)

                Zs_inters = Zs_inters / np.maximum(Zs_inters, 1)
                area_inters = np.sum(Zs_inters) * asc_file.cell**2

                # area_inters = np.count_nonzero(Zs_inters) * cell**2
                fitting_parameter = area_inters / area_union

                print("--------------------------------")
                print("With full output")
                print("Union area", area_union, "Intersect. area", area_inters)
                print("Fitting parameter", fitting_parameter)

                Zs1_mean = (
                    np.mean(Zs1 * Zs_inters)
                    * asc_file.nx
                    * asc_file.ny
                    / np.count_nonzero(Zs_inters)
                )
                Zs2_mean = (
                    np.mean(Zs2 * Zs_inters)
                    * asc_file.nx
                    * asc_file.ny
                    / np.count_nonzero(Zs_inters)
                )

                Zs1_vol = Zs1_mean * area_inters
                Zs2_vol = Zs2_mean * area_inters

                print(
                    "Volume 1 in intersection",
                    Zs1_vol,
                    "Volume 2 in intersection",
                    Zs2_vol,
                )

                Zs_diff = np.abs(Zs1 - Zs2)

                Zs_diff = Zs_diff * Zs_inters

                avg_thick_diff = (
                    np.mean(Zs_diff)
                    * asc_file.nx
                    * asc_file.ny
                    / np.count_nonzero(Zs_inters)
                )
                std_thick_diff = (
                    np.std(Zs_diff)
                    * asc_file.nx
                    * asc_file.ny
                    / np.count_nonzero(Zs_inters)
                )
                vol_diff = avg_thick_diff * area_inters

                rel_err_vol = vol_diff / np.maximum(Zs1_vol, Zs2_vol)

                print("Thickness relative error", rel_err_vol)
                print("--------------------------------")

            else:
                print("Union_diff_file not defined")
                flag_union_diff = False

            if isinstance(input.masking_threshold, float):
                input.masking_threshold = [input.masking_threshold]

            n_masking = len(input.masking_threshold)

            for i_thr in range(n_masking):
                if input.masking_threshold[i_thr] < 1:
                    max_lobes = int(np.floor(np.max(Zflow / self.avg_lobe_thickness)))

                    for i in range(1, 10 * max_lobes):
                        masked_Zflow = ma.masked_where(
                            Zflow < i * 0.1 * self.avg_lobe_thickness, Zflow
                        )

                        total_Zflow = np.sum(Zflow)

                        if input.flag_threshold == 1:
                            volume_fraction = np.sum(masked_Zflow) / total_Zflow

                            coverage_fraction = volume_fraction

                        else:
                            area_fraction = np.true_divide(
                                np.sum(masked_Zflow > 0), np.sum(Zflow > 0)
                            )

                            coverage_fraction = area_fraction
                            # print (coverage_fraction)

                        if coverage_fraction < input.masking_threshold[i_thr]:
                            if input.flag_threshold == 1:
                                print("")
                                print(
                                    "Masking threshold", input.masking_threshold[i_thr]
                                )
                                print(
                                    "Total volume",
                                    asc_file.cell**2 * total_Zflow,
                                    " m3 Masked volume",
                                    asc_file.cell**2 * np.sum(masked_Zflow),
                                    " m3 Volume fraction",
                                    coverage_fraction,
                                )
                                print(
                                    "Total area",
                                    asc_file.cell**2 * np.sum(Zflow > 0),
                                    " m2 Masked area",
                                    asc_file.cell**2 * np.sum(masked_Zflow > 0),
                                    " m2",
                                )
                                print(
                                    "Average thickness full",
                                    total_Zflow / np.sum(Zflow > 0),
                                    " m Average thickness mask",
                                    np.sum(masked_Zflow) / np.sum(masked_Zflow > 0),
                                    " m",
                                )

                            output_thickness = input.run_name + "_avg_thick.txt"
                            with open(output_thickness, "a") as the_file:
                                if i_thr == 0:
                                    the_file.write(
                                        "Average lobe thickness = "
                                        + str(self.avg_lobe_thickness)
                                        + " m\n"
                                    )
                                    the_file.write(
                                        "Total volume = "
                                        + str(asc_file.cell**2 * total_Zflow)
                                        + " m3\n"
                                    )
                                    the_file.write(
                                        "Total area = "
                                        + str(asc_file.cell**2 * np.sum(Zflow > 0))
                                        + " m2\n"
                                    )
                                    the_file.write(
                                        "Average thickness full = "
                                        + str(total_Zflow / np.sum(Zflow > 0))
                                        + " m\n"
                                    )

                                the_file.write(
                                    "Masking threshold = "
                                    + str(input.masking_threshold[i_thr])
                                    + "\n"
                                )
                                the_file.write(
                                    "Masked volume = "
                                    + str(asc_file.cell**2 * np.sum(masked_Zflow))
                                    + " m3\n"
                                )
                                the_file.write(
                                    "Masked area = "
                                    + str(asc_file.cell**2 * np.sum(masked_Zflow > 0))
                                    + " m2\n"
                                )
                                the_file.write(
                                    "Average thickness mask = "
                                    + str(
                                        np.sum(masked_Zflow) / np.sum(masked_Zflow > 0)
                                    )
                                    + " m\n"
                                )

                            output_masked = (
                                input.run_name
                                + "_thickness_masked"
                                + "_"
                                + str(input.masking_threshold[i_thr]).replace(".", "_")
                                + ".asc"
                            )

                            np.savetxt(
                                output_masked,
                                np.flipud((1 - masked_Zflow.mask) * Zflow),
                                header=header,
                                fmt="%1.5f",
                                comments="",
                            )

                            print("")
                            print(output_masked + " saved")

                            break

                    if flag_union_diff:
                        Zs2 = (1 - masked_Zflow.mask) * Zflow
                        Zs_union = np.maximum(Zs1, Zs2)

                        Zs_union = Zs_union / np.maximum(Zs_union, 1)
                        area_union = np.sum(Zs_union) * asc_file.cell**2

                        # area_union = np.count_nonzero(Zs_union) * asc_file.cell**2

                        Zs_inters = np.minimum(Zs1, Zs2)

                        Zs_inters = Zs_inters / np.maximum(Zs_inters, 1)
                        area_inters = np.sum(Zs_inters) * asc_file.cell**2

                        # area_inters = np.count_nonzero(Zs_inters) * asc_file.cell**2
                        fitting_parameter = area_inters / area_union

                        print("--------------------------------")
                        print("With masking threshold", input.masking_threshold[i_thr])
                        print("Union area", area_union, "Intersect. area", area_inters)
                        print("Fitting parameter", fitting_parameter)

                        Zs1_mean = (
                            np.mean(Zs1 * Zs_inters)
                            * asc_file.nx
                            * asc_file.ny
                            / np.count_nonzero(Zs_inters)
                        )
                        Zs2_mean = (
                            np.mean(Zs2 * Zs_inters)
                            * asc_file.nx
                            * asc_file.ny
                            / np.count_nonzero(Zs_inters)
                        )

                        Zs1_vol = Zs1_mean * area_inters
                        Zs2_vol = Zs2_mean * area_inters

                        print(
                            "Vol 1 in intersect.",
                            Zs1_vol,
                            "Vol 2 in intersect.",
                            Zs2_vol,
                        )

                        Zs_diff = np.abs(Zs1 - Zs2)

                        Zs_diff = Zs_diff * Zs_inters

                        avg_thick_diff = (
                            np.mean(Zs_diff)
                            * asc_file.nx
                            * asc_file.ny
                            / np.count_nonzero(Zs_inters)
                        )
                        std_thick_diff = (
                            np.std(Zs_diff)
                            * asc_file.nx
                            * asc_file.ny
                            / np.count_nonzero(Zs_inters)
                        )
                        vol_diff = avg_thick_diff * area_inters

                        rel_err_vol = vol_diff / np.maximum(Zs1_vol, Zs2_vol)

                        print("Thickness relative error", rel_err_vol)
                        print("--------------------------------")

            output_dist = input.run_name + "_dist_full.asc"

            # ST skipped this (and conjugated masked) to save up disk
            # space (poorly used so far):
            """
            np.savetxt(output_dist,
                    np.flipud(Zdist),
                    header=header,
                    fmt='%4i',
                    comments='')

            print('')
            print(output_dist + ' saved')

            output_dist = input.run_name + '_dist_masked.asc'

            if (input.masking_threshold < 1):

                Zdist = (1 - masked_Zflow.mask) * Zdist + masked_Zflow.mask * 0

                np.savetxt(output_dist,
                        np.flipud(Zdist),
                        header=header,
                        fmt='%4i',
                        comments='')

                print('')
                print(output_dist + ' saved')

            """

            if input.hazard_flag:
                output_haz = input.run_name + "_hazard_full.asc"

                np.savetxt(
                    output_haz,
                    np.flipud(Zhazard),
                    header=header,
                    fmt="%1.5f",
                    comments="",
                )

                print("")
                print(output_haz + " saved")

                for i_thr in range(n_masking):
                    if input.masking_threshold[i_thr] < 1:
                        max_Zhazard = int(np.floor(np.max(Zhazard)))

                        total_Zflow = np.sum(Zflow)

                        # for i in range(1,max_Zhazard):
                        for i in np.unique(Zhazard):
                            masked_Zflow = ma.masked_where(Zhazard < i, Zflow)

                            if input.flag_threshold == 1:
                                volume_fraction = np.sum(masked_Zflow) / total_Zflow

                                coverage_fraction = volume_fraction

                            else:
                                area_fraction = np.true_divide(
                                    np.sum(masked_Zflow > 0), np.sum(Zflow > 0)
                                )

                                coverage_fraction = area_fraction

                            if coverage_fraction < input.masking_threshold:
                                break

                        output_haz_masked = (
                            input.run_name
                            + "_hazard_masked"
                            + "_"
                            + str(input.masking_threshold[i_thr]).replace(".", "_")
                            + ".asc"
                        )

                        np.savetxt(
                            output_haz_masked,
                            np.flipud((1 - masked_Zflow.mask) * Zhazard),
                            header=header,
                            fmt="%1.5f",
                            comments="",
                        )

                        print("")
                        print(output_haz_masked + " saved")

            # this is to save an additional output for the cumulative deposit,
            # if restart_files is not empty load restart files (if existing)
            if len(input.restart_files) > 0:
                for i_restart in range(0, len(input.restart_files)):
                    Zflow_old = np.zeros((asc_file.ny, asc_file.nx))

                    input.source = input.restart_files[i_restart]

                    file_exists = exists(input.source)
                    if not file_exists:
                        print(input.source + " not found.")
                        quit()

                    hdr = [getline(input.source, i) for i in range(1, 7)]
                    try:
                        values = [float(h.split(" ")[-1].strip()) for h in hdr]
                    except ValueError:
                        print("An problem occurred with header of file ", input.source)
                        print(hdr)

                    nd = values[5]

                    # Load the previous flow thickness into a numpy array
                    arr = np.loadtxt(input.source, skiprows=6)
                    arr[arr == nd] = 0.0

                    Zflow_old = np.flipud(arr)

                    if input.crop_flag:
                        Zflow_old = Zflow_old[
                            asc_file.jS : asc_file.jN, asc_file.iW : asc_file.iE
                        ]

                    Zflow = Zflow + Zflow_old

                output_full = input.run_name + "_thickness_cumulative.asc"

                np.savetxt(
                    output_full,
                    np.flipud(Zflow),
                    header=header,
                    fmt="%1.5f",
                    comments="",
                )

                output_thickness_cumulative = (
                    input.run_name + "_thickness_cumulative.asc"
                )

                print("")
                print(output_thickness_cumulative + " saved")


def main():
    input = parse_input()
    mrlavaloba = MrLavaLoba(input)
    mrlavaloba.run()


if __name__ == "__main__":
    main()
