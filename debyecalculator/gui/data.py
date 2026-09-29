"""
Experimental data for comparison with calculated patterns: loading of two-column text files and a
least-squares scale fit with difference curve and Rw.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DATA_SUFFIXES = ('.gr', '.fq', '.sq', '.iq', '.xy', '.xye', '.chi', '.dat', '.csv', '.txt')

# Function guessed from the file suffix; everything else is taken as I(Q)
SUFFIX_FUNCTIONS = {'.gr': 'g', '.fq': 'f', '.sq': 's'}


@dataclass
class Comparison:
    scale: float
    x: np.ndarray            # data x within the overlap with the calculation
    difference: np.ndarray   # data - scale * calculation, at x
    rw: float


def guess_function(path: str) -> str:
    return SUFFIX_FUNCTIONS.get(Path(path).suffix.lower(), 'i')


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the first two numeric columns of a text file. Lines that do not start with two numbers (headers,
    comments, metadata blocks such as in PDFgetX3 .gr files) are skipped. Columns may be separated by
    whitespace, commas or semicolons.
    """
    xs, ys = [], []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            fields = line.replace(',', ' ').replace(';', ' ').split()
            if len(fields) < 2:
                continue
            try:
                x, y = float(fields[0]), float(fields[1])
            except ValueError:
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    if len(xs) < 2:
        raise ValueError(f'No two-column numeric data found in {Path(path).name}')
    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x, kind='stable')
    return x[order], y[order]


def two_theta_to_q(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
    return 4 * np.pi * np.sin(np.radians(two_theta) / 2) / wavelength


def q_to_two_theta(q: np.ndarray, wavelength: float) -> np.ndarray:
    """
    2θ in degrees; NaN where Q is beyond the reach of the wavelength (Qλ/4π > 1).
    """
    argument = np.asarray(q) * wavelength / (4 * np.pi)
    with np.errstate(invalid='ignore'):
        return np.where(np.abs(argument) <= 1, np.degrees(2 * np.arcsin(np.clip(argument, -1, 1))), np.nan)


def compare(data_x: np.ndarray, data_y: np.ndarray, calc_x: np.ndarray, calc_y: np.ndarray,
            fit_scale: bool = True) -> Optional[Comparison]:
    """
    Compare data with a calculation on the data points inside the calculated range. With fit_scale, the
    calculation is scaled by the least-squares factor sum(d m) / sum(m m); otherwise the scale is 1.
    Rw = sqrt(sum((d - s m)^2) / sum(d^2)).
    """
    inside = (data_x >= np.min(calc_x)) & (data_x <= np.max(calc_x))
    if np.count_nonzero(inside) < 2:
        return None
    x, d = data_x[inside], data_y[inside]
    m = np.interp(x, calc_x, calc_y)
    scale = float(np.dot(d, m) / np.dot(m, m)) if fit_scale and np.dot(m, m) > 0 else 1.0
    difference = d - scale * m
    norm = np.dot(d, d)
    rw = float(np.sqrt(np.dot(difference, difference) / norm)) if norm > 0 else float('nan')
    return Comparison(scale=scale, x=x, difference=difference, rw=rw)
