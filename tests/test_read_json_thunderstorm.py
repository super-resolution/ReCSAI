from src.utility import read_thunderstorm_drift
import json
from scipy.interpolate import interp1d,BSpline,CubicSpline
import numpy as np
import matplotlib.pyplot as plt
# test_file = r"D:\Daten\Dominik_B\Values.csv"
# drift = read_thunderstorm_drift(test_file)
# x=0

with open(r"D:\Daten\Dominik_B\drift.json", 'r') as f:
  data = json.load(f)
def get_knots_drift(name):
    knots = data[name]["knots"]
    drift = []
    polynom = data[name]['polynomials']

    for poly in polynom:
        coeff = poly["coefficients"]
        drift.append(coeff[0])
    drift.append(coeff[0] + coeff[1] * (knots[-1] - knots[-2] - 1))
    return knots, drift

knots_x, drift_x = get_knots_drift("xFunction")

x = np.arange(knots_x[0],knots_x[-1]+1)
poly_x = CubicSpline(knots_x, drift_x)
x_drift = poly_x(x)
knots_y, drift_y = get_knots_drift("yFunction")
poly_y = CubicSpline(knots_y, drift_y)
y_drift = poly_y(x)
plt.plot(x, y_drift)
plt.plot(x, x_drift)

plt.show()
