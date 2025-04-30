import json
import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983 as NOJ

class CoastalVirginiaWindVisualizer:
    def __init__(self, boundary_file="Coastal_Virginia_polygon.geojson", turbines_file="Coastal_Virginia_point.geojson"):
        self.boundary_file = boundary_file
        self.turbines_file = turbines_file
        self.boundary_array = self._load_boundary()
        self.turbines_array = self._load_turbines()

    def _load_boundary(self):
        with open(self.boundary_file, 'r') as f:
            geojson = json.load(f)
        for feature in geojson["features"]:
            if feature["geometry"]["type"] == "Polygon":
                coords = feature["geometry"]["coordinates"]
                if isinstance(coords[0][0], (float, int)):
                    return np.array(coords)
                else:
                    return np.array(coords[0])
        return np.array([])

    def _load_turbines(self):
        with open(self.turbines_file, 'r') as f:
            geojson = json.load(f)
        points = []
        for feature in geojson["features"]:
            if feature["geometry"]["type"] == "Point":
                x, y = feature["geometry"]["coordinates"]
                points.append([x, y])
        return np.array(points)

    def get_arrays(self):
        return self.boundary_array, self.turbines_array

    def plot_layout_with_aep(self, aep_value):
        plt.figure(figsize=(10, 8))
        if self.boundary_array.ndim == 2:
            plt.plot(self.boundary_array[:, 0], self.boundary_array[:, 1], 'b-', label='Boundary')
        plt.scatter(self.turbines_array[:, 0], self.turbines_array[:, 1], color='darkorange', label='Turbines', zorder=5)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Coastal Virginia Wind Farm Layout\nTotal AEP: {aep_value:.2f} GWh/year")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

# ---- Main Execution ----

if __name__ == "__main__":
    visualizer = CoastalVirginiaWindVisualizer()
    boundary, turbines = visualizer.get_arrays()

    x = turbines[:, 0]
    y = turbines[:, 1]

    windTurbines = V80()
    site = Hornsrev1Site()
    noj = NOJ(site, windTurbines)

    simulationResult = noj(x, y)
    aep = simulationResult.aep().sum()

    print(f"Total AEP (GWh/year): {aep:.2f}")
    visualizer.plot_layout_with_aep(aep)
