from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Define RPP
def RPP(x, a, b):
    return (b**a) / gamma(a) * (x**(a-1)) * np.exp(-b * x)

# Define RPP_c
def RPP_c(x, a, b, c):
    return c*(b**(a/c))/gamma(a/c)*(x**(a-1))*np.exp(-b*(x**c))

# Define parameters for the run
N = 80
frames = 10000
high = [1024, 1024]
all_areas = []
all_normalized_areas = []
popt_a = []
popt_b = []
popt_ca = []
popt_cb = []
popt_cc = []

# Run simulation
for f in tqdm(range(frames)): 
    # Create random data 
    data = np.random.uniform(low=[0, 0], high=high, size=(N, 2))
    
    # Create Delaunay object
    dela = Delaunay(data)
    vertices = data[dela.simplices]

    # Get triangle areas
    areas_per_frame = []
    for i, vertex in enumerate(vertices):
        to_det = np.concatenate((vertex, np.ones((1, np.size(vertex, axis=0))).T), axis=1)
        area = 0.5 * np.linalg.det(to_det)
        
        all_areas.append(area)
        areas_per_frame.append(area)
    
    all_normalized_areas.append(areas_per_frame / np.mean(areas_per_frame))

    # Save some of the variables to check convergence
    if (f % 100 == 0): 
        # Ravel all the np array
        concatenated_normalized_areas = np.concatenate(all_normalized_areas).ravel()
        
        # Compile data and fit RPP
        bins = np.geomspace(1e-2, 12, 200)
        count, _ = np.histogram(concatenated_normalized_areas, bins=bins, density=True)
        
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(RPP, bins[:-1], count)

        popt_a.append(popt[0])
        popt_b.append(popt[1])
        
        popt_c, pcov_c = curve_fit(RPP_c, bins[:-1], count)
        
        popt_ca.append(popt_c[0])
        popt_cb.append(popt_c[1])
        popt_cc.append(popt_c[2])

print(popt)
print(popt_c)

plt.figure()
plt.loglog(bins[:-1], count, '.-')
plt.loglog(bins, RPP(bins, *popt), '.-')
plt.loglog(bins, RPP_c(bins, *popt_c), '.-')
plt.grid()

plt.figure()
plt.plot(np.abs(np.diff(popt_a)), '.-', label='a')
plt.plot(np.abs(np.diff(popt_b)), '.-', label='b')
plt.grid()
plt.legend()

plt.figure()
plt.plot(np.abs(np.diff(popt_a)), '.-', label='a')
plt.plot(np.abs(np.diff(popt_b)), '.-', label='b')
plt.plot(np.abs(np.diff(popt_c)), '.-', label='c')
plt.grid()
plt.legend()

plt.show()
