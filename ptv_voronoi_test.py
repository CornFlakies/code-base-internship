from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, Voronoi
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
N = 100
frames = 1000
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
    
    vor = Voronoi(data)
    
    regions_not = []
    empty_regions = []
    points_not = []
    regions_yes = []
    points_yes = []
    
    maxDF = 1023
    minDF = 0
    
    # VORONOI VOLUMES 
    for idx_input_point, reg_num in enumerate(vor.point_region):
            indices = vor.regions[reg_num]
            vertices = vor.vertices
            cond = (-1 not in indices) and (all((vertices[indices]<maxDF).ravel())) and (all((vertices[indices]>minDF).ravel()))       
            # there may be an empty region, bounded entirely by points at inifinity
            if not indices:
                    regions_not.append(reg_num)
                    empty_regions.append(reg_num)
                    points_not.append(idx_input_point)
    
            elif cond:
                    regions_yes.append(reg_num)
                    points_yes.append(idx_input_point)
    
            # if -1 in indices then region is closed at infinity
            elif not cond:
                    regions_not.append(reg_num)
                    points_not.append(idx_input_point)
                        
    
    all_regions = len(regions_yes) 
    if all_regions > 0:
        vols = np.zeros(all_regions)
        for nn, reg in enumerate(tqdm(regions_yes, desc='Computing volumes', disable=True, leave=False)):
                idx_vert_reg = vor.regions[reg]
                vols[nn] = ConvexHull(vor.vertices[idx_vert_reg]).volume
        v_vols = len(vols)
        all_normalized_areas.append(vols / np.mean(vols))
        
    if (f % 100 == 0): 
        # Ravel all the np array
        concatenated_normalized_areas = np.concatenate(all_normalized_areas).ravel()
        
        # Compile data and fit RPP
        bins = np.geomspace(1e-1, 12, 200)
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
plt.semilogy(bins[:-1], count, '.-')
#plt.loglog(bins, RPP(bins, *popt), '.-')
#plt.loglog(bins, RPP_c(bins, *popt_c), '.-')
plt.grid()

plt.figure()
plt.plot(np.abs(np.diff(popt_a)), '.-', label='a')
plt.plot(np.abs(np.diff(popt_b)), '.-', label='b')
plt.grid()
plt.legend()
plt.figure()
plt.plot(np.abs(np.diff(popt_a)), '.-', label='a')
plt.plot(np.abs(np.diff(popt_b)), '.-', label='b')
plt.grid()
plt.legend()
plt.show()
