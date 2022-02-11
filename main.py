import numpy as np
import matplotlib.pyplot as plt
import projections
import elevation
import visualisation
import utils

time = utils.currentTime()
time0 = time
dlat = 4
dlon = 4
max_h = 8000
mean_sea_h = -4000
min_h = -11000
lats = np.arange(90-dlat/2., -90-dlat/2., -dlat)
lons = np.arange(-180+dlon/2., 180+dlon/2., dlon)
R = 6371 # km
Lat, Lon = np.meshgrid(lats, lons, indexing='ij')
Xs, Ys, Zs = utils.latLonRToXYZ(Lat, Lon, R)
#X, Y = projections.winkelTripel(Lat, Lon)

seeds = 8
alpha = 0.5
time = utils.intervalMessage(time, "setup:")
voronoi, borders, seedCoords = elevation.jumpFlood(seeds, Xs, Ys, Zs)
time = utils.intervalMessage(time, "jump flood voronoi:")
dijkstra = elevation.dijkstraMap(voronoi, borders, seedCoords)
time = utils.intervalMessage(time, "dijkstra:")
boundaries, velocitiesUV, relativeSpeed, colIndex = elevation.voronoiPlateBoundaries(voronoi, borders, seedCoords, seeds, Lat, Lon, strikeSlipHalfAngle=0)
time = utils.intervalMessage(time, "plate boundaries:")
divergent, convergent = elevation.voronoiBoundaryDistances(voronoi, boundaries, seeds, velocitiesUV)
time = utils.intervalMessage(time, "boundary distances:")
height, rawHeight = elevation.voronoiPlateHeight(voronoi, divergent, convergent, Lat, Lon, R, min_h, mean_sea_h, max_h, alpha=alpha)
time = utils.intervalMessage(time, "plate height:")
vys = [Lat[i,j] for (i,j) in seedCoords]
vxs = [Lon[i,j] for (i,j) in seedCoords]
us = [v[0] for v in velocitiesUV]
vs = [v[1] for v in velocitiesUV]
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
visualisation.colourAndVectorMap(Lon, Lat, boundaries, vxs, vys, us, vs, 'rainbow', fig=fig, ax=ax0)
visualisation.terrainMap(Lon, Lat, height, min_h, max_h, fig=fig, ax=ax1)
visualisation.colourAndVectorMap(Lon, Lat, rawHeight, vxs, vys, us, vs, 'rainbow', fig=fig, ax=ax2)
ax1.contour(Lon, Lat, height, colors='black', levels=[0,])
ax2.contour(Lon, Lat, rawHeight, colors='black', levels=[0,])
ax0.set_title("boundaries")
ax1.set_title("height")
ax2.set_title("raw height")
utils.intervalMessage(time, "plotting:")
utils.intervalMessage(time0, "total time:")
plt.show()