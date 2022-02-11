import numpy as np
import time

degToRad = np.pi/180
radToDeg = 180/np.pi

def currentTime():
    return time.time()

def intervalMessage(initialTime, message):
    t1 = currentTime()
    print(message, round(t1-initialTime, 3))
    return t1

def latLonRToXYZ(lat, lon, r):
    x = r*np.cos(lat*degToRad)*np.cos(lon*degToRad)
    y = r*np.cos(lat*degToRad)*np.sin(lon*degToRad)
    z = r*np.cos((lat-90)*degToRad)
    return x, y, z

def xyzToLatLonR(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z/r)*radToDeg
    lon = np.arctan2(y,x)*radToDeg
    return lat, lon, r

def randomPointOnSphere(r=1, grid=None):
    lat = np.random.random()*180-90
    lon = np.random.random()*360-180
    if grid != None:
        lat -= lat%grid[0]
        lon -= lon%grid[1]
        lat += grid[0]/2.
        lon += grid[1]/2.
    return (lat, lon, r)

def haversine(a):
    return np.sin(a/2.)**2

def normArray(x: np.array):
    return x/arrayMagnitude(x)

def angleBetweenBearings(b1, b2):
    return min((b1-b2+360)%360, (b2-b1+360)%360)

def bearingBetween(lat1, lon1, lat2, lon2):
    lon1r = lon1 * degToRad
    lat1r = lat1 * degToRad
    lon2r = lon2 * degToRad
    lat2r = lat2 * degToRad
    dlonr = lon2r - lon1r
    theta = np.arctan2(np.sin(dlonr)*np.cos(lat2r), np.cos(lat1r)*np.sin(lat2r)-np.sin(lat1r)*np.cos(lat2r)*np.cos(dlonr))
    return (theta*radToDeg + 360) % 360

def rasterize(lat, lon, grid):
    # convert lat and lon into grid coords
    adjlat = lat+90
    adjlon = lon+180
    i = np.clip(int(adjlat/grid[0]), 0, int(180/grid[0])-1)
    j = np.clip(int(adjlon/grid[1]), 0, int(360/grid[1])-1)
    return i, j

def gridPointToLatLon(i, j, rows, cols):
    # cell centers, not edges
    dlat = (180-2*180/rows)/rows
    dlon = (360-2*180/cols)/cols
    lat = (rows - i)*dlat - 90 + 180./rows
    lon = j*dlon - 180 + 360./rows
    return lat, lon

def lonGridSpacing(i, Lat, Lon, R):
    dlon = (Lon[0,1] - Lon[0,0])*degToRad
    r = R*np.cos(Lat[i,0]*degToRad)
    return r*dlon

def latGridSpacing(Lat, R):
    dlat = (Lat[0,0] - Lat[1,0])*degToRad
    return R*dlat

def areaProportions(Lat, Lon, R):
    shape = Lat.shape
    dlat = latGridSpacing(Lat, R)
    arr = np.array([dlat*lonGridSpacing(i, Lat, Lon, R) for i in range(shape[0])])
    return np.tile(arr.reshape(-1,1), (1,shape[1]))

def arrayMagnitude(x: np.array):
    return np.sqrt(np.dot(x, x))

def bearingToQuadrant(bearing):
    # Convention is bearing 0 = quadrant 0, clockwise
    if bearing <= 45: return 0
    if bearing <= 135: return 1
    if bearing <= 225: return 2
    if bearing <= 315: return 3
    if bearing <= 360: return 0

def bearingToOctant(bearing):
    # Convention is bearing 0 = octant 0, clockwise
    if bearing <= 22.5: return 0
    if bearing <= 67.5: return 1
    if bearing <= 112.5: return 2
    if bearing <= 157.5: return 3
    if bearing <= 202.5: return 4
    if bearing <= 247.5: return 5
    if bearing <= 292.5: return 6
    if bearing <= 337.5: return 7
    if bearing <= 360: return 0

def haversineDist(lat1, lon1, lat2, lon2, r):
    lon1r = lon1 * degToRad
    lat1r = lat1 * degToRad
    lon2r = lon2 * degToRad
    lat2r = lat2 * degToRad
    dlatr = lat1r - lat2r
    dlonr = lon1r - lon2r
    a = haversine(dlatr) + np.cos(lat1r)*np.cos(lat2r)*haversine(dlonr)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist = r*c
    return dist

def gCircleDist(xyz1, xyz2):
    mag1 = arrayMagnitude(xyz1)
    mag2 = arrayMagnitude(xyz2)
    costheta = np.dot(xyz1, xyz2)/(mag1*mag2)
    costheta = np.clip(costheta, -1, 1)
    return mag1*np.arccos(costheta)
