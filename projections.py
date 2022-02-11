import numpy as np
from typing import Iterable
from utils import degToRad

def sinc(xs):
    if isinstance(xs, Iterable):
        return [sinc(x) for x in xs]
    if xs != 0:
        return np.sin(xs)/xs
    else:
        return 1.0

def winkelTripel(Lat, Lon):
    LatR = Lat*degToRad
    LonR = Lon*degToRad
    A = np.arccos(np.cos(LatR)*np.cos(LonR/2.))
    X = 0.5*(LonR*(2./np.pi)+2*np.cos(LatR)*np.sin(LonR/2.)/sinc(A))
    Y = 0.5*(LatR+np.sin(LatR)/sinc(A))
    return X, Y

def Gnomonic(Lat, Lon, lat0, lon0):
    # https://mathworld.wolfram.com/GnomonicProjection.html
    LatR = Lat*degToRad
    LonR = Lon*degToRad
    lat0r = lat0*degToRad
    lon0r = lon0*degToRad
    C = np.arccos(np.sin(lat0r)*np.sin(LatR)+np.cos(lat0r)*np.cos(LatR)*np.cos(LonR-lon0r))
    X = np.cos(LatR)*np.sin(LonR-lon0r)/np.cos(C)
    Y = (np.cos(lat0r)*np.sin(LatR)-np.sin(lat0r)*np.cos(LatR)*np.cos(LonR-lon0r))/np.cos(C)
    return X, Y