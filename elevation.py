import numpy as np
import utils
import elevation
from noise import perlin

boundaryBearings = {}
with open(file="C:\\Users\\domin\\Documents\\Projects\\Python\\3D Planet Git Ready\\boundaryBearings.csv", encoding='utf-8-sig') as f:
    for l in f.readlines(): 
        arr = l.split(',')
        for i in range(1,len(arr)):
            if arr[i] == '\n' or arr[i] == '': continue
            boundaryBearings[int(arr[i])] = float(arr[0])

def jumpFlood(seeds, XYZ):
    shape = XYZ.shape[:2]
    voronoi = np.full(shape, -1)
    borders = np.full_like(voronoi, 0)
    seedCoords = []
    for s in range(seeds):
        i = np.random.randint(0, shape[0]-1)
        j = np.random.randint(0, shape[1]-1)
        seedCoords.append((i, j))
        voronoi[i,j] = s
    N = shape[0]
    # We're doing jump flood +1 to reduce errors
    # Extra step is so borders is correct
    steps = int(np.log2(N))+2
    # Start jumping
    for step in range(steps):
        jump = int(np.ceil(2**(np.log2(N)-1-step)))
        for i in range(shape[0]):
            for j in range(shape[1]):
                cardinals = []
                bestDistance = 9999
                bestSeed = -1
                last = step == steps-1
                for v in range(-1, 2):
                    inew = i+v*jump
                    if inew < 0 or inew >= shape[0]:
                        continue
                    for u in range(-1, 2):
                        jnew = (j+u*jump)%shape[1]
                        sampleSeed = voronoi[inew,jnew]
                        if (sampleSeed == -1):
                            continue
                        distance = dist((i, j), seedCoords[sampleSeed], XYZ)
                        if distance < bestDistance:
                            bestSeed = sampleSeed
                            bestDistance = distance
                        if u*v == 0 and u+v != 0:
                            cardinals.append(sampleSeed)
                voronoi[i,j] = bestSeed
                if last:
                    neighbours_set = set(cardinals)
                    if len(neighbours_set) >= 2: borders[i,j] = 1
    return voronoi, borders, seedCoords

def dist(p1, p2, XYZ):
    xyz1 = XYZ[p1]
    xyz2 = XYZ[p2]
    return -np.dot(xyz1, xyz2)/(np.dot(xyz1, xyz1)*np.dot(xyz2,xyz2))

def updateSeed(voronoi, seedCoords, point, newSeed, Xs, Ys, Zs):
    seed = voronoi[point]
    if seed == -1 or dist(point, seedCoords[seed], Xs, Ys, Zs) > dist(point, seedCoords[newSeed], Xs, Ys, Zs):
        voronoi[point] = newSeed
        return newSeed
    return seed

def dijkstraMap(voronoi, goals, seedCoords, walls=[], bounds=[], relaxCentres=True):
    shape = voronoi.shape
    goals = (1-goals)*1e9
    dijkstra = np.zeros_like(voronoi, dtype=int)
    if len(bounds) != 0:
        minI, minJ = bounds[0], bounds[1]
        maxI, maxJ = bounds[2]+1, bounds[3]+1
    else:
        minI, minJ = 0, 0
        maxI, maxJ = shape
    # print (minI, maxI, minJ, maxJ)
    while True:
        changed = False
        for i in range(minI, maxI):
            for j in range(minJ, maxJ):
                if voronoi[i,j] in walls: continue
                lowest = 1e10
                if i - 1 >= 0 and goals[i-1,j] < lowest:
                    # top
                    lowest = goals[i-1,j]
                if goals[i,(j-1)%shape[1]] < lowest:
                    # left
                    lowest = goals[i,(j-1)%shape[1]]
                if goals[i,(j+1)%shape[1]] < lowest:
                    # right
                    lowest = goals[i,(j+1)%shape[1]]
                if i + 1 < shape[0] and goals[i+1,j] < lowest:
                    # bottom
                    lowest = goals[i+1,j]
                if goals[i,j] > lowest + 1:
                    goals[i,j] = lowest + 1
                    dijkstra[i,j] = lowest + 1
                    changed = True
                    if relaxCentres:
                        seedCoords[voronoi[i,j]] = (i,j)
        if not changed:
            break
    return dijkstra

def findBounds(voronoi, seeds):
    shape = voronoi.shape
    seeds_seen = []
    bounds = np.empty((seeds, 4), dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            seed = voronoi[i,j]
            if seed not in seeds_seen:
                seeds_seen.append(seed)
                bounds[seed,0] = i # up left i
                bounds[seed,1] = j # up left j
                bounds[seed,2] = i # low right i
                bounds[seed,3] = j # low right j
            else:
                # first encountered i will always be lowest by definition
                if i > bounds[seed,2]:
                    bounds[seed,2] = i
                if j < bounds[seed,1]:
                    bounds[seed,1] = j
                if j > bounds[seed,3]:
                    bounds[seed,3] = j
    return bounds

def voronoiPlateBoundaries(voronoi, borders, seedCoords, seeds, Lat, Lon, strikeSlipHalfAngle = 10):
    # Setup
    R = 6371e3 # m
    shape = voronoi.shape
    seedLatLons = [(Lat[i,j], Lon[i,j]) for (i, j) in seedCoords]
    # Assign plate velocities
    velocitiesRT = []
    velocitiesUV = []
    for i in range(seeds):
        # plate velocity in m/yr
        R = 1 # (np.random.random()*(10 - 1) + 1) * 0.01
        theta = np.random.random()*360
        velocitiesRT.append(np.array([R, theta]))
        velocitiesUV.append(np.array([R*np.sin(theta*utils.degToRad), R*np.cos(theta*utils.degToRad)]))
    boundaries = np.full_like(voronoi, -1, dtype=float)
    collisionIndex = np.full_like(voronoi, -1, dtype=int)
    relativeSpeed = np.zeros_like(voronoi, dtype=float)
    # by convention i,j will always be clockwise / on the rhs of the boundary bearing
    for i in range(shape[0]):
        for j in range(shape[1]):
            if borders[i,j] == 0: continue
            ourplate = voronoi[i,j]
            neighbours = []
            #ourdiv = divergence[i,j]
            # left
            flag = 0
            theirplate = voronoi[i,(j-1)%shape[1]]
            if theirplate != ourplate:
                neighbours.append(theirplate)
                flag |= (1<<6)
            # bottom
            if i + 1 < shape[0]:
                theirplate = voronoi[i+1,j]
                if theirplate != ourplate:
                    neighbours.append(theirplate)
                    flag |= (1<<4)
                # bottom left
                theirplate = voronoi[i+1,(j-1)%shape[1]]
                if theirplate != ourplate:
                    neighbours.append(theirplate)
                    flag |= (1<<5)
                # bottom right
                theirplate = voronoi[i+1,(j+1)%shape[1]]
                if theirplate != ourplate:
                    neighbours.append(theirplate)
                    flag |= (1<<3)
            # right
            theirplate = voronoi[i,(j+1)%shape[1]]
            if theirplate != ourplate:
                neighbours.append(theirplate)
                flag |= (1<<2)
            # top
            if i - 1 >= 0:
                theirplate = voronoi[i-1,j]
                if theirplate != ourplate:
                    neighbours.append(theirplate)
                    flag |= 1
                # top left
                theirplate = voronoi[i-1,(j-1)%shape[1]]
                if theirplate != ourplate:
                    neighbours.append(theirplate)
                    flag |= (1<<7)
                # top right
                theirplate = voronoi[i-1,(j+1)%shape[1]]
                if theirplate != ourplate:
                    neighbours.append(theirplate)
                    flag |= (1<<1)
            if not neighbours: continue
            mostNeighbour = max(set(neighbours), key=neighbours.count)
            theirVelocityBearing = velocitiesRT[mostNeighbour][1]
            ourVelocityBearing = velocitiesRT[ourplate][1]
            try:
                boundaryBearing = boundaryBearings[flag]
            except:
                continue
            collisionIndex[i,j] = mostNeighbour

            ourDeltaBearing = ourVelocityBearing - boundaryBearing + 360 if ourVelocityBearing - boundaryBearing < 0 else ourVelocityBearing - boundaryBearing
            if ourDeltaBearing > 180: ourDeltaBearing = ourDeltaBearing - 360
            theirDeltaBearing = theirVelocityBearing - boundaryBearing + 360 if theirVelocityBearing - boundaryBearing < 0 else theirVelocityBearing - boundaryBearing
            if theirDeltaBearing > 180: theirDeltaBearing = theirDeltaBearing - 360

            boundaryPerpVec = np.array([np.sin((boundaryBearing+90)*utils.degToRad), np.cos((boundaryBearing+90)*utils.degToRad)])
            relativeSpeed[i,j] = np.dot(velocitiesUV[ourplate], boundaryPerpVec)
            if abs(ourDeltaBearing) < strikeSlipHalfAngle and abs(theirDeltaBearing) < strikeSlipHalfAngle:
                # strike slip
                boundary = 2
            elif ourDeltaBearing < 0 and theirDeltaBearing > 0:
                # converging
                boundary = 0
            elif ourDeltaBearing > 0 and theirDeltaBearing < 0:
                # diverging
                boundary = 1
            else:
                # chasing/running
                ourProj = relativeSpeed[i,j]
                theirProj = np.dot(velocitiesUV[mostNeighbour], boundaryPerpVec)
                if ourProj > theirProj:
                    boundary = 1
                else:
                    boundary = 0

            boundaries[i,j] = boundary
            
    return boundaries, velocitiesUV, np.abs(relativeSpeed), collisionIndex

def voronoiBoundaryDistances(voronoi, boundaries, seeds, velocitiesUV):
    seedsArr = np.arange(0,seeds)
    divergent = np.ma.filled(np.ma.masked_not_equal(boundaries, 1), 0).astype(int)
    convergent = np.ma.filled(np.ma.masked_not_equal(boundaries+1, 1), 0).astype(int)
    trenchDistance = np.zeros_like(voronoi, dtype=float)
    mountainDistance = np.zeros_like(voronoi, dtype=float)
    for i in range(seeds):
        walls = seedsArr[seedsArr!=i]
        mask = np.ma.masked_equal(voronoi, i).mask
        # TODO Another dijkstra implementation that can do both in one pass?
        trenchDijkstra = dijkstraMap(voronoi, divergent, None, walls=walls, relaxCentres=False)
        mountainDijkstra = dijkstraMap(voronoi, convergent, None, walls=walls, relaxCentres=False)
        speed = utils.arrayMagnitude(velocitiesUV[i])
        trenchDistance += trenchDijkstra*mask / speed
        mountainDistance += mountainDijkstra*mask / speed
    return trenchDistance, mountainDistance

def voronoiPlateHeight(voronoi, divergent, convergent, Lat, Lon, R, min_h, mean_sea_h, max_h, alpha=0.5, beta=0.05, seaProp=0.7, freq=0.5, octs=6):
    areaProportions = utils.areaProportions(Lat, Lon, R)
    # TODO plates that don't share a convergent boundary with each other
    oceanPlateIDs = [4]
    oceanPlateMask = np.zeros_like(voronoi, dtype=int)
    for i in oceanPlateIDs:
        oceanPlateMask += np.ma.masked_equal(voronoi, i).mask
    noise = 2*np.array(elevation.noiseMap().getNoise(Lat[:,0], Lon[0,:], freq, octs))-1
    # sea level
    height0 = divergent**alpha - convergent**alpha
    seaLevel = findSeaLevel(height0, seaProp, areaProportions)
    height0 -= seaLevel
    # masks
    oceanMask = np.ma.masked_less(height0, 0).mask
    allOceanMask = oceanMask | oceanPlateMask
    # land height
    land = height0*(1-allOceanMask)
    landn = land/np.amax(land)
    landh = landn*max_h
    # oceanic crust on plates with continent height
    ocean = height0*oceanMask
    oceann = ocean/np.amin(ocean)
    oceann = 1 - (2*(oceann-0.5))**2
    oceanh = oceann*mean_sea_h*oceanMask
    # oceanic crust on plates without continent height
    oceanic = height0*oceanPlateMask
    oceanicMin = np.amin(oceanic)
    oceanicn = (oceanic-oceanicMin)/(np.amax(oceanic) - oceanicMin)
    ridgeSide = 2*oceanicn**2
    trenchSide = 2*(oceanicn-0.5)**2+0.5
    ridgeMask = np.ma.masked_less_equal(oceanicn, 0.5).mask
    oceanicn = ridgeSide*ridgeMask + trenchSide*(1-ridgeMask)
    oceanich = oceanicn*min_h*oceanPlateMask
    # total height map
    height1 = landh + oceanh + oceanich
    height1 += noise * beta * (max_h - min_h)
    
    return height1, height0

def findSeaLevel(height, seaProp, areaProportions):
    hist, bin_edges = np.histogram(height, weights=areaProportions, density=True)
    hist_sum = sum(hist)
    count = 0
    seaLevel = 0
    for i in range(len(hist)):
        count += hist[i]
        if count >= seaProp*hist_sum:
            seaLevel = (bin_edges[i] + bin_edges[i-1]) / 2
            break
    return seaLevel

def voronoiPlateAreas(voronoi, seeds, areaProportions):
    shape = voronoi.shape
    areas = np.zeros((seeds), dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            p = voronoi[i,j]
            areas[p] += areaProportions[i,j]
    return areas

class noiseMap:

    def __init__(self):
        self.simplex = perlin.SimplexNoise()
        self.simplex.randomize()

        self.heights = []
        with open(file="C:\\Users\\domin\\Documents\\Projects\\Python\\3D Planet\\hypso2.txt") as f:
            for l in f.readlines(): self.heights.extend(l.split())

    def fBm(self, x, y, z, freq, octs):
        freq = 1/freq
        x *= freq
        y *= freq
        z *= freq
        val = 0
        for o in range(octs):
            mod = 2**o
            val += 0.5**o * self.simplex.noise3(x*mod, y*mod, z*mod)
        return val

    def hypso(self, x, max_height, min_height):
        index = int(x*(len(self.heights)-1))
        h = float(self.heights[index])
        h *= max_height-min_height
        h += min_height
        return h

    def getNoise(self, lats, lons, freq, octs):
        # Sample 3D simplex noise from a unit sphere surface
        ns = []
        for lat in lats:
            row = []
            for lon in lons:
                xv, yv, zv = utils.latLonRToXYZ(lat, lon, 1)
                n = self.fBm(xv, yv, zv, freq, octs)
                row.append(n)
            ns.append(row)
        # Normalise the noise
        ns = (ns - np.min(ns))/(np.max(ns)-np.min(ns))
        return ns

    def getElevation(self, lats, lons, max_h, min_h):
        ns = self.getNoise(lats, lons)
        # Apply the hypsometric curve
        es = []
        for i in range(len(lats)):
            row = []
            for j in range(len(lons)):
                row.append(self.hypso(ns[i][j], max_h, min_h))
            es.append(row)
        E = np.array(es)
        return E