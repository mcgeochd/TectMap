import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colours

def contourMap(xs, ys, vs, levels, cmap, fig=None, ax=None):
    if fig == None: fig, ax = plt.subplots()
    im = ax.contourf(xs, ys, vs, levels, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_aspect(1/1.2)
    fig.tight_layout()
    if fig == None: plt.show()

def colourAndVectorMap(xs, ys, cs, vxs, vys, us, vs, cmap, fig=None, ax=None):
    if fig == None: fig, ax = plt.subplots()
    im = ax.pcolormesh(xs, ys, cs, rasterized=True, cmap=cmap, shading='auto')
    quiv = ax.quiver(vxs, vys, us, vs)
    fig.colorbar(im, ax=ax)
    ax.set_aspect(1/1.2)
    fig.tight_layout()
    if fig == None: plt.show()

def colourScaleMap(xs, ys, vs, cmap):
    fig, ax = plt.subplots()
    im = ax.pcolormesh(xs, ys, vs, rasterized=True, cmap=cmap, shading='auto')
    fig.colorbar(im, ax=ax)
    ax.set_aspect(1/1.2)
    fig.tight_layout()
    plt.show()

def terrainMap(xs, ys, es, min_h, max_h, fig=None, ax=None):
    if fig == None: fig, ax = plt.subplots()
    colours_undersea = plt.cm.gist_earth(np.linspace(0, 0.17, 256))
    colours_land = plt.cm.gist_earth(np.linspace(0.25, 1, 256))
    all_colours = np.vstack((colours_undersea, colours_land))
    terrain_map = colours.LinearSegmentedColormap.from_list('terrain_map', all_colours)
    # make the norm:  Note the center is offset so that the land has more
    # dynamic range:
    divnorm = colours.TwoSlopeNorm(vmin=min_h, vcenter=0, vmax=max_h)
    im = ax.pcolormesh(xs, ys, es, rasterized=True, norm=divnorm, cmap=terrain_map, shading='gouraud')
    #cf = ax1.contour(xs, ys, es, colors=['red'], levels=[0,])
    #ax2.plot(xs, ys, 'o')
    fig.colorbar(im, ax=ax)
    ax.set_aspect(1/1.2)
    fig.tight_layout()
    if fig == None: plt.show()

def pointCloud(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in zip(xs, ys, zs): ax.scatter(x, y, z, c='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()