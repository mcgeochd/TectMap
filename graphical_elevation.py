import panda3d.core as pc
from direct.showbase.ShowBase import ShowBase
from PIL import Image
import numpy as np
import utils

texW = 1024
texH = 512

base = ShowBase()
# Request 8 RGB bits, no alpha bits, and a depth buffer.
fb_prop = pc.FrameBufferProperties()
fb_prop.setRgbColor(True)
fb_prop.setRgbaBits(8, 8, 8, 0)
fb_prop.setDepthBits(16)

# Create a WindowProperties object set to size.
win_prop = pc.WindowProperties(size=(texW, texH))

# Don't open a window - force it to be an offscreen buffer.
flags = pc.GraphicsPipe.BF_refuse_window

# Create a GraphicsBuffer to render to, we'll get the textures out of this
buffer1 = base.graphicsEngine.make_output(base.pipe, "Buffer1", -100, fb_prop, win_prop, flags, base.win.getGsg(), base.win)

# Create a camera, a card to render to, and a display region for them
cm = pc.CardMaker('card')
card = base.render2d.attachNewNode(cm.generate())
region = buffer1.makeDisplayRegion()
cam2D = pc.NodePath(pc.Camera('cam'))
lens = pc.OrthographicLens()
lens.setFilmSize(2, 2)
lens.setNearFar(-1000, 1000)
cam2D.node().setLens(lens)
base.render2d.setDepthTest(False)
base.render2d.setDepthWrite(False)
cam2D.reparentTo(base.render2d)
region.setCamera(cam2D)

# Load the shader steps
voronoiShader = pc.Shader.load(pc.Shader.SL_GLSL, vertex="quad.vert", fragement="voronoi.frag")

# Generate cone triangle fan
# def genCone(halfAngle=10, height=1, subDivs=32):
#     radius = height*np.tan(halfAngle*utils.degToRad)
#     vertices = np.array([[0, 0, 0]])
#     indices = np.array([0])
#     for i in range(subDivs):
#         theta = i*360*utils.degToRad/subDivs
#         vertex = np.array([radius*np.sin(theta), radius*np.cos(theta), -height])
#         vertices.append(vertex)
#         indices.append(i+1)
#     return vertices, indices

# def jumpFlood(seeds: int, XYZ: np.ndarray):
#     # Seeds
#     texArr = np.zeros((512, 1024, 3), dtype=bytes)
#     for seed in range(min(seeds, 255)):
#         i = np.random.randint(0, 512)
#         j = np.random.randint(0, 1024)
#         texArr[i,j,0] = 1 + seed
#     tex = pc.Texture("jump-flood")
#     tex.setup2dTexture(1024, 512, pc.Texture.T_unsigned_byte, pc.Texture.F_rgb8)
#     tex.setRamImageAs(texArr, 'RGB')
#     # XYZ
#     XYZ = (XYZ + 1) * 0.5 * 255
#     xyzArr = XYZ.astype(dtype=np.dtype('B'))
#     sphereXYZ = pc.Texture("sphereXYZ")
#     sphereXYZ.setup2dTexture(1024, 512, pc.Texture.T_unsigned_byte, pc.Texture.F_rgb32)
#     sphereXYZ.setRamImageAs(xyzArr, 'RGB')

#     # Create a dummy node and apply the shader to it
#     shader = pc.Shader.load_compute(pc.Shader.SL_GLSL, "compute.glsl")
#     dummy = pc.NodePath("dummy")
#     dummy.set_shader(shader)
#     dummy.set_shader_input("tex", tex)
#     dummy.set_shader_input("sphereXYZ", sphereXYZ)

#     # Retrieve the underlying ShaderAttrib
#     sattr = dummy.get_attrib(pc.ShaderAttrib)

#     # Dispatch the compute shader, right now!
#     for level in range(10):
#         dummy.set_shader_input("level", float(level))
#         base.graphicsEngine.dispatch_compute((1, 1, 1), sattr, base.win.get_gsg())

#     return tex

def storeTextureAsImage(texture: pc.Texture, mode: str, filename: str):
    data = np.array(texture.getRamImageAs(mode))
    im = Image.frombytes(mode, (texW, texH), data)
    im.save(filename+'.png')

dlat = 180/512.
dlon = 360/1024.
lats = np.linspace(90-dlat/2., -90-dlat/2., 512)
lons = np.linspace(-180+dlon/2., 180+dlon/2., 1024)
Lat, Lon = np.meshgrid(lats, lons, indexing='ij')
Xs, Ys, Zs = utils.latLonRToXYZ(Lat, Lon, 1)
# Stack and remove sign
XYZ = np.dstack((Xs, Ys, Zs))

# tex = jumpFlood(8, XYZ)
# storeTextureAsImage(tex, 'RGB', 'jumpflood')