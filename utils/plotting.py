import numpy as np



def set_axes_equal(ax):
    ''' Makes sure the axis of the 3D plot have equal lengths.'''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_colours(srcPts):
	''' Creates a colour scheme for plotting '''
	srcPts = srcPts.reshape(-1, 2)
	blue  = [img1[int(Pt[1]), int(Pt[0]), 0] for Pt in srcPts]
	green = [img1[int(Pt[1]), int(Pt[0]), 1] for Pt in srcPts]
	red   = [img1[int(Pt[1]), int(Pt[0]), 2] for Pt in srcPts]
	colour3D = []
	for i in range(len(srcPts)):
	    colour3D.append([red[i] /255, green[i] /255, blue[i] /255])
	return colour3D