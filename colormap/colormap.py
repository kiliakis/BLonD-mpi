'''
Custom colormap
'''

from matplotlib.colors import LinearSegmentedColormap     
import numpy as np
import os

present_folder = os.path.dirname(os.path.realpath(__file__))

cmap_white_blue_red_import = np.loadtxt(present_folder+'/colormap.txt')

cmap_white_blue_red_interparray = np.array(np.linspace(0.,1.,len(cmap_white_blue_red_import[:,0])), ndmin=2)
cmap_white_blue_red_red = np.hstack((cmap_white_blue_red_interparray.T, np.array(cmap_white_blue_red_import[:,0],ndmin=2).T, np.array(cmap_white_blue_red_import[:,0],ndmin=2).T))
cmap_white_blue_red_green = np.hstack((cmap_white_blue_red_interparray.T, np.array(cmap_white_blue_red_import[:,1],ndmin=2).T, np.array(cmap_white_blue_red_import[:,1],ndmin=2).T))
cmap_white_blue_red_blue = np.hstack((cmap_white_blue_red_interparray.T, np.array(cmap_white_blue_red_import[:,2],ndmin=2).T, np.array(cmap_white_blue_red_import[:,2],ndmin=2).T))

cmap_white_blue_red_red = tuple(map(tuple, cmap_white_blue_red_red))
cmap_white_blue_red_green = tuple(map(tuple, cmap_white_blue_red_green))
cmap_white_blue_red_blue = tuple(map(tuple, cmap_white_blue_red_blue))

cdict_white_blue_red = {'red':cmap_white_blue_red_red, 'green':cmap_white_blue_red_green,'blue':cmap_white_blue_red_blue}

cmap_white_blue_red = LinearSegmentedColormap('WhiteBlueRed', cdict_white_blue_red)
