# Program to return images and graphs given an OSM PBF file.
from pyrosm import OSM, get_data
import matplotlib.pyplot as plt
import osmnx as ox

# Initialize reader
dir_name = "data/osm/"
map_name = "SanJose"
osm = OSM(dir_name+map_name+".osm.pbf")
for i in range(len(osm.get_boundaries()['osm_type'])):
    if(osm.get_boundaries()['osm_type'][i]=='relation'): 
        osm_data = OSM(dir_name+map_name+".osm.pbf", osm.get_boundaries()['geometry'][i])
        net = osm_data.get_network(network_type="driving")
        if net is not None:
            net.plot(linewidth=0.5)
            plt.axis('off')
            plt.savefig('data/output/'+map_name+str(i)+'.png', bbox_inches='tight', pad_inches=0, dpi=400)



