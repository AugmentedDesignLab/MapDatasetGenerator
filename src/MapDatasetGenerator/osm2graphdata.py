from pyrosm import OSM, get_data
import networkx as nx

# Initialize reader
dir_name = "../data/osm/"
map_name = "SantaBarbara"
osm = OSM(dir_name+map_name+".osm.pbf")
for i in range(len(osm.get_boundaries()['osm_type'])):
    if(osm.get_boundaries()['osm_type'][i]=='relation'): 
        osm_data = OSM(dir_name+map_name+".osm.pbf", osm.get_boundaries()['geometry'][i])
        nodes, edges = osm_data.get_network(nodes=True, network_type="driving")
        nx_graph = osm.to_graph(nodes, edges, graph_type="networkx")
        nx.write_gpickle(nx_graph, 'data/output/'+map_name+str(i)+'.gpickle')
        
