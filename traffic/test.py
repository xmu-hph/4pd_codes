import traci
# 使用sumolib创建一个路网和路由文件
# pip install sumolib lxml
import lxml.etree as ET
import sumolib

# 生成节点文件
def create_nodes_file(file_path):
    nodes = ET.Element("nodes", attrib={"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd"})
    node1 = ET.SubElement(nodes, "node", id="1", x="0.0", y="0.0")
    node2 = ET.SubElement(nodes, "node", id="2", x="100.0", y="0.0")
    node3 = ET.SubElement(nodes, "node", id="3", x="50.0", y="50.0")

    tree = ET.ElementTree(nodes)
    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

# 生成边文件
def create_edges_file(file_path):
    edges = ET.Element("edges", attrib={"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd"})
    edge1 = ET.SubElement(edges, "edge", id="1to2", from_="1", to="2")
    edge2 = ET.SubElement(edges, "edge", id="1to3", from_="1", to="3")
    edge3 = ET.SubElement(edges, "edge", id="3to2", from_="3", to="2")

    tree = ET.ElementTree(edges)
    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

# 生成路由流量文件
def create_routes_file(file_path):
    routes = ET.Element("routes", attrib={"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                          "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"})
    vType = ET.SubElement(routes, "vType", id="car", accel="1.0", decel="5.0", sigma="0.5", length="5.0", minGap="2.5", maxSpeed="25.0", color="1,0,0")
    route = ET.SubElement(routes, "route", id="route0", edges="1to2 2to3")
    vehicle = ET.SubElement(routes, "vehicle", id="veh0", type="car", route="route0", depart="0")

    tree = ET.ElementTree(routes)
    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

# 生成配置文件
def create_config_file(file_path, net_file, route_file):
    config = ET.Element("configuration", attrib={"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                                 "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"})
    input_elem = ET.SubElement(config, "input")
    net_file_elem = ET.SubElement(input_elem, "net-file", value=net_file)
    route_files_elem = ET.SubElement(input_elem, "route-files", value=route_file)

    time_elem = ET.SubElement(config, "time")
    begin_elem = ET.SubElement(time_elem, "begin", value="0")
    end_elem = ET.SubElement(time_elem, "end", value="1000")

    tree = ET.ElementTree(config)
    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

# 生成文件路径
nodes_file = "your_nodes.nod.xml"
edges_file = "your_edges.edg.xml"
net_file = "your_net.net.xml"
routes_file = "your_routes.rou.xml"
config_file = "your_config_file.sumocfg"

# 生成节点和边文件
create_nodes_file(nodes_file)
create_edges_file(edges_file)

# 使用 netconvert 生成路网文件
sumolib.net.convertNet([nodes_file], [edges_file], net_file)

# 生成路由流量文件和配置文件
create_routes_file(routes_file)
create_config_file(config_file, net_file, routes_file)

print("所有文件已生成")

import cityflow
