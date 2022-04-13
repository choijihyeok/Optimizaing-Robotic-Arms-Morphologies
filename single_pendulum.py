#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:58:55 2022

@author: leonast
"""


from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os
import sys
import numpy as np
from bs4 import BeautifulSoup

xml_path = "single_pendulum.xml"
xml_path2 = "example_pendulum.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)

t = 0
viewer = MjViewer(sim)
ls_wow=[]

# def _get_motor_names(xml_soup):
#     # find actuator and motor
#     if xml_soup.find('actuator'):
#         motors = xml_soup.find('actuator').find_all('motor')
#         name_list = [i_motor['joint'] for i_motor in motors]
#         return name_list
#     else:
#         return None
#
# def _get_tree_structure(xml_soup, node_type_allowed):
#     mj_soup = xml_soup.find('worldbody').find('body')
#     tree = []  # NOTE: the order in the list matters!
#     tree_id = 0
#     nodes = dict()
#
#     # step 0: set the root node
#     motor_names = _get_motor_names(xml_soup)
#     node_info = {'type': 'root', 'is_output_node': False,
#                  'name': 'root_mujocoroot',
#                  'neighbour': [], 'id': 0,
#                  'info': mj_soup.attrs, 'raw': mj_soup}
#     # find joint from motor
#     node_info['attached_joint_name'] = \
#         [i_joint['name']
#          for i_joint in mj_soup.find_all('joint', recursive=False)
#          if i_joint['name'] not in motor_names]
#     for key in JOINT_KEY:
#         node_info[key + '_size'] = 0
#     node_info['attached_joint_info'] = []
#     node_info['tendon_nodes'] = []
#     tree.append(node_info)
#     tree_id += 1
#
#     # step 1: set the 'node_type_allowed' nodes
#     for i_type in node_type_allowed:
#         nodes[i_type] = mj_soup.find_all(i_type)
#         if i_type == 'body':
#             # the root body should be added to the body
#             nodes[i_type] = [mj_soup] + nodes[i_type]
#         if len(nodes[i_type]) == 0:
#             continue
#         for i_node in nodes[i_type]:
#             node_info = dict()
#             node_info['type'] = i_type
#             node_info['is_output_node'] = False
#             node_info['raw_name'] = i_node['name']
#             node_info['name'] = node_info['type'] + '_' + i_node['name']
#             node_info['tendon_nodes'] = []
#             # this id is the same as the order in the tree
#             node_info['id'] = tree_id
#             node_info['parent'] = None
#
#             # additional debug information, should not be used during training
#             node_info['info'] = i_node.attrs
#             node_info['raw'] = i_node
#
#             # NOTE: the get the information about the joint that is directly
#             # attached to 'root' node. These joints will be merged into the
#             # 'root' node
#             if i_type == 'joint' and \
#                     i_node['name'] in tree[0]['attached_joint_name']:
#                 tree[0]['attached_joint_info'].append(node_info)
#                 for key in JOINT_KEY:
#                     tree[0][key + '_size'] += ROOT_OB_SIZE[key][i_node['type']]
#                 continue
#
#             # currently, only 'hinge' type is supported
#             if i_type == 'joint' and i_node['type'] not in ALLOWED_JOINT_TYPE:
#                 logger.warning(
#                     'NOT IMPLEMENTED JOINT TYPE: {}'.format(i_node['type'])
#                 )
#
#             tree.append(node_info)
#             tree_id += 1
#
#         logger.info('{} {} found'.format(len(nodes[i_type]), i_type))
#     node_type_dict = {}
#
#     # step 2: get the node_type dict ready
#     for i_key in node_type_allowed:
#         node_type_dict[i_key] = [i_node['id'] for i_node in tree
#                                  if i_key == i_node['type']]
#         assert len(node_type_dict) >= 1, logger.error(
#             'Missing node type {}'.format(i_key))
#     return tree, node_type_dict


# load xml file
infile = open(xml_path, 'r')
xml_soup = BeautifulSoup(infile.read(), "html.parser")

# note type
node_type_allowed = ['root', 'joint', 'body', 'geom']

# get the basic information of the nodes ready
# tree, node_type_dict = _get_tree_structure(xml_soup, node_type_allowed)


print('process')

while True:
    t += 1
    sim.step()
    viewer.render()



