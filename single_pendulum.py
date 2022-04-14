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

# step 0: settings about the joint
JOINT_KEY = ['qpos', 'qvel', 'qfrc_constr', 'qfrc_act']
ROOT_OB_SIZE = {
    'qpos': {'free': 7, 'hinge': 1, 'slide': 1},
    'qvel': {'free': 6, 'hinge': 1, 'slide': 1},
    'qfrc_act': {'free': 6, 'hinge': 1, 'slide': 1},
    'qfrc_constr': {'free': 6, 'hinge': 1, 'slide': 1}
}
EDGE_TYPE = {'self_loop': 0, 'root-root': 0,  # root-root is loop, also 0
             'joint-joint': 1, 'geom-geom': 2, 'body-body': 3, 'tendon': 4,
             'joint-geom': 5, 'geom-joint': -5,  # pc-relationship
             'joint-body': 6, 'body-joint': -6,
             'body-geom': 7, 'geom-body': -7,
             'root-geom': 8, 'geom-root': -8,
             'root-joint': 9, 'joint-root': -9,
             'root-body': 10, 'body-root': -10}

xml_path = "single_pendulum.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)

t = 0
viewer = MjViewer(sim)
ls_wow=[]

def _get_motor_names(xml_soup):
    # find actuator and motor
    if xml_soup.find('actuator'):
        motors = xml_soup.find('actuator').find_all('motor')
        name_list = [i_motor['joint'] for i_motor in motors]
        return name_list
    else:
        return None

def _get_tree_structure(xml_soup, node_type_allowed):
    # development status
    ALLOWED_JOINT_TYPE = ['hinge', 'free', 'slide']

    mj_soup = xml_soup.find('worldbody').find('body')
    tree = []  # NOTE: the order in the list matters!
    tree_id = 0
    nodes = dict()

    # step 0: set the root node
    motor_names = _get_motor_names(xml_soup)
    node_info = {'type': 'root', 'is_output_node': False,
                 'name': 'root_mujocoroot',
                 'neighbour': [], 'id': 0,
                 'info': mj_soup.attrs, 'raw': mj_soup}
    # find joint from motor
    node_info['attached_joint_name'] = \
        [i_joint['name']
         for i_joint in mj_soup.find_all('joint', recursive=False)
         if i_joint['name'] not in motor_names]
    for key in JOINT_KEY:
        node_info[key + '_size'] = 0
    node_info['attached_joint_info'] = []
    node_info['tendon_nodes'] = []
    tree.append(node_info)
    tree_id += 1

    # step 1: set the 'node_type_allowed' nodes
    for i_type in node_type_allowed:
        nodes[i_type] = mj_soup.find_all(i_type)
        if i_type == 'body':
            # the root body should be added to the body
            nodes[i_type] = [mj_soup] + nodes[i_type]
        if len(nodes[i_type]) == 0:
            continue
        for i_node in nodes[i_type]:
            node_info = dict()
            node_info['type'] = i_type
            node_info['is_output_node'] = False
            node_info['raw_name'] = i_node['name']
            node_info['name'] = node_info['type'] + '_' + i_node['name']
            node_info['tendon_nodes'] = []
            # this id is the same as the order in the tree
            node_info['id'] = tree_id
            node_info['parent'] = None

            # additional debug information, should not be used during training
            node_info['info'] = i_node.attrs
            node_info['raw'] = i_node

            # NOTE: the get the information about the joint that is directly
            # attached to 'root' node. These joints will be merged into the
            # 'root' node
            if i_type == 'joint' and \
                    i_node['name'] in tree[0]['attached_joint_name']:
                tree[0]['attached_joint_info'].append(node_info)
                for key in JOINT_KEY:
                    tree[0][key + '_size'] += ROOT_OB_SIZE[key][i_node['type']]
                continue

            # currently, only 'hinge' type is supported
            if i_type == 'joint' and i_node['type'] not in ALLOWED_JOINT_TYPE:
                print(
                    'NOT IMPLEMENTED JOINT TYPE: {}'.format(i_node['type'])
                )

            tree.append(node_info)
            tree_id += 1

        print('{} {} found'.format(len(nodes[i_type]), i_type))
    node_type_dict = {}

    # step 2: get the node_type dict ready
    for i_key in node_type_allowed:
        node_type_dict[i_key] = [i_node['id'] for i_node in tree
                                 if i_key == i_node['type']]
        if len(node_type_dict) < 1 :
            print('Missing node type {}'.format(i_key))
    return tree, node_type_dict

def _append_tree_relation(tree, node_type_allowed, root_connection_option):
    '''
        @brief:
            build the relationship matrix and append relationship attribute
            to the nodes of the tree

        @input:
            @root_connection_option:
                'nN, Rn': without neighbour, no additional connection
                'nN, Rb': without neighbour, root connected to all body
                'nN, Ra': without neighbour, root connected to all node
                'yN, Rn': with neighbour, no additional connection
    '''
    num_node = len(tree)
    relation_matrix = np.zeros([num_node, num_node], dtype=np.int)

    # step 1: set graph connection relationship
    for i_node in tree:
        # step 1.1: get the id of the children
        children = i_node['raw'].find_all(recursive=False)
        if len(children) == 0:
            continue
        children_names = [i_children.name + '_' + i_children['name']
                          for i_children in children
                          if i_children.name in node_type_allowed]
        children_id_list = [
            [node['id'] for node in tree if node['name'] == i_children_name]
            for i_children_name in children_names
        ]

        i_node['children_id_list'] = children_id_list = \
            sum(children_id_list, [])  # squeeze the list
        current_id = i_node['id']
        current_type = tree[current_id]['type']

        # step 1.2: set the children-parent relationship edges
        for i_children_id in i_node['children_id_list']:
            relation_matrix[current_id, i_children_id] = \
                EDGE_TYPE[current_type + '-' + tree[i_children_id]['type']]
            relation_matrix[i_children_id, current_id] = \
                EDGE_TYPE[tree[i_children_id]['type'] + '-' + current_type]
            if tree[current_id]['type'] == 'body':
                tree[i_children_id]['parent'] = current_id

        # step 1.3 (optional): set children connected if needed
        if 'yN' in root_connection_option:
            for i_node_in_use_1 in i_node['children_id_list']:
                for i_node_in_use_2 in i_node['children_id_list']:
                    relation_matrix[i_node_in_use_1, i_node_in_use_2] = \
                        EDGE_TYPE[tree[i_node_in_use_1]['type'] + '-' +
                                  tree[i_node_in_use_2]['type']]

        else:
            if 'nN' not in root_connection_option:
                print(
                'Unrecognized root_connection_option: {}'.format(
                    root_connection_option
                )
            )

    # step 2: set root connection
    if 'Ra' in root_connection_option:
        # if root is connected to all the nodes
        for i_node_in_use_1 in range(len(tree)):
            target_node_type = tree[i_node_in_use_1]['type']

            # add connections between all nodes and root
            relation_matrix[0, i_node_in_use_1] = \
                EDGE_TYPE['root' + '-' + target_node_type]
            relation_matrix[i_node_in_use_1, 0] = \
                EDGE_TYPE[target_node_type + '-' + 'root']

    elif 'Rb' in root_connection_option:
        for i_node_in_use_1 in range(len(tree)):
            target_node_type = tree[i_node_in_use_1]['type']

            if not target_node_type == 'body':
                continue

            # add connections between body and root
            relation_matrix[0, i_node_in_use_1] = \
                EDGE_TYPE['root' + '-' + target_node_type]
            relation_matrix[i_node_in_use_1, 0] = \
                EDGE_TYPE[target_node_type + '-' + 'root']
    else:
        if 'Rn' not in root_connection_option:
            print(
            'Unrecognized root_connection_option: {}'.format(
                root_connection_option
            )
        )

    # step 3: unset the diagonal terms back to 'self-loop'
    np.fill_diagonal(relation_matrix, EDGE_TYPE['self_loop'])

    return tree, relation_matrix


# load xml file
infile = open(xml_path, 'r')
xml_soup = BeautifulSoup(infile.read(), "html.parser")

# note type
node_type_allowed = ['root', 'joint', 'body', 'geom']

# get the basic information of the nodes ready
tree, node_type_dict = _get_tree_structure(xml_soup, node_type_allowed)

# step 1: get the neighbours and relation tree
# root_connection_option :
# yN : with neighbor
# nN : without neighbour
# Rn : no additional connection
# Rb : root connected to all body
# Ra : root connected to all node
tree, relation_matrix = _append_tree_relation(tree, node_type_allowed, root_connection_option = 'nN, Rb, uE')
print('process')

while True:
    t += 1
    sim.step()
    viewer.render()



