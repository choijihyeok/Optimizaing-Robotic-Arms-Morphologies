#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os
import sys
import numpy as np
from bs4 import BeautifulSoup
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# step 0: settings about the joint
# qpos : position of each joint in the simulation
# qvel : velocity of joints
# qfrc_constr : constraint force
# qfrc_act : actuator force
JOINT_KEY = ['qpos', 'qvel', 'qfrc_constr', 'qfrc_act']
# cinert : com-based body inertia and mass
# cvel : com-based velocity[3D rot, 3D tran]
# cfrc : com-based force
BODY_KEY = ['cinert', 'cvel', 'cfrc']

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

OB_MAP = {
        'Humanoid-v1':
            ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_act', 'cfrc'],
        'HumanoidStandup-v1':
            ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_act', 'cfrc'],
        'HalfCheetah-v1': ['qpos', 'qvel'],
        'Hopper-v1': ['qpos', 'qvel'],
        'Walker2d-v1': ['qpos', 'qvel'],
        'AntS-v1': ['qpos', 'qvel', 'cfrc'],
        'Swimmer-v1': ['qpos', 'qvel'],

        'WalkersHopper-v1': ['qpos', 'qvel'],
        'WalkersHalfhumanoid-v1': ['qpos', 'qvel'],
        'WalkersHalfcheetah-v1': ['qpos', 'qvel'],
        'WalkersFullcheetah-v1': ['qpos', 'qvel'],
        'WalkersOstrich-v1': ['qpos', 'qvel'],
        'WalkersKangaroo-v1': ['qpos', 'qvel']
}

# step 1: register the settings for traditional environments
SYMMETRY_MAP = {'Humanoid-v1': 2,
                'HumanoidStandup-v1': 2,
                'HalfCheetah-v1': 1,
                'Hopper-v1': 1,
                'Walker2d-v1': 1,
                'AntS-v1': 2,
                'Swimmer-v1': 2,

                'WalkersHopper-v1': 1,
                'WalkersHalfhumanoid-v1': 1,
                'WalkersHalfcheetah-v1': 1,
                'WalkersFullcheetah-v1': 1,
                'WalkersOstrich-v1': 1,
                'WalkersKangaroo-v1': 1}

PARAMETERS_DEFAULT_DICT = {
    'root': {},
    'body': {'pos': 'NON_DEFAULT'},
    'geom': {
        'fromto': '-1 -1 -1 -1 -1 -1',
        'size': 'NON_DEFAULT',
        'type': 'NON_DEFAULT'
    },
    'joint': {
        'armature': '-1',
        'axis': 'NON_DEFAULT',
        'damping': '-1',
        'pos': 'NON_DEFAULT',
        'stiffness': '-1',
        'range': '-1 -1'
    }
}

GEOM_TYPE_ENCODE = {
    'capsule': [0.0, 1.0],
    'sphere': [1.0, 0.0],
}

xml_path = "single_pendulum.xml"
xml_path2 = "double_pendulum.xml"
xml_path3 = "swimmer.xml"
xml_path4 = "half_cheetah.xml"
xml_path5 = "hopper.xml"

xml_path = xml_path5
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

def _append_tendon_relation(tree, relation_matrix, xml_soup):
    '''
        @brief:
            build the relationship of tendon (the spring)
    '''
    tendon = xml_soup.find('tendon')
    if tendon is None:
        return tree, relation_matrix
    tendon_list = tendon.find_all('fixed')

    for i_tendon in tendon_list:
        # find the id
        joint_name = ['joint_' + joint['joint']
                      for joint in i_tendon.find_all('joint')]
        joint_id = [node['id'] for node in tree if node['name'] in joint_name]
        if len(joint_id) != 2: print(
            'Unsupported tendon: {}'.format(i_tendon))

        # update the tree and the relationship matrix
        relation_matrix[joint_id[0], joint_id[1]] = EDGE_TYPE['tendon']
        relation_matrix[joint_id[1], joint_id[0]] = EDGE_TYPE['tendon']
        tree[joint_id[0]]['tendon_nodes'].append(joint_id[1])
        tree[joint_id[1]]['tendon_nodes'].append(joint_id[0])

        print(
            'new tendon found between: {} and {}'.format(
                tree[joint_id[0]]['name'], tree[joint_id[1]]['name']
            )
        )

    return tree, relation_matrix

def _get_input_info(tree, task_name):
    input_dict = {}

    joint_id = [node['id'] for node in tree if node['type'] == 'joint']
    body_id = [node['id'] for node in tree if node['type'] == 'body']
    root_id = [node['id'] for node in tree if node['type'] == 'root'][0]

    # init the input dict
    input_dict[root_id] = []
    if 'cinert' in OB_MAP[task_name] or \
            'cvel' in OB_MAP[task_name] or 'cfrc' in OB_MAP[task_name]:
        candidate_id = joint_id + body_id
    else:
        candidate_id = joint_id
    for i_id in candidate_id:
        input_dict[i_id] = []

    print('scanning ob information...')
    current_ob_id = 0
    for ob_type in OB_MAP[task_name]:

        if ob_type in JOINT_KEY:
            # step 1: collect the root ob's information. Some ob might be
            # ignore, which is specify in the SYMMETRY_MAP
            ob_step = tree[0][ob_type + '_size'] - \
                (ob_type == 'qpos') * SYMMETRY_MAP[task_name]
            input_dict[root_id].extend(
                range(current_ob_id, current_ob_id + ob_step)
            )
            current_ob_id += ob_step

            # step 2: collect the joint ob's information
            for i_id in joint_id:
                input_dict[i_id].append(current_ob_id)
                current_ob_id += 1

        elif ob_type in BODY_KEY:

            BODY_OB_SIZE = 10 if ob_type == 'cinert' else 6

            # step 0: skip the 'world' body
            current_ob_id += BODY_OB_SIZE

            # step 1: collect the root ob's information, note that the body will
            # still take this ob
            input_dict[root_id].extend(
                range(current_ob_id, current_ob_id + BODY_OB_SIZE)
            )
            # current_ob_id += BODY_OB_SIZE

            # step 2: collect the body ob's information
            for i_id in body_id:
                input_dict[i_id].extend(
                    range(current_ob_id, current_ob_id + BODY_OB_SIZE)
                )
                current_ob_id += BODY_OB_SIZE
        else:
            if 'add' not in ob_type: \
                print('TYPE {BODY_KEY} NOT RECGNIZED'.format(ob_type))
            addition_ob_size = int(ob_type.split('_')[-1])
            input_dict[root_id].extend(
                range(current_ob_id, current_ob_id + addition_ob_size)
            )
            current_ob_id += addition_ob_size
        print(
            'after {}, the ob size is reaching {}'.format(
                ob_type, current_ob_id
            )
        )
    return input_dict, current_ob_id  # to debug if the ob size is matched

def _get_output_info(tree, xml_soup, gnn_output_option):
    output_list = []
    output_type_dict = {}
    motors = xml_soup.find('actuator').find_all('motor')

    for i_motor in motors:
        joint_id = [i_node['id'] for i_node in tree
                    if 'joint_' + i_motor['joint'] == i_node['name']]
        if len(joint_id) == 0:
            # joint_id = 0  # it must be the root if not found
            print(
                'Motor {} not found!'.format(i_motor['joint'])
            )
        else:
            joint_id = joint_id[0]
        tree[joint_id]['is_output_node'] = True
        output_list.append(joint_id)

        # construct the output_type_dict
        if gnn_output_option == 'shared':
            motor_type_name = i_motor['joint'].split('_')[0]
            if motor_type_name in output_type_dict:
                output_type_dict[motor_type_name].append(joint_id)
            else:
                output_type_dict[motor_type_name] = [joint_id]
        elif gnn_output_option == 'separate':
            motor_type_name = i_motor['joint']
            output_type_dict[motor_type_name] = [joint_id]
        else:
            if gnn_output_option != 'unified': print(
                'Invalid output type: {}'.format(gnn_output_option)
            )
            if 'unified' in output_type_dict:
                output_type_dict['unified'].append(joint_id)
            else:
                output_type_dict['unified'] = [joint_id]

    return tree, output_list, output_type_dict, len(motors)

def _get_para_list(xml_soup, node_type_allowed):
    '''
        @brief:
            for each type in the node_type_allowed, we find the attributes that
            shows up in the xml
            below is the node parameter info list:

            @root (size 0):
                More often the case, the root node is the domain root, as
                there is the 2d/3d information in it.

            @body (max size: 3):
                @pos: 3

            @geom (max size: 9):
                @fromto: 6
                @size: 1
                @type: 2

            @joint (max size: 11):
                @armature: 1
                @axis: 3
                @damping: 1
                @pos: 3
                @stiffness: 1  # important
                @range: 2
    '''
    # step 1: get the available parameter list for each node
    para_list = {node_type: [] for node_type in node_type_allowed}
    mj_soup = xml_soup.find('worldbody').find('body')
    for node_type in node_type_allowed:
        # search the node with type 'node_type'
        node_list = mj_soup.find_all(node_type)  # all the nodes
        for i_node in node_list:
            # deal with each node
            for key in i_node.attrs:
                # deal with each attributes
                if key not in para_list[node_type] and \
                        key in PARAMETERS_DEFAULT_DICT[node_type]:
                    para_list[node_type].append(key)

    # step 2: get default parameter settings
    default_dict = PARAMETERS_DEFAULT_DICT
    default_soup = xml_soup.find('default')
    if default_soup != None:
        for node_type, para_type_list in para_list.items():
            # find the default str if possible
            type_soup = default_soup.find(node_type)
            if type_soup != None:
                for para_type in para_type_list:
                    if para_type in type_soup.attrs:
                        default_dict[node_type][para_type] = \
                            type_soup[para_type]
            else:
                print(
                    'No default settings available for type {}'.format(
                        node_type
                    )
                )
    else:
        print('No default settings available for this xml!')

    return para_list, default_dict

def _collect_parameter_info(output_parameter,
                            parameter_type,
                            node_type,
                            default_dict,
                            info_dict):
    # step 1: get the parameter str
    if parameter_type in info_dict:
        # append the default setting into default_dict
        para_str = info_dict[parameter_type]
    elif parameter_type in default_dict[node_type]:
        para_str = default_dict[node_type][parameter_type]
    else:
        if True: print(
            'no information available for node: {}, para: {}'.format(
                node_type, parameter_type
            )
        )

    # step 2: parse the str into the parameter numbers
    if node_type == 'geom' and para_str in GEOM_TYPE_ENCODE:
        output_parameter.extend(GEOM_TYPE_ENCODE[para_str])
    else:
        output_parameter.extend(
            [float(element) for element in para_str.split(' ')]
        )

    return output_parameter


def _append_node_parameters(tree,
                            xml_soup,
                            node_type_allowed,
                            gnn_embedding_option):
    '''
        @brief:
            the output of this function is a dictionary.
        @output:
            e.g.: node_parameters['geom'] is a numpy array, which has the shape
            of (num_nodes, para_size_of_'geom')
            the node is ordered in the relative position in the tree
    '''
    assert node_type_allowed.index('joint') < node_type_allowed.index('body')

    if gnn_embedding_option == 'parameter':
        # step 0: get the para list and default setting for this mujoco xml
        PARAMETERS_LIST, default_dict = _get_para_list(xml_soup,
                                                       node_type_allowed)

        # step 2: get the node_parameter_list ready, they are in the node_order
        node_parameters = {node_type: [] for node_type in node_type_allowed}
        for node_id in range(len(tree)):
            output_parameter = []

            for i_parameter_type in PARAMETERS_LIST[tree[node_id]['type']]:
                # collect the information one by one
                output_parameter = _collect_parameter_info(
                    output_parameter, i_parameter_type,
                    tree[node_id]['type'], default_dict, tree[node_id]['info']
                )

            # this node is finished
            node_parameters[tree[node_id]['type']].append(output_parameter)

        # step 3: numpy the elements, and do validation check
        for node_type in node_type_allowed:
            node_parameters[node_type] = np.array(node_parameters[node_type],
                                                  dtype=np.float32)

        # step 4: get the size of parameters logged
        para_size_dict = {
            node_type: len(node_parameters[node_type][0])
            for node_type in node_type_allowed
        }

        # step 5: trick, root para is going to receive a dummy para [1]
        para_size_dict['root'] = 1
        node_parameters['root'] = np.ones([1, 1])
    elif gnn_embedding_option in \
            ['shared', 'noninput_separate', 'noninput_shared']:
        # step 1: preprocess, register the node, get the number of bits for
        # encoding needed
        struct_name_list = {node_type: [] for node_type in node_type_allowed}
        for node_id in range(len(tree)):
            name = tree[node_id]['name'].split('_')
            type_name = name[0]

            if gnn_embedding_option in ['noninput_separate']:
                register_name = name
                struct_name_list[type_name].append(register_name)
            else:  # shared
                register_name = type_name + '_' + name[1]
                if register_name not in struct_name_list[type_name]:
                    struct_name_list[type_name].append(register_name)
            tree[node_id]['register_embedding_name'] = register_name

        struct_name_list['root'] = [tree[0]['name']]  # the root
        tree[0]['register_embedding_name'] = tree[0]['name']

        # step 2: estimate the encoding length
        num_type_bits = 2
        para_size_dict = {  # 2 bits for type encoding
            i_node_type: num_type_bits + 8
            for i_node_type in node_type_allowed
        }

        # step 3: get the parameters
        node_parameters = {i_node_type: []
                           for i_node_type in node_type_allowed}
        appear_str = []
        for node_id in range(len(tree)):
            type_name = tree[node_id]['type']
            type_str = str(bin(node_type_allowed.index(type_name)))
            type_str = (type_str[2:]).zfill(num_type_bits)
            node_str = str(bin(struct_name_list[type_name].index(
                tree[node_id]['register_embedding_name']
            )))
            node_str = (node_str[2:]).zfill(
                para_size_dict[tree[node_id]['type']] - 2
            )

            if node_id == 0 or para_size_dict[type_name] == 2:
                node_str = ''

            final_str = type_str + node_str
            if final_str not in appear_str:
                appear_str.append(final_str)

            if 'noninput_shared_multi' in gnn_embedding_option:
                node_parameters[type_name].append(
                    tree[node_id]['register_embedding_name']
                )
            elif 'noninput' in gnn_embedding_option:
                node_parameters[type_name].append([appear_str.index(final_str)])
            else:
                node_parameters[type_name].append(
                    [int(i_char) for i_char in final_str]
                )

        # step 4: numpy the elements, and do validation check
        if gnn_embedding_option != 'noninput_shared_multi':
            para_dtype = np.float32 \
                if gnn_embedding_option in ['parameter', 'shared'] \
                else np.int32
            for node_type in node_type_allowed:
                node_parameters[node_type] = \
                    np.array(node_parameters[node_type], dtype=para_dtype)
    else:
        if True: print(
            'Invalid option: {}'.format(gnn_embedding_option)
        )

    # step 5: postprocess
    # NOTE: make the length of the parameters the same
    if gnn_embedding_option in ['parameter', 'shared']:
        max_length = max([para_size_dict[node_type]
                         for node_type in node_type_allowed])
        for node_type in node_type_allowed:
            shape = node_parameters[node_type].shape
            new_node_parameters = np.zeros([shape[0], max_length], dtype=np.int)
            new_node_parameters[:, 0: shape[1]] = node_parameters[node_type]
            node_parameters[node_type] = new_node_parameters
            para_size_dict[node_type] = max_length
    else:
        para_size_dict = {i_node_type: 1 for i_node_type in node_type_allowed}

    return node_parameters, para_size_dict

def _prune_body_nodes(tree,
                      relation_matrix,
                      node_type_dict,
                      input_dict,
                      node_parameters,
                      para_size_dict,
                      root_connection_option):
    '''
        @brief:
            In this function, we will have to remove the body node.
            1. delete all the the bodys, whose ob will be placed into its kid
                joint (multiple joints possible)
            2. for the root node, if kid joint exists, transfer the ownership
                body ob into the kids
    '''
    # make sure the tree is structured in a 'root', 'joint', 'body' order
    assert node_type_dict['root'] == [0] and \
        max(node_type_dict['joint']) < min(node_type_dict['body']) and \
        'geom' not in node_type_dict

    # for each joint, let it eat its father body root, the relation_matrix and
    # input_dict need to be inherited
    for node_id, i_node in enumerate(tree[0: min(node_type_dict['body'])]):
        if i_node['type'] != 'joint':
            assert i_node['type'] == 'root'
            continue

        # find all the parent
        parent = i_node['parent']

        # inherit the input observation
        if parent in input_dict:
            input_dict[node_id] += input_dict[parent]
        '''
            1. inherit the joint with shared body, only the first joint will
                inherit the AB_body's relationship. other joints will be
                attached to the first joint
                A_joint ---- AB_body ---- B_joint. On
            2. inherit joint-joint relationships for sybling joints:
                A_joint ---- A_body ---- B_body ---- B_joint
            3. inherit the root-joint connection
                A_joint ---- A_body ---- root
        '''
        # step 1: check if there is brothers / sisters of this node
        children = np.where(
            relation_matrix[parent, :] == EDGE_TYPE['body-joint']
        )[0]
        first_brother = [child_id for child_id in children
                         if child_id != node_id and child_id < node_id]
        if len(first_brother) > 0:
            first_brother = min(first_brother)
            relation_matrix[node_id, first_brother] = EDGE_TYPE['joint-joint']
            relation_matrix[first_brother, node_id] = EDGE_TYPE['joint-joint']
            continue

        # step 2: the type 2 relationship, note that only the first brother
        # will be considered
        uncles = np.where(
            relation_matrix[parent, :] == EDGE_TYPE['body-body']
        )[0]
        for i_uncle in uncles:
            syblings = np.where(
                relation_matrix[i_uncle, :] == EDGE_TYPE['body-joint']
            )[0]
            if len(syblings) > 0:
                sybling = syblings[0]
            else:
                continue
            if tree[sybling]['parent'] is tree[i_uncle]['parent']:
                continue
            relation_matrix[node_id, sybling] = EDGE_TYPE['joint-joint']
            relation_matrix[sybling, node_id] = EDGE_TYPE['joint-joint']

        # step 3: the type 3 relationship
        uncles = np.where(
            relation_matrix[parent, :] == EDGE_TYPE['body-root']
        )[0]
        assert len(uncles) <= 1
        for i_uncle in uncles:
            relation_matrix[node_id, i_uncle] = EDGE_TYPE['joint-root']
            relation_matrix[i_uncle, node_id] = EDGE_TYPE['root-joint']

    # remove all the body root
    first_body_node = min(node_type_dict['body'])
    tree = tree[:first_body_node]
    relation_matrix = relation_matrix[:first_body_node, :first_body_node]
    for i_body_node in node_type_dict['body']:
        if i_body_node in input_dict:
            input_dict.pop(i_body_node)
    node_parameters.pop('body')
    node_type_dict.pop('body')
    para_size_dict.pop('body')

    for i_node in node_type_dict['joint']:
        assert len(input_dict[i_node]) == len(input_dict[1])

    return tree, relation_matrix, node_type_dict, \
        input_dict, node_parameters, para_size_dict

def construct_ob_size_dict(node_info, input_feat_dim):
    '''
        @brief: for each node type, we collect the ob size for this type
    '''
    node_info['ob_size_dict'] = {}
    for node_type in node_info['node_type_dict']:
        node_ids = node_info['node_type_dict'][node_type]

        # record the ob_size for each type of node
        if node_ids[0] in node_info['input_dict']:
            node_info['ob_size_dict'][node_type] = \
                len(node_info['input_dict'][node_ids[0]])
        else:
            node_info['ob_size_dict'][node_type] = 0

        node_ob_size = [
            len(node_info['input_dict'][node_id])
            for node_id in node_ids if node_id in node_info['input_dict']
        ]

        if len(node_ob_size) == 0:
            continue

        assert node_ob_size.count(node_ob_size[0]) == len(node_ob_size), \
            print('Nodes (type {}) have wrong ob size: {}!'.format(
                node_type, node_ob_size
            ))

    return node_info

def get_inverse_type_offset(node_info, mode):
    if mode not in ['output', 'node']: print(
        'Invalid mode: {}'.format(mode)
    )
    inv_extype_offset = 'inverse_' + mode + '_extype_offset'
    inv_intype_offset = 'inverse_' + mode + '_intype_offset'
    inv_self_offset = 'inverse_' + mode + '_self_offset'
    inv_original_id = 'inverse_' + mode + '_original_id'
    node_info[inv_extype_offset] = []
    node_info[inv_intype_offset] = []
    node_info[inv_self_offset] = []
    node_info[inv_original_id] = []
    current_offset = 0
    for mode_type in node_info[mode + '_type_dict']:
        i_length = len(node_info[mode + '_type_dict'][mode_type])
        # the original id
        node_info[inv_original_id].extend(
            node_info[mode + '_type_dict'][mode_type]
        )

        # In one batch, how many element is listed before this type?
        # e.g.: [A, A, C, B, C, A], with order [A, B, C] --> [0, 0, 4, 3, 4, 0]
        node_info[inv_extype_offset].extend(
            [current_offset] * i_length
        )

        # In current type, what is the position of this node?
        # e.g.: [A, A, C, B, C, A] --> [0, 1, 0, 0, 1, 2]
        node_info[inv_intype_offset].extend(
            range(i_length)
        )

        # how many nodes are in this type?
        # e.g.: [A, A, C, B, C, A] --> [3, 3, 2, 1, 2, 3]
        node_info[inv_self_offset].extend(
            [i_length] * i_length
        )
        current_offset += i_length

    sorted_id = np.array(node_info[inv_original_id])
    sorted_id.sort()
    node_info[inv_original_id] = [
        node_info[inv_original_id].index(i_node)
        for i_node in sorted_id
    ]

    node_info[inv_extype_offset] = np.array(
        [node_info[inv_extype_offset][i_node]
         for i_node in node_info[inv_original_id]]
    )
    node_info[inv_intype_offset] = np.array(
        [node_info[inv_intype_offset][i_node]
         for i_node in node_info[inv_original_id]]
    )
    node_info[inv_self_offset] = np.array(
        [node_info[inv_self_offset][i_node]
         for i_node in node_info[inv_original_id]]
    )

    return node_info

def get_receive_send_idx(node_info):
    # register the edges that shows up, get the number of edge type
    edge_dict = EDGE_TYPE
    edge_type_list = []  # if one type of edge exist, register

    for edge_id in edge_dict.values():
        if edge_id == 0:
            continue  # the self loop is not considered here
        if (node_info['relation_matrix'] == edge_id).any():
            edge_type_list.append(edge_id)

    node_info['edge_type_list'] = edge_type_list
    node_info['num_edge_type'] = len(edge_type_list)

    receive_idx_raw = {}
    receive_idx = []
    send_idx = {}
    for edge_type in node_info['edge_type_list']:
        receive_idx_raw[edge_type] = []
        send_idx[edge_type] = []
        i_id = np.where(node_info['relation_matrix'] == edge_type)
        for i_edge in range(len(i_id[0])):
            send_idx[edge_type].append(i_id[0][i_edge])
            receive_idx_raw[edge_type].append(i_id[1][i_edge])
            receive_idx.append(i_id[1][i_edge])

    node_info['receive_idx'] = receive_idx
    node_info['receive_idx_raw'] = receive_idx_raw
    node_info['send_idx'] = send_idx
    node_info['num_edges'] = len(receive_idx)

    return node_info

hidden_dim = 64

def prepare_placeholders():
    '''
        @brief:
            get the input placeholders ready. The _input placeholder has
            different size from the input we use for the general network.
    '''
    # step 1: build the input_obs and input_parameters
    input_obs = {
        node_type: tf.placeholder(
            tf.float32,
            [None, node_info['ob_size_dict'][node_type]],
            name='input_ob_placeholder_ggnn'
        )
        for node_type in node_info['node_type_dict']
    }

    input_hidden_state = {
        node_type: tf.placeholder(
            tf.float32,
            [None, hidden_dim],
            name='input_hidden_dim_' + node_type
        )
        for node_type in node_info['node_type_dict']
    }

    input_parameter_dtype = tf.int32 \
        if 'noninput' in gnn_embedding_option else tf.float32
    input_parameters = {
        node_type: tf.placeholder(
            input_parameter_dtype,
            [None, node_info['para_size_dict'][node_type]],
            name='input_para_placeholder_ggnn')
        for node_type in node_info['node_type_dict']
    }

    # step 2: the receive and send index
    receive_idx = tf.placeholder(
        tf.int32, shape=(None), name='receive_idx'
    )
    send_idx = {
        edge_type: tf.placeholder(
            tf.int32, shape=(None),
            name='send_idx_{}'.format(edge_type))
        for edge_type in node_info['edge_type_list']
    }

    # step 3: the node type index and inverse node type index
    node_type_idx = {
        node_type: tf.placeholder(
            tf.int32, shape=(None),
            name='node_type_idx_{}'.format(node_type))
        for node_type in node_info['node_type_dict']
    }
    inverse_node_type_idx = tf.placeholder(
        tf.int32, shape=(None), name='inverse_node_type_idx'
    )

    # step 4: the output node index
    output_type_idx = {
        output_type: tf.placeholder(
            tf.int32, shape=(None),
            name='output_type_idx_{}'.format(output_type)
        )
        for output_type in node_info['output_type_dict']
    }
    inverse_output_type_idx = tf.placeholder(
        tf.int32, shape=(None), name='inverse_output_type_idx'
    )

    # step 5: batch_size
    batch_size_int = tf.placeholder(
        tf.int32, shape=(), name='batch_size_int'
    )
    return input_obs, input_hidden_state, input_parameters, receive_idx, send_idx, node_type_idx, inverse_node_type_idx, output_type_idx, inverse_output_type_idx, batch_size_int

class MLP(object):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                input dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which outputs a list of N tensors, each is the hidden
                activation of one layer
    """

    def __init__(self,
                 dims,
                 init_method,
                 act_func=None,
                 add_bias=True,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="MLP",
                 use_dropout=False):

        self._init_method = init_method

        self._scope = scope
        self._add_bias = add_bias
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer
        self._act_func = [None] * self._num_layer
        self._use_dropout = use_dropout

        # initialize variables
        with tf.variable_scope(scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in = int(dims[ii - 1])
                    dim_out = int(dims[ii])

                    self._w[ii] = weight_variable(
                        [dim_in, dim_out], init_method=self._init_method,
                        init_para={"mean": 0.0, "stddev": init_std},
                        wd=wd, name="w", dtype=dtype
                    )

                    if add_bias:
                        self._b[ii] = weight_variable(
                            [dim_out], init_method="constant",
                            init_para={"val": 1.0e-2},
                            wd=wd, name="b", dtype=dtype
                        )

                    if act_func and act_func[ii] is not None:
                        if act_func[ii] == "relu":
                            self._act_func[ii] = tf.nn.relu
                        elif act_func[ii] == "sigmoid":
                            self._act_func[ii] = tf.sigmoid
                        elif act_func[ii] == "tanh":
                            self._act_func[ii] = tf.tanh
                        else:
                            raise ValueError("Unsupported activation method!")

    def __call__(self, x, dropout_mask=None):
        h = [None] * self._num_layer

        with tf.variable_scope(self._scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    if ii == 0:
                        input_vec = x
                    else:
                        input_vec = h[ii - 1]

                    if self._use_dropout:
                        if dropout_mask is not None:
                            input_vec = 2 * input_vec * dropout_mask[ii]
                        else:
                            input_vec = tf.nn.dropout(
                                input_vec, 0.5,
                                name='training_dropout_' + str(ii)
                            )

                    h[ii] = tf.matmul(input_vec, self._w[ii])

                    if self._add_bias:
                        h[ii] += self._b[ii]

                    if self._act_func[ii] is not None:
                        h[ii] = self._act_func[ii](h[ii])

        return h

def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_para=None,
                    wd=None,
                    seed=1234,
                    name=None,
                    trainable=True):
    """ Initialize Weights

        Input:
                shape: list of int, shape of the weights
                init_method: string, indicates initialization method
                init_para: a dictionary,
                init_val: if it is not None, it should be a tensor
                wd: a float, weight decay
                name:
                trainable:

        Output:
                var: a TensorFlow Variable
    """

    if init_method is None:
        initializer = tf.zeros_initializer(shape, dtype=dtype)
    elif init_method == "normal":
        initializer = tf.random_normal_initializer(
            mean=init_para["mean"],
            stddev=init_para["stddev"],
            seed=seed,
            dtype=dtype
        )
    elif init_method == "truncated_normal":
        initializer = tf.truncated_normal_initializer(
            mean=init_para["mean"],
            stddev=init_para["stddev"],
            seed=seed,
            dtype=dtype
        )
    elif init_method == "uniform":
        initializer = tf.random_uniform_initializer(
            minval=init_para["minval"],
            maxval=init_para["maxval"],
            seed=seed,
            dtype=dtype
        )
    elif init_method == "constant":
        initializer = tf.constant_initializer(value=init_para["val"],
                                              dtype=dtype)
    elif init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=True, seed=seed, dtype=dtype
        )
    elif init_method == 'orthogonal':
        initializer = tf.orthogonal_initializer(
            gain=1.0, seed=seed, dtype=dtype
        )
    else:
        raise ValueError("Unsupported initialization method!")

    var = tf.Variable(initializer(shape), name=name, trainable=trainable)

    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_decay")
        tf.add_to_collection("losses", weight_decay)

    return var


class GRU(object):
    """ Gated Recurrent Units (GRU)

        Input:
                input_dim: input dimension
                hidden_dim: hidden dimension
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which computes the output of GRU with one step
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 init_method,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="GRU"):

        self._init_method = init_method

        # initialize variables
        with tf.variable_scope(scope):
            self._w_xi = weight_variable(
                [input_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_xi", dtype=dtype
            )
            self._w_hi = weight_variable(
                [hidden_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_hi", dtype=dtype
            )
            self._b_i = weight_variable(
                [hidden_dim], init_method="constant",
                init_para={"val": 0.0}, wd=wd, name="b_i", dtype=dtype
            )

            self._w_xr = weight_variable(
                [input_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_xr", dtype=dtype
            )
            self._w_hr = weight_variable(
                [hidden_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_hr", dtype=dtype
            )
            self._b_r = weight_variable(
                [hidden_dim], init_method="constant", init_para={"val": 0.0},
                wd=wd, name="b_r", dtype=dtype
            )

            self._w_xu = weight_variable(
                [input_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_xu", dtype=dtype
            )
            self._w_hu = weight_variable(
                [hidden_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_hu", dtype=dtype
            )
            self._b_u = weight_variable(
                [hidden_dim], init_method="constant", init_para={"val": 0.0},
                wd=wd, name="b_u", dtype=dtype
            )

    def __call__(self, x, state, is_batch_mul=False):
        # update gate
        g_i = tf.sigmoid(
            tf.matmul(x, self._w_xi) + tf.matmul(state, self._w_hi) + self._b_i
        )

        # reset gate
        g_r = tf.sigmoid(
            tf.matmul(x, self._w_xr) + tf.matmul(state, self._w_hr) + self._b_r
        )

        # new memory implementation 1
        u = tf.tanh(
            tf.matmul(x, self._w_xu) + tf.matmul(g_r * state, self._w_hu) +
            self._b_u
        )

        # hidden state
        new_state = state * g_i + u * (1 - g_i)

        return new_state

# parameters for build_network_weights
name_scope = 'evolutionary_agent_policy'
input_feat_dim = 64
gnn_embedding_option = 'shared'
MLP_embedding = None
init_method = init_methods='orthogonal'
agent_id = 0
seed = 0
npr = np.random.RandomState(seed + agent_id)
trainable = True
network_shape = [64, 64]
node_update_method = 'GRU'
Node_update = None
gnn_output_per_node = 3
output_size = 1
MLP_ob_mapping = None
MLP_prop = None
MLP_Out = None
action_dist_logstd = None

def build_network_weights():
    '''
        @brief: build the network
        @weights:
            _MLP_embedding (1 layer)
            _MLP_ob_mapping (1 layer)
            _MLP_prop (2 layer)
            _MLP_output (2 layer)
    '''
    # step 1: build the weight parameters (mlp, gru)
    with tf.variable_scope(name_scope):
        # step 1_1: build the embedding matrix (mlp)
        # tensor shape (None, para_size) --> (None, input_dim - ob_size)
        if input_feat_dim % 2 != 0: print("assert")
        if 'noninput' not in gnn_embedding_option:
            MLP_embedding = {
                node_type: MLP(
                    [input_feat_dim / 2,
                     node_info['para_size_dict'][node_type]],
                    init_method=init_method,
                    act_func=['tanh'] * 1,  # one layer at most
                    add_bias=True,
                    scope='MLP_embedding_node_type_{}'.format(node_type)
                )
                for node_type in node_info['node_type_dict']
                if node_info['ob_size_dict'][node_type] > 0
            }
            MLP_embedding.update({
                node_type: MLP(
                    [input_feat_dim,
                     node_info['para_size_dict'][node_type]],
                    init_method=init_method,
                    act_func=['tanh'] * 1,  # one layer at most
                    add_bias=True,
                    scope='MLP_embedding_node_type_{}'.format(node_type)
                )
                for node_type in node_info['node_type_dict']
                if node_info['ob_size_dict'][node_type] == 0
            })
        else:
            embedding_vec_size = max(
                np.reshape(
                    [max(node_info['node_parameters'][i_key])
                     for i_key in node_info['node_parameters']],
                    [-1]
                )
            ) + 1
            embedding_vec_size = int(embedding_vec_size)
            embedding_variable = {}

            out = npr.randn(
                embedding_vec_size, int(input_feat_dim / 2)
            ).astype(np.float32)
            out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            embedding_variable[False] = tf.Variable(
                out, name='embedding_HALF', trainable=trainable
            )

            if np.any([node_size == 0 for _, node_size
                       in node_info['ob_size_dict'].items()]):

                out = npr.randn(
                    embedding_vec_size, input_feat_dim
                ).astype(np.float32)
                out *= 1.0 / np.sqrt(np.square(out).sum(axis=0,
                                                        keepdims=True))
                embedding_variable[True] = tf.Variable(
                    out, name='embedding_FULL', trainable=trainable
                )

        # step 1_2: build the ob mapping matrix (mlp)
        # tensor shape (None, para_size) --> (None, input_dim - ob_size)
        MLP_ob_mapping = {
            node_type: MLP(
                [input_feat_dim / 2,
                 node_info['ob_size_dict'][node_type]],
                init_method=init_method,
                act_func=['tanh'] * 1,  # one layer at most
                add_bias=True,
                scope='MLP_embedding_node_type_{}'.format(node_type)
            )
            for node_type in node_info['node_type_dict']
            if node_info['ob_size_dict'][node_type] > 0
        }

        # step 1_4: build the mlp for the propogation between nodes
        MLP_prop_shape = network_shape + \
            [hidden_dim] + [hidden_dim]
        MLP_prop = {
            i_edge: MLP(
                MLP_prop_shape,
                init_method=init_method,
                act_func=['tanh'] * (len(MLP_prop_shape) - 1),
                add_bias=True,
                scope='MLP_prop_edge_{}'.format(i_edge)
            )
            for i_edge in node_info['edge_type_list']
        }

        # step 1_5: build the node update function for each node type
        if node_update_method == 'GRU':
            Node_update = {
                i_node_type: GRU(
                    hidden_dim * 2,  # for both the message and ob
                    hidden_dim,
                    init_method=init_method,
                    scope='GRU_node_{}'.format(i_node_type)
                )
                for i_node_type in node_info['node_type_dict']
            }
        else:
            assert False

        # step 1_6: the mlp for the mu of the actions
        # (l_1, l_2, ..., l_o, l_i)
        MLP_out_shape = network_shape + \
            [gnn_output_per_node] + [hidden_dim]
        MLP_out_act_func = ['tanh'] * (len(MLP_out_shape) - 1)
        MLP_out_act_func[-1] = None

        MLP_Out = {
            output_type: MLP(
                MLP_out_shape,
                init_method=init_method,
                act_func=MLP_out_act_func,
                add_bias=True,
                scope='MLP_out'
            )
            for output_type in node_info['output_type_dict']
        }

        # step 1_8: build the log std for the actions
        action_dist_logstd = tf.Variable(
            (0.0 * npr.randn(1, output_size)).astype(
                np.float32
            ),
            name="policy_logstd",
            trainable=trainable
        )
    return MLP_embedding, None, MLP_ob_mapping, MLP_prop, Node_update, MLP_Out, action_dist_logstd

input_embedding = None
embedding_variable = None
ob_feat = None
input_feat = None
nstep = 1
input_feat_list = None


def build_input_graph():
    if 'noninput' not in gnn_embedding_option:
        input_embedding = {
            node_type: MLP_embedding[node_type](
                input_parameters[node_type]
            )[-1]
            for node_type in node_info['node_type_dict']
        }
    else:
        input_embedding = {
            node_type: tf.gather(
                embedding_variable[
                    node_info['ob_size_dict'][node_type] == 0
                ],
                tf.reshape(input_parameters[node_type], [-1])
            )
            for node_type in node_info['node_type_dict']
        }

    # shape: [n_step, node_num, embedding_size + ob_size]
    ob_feat = {
        node_type: MLP_ob_mapping[node_type](
            input_obs[node_type]
        )[-1]
        for node_type in node_info['node_type_dict']
        if node_info['ob_size_dict'][node_type] > 0
    }
    ob_feat.update({
        node_type: input_obs[node_type]
        for node_type in node_info['node_type_dict']
        if node_info['ob_size_dict'][node_type] == 0
    })

    input_feat = {
        node_type: tf.concat([
            tf.reshape(
                input_embedding[node_type],
                [-1, nstep *
                 len(node_info['node_type_dict'][node_type]),
                 int(input_feat_dim / 2)],
            ),
            tf.reshape(
                ob_feat[node_type],
                [-1, nstep *
                 len(node_info['node_type_dict'][node_type]),
                 int(input_feat_dim / 2)],
            )
        ], axis=2)
        for node_type in node_info['node_type_dict']
    }

    split_feat_list = {
        node_type: tf.split(
            input_feat[node_type],
            nstep,
            axis=1,
            name='split_into_nstep' + node_type
        )
        for node_type in node_info['node_type_dict']
    }
    feat_list = []
    for i_step in range(nstep):
        # for node_type in self._node_info['node_type_dict']:
        feat_list.append(
            tf.concat(
                [tf.reshape(split_feat_list[node_type][i_step],
                            [-1, input_feat_dim])
                 for node_type in node_info['node_type_dict']],
                axis=0  # the node
            )
        )
    input_feat_list = [
        tf.gather(  # get node order into graph order
            i_step_data,
            inverse_node_type_idx,
            name='get_order_back_gather_init' + str(i_step),
        )
        for i_step, i_step_data in enumerate(feat_list)
    ]

    current_hidden_state = tf.concat(
        [input_hidden_state[node_type]
         for node_type in node_info['node_type_dict']],
        axis=0
    )
    current_hidden_state = tf.gather(  # get node order into graph order
        current_hidden_state,
        inverse_node_type_idx,
        name='get_order_back_gather_init'
    )
    return current_hidden_state, input_feat_list

concat_msg = None
batch_size_int = None
inverse_node_type_idx = None


def build_network_graph():
    current_hidden_state, input_feat_list = build_input_graph()

    # step 3: unroll the propogation
    action_mu_output_lst = []  # [nstep, None, n_action_size]

    for tt in range(nstep):
        current_input_feat = input_feat_list[tt]
        prop_msg = []
        for ee, i_edge_type in enumerate(node_info['edge_type_list']):
            node_activate = \
                tf.gather(
                    current_input_feat,
                    send_idx[i_edge_type],
                    name='edge_id_{}_prop_steps_{}'.format(i_edge_type, tt)
                )
            prop_msg.append(
                MLP_prop[i_edge_type](node_activate)[-1]
            )

        # aggregate messages
        concat_msg = tf.concat(prop_msg, 0)
        # self.concat_msg = concat_msg
        message = tf.unsorted_segment_sum(
            concat_msg, receive_idx,
            node_info['num_nodes'] * batch_size_int
        )

        denom_const = tf.unsorted_segment_sum(
            tf.ones_like(concat_msg), receive_idx,
            node_info['num_nodes'] * batch_size_int
        )
        message = tf.div(message, (denom_const + tf.constant(1.0e-10)))
        node_update_input = tf.concat([message, current_input_feat], axis=1,
                                      name='ddbug' + str(tt))

        # update the hidden states via GRU
        new_state = []
        for i_node_type in node_info['node_type_dict']:
            new_state.append(
                Node_update[i_node_type](
                    tf.gather(
                        node_update_input,
                        node_type_idx[i_node_type],
                        name='GRU_message_node_type_{}_prop_step_{}'.format(
                            i_node_type, tt
                        )
                    ),
                    tf.gather(
                        current_hidden_state,
                        node_type_idx[i_node_type],
                        name='GRU_feat_node_type_{}_prop_steps_{}'.format(
                            i_node_type, tt
                        )
                    )
                )
            )
        output_hidden_state = {
            node_type: new_state[i_id]
            for i_id, node_type
            in enumerate(node_info['node_type_dict'])
        }
        new_state = tf.concat(new_state, 0)  # BTW, the order is wrong
        # now, get the orders back
        current_hidden_state = tf.gather(
            new_state, inverse_node_type_idx,
            name='get_order_back_gather_prop_steps_{}'.format(tt)
        )

        # self._action_mu_output = []  # [nstep, None, n_action_size]
        action_mu_output = []
        for output_type in node_info['output_type_dict']:
            action_mu_output.append(
                MLP_Out[output_type](
                    tf.gather(
                        current_hidden_state,
                        output_type_idx[output_type],
                        name='output_type_{}'.format(output_type)
                    )
                )[-1]
            )

        action_mu_output = tf.concat(action_mu_output, 0)
        action_mu_output = tf.gather(
            action_mu_output,
            inverse_output_type_idx,
            name='output_inverse'
        )

        action_mu_output = tf.reshape(action_mu_output,
                                      [batch_size_int, -1])
        action_mu_output_lst.append(action_mu_output)

    action_mu_output_lst = tf.reshape(
        tf.concat(action_mu_output_lst, axis=1), [-1, output_size]
    )
    # step 4: build the log std for the actions
    action_dist_logstd_param = tf.reshape(
        tf.tile(
            tf.reshape(action_dist_logstd, [1, 1, output_size],
                       name='test'),
            [nstep, batch_size_int, 1]
        ), [-1, output_size]
    )
    return action_mu_output_lst, action_dist_logstd_param




'''
    @brief:
        get the tree of "geom", "body", "joint" built up.

    @return:
        @tree: This function will return a list of dicts. Every dict
            contains the information of node.

            tree[id_of_the_node]['name']: the unique identifier of the node

            tree[id_of_the_node]['neighbour']: is the list for the
            neighbours

            tree[id_of_the_node]['type']: could be 'geom', 'body' or
            'joint'

            tree[id_of_the_node]['info']: debug info from the xml. it
            should not be used during model-free optimization

            tree[id_of_the_node]['is_output_node']: True or False

        @input_dict: input_dict['id_of_the_node'] = [position of the ob]

        @output_list: This correspond to the id of the node where a output
            is available
'''
# build session
session = None
use_gpu = 0
if use_gpu:
    config = tf.ConfigProto(device_count={'GPU': 1})
else:
    config = tf.ConfigProto(device_count={'GPU': 0})
config.gpu_options.allow_growth = True  # don't take full gpu memory
session = tf.Session(config=config)

# load xml file
# parse_mujoco_template
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

# step 2: get the tendon relationship ready
tree, relation_matrix = _append_tendon_relation(tree,
                                                relation_matrix,
                                                xml_soup)

import matplotlib.pyplot as plt

plotMatrix = np.array(relation_matrix)
fig, ax = plt.subplots()
ax.matshow(plotMatrix, cmap=plt.cm.Blues)
for i in range(len(plotMatrix)):
    for j in range(len(plotMatrix[0])):
        c = plotMatrix[j, i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.title('Relation Matrix')
plt.show()
# plt.savefig('relationMatrix.png')

# step 3: get the input list ready
# task_name is specific xml model.
task_name = 'Humanoid-v1'
input_dict, ob_size = _get_input_info(tree, task_name)

# step 4: get the output list ready
gnn_output_option='shared'
tree, output_list, output_type_dict, action_size = \
    _get_output_info(tree, xml_soup, gnn_output_option)

# step 5: get the node parameters
gnn_embedding_option = 'shared'
node_parameters, para_size_dict = _append_node_parameters(
    tree, xml_soup, node_type_allowed, gnn_embedding_option
)

debug_info = {'ob_size': ob_size, 'action_size': action_size}

# step 6: prune the body nodes
gnn_node_option='nG,yB'
root_connection_option='nN, Rn, sE'
if 'nB' in gnn_node_option:
    tree, relation_matrix, node_type_dict, \
        input_dict, node_parameters, para_size_dict = \
        _prune_body_nodes(tree=tree,
                          relation_matrix=relation_matrix,
                          node_type_dict=node_type_dict,
                          input_dict=input_dict,
                          node_parameters=node_parameters,
                          para_size_dict=para_size_dict,
                          root_connection_option=root_connection_option)

# step 7: (optional) uni edge type?
if 'uE' in root_connection_option:
    relation_matrix[np.where(relation_matrix != 0)] = 1
else:
    if 'sE' not in root_connection_option:
        print('assert')

node_info = dict(tree=tree,
            relation_matrix=relation_matrix,
            node_type_dict=node_type_dict,
            output_type_dict=output_type_dict,
            input_dict=input_dict,
            output_list=output_list,
            debug_info=debug_info,
            node_parameters=node_parameters,
            para_size_dict=para_size_dict,
            num_nodes=len(tree))

# step 1: check that the input and output size is matched
is_baseline = False
# gnn_util.io_size_check(self._input_size, self._output_size, node_info, is_baseline)

# step 2: check for ob size for each node type, construct the node dict
input_feat_dim=64
node_info = construct_ob_size_dict(node_info, input_feat_dim)

# step 3: get the inverse node offsets (used to construct gather idx)
node_info = get_inverse_type_offset(node_info, 'node')

# step 4: get the inverse node offsets (used to gather output idx)
node_info = get_inverse_type_offset(node_info, 'output')

# step 5: register existing edge and get the receive and send index
node_info = get_receive_send_idx(node_info)

# prepare the network's input and output
input_obs, input_hidden_state, input_parameters, receive_idx, send_idx, node_type_idx, inverse_node_type_idx, output_type_idx, inverse_output_type_idx, batch_size_int = prepare_placeholders()

# Visualization Graph
import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt

# changeing send, receive index to list
send_idx_list = []
receive_idx_list = list(node_info['receive_idx'])
for key, value in node_info['send_idx'].items():
    send_idx_list.extend(value)
print('send:',send_idx_list)
print('recv:', receive_idx_list)
# Graph connectivity [2, num_edges]
edge_index = torch.tensor([send_idx_list,
                           receive_idx_list])
# Node feature
# x = torch.tensor([[3], [4], [5]])
# Labels
labels = {}
for i in range(node_info['num_nodes']):
    labels[i] = i
data = torch_geometric.data.Data(edge_index=edge_index)
g = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g, with_labels=True, node_size=100, alpha=1, linewidths=10, labels=labels)
plt.show()
# plt.savefig('graph.png')




# define the network here
MLP_embedding, embedding_variable, MLP_ob_mapping, MLP_prop, Node_update, MLP_Out, action_dist_logstd = build_network_weights()

action_mu_output, action_dist_logstd_param = build_network_graph()

# get the variable list ready
# collect the tf variable and the trainable tf variable
trainable_var_list = [var for var in tf.trainable_variables()
                                    if name_scope in var.name]

all_var_list = [var for var in tf.global_variables()
                              if name_scope in var.name]



# making species information for hopper.

# root infomation of creature
CREATURE_ROOT_INFO = {
    'fish': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.01,
        'b_size': 0.08,
        'c_size': 0.04,
        'joint_range': 60
    },
    'walker': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.07,
        'b_size': 0.3,
        'c_size': 233,
        'joint_range': 60
    },
    'hopper': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.0653,
        'b_size': 0.1025,
        'c_size': 0.03,
        'joint_range': 60
    },
    'cheetah': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.046,
        'b_size': 0.5,
        'c_size': 0.03,
        'joint_range': 60
    }
}

# attributes of creature
CREATURE_ORIGINAL_ATTR = {
    'fish': [
        # tail1
        {
            'u': 3 * np.pi / 2,
            'v': np.pi / 2,
            'axis_x': -0.99,
            'axis_y': 1e-16,
            'a_size': 0.001,
            'b_size': 0.008,
            'c_size': 0.016,
            'joint_range': 30,
            'geom_type': 0
        },
        # tail2
        {
            'u': np.pi / 2,
            'v': np.pi / 2,
            'axis_x': 0.99,
            'axis_y': 1e-16,
            'a_size': 0.001,
            'b_size': 0.018,
            'c_size': 0.035,
            'joint_range': 30,
            'geom_type': 0   # the original attribute is defined using stiffness
        },
        # left fin
        {
            'u': np.pi,
            'v': np.pi / 2,
            'axis_x': 1e-16,
            'axis_y': 0.99,
            'a_size': 0.02,
            'b_size': 0.015,
            'c_size': 0.001,
            'joint_range': 30,
            'geom_type': 0 # the original attribute is defined using tendon
        },
        # right fin
        {
            'u': 0,
            'v': np.pi / 2,
            'axis_x': 1e-16,
            'axis_y': -0.99,
            'a_size': 0.02,
            'b_size': 0.015,
            'c_size': 0.001,
            'joint_range': 30,
            'geom_type': 0
        }
    ],
    'walker': [
        # left thigh
        # NOTE: left and right thigh should be symmetric
        {
            'u': 4 * np.pi / 9,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.05,
            'b_size': 0.225,
            'c_size': 40,
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.04,
            'b_size': 0.25,
            'c_size': 40,
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 0,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.05,
            'b_size': 0.08,
            'c_size': 40,
            'joint_range': 30,
            'geom_type': 0
        }
    ],
    'hopper': [
        {
            'u': np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.065,
            'b_size': 0.075,
            'c_size': 30,
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 0,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.04,
            'b_size': 0.165,
            'c_size': 30,
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.03,
            'b_size': 0.16,
            'c_size': 30,
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 0,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.04,
            'b_size': 0.09,
            'c_size': 30,
            'joint_range': 30,
            'geom_type': 0
        }
    ],
    'cheetah': [
        {# back thigh
            'u': -4 * np.pi / 9,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.046,
            'b_size': 0.145,
            'c_size': 120,
            'joint_range': 30,
            'geom_type': 0
        },
        {# back shin
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.046,
            'b_size': 0.15,
            'c_size': 90,
            'joint_range': 30,
            'geom_type': 0
        },
        {# back foot
            'u': -4 * np.pi / 9,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1,
            'a_size': 0.046,
            'b_size': 0.15,
            'c_size': 60,
            'joint_range': 30,
            'geom_type': 0
        },
        {# front thigh
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': 1,
            'axis_y': -1,
            'a_size': 0.046,
            'b_size': 0.145,
            'c_size': 90,
            'joint_range': 30,
            'geom_type': 0
        },
        {# front shin
            'u': 0,
            'v': np.pi,
            'axis_x': 1,
            'axis_y': -1,
            'a_size': 0.046,
            'b_size': 0.106,
            'c_size': 60,
            'joint_range': 30,
            'geom_type': 0
        },
        {# front foot
            'u': 7 * 2 * np.pi / 9,
            'v': np.pi,
            'axis_x': 1,
            'axis_y': -1,
            'a_size': 0.046,
            'b_size': 0.07,
            'c_size': 30,
            'joint_range': 30,
            'geom_type': 0
        }
    ]
}

CREATURE_HARD_CONSTRAINT = {
    'fish': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.002, 0.03),
        'b_size': (0.002, 0.03),
        'c_size': (0.002, 0.03),
        'joint_range': (30, 120)
    },
    'walker': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.03, 0.07),
        'b_size': (0.1, 0.3),
        'c_size': (30, 40), # c_size server as the value for gear
        'joint_range': (30, 90)
    },
    'hopper': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.03, 0.07),
        'b_size': (0.010, 0.035),
        'c_size': (0.002, 0.03),
        'joint_range': (30, 120)
    },
    'cheetah': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.002, 0.03),
        'b_size': (0.03, 0.15),
        'c_size': (30, 120),
        'joint_range': (30, 120)
    }
}


# making node and tree


class Node:
    def __init__(self, node_id, attr):
        self.id = node_id
        self.attr = attr
        self.child_list = []

    def get_all_descendents(self):
        '''
        '''
        all_nodes = []
        p_queue = [self]
        while len(p_queue) != 0:
            node = p_queue.pop(0)
            for c_node in node.get_child_list():
                p_queue.append(c_node)

            all_nodes.append(node)

        return all_nodes
    def get_all_node_id(self):
        ''' return a list of node_ids that are part of the node-defined
        subtree
        NOTE: this method returns the node in breadth first search (BFS) order!
        '''
        # get all descendents list
        all_nodes = self.get_all_descendents()
        all_id_list = [x.get_node_id() for x in all_nodes]

        # add its own id
        # all_id_list = [self.get_node_id()] + all_descendents_id
        return all_id_list

    def is_same_size(self, other_node):
        ''' return true if the two nodes have the same
        'a_size', 'b_size', 'c_size' in its attribute
        '''
        if np.abs(self.attr['a_size'] - \
            other_node.get_attr()['a_size']) > 1e-7:
            return False
        if np.abs(self.attr['b_size'] - \
            other_node.get_attr()['b_size']) > 1e-7:
            return False
        if np.abs(self.attr['c_size'] - \
            other_node.get_attr()['c_size']) > 1e-7:
            return False
        return True

    def set_id(self, new_id):
        '''
        '''
        self.id = new_id
        return new_id + 1

    def add_child(self, node):
        self.child_list.append(node)

    def get_child_list(self):
        return self.child_list

    def get_node_id(self):
        return self.id

    def get_attr(self):
        return self.attr

    def set_attr(self, new_attr):
        self.attr = new_attr
        return

    def gen_test_node_attr(self, task='fish', node_num=5, discrete_rv=True):
        '''
        '''

        def get_uniform(low, high, discrete=True, total_lvl=6, avoid_zero=True):
            ''' uniformly sample from [low, high]
            if given discrete, it will sample from [low, high] at total_lvl
            steps
            '''
            assert total_lvl > 0, 'Invalid number of step %d' % total_lvl

            if discrete:
                step = (high - low) / total_lvl
                val = low + step * random.randint(1, total_lvl)
            else:
                val = float(np.random.uniform(low, high, 1))

            if avoid_zero:
                val += 1e-9
            return val

        def root_default(task):
            root_attr = {}

            if 'fish' in task:
                # trivial information that shouldn't be used
                root_attr['geom_type'] = -1
                root_attr['u'] = -1
                root_attr['v'] = -1
                root_attr['axis_x'] = -1
                root_attr['axis_y'] = -1
                # this is the default value as in dm_control original repo
                root_attr['a_size'] = 0.0075
                root_attr['b_size'] = 0.06
                root_attr['c_size'] = root_attr['a_size'] * 4
                #
                root_attr['joint_range'] = 60
            elif 'walker' in task:
                # some trivial attributes
                root_attr['geom_type'] = -1
                root_attr['u'] = -1
                root_attr['v'] = -1
                root_attr['axis_x'] = -1
                root_attr['axis_y'] = -1
                root_attr['joint_range'] = 60
                # setting the torso, using ellipsoid to approximate cylinder
                root_attr['a_size'] = 0.07
                root_attr['b_size'] = 0.3
                root_attr['c_size'] = -1
            elif 'hopper' in task:
                root_attr = CREATURE_ROOT_INFO['hopper']
            elif 'cheetah' in task:
                root_attr = CREATURE_ROOT_INFO['cheetah']
            else:
                assert 0, 'task: %s, not supported' % task

            return root_attr

        def gen_one_attr(task, discrete_rv=True):
            '''
            '''
            node_attr = {}

            # define geom type
            node_attr['geom_type'] = random.randint(0, 0)

            if 'fish' in task:
                constraint = CREATURE_HARD_CONSTRAINT['fish']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])

            elif 'walker' in task:
                constraint = CREATURE_HARD_CONSTRAINT['walker']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])

            elif 'hopper' in task:
                constraint = CREATURE_HARD_CONSTRAINT['hopper']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])
            elif 'cheetah' in task:
                constraint = CREATURE_HARD_CONSTRAINT['walker']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])

            return node_attr

        node_attr_list = []
        for i in range(node_num):
            node_attr = gen_one_attr(task, discrete_rv=discrete_rv)
            node_attr_list.append(node_attr)

        node_attr_list[0] = root_default(task)
        return node_attr_list

    def make_symmetric(self, task='fish', discrete=True):
        ''' make the symmetric node w.r.t the self node
        symmetric node and return the new node
        '''
        new_attr = self.gen_test_node_attr(
            task=task,
            node_num=2,
            discrete_rv=discrete
        )[-1]

        same_attr = ['a_size', 'b_size', 'c_size', 'geom_type',
                     'joint_range']
        for item in same_attr:
            new_attr[item] = self.attr[item]

        # set certain attributes
        new_attr['u'] = np.pi - self.attr['u']
        while new_attr['u'] < 0: new_attr['u'] += 2 * np.pi
        new_attr['v'] = self.attr['v']
        new_attr['axis_x'] = -self.attr['axis_x']
        new_attr['axis_y'] = self.attr['axis_y']

        # create the new node
        new_node = Node(-1, new_attr)

        return new_node

def node_count(node, counter_start=0):
    ''' count the number of nodes in a tree
    '''
    total_num = counter_start
    parent_list = [node]
    while len(parent_list) != 0:
        node = parent_list.pop(0)
        child_list = node.get_child_list()
        for child in child_list:
            parent_list.append(child)

        total_num = node.set_id(total_num)
    return total_num

class Tree:
    def __init__(self, node=None):
        self.root = node
        self.total_num = 0
        self.total_num = node_count(self.root, self.total_num)

    def add_sub_tree(self, parent, child):
        ''' add the child (a node containing subtree)
        to the parent's child_list
        return: list of id values that is assigned to child
        '''
        # traverse child's subtree and update the id
        prev_node_num = self.total_num
        self.total_num = node_count(child, self.total_num)
        parent.add_child(child)

        return list(range(prev_node_num, self.total_num))

    def remove_sub_tree(self, parent, child):
        ''' remove the parent child relationship
        '''
        # figure out the place child within parent's child list
        child_list = parent.get_child_list()
        child_id_list = [item.get_node_id() for item in child_list]
        try:
            idx = child_id_list.index(child.get_node_id())
        except:
            raise RuntimeError('child doesn\'t exist in parent')
        child_list.pop(idx)
        self.total_num = 0
        self.total_num = node_count(self.root, self.total_num)
        return

    def sample_node(self, p=0.5):
        ''' randomly sample a node
        '''

        def check_node(node, p=0.5):
            if np.random.binomial(1, p) == 1:
                return node
            res = None
            for child in node.child_list:
                res = check_node(child)
                if res is not None: return res
            return res

        # WARNING:
        # this method has different weights for nodes at different depth
        # sample_node = None
        # while sample_node is None:
        #     sample_node = check_node(self.root, p=p)
        all_nodes = self.get_all_nodes()
        sample_node = random.choice(all_nodes)

        return sample_node

    def get_all_nodes(self):
        ''' return all the nodes in the tree
        (order is not guaranteed)
        '''
        all_nodes = []
        node_list = [self.root]
        while len(node_list) != 0:
            node = node_list.pop(0)
            for c_node in node.get_child_list():
                node_list.append(c_node)
            all_nodes.append(node)
        return all_nodes

    def get_pc_relation(self):
        ''' return a list of strings '%d-%d'
        meaning parent-child connection
        '''
        pc_rel = []
        p_node_list = [self.root]
        while len(p_node_list) != 0:
            p_node = p_node_list.pop(0)
            for c_node in p_node.get_child_list():
                p_id = p_node.get_node_id()
                c_id = c_node.get_node_id()
                rel = '%d-%d' % (p_id, c_id)
                pc_rel.append(rel)
                p_node_list.append(c_node)
        return pc_rel

    def mirror_mat(self, mat):
        ''' input:  1. numpy array matrix of N x N size
            output: 1. matrix of N x N size
                       keep the other half of the original mat
                       and make it symmetric along the diagonal line
        '''
        N, _ = mat.shape
        assert N == _, 'Not a square matrix, it cannot be symmetric!'

        for i in range(N):
            for j in range(i, N):
                mat[j, i] = mat[i, j]
        return mat

    def to_mujoco_format(self):
        ''' create the 1. adj_mat
                       2. node_attr
        to get ready for model_gen
        '''
        N = self.total_num

        pc_relation = self.get_pc_relation()
        all_nodes = self.get_all_nodes()

        # generate the adj_matrix
        adj_mat = np.zeros((N, N))
        for rel in pc_relation:
            i, j = rel.split('-')
            i, j = int(i), int(j)
            adj_mat[i, j] = 1
        adj_mat = self.mirror_mat(adj_mat)
        adj_mat[adj_mat > 0] = 7

        # generate the attributes
        sorted_nodes = sorted(all_nodes, key=lambda x: x.get_node_id())
        node_attr = [item.get_attr() for item in sorted_nodes]
        return adj_mat.astype('int'), node_attr



# generate xml hopper structure
from lxml import etree
import random

def hopper_xml_generator(adj_matrix, node_attr, options=None, filename=None):
    ''' generate xml for hopper
    '''

    def get_encoding(val):
        ''' from a value (0-7) to binary one-hot encoding
        eg. input 6
            output [1, 1, 0]
        '''
        onehot = [int(item) for item in list(bin(val)[2:])]
        onehot = [0 for i in range(3 - len(onehot))] + onehot
        onehot = list(reversed(onehot))
        return onehot



    def add_mujoco_header(root):
        incl_1 = etree.Element('include', file='./common/skybox.xml')
        incl_2 = etree.Element('include', file='./common/visual.xml')
        incl_3 = etree.Element('include', file='./common/materials.xml')
        root.append(incl_1)
        root.append(incl_2)
        root.append(incl_3)
        return root

    def add_mujoco_options(root, options):
        # similar to walker but the attribute value is different
        option = etree.Element('option', timestep='0.005')
        statistic = etree.Element('statistic', extent='2', center='0 0 0.5')
        root.append(option)
        root.append(statistic)

        # default
        default_sec = etree.Element('default')

        # default 1: hopper class
        default_1 = etree.fromstring('<default class="hopper"></default>')
        default_1_joint = etree.Element('joint', type='hinge',
                                        axis='0 1 0', limited='true', damping='0.05', armature='.2'
                                        )
        default_1_geom = etree.Element('geom', type='capsule', material='self')
        default_1_site = etree.Element('site', type='sphere', size='0.05', group='3')
        default_1.append(default_1_joint)
        default_1.append(default_1_geom)
        default_1.append(default_1_site)

        default_2 = etree.fromstring('<default class="free"></default>')
        default_2_joint = etree.Element('joint', limited='false', damping='0',
                                        armature='0', stiffness='0'
                                        )
        default_2.append(default_2_joint)

        default_motor = etree.Element('motor', ctrlrange='-1 1', ctrllimited='true')

        default_sec.append(default_1)
        default_sec.append(default_2)
        default_sec.append(default_motor)

        root.append(default_sec)
        return root

    def add_worldbody(root, adj_matrix, node_attr_list):
        '''
        '''

        def worldbody_preliminary(worldbody):
            '''
            '''
            geom = etree.Element('geom', name='floor', type='plane',
                                 conaffinity='1', pos='48 0 0', size='50 1 .2', material='grid'
                                 )
            worldbody.append(geom)

            camera1 = etree.Element('camera', name='cam0',
                                    pos='0 -2.8 0.8', euler='90 0 0',
                                    mode='trackcom'
                                    )
            camera2 = etree.Element('camera', name='back',
                                    pos='-2 -.2 1.2', xyaxes='0.2 -1 0 .5 0 2',
                                    mode='trackcom'
                                    )
            worldbody.append(camera1)
            worldbody.append(camera2)

            return worldbody

        def torso_preliminary(torso):
            light = etree.Element('light', name='top', pos='0 0 2', mode='trackcom')
            torso.append(light)

            joint_x = etree.fromstring('<joint name="rootx" type="slide" axis="1 0 0" class="free"/>')
            joint_y = etree.fromstring('<joint name="rooty" type="hinge" axis="0 1 0" class="free"/>')
            joint_z = etree.fromstring('<joint name="rootz" type="slide" axis="0 0 1" class="free"/>')
            torso.append(joint_x)
            torso.append(joint_z)
            torso.append(joint_y)

            geom1 = etree.Element('geom', name='torso', fromto='0 0 -.05 0 0 .2', size='0.0653')
            geom2 = etree.Element('geom', name='nose', fromto='.08 0 .13 .15 0 .14', size='0.03')
            torso.append(geom1)
            torso.append(geom2)

            return torso

        ########### START OF PARSING THE WORLDBODY ###########
        # this part should be identical to that of walker
        worldbody = etree.Element('worldbody')
        worldbody = worldbody_preliminary(worldbody)

        # parse the matrix
        N, _ = adj_matrix.shape

        body_dict = {}
        info_dict = {}  # log the needed information given a node

        # the root of the model is always fixed
        body_root = etree.Element('body', name='torso', pos='0 0 1',
                                  childclass='hopper')
        body_root = torso_preliminary(body_root)
        root_info = {}

        root_info['a_size'] = node_attr_list[0]['a_size']
        root_info['b_size'] = node_attr_list[0]['b_size']
        root_info['c_size'] = node_attr_list[0]['c_size']
        # for determining the center of the capsule relative to the body joint
        root_info['center_rel_pos'] = 0

        info_dict[0] = root_info
        body_dict[0] = body_root

        # initilize the parent list to go throught the entire matrix
        parent_list = [0]
        while len(parent_list) != 0:
            parent_node = parent_list.pop(0)

            parent_row = np.copy(adj_matrix[parent_node])
            for i in range(parent_node + 1): parent_row[i] = 0
            child_list = np.where(parent_row)[0].tolist()

            while True:

                try:
                    child_node = child_list.pop(0)
                except:
                    break

                # parent-child relationship
                # print('P-C relationship:', parent_node, child_node)
                node_attr = node_attr_list[child_node]
                node_name = 'node-%d' % (child_node)

                # this is parent's ellipsoid information
                parent_info = info_dict[parent_node]
                a_parent = parent_info['a_size']
                b_parent = parent_info['b_size']
                c_parent = parent_info['c_size']
                center_rel_pos = parent_info['center_rel_pos']

                # getting node attributes from the list
                u = node_attr['u']  # using these 2 values for determining the range
                v = node_attr['v']  # of the joint

                axis_x = node_attr['axis_x']  # used to determine the relative position
                axis_y = node_attr['axis_y']  # w.r.t the parent capsule

                a_child = node_attr['a_size']  # using this as the capsule radius
                b_child = node_attr['b_size']  # using this as the capsule h
                c_child = node_attr['c_size']

                # compute the translational and rotational matrix
                child_info = {}

                # store attributes that defines the child's geom
                child_info['a_size'] = a_child
                child_info['b_size'] = b_child
                child_info['c_size'] = c_child

                # set the stitching point relative to parent
                a = min([node_attr['axis_x'], node_attr['axis_y']])
                b = max([node_attr['axis_x'], node_attr['axis_y']])
                stitch_ratio = node_attr['axis_x'] / 1

                if not (stitch_ratio <= 1.01 and stitch_ratio >= -1.01):
                    import pdb;
                    pdb.set_trace()
                stitch_pt = b_parent * stitch_ratio + center_rel_pos
                body_pos = '0 0 %f' % (stitch_pt)

                # body translation
                if node_attr['axis_x'] * node_attr['axis_y'] >= 0:
                    geom_pos = '0 0 -%f' % (b_child)
                    child_info['center_rel_pos'] = -b_child
                else:
                    geom_pos = '0 0 %f' % (b_child)
                    child_info['center_rel_pos'] = b_child

                joint_pos = '0 0 0'

                # now create the body
                body_child = etree.Element('body', name=node_name, pos=body_pos)

                # add geom
                geom_type = 2  # for all planar creates, we use capsule
                capsule_size = '%f %f' % (a_child, b_child)
                geom = etree.Element('geom', name=node_name, pos=geom_pos, size=capsule_size)
                body_child.append(geom)

                # add joints
                joint_type = adj_matrix[parent_node, child_node]
                joint_axis = get_encoding(joint_type)
                joint_range = node_attr['joint_range']
                range1 = node_attr['u'] / (2.0 * np.pi) * -90
                range2 = node_attr['v'] / (1.0 * np.pi) * 150 + 1
                range2 = range1 + 90 + 1
                if joint_axis[0] == 1:
                    x_joint = etree.fromstring("<joint name='%d-%d_x' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                                               (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(x_joint)
                if joint_axis[1] == 1:
                    y_joint = etree.fromstring("<joint name='%d-%d_y' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                                               (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(y_joint)
                if joint_axis[2] == 1:
                    z_joint = etree.fromstring("<joint name='%d-%d_z' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                                               (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(z_joint)

                site1_pos = '0 0 0'
                site2_pos = '0 0 %f' % (2 * child_info['center_rel_pos'])
                site1 = etree.Element('site', name='touch1-%s' % (node_name), pos=site1_pos)
                site2 = etree.Element('site', name='touch2-%s' % (node_name), pos=site2_pos)
                body_child.append(site1)
                body_child.append(site2)

                # logging the information
                body_dict[parent_node].append(body_child)
                body_dict[child_node] = body_child  # register child's body struct in case it has child
                info_dict[child_node] = child_info
                # child becomes the parent for further examination
                parent_list.append(child_node)

        worldbody.append(body_dict[0])
        root.append(worldbody)
        return root

    def dfs_order2(adj_mat):
        '''
            return the order of parent-child relationship in the tree structure
            described by the adj_matrix
            input:
                1. adj_mat an N x N matrix in which index 0 is the root
                2. the matrix is symmetric
            output:
                1. a list of '%d-%d's decribing the order of parent-child relationship
                   run using dfs
        '''

        def dfs_order_helper(adj_mat, node_id, cur_order):
            '''
            '''
            # get child_list
            node_row = np.copy(adj_mat[node_id])
            for i in range(node_id + 1): node_row[i] = 0
            child_list = np.where(node_row)[0].tolist()

            for child_node in child_list:
                edge = '%d-%d' % (node_id, child_node)
                cur_order.append(edge)
                dfs_order_helper(adj_mat, child_node, cur_order)

            return cur_order

        # using recursion to solve the problem
        dfs_order = []
        dfs_order = dfs_order_helper(adj_mat, 0, dfs_order)
        return dfs_order

    def add_mujoco_sensor(root, adj_matrix):
        ''' this is specific for planar creature
        the body parts need to be informed with their collision with the ground
        '''

        dfs_order = dfs_order2(adj_matrix)
        node_order = [0] + [int(item.split('-')[-1]) for item in dfs_order]

        sensor = etree.Element('sensor')

        for i in range(1, len(node_order)):
            node_name = 'node-%d' % (i)

            site1 = etree.Element('touch', name='touch1-%s' % (node_name),
                                  site='touch1-%s' % (node_name)
                                  )
            site2 = etree.Element('touch', name='touch2-%s' % (node_name),
                                  site='touch2-%s' % (node_name)
                                  )
            sensor.append(site1)
            sensor.append(site2)

        root.append(sensor)
        return root

    def add_mujoco_actuator(root, adj_matrix):
        ''' this part should be exactly the same as walker
        '''
        dfs_order = dfs_order2(adj_matrix)
        actuator = etree.Element('actuator')

        for edge in dfs_order:

            p, c = [int(x) for x in edge.split('-')]

            joint_type = adj_matrix[p, c]
            joint_axis = get_encoding(joint_type)
            edges = []
            if joint_axis[0] == 1:
                edge_x = '%s_x' % edge
                edges.append(edge_x)
            elif joint_axis[1] == 1:
                edge_y = '%s_y' % edge
                edges.append(edge_y)
            elif joint_axis[2] == 1:
                edge_z = '%s_z' % edge
                edges.append(edge_z)

            positions = [etree.Element('motor', name=item, joint=item, gear='30')
                         for item in edges]
            for item in positions: actuator.append(item)
        root.append(actuator)
        return root

    ################## Actual codes for generating hopper ##################

    root = etree.Element('mujoco', model='planar hopper')
    root = add_mujoco_header(root)
    root = add_mujoco_options(root, options)
    root = add_worldbody(root, adj_matrix, node_attr)
    root = add_mujoco_sensor(root, adj_matrix)
    root = add_mujoco_actuator(root, adj_matrix)

    if filename is not None:
        tree = etree.ElementTree(root)
        tree.write('./assets/gen' + filename + '.xml',
                   pretty_print=True,
                   xml_declaration=True, encoding='utf-8'
                   )
    return root

from copy import deepcopy

# making species
class Species:
    '''
    '''

    def __init__(self,
                 body_num=3,
                 discrete=True,
                 allow_hierarchy=True,
                 filter_ratio=0.5):
        # filter ratio
        # self.args = args
        self.p = 0.5
        self.allow_hierarchy = True
        self.discrete = True
        self.mutation_add_ratio = 0.2
        self.mutation_delete_ratio = 0.2
        self.self_cross_ratio = 0.0
        self.force_symmetric = False

        # if args.optimize_creature == False:
        #     # set up the tree (if no starting species is specified)
        #     r_node = Node(0,
        #                   model_gen.gen_test_node_attr(task=args.task, node_num=2, discrete_rv=discrete)[0]
        #                   )
        #     self.struct_tree = Tree(r_node)
        #
        #     if 'walker' in self.args.task:
        #         while len(self.struct_tree.get_root().get_all_descendents()) <= \
        #                 body_num:
        #             self.perturb_add(no_selfcross=True)
        #     else:
        #         for i in range(1, body_num):
        #             if self.args.force_symmetric:
        #                 node_list = self.generate_one_node()
        #             else:
        #                 node_list = self.generate_one_node(only_one=True)
        #             for c_node in node_list:
        #                 self.struct_tree.add_sub_tree(r_node, c_node)
        # else:
        # if 'fish' in args.task:
        #     self.struct_tree = Tree(get_original_fish())
        #
        # elif 'walker' in args.task:
        #     self.struct_tree = Tree(get_original_walker())
        #
        # elif 'cheetah' in args.task:
        #     self.struct_tree = Tree(get_original_cheetah())

        # if 'hopper' in args.task:
        self.struct_tree = Tree(get_original_hopper())

        return None

    def gen_test_node_attr(self, task='fish', node_num=5, discrete_rv=True):
        '''
        '''

        def get_uniform(low, high, discrete=True, total_lvl=6, avoid_zero=True):
            ''' uniformly sample from [low, high]
            if given discrete, it will sample from [low, high] at total_lvl
            steps
            '''
            assert total_lvl > 0, 'Invalid number of step %d' % total_lvl

            if discrete:
                step = (high - low) / total_lvl
                val = low + step * random.randint(1, total_lvl)
            else:
                val = float(np.random.uniform(low, high, 1))

            if avoid_zero:
                val += 1e-9
            return val

        def root_default(task):
            root_attr = {}

            if 'fish' in task:
                # trivial information that shouldn't be used
                root_attr['geom_type'] = -1
                root_attr['u'] = -1
                root_attr['v'] = -1
                root_attr['axis_x'] = -1
                root_attr['axis_y'] = -1
                # this is the default value as in dm_control original repo
                root_attr['a_size'] = 0.0075
                root_attr['b_size'] = 0.06
                root_attr['c_size'] = root_attr['a_size'] * 4
                #
                root_attr['joint_range'] = 60
            elif 'walker' in task:
                # some trivial attributes
                root_attr['geom_type'] = -1
                root_attr['u'] = -1
                root_attr['v'] = -1
                root_attr['axis_x'] = -1
                root_attr['axis_y'] = -1
                root_attr['joint_range'] = 60
                # setting the torso, using ellipsoid to approximate cylinder
                root_attr['a_size'] = 0.07
                root_attr['b_size'] = 0.3
                root_attr['c_size'] = -1
            elif 'hopper' in task:
                root_attr = CREATURE_ROOT_INFO['hopper']
            elif 'cheetah' in task:
                root_attr = CREATURE_ROOT_INFO['cheetah']
            else:
                assert 0, 'task: %s, not supported' % task

            return root_attr

        def gen_one_attr(task, discrete_rv=True):
            '''
            '''
            node_attr = {}

            # define geom type
            node_attr['geom_type'] = random.randint(0, 0)

            if 'fish' in task:
                constraint = CREATURE_HARD_CONSTRAINT['fish']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])

            elif 'walker' in task:
                constraint = CREATURE_HARD_CONSTRAINT['walker']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])

            elif 'hopper' in task:
                constraint = CREATURE_HARD_CONSTRAINT['hopper']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])
            elif 'cheetah' in task:
                constraint = CREATURE_HARD_CONSTRAINT['walker']
                needed_attr = ['u', 'v', 'axis_x', 'axis_y',
                               'a_size', 'b_size', 'c_size', 'joint_range']
                for item in needed_attr:
                    low, high = constraint[item]
                    node_attr[item] = get_uniform(
                        low, high, discrete=discrete_rv
                    )
                    if item == 'joint_range':
                        node_attr[item] = int(node_attr[item])

            return node_attr

        node_attr_list = []
        for i in range(node_num):
            node_attr = gen_one_attr(task, discrete_rv=discrete_rv)
            node_attr_list.append(node_attr)

        node_attr_list[0] = root_default(task)
        return node_attr_list

    def get_gene(self):
        '''
        '''
        adj_mat, node_attr = self.struct_tree.to_mujoco_format()

        # if 'fish' in self.args.task:
        #     adj_mat[adj_mat > 0] = 7
        # elif 'walker' in self.args.task or \
        #         'cheetah' in self.args.task or \
        #         'hopper' in self.args.task:
        adj_mat[adj_mat > 0] = 2

        return adj_mat, node_attr

    def get_xml(self):
        '''
        '''
        adj_mat, node_attr = self.get_gene()

        # if 'fish' in self.args.task:
        #     xml_struct = model_gen.fish_xml_generator(adj_mat, node_attr, options=None)
        # elif 'walker' in self.args.task:
        #     xml_struct = model_gen.walker_xml_generator(adj_mat, node_attr, options=None)
        # elif 'hopper' in self.args.task:
        #     xml_struct = model_gen.hopper_xml_generator(adj_mat, node_attr, options=None)
        # elif 'cheetah' in self.args.task:
        #     xml_struct = model_gen.cheetah_xml_generator(adj_mat, node_attr, options=None)

        xml_struct = hopper_xml_generator(adj_mat, node_attr, options=None)
        xml_str = etree.tostring(xml_struct, pretty_print=True)
        return xml_struct, xml_str

    def dfs_order2(self, adj_mat):
        '''
            return the order of parent-child relationship in the tree structure
            described by the adj_matrix
            input:
                1. adj_mat an N x N matrix in which index 0 is the root
                2. the matrix is symmetric
            output:
                1. a list of '%d-%d's decribing the order of parent-child relationship
                   run using dfs
        '''

        def dfs_order_helper(adj_mat, node_id, cur_order):
            '''
            '''
            # get child_list
            node_row = np.copy(adj_mat[node_id])
            for i in range(node_id + 1): node_row[i] = 0
            child_list = np.where(node_row)[0].tolist()

            for child_node in child_list:
                edge = '%d-%d' % (node_id, child_node)
                cur_order.append(edge)
                dfs_order_helper(adj_mat, child_node, cur_order)

            return cur_order

        # using recursion to solve the problem
        dfs_order = []
        dfs_order = dfs_order_helper(adj_mat, 0, dfs_order)
        return dfs_order

    def gaussian_noise(self, mu, std, discrete, step_size=None, ):
        '''
        '''
        if discrete and (step_size is None):
            raise RuntimeError('Gaussian noise cannot handle discrete without a step')

        if not discrete:
            noise = float(np.random.normal(mu, std, 1))
            return noise

        # sample by a step size if discrete
        noise = int(np.around(np.random.normal(0, 1, 1)))
        return noise * step_size

    def perturb_one_local(self, node_attr, task='fish', perturb_geom=False, discrete=True):
        ''' perform local perturbation according to current attributes
        '''
        new_attr = deepcopy(node_attr)

        # if 'fish' in task:
        #     constraint = CREATURE_HARD_CONSTRAINT['fish']
        # elif 'walker' in task:
        #     constraint = CREATURE_HARD_CONSTRAINT['walker']
        # elif 'cheetah' in task:
        #     constraint = CREATURE_HARD_CONSTRAINT['cheetah']
        # elif 'hopper' in task:
        constraint = CREATURE_HARD_CONSTRAINT['hopper']

        attr_list = ['u', 'v', 'axis_x', 'axis_y',
                     'a_size', 'b_size', 'c_size', 'joint_range']

        for attr in attr_list:
            low, high = constraint[attr]
            step_size = (high - low) / 6  # ideally this should be a hyperparameter to tune

            new_attr[attr] = node_attr[attr] + \
                             self.gaussian_noise(0, step_size / 2,
                                                           discrete, step_size)
            new_attr[attr] = \
                float(np.clip(new_attr[attr], low, high))
            if 'joint_range' == attr:
                new_attr[attr] = int(new_attr[attr])

        return new_attr

    def perturb_one_attr(self, node_attr, task='fish', perturb_geom=False, perturb_discrete=True):
        '''
            brief: perturb the 7D parameters

            NOTE: some basic stuff
                        1. consider whether the perturbation physically makes sense
                            (the perturbation will always makes sense
                            if we have reasonable upper and lower bounds)
        '''

        new_attr = self.perturb_one_local(node_attr, task=task,
                                     discrete=perturb_discrete
                                     )
        return new_attr



    def generate_one_node(self, only_one=False):
        ''' return a list of new nodes being generated
        '''
        NEW_STRUCT_OPTS = ['l1-basic', 'l2-symmetry']
        # different policy for adding new hierarchical structure
        if self.allow_hierarchy:
            choice = random.choice(NEW_STRUCT_OPTS)
        else:
            choice = NEW_STRUCT_OPTS[0]

        if self.force_symmetric:
            choice = 'l2-symmetry'
        if only_one:
            choice = 'l1-basic'
        # if self.args.walker_force_no_sym:
        #     choice = 'l1-basic'

        new_node_list = []
        if choice == 'l1-basic':
            # adding one basic node
            new_node = Node(-1,
                            self.gen_test_node_attr(task='hopper',
                                                         node_num=2, discrete_rv=self.discrete
                                                         )[-1]
                            )
            new_node_list.append(new_node)
        elif choice == 'l2-symmetry':
            # adding 2 symmetric nodes
            attr1, attr2 = [
                self.gen_test_node_attr(task='hopper',
                                             node_num=2,
                                             discrete_rv=self.discrete
                                             )[-1]
                for i in range(2)
            ]
            same_attr = ['a_size', 'b_size', 'c_size', 'geom_type',
                         'joint_range']
            for item in same_attr:
                attr2[item] = attr1[item]
            # set certain attributes
            attr2['u'] = np.pi - attr1['u']
            while attr2['u'] < 0: attr2['u'] += 2 * np.pi
            attr2['v'] = attr1['v']
            attr2['axis_x'] = attr1['axis_x']
            attr2['axis_y'] = -attr1['axis_y']
            # create the new node with corresponding attributes
            node1 = Node(-1, attr1)
            node2 = Node(-1, attr2)
            new_node_list.append(node1)
            new_node_list.append(node2)
        else:
            raise NotImplementedError

        # if self.args.force_grow_at_ends:
        #     for node in new_node_list:
        #         node.attr['axis_x'] = 1 if node.attr['axis_x'] >= 0 else -1

        return new_node_list

    def perturb_add(self, no_selfcross=False):
        '''
        '''
        debug_info = {}
        debug_info['op'] = 'add_node'

        # prepare debug info -- for updating weights in GGNN
        edge_dfs = self.dfs_order2(self.get_gene()[0])
        dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
        debug_info['old_order'] = deepcopy(dfs_order)
        debug_info['old_nodes'] = sorted(self.struct_tree.root.get_all_node_id())
        debug_info['new_nodes'] = sorted(self.struct_tree.root.get_all_node_id())

        debug_info['parent'] = []

        all_nodes = self.struct_tree.get_all_nodes()
        one_node_added = False

        while one_node_added == False:
            # iterate through all the nodes
            for node in all_nodes:
                not_add_node = np.random.binomial(1, 1 - self.p)

                if not_add_node:
                    continue

                # if 'walker' in self.args.task and self.args.walker_more_constraint:
                #     # check the node type and childs
                #     if len(node.get_child_list()) >= 2: continue
                #     if node is not self.struct_tree.get_root():
                #         if len(node.get_child_list()) >= 1: continue

                one_node_added = True

                # sample a place to add the new struct
                one_node = node
                debug_info['parent'].append(one_node.get_node_id())


                use_selfcross = np.random.binomial(1, self.self_cross_ratio)
                if no_selfcross or not use_selfcross:
                    new_node_list = self.generate_one_node()
                else:
                    debug_info['other'] = 'using self cross'
                    self_struct = random.choice(all_nodes)
                    while self_struct is self.struct_tree.get_root():
                        self_struct = random.choice(all_nodes)
                    self_struct = deepcopy(self_struct)
                    new_node_list = [self_struct]

                for node in new_node_list:
                    self.struct_tree.add_sub_tree(one_node, node)
                    debug_info['new_nodes'].append(-1)

        # new order, still the dfs order with new element value being -1
        prev_node_num = len(debug_info['old_order'])
        edge_dfs = self.dfs_order2(self.get_gene()[0])
        dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
        debug_info['new_order'] = deepcopy(dfs_order)
        debug_info['new_order'] = [-1 if x >= prev_node_num else x
                                   for x in debug_info['new_order']
                                   ]

        return debug_info

    def perturb_remove(self):
        '''
        '''
        # randomly sample a node and remove all its subtree
        debug_info = {}
        debug_info['op'] = 'rm_node'
        edge_dfs = self.dfs_order2(self.get_gene()[0])
        dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
        debug_info['old_order'] = deepcopy(dfs_order)
        debug_info['new_order'] = deepcopy(dfs_order)
        debug_info['old_nodes'] = sorted(self.struct_tree.root.get_all_node_id())
        debug_info['new_nodes'] = sorted(self.struct_tree.root.get_all_node_id())

        # sample the parent node to remove from
        one_node = self.struct_tree.sample_node()
        while len(one_node.get_child_list()) == 0:
            one_node = self.struct_tree.sample_node()

        delete_list = []
        bak_list = []
        remove_node_id_list = []
        # sample a child to start with
        child_node = random.choice(one_node.get_child_list())
        delete_list.append(child_node)
        bak_list.append(deepcopy(child_node))
        remove_node_id_list += child_node.get_all_node_id()

        if self.force_symmetric:
            # also find the node of the same size among the child
            for node in one_node.get_child_list():
                if node is child_node:
                    continue
                if node.is_same_size(child_node):
                    delete_list.append(node)
                    bak_list.append(deepcopy(node))
                    remove_node_id_list += node.get_all_node_id()
                    break

        # remove subtree from the parent
        for node_to_delete in delete_list:
            self.struct_tree.remove_sub_tree(one_node, node_to_delete)

        # if there is only root left, add everything back
        if self.struct_tree.total_num == 1:
            debug_info['op'] = 'none'
            root = self.struct_tree.sample_node()
            for bak_node in bak_list:
                self.struct_tree.add_sub_tree(root, bak_node)
            return debug_info

        debug_info['delete_node'] = []
        for rm_node_id in remove_node_id_list:
            debug_info['new_nodes'].remove(rm_node_id)
            debug_info['new_order'].remove(rm_node_id)
            debug_info['delete_node'].append(rm_node_id)
        return debug_info

    def perturb_attr(self):
        '''
        '''
        debug_info = {}
        debug_info['op'] = 'change_attr'
        all_nodes = self.struct_tree.get_all_nodes()

        one_node_mutate = False

        while one_node_mutate == False:
            for node in all_nodes:
                # no need to perturb the root
                if node is self.struct_tree.root: continue

                if np.random.binomial(1, self.p):
                    one_node_mutate = True

                    if self.force_symmetric:
                        # search for the one with the same size
                        node2 = None
                        for another_node in all_nodes:
                            if another_node is node: continue
                            if another_node.is_same_size(node):
                                node2 = another_node
                        if node2 is None:
                            import pdb;
                            pdb.set_trace()
                    # regardless of forcing symmetry or not, we need to make the
                    # perturbation
                    node.set_attr(
                        self.perturb_one_attr(node.get_attr(),
                                                       task='hopper',
                                                       perturb_discrete=self.discrete
                                                       )
                    )

                    if self.force_symmetric:
                        node2.set_attr(
                            node.make_symmetric('hopper',
                                                self.discrete
                                                ).get_attr()
                        )

                else:
                    pass

        return debug_info

    def mutate(self):
        '''
        '''
        # op = np.random.choice(
        #     ['add', 'remove', 'perturb'], 1,
        #     p=[self.mutation_add_ratio,
        #        self.mutation_delete_ratio, 1 - self.mutation_add_ratio - self.mutation_delete_ratio]
        # )

        op = 'perturb'
        debug_info = {}
        debug_info['old_mat'] = self.get_gene()[0]
        debug_info['old_pc'] = self.struct_tree.get_pc_relation()



        # setting some hard constraint,
        # fixing the maximum and minimum number of nodes allowed in the body
        # if len(self.get_gene()[1]) >= 10 and op == 'add':
        #     op = 'perturb'
        # if len(self.get_gene()[1]) == 2 and op == 'remove':
        #     op = 'perturb'

        if op == 'add':
            perturb_info = self.perturb_add()
        elif op == 'remove':
            perturb_info = self.perturb_remove()
        elif op == 'perturb':
            perturb_info = self.perturb_attr()

        debug_info = {**debug_info, **perturb_info}

        return debug_info

def get_original_hopper():
    '''
    '''
    r_node = Node(0, CREATURE_ROOT_INFO['hopper'])
    pelvis = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][0])
    thigh = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][1])
    calf = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][2])
    foot = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][3])

    calf.add_child(foot)
    thigh.add_child(calf)
    pelvis.add_child(thigh)
    r_node.add_child(pelvis)

    return r_node

import os.path as osp
import sys
import datetime
import pdb

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# get current time
running_start_time = datetime.datetime.now()
time = str(running_start_time.strftime("%Y_%m_%d-%X"))

# get current file path
_this_dir = osp.dirname(__file__)
_base_dir = osp.join(_this_dir)
add_path(_base_dir)
print(_base_dir)


def init_path():
    ''' function to be called in the beginning of the file
    '''
    _this_dir = osp.dirname(__file__)
    _base_dir = osp.join(_this_dir, '..')
    add_path(_base_dir)
    return None


def bypass_frost_warning():
    return 0


def get_base_dir():
    return _base_dir


def get_time():
    return time


def get_abs_base_dir():
    return osp.abspath(_base_dir)


def xml_string_to_file(xml, filename):
    '''
    '''
    if filename == None:
        return

    with open(filename, 'wb') as fd:
        fd.write(xml)
    return




# making struct_tree
body_part_num = 10
spc = Species(body_num=body_part_num)
max_evo_step = 2

for i in range(max_evo_step):
    # generate hopper xml file
    adj_mat, node_attr = spc.get_gene()
    xml_struct, xml_str = spc.get_xml()
    file_path = os.path.join(get_base_dir(),
        'test_hierarchy_perturb.xml'
    )
    xml_string_to_file(xml_str, file_path)

    debug_info = spc.mutate()
    print('step # %d, Mutate option: %s' %(i,debug_info['op']))

xml_path = file_path
model = load_model_from_path(xml_path)
sim = MjSim(model)

t = 0
viewer = MjViewer(sim)
ls_wow=[]


print('process')

# print(node_info['tree'])
while True:
    t += 1
    sim.step()
    viewer.render()



