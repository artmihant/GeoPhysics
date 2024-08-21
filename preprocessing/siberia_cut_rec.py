from typing import Callable
import numpy as np
from numpy import ndarray
from cae_models import FCModel
from scipy.spatial import KDTree
from os.path import join

from cae_models.fc import FCBlock


def local_coords_2D(ax,ay, bx,by, cx,cy, px,py):
    p = np.linalg.det([
        [ax,ay,1],
        [bx,by,1],
        [cx,cy,1],
    ])
    if abs(p) > 1000:

        a = np.linalg.det([
            [px,py,1],
            [bx,by,1],
            [cx,cy,1],
        ])

        b = np.linalg.det([
            [ax,ay,1],
            [px,py,1],
            [cx,cy,1],
        ])

        c = np.linalg.det([
            [ax,ay,1],
            [bx,by,1],
            [px,py,1],
        ])

        return [
            a/p, b/p, c/p
        ]
    else:
        p = np.linalg.det([
            [ax,1],
            [bx,1],
        ])
        if abs(p) > 0:

            a = np.linalg.det([
                [px,1],
                [bx,1],
            ])

            b = np.linalg.det([
                [ax,1],
                [px,1],
            ])

            return [
                a/p, b/p, 0
            ]
        else:
            return [1,0,0]


class CONFIG:
    datapath = "/home/artem/Projects/GeoPhysics/preprocessing/data"

    # fc_main_src = join(datapath, 'siberia_src_rec_fix_bc.fc')
    fc_main_zero = join(datapath, 'siberia_zero.fc')
    fc_top_zero = join(datapath, 'siberia_zero_top.fc')
    fc_top_zero_cut = join(datapath, 'siberia_cut8_top.fc')

    fc_main_zero_cut = join(datapath, 'siberia_cut8.fc')
    fc_main_cut_ab = join(datapath, 'siberia_cut8_abs.fc')
    fc_main_cut_ab_res = join(datapath, 'siberia_cut8_rec212.fc')

    center = [
        439021.25,
        7839164.0
    ]

    depth = -2

    # depth_rec = -2
    # depth_bang = -10

    recievers_area = 141, 22
    step = 50, 300

    # Число слоев элементов в модели
    layers = 143

    radius = 4000

def cut(mesh, nodes_mask):

    mesh.nodes.coords = mesh.nodes.coords[nodes_mask]
    mesh.nodes.ids = mesh.nodes.ids[nodes_mask]

    node_set = set(mesh.nodes.ids)

    size = mesh.elems.sizes[0]

    assert len(mesh.elems.nodes) == size*len(mesh.elems)

    elems_nodes = mesh.elems.nodes.reshape(-1,size)

    elems_mask = [node_set.issuperset(elem) for elem in elems_nodes]

    mesh.elems.blocks = mesh.elems.blocks[elems_mask]
    mesh.elems.orders = mesh.elems.orders[elems_mask]
    mesh.elems.parent_ids = mesh.elems.parent_ids[elems_mask]
    mesh.elems.types = mesh.elems.types[elems_mask]

    mesh.elems.ids = mesh.elems.ids[elems_mask]

    mesh.elems.nodes = elems_nodes[elems_mask].reshape(-1)

    mesh.elems.update()

    return mesh


def main():

    ## Удаляем старые ресиверы и абсорбцию и реиндексируем узлы ##

    if False:

        fc_main = FCModel(CONFIG.fc_main_src)

        fc_main.loads = []
        fc_main.receivers = []

        sorted_node_ids = np.sort(fc_main.mesh.nodes.ids)
        
        fc_main.mesh.nodes.ids = np.searchsorted(sorted_node_ids, fc_main.mesh.nodes.ids).astype(np.int32)+1
        fc_main.mesh.elems.nodes = np.searchsorted(sorted_node_ids, fc_main.mesh.elems.nodes).astype(np.int32)+1

        fc_main.save(CONFIG.fc_main_zero)

    ## За кадром: извлекаем из fc_main_zero top - верхнюю грань

    ## Обрезаем в размерах верхнюю грань ##

    if False:

        fc_top = FCModel(CONFIG.fc_top_zero)

        def area_selector(coord):
            return abs(coord[0]-CONFIG.center[0]) < CONFIG.radius and abs(coord[1]-CONFIG.center[1]) < CONFIG.radius

        nodes_mask = [area_selector(coord) for coord in fc_top.mesh.nodes.coords]

        fc_top.mesh = cut(fc_top.mesh, nodes_mask)

        fc_top.save(CONFIG.fc_top_zero_cut)

    ## Обрезаем в размерах всё по верхней грани ##

    if False:

        fc_top = FCModel(CONFIG.fc_top_zero_cut)
        fc_main = FCModel(CONFIG.fc_main_zero)

        nids = fc_top.mesh.nodes.ids

        area = len(fc_main.mesh.nodes)//(CONFIG.layers+1)

        def index_selector(index):
            return (index%area + 1) in nids

        nodes_mask = np.vectorize(index_selector)(fc_main.mesh.nodes.ids)

        fc_main.mesh = cut(fc_main.mesh, nodes_mask)

        material = fc_main.materials[0]

        for key in material['properties']:
            for property in material['properties'][key]:

                if isinstance(property['dependency'], list) and property['dependency']:
                    for dep in property['dependency']:

                        dep_data = dep['data']
                        property_data = property['data']

                        if isinstance(dep_data, ndarray) and isinstance(property_data, ndarray):

                            if dep['type'] == 10:
                                mat_mask = np.isin(dep_data, fc_main.mesh.elems.ids, assume_unique=True)
                                dep['data'] = dep_data[mat_mask]
                                property['data'] = property_data[mat_mask]

        sorted_node_ids = np.sort(fc_main.mesh.nodes.ids)
        
        fc_main.mesh.nodes.ids = np.searchsorted(sorted_node_ids, fc_main.mesh.nodes.ids).astype(np.int32)+1
        fc_main.mesh.elems.nodes = np.searchsorted(sorted_node_ids, fc_main.mesh.elems.nodes).astype(np.int32)+1


        fc_main.mesh.elems.ids = np.arange(len(fc_main.mesh.elems), dtype=np.int32) + 1

        fc_main.materials[0]['properties']['common'][0]['dependency'][0]['data'] = np.arange(len(fc_main.mesh.elems), dtype=np.float64) + 1
        fc_main.materials[0]['properties']['elasticity'][0]['dependency'][0]['data'] = np.arange(len(fc_main.mesh.elems), dtype=np.float64) + 1
        fc_main.materials[0]['properties']['elasticity'][1]['dependency'][0]['data'] = np.arange(len(fc_main.mesh.elems), dtype=np.float64) + 1

        fc_main.save(CONFIG.fc_main_zero_cut)

    ## За кадром: расставляем абсорбции

    # # Расставляем приемники

    if True:

        fc_main = FCModel(CONFIG.fc_main_cut_ab)
        fc_top = FCModel(CONFIG.fc_top_zero_cut)

        absorption = fc_main.loads[0]['apply_to'].astype(np.int32)

        print(absorption, len(absorption))

        top_coords = fc_top.mesh.nodes.coords

        recievers = np.zeros((CONFIG.recievers_area[0]*CONFIG.recievers_area[1],3), dtype=np.float64)

        kd_tree_coord = KDTree(top_coords[:,:2])

        for j in range(CONFIG.recievers_area[1]):
            for i in range(CONFIG.recievers_area[0]):

                x = CONFIG.center[0]+(i-(CONFIG.recievers_area[0]-1)/2)*CONFIG.step[0]
                y = CONFIG.center[1]+(j-(CONFIG.recievers_area[1]-1)/2)*CONFIG.step[1]

                closed = kd_tree_coord.query([x,y], k=3)

                point0 = top_coords[closed[1][0]]
                point1 = top_coords[closed[1][1]]
                point2 = top_coords[closed[1][2]]

                l = local_coords_2D(
                    point0[0],point0[1],
                    point1[0],point1[1], 
                    point2[0],point2[1], 
                    x, y
                )

                z = l[0]*point0[2] + l[1]*point1[2] + l[2]*point2[2] + CONFIG.depth

                print(f'"{i}_{j}": [{x}, {y}, {z}],')

                recievers[j*CONFIG.recievers_area[0]+i,:] = x,y,z


        node_index = fc_main.mesh.nodes.ids.max()
        element_index = fc_main.mesh.elems.ids.max()

        nodes_ids = []
        nodes_coords = []

        elems_blocks = []
        elems_orders = []
        elems_parent_ids = []
        elems_types = []
        elems_ids = []
        elems_nodes = []
        
        for node in recievers:
            node_index += 1
            element_index += 1

            nodes_ids.append(node_index)
            nodes_coords.append(node)

            elems_blocks.append(2)
            elems_orders.append(1)
            elems_parent_ids.append(2)
            elems_types.append(101)
            elems_ids.append(element_index)
            elems_nodes.append(node_index)


        fc_main.mesh.nodes.ids = np.concatenate((fc_main.mesh.nodes.ids, nodes_ids), dtype=np.int32) 
        fc_main.mesh.nodes.coords = np.concatenate((fc_main.mesh.nodes.coords, nodes_coords), dtype=np.float64) 

        fc_main.mesh.elems.blocks = np.concatenate((fc_main.mesh.elems.blocks, elems_blocks), dtype=np.int32)
        fc_main.mesh.elems.orders = np.concatenate((fc_main.mesh.elems.orders, elems_orders), dtype=np.int32)
        fc_main.mesh.elems.parent_ids = np.concatenate((fc_main.mesh.elems.parent_ids, elems_parent_ids), dtype=np.int32)
        fc_main.mesh.elems.types = np.concatenate((fc_main.mesh.elems.types, elems_types), dtype=np.int8)
        fc_main.mesh.elems.ids = np.concatenate((fc_main.mesh.elems.ids, elems_ids), dtype=np.int32)
        fc_main.mesh.elems.nodes = np.concatenate((fc_main.mesh.elems.nodes, elems_nodes), dtype=np.int32)

        fc_main.receivers = [{
            'apply_to':np.array(nodes_ids, dtype=np.int32),
            'dofs': [1,1,1],
            'id': 1,
            'name': 'Receivers',
            'type': 1
        }]

        fc_main.blocks.append(FCBlock({
            'id':2,
            'cs_id':1,
            'material_id':0,
            'property_id':-1
        }))


        fc_main.save(CONFIG.fc_main_cut_ab_res)

    ## Готово, можно отправлять на расчет!

if __name__ == '__main__':
    main()


