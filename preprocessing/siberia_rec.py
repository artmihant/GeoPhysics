import numpy as np
from cae_models import FCModel
from scipy.spatial import KDTree

from cae_models.fc import FCBlock
from os.path import join

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
        # print(p)
        p = np.linalg.det([
            [ax,1],
            [bx,1],
        ])
        # print(p)
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

    fc_main_zero = join(datapath, 'siberia_main_rec212_bc.fc')
    fc_top_zero = join(datapath, 'siberia_zero_top.fc')
    # fc_main_rec = join(datapath, 'model5_rec.fc')

    # center = [
    #     439025.000000,
    #     7839165.000000
    # ]
    
    center = [
        439021.25,
        7839164.0,
    ]

    depth = -2

    recievers_area = 231, 52
    step = 50, 300

    # recievers_area = 400, 400
    # step = 25, 25

    recievers_area = 35, 315
    # recievers_area = 1, 1

    step = 300, 50


    # Число слоев элементов в модели
    layers = 143

    radius = 4000


def main():

    fc_main = FCModel(CONFIG.fc_main_zero)

    fc_main.save(CONFIG.fc_main_zero)
    fc_top = FCModel(CONFIG.fc_top_zero)

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

    # fc_main.save(CONFIG.fc_main_rec)


if __name__ == '__main__':
    main()


