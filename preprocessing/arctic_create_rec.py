from os.path import join
import segyio
import numpy as np

from cae_models import FCModel
from cae_models.fc import FCBlock
from scipy.spatial import KDTree


# def local_coords_2D(ax,ay, bx,by, cx,cy, px,py):

#     pass
#     p = np.linalg.det([
#         [ax,ay,1],
#         [bx,by,1],
#         [cx,cy,1],
#     ])
#     if abs(p) > 10:

#         a = np.linalg.det([
#             [px,py,1],
#             [bx,by,1],
#             [cx,cy,1],
#         ])

#         b = np.linalg.det([
#             [ax,ay,1],
#             [px,py,1],
#             [cx,cy,1],
#         ])

#         c = np.linalg.det([
#             [ax,ay,1],
#             [bx,by,1],
#             [px,py,1],
#         ])

#         return [
#             a/p, b/p, c/p
#         ]
#     else:
#         p = np.linalg.det([
#             [ax,1],
#             [bx,1],
#         ])
#         if abs(p) > 0:

#             a = np.linalg.det([
#                 [px,1],
#                 [bx,1],
#             ])

#             b = np.linalg.det([
#                 [ax,1],
#                 [px,1],
#             ])

#             return [
#                 a/p, b/p, 0
#             ]
#         else:
#             return [1,0,0]


class CONFIG:
    datapath = "/home/artem/Projects/GeoPhysics/preprocessing/data"

    top = 98
    height = 2

    samples = 50

    # samples = 501

    elems_shape = (samples, 347, 283)

    nodes_shape = (elems_shape[0]+1, elems_shape[1]+1, elems_shape[2]+1)

    elems_count = elems_shape[0]*elems_shape[1]*elems_shape[2]
    nodes_count = nodes_shape[0]*nodes_shape[1]*nodes_shape[2]

    traces = elems_shape[1]*elems_shape[2]

    rho_path = join(datapath, 'sem3d_arctic_Rho.sgy')
    vp_path  = join(datapath, 'sem3d_arctic_Vp.sgy')
    vs_path  = join(datapath, 'sem3d_arctic_Vs.sgy')

    fc_zero  = join(datapath, 'arctic_zero_light.fc')
    fc_top  = join(datapath, 'arctic_top_light.fc')
    fc_abs  = join(datapath, 'arctic_abs_light.fc')
    fc_abs_res  = join(datapath, 'arctic_abs_res_light.fc')

    recievers_area = 100, 1
    step = 50, 1

    center = [
        603200.0,
        522825.0
    ]

    depth = -1

def main():

    ## Создаем модельку

    if True:

        ## Создаем геометрию модели

        top_coords = np.zeros((CONFIG.traces, 2), dtype=np.float32)

        with segyio.open(CONFIG.rho_path) as segyfile_rho:
            for i, header in enumerate(segyfile_rho.header):
                top_coords[i] = [
                    header[segyio.TraceField.CDP_X],
                    header[segyio.TraceField.CDP_Y],
                ]

        top_coords = top_coords.reshape(1, *CONFIG.elems_shape[1:], 2)

        elements_coords = np.zeros((*CONFIG.elems_shape, 3), dtype=np.float64)

        elements_coords[:,:,:,:2] = top_coords
        elements_coords[:,:,:,2] = np.arange(CONFIG.top,CONFIG.top-CONFIG.height*CONFIG.samples ,-CONFIG.height, dtype=np.float32).reshape(-1,1,1)

        nodes_coords = np.zeros((*CONFIG.nodes_shape, 3), dtype=np.float64)
        
        nodes_coords[1:-1,1:-1,1:-1] = (
            elements_coords[:-1,:-1,:-1] +
            elements_coords[:-1,:-1, 1:] +
            elements_coords[:-1, 1:,:-1] +
            elements_coords[:-1, 1:, 1:] +
            elements_coords[ 1:,:-1,:-1] +
            elements_coords[ 1:,:-1, 1:] +
            elements_coords[ 1:, 1:,:-1] +
            elements_coords[ 1:, 1:, 1:]
        )/8

        nodes_coords[(0,-1),:,:] = nodes_coords[(1,-2),:,:] + nodes_coords[(2,-3),:,:] - nodes_coords[(3,-4),:,:]

        nodes_coords[:,(0,-1),:] = nodes_coords[:,(1,-2),:] + nodes_coords[:,(2,-3),:] - nodes_coords[:,(3,-4),:]

        nodes_coords[:,:,(0,-1)] = nodes_coords[:,:,(1,-2)] + nodes_coords[:,:,(2,-3)] - nodes_coords[:,:,(3,-4)]

        nodes_ids = np.arange(CONFIG.nodes_count, dtype=np.int32).reshape(CONFIG.nodes_shape)+1

        fc_model = FCModel()

        fc_model.mesh.nodes.ids = nodes_ids.reshape(-1)
        fc_model.mesh.nodes.coords = nodes_coords.reshape(-1)

        elems_ids = np.arange(CONFIG.elems_count, dtype=np.int32).reshape(CONFIG.elems_shape)+1
        fc_model.mesh.elems.ids = elems_ids.reshape(-1)

        fc_model.mesh.elems.blocks = np.full(CONFIG.elems_count, 1, dtype=np.int32)
        fc_model.mesh.elems.orders = np.full(CONFIG.elems_count, 1, dtype=np.int32)
        fc_model.mesh.elems.parent_ids = np.full(CONFIG.elems_count, 1, dtype=np.int32)
        fc_model.mesh.elems.types = np.full(CONFIG.elems_count, 3, dtype=np.int8)

        elems_nodes = np.zeros((*CONFIG.elems_shape,8),dtype=np.int32)

        top_coords = np.zeros((CONFIG.traces, 2), dtype=np.float32)

        with segyio.open(CONFIG.rho_path) as segyfile_rho:
            for i, header in enumerate(segyfile_rho.header):
                top_coords[i] = [
                    header[segyio.TraceField.CDP_X],
                    header[segyio.TraceField.CDP_Y],
                ]

        top_coords = top_coords.reshape(1, *CONFIG.elems_shape[1:], 2)

        elements_coords = np.zeros((*CONFIG.elems_shape, 3), dtype=np.float64)

        elements_coords[:,:,:,:2] = top_coords
        elements_coords[:,:,:,2] = np.arange(CONFIG.top,CONFIG.top-CONFIG.height*CONFIG.samples ,-CONFIG.height, dtype=np.float32).reshape(-1,1,1)

        nodes_coords = np.zeros((*CONFIG.nodes_shape, 3), dtype=np.float64)
        
        nodes_coords[1:-1,1:-1,1:-1] = (
            elements_coords[:-1,:-1,:-1] +
            elements_coords[:-1,:-1, 1:] +
            elements_coords[:-1, 1:,:-1] +
            elements_coords[:-1, 1:, 1:] +
            elements_coords[ 1:,:-1,:-1] +
            elements_coords[ 1:,:-1, 1:] +
            elements_coords[ 1:, 1:,:-1] +
            elements_coords[ 1:, 1:, 1:]
        )/8

        nodes_coords[(0,-1),:,:] = nodes_coords[(1,-2),:,:] + nodes_coords[(2,-3),:,:] - nodes_coords[(3,-4),:,:]

        nodes_coords[:,(0,-1),:] = nodes_coords[:,(1,-2),:] + nodes_coords[:,(2,-3),:] - nodes_coords[:,(3,-4),:]

        nodes_coords[:,:,(0,-1)] = nodes_coords[:,:,(1,-2)] + nodes_coords[:,:,(2,-3)] - nodes_coords[:,:,(3,-4)]

        nodes_ids = np.arange(CONFIG.nodes_count, dtype=np.int32).reshape(CONFIG.nodes_shape)+1

        fc_model = FCModel()

        fc_model.mesh.nodes.ids = nodes_ids.reshape(-1)
        fc_model.mesh.nodes.coords = nodes_coords.reshape(-1)

        elems_ids = np.arange(CONFIG.elems_count, dtype=np.int32).reshape(CONFIG.elems_shape)+1
        fc_model.mesh.elems.ids = elems_ids.reshape(-1)

        fc_model.mesh.elems.blocks = np.full(CONFIG.elems_count, 1, dtype=np.int32)
        fc_model.mesh.elems.orders = np.full(CONFIG.elems_count, 1, dtype=np.int32)
        fc_model.mesh.elems.parent_ids = np.full(CONFIG.elems_count, 1, dtype=np.int32)
        fc_model.mesh.elems.types = np.full(CONFIG.elems_count, 3, dtype=np.int8)

        elems_nodes = np.zeros((*CONFIG.elems_shape,8),dtype=np.int32)

        elems_nodes[:,:,:,0] = nodes_ids[:-1,:-1,1:]
        elems_nodes[:,:,:,1] = nodes_ids[:-1,:-1,:-1]
        elems_nodes[:,:,:,2] = nodes_ids[:-1,1:,:-1]
        elems_nodes[:,:,:,3] = nodes_ids[:-1,1:,1:]
        elems_nodes[:,:,:,4] = nodes_ids[1:,:-1,1:]
        elems_nodes[:,:,:,5] = nodes_ids[1:,:-1,:-1]
        elems_nodes[:,:,:,6] = nodes_ids[1:,1:,:-1]
        elems_nodes[:,:,:,7] = nodes_ids[1:,1:,1:]

        elems_nodes = elems_nodes.reshape(-1,8)


        fc_model.mesh.elems.nodes = elems_nodes

        ## Задаем материалы модели

        Rho_data = np.zeros((CONFIG.samples, CONFIG.traces), dtype=np.float64)
        Vp_data  = np.zeros((CONFIG.samples, CONFIG.traces), dtype=np.float64)
        Vs_data  = np.zeros((CONFIG.samples, CONFIG.traces), dtype=np.float64)


        with segyio.open(CONFIG.rho_path) as segyfile_rho:
            for index, trace in enumerate(segyfile_rho.trace):
                Rho_data[:, index] = trace[:CONFIG.samples]

        Rho_data = Rho_data.reshape(CONFIG.elems_shape)

        with segyio.open(CONFIG.vp_path) as segyfile_vp:
            for index, trace in enumerate(segyfile_vp.trace):
                Vp_data[:, index] = trace[:CONFIG.samples]

        Vp_data = Vp_data.reshape(CONFIG.elems_shape)



        with segyio.open(CONFIG.vs_path) as segyfile_vs:
            for index, trace in enumerate(segyfile_vs.trace):
                Vs_data[:, index] = trace[:CONFIG.samples]

        Vs_data = Vs_data.reshape(CONFIG.elems_shape)

        E_data = np.zeros(CONFIG.elems_shape, dtype=np.float64)
        Nu_data = np.zeros(CONFIG.elems_shape, dtype=np.float64)

        V_delta = Vp_data ** 2 - Vs_data ** 2

        elems_mask = V_delta > 0

        E_data[elems_mask] = (Rho_data * Vs_data ** 2 * (3 * Vp_data ** 2 - 4 * Vs_data ** 2))[elems_mask] / V_delta[elems_mask]
        Nu_data[elems_mask] = ((Vp_data**2 - 2 * Vs_data ** 2) / 2)[elems_mask] / V_delta[elems_mask]

        E_data = E_data[elems_mask]
        Nu_data = Nu_data[elems_mask]
        Rho_data = Rho_data[elems_mask]
        Vp_data = Vp_data[elems_mask]
        Vs_data = Vs_data[elems_mask]

        print(Vp_data.min(), Vp_data.max())
        print(Vs_data.min(), Vs_data.max())

        elems_mask = elems_mask.reshape(-1)

        fc_model.mesh.elems.blocks = fc_model.mesh.elems.blocks[elems_mask]
        fc_model.mesh.elems.orders = fc_model.mesh.elems.orders[elems_mask]
        fc_model.mesh.elems.parent_ids = fc_model.mesh.elems.parent_ids[elems_mask]
        fc_model.mesh.elems.types = fc_model.mesh.elems.types[elems_mask]
        fc_model.mesh.elems.ids = fc_model.mesh.elems.ids[elems_mask]
        fc_model.mesh.elems.nodes = fc_model.mesh.elems.nodes[elems_mask]

        material_nids = np.arange(CONFIG.elems_count, dtype=np.float64)+1

        material_nids = material_nids[elems_mask]

        fc_model.materials = [{
            'id': 1,
            'name': 'Soil',
            'properties':{
                'common':[{
                    'name':0,
                    'data':Rho_data.reshape(-1),
                    'type': 0,
                    'dependency':[{
                        'type': 10,
                        'data': material_nids
                    }]
                }],
                'elasticity':[{
                    'name':0,
                    'data':E_data.reshape(-1),
                    'type': 0,
                    'dependency':[{
                        'type': 10,
                        'data': material_nids
                    }]
                },{
                    'name':1,
                    'data':Nu_data.reshape(-1),
                    'type': 0,
                    'dependency':[{
                        'type': 10,
                        'data': material_nids
                    }]
                }],
            },

        }]

        fc_model.blocks = [FCBlock({
            "cs_id" : 1,
            "id" : 1,
            "material_id" : 1,
            "property_id" : -1
        })]
        

        fc_model.mesh.elems.ids = np.arange(len(fc_model.mesh.elems), dtype=np.int32) + 1

        if type(fc_model.materials[0]['properties']['common'][0]['dependency']) == list:
            fc_model.materials[0]['properties']['common'][0]['dependency'][0]['data'] = np.arange(len(fc_model.mesh.elems), dtype=np.float64) + 1

        if type(fc_model.materials[0]['properties']['elasticity'][0]['dependency']) == list:
            fc_model.materials[0]['properties']['elasticity'][0]['dependency'][0]['data'] = np.arange(len(fc_model.mesh.elems), dtype=np.float64) + 1

        if type(fc_model.materials[0]['properties']['elasticity'][1]['dependency']) == list:
            fc_model.materials[0]['properties']['elasticity'][1]['dependency'][0]['data'] = np.arange(len(fc_model.mesh.elems), dtype=np.float64) + 1

        fc_model.save(CONFIG.fc_zero)

    ## За кадром: расставляем абсорбции и считываем верхнюю грань

    # Расставляем приемники

    if False:

        print('start')

        fc_main = FCModel(CONFIG.fc_abs)
        print('Read main complite')

        fc_top = FCModel(CONFIG.fc_top)
        print('Read top complite')

        top_coords = fc_top.mesh.nodes.coords

        recievers = np.zeros((CONFIG.recievers_area[0]*CONFIG.recievers_area[1],3), dtype=np.float64)

        kd_tree_coord = KDTree(top_coords[:,:2])

        for j in range(CONFIG.recievers_area[1]):
            for i in range(CONFIG.recievers_area[0]):

                x = CONFIG.center[0]+(i-(CONFIG.recievers_area[0]-1)/2)*CONFIG.step[0]
                y = CONFIG.center[1]+(j-(CONFIG.recievers_area[1]-1)/2)*CONFIG.step[1]

                closed = kd_tree_coord.query([x,y], k=4)

                point0 = top_coords[closed[1][0]]
                point1 = top_coords[closed[1][1]]
                point2 = top_coords[closed[1][2]]
                point3 = top_coords[closed[1][3]]

                # l = local_coords_2D(
                #     point0[0],point0[1],
                #     point1[0],point1[1], 
                #     point2[0],point2[1], 
                #     x, y
                # )

                z = min(point0[2], point1[2], point2[2], point3[2]) + CONFIG.depth

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


        fc_main.save(CONFIG.fc_abs_res)



if __name__ == '__main__':
    main()