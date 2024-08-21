import json
from base64 import b64decode, b64encode
from typing import Callable, Optional, TypeVar, TypedDict
import numpy as np
from .element_types import ELEMENT_TYPES

from numpy import ndarray, dtype, int8, int32, int64, float64

D = TypeVar('D', bound=dtype)


def isBase64(sb):
    try:
        if isinstance(sb, str):
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return b64encode(b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False

def decode(src:str, dtype: D = dtype('int32')) -> ndarray[int, D]:
    return np.array([], dtype=dtype) if src == '' else np.frombuffer(b64decode(src), dtype).copy() #type:ignore


def fdecode(src:str, dtype: D = dtype('int32')) -> ndarray[int, D] | str:
    return np.array([], dtype=dtype) if src == '' else decode(src, dtype) if isBase64(src) else src #type:ignore


def encode(data: ndarray) -> str:
    return b64encode(data.tobytes()).decode()


def fencode(data: ndarray | str) -> str:
    if isinstance(data, str):
        return data
    elif isinstance(data, ndarray):
        if len(data) == 0:
            return ''
        return encode(data)


FC_ELEMENT_TYPES = {}

for ELEMENT_TYPE in ELEMENT_TYPES:
    for i in ELEMENT_TYPE['fc_id']:
        FC_ELEMENT_TYPES[i] = ELEMENT_TYPE


class FCElems:
    blocks: ndarray[int, dtype[int32]]
    orders: ndarray[int, dtype[int32]]
    parent_ids: ndarray[int, dtype[int32]]
    types: ndarray[int, dtype[int8]]
    ids: ndarray[int, dtype[int32]]
    # nodes: list[ndarray[int, dtype[int32]]]
    nodes: ndarray[int, dtype[int32]]


    def __init__(self):
        self.blocks = np.array([], dtype=int32)
        self.orders = np.array([], dtype=int32)
        self.parent_ids = np.array([], dtype=int32)
        self.types = np.array([], dtype=int8)
        self.ids = np.array([], dtype=int32)
        self.nodes = np.array([], dtype=int32)

        self.offsets = np.array([], dtype=int32)
        self.sizes = np.array([], dtype=int32)

    def __len__(self):
        return len(self.ids)

    def __repr__(self) -> str:
        return f"<FCElems: {len(self)} elements>"

    def update(self):

        self.sizes = np.vectorize(lambda t:  FC_ELEMENT_TYPES[t]['nodes'])(self.types)
        self.offsets = np.cumsum(self.sizes) - self.sizes


    # def __getitem__(self, mask):
    #     item = FCElems()
    #     item.ids = self.ids[mask]
    #     item.types = self.types[mask]
    #     item.blocks = self.blocks[mask]
    #     item.orders = self.orders[mask]
    #     item.parent_ids = self.parent_ids[mask]

    #     item.ids

    #     item.nodes = self.nodes[nodes_mask]

    #     item.update()
    #     return item



class FCNodes:
    ids: ndarray[int, dtype[int32]]
    coords: ndarray[int, dtype[float64]]

    def __init__(self):
        self.ids = np.array([], dtype=int32)
        self.coords = np.array([], dtype=float64)

    def __len__(self):
        return len(self.ids)

    def __repr__(self) -> str:
        return f"<FCNodes: {len(self)} nodes>"

    # def __getitem__(self, mask):
    #     item = FCNodes()
    #     item.ids = self.ids[mask]
    #     item.coords = self.coords[mask]
    #     return item



class FCMesh:

    def __init__(self, data = None):
        self.nodes = FCNodes()
        self.elems = FCElems()

        if data:
            self.decode(data)


    def __repr__(self) -> str:
        return "<FCMesh>"

    nodes: FCNodes
    elems: FCElems


    def decode(self, data = {}):


        self.elems.blocks = decode(data['elem_blocks'])
        self.elems.orders = decode(data['elem_orders'])
        self.elems.parent_ids = decode(data['elem_parent_ids'])
        self.elems.types = decode(data['elem_types'], dtype(int8))
        self.elems.ids = decode(data['elemids'])
        self.elems.nodes = decode(data['elems'])

        self.nodes.ids = decode(data['nids'])
        self.nodes.coords = decode(data['nodes'], dtype(float64)).reshape(-1,3)

        self.elems.update()



    def encode(self):

        return {
            "elem_blocks": encode(self.elems.blocks),
            "elem_orders": encode(self.elems.orders),
            "elem_parent_ids": encode(self.elems.parent_ids),
            "elem_types": encode(self.elems.types),
            "elemids": encode(self.elems.ids),
            "elems": encode(self.elems.nodes),
            "elems_count": len(self.elems.ids),
            "nids": encode(self.nodes.ids),
            "nodes": encode(self.nodes.coords),
            "nodes_count": len(self.nodes.ids),
        }


class FCBlock:
    id: int
    cs_id : int
    material_id: int
    property_id: int

    def __init__(self, data = None):
        if data:
            self.decode(data)
 
    def decode(self, data):
        self.id = data['id']
        self.cs_id = data['cs_id']
        self.material_id = data['material_id']
        self.property_id = data['property_id']

    def encode(self):
        return {
            'id': self.id,
            'cs_id': self.cs_id,
            'material_id': self.material_id,
            'property_id': self.property_id
        }



class FCCoordinateSystem(TypedDict):
    dir1: ndarray[int, dtype[float64]]
    dir2: ndarray[int, dtype[float64]]
    id: int
    name: str
    origin: ndarray[int, dtype[float64]]
    type: str


class FCDependency(TypedDict):
    type: int
    data: ndarray[int, dtype[float64]] | str


class FCMaterialProperty(TypedDict):
    type : int
    name: int
    data : ndarray[int, dtype[float64]] | str
    dependency: list[FCDependency] | int


class FCMaterial(TypedDict):
    id: int
    name: str
    properties: dict[str, list[FCMaterialProperty]]


class FCLoadAxis(TypedDict):
    data: ndarray[int, dtype[float64]] | str
    dependency: list[FCDependency] | int


class FCLoad(TypedDict):
    apply_to: ndarray[int, dtype[int64]] | str
    cs: Optional[int]
    name: str
    type: int
    id: int
    axes: list[FCLoadAxis]


class FCRestrainAxis(TypedDict):
    data: ndarray[int, dtype[float64]] | str
    dependency: list[FCDependency] | int
    flag: bool


class FCRestraint(TypedDict):
    apply_to: ndarray[int, dtype[int64]] | str
    cs: Optional[int]
    name: str
    id: int
    axes: list[FCRestrainAxis]


class FCNodeset(TypedDict):
    apply_to: ndarray[int, dtype[int64]] | str
    id: int
    name: str

class FCSideset(TypedDict):
    apply_to: ndarray[int, dtype[int32]] | str
    id: int
    name: str

class FCReciver(TypedDict):
    apply_to: ndarray[int, dtype[int32]] | str
    dofs: list[int]
    id: int
    name: str
    type: int


def decode_dependency(deps_types: list | int, dep_data) -> list[FCDependency] | int :

    if isinstance(deps_types, list):

        return [{
            "type": deps_type,
            "data": fdecode(dep_data[j], dtype(float64))
        } for j, deps_type in enumerate(deps_types)]

    elif isinstance(deps_types, int):
        return deps_types


def encode_dependency(dependency: list[FCDependency] | int):

    if isinstance(dependency, int):
        return dependency, ''
    elif isinstance(dependency, list):
        return [deps['type'] for deps in dependency], [fencode(deps['data']) for deps in dependency]


class FCModel:


    header = {
      "binary" : True,
      "description" : "Fidesys Case Format",
      "types" : { "char":1, "double":8, "int":4, "short_int":2 },
      "version" : 3
    }

    settings = {}

    blocks: list[FCBlock] = []

    coordinate_systems: list[FCCoordinateSystem]= []

    materials: list[FCMaterial] = []

    restraints: list[FCRestraint] = []
    loads: list[FCLoad] = []

    receivers: list[FCReciver] = []


    mesh: FCMesh = FCMesh()
    

    def __init__(self, filepath=None):

        if filepath:
            with open(filepath, "r") as f:
                src_data = json.load(f)

            self.src_data = src_data

            self._decode_header(src_data)
            self._decode_blocks(src_data)
            self._decode_coordinate_systems(src_data)
            self._decode_mesh(src_data)
            self._decode_settings(src_data)
            self._decode_materials(src_data)
            self._decode_restraints(src_data)
            self._decode_loads(src_data)
            self._decode_receivers(src_data)

    def save(self, filepath):

        src_data = {}

        self._encode_header(src_data)
        self._encode_blocks(src_data)
        self._encode_coordinate_systems(src_data)
        self._encode_mesh(src_data)
        self._encode_settings(src_data)
        self._encode_materials(src_data)
        self._encode_restraints(src_data)
        self._encode_loads(src_data)
        self._encode_receivers(src_data)

        with open(filepath, "w") as f:
            json.dump(src_data, f, indent=4)


    def _decode_header(self, src_data):
        self.header = src_data.get('header')


    def _encode_header(self, src_data):
        src_data['header'] = self.header


    def _decode_blocks(self, src_data):
        self.blocks = [FCBlock(src) for src in src_data.get('blocks', [])]


    def _encode_blocks(self, src_data):
        if self.blocks:
            src_data['blocks'] = [block.encode() for block in self.blocks]


    def _decode_coordinate_systems(self, src_data):

        self.coordinate_systems = [{
            'dir1': decode(cs['dir1'],   dtype(float64)),
            'dir2': decode(cs['dir2'],   dtype(float64)),
            'origin': decode(cs['origin'],   dtype(float64)),
            "id" : cs['id'],
            "name": cs['name'],
            "type": cs['type']
        } for cs in src_data.get('coordinate_systems') ]


    def _encode_coordinate_systems(self, src_data):
        if self.coordinate_systems:
            src_data['coordinate_systems'] = [{
                'dir1': encode(cs['dir1']),
                'dir2': encode(cs['dir2']),
                'origin': encode(cs['origin']),
                "id" : cs['id'],
                "name": cs['name'],
                "type": cs['type']
            } for cs in self.coordinate_systems ]


    def _decode_mesh(self, src_data):
        self.mesh = FCMesh(src_data['mesh'])


    def _encode_mesh(self, src_data):
        src_data['mesh'] = self.mesh.encode()


    def _decode_settings(self, data):
        self.settings = data.get('settings')
        assert self.settings


    def _encode_settings(self, src_data):
        settings = self.settings
        src_data['settings'] = settings


    def _decode_materials(self, src_data):

        self.materials = []

        for material_src in src_data.get('materials', []):

            properties: dict[str, list[FCMaterialProperty]] = {}

            for property_name in material_src:
                properties_src = material_src[property_name]

                if type(properties_src) != list:
                    continue

                properties[property_name] = [{
                        "name": property_src["const_names"][i],
                        "data": decode(constants, dtype(float64)),
                        "type": property_src["type"],
                        "dependency": decode_dependency(
                            property_src["const_types"][i], 
                            property_src["const_dep"][i]
                        )
                    }
                    for property_src in properties_src
                    for i, constants in enumerate(property_src["constants"])
                ]

            self.materials.append({
                "id": material_src['id'],
                "name": material_src['name'],
                "properties": properties
            })


    def _encode_materials(self, src_data):
        if self.materials:

            src_data['materials'] = []

            for material in self.materials:

                material_src = {
                    "id": material['id'],
                    "name": material['name'],
                }

                for property_name in material["properties"]:

                    material_src[property_name] = []

                    for material_property in material["properties"][property_name]:

                        const_types, const_dep = encode_dependency(material_property["dependency"])
                        
                        material_src[property_name].append({
                            "const_dep": [const_dep],
                            "const_dep_size": [len(material_property["data"])],
                            "const_names": [material_property["name"]],
                            "const_types": [const_types],
                            "constants": [fencode(material_property["data"])],
                            "type": material_property["type"]
                        })
            
                src_data['materials'].append(material_src)


    def _decode_restraints(self, src_data):

        self.restraints = []

        for restraint_src in src_data.get('restraints', []):

            axes: list[FCRestrainAxis] = []

            for i, dep_var_num in enumerate(restraint_src['dep_var_num']):

                axis_data = restraint_src['data'][i] \
                    if restraint_src["dependency_type"][i] == 6 \
                    else fdecode(restraint_src['data'][i], dtype('float64'))

                axes.append({
                    "data": axis_data,
                    "dependency": decode_dependency(restraint_src["dependency_type"][i], dep_var_num),
                    "flag": restraint_src['flag'][i],
                })

            apply_to = fdecode(restraint_src['apply_to'], dtype('int64'))
            assert len(apply_to) == restraint_src['apply_to_size']

            self.restraints.append({
                "id": restraint_src['id'],
                "name": restraint_src['name'],
                "cs": restraint_src.get('cs', 0),
                "apply_to": apply_to,
                "axes": axes
            })


    def _encode_restraints(self, src_data):
        if self.restraints:

            src_data['restraints'] = []
            
            for restraint in self.restraints:

                restraint_src = {
                    'id': restraint['id'],
                    'name': restraint['name'],
                    'cs': restraint['cs'],
                    'apply_to': fencode(restraint['apply_to']),
                    'apply_to_size': len(restraint['apply_to']),
                    'data': [],
                    'flag': [],
                    'dependency_type': [],
                    'dep_var_num': [],
                    'dep_var_size': [],
                }

                for axis in restraint['axes']:
                    restraint_src['data'].append(fencode(axis['data']))
                    restraint_src['flag'].append(axis['flag'])

                    const_types, const_dep = encode_dependency(restraint_src["dependency"])

                    restraint_src['dependency_type'].append(const_types)
                    restraint_src['dep_var_num'].append(const_dep)
                    restraint_src['dep_var_size'].append(len(const_dep))

                src_data['restraints'].append(restraint_src)
            

    def _decode_loads(self, src_data):

        self.loads = []

        for load_src in src_data.get('loads', []):

            axes: list[FCLoadAxis] = []
            if 'dep_var_num' in load_src:
                for i, dep_var_num in enumerate(load_src['dep_var_num']):
                    axes.append({
                        "data": fdecode(load_src['data'][i], dtype('float64')),
                        "dependency": decode_dependency(load_src["dependency_type"][i], dep_var_num),
                    })

            apply_to = fdecode(load_src['apply_to'], dtype('int64'))

            assert len(apply_to) == load_src['apply_to_size']

            self.loads.append({
                "id": load_src['id'],
                "name": load_src['name'],
                "cs": load_src['cs'] if 'cs' in load_src else 0,
                "apply_to": apply_to,
                "axes": axes,
                "type": load_src['type'],
            })


    def _encode_loads(self, src_data):
        if self.loads:

            src_data['loads'] = []
            
            for load in self.loads:

                load_src = {
                    'id': load['id'],
                    'name': load['name'],
                    'cs': load['cs'],
                    'type': load['type'],
                    'apply_to': fencode(load['apply_to']),
                    'apply_to_size': len(load['apply_to']),
                    'data': [],
                    'dependency_type': [],
                    'dep_var_num': [],
                    'dep_var_size': [],
                }

                for axis in load['axes']:
                    load_src['data'].append(fencode(axis['data']))

                    const_types, const_dep = encode_dependency(load_src["dependency"])

                    load_src['dependency_type'].append(const_types)
                    load_src['dep_var_num'].append(const_dep)
                    load_src['dep_var_size'].append(len(const_dep))

                src_data['loads'].append(load_src)
            


    # def _decode_nodesets(self, data):
    #     pass


    # def _encode_nodesets(self, data):
    #     pass

    # def _encode_sidesets(self, data):
    #     pass


    def _decode_receivers(self, src_data):

        self.receivers = [{
            'apply_to': fdecode(cs['apply_to']),
            'dofs': cs['dofs'],
            "id" : cs['id'],
            "name": cs['name'],
            "type": cs['type']
        } for cs in src_data.get('receivers',[]) ]


    def _encode_receivers(self, src_data):

        if self.receivers:
            src_data['receivers'] = [{
                'apply_to': fencode(cs['apply_to']),
                'apply_to_size': len(cs['apply_to']),
                'dofs': cs['dofs'],
                "id" : cs['id'],
                "name": cs['name'],
                "type": cs['type']
            } for cs in self.receivers ]



    def cut(self, cut_function: Callable):

        nodes_mask = [cut_function(
            self.mesh.nodes.ids[i],
            self.mesh.nodes.coords[i]
        ) for i in range(len(self.mesh.nodes))]

        self.mesh.nodes.coords = self.mesh.nodes.coords[nodes_mask]
        self.mesh.nodes.ids = self.mesh.nodes.ids[nodes_mask]

        node_set = set(self.mesh.nodes.ids)

        elems_mask = [node_set.issuperset(elem) for elem in self.mesh.elems.nodes]

        self.mesh.elems.blocks = self.mesh.elems.blocks[elems_mask]
        self.mesh.elems.orders = self.mesh.elems.orders[elems_mask]
        self.mesh.elems.parent_ids = self.mesh.elems.parent_ids[elems_mask]
        self.mesh.elems.types = self.mesh.elems.types[elems_mask]
        self.mesh.elems.ids = self.mesh.elems.ids[elems_mask]

        # self.mesh.elems.nodes = [elem for i, elem in enumerate(self.mesh.elems.nodes) if elems_mask[i]]
        self.mesh.elems.nodes = self.mesh.elems.nodes[elems_mask]


        for material in self.materials:
            for key in material['properties']:
                for property in material['properties'][key]:

                    if isinstance(property['dependency'], list) and property['dependency']:
                        for dep in property['dependency']:

                            dep_data = dep['data']
                            property_data = property['data']

                            if isinstance(dep_data, ndarray) and isinstance(property_data, ndarray):

                                if dep['type'] == 10:
                                    mat_mask = np.isin(dep_data, self.mesh.elems.ids, assume_unique=True)
                                    dep['data'] = dep_data[mat_mask]
                                    property['data'] = property_data[mat_mask]

                                if dep['type'] == 11:
                                    mat_mask = np.isin(dep_data, self.mesh.nodes.ids, assume_unique=True)
                                    dep['data'] = dep_data[mat_mask]
                                    property['data'] = property_data[mat_mask]


    def compress(self):
        nodes_id_map = {(index):i+1 for i, index in enumerate(self.mesh.nodes.ids)}
        elems_id_map = {(index):i+1 for i, index in enumerate(self.mesh.elems.ids)}

        self.mesh.nodes.ids = np.arange(len(self.mesh.nodes), dtype=np.int32)+1
        self.mesh.elems.ids = np.arange(len(self.mesh.elems), dtype=np.int32)+1

        for i, n in enumerate(self.mesh.elems.nodes):
            node = nodes_id_map[n]
            self.mesh.elems.nodes[i] = node


        for material in self.materials:
            for key in material['properties']:
                for property in material['properties'][key]:

                    if isinstance(property['dependency'], list) and property['dependency']:
                        for dep in property['dependency']:
                            if isinstance(dep['data'], ndarray):
                                if dep['type'] == 10:
                                    for i, n in enumerate(dep['data']):
                                        dep['data'][i] = elems_id_map[int(n)]
                                if dep['type'] == 11:
                                    for i, n in enumerate(dep['data']):
                                        dep['data'][i] = nodes_id_map[int(n)]
                        
