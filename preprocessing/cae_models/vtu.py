# -*- coding: utf-8 -*-
import json
import os
import shutil
from copy import copy
import base64
import zlib

import pandas as pd
import numpy as np
import xml.etree.cElementTree as ET


class Vtu:

    def cell_type(self, code):
        assert code in self.vtk_cell_types, f'unknown vkt element {code}'

        cell_type = self.vtk_cell_types[code]

        name = cell_type[0]
        nodes_count = cell_type[1]
        edges_struct = cell_type[2]
        surface_struct = cell_type[3]
        edges = []

        def split_line(line):
            if len(line) < 2:
                return []
            return list(map(list, zip(line, line[1:])))

        for polyedge in edges_struct:
            edges.extend(split_line(polyedge))

        polygon_struct = {}

        for shape in surface_struct:
            if len(shape) not in polygon_struct:
                polygon_struct[len(shape)] = []
            polygon_struct[len(shape)].extend(list(shape))

        edges_template = np.array(edges).reshape(-1)

        return {
            'name': name,
            'nodes_count': nodes_count,
            'edges_struct': edges_struct,
            'surface_struct': surface_struct,
            'edges_template': edges_template,
            'polygon_struct': polygon_struct,
        }

    vtk_cell_types = {
        # https://www.cs.auckland.ac.nz/compsci716s2t/resources/VTK_file-formats.pdf
        # https://kitware.github.io/vtk-examples/site/VTKFileFormats/
        # vtk_code: (name, count_nodes, edges_struct, trangles_struct)
        1: ("vertex", 1, [], []),
        3: ("edge", 2, [(0, 1)], []),
        5: ("triangle", 3,# -*- coding: utf-8 -*-
import json
import os
import shutil
from copy import copy
import base64
import zlib

import pandas as pd
import numpy as np
import xml.etree.cElementTree as ET


class VTUModel:

    def cell_type(self, code):
        assert code in self.vtk_cell_types, f'unknown vkt element {code}'

        cell_type = self.vtk_cell_types[code]

        name = cell_type[0]
        nodes_count = cell_type[1]
        edges_struct = cell_type[2]
        surface_struct = cell_type[3]
        edges = []

        def split_line(line):
            if len(line) < 2:
                return []
            return list(map(list, zip(line, line[1:])))

        for polyedge in edges_struct:
            edges.extend(split_line(polyedge))

        polygon_struct = {}

        for shape in surface_struct:
            if len(shape) not in polygon_struct:
                polygon_struct[len(shape)] = []
            polygon_struct[len(shape)].extend(list(shape))

        edges_template = np.array(edges).reshape(-1)

        return {
            'name': name,
            'nodes_count': nodes_count,
            'edges_struct': edges_struct,
            'surface_struct': surface_struct,
            'edges_template': edges_template,
            'polygon_struct': polygon_struct,
        }

    vtk_cell_types = {
        # https://www.cs.auckland.ac.nz/compsci716s2t/resources/VTK_file-formats.pdf
        # https://kitware.github.io/vtk-examples/site/VTKFileFormats/
        # vtk_code: (name, count_nodes, edges_struct, trangles_struct)
        1: ("vertex", 1, [], []),
        3: ("edge", 2, [(0, 1)], []),
        5: ("triangle", 3,
            [(0, 1, 2, 0)],
            [(0, 1, 2)]),
        9: ("quad", 4,
            [(0, 1, 2, 3, 0)],
            [(0, 1, 2, 3)]),
        10: ("tetra", 4,
             [(0, 1, 2, 0), (0, 3), (1, 3), (2, 3)],
             [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]),
        12: ("hexahedron", 8,
             [(0, 1, 2, 3, 0), (4, 5, 6, 7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
             [(3, 2, 1, 0), (4, 5, 6, 7), (1, 2, 6, 5), (0, 1, 5, 4), (0, 4, 7, 3), (2, 3, 7, 6)]),
        13: ("wedge", 6,
             [(0, 1, 2, 0), (3, 4, 5, 3), (0, 3), (1, 4), (2, 5)],
             [(0, 1, 2), (5, 4, 3), (0, 2, 5, 3), (0, 3, 4, 1), (1, 4, 5, 2)]),
        14: ("pyramid", 5,
             [(0, 1, 2, 3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
             [(3, 2, 1, 0), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]),
        21: ("edge3", 3, [(0, 2, 1)], []),
        22: ("triangle6", 6,
             [(0, 3, 1, 4, 2, 5, 0)],
             [(0, 3, 1, 4, 2, 5)]),
        23: ("quad8", 8,
             [(0, 4, 1, 5, 2, 6, 3, 7, 0)],
             [(0, 4, 1, 5, 2, 6, 3, 7)]),
        24: ("tetra10", 10,
             [(0, 4, 1, 5, 2, 6, 0), (0, 7, 3), (1, 8, 3), (2, 9, 3)],
             [(0, 6, 2, 5, 1, 4), (0, 4, 1, 8, 3, 5), (1, 5, 2, 9, 3, 8), (2, 6, 0, 5, 3, 9)]),
        25: ("hexahedron20", 20,
             [(0, 8, 1, 9, 2, 10, 3, 11, 0), (4, 12, 5, 13, 6, 14, 7, 15, 4),
              (0, 16, 4), (1, 17, 5), (2, 18, 6), (3, 19, 7)],
             [(3, 10, 2, 9, 1, 8, 0, 11), (4, 12, 5, 13, 6, 14, 7, 15), (1, 9, 2, 18, 6, 13, 5, 17),
              (0, 8, 1, 17, 5, 12, 4, 16), (0, 16, 4, 15, 7, 19, 3, 11), (2, 10, 3, 19, 7, 14, 6, 18)]),
        26: ("wedge15", 15,
             [(0, 6, 1, 7, 2, 8, 0), (3, 9, 4, 10, 5, 11, 3), (0, 12, 3), (1, 13, 4), (2, 14, 5)],
             [(0, 6, 1, 7, 2, 8), (5, 10, 4, 9, 3, 11),
              (0, 8, 2, 14, 5, 11, 3, 12), (0, 12, 3, 9, 4, 13, 1, 6), (1, 13, 4, 10, 5, 14, 2, 7)]),
        27: ("pyramid13", 13,
             [(0, 5, 1, 6, 2, 7, 3, 8, 0), (0, 9, 4), (1, 10, 4), (2, 11, 4), (3, 12, 4)],
             [(3, 7, 2, 6, 1, 5, 0, 8),
              (0, 5, 1, 10, 4, 9), (1, 6, 2, 11, 4, 10), (2, 7, 3, 12, 4, 11), (3, 8, 0, 9, 4, 12)]),
    }

    def __init__(self, filename: str):

        with open(filename, 'r') as file_read:
            self.tree = self.scan_file(file_read)

        PointData = self.tree['Piece']['PointData']
        CellData = self.tree['Piece']['CellData']
        Points = self.tree['Piece']['Points']
        Cells = self.tree['Piece']['Cells']

        point_data = copy(PointData)

        for key in point_data:
            if 'Node' in key:
                del PointData[key]

        self.cells = pd.DataFrame({
            'type': Cells['types']['value'],
            'nodes': pd.Series(np.split(Cells['connectivity']['value'].values, Cells['offsets']['value'].values[:-1]))
        })

        self.cell_data = pd.DataFrame({name: CellData[name]['value'] for name in CellData})

        self.points = Points['Points']
        self.point_data = PointData
        for attr in self.point_data:
            attr_data = self.point_data[attr]
            if 'X' in attr_data and 'Y' in attr_data and 'Z' in attr_data and 'Magnitude' not in attr_data:
                self.point_data[attr]['Magnitude'] = (attr_data['X']**2 + attr_data['Y']**2 + attr_data['Z']**2)**(1/2)

    def add_safety_factor(self, materials):

        material_nodes = {'index': np.array([], dtype=np.int32), 'material': np.array([], dtype=np.int32)}

        for vol in self.cell_data['Material ID'].unique():

            data = np.array([], dtype=np.int32)
            cells = self.cells['nodes'][self.cell_data['Material ID'] == vol]
            for cell in cells:
                data = np.concatenate((data, cell))
            data = np.unique(data)
            material_nodes['index'] = np.concatenate((material_nodes['index'], data))
            material_nodes['material'] = np.concatenate((material_nodes['material'], np.full(len(data), vol)))

        material_nodes = pd.DataFrame(material_nodes)
        material_nodes = material_nodes.set_index('index')

        # Добавляем первичный критерий прочности
        def calc_safety_factor_ultimate(s):

            if s['stress'] == 0:
                return 1e20
            if not materials[int(s['material']) - 1]['value']['ultimate_stress']:
                return 0.0
            value = materials[int(s['material']) - 1]['value']['ultimate_stress'] / s['stress']
            if value > 1e20:
                return 1e20
            return value

        def calc_safety_factor_yield(s):
            if s['stress'] == 0:
                return 1e20
            if not materials[int(s['material']) - 1]['value']['yield_stress']:
                return 0.0
            value = materials[int(s['material']) - 1]['value']['yield_stress'] / s['stress']
            if value > 1e20:
                return 1e20
            return value

        if 'Principal stress vector 1' in self.point_data:
            material_nodes['stress'] = self.point_data['Principal stress vector 1']['Magnitude']

            self.point_data['Safety Factor 1-Principal'] = pd.DataFrame({
                'Ultimate': material_nodes.apply(calc_safety_factor_ultimate, axis=1),
                'Yield': material_nodes.apply(calc_safety_factor_yield, axis=1)
            })

        if 'Stress' in self.point_data:
            material_nodes['stress'] = self.point_data['Stress']['Mises']

            self.point_data['Safety Factor Mises'] = pd.DataFrame({
                'Ultimate': material_nodes.apply(calc_safety_factor_ultimate, axis=1),
                'Yield': material_nodes.apply(calc_safety_factor_yield, axis=1)
            })


    def surface(self):

        surface_geometry = {}

        nodes_offset = 0

        for vol in self.cell_data['Parent ID'].unique():

            surface_nodes = []

            volume_nodes = self.cells['nodes'][self.cell_data['Parent ID'] == vol]

            cell_types = self.cells['type'].unique()

            polygon_struct = {}

            for code in cell_types:
                cell_type = self.cell_type(code)

                volume_node = volume_nodes[self.cells['type'] == code].values
                if not len(volume_node):
                    continue

                cell_nodes = np.vstack(volume_node)

                for n in cell_type['polygon_struct']:
                    if n not in polygon_struct:
                        polygon_struct[n] = []

                    polygon_struct[n].extend(cell_nodes[:, cell_type['polygon_struct'][n]])

            for n in polygon_struct:
                polygon_struct[n] = np.array(polygon_struct[n]).reshape(-1, n)

                polygons = np.copy(polygon_struct[n])
                if not len(polygons):
                    continue

                polygons.sort()

                polygons, uniq_cnt = np.unique(polygons, axis=0, return_counts=True)
                polygons = polygons[uniq_cnt == 1]

                surface_nodes.extend(list(polygons.reshape(-1)))

            surface_nodes = np.unique(surface_nodes)

            def compact(v):
                r = {node: i for i, node in enumerate(surface_nodes)}
                return np.vectorize(lambda x: r.get(x, -1))(v)

            def in_surface(v):
                v = compact(v)
                return v[np.where(np.all(v != -1, axis=1))]

            edges = []
            trangles = []

            for code in cell_types:
                cell_type = self.cell_type(code)

                edges_template = cell_type['edges_template']

                cells_node = volume_nodes[self.cells['type'] == code].values

                edges.extend([cell[edges_template] for cell in cells_node])

            for n in polygon_struct:
                polygons = polygon_struct[n]

                polygons = in_surface(polygons)

                def split_shape(shape):
                    return [shape[-1], *shape[:2], *split_shape((*shape[2:], shape[1]))] if len(shape) >= 3 else []

                trangles_template = split_shape(list(range(n)))

                trangles.extend(polygons[:, trangles_template])

            edges = np.concatenate(edges).reshape(-1, 2)
            edges = in_surface(edges)
            trangles = np.concatenate(trangles).reshape(-1, 3)

            edges += nodes_offset
            trangles += nodes_offset
            nodes_offset += len(surface_nodes)

            surface_geometry[vol] = {
                'point_data': {attr: self.point_data[attr].loc[surface_nodes] for attr in self.point_data},
                'points': self.points.loc[surface_nodes],
                'edges': edges,
                'trangles': trangles,
            }

        return surface_geometry

    @property
    def elements_count(self):
        elements = {}
        for element_type_code in self.cells["type"].tolist():
            element_type_name = self.cell_type(element_type_code)["name"]
            if element_type_name not in elements:
                elements[element_type_name] = 1
            else:
                elements[element_type_name] += 1
        return elements

    def surface_save_to_np(self, path):
        surfaces = self.surface()
        trangles = []
        edges = []
        points = []
        point_data = {}
        for vol in surfaces:
            surface = surfaces[vol]
            trangles.append(surface['trangles'])
            edges.append(surface['edges'])
            points.append(surface['points'].values)
            for attr in surface['point_data']:
                if attr not in point_data:
                    point_data[attr] = []
                point_data[attr].append(surface['point_data'][attr].values)

        points = np.concatenate(points)
        edges = np.concatenate(edges)
        trangles = np.concatenate(trangles)

        point_data = {attr: np.concatenate(point_data[attr]) for attr in point_data}

        if os.path.exists(os.path.join(path, 'attrs')):
            shutil.rmtree(os.path.join(path, 'attrs'))
        os.makedirs(os.path.join(path, 'attrs'))

        model_file = os.path.join(path, 'model.npz')
        np.savez(model_file, points=points, trangles=trangles, edges=edges)

        for attr in point_data:
            attr_file = os.path.join(path, 'attrs', f'{attr}.npy')
            np.save(attr_file, point_data[attr])

        extremes = self.extremes
        attrs = self.attrs

        with open(os.path.join(path, 'attrs.json'), "w") as write_file:
            json.dump({
                attr: {
                    'axes': attrs[attr],
                    'extremes': extremes[attr]
                }
                for attr in attrs
            }, write_file, indent=4)

        with open(os.path.join(path, 'extremes.json'), "w") as write_file:
            json.dump({attr: extremes[attr] for attr in attrs}, write_file, indent=4)

    def scan_file(self, file_read):

        head = ""
        for line in file_read:
            if '<AppendedData' in line:
                head += '</VTKFile>'
                break
            head += line
        head = ET.fromstring(head)

        assert head.attrib["type"] == "UnstructuredGrid"

        UnstructuredGrid = head[0]

        offset_map = self._scan_offset_map(UnstructuredGrid)

        while file_read.read(1) == ' ':
            pass

        return self._scan_tree(UnstructuredGrid, offset_map, file_read)

    def _scan_offset_map(self, xml):
        offsets = []
        for leaf in xml:
            if 'offset' in leaf.attrib:
                offsets.append(int(leaf.attrib['offset']))
            else:
                offsets.extend(self._scan_offset_map(leaf))
        return offsets

    def _scan_tree(self, xml, offset_map, source_file):
        data = {}
        for leaf in xml:
            if 'offset' in leaf.attrib:
                assert leaf.tag in ['Array', 'DataArray']
                assert 'Name' in leaf.attrib

                offset = int(leaf.attrib['offset'])
                index = offset_map.index(offset)
                if index < len(offset_map) - 1:
                    block = source_file.read(
                        offset_map[offset_map.index(offset) + 1] - offset)
                else:
                    block = source_file.readline()[:-1]

                if leaf.tag == 'DataArray':
                    data[leaf.attrib['Name']] = self._convert_data_array(block, leaf.attrib)
            else:
                data[leaf.tag] = self._scan_tree(leaf, offset_map, source_file)
        return data

    @staticmethod
    def _decode_data_block(data, data_type, header_type='UInt32'):
        # using dark magic
        def vtu_to_np_type(name):
            return np.dtype(getattr(np, name.lower()))

        def num_bytes_to_num_base64_chars(num_bytes):
            return -(-num_bytes // 3) * 4

        dtype = vtu_to_np_type(header_type)
        num_bytes_per_item = np.dtype(dtype).itemsize
        num_chars = num_bytes_to_num_base64_chars(num_bytes_per_item)
        byte_string = base64.b64decode(data[:num_chars])[:num_bytes_per_item]
        num_blocks = np.frombuffer(byte_string, dtype)[0]

        num_header_items = 3 + num_blocks
        num_header_bytes = num_bytes_per_item * num_header_items
        num_header_chars = num_bytes_to_num_base64_chars(num_header_bytes)
        byte_string = base64.b64decode(data[:num_header_chars])
        header = np.frombuffer(byte_string, dtype)

        block_sizes = header[3:]

        byte_array = base64.b64decode(data[num_header_chars:])
        dtype = vtu_to_np_type(data_type)

        byte_offsets = np.empty(block_sizes.shape[0] + 1, dtype=block_sizes.dtype)
        byte_offsets[0] = 0
        np.cumsum(block_sizes, out=byte_offsets[1:])

        # process the compressed data
        block_data = np.concatenate([
            np.frombuffer(
                zlib.decompress(byte_array[byte_offsets[k]: byte_offsets[k + 1]]),
                dtype=dtype,
            )
            for k in range(num_blocks)
        ])

        return block_data

    def _convert_data_array(self, block, attrib):
        data_block = self._decode_data_block(block, attrib['type'], "UInt32")

        if 'NumberOfComponents' in attrib:
            dim = int(attrib['NumberOfComponents'])
            columns = list(range(dim))
            for key in attrib:
                if key.startswith('ComponentName'):
                    columns[int(key[len('ComponentName'):])] = attrib[key]

            data = np.reshape(data_block, (-1, dim))

            if dim == 3 and columns[0] == 0:

                if 'Name' in attrib and attrib['Name'] == 'Points':
                    columns = ['X', 'Y', 'Z']
                else:
                    columns = ['Magnitude', 'X', 'Y', 'Z']
                    m = np.apply_along_axis(
                        lambda x: [(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5],
                        1, data)
                    data = np.hstack([m, data])

            return pd.DataFrame(
                data=data,
                columns=columns)

        return pd.DataFrame(data={'value': pd.Series(data_block)})


    @property
    def attributes(self):
        attributes = {}
        for field in self.point_data:
            attributes[field] = []

            axes = list(self.point_data[field].columns)

            index = 0

            for axis in axes:
                column = self.point_data[field][axis]

                extremes = {
                    'min': {
                        'node': int(column.idxmin()),
                        'point': self.points.loc[column.idxmin()].to_list(),
                        'value': column.min()
                    },
                    'max': {
                        'node': int(column.idxmax()),
                        'point': self.points.loc[column.idxmax()].to_list(),
                        'value': column.max()
                    },
                }

                attributes[field].append({
                    'id': axis,
                    'extremes': extremes,
                    'index': index
                })

                index += 1


        return attributes


if __name__ == '__main__':
    materials = [{'uuid': '8aef3dce-2402-4b9a-8551-a801ff76c2cf', 'markers': '*', 'markers_type': 'volume',
                  'name': 'Material 01', 'entity_type': 'Material',
                  'value': {'youngs_modulus': 200000000000, 'poissons_ratio': 0.3, 'density': 8000,
                            'ultimate_stress': 450000000, 'ultimate_strain': 0.4, 'yield_stress': 160000000,
                            'specific_heat_capacity': 500, 'thermal_conductivity': 45,
                            'thermal_expansion_coefficient': 1.2e-05}, 'enabled': True}]

    vtu_input = '/home/artem/ProveDesign/alpha/data/storage/bbcb00f3fdbc/result/fb62cca7e82f/calc.vtu'

    result_vtu = Vtu(vtu_input)

    result_path = '/home/artem/ProveDesign/alpha/data/storage/bbcb00f3fdbc/result/fb62cca7e82f'

    result_vtu.surface_save_to_np(result_path)

            [(0, 1, 2, 0)],
            [(0, 1, 2)]),
        9: ("quad", 4,
            [(0, 1, 2, 3, 0)],
            [(0, 1, 2, 3)]),
        10: ("tetra", 4,
             [(0, 1, 2, 0), (0, 3), (1, 3), (2, 3)],
             [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]),
        12: ("hexahedron", 8,
             [(0, 1, 2, 3, 0), (4, 5, 6, 7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
             [(3, 2, 1, 0), (4, 5, 6, 7), (1, 2, 6, 5), (0, 1, 5, 4), (0, 4, 7, 3), (2, 3, 7, 6)]),
        13: ("wedge", 6,
             [(0, 1, 2, 0), (3, 4, 5, 3), (0, 3), (1, 4), (2, 5)],
             [(0, 1, 2), (5, 4, 3), (0, 2, 5, 3), (0, 3, 4, 1), (1, 4, 5, 2)]),
        14: ("pyramid", 5,
             [(0, 1, 2, 3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
             [(3, 2, 1, 0), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]),
        21: ("edge3", 3, [(0, 2, 1)], []),
        22: ("triangle6", 6,
             [(0, 3, 1, 4, 2, 5, 0)],
             [(0, 3, 1, 4, 2, 5)]),
        23: ("quad8", 8,
             [(0, 4, 1, 5, 2, 6, 3, 7, 0)],
             [(0, 4, 1, 5, 2, 6, 3, 7)]),
        24: ("tetra10", 10,
             [(0, 4, 1, 5, 2, 6, 0), (0, 7, 3), (1, 8, 3), (2, 9, 3)],
             [(0, 6, 2, 5, 1, 4), (0, 4, 1, 8, 3, 5), (1, 5, 2, 9, 3, 8), (2, 6, 0, 5, 3, 9)]),
        25: ("hexahedron20", 20,
             [(0, 8, 1, 9, 2, 10, 3, 11, 0), (4, 12, 5, 13, 6, 14, 7, 15, 4),
              (0, 16, 4), (1, 17, 5), (2, 18, 6), (3, 19, 7)],
             [(3, 10, 2, 9, 1, 8, 0, 11), (4, 12, 5, 13, 6, 14, 7, 15), (1, 9, 2, 18, 6, 13, 5, 17),
              (0, 8, 1, 17, 5, 12, 4, 16), (0, 16, 4, 15, 7, 19, 3, 11), (2, 10, 3, 19, 7, 14, 6, 18)]),
        26: ("wedge15", 15,
             [(0, 6, 1, 7, 2, 8, 0), (3, 9, 4, 10, 5, 11, 3), (0, 12, 3), (1, 13, 4), (2, 14, 5)],
             [(0, 6, 1, 7, 2, 8), (5, 10, 4, 9, 3, 11),
              (0, 8, 2, 14, 5, 11, 3, 12), (0, 12, 3, 9, 4, 13, 1, 6), (1, 13, 4, 10, 5, 14, 2, 7)]),
        27: ("pyramid13", 13,
             [(0, 5, 1, 6, 2, 7, 3, 8, 0), (0, 9, 4), (1, 10, 4), (2, 11, 4), (3, 12, 4)],
             [(3, 7, 2, 6, 1, 5, 0, 8),
              (0, 5, 1, 10, 4, 9), (1, 6, 2, 11, 4, 10), (2, 7, 3, 12, 4, 11), (3, 8, 0, 9, 4, 12)]),
    }

    def __init__(self, filename: str):

        with open(filename, 'r') as file_read:
            self.tree = self.scan_file(file_read)

        PointData = self.tree['Piece']['PointData']
        CellData = self.tree['Piece']['CellData']
        Points = self.tree['Piece']['Points']
        Cells = self.tree['Piece']['Cells']

        point_data = copy(PointData)

        for key in point_data:
            if 'Node' in key:
                del PointData[key]

        self.cells = pd.DataFrame({
            'type': Cells['types']['value'],
            'nodes': pd.Series(np.split(Cells['connectivity']['value'].values, Cells['offsets']['value'].values[:-1]))
        })

        self.cell_data = pd.DataFrame({name: CellData[name]['value'] for name in CellData})

        self.points = Points['Points']
        self.point_data = PointData
        for attr in self.point_data:
            attr_data = self.point_data[attr]
            if 'X' in attr_data and 'Y' in attr_data and 'Z' in attr_data and 'Magnitude' not in attr_data:
                self.point_data[attr]['Magnitude'] = (attr_data['X']**2 + attr_data['Y']**2 + attr_data['Z']**2)**(1/2)

    def add_safety_factor(self, materials):

        material_nodes = {'index': np.array([], dtype=np.int32), 'material': np.array([], dtype=np.int32)}

        for vol in self.cell_data['Material ID'].unique():

            data = np.array([], dtype=np.int32)
            cells = self.cells['nodes'][self.cell_data['Material ID'] == vol]
            for cell in cells:
                data = np.concatenate((data, cell))
            data = np.unique(data)
            material_nodes['index'] = np.concatenate((material_nodes['index'], data))
            material_nodes['material'] = np.concatenate((material_nodes['material'], np.full(len(data), vol)))

        material_nodes = pd.DataFrame(material_nodes)
        material_nodes = material_nodes.set_index('index')

        # Добавляем первичный критерий прочности
        def calc_safety_factor_ultimate(s):

            if s['stress'] == 0:
                return 1e20
            if not materials[int(s['material']) - 1]['value']['ultimate_stress']:
                return 0.0
            value = materials[int(s['material']) - 1]['value']['ultimate_stress'] / s['stress']
            if value > 1e20:
                return 1e20
            return value

        def calc_safety_factor_yield(s):
            if s['stress'] == 0:
                return 1e20
            if not materials[int(s['material']) - 1]['value']['yield_stress']:
                return 0.0
            value = materials[int(s['material']) - 1]['value']['yield_stress'] / s['stress']
            if value > 1e20:
                return 1e20
            return value

        if 'Principal stress vector 1' in self.point_data:
            material_nodes['stress'] = self.point_data['Principal stress vector 1']['Magnitude']

            self.point_data['Safety Factor 1-Principal'] = pd.DataFrame({
                'Ultimate': material_nodes.apply(calc_safety_factor_ultimate, axis=1),
                'Yield': material_nodes.apply(calc_safety_factor_yield, axis=1)
            })

        if 'Stress' in self.point_data:
            material_nodes['stress'] = self.point_data['Stress']['Mises']

            self.point_data['Safety Factor Mises'] = pd.DataFrame({
                'Ultimate': material_nodes.apply(calc_safety_factor_ultimate, axis=1),
                'Yield': material_nodes.apply(calc_safety_factor_yield, axis=1)
            })


    def surface(self):

        surface_geometry = {}

        nodes_offset = 0

        for vol in self.cell_data['Parent ID'].unique():

            surface_nodes = []

            volume_nodes = self.cells['nodes'][self.cell_data['Parent ID'] == vol]

            cell_types = self.cells['type'].unique()

            polygon_struct = {}

            for code in cell_types:
                cell_type = self.cell_type(code)

                volume_node = volume_nodes[self.cells['type'] == code].values
                if not len(volume_node):
                    continue

                cell_nodes = np.vstack(volume_node)

                for n in cell_type['polygon_struct']:
                    if n not in polygon_struct:
                        polygon_struct[n] = []

                    polygon_struct[n].extend(cell_nodes[:, cell_type['polygon_struct'][n]])

            for n in polygon_struct:
                polygon_struct[n] = np.array(polygon_struct[n]).reshape(-1, n)

                polygons = np.copy(polygon_struct[n])
                if not len(polygons):
                    continue

                polygons.sort()

                polygons, uniq_cnt = np.unique(polygons, axis=0, return_counts=True)
                polygons = polygons[uniq_cnt == 1]

                surface_nodes.extend(list(polygons.reshape(-1)))

            surface_nodes = np.unique(surface_nodes)

            def compact(v):
                r = {node: i for i, node in enumerate(surface_nodes)}
                return np.vectorize(lambda x: r.get(x, -1))(v)

            def in_surface(v):
                v = compact(v)
                return v[np.where(np.all(v != -1, axis=1))]

            edges = []
            trangles = []

            for code in cell_types:
                cell_type = self.cell_type(code)

                edges_template = cell_type['edges_template']

                cells_node = volume_nodes[self.cells['type'] == code].values

                edges.extend([cell[edges_template] for cell in cells_node])

            for n in polygon_struct:
                polygons = polygon_struct[n]

                polygons = in_surface(polygons)

                def split_shape(shape):
                    return [shape[-1], *shape[:2], *split_shape((*shape[2:], shape[1]))] if len(shape) >= 3 else []

                trangles_template = split_shape(list(range(n)))

                trangles.extend(polygons[:, trangles_template])

            edges = np.concatenate(edges).reshape(-1, 2)
            edges = in_surface(edges)
            trangles = np.concatenate(trangles).reshape(-1, 3)

            edges += nodes_offset
            trangles += nodes_offset
            nodes_offset += len(surface_nodes)

            surface_geometry[vol] = {
                'point_data': {attr: self.point_data[attr].loc[surface_nodes] for attr in self.point_data},
                'points': self.points.loc[surface_nodes],
                'edges': edges,
                'trangles': trangles,
            }

        return surface_geometry

    @property
    def elements_count(self):
        elements = {}
        for element_type_code in self.cells["type"].tolist():
            element_type_name = self.cell_type(element_type_code)["name"]
            if element_type_name not in elements:
                elements[element_type_name] = 1
            else:
                elements[element_type_name] += 1
        return elements

    def surface_save_to_np(self, path):
        surfaces = self.surface()
        trangles = []
        edges = []
        points = []
        point_data = {}
        for vol in surfaces:
            surface = surfaces[vol]
            trangles.append(surface['trangles'])
            edges.append(surface['edges'])
            points.append(surface['points'].values)
            for attr in surface['point_data']:
                if attr not in point_data:
                    point_data[attr] = []
                point_data[attr].append(surface['point_data'][attr].values)

        points = np.concatenate(points)
        edges = np.concatenate(edges)
        trangles = np.concatenate(trangles)

        point_data = {attr: np.concatenate(point_data[attr]) for attr in point_data}

        if os.path.exists(os.path.join(path, 'attrs')):
            shutil.rmtree(os.path.join(path, 'attrs'))
        os.makedirs(os.path.join(path, 'attrs'))

        model_file = os.path.join(path, 'model.npz')
        np.savez(model_file, points=points, trangles=trangles, edges=edges)

        for attr in point_data:
            attr_file = os.path.join(path, 'attrs', f'{attr}.npy')
            np.save(attr_file, point_data[attr])

        extremes = self.extremes
        attrs = self.attrs

        with open(os.path.join(path, 'attrs.json'), "w") as write_file:
            json.dump({
                attr: {
                    'axes': attrs[attr],
                    'extremes': extremes[attr]
                }
                for attr in attrs
            }, write_file, indent=4)

        with open(os.path.join(path, 'extremes.json'), "w") as write_file:
            json.dump({attr: extremes[attr] for attr in attrs}, write_file, indent=4)

    def scan_file(self, file_read):

        head = ""
        for line in file_read:
            if '<AppendedData' in line:
                head += '</VTKFile>'
                break
            head += line
        head = ET.fromstring(head)

        assert head.attrib["type"] == "UnstructuredGrid"

        UnstructuredGrid = head[0]

        offset_map = self._scan_offset_map(UnstructuredGrid)

        while file_read.read(1) == ' ':
            pass

        return self._scan_tree(UnstructuredGrid, offset_map, file_read)

    def _scan_offset_map(self, xml):
        offsets = []
        for leaf in xml:
            if 'offset' in leaf.attrib:
                offsets.append(int(leaf.attrib['offset']))
            else:
                offsets.extend(self._scan_offset_map(leaf))
        return offsets

    def _scan_tree(self, xml, offset_map, source_file):
        data = {}
        for leaf in xml:
            if 'offset' in leaf.attrib:
                assert leaf.tag in ['Array', 'DataArray']
                assert 'Name' in leaf.attrib

                offset = int(leaf.attrib['offset'])
                index = offset_map.index(offset)
                if index < len(offset_map) - 1:
                    block = source_file.read(
                        offset_map[offset_map.index(offset) + 1] - offset)
                else:
                    block = source_file.readline()[:-1]

                if leaf.tag == 'DataArray':
                    data[leaf.attrib['Name']] = self._convert_data_array(block, leaf.attrib)
            else:
                data[leaf.tag] = self._scan_tree(leaf, offset_map, source_file)
        return data

    @staticmethod
    def _decode_data_block(data, data_type, header_type='UInt32'):
        # using dark magic
        def vtu_to_np_type(name):
            return np.dtype(getattr(np, name.lower()))

        def num_bytes_to_num_base64_chars(num_bytes):
            return -(-num_bytes // 3) * 4

        dtype = vtu_to_np_type(header_type)
        num_bytes_per_item = np.dtype(dtype).itemsize
        num_chars = num_bytes_to_num_base64_chars(num_bytes_per_item)
        byte_string = base64.b64decode(data[:num_chars])[:num_bytes_per_item]
        num_blocks = np.frombuffer(byte_string, dtype)[0]

        num_header_items = 3 + num_blocks
        num_header_bytes = num_bytes_per_item * num_header_items
        num_header_chars = num_bytes_to_num_base64_chars(num_header_bytes)
        byte_string = base64.b64decode(data[:num_header_chars])
        header = np.frombuffer(byte_string, dtype)

        block_sizes = header[3:]

        byte_array = base64.b64decode(data[num_header_chars:])
        dtype = vtu_to_np_type(data_type)

        byte_offsets = np.empty(block_sizes.shape[0] + 1, dtype=block_sizes.dtype)
        byte_offsets[0] = 0
        np.cumsum(block_sizes, out=byte_offsets[1:])

        # process the compressed data
        block_data = np.concatenate([
            np.frombuffer(
                zlib.decompress(byte_array[byte_offsets[k]: byte_offsets[k + 1]]),
                dtype=dtype,
            )
            for k in range(num_blocks)
        ])

        return block_data

    def _convert_data_array(self, block, attrib):
        data_block = self._decode_data_block(block, attrib['type'], "UInt32")

        if 'NumberOfComponents' in attrib:
            dim = int(attrib['NumberOfComponents'])
            columns = list(range(dim))
            for key in attrib:
                if key.startswith('ComponentName'):
                    columns[int(key[len('ComponentName'):])] = attrib[key]

            data = np.reshape(data_block, (-1, dim))

            if dim == 3 and columns[0] == 0:

                if 'Name' in attrib and attrib['Name'] == 'Points':
                    columns = ['X', 'Y', 'Z']
                else:
                    columns = ['Magnitude', 'X', 'Y', 'Z']
                    m = np.apply_along_axis(
                        lambda x: [(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5],
                        1, data)
                    data = np.hstack([m, data])

            return pd.DataFrame(
                data=data,
                columns=columns)

        return pd.DataFrame(data={'value': pd.Series(data_block)})


    @property
    def attributes(self):
        attributes = {}
        for field in self.point_data:
            attributes[field] = []

            axes = list(self.point_data[field].columns)

            index = 0

            for axis in axes:
                column = self.point_data[field][axis]

                extremes = {
                    'min': {
                        'node': int(column.idxmin()),
                        'point': self.points.loc[column.idxmin()].to_list(),
                        'value': column.min()
                    },
                    'max': {
                        'node': int(column.idxmax()),
                        'point': self.points.loc[column.idxmax()].to_list(),
                        'value': column.max()
                    },
                }

                attributes[field].append({
                    'id': axis,
                    'extremes': extremes,
                    'index': index
                })

                index += 1


        return attributes


if __name__ == '__main__':
    materials = [{'uuid': '8aef3dce-2402-4b9a-8551-a801ff76c2cf', 'markers': '*', 'markers_type': 'volume',
                  'name': 'Material 01', 'entity_type': 'Material',
                  'value': {'youngs_modulus': 200000000000, 'poissons_ratio': 0.3, 'density': 8000,
                            'ultimate_stress': 450000000, 'ultimate_strain': 0.4, 'yield_stress': 160000000,
                            'specific_heat_capacity': 500, 'thermal_conductivity': 45,
                            'thermal_expansion_coefficient': 1.2e-05}, 'enabled': True}]

    vtu_input = '/home/artem/ProveDesign/alpha/data/storage/bbcb00f3fdbc/result/fb62cca7e82f/calc.vtu'

    result_vtu = Vtu(vtu_input)

    result_path = '/home/artem/ProveDesign/alpha/data/storage/bbcb00f3fdbc/result/fb62cca7e82f'

    result_vtu.surface_save_to_np(result_path)
