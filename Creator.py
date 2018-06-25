import numpy as np
import re
from random import sample
import os
import itertools


class Creator:
    def __init__(self, CONFIG_path, surf_index, CONTROL_path=None, FIELD_path=None):
        if CONTROL_path is None:
            CONTROL_path = f'../../MD_Al2O3/{surf_index}/Pristine_with_surf/CONTROL'
        self.CONFIG_path = CONFIG_path
        self.CONTROL_path = CONTROL_path
        self.FIELD_path = FIELD_path
        self.surf_index = surf_index

    @staticmethod
    def CONFIG_extract(filename: str):
        CONFIG = open(filename, 'r')
        for i in range(5):
            CONFIG.readline()
        data = CONFIG.readlines()
        for i, line in enumerate(data):
            data[i] = re.split(' ', line)
        for line in data:
            length = len(line)
            while length > 0:
                length -= 1
                if line[length] == "":
                    line.pop(length)

        names = []
        numbers = []
        x = []
        y = []
        z = []
        for i in range(0, len(data) - 1, 2):
            names.append(data[i][0])
            numbers.append(int(data[i][1]))
            x.append(float(data[i + 1][0]))
            y.append(float(data[i + 1][1]))
            z.append(float(data[i + 1][2]))
        return x, y, z, names, numbers

    @staticmethod
    def nearest_from(target, tol, x, y, z, names, numbers, atom_type=None):
        """
        :return: Numbers of nearest atoms
        """
        target -= 1
        returned_numbers = []
        for x_, y_, z_, number, name in zip(x, y, z, numbers, names):
            if np.sqrt((x_ - (x[target]))**2 + (y_ - (y[target]))**2 + (z_ - (z[target]))**2) < tol:
                if atom_type is None:
                    returned_numbers.append(number)
                else:
                    if name == atom_type:
                        returned_numbers.append(number)
        return returned_numbers

    def __make_CONFIG(self, folder_path, atoms_for_repl=None, atoms_names=None,
                      imp_names=None, atoms_for_del=None):
        """
        :param folder_path: str
        :param atoms_for_repl: list
        :param atoms_names: list
        :param imp_names: list
        :param atoms_for_del: list
        :return: unique_imps, list(number of each imp), int(number of Al), int(number of O)
        """
        CONFIG = open(self.CONFIG_path, 'r')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        OUTPUT = open(folder_path + '/CONFIG', 'w')
        data = CONFIG.readlines()
        CONFIG.close()
        atoms_numbers = {'Al': 13, 'O': 8, 'O_shl': 8, 'Mg': 12, 'Co': 27, 'Fe': 26, 'Cd': 28,
                         'Ca': 20, 'Sr': 38, 'Ba': 56}

        if atoms_for_repl is not None:
            for replacing_atom, atom_name, imp_name, in zip(atoms_for_repl, atoms_names, imp_names):
                replaced_string = f'{atom_name}\s+{replacing_atom}\s+{atoms_numbers[atom_name]}\n'
                for i in range(len(data)):
                    data[i] = re.sub(re.compile(replaced_string),
                                     f'{imp_name}               {replacing_atom}        {atoms_numbers[imp_name]}\n',
                                     data[i])

        if atoms_for_del is not None:
            atoms_for_del.sort()
            atoms_for_del = atoms_for_del[::-1]

            for deleting_atom in atoms_for_del:
                data.pop(6+2*(deleting_atom-1))
                data.pop(5+2*(deleting_atom-1))

        data_al = []
        data_o = []
        data_o_shl = []

        unique_imps = list(set(imp_names))
        data_imps = [[] for _ in range(len(unique_imps))]

        for i in range(len(data)):
            if 'Al' in data[i]:
                data_al.append(data[i])
                data_al.append(data[i + 1])
            if 'O ' in data[i]:
                data_o.append(data[i])
                data_o.append(data[i + 1])
            if 'O_shl' in data[i]:
                data_o_shl.append(data[i])
                data_o_shl.append(data[i + 1])
            for j, imp in enumerate(unique_imps):
                if imp in data[i]:
                    data_imps[j].append(data[i])
                    data_imps[j].append(data[i + 1])

        for line in data[:5]:
            OUTPUT.write(line)
        for imp_type in data_imps:
            for line in imp_type:
                OUTPUT.write(line)
        for line in data_al:
            OUTPUT.write(line)

        for i in np.arange(0, len(data_o), 2):
            OUTPUT.write(data_o[i])
            OUTPUT.write(data_o[i + 1])
            OUTPUT.write(data_o_shl[i])
            OUTPUT.write(data_o_shl[i + 1])

        OUTPUT.close()

        return unique_imps, list(map(lambda x: int(0.5*x), list(map(len, data_imps)))),\
               int(len(data_al)/2), int(len(data_o)/2)

    def __make_FIELD(self, folder_path, unique_imps, len_imps, potentials,
                     masses, charges, len_Al, len_O):
        OUTPUT = open(folder_path + '/FIELD', 'w')
        OUTPUT.write(f'DL_POLY Al2O3 with {unique_imps}\n')
        OUTPUT.write('UNITS eV\n')
        OUTPUT.write(f'MOLECULES {len(unique_imps) + 2}\n')

        for imp, amount_imp, mass, charge in zip(unique_imps, len_imps, masses, charges):
            OUTPUT.write(f'{imp}_mol\n')
            OUTPUT.write('NUMMOLS    1\n')
            OUTPUT.write(f'ATOMS       {amount_imp}\n')
            OUTPUT.write(f'{imp}     {mass}   {charge}     {amount_imp}\n')
            OUTPUT.write('FINISH\n')

        OUTPUT.write('Al_mol\n')
        OUTPUT.write('NUMMOLS    1\n')
        OUTPUT.write(f'ATOMS       {len_Al}\n')
        OUTPUT.write(f'Al     26.982   3     {len_Al}\n')
        OUTPUT.write('FINISH\n')

        OUTPUT.write('O_mol\n')
        OUTPUT.write(f'NUMMOLS    {len_O}\n')
        OUTPUT.write('ATOMS 2\n')
        OUTPUT.write('O      15.999  -2.04  1\n')
        OUTPUT.write('O_shl  0        0.04  1\n')
        OUTPUT.write('SHELL 1 2\n')
        OUTPUT.write('1  2  6.3  0\n')
        OUTPUT.write('FINISH\n')

        OUTPUT.write(f'VDW {len(unique_imps) + 2}\n')
        OUTPUT.write('O  O  buck  9547.96 0.2192 32.0\n')
        OUTPUT.write('Al O  buck  1120.04 0.3125 0.0\n')
        for pot in potentials:
            OUTPUT.write(f'{pot}')
        OUTPUT.write('CLOSE')

        OUTPUT.close()

    def __make_CONTROL(self, folder_path, traj_step):
        OUTPUT = open(folder_path + '/CONTROL', 'w')
        OUTPUT.write('CONTROL file generated by DL_POLY/java utility\n\n')

        OUTPUT.write('steps              2000\n')
        OUTPUT.write('multiple step         1\n')
        OUTPUT.write('print                50\n')
        OUTPUT.write('stack                50\n')
        OUTPUT.write('stats                 1\n')
        OUTPUT.write(f'traj             0 {traj_step} 0\n\n')

        OUTPUT.write('optim energy      0.1\n\n')

        OUTPUT.write('timestep         0.0010\n')
        OUTPUT.write('cutoff           15.000\n')
        OUTPUT.write('delr width       15.000\n')
        OUTPUT.write('rvdw cutoff      15.000\n')
        OUTPUT.write('ewald precision  1.0E-5\n')
        OUTPUT.write('shake tolerance  1.0E-5\n')
        OUTPUT.write('quaternion tolerance  1.0E-5\n\n')

        OUTPUT.write('job time              400000.0\n')
        OUTPUT.write('close time               500.0\n\n')

        OUTPUT.write('finish')
        OUTPUT.close()

    def fractional_coverage(self, impurities_list, potentials, masses,
                            coverages, atoms_for_replacement, copies_range):
        for imp, pot, mass in zip(impurities_list, potentials, masses):
            for coverage in coverages:
                for copy in range(copies_range[0], copies_range[1]):
                    folder_path = f'{self.surf_index}/{imp}_{coverage}_{copy}'
                    sequence = sample(atoms_for_replacement, coverage)
                    unique_imps, len_imps, len_Al, len_O = self.__make_CONFIG(folder_path, sequence,
                                                                              ['Al'] * coverage, [imp] * coverage)
                    self.__make_FIELD(folder_path, unique_imps, len_imps, [pot], [mass], [3], len_Al, len_O)
                    self.__make_CONTROL(folder_path, 50)

    def bulk_state_for_one_imp(self, impurities_list, potentials, masses,
                               atom_for_replacement):
        for imp, pot, mass in zip(impurities_list, potentials, masses):
            folder_path = f'{self.surf_index}/bulk_{imp}'
            unique_imps, len_imps, len_Al, len_O = self.__make_CONFIG(folder_path, [atom_for_replacement],
                                                                      ['Al'], [imp])
            self.__make_FIELD(folder_path, unique_imps, len_imps, [pot], [mass], [3], len_Al, len_O)
            self.__make_CONTROL(folder_path, 50)

    def surf_state_for_one_imp(self, impurities_list, potentials, masses, atom_for_replacement):
        for imp, pot, mass in zip(impurities_list, potentials, masses):
            folder_path = f'{self.surf_index}/{imp}_1'
            unique_imps, len_imps, len_Al, len_O = self.__make_CONFIG(folder_path, [atom_for_replacement],
                                                                      ['Al'], [imp])
            self.__make_FIELD(folder_path, unique_imps, len_imps, [pot], [mass], [3], len_Al, len_O)
            self.__make_CONTROL(folder_path, 50)

    def cluster_Vo_Al_imp(self, O_for_del, O_shl_for_del, impurities_list, potentials, masses):
        x, y, z, names, numbers = self.CONFIG_extract(self.CONFIG_path)
        atoms_names = ['Al'] * 2
        for O, O_shl in zip(O_for_del, O_shl_for_del):
            atoms_for_del = [O, O_shl]
            Al_for_replace = self.nearest_from(O, 3, x, y, z, names, numbers, atom_type='Al')
            for imp, pot, mass in zip(impurities_list, potentials, masses):
                for count, Al_combination in enumerate(itertools.combinations(Al_for_replace, 2)):
                    Al_for_change = list(Al_combination)
                    folder_path = f'{self.surf_index}/2{imp}_Vo/{O}/{count}'
                    unique_imps, len_imps, len_Al, len_O = self.__make_CONFIG(folder_path, Al_for_change,
                                                                              atoms_names, [imp] * 2,
                                                                              atoms_for_del)
                    self.__make_FIELD(folder_path, unique_imps, len_imps, [pot], [mass], [2], len_Al, len_O)
                    self.__make_CONTROL(folder_path, 50)
