import numpy as np
import re


class Selector:
    def __init__(self, config_path: str, surf_index: str):
        self.config_path = config_path
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

    def select_surf_atoms(self, tol=0.1):

        def select_surf_atoms_012(tol):

            def surf_equation(x, y, z, tol):
                result = -40.629 * x - 26.618 * z + 968.607
                if np.abs(result) < tol:
                    return 1
                else:
                    return 0

            x, y, z, names, numbers = self.CONFIG_extract(self.config_path)
            is_surf = []
            for x_, y_, z_ in zip(x, y, z):
                is_surf.append(surf_equation(x_, y_, z_, tol))
            list_of_indices = [i for i, x in enumerate(is_surf) if x == 1]
            return [numbers[i] for i in list_of_indices]

        def select_surf_atoms_113(tol):

            def surf_equation(x, y, z, tol):

                result = -0.002*x + 34.996*y + 19.748*z - 493.743
                if np.abs(result) < tol:
                    return 1
                else:
                    return 0

            x, y, z, names, numbers = self.CONFIG_extract(self.config_path)
            is_surf = []
            for x_, y_, z_ in zip(x, y, z):
                is_surf.append(surf_equation(x_, y_, z_, tol))
            list_of_indices = [i for i, x in enumerate(is_surf) if x == 1]
            return [numbers[i] for i in list_of_indices]

        def select_surf_atoms_001(tol):

            def surf_equation(x, y, z, tol):
                if z > 25:
                    return 1
                else:
                    return 0

            x, y, z, names, numbers = self.CONFIG_extract(self.config_path)
            is_surf = []
            for x_, y_, z_ in zip(x, y, z):
                is_surf.append(surf_equation(x_, y_, z_, tol))
            list_of_indices = [i for i, x in enumerate(is_surf) if x == 1]
            return [numbers[i] for i in list_of_indices]

        options = {'012': select_surf_atoms_012,
                   '113': select_surf_atoms_113,
                   '001': select_surf_atoms_001}

        return options.get(self.surf_index, lambda: None)(tol)

    def select_xyz(self, xrange: list, yrange: list, zrange: list, correct_name: str):
        """
        Return numbers of atoms 
        """
        def check(variables):
            x, y, z, name = variables
            if xrange[0] < x < xrange[1] and yrange[0] < y < yrange[1] and \
                                    zrange[0] < z < zrange[1] and name == correct_name:
                return 1
            else:
                return 0

        x, y, z, names, numbers = self.CONFIG_extract(self.config_path)
        is_correct = list(map(check, zip(x, y, z, names)))
        list_of_indices = [i for i, x in enumerate(is_correct) if x == 1]
        return [numbers[i] for i in list_of_indices]
