import numpy as np
import re
import os
import pandas as pd
from scipy.optimize import minimize
# from multiprocessing import Process, Array


class Analyzer:
    def __init__(self, imp_list: list, surf_index: str, solution_energy_method, clear_without_path=None, clear_with_path=None):
        if clear_without_path is None:
            clear_without_path = f'../../MD_Al2O3/{surf_index}/Pristine_without_surf/STATIS'
        if clear_with_path is None:
            clear_with_path = f'../../MD_Al2O3/{surf_index}/Pristine_with_surf/STATIS'
        self.imp_list = imp_list
        self.solution_energy_method = solution_energy_method
        self.clear_without = Analyzer.extract_energies_from_STATIS(clear_without_path)[-1]
        self.clear_with = Analyzer.extract_energies_from_STATIS(clear_with_path)[-1]
        self.surf_index = surf_index
        self.bulk_energies = None
        self.energy_table = None
        self.solution_energies = None
        self.delta_H_seg = None
        self.coverage_atoms = None
        self.results = None

    def info(self):
        print('Surface index: ', self.surf_index)
        print('Energy of clear crystal without surface:', self.clear_without)
        print('Energy of clear crystal with surface:\t', self.clear_with)
        print('\nImp\tSolution Energy')
        for imp, sol_en in zip(self.imp_list, self.solution_energies):
            print(imp, ':\t', sol_en)
        print('\nSegregation Energy')
        for imp, seg_ens in zip(self.imp_list, self.delta_H_seg):
            print(imp)
            for cov, en in zip(self.coverage_atoms, seg_ens):
                print(cov, ':\t', en)

    @staticmethod
    def extract_energies_from_STATIS(filename):
        STATIS = open(filename, 'r')
        data = STATIS.readlines()
        data_length = len(data)
        for i, line in enumerate(data):
            data[i] = re.split(' ', line)
        for line in data:
            length = len(line)
            while length > 0:
                length -= 1
                if line[length] == "":
                    line.pop(length)
        energy = []
        for i in range(3, data_length, 9):
            energy.append(float(data[i][2]))
        STATIS.close()
        return energy

    @staticmethod
    def minimum_energy_search(surf_index, imp, coverage, copies_range):

        filenames = []
        for j in range(copies_range[0], copies_range[1]):
            filenames.append(f'../../MD_Al2O3/{surf_index}/{imp}/{imp}_{coverage}_{j}/STATIS')

        energies = []
        for filename in filenames:
            tmp_energy = Analyzer.extract_energies_from_STATIS(filename)
            if np.abs(tmp_energy[-1]) < 1e6:
                energies.append(tmp_energy[-1])
            else:
                energies.append(0)

        for i, en in enumerate(energies):
            print(i, ': ', en)
        print('')
        print('min is:', np.min(energies), np.argmin(energies))

    def solution_energy_calc(self, verbose=False):

        def solution_energy_bulk_phase(verbose):
            bulk_energies = []
            if verbose:
                for imp in self.imp_list:
                    file_path = f'../../MD_Al2O3/{self.surf_index}/{imp}/bulk_{imp}/STATIS'
                    tmp_energy = self.extract_energies_from_STATIS(file_path)[-1]
                    if tmp_energy > -1e19:
                        bulk_energies.append(tmp_energy)
                        print(tmp_energy, ' ', file_path)
                    else:
                        bulk_energies.append(-1)
                        print('convergence has not been achieved at ', file_path)
            else:
                for imp in self.imp_list:
                    file_path = f'../../MD_Al2O3/{self.surf_index}/{imp}/bulk_{imp}/STATIS'
                    tmp_energy = self.extract_energies_from_STATIS(file_path)[-1]
                    if tmp_energy > -1e19:
                        bulk_energies.append(tmp_energy)
                    else:
                        bulk_energies.append(-1)

            sol_energies = []

            for bulk_en in bulk_energies:
                sol_energies.append(bulk_en - self.clear_without)

            self.solution_energies = sol_energies
            return sol_energies

        def solution_energy_surf_phase(verbose):
            bulk_energies = []
            if verbose:
                for imp in self.imp_list:
                    file_path = f'../../MD_Al2O3/{self.surf_index}/bulk_one_imp_with_surf/bulk_{imp}/STATIS'
                    tmp_energy = self.extract_energies_from_STATIS(file_path)[-1]
                    if tmp_energy > -1e19:
                        bulk_energies.append(tmp_energy)
                        print(tmp_energy, ' ', file_path)
                    else:
                        bulk_energies.append(-1)
                        print('convergence has not been achieved at ', file_path)
            else:
                for imp in self.imp_list:
                    file_path = f'../../MD_Al2O3/{self.surf_index}/bulk_one_imp_with_surf/bulk_{imp}/STATIS'
                    tmp_energy = self.extract_energies_from_STATIS(file_path)[-1]
                    if tmp_energy > -1e19:
                        bulk_energies.append(tmp_energy)
                    else:
                        bulk_energies.append(-1)

            sol_energies = []

            for bulk_en in bulk_energies:
                sol_energies.append(bulk_en - self.clear_with)

            self.solution_energies = sol_energies
            return sol_energies

        options = {'bulk_method': solution_energy_bulk_phase,
                   'surf_method': solution_energy_surf_phase}

        return options.get(self.solution_energy_method, lambda: None)(verbose)

    def segr_analysis_with_coverage(self, coverage_atoms, copies_range, verbose=False):

        self.coverage_atoms = coverage_atoms

        energies = pd.DataFrame(columns=coverage_atoms, index=self.imp_list)

        filenames = []
        for imp in self.imp_list:
            filenames.append(f'../../MD_Al2O3/{self.surf_index}/{imp}/{imp}_1/STATIS')
        for filename, imp in zip(filenames, self.imp_list):
            tmp_energy = Analyzer.extract_energies_from_STATIS(filename)
            energies.at[imp, 1] = tmp_energy[-1]

        filenames = []
        for imp in self.imp_list:
            tmp = []
            for coverage in coverage_atoms[1::]:
                tmp_2 = []
                for j in range(copies_range[0], copies_range[1]):
                    tmp_2.append(f'../../MD_Al2O3/{self.surf_index}/{imp}/{imp}_{coverage}_{j}/STATIS')
                tmp.append(tmp_2)
            filenames.append(tmp)
        for filename, imp in zip(filenames, self.imp_list):
            for name, cov in zip(filename, coverage_atoms[1::]):
                tmp = []
                for n in name:
                    tmp_energy = self.extract_energies_from_STATIS(n)
                    if np.abs(tmp_energy[-1]) < 1e6:
                        tmp.append(tmp_energy[-1])
                if len(tmp) == 0:
                    tmp.append(0)
                if verbose:
                    print(imp, cov, np.argmin(tmp))
                energies.at[imp, cov] = np.min(tmp)

        self.energy_table = energies

        solution_energy = self.solution_energy_calc()

        delta_H_seg = []
        for imp, sol_en in zip(self.imp_list, solution_energy):
            delta_H_seg_tmp = []
            for cov in coverage_atoms:
                delta_H_seg_tmp.append(1 / cov * (energies.at[imp, cov] - self.clear_with - cov * sol_en))
                if verbose:
                    print(imp, '  ', cov, '  ', delta_H_seg_tmp[-1])
            if verbose:
                print('')
            delta_H_seg.append(delta_H_seg_tmp)

        self.delta_H_seg = delta_H_seg

        return delta_H_seg

    def segr_analysis_with_coverage_parallel(self, coverage_atoms, copies_range, n_proc, verbose=False):
        pass

    def fit(self, func_type: str, number_surf_atoms: int):
        """
        func_type:
            1: c[0] + c[1] * np.log(x)
            2: c[0] + c[1] * x + c[2] * x * x + c[3] * np.log(x) + c[4] * np.tanh(x)
            3: c[0] + c[1] * x + c[2] * x * x + c[3] * np.log(x) + c[4] * np.tanh(x) + c[5] * x ** (-0.5)
            4: c[0] + c[1] * np.log(x) + c[2] * x ** (-0.5)
        :param func_type: 1,2,3,4
        :param number_surf_atoms: integer number
        :return: X, Y
        """

        def fit_with_func_1(segr_en):

            def error_func(c, x_exp, y_exp):
                error = 0
                for x_, y_ in zip(x_exp, y_exp):
                    tmp = (y_ - c[0] - c[1] * np.log(x_)) ** 2
                    error += tmp
                return error

            def f(x, c):
                return c[0] + c[1] * np.log(x)

            X_i = np.array(self.coverage_atoms) / (number_surf_atoms - np.array(self.coverage_atoms))
            results = []
            for en in segr_en:
                result = minimize(error_func, np.array([1, 1]), args=(X_i, en))
                results.append(result.x)
            x = np.arange(0.005, 7.5, 0.001)
            y = []
            for result in results:
                y.append(f(x, result))

            self.results = results

            return x, y

        def fit_with_func_2(segr_en):

            def error_func(c, x_exp, y_exp):
                error = 0
                for x_, y_ in zip(x_exp, y_exp):
                    tmp = (y_ - c[0] - c[1] * x_ - c[2] * x_ * x_ - c[3] * np.log(x_) - c[4] * np.tanh(x_)) ** 2
                    error += tmp
                return error

            def f(x, c):
                return c[0] + c[1] * x + c[2] * x * x + c[3] * np.log(x) + c[4] * np.tanh(x)

            X_i = np.array(self.coverage_atoms) / (number_surf_atoms - np.array(self.coverage_atoms))
            results = []
            for en in segr_en:
                result = minimize(error_func, np.array([1, 1, 1, 1, 1]), args=(X_i, en))
                results.append(result.x)
            x = np.arange(0.005, 7.5, 0.001)
            y = []
            for result in results:
                y.append(f(x, result))

            self.results = results

            return x, y

        def fit_with_func_3(segr_en):

            def error_func(c, x_exp, y_exp):
                error = 0
                for x_, y_ in zip(x_exp, y_exp):
                    tmp = (y_ - c[0] - c[1] * x_ - c[2] * x_ * x_ - c[3] * np.log(x_) -
                           c[4] * np.tanh(x_) - c[5] * x_ ** (-0.5)) ** 2
                    error += tmp
                return error

            def f(x, c):
                return c[0] + c[1] * x + c[2] * x * x + c[3] * np.log(x) + c[4] * np.tanh(x) + \
                       c[5] * x ** (-0.5)

            X_i = np.array(self.coverage_atoms) / (number_surf_atoms - np.array(self.coverage_atoms))
            results = []
            for en in segr_en:
                result = minimize(error_func, np.array([1, 1, 1, 1, 1, 1]), args=(X_i, en))
                results.append(result.x)
            x = np.arange(0.005, 7.5, 0.001)
            y = []
            for result in results:
                y.append(f(x, result))

            self.results = results

            return x, y

        def fit_with_func_4(segr_en):

            def error_func(c, x_exp, y_exp):
                error = 0
                for x_, y_ in zip(x_exp, y_exp):
                    tmp = (y_ - c[0] - c[1] * np.log(x_) - c[2] * x_ ** (-0.5)) ** 2
                    error += tmp
                return error

            def f(x, c):
                return c[0] + c[1] * np.log(x) + c[2] * x ** (-0.5)

            X_i = np.array(self.coverage_atoms) / (number_surf_atoms - np.array(self.coverage_atoms))
            results = []
            for en in segr_en:
                result = minimize(error_func, np.array([1, 1, 1]), args=(X_i, en))
                results.append(result.x)
            x = np.arange(1 / 108, 7.5, 0.001)
            y = []
            for result in results:
                y.append(f(x, result))

            self.results = results

            return x, y

        options = {'1': fit_with_func_1, '2': fit_with_func_2,
                   '3': fit_with_func_3, '4': fit_with_func_4}

        return options.get(func_type, lambda: None)(self.delta_H_seg)

    def cluster_Vo_Al_imp_calc(self, deleted_O, verbose=False):
        energy = []
        for imp in self.imp_list:
            energy_for_each_O = []
            for O in deleted_O:
                tmp_energy = []
                list_dir = os.listdir(f'../../MD_Al2O3/{self.surf_index}/2{imp}_Vo/{O}')
                for folder in list_dir:
                    file_path = f'../../MD_Al2O3/{self.surf_index}/2{imp}_Vo/{O}/{folder}/STATIS'
                    en = self.extract_energies_from_STATIS(file_path)
                    tmp_energy.append(en[-1])
                energy_for_each_O.append(np.min(tmp_energy))
                if verbose:
                    print(f'{imp} {O}: ', np.argmin(tmp_energy))
            energy.append(energy_for_each_O)
        return energy

    def segr_en_from_radius(self):
        delta_H_seg = []

        delta_H_bulk = self.solution_energy_calc()

        for imp, bulk_en in zip(self.imp_list, delta_H_bulk):
            file_path = f'../../MD_Al2O3/{self.surf_index}/{imp}/{imp}_1/STATIS'
            surf_en = self.extract_energies_from_STATIS(file_path)[-1]
            delta_H_seg.append(surf_en - self.clear_with - bulk_en)

        return delta_H_seg

    def calc_depth(self, Al_right_order):
        energy = []
        solution_energy = self.solution_energy_calc()

        for imp, sol_en in zip(self.imp_list, solution_energy):
            energy_for_each_imp = []
            for Al in Al_right_order:
                file_path = f'../../MD_Al2O3/{self.surf_index}/{imp}_depth/{Al}/STATIS'
                energy_for_each_imp.append(self.extract_energies_from_STATIS(file_path)[-1] - self.clear_with - sol_en)
            # energy_for_each_imp = np.array(energy_for_each_imp)
            # energy_for_each_imp = energy_for_each_imp - energy_for_each_imp[-1]
            energy.append(energy_for_each_imp)

        return energy
