import csv
import pandas as pd
import numpy as np

from sacred import Ingredient

ingredient = Ingredient('extras')

@ingredient.capture
class Extras():
    @ingredient.capture
    def __init__(self, excel_path, extras, num_extra):
        self.extras_csv = pd.read_excel(excel_path)
        self.extras = extras
        self.num_extra = num_extra

    # extras
    @ingredient.capture
    def _bcva(self, id): # m0
        return self.get_extra_value('Baseline BCVA (LogMAR)', id)
    @ingredient.capture
    def _bcva_m3(self, id):
        return self.get_extra_value('Visual acuity Month 3 (logMAR)', id)
    @ingredient.capture
    def _bcva_m12(self, id):
        return self.get_extra_value('Visual acuity Month 12 (logMAR)', id)
    @ingredient.capture
    def _bcva_delta_m0_m12(self, id):
        return self.get_extra_value('Visual acuity change from Month 0 to Month 12 (letters)', id)
    # Central subfield Thickness baseline (μm)
    @ingredient.capture
    def _cstb(self, id):
        return self.get_extra_value('Central subfield Thickness baseline (µm)', id) / 1000
    # Maximal retina thickness, baseline (µm)
    @ingredient.capture
    def _mrtb(self, id):
        return self.get_extra_value('Maximal retina thickness, baseline (µm)', id) / 1000
    # HbA1c at DME diagnosis, (%)
    @ingredient.capture
    def _hba1c(self, id):
        array = self.get_extra_value('HbA1c at DME diagnosis, (%)', id)
        return array
    # Status post PRP (1-yes, 2-no)
    @ingredient.capture
    def _prp(self, id):
        return self.get_extra_value('Status post PRP (1-yes, 2-no)', id)
    # Lens status (1-phakic, 2- pseudophakic)
    @ingredient.capture
    def _lens(self, id):
        return self.get_extra_value('Lens status (1-phakic, 2- pseudophakic)', id)   
    # Diabetic retinopathy (1-NPDR, 2-PDR)
    @ingredient.capture
    def _pdr(self, id):
        return self.get_extra_value('Diabetic retinopathy (1-NPDR, 2-PDR)', id)
    # Gender (1-male, 2-female)
    @ingredient.capture
    def _gender(self, id):
        return self.get_extra_value('Gender (1-male, 2-female)', id)
    # Anti-VEGF drug intially injected (1-ranibizumab, 2-aflibercept, 3-bevacizumab)
    @ingredient.capture
    def _avegf(self, id):
        return self.get_extra_value('Anti-VEGF drug intially injected (1-ranibizumab, 2-aflibercept, 3-bevacizumab)', id)
    # Age at DME diagnosis (years)
    @ingredient.capture
    def _age(self, id):
        return self.get_extra_value('Age at DME diagnosis (years)', id)
    # duration of DM (months)
    @ingredient.capture
    def _duration(self, id):
        return self.get_extra_value('duration of DM (months)', id) / 492
    # Surgery during 12 months? (1-yes, 2-no)
    @ingredient.capture
    def _surgery(self, id):
        return self.get_extra_value('Surgery during 12 months? (1-yes, 2-no)', id)
        
    def get_extra_value(self, column_name, id):
        value = self.extras_csv.loc[self.extras_csv['ID'] == id][column_name].values[0]
        mean = self.extras_csv[column_name].mean()
        std = self.extras_csv[column_name].std()
        
        # print('v, avg, std', value, mean, std)
        if np.isnan(value):
            return [0, 0]
        else:
            return [value / std - mean, 1]

    def get_column_average(self, column_name):
        return self.extras_csv[column_name].mean()

    def get_column_std(self, column_name):
        return self.extras_csv[column_name].std()

    @ingredient.capture
    def get_extras(self,id):
        extras = np.zeros((self.num_extra, 2))
        for i, extra in enumerate(self.extras):
            # if extra == 'bcva':
            #     extras.append(self._bcva(id))
            # if extra == 'bcva_m3':
            #     extras.append(self._bcva_m3(id))
            # if extra == 'bcva_m12':
            #     extras.append(self._bcva_m12(id))
            # if extra == 'bcva_delta_m0_m12':
            #     extras.append(self._bcva_delta_m0_m12(id))
            # if extra == 'cstb':
            #     extras.append(self._cstb(id))
            # if extra == 'mrtb':
            #     extras.append(self._mrtb(id))
            if extra == 'hba1c':
                extras[i] = self._hba1c(id)
            # if extra == 'age':
            #     extras.append(self._age(id))
            # if extra == 'duration':
            #     extras.append(self._duration(id))
            # if extra == 'prp':
            #     prp = self._prp(id)
            #     if prp == 1: # 1-yes, 2-no
            #         extras.append(1)
            #         extras.append(0)
            #     else:
            #         extras.append(0)
            #         extras.append(1)
            # if extra == 'lens': # 1-phakic, 2-pseudophakic
            #     lens = self._lens(id)
            #     if lens == 1: 
            #         extras.append(1)
            #         extras.append(0)
            #     else:
            #         extras.append(0)
            #         extras.append(1)
            # if extra == 'pdr': # 1-NPDR, 2-PDR
            #     pdr = self._pdr(id)
            #     if pdr == 1: 
            #         extras.append(1)
            #         extras.append(0)
            #     else:
            #         extras.append(0)
            #         extras.append(1)
            # if extra == 'gender': # 1-male, 2-female
            #     gender = self._gender(id)
            #     if gender == 1: 
            #         extras.append(1)
            #         extras.append(0)
            #     else:
            #         extras.append(0)
            #         extras.append(1)
            # if extra == 'avegf': # 1-ranibizumab, 2-aflibercept, 3-bevacizumab
            #     avegf = self._avegf(id)
            #     if avegf == 1: 
            #         extras.append(1)
            #         extras.append(0)
            #         extras.append(0)
            #     elif avegf == 2:
            #         extras.append(0)
            #         extras.append(1)
            #         extras.append(0)
            #     else:
            #         extras.append(0)
            #         extras.append(0)
            #         extras.append(1)
            # if extra == 'surgery':
            #     surgery = self._surgery(id)
            #     if surgery == 1: 
            #         extras.append(1)
            #         extras.append(0)
            #     else:
            #         extras.append(0)
            #         extras.append(1)
            # if extra == 'no-extras':
            #     extras.append(0)
                
        return extras