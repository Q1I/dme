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
    def _bcva(self, id, encode = True): # m0
        return self.get_extra_value('Baseline BCVA (LogMAR)', id, encode)
    @ingredient.capture
    def _bcva_m3(self, id):
        return self.get_extra_value('Visual acuity Month 3 (logMAR)', id)
    @ingredient.capture
    def _bcva_m12(self, id):
        return self.get_extra_value('Visual acuity Month 12 (logMAR)', id)
    @ingredient.capture
    def _bcva_delta_m0_m12(self, id, encode = True):
        return self.get_extra_value('Visual acuity change from Month 0 to Month 12 (letters)', id, encode)
    @ingredient.capture
    def _bcva_delta_m0_m3(self, id, encode = True):
        return self.get_extra_value('Visual acuity change baseline to month 3, letters', id, encode)
    # Eye (1-right, 2-left)
    @ingredient.capture
    def _eye(self, id):
        return self.get_extra_value('Eye (1-right, 2- left)', id, False)
    # Central subfield Thickness baseline (μm)
    @ingredient.capture
    def _cstb(self, id, encode = True):
        return self.get_extra_value('Central subfield Thickness baseline (µm)', id, encode)
    # Maximal retina thickness, baseline (µm)
    @ingredient.capture
    def _mrtb(self, id, encode = True):
        return self.get_extra_value('Maximal retina thickness, baseline (µm)', id, encode)
    # HbA1c at DME diagnosis, (%)
    @ingredient.capture
    def _hba1c(self, id, encode = True):
        array = self.get_extra_value('HbA1c at DME diagnosis, (%)', id, encode)
        return array
    # Status post PRP (1-yes, 2-no)
    @ingredient.capture
    def _prp(self, id):
        return self.get_extra_value('Status post PRP (1-yes, 2-no)', id, False)
    # Lens status (1-phakic, 2- pseudophakic)
    @ingredient.capture
    def _lens(self, id):
        return self.get_extra_value('Lens status (1-phakic, 2- pseudophakic)', id, False)   
    # Diabetic retinopathy (1-NPDR, 2-PDR)
    @ingredient.capture
    def _pdr(self, id):
        return self.get_extra_value('Diabetic retinopathy (1-NPDR, 2-PDR)', id, False)
    # Gender (1-male, 2-female)
    @ingredient.capture
    def _gender(self, id):
        return self.get_extra_value('Gender (1-male, 2-female)', id, False)
    # Anti-VEGF drug intially injected (1-ranibizumab, 2-aflibercept, 3-bevacizumab)
    @ingredient.capture
    def _avegf(self, id):
        return self.get_extra_value('Anti-VEGF drug intially injected (1-ranibizumab, 2-aflibercept, 3-bevacizumab)', id, False)
    # Age at DME diagnosis (years)
    @ingredient.capture
    def _age(self, id, encode):
        return self.get_extra_value('Age at DME diagnosis (years)', id, encode)
    # duration of DM (months)
    @ingredient.capture
    def _duration(self, id):
        return self.get_extra_value('duration of DM (months)', id) / 492
    # Surgery during 12 months? (1-yes, 2-no)
    @ingredient.capture
    def _surgery(self, id):
        return self.get_extra_value('Surgery during 12 months? (1-yes, 2-no)', id, False)
        
    def get_extra_value(self, column_name, id, encode = True):
        value = self.extras_csv.loc[self.extras_csv['ID'] == id][column_name].values[0]
        mean = self.extras_csv[column_name].mean()
        std = self.extras_csv[column_name].std()
        
        # print('v, avg, std', value, mean, std)
        if np.isnan(value):
            return 0, 0
        else:
            extra = ((value - mean) / std ) if encode else value
            return extra, 1

    def get_column_average(self, column_name):
        return self.extras_csv[column_name].mean()

    def get_column_std(self, column_name):
        return self.extras_csv[column_name].std()

    @ingredient.capture
    def _hba1c_orig(self, id):
        array = self.get_extra_value('HbA1c at DME diagnosis, (%)', id, False)
        return array

    @ingredient.capture
    def get_extras_orig(self,id):
        extras = np.zeros((self.num_extra, 2))
        for i, extra in enumerate(self.extras):
            if extra == 'hba1c':
                extras[i] = self._hba1c_orig(id)
        return extras

    @ingredient.capture
    def get_extras(self,id, encode = True):
        extras = np.zeros((self.num_extra, ))
        extras_msk = np.zeros((self.num_extra, ))
        for i, extra in enumerate(self.extras):
            if extra == 'bcva':
                extras[i], extras_msk[i] = self._bcva(id, encode)
            # if extra == 'bcva_m3':
            #     extras[i] = self._bcva_m3(id))
            # if extra == 'bcva_m12':
            #     extras[i] = self._bcva_m12(id))
            if extra == 'bcva_delta_m0_m12':
                extras[i], extras_msk[i] = self._bcva_delta_m0_m12(id, encode)
            if extra == 'bcva_delta_m0_m3':
                extras[i], extras_msk[i] = self._bcva_delta_m0_m3(id, encode)
            if extra == 'cstb':
                extras[i], extras_msk[i] = self._cstb(id, encode)
            if extra == 'mrtb':
                extras[i], extras_msk[i] = self._mrtb(id, encode)
            if extra == 'hba1c':
                extras[i], extras_msk[i] = self._hba1c(id, encode)
            if extra == 'age':
                extras[i], extras_msk[i] = self._age(id, encode)
            # if extra == 'duration':
            #     extras[i] = self._duration(id))
            if extra.startswith('eye_'):
                eye = self._eye(id)
                if eye[0] == 1: # 1-right, 2-left
                    eye_right = 1
                    eye_left = 0
                else:
                    eye_right = 0
                    eye_left = 1
                extras[i] = eye_right if extra == 'eye_right' else eye_left
                extras_msk[i] = eye[1]
            if extra.startswith('prp_'):
                prp = self._prp(id)
                if prp[0] == 1: # 1-yes, 2-no
                    prp_yes = 1
                    prp_no = 0
                else:
                    prp_yes = 0
                    prp_no = 1
                extras[i] = prp_yes if extra == 'prp_yes' else prp_no
                extras_msk[i] = prp[1]
            if extra.startswith('lens_'): # 1-phakic, 2-pseudophakic
                lens = self._lens(id)
                if lens[0] == 1: 
                    lens_phakic = 1
                    lens_pseudophakic = 0
                else:
                    lens_phakic = 0
                    lens_pseudophakic = 1
                extras[i] = lens_phakic if extra == 'lens_phakic' else lens_pseudophakic    
                extras_msk[i] = lens[1]
            if extra.startswith('pdr_'): # 1-NPDR, 2-PDR
                pdr = self._pdr(id)
                if pdr[0] == 1: 
                    pdr_n = 1
                    pdr_p = 0
                else:
                    pdr_n = 0
                    pdr_p = 1
                extras[i] = pdr_n if extra == 'pdr_n' else pdr_p
                extras_msk[i] = pdr[1]
            if extra.startswith('gender_'): # 1-male, 2-female
                gender = self._gender(id)
                if gender[0] == 1: 
                    gender_male = 1
                    gender_female = 0
                else:
                    gender_male = 0
                    gender_female = 1
                extras[i] = gender_male if extra == 'gender_male' else gender_female
                extras_msk[i] = gender[1]
            if extra.startswith('avegf_'): # 1-ranibizumab, 2-aflibercept, 3-bevacizumab
                avegf = self._avegf(id)
                if avegf[0] == 1: 
                    avegf_ranibizumab = 1
                    avegf_aflibercept = 0
                    avegf_bevacizumab = 0
                elif avegf[0]  == 2:
                    avegf_ranibizumab = 0
                    avegf_aflibercept = 1
                    avegf_bevacizumab = 0
                else:
                    avegf_ranibizumab = 0
                    avegf_aflibercept = 0
                    avegf_bevacizumab = 1
                if extra == 'avegf_ranibizumab':
                    extras[i] = avegf_ranibizumab
                if extra == 'avegf_aflibercept':
                    extras[i] = avegf_aflibercept
                if extra == 'avegf_bevacizumab':
                    extras[i] = avegf_bevacizumab
                extras_msk[i] = avegf[1]
            if extra.startswith('surgery_'): # 1-yes, 2-no
                surgery = self._surgery(id)
                if surgery[0] == 1: 
                    surgery_yes = 1
                    surgery_no = 0
                else:
                    surgery_yes = 0
                    surgery_no = 1
                extras[i] = surgery_no if extra == 'surgery_no' else surgery_yes
                extras_msk[i] = surgery[1]
            if extra == 'no-extras':
                extras[i] = 0
            # print(i, extra, extras)
        return extras, 0 if 0 in extras_msk else 1