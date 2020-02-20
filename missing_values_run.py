from sacred import Experiment
import json
from sacred.observers import FileStorageObserver

# from ingredients.parse_months import ingredient as parse_months_ingredient, parse_months_run
from ingredients.dme import ingredient as dme_ingredient, dme_run
from ingredients.extras import ingredient as extras_ingredient
from ingredients.logging import ingredient as logging_ingredient
from ingredients.missing_values import ingredient as missing_values_ingredient, missing_values_run

#################
## ingredients ##
#################
# ingredients = [parse_months_ingredient, dme_ingredient]
ingredients = [missing_values_ingredient, extras_ingredient, logging_ingredient]

ex = Experiment('dme',  ingredients)

# path = ''
path = '/scratch/ws/trng859b-dme/'

##############
## observer ##
##############
ex.observers.append(FileStorageObserver(path + 'logs'))

############
## config ##
############
ex.add_config('core/config.json')

# extras = ['bcva','cstb','mrtb','hba1c','prp', 'lens', 'pdr', 'gender', 'avegf', 'age', 'duration']
# num_extra = len(extras) + 6 #(prp_yes, prp_no, lens_phakic, lens_pseudophakic, pdr_npdr, pdr_pdr, gender_male, gender_female, avegf_ranibizumab, avegf_aflibercept, avegf_bevacizumab)
extras = [
'age',
'bcva',
'bcva_delta_m0_m12',
'cstb',
'mrtb',
'hba1c',
'prp_yes', 'prp_no',
'lens_phakic', 'lens_pseudophakic',
'pdr_n', 'pdr_p',
'gender_male', 'gender_female',
'avegf_ranibizumab', 'avegf_aflibercept', 'avegf_bevacizumab']
num_extra = len(extras)

@ex.config
def default():
    """Default Configuration"""
    title = 'dme'

# @parse_months_ingredient.config
# def update_cfg():
#     start_at_x = 510
#     cut_y = 124
#     file_path = '/home/q1/Python/dl/data/uniklinik_augen_unique'

@missing_values_ingredient.config
def update_cfg():
    """Configuration >> DME"""
    num_examples = 3
    input_size = 128 # 128
    batch_size = 16 # 16
    numpy_source_path = path + 'data/parsed'
    dropout_rate = 0.2
    filters = 32
    epochs = 100
    excel_path = path + 'data/dme-extras.xlsx'
    model_save_path = path + 'data/models/'
    history_save_path = path + 'logs/'
    predictions_save_path = path + 'data/predictions/'
    verbose = 2
    patience = 60
    evenly_distributed = False
    test_all = False
    extras = extras
    num_extra = num_extra
    n_splits = 5
    n_repeats = 1
    validation_ids = ['A063', 'A064', 'A065', 'A066', 'A067', 'A090', 'A091', 'A092', 'A093', 'A094', 'A095', 'A096', 'A097', 'A098', 'A099', 'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111']
    # validation_ids = None
    use_validation = False

@extras_ingredient.config
def update_cfg():
    excel_path = path + 'data/dme-extras.xlsx'

@logging_ingredient.config
def update_cfg():
    scores_save_path = path + 'logs/'
    extras = extras

@ex.automain
def run(_run, title, missing_values):
    # parse_months_run()
    for label in missing_values['extras']:
        title += ' - ' + label
    print('Start experiment: %s' % title)
    missing_values_run(_run, title)
    print('End experiment: %s' % title)
    # dme_predict(_run)

