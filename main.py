from sacred import Experiment
import json
from sacred.observers import FileStorageObserver

# from ingredients.parse_months import ingredient as parse_months_ingredient, parse_months_run
from ingredients.dme import ingredient as dme_ingredient, dme_run

#################
## ingredients ##
#################
# ingredients = [parse_months_ingredient, dme_ingredient]
ingredients = [dme_ingredient]

ex = Experiment('dme',  ingredients)

path = ''
# path = '/scratch/ws/trng859b-dme/'

##############
## observer ##
##############
ex.observers.append(FileStorageObserver(path + 'logs'))

############
## config ##
############
ex.add_config('core/config.json')

@ex.config
def default():
    """Default Configuration"""
    title = 'dme'

# @parse_months_ingredient.config
# def update_cfg():
#     start_at_x = 510
#     cut_y = 124
#     file_path = '/home/q1/Python/dl/data/uniklinik_augen_unique'

@dme_ingredient.config
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
    verbose = 2
    patience = 40
    use_baseline = True
    evenly_distributed = False

@ex.automain
def run(_run, title, dme):
    # parse_months_run()
    title += ' - visus' if dme['use_baseline'] else ' - no extras'
    title += ' - evenly distributed examples' if dme['evenly_distributed'] else ''
    print('Start experiment: %s' % title)
    dme_run(_run, title)
    print('End experiment: %s' % title)

