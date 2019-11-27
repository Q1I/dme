from sacred import Experiment
import json
from sacred.observers import FileStorageObserver

from ingredients.parse_months import ingredient as parse_months_ingredient, parse_months_run
from ingredients.dme import ingredient as dme_ingredient, dme_run

#################
## ingredients ##
#################
ingredients = [parse_months_ingredient, dme_ingredient]

ex = Experiment('dme',  ingredients)

##############
## observer ##
##############
ex.observers.append(FileStorageObserver('logs'))

############
## config ##
############
ex.add_config('core/config.json')

@ex.config
def default():
    """Default Configuration"""
    title = 'dme - visus'

@parse_months_ingredient.config
def update_cfg():
    start_at_x = 510
    cut_y = 124
    file_path = '/home/q1/Python/dl/data/uniklinik_augen'

@dme_ingredient.config
def update_cfg():
    """Configuration >> DME"""
    num_examples = 2
    input_size = 30
    generator_batch_size = 109
    numpy_source_path = 'data/parsed'
    dropout_rate = 0.5
    filters = 32
    fit_batch_size = 32
    epochs = 8
    steps_per_epoch = 100
    excel_path = 'data/dme-extras.xlsx'
    model_save_path = 'data/models/'

@ex.automain
def run(_run, title):
    # parse_months_run()
    print('Start experiment: %s' % title)
    dme_run(_run._id)

