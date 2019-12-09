from sacred import Experiment
import json
from sacred.observers import FileStorageObserver

from ingredients.parse_months import ingredient as parse_months_ingredient, parse_months_run

#################
## ingredients ##
#################
ingredients = [parse_months_ingredient]

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

@parse_months_ingredient.config
def update_cfg():
    start_at_x = 510
    cut_y = 124
    file_path = '/home/q1/Python/dl/data/uniklinik_augen/dme-data'

@ex.automain
def run():
    parse_months_run()

