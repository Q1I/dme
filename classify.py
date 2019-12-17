from sacred import Experiment
import json
from sacred.observers import FileStorageObserver

# from ingredients.parse_months import ingredient as parse_months_ingredient, parse_months_run
from ingredients.dme import ingredient as dme_ingredient, dme_predict

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
    patience = 30
    evenly_distributed = True
    test_all = False
    extras = ['no-extras']
    validation_ids = ['A063', 'A064', 'A065', 'A066', 'A067', 'A091', 'A091', 'A092', 'A093', 'A094', 'A095', 'A096', 'A097', 'A098', 'A099', 'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111']
    use_validation = False

@ex.automain
def run(_run, title, dme):
    dme_predict(_run)

