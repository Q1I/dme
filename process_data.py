from sacred import Experiment
from ingredients.parse_months import ingredient as parse_months_ingredient, parse_months_run

#################
## ingredients ##
#################
ingredients = [parse_months_ingredient]

ex = Experiment('parse_months',  ingredients)

@parse_months_ingredient.config
def update_cfg():
    start_at_x = 510
    cut_y = 124
    file_path = '/home/q1/Python/dl/data/uniklinik_augen_unique'

@ex.automain
def run():
    parse_months_run()

