import csv
import numpy as np
import os

from ingredients.prediction import get_prediction, predictions_summary

from sacred import Ingredient

ingredient = Ingredient('logging')

@ingredient.config
def cfg():
    scores_save_path = 'path_to_scores'
    extras = ['bcva','cstb','mrtb','hba1c','prp', 'lens', 'pdr', 'gender', 'avegf', 'age', 'duration']
    
@ingredient.capture
def log_average_scores(keys, scores, id = None, write_to_file = False, prediction = None):
    print('### average:')
    tmp = {}
    logs = {}
    for key in keys:
        for i, score in enumerate(scores):
            if key not in tmp:
                tmp[key]=[]
            tmp[key].append(score[key])
        mean = np.mean(tmp[key])
        std = np.std(tmp[key])
        max = np.max(tmp[key])
        min = np.min(tmp[key])

        if key not in logs:
            logs[key] = {}    
        logs[key]['val'] = mean
        logs[key]['std'] = std
        logs[key]['max'] = max
        logs[key]['min'] = min

        if 'ca' in key:
            print('avg %s:  %.2f%% (+/- %.2f) (max: %.2f%%) (min: %.2f%%)'  % (key, mean * 100, std * 100, max * 100, min * 100))
        else:
            print('avg %s:  %.2f (+/- %.2f) (max: %.2f) (min: %.2f)'  % (key, mean, std, max, min))
    
    if write_to_file:
        write_average_scores(logs, id, prediction)

@ingredient.capture
def write_average_scores(logs, id, prediction, scores_save_path, extras):
    stats_file = '%sstatistics.csv' % (scores_save_path)
    with open(stats_file, mode='a') as csv_file:
        fieldnames = ['id', 'val_ca', 'val_ca_std', 'val_ca_min', 'val_ca_max', 'val_loss', 'val_loss_std', 'val_loss_min', 'val_loss_max', 'loss', 'loss_std', 'loss_min', 'loss_max', 'ca', 'ca_std', 'ca_min', 'ca_max', 'prediction', 'extras']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if os.stat(stats_file).st_size == 0:
            writer.writeheader()
        row = {} 
        row['id'] = id
        row['prediction'] = prediction
        seperator = ';'
        row['extras'] = seperator.join(sorted(extras))
        for key in logs:
            if 'lr' in key:
                continue
            row[key] = logs[key]['val'] 
            row[key + '_std'] = logs[key]['std']
            row[key + '_min'] = logs[key]['min']
            row[key + '_max'] = logs[key]['max']
        writer.writerow(row)

@ingredient.capture
def log_average_predictions(predictions, predictions_save_path, run_id, validations = {}):
    data = []
    misses = []
    prob = {}
    fieldnames = ['id', 'responder prediction (rp)', 'rp-std', 'rp-min', 'rp-max', 'non-responder prediction (nrp)', 'nrp-std', 'nrp-min', 'nrp-max', 'correct']
    if not os.path.exists(predictions_save_path):
        os.makedirs(predictions_save_path)
    for i,id in enumerate(predictions):
        responder_predictions = []
        non_responder_predictions = []
        for p in predictions[id]:
            responder_predictions.append(p[0,0])
            non_responder_predictions.append(p[0,1])
        r_p_mean = round(np.mean(responder_predictions) * 100, 2)
        r_p_std = round(np.std(responder_predictions) * 100, 2)
        r_p_min = round(np.min(responder_predictions) * 100, 2)
        r_p_max = round(np.max(responder_predictions) * 100, 2)
        nr_p_mean = round(np.mean(non_responder_predictions) * 100, 2)
        nr_p_std = round(np.std(non_responder_predictions) * 100, 2)
        nr_p_min = round(np.min(non_responder_predictions) * 100, 2)
        nr_p_max = round(np.max(non_responder_predictions) * 100, 2)
        validation = validations[id]
        # get prediction of mean
        prob[0, 0] = r_p_mean
        prob[0, 1] = nr_p_mean
        prediction, p_score = get_prediction(prob)
        if validation == ' ':
            misses.append(id)
        print('%s => %s [%s] : [r] = %.2f%% (+/- %.2f) (min: %.2f%%) (max: %.2f%%) [n-r] = %.2f%% (+/- %.2f) (min: %.2f%%) (max: %.2f%%)'  % (id, prediction, validation, r_p_mean, r_p_std, r_p_min, r_p_max, nr_p_mean, nr_p_std, nr_p_min, nr_p_max))
        data.append({fieldnames[0]: id, fieldnames[1]: r_p_mean, fieldnames[2]: r_p_std, fieldnames[3]: r_p_min, fieldnames[4]: r_p_max, fieldnames[5]: nr_p_mean, fieldnames[6]: nr_p_std, fieldnames[7]: nr_p_min, fieldnames[8]: nr_p_max, fieldnames[9]: validation})
    # write to csv
    with open('%spredictions-%s.csv' % (predictions_save_path,run_id), mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    # stats
    prediction_accuracy = predictions_summary(predictions, misses)

    return prediction_accuracy
