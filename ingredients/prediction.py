from sacred import Ingredient

ingredient = Ingredient('prediction')

@ingredient.capture
def get_prediction(prediction):
    if prediction[0, 0] > prediction[0, 1]:
        return 'r ', prediction[0, 0]
    else:
        return 'nr', prediction[0, 1]

def get_missing_values_prediction(value, avg, std):
    return (value + avg) * std

@ingredient.capture
def predictions_summary(predictions, misses):
    print('#### Stats')
    count_success = len(predictions) - len(misses)
    acc = count_success / len(predictions)
    print('Accuracy: %.2f%%' % acc)
    print('Success: ', count_success)
    print('Miss: ', len(misses))
    print(sorted(misses))
    return acc