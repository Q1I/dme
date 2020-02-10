# import matplotlib.pyplot as plt
from sacred import Ingredient

ingredient = Ingredient('plotting')

@ingredient.capture
def plot(predictions_ids, predictions_responder_values, predictions_non_responder_values, predictions_miss):
    # r/n-r
    plot_predictions('responder', '', predictions_ids, predictions_responder_values, predictions_miss['x'], predictions_miss['y_r'])
    plot_predictions('non-responder', '', predictions_ids, predictions_non_responder_values, predictions_miss['x'], predictions_miss['y_nr'])
    # sorted
    top_responder = ''
    top_non_responder = ''
    list1, list2 = zip(*sorted(zip(predictions_responder_values, predictions_ids), reverse=True))
    plot_predictions('sorted-responder', 'top %s' % top_responder, list2, list1, predictions_miss['x'], predictions_miss['y_r'])
    
    # for i in list2:
    #     plot_image(i, list1[i], 'n-r' if i in predictions_miss else 'r')

    list1, list2 = zip(*sorted(zip(predictions_non_responder_values, predictions_ids), reverse=True))
    plot_predictions('sorted-non-responder', 'top %s' % top_non_responder, list2, list1, predictions_miss['x'], predictions_miss['y_nr'])
    
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label, img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def plot_predictions(title, label, ids, values, misses_ids, misses_values):
    plt.xlabel('ID')
    plt.ylabel('Confidence')

    plt.title('Prediction: %s %s' % (title, label) )
    plt.plot(ids, values, 'bs')
    tmp_ids = []
    tmp_values = []
    for i, id in enumerate(misses_ids):
        if id in ids:
            tmp_ids.append(id)
            tmp_values.append(misses_values[i])
    plt.plot(tmp_ids, tmp_values, 'ro')
    plt.savefig('prediction-%s.png' % title, bbox_inches='tight')
    # plt.show()
    plt.clf()


def plot_bk(history, history_save_path, id, counter):
    # summarize history for accuracy
    plt.plot(history.history['ca'])
    plt.plot(history.history['val_ca'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('%s%s/accuracy-%i.png' % (history_save_path, id, counter))
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('%s%s/loss-%i.png' % (history_save_path, id, counter))
    plt.clf()