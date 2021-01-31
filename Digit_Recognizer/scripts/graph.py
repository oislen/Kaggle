import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import itertools

#from scores import average_precision, auc

def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def visualize_samples(data, n_cols=5, n_rows=1):
    """Visualize samples."""
    plt.figure(figsize = (3*n_cols,3*n_rows))
    for n,i in enumerate(np.random.randint(len(data), size = n_cols*n_rows)):
        plt.subplot(n_rows,n_cols,n+1)
        plt.axis('off')
        plt.imshow(data[i])
    plt.show()
    
def visualize_heatmap(images, heatmap, n_cols=5, n_rows=1):
    """Visualize heatmap"""
    plt.figure(figsize=(3 * n_cols, 2 * 3 * n_rows))
    for n,i in enumerate(np.arange(n_cols * n_rows)):
        plt.subplot(2 * n_rows, n_cols, n + 1)
        plt.axis('off')
        plt.imshow(images[i])
        
        plt.subplot(2 * n_rows, n_cols, n + 1 + n_cols)
        plt.axis('off')
        plt.imshow(heatmap[i])
    plt.show()

def show_bboxes(bboxes, ax, color="black", text=None):
    for i, bbox in enumerate(bboxes):
        ax.add_patch(patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], fill=False, color=color))
        if text is not None:
            ax.text(bbox[1], bbox[0], text[i], color=color)
        
def visualize_bboxes(images, pred_bboxes=None, true_bboxes=None, decision_function=None, n_cols=5, n_rows=1):
    plt.figure(figsize = (3*n_cols,3*n_rows))
    
    if pred_bboxes is not None:
        pred_bboxes = np.array(pred_bboxes, dtype=np.int32)
    if true_bboxes is not None:
        true_bboxes = np.array(true_bboxes, dtype=np.int32)

    for n,i in enumerate(np.random.choice(range(len(images)), size=n_cols * n_rows, replace=False)):
        ax = plt.subplot(n_rows,n_cols,n+1)
        plt.axis('off')
        plt.imshow(images[i])

        if pred_bboxes is not None:
            _text = (["{0:0.2f}".format(decision_function[prec]) for prec in np.where(pred_bboxes[:, 0] == i)[0]]
                     if decision_function is not None else None)
            show_bboxes(bboxes=pred_bboxes[pred_bboxes[:, 0] == i, 1:], ax=ax, color="blue", text=_text)
        
        if true_bboxes is not None:
            show_bboxes(bboxes=true_bboxes[true_bboxes[:, 0] == i, 1:], ax=ax, color="red")
    plt.show()

def plot_precision_recall(precision, recall):
    _auc = auc(y=precision, x=recall)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.plot(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.grid(color="white")
    plt.title('Precision-Recall curve: AUC-PR={0:0.2f}'.format(_auc))
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1