# TEST VERSION WITH LIST OF 4 Training points
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
import itertools

excel_read = pd.read_excel(r"C:\Users\stoian\Desktop\Python_Projects\ker_nn_01\ker_train_01.xlsx",sheetname = 'Data')

# X_Train = excel_read.iloc[:900,0:3].copy()
# Y_Train = excel_read.iloc[:900,3].copy()
# X_Test = excel_read.iloc[900:,0:3].copy()
# Y_Test = excel_read.iloc[900:,3].copy()

X_Train = [1,2,3,4]
Y_Train = [2,4,6,8]
X_Test = [5,6]
Y_Test = [10,12]

# input_dim = X_Train.shape[1]
input_dim = 1

model = Sequential()
model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')


class BatchLogger(Callback):

    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])

    def get_values(self, metric_name, window):
        d = pd.Series(self.log_values[metric_name])
        return d.rolling(window, center=False).mean()


bl = BatchLogger()

print (np.array(X_Train))
print (np.array(Y_Train))


history = model.fit(
              np.array(X_Train), np.array(Y_Train),
              batch_size=25, epochs=1000, verbose=1, callbacks=[bl],
              validation_data=(np.array(X_Test), np.array(Y_Test)))

score = model.evaluate(np.array(X_Test), np.array(Y_Test), verbose=1)
# print('Test log loss:', score[0])
# print('Test accuracy:', score[1])


#_________________________________________ PLOT ___________________________________

def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    y_pred_labels = (y_pred > th).astype(int)

    cm = confusion_matrix(y_true, y_pred_labels)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

# ________________________________________________________ PLOT CM ___________________________



y_train_pred = model.predict_on_batch(np.array(X_Train))[:, 0]
y_test_pred = model.predict_on_batch(np.array(X_Test))[:, 0]

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(15, 5)

plot_cm(ax[0], Y_Train, y_train_pred, [0, 1], 'Confusion matrix (TRAIN)')
plot_cm(ax[1], Y_Test, y_test_pred, [0, 1], 'Confusion matrix (TEST)')

#plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)

plt.tight_layout()
plt.show()

# __________________________________________ END PLOT _________________________________________________________


res = model.predict_on_batch(np.array(X_Test))
print (res)




