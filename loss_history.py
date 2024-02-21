"""
Dense6DPose (c) by Metwalli Al-Selwi
contact: Metwalli.msn@gmail.com

Dense6DPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
"""

from tensorflow.keras.callbacks import Callback
import json
import codecs
import os
class LossHistory(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.history = self.loadHist(filename)

    def on_epoch_end(self, epoch, logs=None):
        new_history = {}
        for k, v in logs.items(): # compile new history from logs
            new_history[k] = [str(v)] # convert values into lists
        self.history = self.appendHist(self.history, new_history) # append the logs
        self.saveHist(self.filename, self.history) # save history from current training

    def get_initial_epoch(self):
        return 0 if self.history == {} else len(self.history['classification_loss'])

    def get_initial_lr(self):
        return 0 if self.history == {} else self.history['lr'][-1]

    def saveHist(self, path, history):
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)

    def loadHist(self, path):
        n = {}  # set history to empty
        if os.path.exists(path):  # reload history if it exists
            with codecs.open(path, 'r', encoding='utf-8') as f:
                n = json.loads(f.read())
        return n

    def appendHist(self, h1, h2):
        if h1 == {}:
            return h2
        else:
            dest = {}
            for key, value in h1.items():
                dest[key] = value + h2[key]
            return dest