#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np
import keras

from keras import backend as K
import tensorflow as tf

from tensorflow.keras.metrics import AUC


def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auc_pr(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred, curve='PR', name="AUC_PR", summation_method="careful_interpolation")[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def auc_roc(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred, curve='ROC', name="AUC_ROC")[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# one output
def auc_skl(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)
