#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:02:20 2017

@author: nerea
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense

from multivariate_gaussian import MultivGauss

web = './web/'
navigation = './navigation/'
user_list = navigation + 'user_list.txt'

general = 'general'
grado = 'grado'
posgrado = 'posgrado'
url_formulario = 'http://canarias.universidadeuropea.es/solicitud-de-informacion'
browsers = ['Chrome', 'Firefox', 'IExplorer', 'Safari']
operatingSystems = ['Windows10', 'Windows7', 'WindowsXP', 'MacOS', 'Ubuntu', 'Fedora', 'Android']
sep = ','
extension = '.txt'

def my_sigmoid(x):
    return 1/(1 + np.exp(-(x-4)/2)) # dibujar y ver ;)
    
def read_file(file_name):
    my_file = open(file_name, 'r')
    lines = my_file.read().split()
    my_file.close()
    return lines

def load_web_tree():
    web_tree = {}
    web_tree[general] = read_file(web + 'general.txt')
    web_tree[grado] = read_file(web + 'grado.txt')
    web_tree[posgrado] = read_file(web + 'postgrado.txt')
    return web_tree

def get_users():
    return read_file(user_list)

def get_navigation(user):
    navigation_file = open(navigation + user + extension)
    hit_list = navigation_file.read().split()
    navigation_file.close()
    return hit_list

def hit_value(hit):
    ue_tree = load_web_tree()
    hitss = hit.split(',')
    return hitss[-1] in ue_tree[general], hitss[-1] in ue_tree[grado], hitss[-1] in ue_tree[posgrado]

def raw_navigation_value(user):
    navigation = get_navigation(user)
    navigation_value = np.random.uniform(0,0.2, size=3)
    for hit in navigation:
        navigation_value = navigation_value + hit_value(hit)
    return navigation_value

def navigation_affinity(user):
    ue, grado, posgrado = raw_navigation_value(user)
    return my_sigmoid(np.array([ue/20.0, grado/4.0, posgrado/4.0])) # que nos queden más o menos entre 0 y 10

def train_set():
    users = get_users()
    X = np.empty((len(users),3))
    Y = np.empty(len(users))
    i = 0
    for user in users:
        X[i,:] = navigation_affinity(user)
        Y[i] = int(user[0])
        i = i+1
    return X,Y

def split_set(x, y):
    '''
    y - boolean labels
    '''
    set_a = x[y.astype(bool)].copy()
    set_b = x[(1 - y).astype(bool)].copy()
    return set_a, set_b

if __name__ == '__main__':
    X, Y = train_set()
    convs, no_convs = split_set(X, Y)
    
    p_convs = MultivGauss(convs)
    p_convs.run()
    p_no_convs = MultivGauss(no_convs)
    p_no_convs.run()
    

    
    X, Y = train_set()
    

    test = []
    for point in X:
        test.append(p_convs.value(point) > p_no_convs.value(point))
    Y_ = np.array(test)
    print( 'Accuracy : ' + str((np.sum(((Y - Y_) == 0).astype(int)) / X.shape[0])*100) + ' over ' + str(X.shape[0]) + ' navigation cases' )