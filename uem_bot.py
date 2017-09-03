#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:02:20 2017

@author: nerea
"""

import numpy as np
import pandas as pd


web = './web/'
navigation = './navigation/'
general = 'general'
grado = 'grado'
posgrado = 'posgrado'
url_formulario = 'http://canarias.universidadeuropea.es/solicitud-de-informacion'
browsers = ['Chrome', 'Firefox', 'IExplorer', 'Safari']
operatingSystems = ['Windows10', 'Windows7', 'WindowsXP', 'MacOS', 'Ubuntu', 'Fedora', 'Android']
sep = ','

def read_file(file_name):
    my_file = open(web + file_name, 'r')
    lines = my_file.read().split()
    my_file.close()
    return lines

def load_web_tree():
    web_tree = {}
    web_tree[general] = read_file('general.txt')
    web_tree[grado] = read_file('grado.txt')
    web_tree[posgrado] = read_file('postgrado.txt')
    return web_tree

class Bot:
    ''' A little bit about Bot class'''
    
    def __init__(self, web_tree, conversor=True, bias='grado'):
        self.conversor = conversor
        self.bias = bias
        self.fullVisitorId = str(int(conversor)) + str(np.random.randint(0,1e9)).zfill(9)
        self.visits = None
        self.hits = None
        self.browser = np.random.choice(browsers)
        self.operatingSystem = np.random.choice(operatingSystems)
        self.isMobile = False
        self.language = 'es-es'
        self.web_tree = web_tree
        
        # fijamos el número de sesiones y hits por sesión
        #   para ello, en primer lugar fijamos los parámetros
        if conversor:
            mu_visit = 8
            sigma_visit = 3
            mu_hit = 16
            sigma_hit = 10
        else :
            mu_visit = 2
            sigma_visit = 1
            mu_hit = 5
            sigma_hit= 2
        
        self.visits = max(1,int(np.round(np.random.normal(mu_visit, sigma_visit))))
        self.hits = np.maximum(1,(np.round(np.random.normal(mu_hit,sigma_hit, size=self.visits)).astype(int)))
        
    def next_hit(self, visitNumber, hitNumber):
        probs = np.array([0.70, 0.025 + 0.25 * self.conversor, 0.025 + 0.25 * (not self.conversor)])
        branch = np.random.choice([general, grado, posgrado], p=probs)
        pagePath = np.random.choice(self.web_tree[branch])
        return self.fullVisitorId + sep + str(visitNumber) + sep + str(self.visits) + sep + str(hitNumber) + sep + str(self.hits[visitNumber]) + \
            sep + self.browser + sep + self.operatingSystem + sep + self.language + sep + pagePath + '\n'
            
    def run(self):
        print("Starting navigation... " + self.fullVisitorId)
        user_list = open(navigation + 'user_list.txt', 'a')
        user_list.write(self.fullVisitorId + '\n')
        user_list.close()
        
        my_file = open(navigation + self.fullVisitorId + '.txt', 'a') # 'a' for appending mode
        for visitNumber in range(self.visits):
            for hitNumber in range(self.hits[visitNumber]):
                my_file.write(self.next_hit(visitNumber, hitNumber))
        my_file.close()
                
    
if __name__ == '__main__':
    ue_canarias = load_web_tree()
    for i in range(100):
        if (i % 4) == 0:
            conversor = True
            bias = grado
        elif (i % 4) == 1:
            conversor = False
            bias == grado
        elif (i % 4) == 2:
            conversor = True
            bias = posgrado
        else:
            conversor = False
            bias = posgrado
            
#        print(pibot.conversor, pibot.bias)
        pibot = Bot(ue_canarias, conversor, bias)
        pibot.run()
        
#    pibot = Bot(ue_canarias, conversor = True, bias='grado')
#    pibot.run()