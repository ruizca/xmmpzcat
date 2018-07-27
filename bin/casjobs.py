#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 19:00:26 2018

@author: alnoah
"""

import os
import requests
import getpass
import subprocess as sp

import utils

def add_table(data_table, name):
    
    data_table.write('temp.csv', format='ascii.csv', overwrite=True)
    
    url='http://mastweb.stsci.edu/ps1casjobs/'
    payload = {'tableTypeDDL': 0, 
               'NameBox': name,
               'DataType': 0,
               'importType': 1}
    file = {'httpBox': open('temp.csv','rb')}

    user = getpass.getpass("casjobs user:")
    passwd = getpass.getpass("Password for " + user + ":")    

    s = requests.Session()
    r = s.get(url + 'login.aspx', params={'userid': user, 'password': passwd})
    r = s.post(url + 'TableImport.aspx', files=file, data=payload)
    
    os.remove('temp.csv')
    
    return r.status_code
    
def drop_table(name):

    cmd = 'java -jar casjobs.jar execute -t "mydb" '
    cmd += '-n "drop query" "drop table {}"'.format(name)    
    sp.call(cmd, shell=True)

def get_table(name, format='fits', folder='.'):
    
    cmd = 'java -jar casjobs.jar extract -table {} -force -type {} -url {}'
    cmd = cmd.format(name, format, folder)
    output = sp.check_output(cmd, shell=True)
    
    for line in output.decode().split('\n'):
        if line.startswith('http'):
            utils.downloadFile(line, folder, 
                               filename='{}.{}'.format(name, format))
    
def run_qry(qry, qry_name):
    
    cmd = 'java -jar casjobs.jar run -n {} "{}"'.format(qry_name, qry)
    sh_return = sp.call(cmd, shell=True)

    return sh_return
