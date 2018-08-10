#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 09 01:10:33 2018

@author: Kumari Anjali

Comiler/Interpreter : Python 3.6.3 64bits, Qt 5.6.2, PyQt5 5.6 on Darwin 
Environment Version : Spyder 3.2.6 
Framework           : Anaconda Navigator 1.8.3
OS Platform         : MacOS High Sierra 10.13.2
May run on MS Window/Unix/Linux/Ununto as well
May not run on earlier version of Python

"""

"""Using Meyer Packard Algorithm to predict stock price of Tesla""" 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         # Import matplotlib ploting library 
# import pandas_datareader as pdr # To use when data to be downloaded from Yahoo/Google
# import datetime   # To use when data to be downloaded from Yahoo/Google
import math as mt
import random as rd
#from array import array
# import quandl   # To use when data to be downloaded from quandl.com

'''WE CAN DOWNLOAD YAHOO/GOOGLE FINANCE DATA WITH BELOW GIVEN CODE'''
#-----------------------------------------------------------
# Downloading Tesla stock data from yahoo/google.com
# start = datetime.datetime(2013, 11, 1)
# end = datetime.datetime(2017, 7, 11)
# dataset = pdr.get_data_yahoo("TSLA", start, end)
# dataset = pdr.DataReader("TSLA", 'yahoo', start, end)
#-----------------------------------------------------------

'''WE CAN DOWNLOAD STOCK DATA FROM QUANDL'''
#-----------------------------------------------------------
#To request specific data from Quandl.com:
# dataset = quandl.get("WIKI/TSLA.4", start_date="2013-11-01", end_date="2017-07-17", authtoken="CzibtKpDF8ssxxmSK3RD")
#-----------------------------------------------------------

# Code to connect offiline data in .CSV format already downloaded. Used when there is no internet
'''FOR OFFLINE SYSTEM, WE HAVE ATTACHED .CSV DATA FILE FOR TRAINING DATA'''

#**********************************************
# MODULE 01 : DATA PULLING AND BASIC STAT
#**********************************************

print("-----------------------------------------------------------------------")
print("GENETIC ALGORITHMS FOR STOCK DATA PREDICTION")
print("-----------------------------------------------------------------------")
print()
dataset = pd.read_csv('TSLA1317.csv')
print(dataset.head(5))
print(dataset.tail(5))

dataset['Daily_Move'] = dataset['Close'] - dataset['Close'].shift(1)

# Analyzing Tesla stock data for daily stock price variation
tesla = dataset['Close']
daily_move_mean = abs(dataset['Daily_Move']).mean()
print()
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
print("TESLA STOCKS DAILY MOVES : MEAN ")
print(daily_move_mean)
print()

stock_price_range = (min(tesla),max(tesla))
print("TESLA STOCKS : PRICE RANGE ")
print(stock_price_range)

# Calculating Standard deviation of all stock prices in the dataset
arr = np.array(tesla)
arr[0:10]
std = np.ndarray.std(arr)
print()
print("STANDARD DEVIATION : ", std)
print()
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
print()

plt.plot(tesla)
plt.title('Plot 1 : Stock Price movement')
plt.xlabel('Period')
plt.ylabel('Return')
plt.show()
print()

#*************************************************************
# MODULE 02 : GENES & CHROMOSOME CREATION : 
#*************************************************************

# Creating genes (individual Genes) with sets of conditions
genes = []
i = 0
while i < 1000:
    i = i + 1
    random_price = rd.uniform(min(tesla),max(tesla))
    r1 = rd.uniform(random_price - daily_move_mean, random_price + daily_move_mean)
    r2 = rd.uniform(r1 - daily_move_mean, r1 + daily_move_mean)
    r3 = rd.uniform(r2 - daily_move_mean, r2 + daily_move_mean)
    r4 = rd.uniform(r3 - daily_move_mean, r3 + daily_move_mean)
    genes.append((random_price,r1,r2,r3,r4))
print()
print("---------------------------------------")
print("INDIVIDUAL CHROMOSOME WITH 5 GENES EACH")
print("---------------------------------------")
print()
print(genes[1:10])
print()  

#*****************************************************************************
# MODULE 03 : NORMAN PACKARD CONDITION TEST FOR EACH DAY CLOSING STOCK PRICE : 
#*****************************************************************************

# FINDING TOTAL POPULATION ON THE CONDITION GIVEN BY NORMAN PACKARD
# Here we are checking whether : Gene-2 <= Closing Price <= Gene+2 
# Input = Closing Price of Tesla Array and Chromosome Tupples Created above 
# Packard Condition Test
def array_match(arr,tup):
    """DocString : iteratively scans consecutive elements of a list and checks whether the
       elements fall in the neighbourhood of the tuple elements. Returns a 
       dictionary of genes and corresponding array elements which comply with the genes.
       Form : array_match(list,tuple)
       Here arr = Closing Price of Tesla in Array format
       tup = Total Chromosome tupple with 5 genes each, created above
       """
    y_values = []
    for a in range(0,len(arr)-5):
        if tup[0] -2  <= arr[a]  <= tup[0] +2 and \
           tup[1] -2 <= arr[a+1] <= tup[1] +2 and \
           tup[2] -2 <= arr[a+2] <= tup[2] +2 and \
           tup[3] -2 <= arr[a+3] <= tup[3] +2 and \
           tup[4] -2 <= arr[a+4] <= tup[4] +2:
               y_values.append(arr[a+5])
    
    return y_values

# Store all closing prices of array fulfilling above criterias : Norman Packard Condition
final_set = {}
for con in genes:
    final_set[con] = array_match(tesla, con)
'''Large Dataset, hence deactivated'''
# print("POPULATION DICTIONARY OF CLOSING PRICE WHICH PASS PACKARD CONDITION")       
# print(final_set)
print()

#************************************************************************************************
# MODULE 04 : MEYER & PACKARAD FITNESS FOR STOCK PRICE POPULATION WHICH PASSED PACKARD CONDITION: 
#************************************************************************************************
                
def fitness(series, std_pop, alpha):
    """Meyer and Packard fitness test. Returns the value of fitness function
       Form : fitness(list,float,float)"""
    sigma = np.std(series)
    if len(series) <= 1:
        return float('nan')
    else:
        # Meyer Packard Test
        fitness = -1* (mt.log2(sigma/std_pop) + alpha/len(series))
        return round(fitness,2)

# Return fitness value for each set of values complying with respective genes

fitness_dict = {}
for it in final_set:
    if -1000 < fitness(final_set[it], std, 10) < 1000:
        fitness_dict[it] = fitness(final_set[it], std, 10)
print()
print("------------------------------------------------------------------------")
print("FITNESS DICTIONAY CONTAINING CHROMOSOME WHICH PASSED MEYER PACKARD TEST ")
print("------------------------------------------------------------------------")
print()
print(fitness_dict)  # DUE TO LARGE DATASET, WE HAVE INACTIVATED THIS OUTPUT
print()
print()

# RANKING OF CHROMOSOMES, WHICH PASSED MEYER PACKARD TEST in above module
print("--------------------------------------------------------")
print("RANKING OF CHROMOSOMES, WHICH PASSED MEYER PACKARD TEST ")
print("--------------------------------------------------------")
print()
fitness_dict_flip = {v: k for k, v in fitness_dict.items()}
fitness_scores = list(fitness_dict_flip.keys())
fitness_scores.sort(reverse = True)
for a in fitness_scores:
    print(a,fitness_dict_flip[a])
print()
    
# WE DISCARD ALL CHROMOSOME WITH LOW RANKS IN MEYER PACKARD TEST
# REMOVE ALL CHROMOSOME BELOW RANK = 1 AND CREATING A NEW SET OF DICTIONARY WITH RANK ABOVE 1
fitness_dict_flip_prune = {}
for b in fitness_scores:
    if b >= 1:
        fitness_dict_flip_prune[b] = fitness_dict_flip[b]
        
print()
print("-----------------------------------")
print("FIT CHROMOSOMES WITH RANK ABOVE 1 ")
print("-----------------------------------")
print()
print(fitness_dict_flip_prune)
print()
# print(type(fitness_dict_flip_prune))


# print the y values of remaining individuals
print("-----------------------------------------------")
print("REMAINING FIT POPULATION WITH ACCEPTABLE SCORE ")
print("-----------------------------------------------")
print()
for m in fitness_dict_flip_prune:
        print(m, final_set[fitness_dict_flip_prune[m]])

print()
print()


#*************************************************************************************************
# MODULE 04 : UNFORM CROSS-OVER OF POPULATION WHICH RANKED ABOVE 1 IN MEYER PACKARAD FITNESS TEST: 
#*************************************************************************************************

def uniform_crossover(tup1,tup2):
    """Improving results of Meyer Packard Algorithm"""
    a1,b1,c1,d1,e1 = tup1
    a2,b2,c2,d2,e2 = tup2
    tup3 = (a1,b1,c1,d2,e2)
    tup4 = (a2,b2,c2,d1,e1)
    tup5 = (a1,b1,c2,d2,e2)
    tup6 = (a2,b2,c1,d1,e1)
    return [tup3,tup4,tup5,tup6]  

# Creating offsprings using uniform crossover and checking their fitness score
# fitness_dict_flip_prune.keys()
# fitness_dict_flip_prune.values()
# fitness_dict_flip_prune.items()

# SKINNING FITTNESS RANK FROM DATA DICTIONARY : fitness_dict_flip_prune AND ARRANGING IN DESCENDING ORDER IN A LIST
a = sorted(fitness_dict_flip_prune, reverse=True)
v1 = a[0]  # Value of Highest Fitness Rank assigned 
v2 = a[1]  # Value of Second Highest Fitness Rank assigned 

# CROSSOVER OF HIGHEST AND SECOND HIGHEST RANKED CHROMOSOME AND RE-ITERATION
offsprings = uniform_crossover(fitness_dict_flip_prune[v1], fitness_dict_flip_prune[v2])
offsprings = offsprings + uniform_crossover(fitness_dict_flip_prune[v1], fitness_dict_flip_prune[v2])
print()
print("-----------------------------------------------------------------------------------")    
print("OFFSPRINGS GENERATED AFTER UNIFORM CROSSOVER WITH CHROMOSOME OF HIGHER FITNESS RANK")
print("-----------------------------------------------------------------------------------")
print()
print(offsprings)
print()
print("----------------------------------------------------------------------")
print("Total Offsprins generated by CrossOver of higher ranked Chromosomes : ")
print("----------------------------------------------------------------------")
print()
print(len(offsprings))
print()

#************************************************************************************************
# MODULE 05 : MUTATION & IMPROVEMENT OF OFFSPRINGS
#************************************************************************************************

# OFFSPRING GENES WHICH PASS NORMAN PACKARD CONDITION 
improved_set = {}
for con in offsprings:
    improved_set[con] = array_match(tesla, con)
print()
print("-----------------------------------------------------")
print("Offsprings which pass condition set by Norman Packard")
print("-----------------------------------------------------")
print()
print(improved_set)
print()
print("Total Number of Offsprings, which passed Norman Packard condition: ", len(improved_set))
print()

# MUTATION  PROCESS OF OFFSPRINGS, WHICH PASSED NORMAL PACKARD CONDITION

fitness_dict_improved = {}
for it in improved_set:
    if -1000 < fitness(improved_set[it], std, 10) < 1000:
        fitness_dict_improved[it] = fitness(improved_set[it],std,10)
print()
print("------------------------------------------------------")
print("OFFSPRINGS WITH IMPROVED FITNESS SCORE AFTER MUTATION ") 
print("------------------------------------------------------")
print()       
print(fitness_dict_improved)
print()
print()
print("Total Number of Offsprings undergone Mutation: ", len(fitness_dict_improved))
print()
print("------------------------------------------------------------------------")
print("------------------------------------------------------------------------")
print("------------------------------ THE END -------------------------------- ")
print()
#************************************************************************************************
# END 
#************************************************************************************************
