# -*- coding: utf-8 -*-
import csv
import logging
import os
import pickle
import random
import time
from collections import Counter
#- add new modules
from collections import OrderedDict
from pathlib import Path
from random import Random

import numpy as np
import torch
from argParser import args
from fllibs import *
from torch.utils.data import DataLoader

import math  

#set up the data generator to have consistent results
seed = 10                             
generator = torch.Generator()
generator.manual_seed(seed)                                          

def seed_worker(worker_id):
    worker_seed = seed #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)                                     
    random.seed(worker_seed)                                      

class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]                               
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""                        

    def __init__(self, data, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)                                        

        self.data = data
        self.labels = self.data.targets

        self.args = args
        self.isTest = isTest
        np.random.seed(seed)                                         

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass                         

        #- set the number of samples per worker
        self.usedSamples = 0                                       

        #- introduce targets dict
        self.targets = OrderedDict()                            
        self.indexToLabel = {}

        # categarize the samples                                  
        # last_label = None
        # count = 0

#        for index, label in enumerate(self.labels):                 
#            if label not in self.targets:
#                self.targets[label] = []                         

#            self.targets[label].append(index)                 
#            self.indexToLabel[index] = label                        


        for index, label in enumerate(self.labels):

            if self.args.data_set == "Mnist":
                label = label.item()
            if self.args.data_set == "fashion_mnist":
                label = label.item()                

            if self.args.data_set == "emnist":
                label = label.item()   

            if label not in self.targets:
                self.targets[label] = []                      

            self.targets[label].append(index)                             
            self.indexToLabel[index] = label

#            logging.info(f"index, label is {index, label}!")            
#            logging.info(f"type of label is {type(label)}!")            
#        logging.info(f"len of self.labels is {len(self.labels)}!")         
#        logging.info(f"len of self.targets is {len(self.targets)}!")      
#        logging.info(f"len of self.targets[1] is {len(self.targets[1])}!") 

    def getNumOfLabels(self):                                           
        return self.numOfLabels                                
  
    def getDataLen(self):
        return self.data_len                                 






    def trace_partition(self, data_map_file, ratio=1.0):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")          

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:                                      
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0         

            for row in csv_reader:                                           
                if read_first:                                                
                    logging.info(f'Trace names are {", ".join(row)}')             
                    read_first = False
                else:
                    client_id = row[0]                                          

                    if client_id not in unique_clientIds:                          
                        unique_clientIds[client_id] = len(unique_clientIds)         
                    
                                                                                      
                    clientId_maps[sample_id] = unique_clientIds[client_id]          
    
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]                 

        for idx in range(len(self.data.data)):                                        
            self.partitions[clientId_maps[idx]].append(idx)                           

        for i in range(len(unique_clientIds)):
            self.rng.shuffle(self.partitions[i])                                     
            takelen = max(0, int(len(self.partitions[i]) * ratio))                   
            self.partitions[i] = self.partitions[i][:takelen]







    #- add data mapping handlers (uniform, zipf, balanced) and class exclusion 
    def partition_data_helper(self, num_clients, data_map_dir=None):
        tasktype = 'train' if not self.isTest else 'test'                          
        data_map_file = None
        if data_map_dir is not None:                                              
            data_map_file = os.path.join(data_map_dir, tasktype + '.csv')          
            logging.info(f"data_map_file is {os.path.exists(data_map_file)}!")
            #- handle the case for reddit dataset where on IBEX mappings are stored on the metadata folder
            if args.data_set == 'reddit' or args.data_set == 'stackoverflow':    
                data_map_dir = os.path.join(args.log_path, 'metadata', args.data_set, tasktype)
                data_map_file = os.path.join(data_map_dir,  'result_' + str(args.process_files_ratio) + '.csv')

        #- apply ratio on the data - manipulate the data per uses
        ratio = 1.0
        if not self.isTest and self.args.train_ratio < 1.0:                 
            ratio = self.args.train_ratio
        elif self.isTest and self.args.test_ratio < 1.0:
            ratio = self.args.test_ratio

        #- introduce the mapping based on other methods rather than read mapping file to partition trace  
        if self.isTest:                                                   
            if self.args.partitioning < 0 or data_map_file is None or num_clients < args.total_worker:      
                
                logging.info(f"Start test uniform_partition!")
                
                self.uniform_partition(num_clients=num_clients, ratio=ratio) 
                

                
                '''

                logging.info(f"custom_partition_test start!")

                if args.data_set == 'cifar10':
                    num_clients_test = math.floor(self.args.total_clients/5) if self.args.total_clients > 0 else math.floor(self.args.total_worker/5)    
                    self.custom_partition_test(num_clients=num_clients_test, ratio=ratio)
                    
                    
                    ini_list = [ii for ii in range(num_clients_test)]
                    random_some_element1 = np.random.choice(ini_list, math.floor(num_clients_test/4))
                    random_some_element2 = np.random.choice(ini_list, math.floor(num_clients_test/4))
                    random_some_element3 = np.random.choice(ini_list, math.floor(num_clients_test/4))                
                    random_some_element4 = np.random.choice(ini_list, math.floor(num_clients_test/4))

                
                elif args.data_set == 'google_speech':
                    num_clients_test = 18
                    self.custom_partition_test(num_clients=num_clients_test, ratio=ratio)
                    
                    ini_list = [ii for ii in range(num_clients_test)]
                    random_some_element1 = np.random.choice(ini_list, math.ceil(num_clients_test/4))
                    random_some_element2 = np.random.choice(ini_list, math.ceil(num_clients_test/4))
                    random_some_element3 = np.random.choice(ini_list, math.ceil(num_clients_test/4))                
                    random_some_element4 = np.random.choice(ini_list, math.ceil(num_clients_test/4))
                
                
                partition_test1 = []
                partition_test2 = []
                partition_test3 = []
                partition_test4 = []


                logging.info(f'length of self.partitions is {len(self.partitions)}!!!')                          
                logging.info(f'length of self.partitions[0] is {len(self.partitions[0])}!!!')                  
                logging.info(f'length of self.partitions[1] is {len(self.partitions[1])}!!!')                 
                logging.info(f'length of self.partitions[2] is {len(self.partitions[2])}!!!')                  
                logging.info(f'length of data is {self.getDataLen()}!!!')                                

                for j1 in random_some_element1:
                    partition_test1 = partition_test1 + self.partitions[j1]
                for j2 in random_some_element2:
                    partition_test2 = partition_test2 + self.partitions[j2]
                for j3 in random_some_element3:
                    partition_test3 = partition_test3 + self.partitions[j3]
                for j4 in random_some_element4:
                    partition_test4 = partition_test4 + self.partitions[j4]

                
                self.partitions = [partition_test1, partition_test2, partition_test3, partition_test4]

                logging.info(f'length of partition_test1 is {len(partition_test1)}!!!')                  
                logging.info(f'length of partition_test2 is {len(partition_test2)}!!!')                  
                logging.info(f'length of partition_test3 is {len(partition_test3)}!!!')                   
                logging.info(f'length of partition_test4 is {len(partition_test4)}!!!')                   

                logging.info(f"custom_partition_test, number of clients_test is {num_clients_test}! Number of partition_test is {len(self.partitions)}!")
                '''

                '''
                
                logging.info(f"custom_partition_test start!")
#                filename = 'helper_test_mapping_part' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
#                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)  
                
#                filename = 'helper_test_mapping_part_6_12_Test174_' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
#                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)   

#                filename = 'helper_test_mapping_part_6_13_Test182_' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
#                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)   

                
#                filename = 'helper_test_mapping_part_6_17_Test190_' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
#                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)  

                
#                filename = 'helper_test_mapping_part_6_19_C7_' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
#                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)   

                
#                filename = 'helper_test_mapping_part_6_19_C10_' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
#                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)   

  
                filename = 'helper_test_mapping_part_6_28_M9_' + str(self.args.partitioning) + '_data' + str(self.data_len) + '_labels'\
                    + str(self.getNumOfLabels()) + '_samples' + str(self.usedSamples)   


                folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')

                if not os.path.isdir(folder):                                     
                    Path(folder).mkdir(parents=True, exist_ok=True)

                custom_mapping_file = os.path.join(folder, filename)                
                if args.this_rank != 1:                                             
                    while (not os.path.exists(custom_mapping_file)):                
                        time.sleep(120)
                if os.path.exists(custom_mapping_file):                             
                    with open(custom_mapping_file, 'rb') as fin:                    
                        
                        logging.info(f'Loading test mapping from file {filename}')
                        self.partitions = pickle.load(fin)                           
                        

                    return  
                
                
                    
                if args.data_set == 'cifar10':
                    num_clients_test = math.floor(self.args.total_clients/5) if self.args.total_clients > 0 else math.floor(self.args.total_worker/5)    
                    self.custom_partition_test(num_clients=num_clients_test, ratio=ratio)

                    ini_list = [ii for ii in range(num_clients_test)]
                    random_some_element1 = np.random.choice(ini_list, math.floor(num_clients_test/4))
                    random_some_element2 = np.random.choice(ini_list, math.floor(num_clients_test/4))
                    random_some_element3 = np.random.choice(ini_list, math.floor(num_clients_test/4))                
                    random_some_element4 = np.random.choice(ini_list, math.floor(num_clients_test/4))

                
                elif args.data_set == 'google_speech':
                    num_clients_test = 18
                    self.custom_partition_test(num_clients=num_clients_test, ratio=ratio)

                    ini_list = [ii for ii in range(num_clients_test)]
                    random_some_element1 = np.random.choice(ini_list, math.ceil(num_clients_test/4))
                    random_some_element2 = np.random.choice(ini_list, math.ceil(num_clients_test/4))
                    random_some_element3 = np.random.choice(ini_list, math.ceil(num_clients_test/4))                
                    random_some_element4 = np.random.choice(ini_list, math.ceil(num_clients_test/4))

                if args.data_set == 'Mnist':      
                    num_clients_test = math.floor(self.args.total_clients/5) if self.args.total_clients > 0 else math.floor(self.args.total_worker/5)    #借用了executor.py中的方法
                    self.custom_partition_test(num_clients=num_clients_test, ratio=ratio)

                    ini_list = [ii for ii in range(num_clients_test)]
                    random_some_element1 = np.random.choice(ini_list, math.floor(num_clients_test/4))
                    random_some_element2 = np.random.choice(ini_list, math.floor(num_clients_test/4))
                    random_some_element3 = np.random.choice(ini_list, math.floor(num_clients_test/4))                
                    random_some_element4 = np.random.choice(ini_list, math.floor(num_clients_test/4))

                
                partition_test1 = []
                partition_test2 = []
                partition_test3 = []
                partition_test4 = []


                logging.info(f'length of self.partitions is {len(self.partitions)}!!!')                           
                logging.info(f'length of self.partitions[0] is {len(self.partitions[0])}!!!')                 
                logging.info(f'length of self.partitions[1] is {len(self.partitions[1])}!!!')                  
                logging.info(f'length of self.partitions[2] is {len(self.partitions[2])}!!!')                
                logging.info(f'length of data is {self.getDataLen()}!!!')                                    

                for j1 in random_some_element1:
                    partition_test1 = partition_test1 + self.partitions[j1]
                for j2 in random_some_element2:
                    partition_test2 = partition_test2 + self.partitions[j2]
                for j3 in random_some_element3:
                    partition_test3 = partition_test3 + self.partitions[j3]
                for j4 in random_some_element4:
                    partition_test4 = partition_test4 + self.partitions[j4]

                
                self.partitions = [partition_test1, partition_test2, partition_test3, partition_test4]

                logging.info(f'length of partition_test1 is {len(partition_test1)}!!!')                  
                logging.info(f'length of partition_test2 is {len(partition_test2)}!!!')                     
                logging.info(f'length of partition_test3 is {len(partition_test3)}!!!')                
                logging.info(f'length of partition_test4 is {len(partition_test4)}!!!')                   

                logging.info(f"custom_partition_test, number of clients_test is {num_clients_test}! Number of partition_test is {len(self.partitions)}!")
                



                #save the partitions as pickle file                           
                if not os.path.exists(custom_mapping_file):                                       
                    with open(custom_mapping_file, 'wb') as fout:
                        pickle.dump(self.partitions, fout)                                              
                    logging.info(f'Storing test_partitioning to file {filename}')       
                '''



            else:          
                self.trace_partition(data_map_file, ratio=ratio)
        elif self.args.partitioning <= 0:                                    
            if self.args.partitioning < 0 or data_map_file is None:          
                self.uniform_partition(num_clients=num_clients, ratio=ratio)
            else:                                                                                         
                self.trace_partition(data_map_file, ratio=ratio)            
        else:                                                             
            self.custom_partition(num_clients=num_clients, ratio=ratio)







    def uniform_partition(self, num_clients, ratio=1.0):                
        # random uniform partition                                    
        numOfLabels = self.getNumOfLabels()                              
        #- update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio)) 
        logging.info(f"Uniform partitioning data, ratio: {ratio} applied for {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")


        logging.info(f"Uniform partitioning, number of clients is {num_clients}!")


        indexes = list(range(data_len))
        self.rng.shuffle(indexes)                                     

        for _ in range(num_clients):
            part_len = int(1. / num_clients * data_len)                                   
            self.partitions.append(indexes[0:part_len])                 
            indexes = indexes[part_len:]





    def custom_partition(self, num_clients, ratio=1.0):                  
        logging.info(f"Start custom_partition!")
        # custom partition                                                
        numOfLabels = self.getNumOfLabels()                              
        #- update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))  
        sizes = [1.0 / num_clients for _ in range(num_clients)]          

        #get # of samples per worker
        #- set the number of samples per worker                  
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)  
        # get number of samples per worker
        if self.usedSamples <= 0:                                     
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                      
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                      

        # filename = 'training_mapping_part_6_10_Test157_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #            + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'training_mapping_part_6_11_Test168_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples) 


#        filename = 'training_mapping_part_6_18_C1_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples) 


#        filename = 'training_mapping_part_6_18_C4_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  


#        filename = 'training_mapping_part_6_23_S1_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  


#        filename = 'training_mapping_part_6_25_M1_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'training_mapping_part_6_27_M4_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  

#        logging.info(f"After file!") 


#        filename = 'training_mapping_part_6_30_t2_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  

#        filename = 'training_mapping_part_7_3_t15_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'training_mapping_part_7_4_t19_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'training_mapping_part_7_5_t22_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)    

#        filename = 'training_mapping_part_7_5_t24_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'training_mapping_part_7_12_f1_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  

        filename = 'training_mapping_part_7_14_e1_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples)  

#        filename = 'training_mapping_part_7_15_e3_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  

#        filename = 'training_mapping_part_7_15_e5_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  

        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')  
        if not os.path.isdir(folder):                                      
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)               
        if args.this_rank != 1:                                           
            while (not os.path.exists(custom_mapping_file)):            
                time.sleep(120)
        if os.path.exists(custom_mapping_file):                        
            with open(custom_mapping_file, 'rb') as fin:                    
            
                logging.info(f'Loading partitioning from file {filename}')
                self.partitions = pickle.load(fin)                           

                for i, part in enumerate(self.partitions):                               
                    labels = [self.indexToLabel[index] for index in part]    
                    #count_elems = Counter(labels)
                    #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')
            return  

        #get targets
        targets = self.getTargets()                                       
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}    

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                           
            keyLength[keyDir[key]] = len(targets[key])                     
            
        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")
                                                                            
        ratioOfClassWorker = self.create_mapping(sizes)                      

#        logging.info(f"After create_mapping!")                              
        if ratioOfClassWorker is None:                                        
            return self.uniform_partition(num_clients=num_clients)            

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)            
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

#        logging.info(f"len of keyDir is {len(keyDir)}!")                   
#        ii = 0
#        for key in keyDir.keys():                                         
#            logging.info(f"{ii} keyDir is {key}!")
#            ii = ii+1
#            if ii == 20:
#                break             


        # split the classes
        for worker in range(len(sizes)):                         
#            logging.info(f"in the for loop!")            
            self.partitions.append([])                                  
            # enumerate the ratio of classes it should take
#            logging.info(f"len of targets is {len(list(targets.keys()))}!")   
            for c in list(targets.keys()):                                   
                takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])   
                takeLength = min(takeLength, keyLength[keyDir[c]])            
#                logging.info(f"before index!") 
                indexes = self.rng.sample(targets[c], takeLength)             
                self.partitions[-1] += indexes                                
                
#                logging.info(f"after partition!") 
                labels = [self.indexToLabel[index] for index in self.partitions[-1]]  
                count_elems = Counter(labels)                                                               
                tempClassPerWorker[worker][keyDir[c]] += takeLength          
#            logging.info(f"after second for loop!")                         
            #logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')

        del tempClassPerWorker                                                
#        logging.info(f"Before create custom_mapping_file!")                
        #save the partitions as pickle file                                  
        if not os.path.exists(custom_mapping_file):                                         
            with open(custom_mapping_file, 'wb') as fout:
                 pickle.dump(self.partitions, fout)                                            
            logging.info(f'Storing partitioning to file {filename}')         


        





    def custom_partition_test(self, num_clients, ratio=1.0):              
        # custom partition                                             
        numOfLabels = self.getNumOfLabels()                          
        #- update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))  
        sizes = [1.0 / num_clients for _ in range(num_clients)]           

        #get # of samples per worker
        #- set the number of samples per worker                     
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1) 
        # get number of samples per worker
        if self.usedSamples <= 0:                                        
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                  
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                                 

#        filename = 'testing_mapping_part' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)

#        filename = 'testing_mapping_part_6_12_Test174_' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   


#        filename = 'testing_mapping_part_6_13_Test182_' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)  


#        filename = 'testing_mapping_part_6_17_Test190_' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'testing_mapping_part_6_19_C7_' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples)   

#        filename = 'testing_mapping_part_6_19_C10_' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                   + str(num_class) + '_samples' + str(self.usedSamples) 


        filename = 'testing_mapping_part_6_28_M9_' + str(self.args.partitioning) + '_clients_test' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                   + str(num_class) + '_samples' + str(self.usedSamples)                    

        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')
 
        if not os.path.isdir(folder):                                    
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)               
        if args.this_rank != 1:                                            
            while (not os.path.exists(custom_mapping_file)):                
                time.sleep(120)
        if os.path.exists(custom_mapping_file):                             
            with open(custom_mapping_file, 'rb') as fin:                    

                logging.info(f'Loading partitioning from file {filename}')
                self.partitions = pickle.load(fin)                           

                for i, part in enumerate(self.partitions):                             
                    labels = [self.indexToLabel[index] for index in part]    
                    #count_elems = Counter(labels)
                    #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')
            return  

        #get targets
        targets = self.getTargets()                                          
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}     

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                           
            keyLength[keyDir[key]] = len(targets[key])                       
               
        logging.info(f"Custom partitioning test {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients_test, use {self.usedSamples} sample per client_test ...")
                                                                             
        ratioOfClassWorker = self.create_mapping(sizes)                      


        if ratioOfClassWorker is None:                                        
            return self.uniform_partition(num_clients=num_clients)           

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)              
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

        # split the classes
        for worker in range(len(sizes)):                                     
            self.partitions.append([])                             
            # enumerate the ratio of classes it should take
            for c in list(targets.keys()):                                
                takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])  
                takeLength = min(takeLength, keyLength[keyDir[c]])            

                indexes = self.rng.sample(targets[c], takeLength)             
                self.partitions[-1] += indexes                            


                labels = [self.indexToLabel[index] for index in self.partitions[-1]] 
                count_elems = Counter(labels)                                 
                tempClassPerWorker[worker][keyDir[c]] += takeLength           

            #logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')

        del tempClassPerWorker                                               

        #save the partitions as pickle file                                   
        if not os.path.exists(custom_mapping_file):                                     
            with open(custom_mapping_file, 'wb') as fout:
                 pickle.dump(self.partitions, fout)                             
            logging.info(f'Storing partitioning to file {filename}')         














    '''
    def create_mapping(self, sizes):                                  
        numOfLabels = self.getNumOfLabels()                       

        ratioOfClassWorker = None                                   
        if self.args.partitioning == 1:                                #1 NONIID-Uniform
            ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)    

        elif self.args.partitioning == 2:                              #2 NONIID-Zipfian   
            ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)  

        elif self.args.partitioning == 3:                              #3 NONIID-Balanced
            ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)
        elif self.args.partitioning == 6:                              
            dirichlet_list = () 
            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen     
            ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        


        num_remove_class=0
        if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
            num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))  
            for w in range(len(sizes)):                           
                # randomly filter classes by forcing zero samples
                wrandom = self.rng.sample(range(numOfLabels), num_remove_class)   
                for wr in wrandom:
                    ratioOfClassWorker[w][wr] = 0.0 #0.001       

        #logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker), repr(ratioOfClassWorker)))
        logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ==== \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))    #np.count_nonzero是统计所有非零元素并返回它们的计数
        return ratioOfClassWorker
    '''






    def create_mapping(self, sizes):                            
        numOfLabels = self.getNumOfLabels()             

        ratioOfClassWorker = None                                    
        if self.args.partitioning == 1:                                #1 NONIID-Uniform
            ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)    

        elif self.args.partitioning == 2:                              #2 NONIID-Zipfian  
            ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)  

        elif self.args.partitioning == 3:                        
            ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)     
        elif self.args.partitioning == 6:                              
            logging.info('Method of create_mapping is 6')
            dirichlet_list = [] 
            up_bound1 = 2                                       
            low_bound1 = 50
            up_bound2 = 100
                                                   
            ratio1 = 0.38                                        
            num1 =  math.floor(len(sizes)*ratio1)                
            num2 = math.ceil(len(sizes)*(1-ratio1))           

#            step1 = 2*(up_bound1-self.args.dirichlet_alpha)/len(sizes)          
#            step2 = 2*(up_bound2-low_bound1)/len(sizes)                    

            step1 = (up_bound1-self.args.dirichlet_alpha)/num1    
            step2 = 2*(up_bound2-low_bound1)/num2                    

            logging.info(f"up_bound1 is {up_bound1}!")            


            ratioOfClassWorker = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
#            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen       
            for j in range(len(sizes)):
#                logging.info("==== length of sizes is:{}\n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))    #np.count_nonzero是统计所有非零元素并返回它们的计数
                if j < (len(sizes)/2):
 #               dirichlet_list.append([self.args.dirichlet_alpha+j*step,]*numOfLabelslen)            
                    ratioOfClassWorker[j] = np.random.dirichlet([self.args.dirichlet_alpha+j*step1,]*numOfLabels).astype(np.float32)
                else:
                    ratioOfClassWorker[j] = np.random.dirichlet([low_bound1+j*step2,]*numOfLabels).astype(np.float32)

#            dirichlet_list.append = []            
#            ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        


        elif self.args.partitioning == 7:                         
            logging.info('Method of create_mapping is 7')

            dirichlet_list = [] 
#            up_bound1 = 10                                        
#            low_bound1 = 50
#            up_bound2 = 100
#            step1 = 2*(up_bound1-self.args.dirichlet_alpha)/len(sizes)
#            step2 = 2*(up_bound2-low_bound1)/len(sizes)

            up_bound1 = 2                                    
            low_bound1 = 50
            up_bound2 = 100
                                                   
            ratio1 = 0.38                                   
            num1 =  math.floor(len(sizes)*ratio1)               
            num2 = math.ceil(len(sizes)*(1-ratio1))            

#            step1 = 2*(up_bound1-self.args.dirichlet_alpha)/len(sizes)       
#            step2 = 2*(up_bound2-low_bound1)/len(sizes)                      

            step1 = (up_bound1-self.args.dirichlet_alpha)/num1       
            step2 = 2*(up_bound2-low_bound1)/num2                 


            ratioOfClassWorker = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
            ratioOfClassWorker1 = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
#            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen     
            for j in range(len(sizes)):
#                logging.info("==== length of sizes is:{}\n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))    #np.count_nonzero是统计所有非零元素并返回它们的计数
                if j < (len(sizes)/2):
 #               dirichlet_list.append([self.args.dirichlet_alpha+j*step,]*numOfLabelslen)            
                    ratioOfClassWorker1[j] = np.random.dirichlet([self.args.dirichlet_alpha+j*step1,]*numOfLabels).astype(np.float32)
                else:
                    ratioOfClassWorker1[j] = np.random.dirichlet([low_bound1+j*step2,]*numOfLabels).astype(np.float32)
            
            for j in range(len(sizes)):
                ratioOfClassWorker[len(sizes)-1-j] = ratioOfClassWorker1[j]

  #-------------------------------------------------------------------------------------------------------------------------------------------------
        limitable = np.zeros([len(sizes), numOfLabels]).astype(np.float32)      
#        filename1 = 'Matrix_training_mapping_part_6_30_t2_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

#        filename1 = 'Matrix_training_mapping_part_7_3_t15_6_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

#        filename1 = 'Matrix_training_mapping_part_7_5_t22_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

#        filename1 = 'Matrix_training_mapping_part_7_5_t24_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

#        filename1 = 'Matrix_training_mapping_part_7_12_f1_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

        filename1 = 'Matrix_training_mapping_part_7_14_e1_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
                    + str(numOfLabels) 

#        filename1 = 'Matrix_training_mapping_part_7_15_e3_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

#        filename1 = 'Matrix_training_mapping_part_7_15_e5_' + str(self.args.partitioning) + '_clients' + str(len(sizes)) + '_data' \
#                   + str(numOfLabels) 

        folder1 = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')
        if not os.path.isdir(folder1):                                   
            Path(folder1).mkdir(parents=True, exist_ok=True)

        matrix_file = os.path.join(folder1, filename1)              
#        if args.this_rank != 1:                                     
#            while (not os.path.exists(matrix_file)): 
#                time.sleep(120)
        if os.path.exists(matrix_file):                       
            with open(matrix_file, 'rb') as fin1:                    
                logging.info(f'Loading partitioning from file {filename1}')
                limitable = pickle.load(fin1)                           

            num_remove_class=0
#            if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
#            num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))  #上面有的操作
            for w in range(len(sizes)):                               
                # randomly filter classes by forcing zero samples
                for r in range(numOfLabels):
                    if limitable[w][r] < 0:
                        ratioOfClassWorker[w][r] = 0.0 #0.001          
        else:
            num_remove_class=0
            if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
                num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))  #上面有的操作
                for w in range(len(sizes)):                           
                # randomly filter classes by forcing zero samples
                    wrandom = self.rng.sample(range(numOfLabels), num_remove_class)   
                    for r in wrandom:
                        ratioOfClassWorker[w][r] = 0.0 #0.001        
                        limitable[w][r] = -1.0 #0.001          

            with open(matrix_file, 'wb') as fout2:
                 pickle.dump(limitable, fout2)                                       
            logging.info(f'Storing partitioning to file {filename1}')  






 #       num_remove_class=0
 #       if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
 #           num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))  #上面有的操作
 #           for w in range(len(sizes)):                              
 #               # randomly filter classes by forcing zero samples
 #               wrandom = self.rng.sample(range(numOfLabels), num_remove_class) 
 #               for wr in wrandom:
 #                   ratioOfClassWorker[w][wr] = 0.0 #0.001         



        #logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker), repr(ratioOfClassWorker)))
        logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ==== \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))    #np.count_nonzero是统计所有非零元素并返回它们的计数
        return ratioOfClassWorker







    def getTargets(self):                           
        tempTarget = self.targets.copy()
        #TODO:why the temp targets are reshuffled each time getTargets is called?
        for key in tempTarget:
             self.rng.shuffle(tempTarget[key])
        return tempTarget

    def log_selection(self,classPerWorker):          
        totalLabels = [0 for i in range(len(classPerWorker[0]))]       
        logging.info("====Total # of workers is :{}, w/ {} labels, {}".format(len(classPerWorker), len(classPerWorker[0]), len(self.partitions)))    #不太明白，如果按照custom_partition的定义，最后一个输出也就是num_client啊？
        for index, row in enumerate(classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(classPerWorker[index]):    
                rowStr += '\t'+str(int(label))                 
                totalLabels[i] += label                              
                numSamples += label                           
            logging.info(str(index) + ':\t' + rowStr + '\t' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index])))
            logging.info("=====================================\n")
        logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        logging.info("=====================================\n")


    def use(self, partition, istest):
        resultIndex = self.partitions[partition]                 
        
        exeuteLength = -1 if not istest else int(len(resultIndex) * self.args.test_ratio)       
        resultIndex = resultIndex[:exeuteLength]                                           
        self.rng.shuffle(resultIndex)                           

        return Partition(self.data, resultIndex)

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}  


def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None, seed=0):
    """Load data given client Id"""                
    partition = partition.use(rank - 1, isTest)
    dropLast = False if isTest or (args.used_samples < 0) else True           

    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)  

    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)

