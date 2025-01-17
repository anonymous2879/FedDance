import logging
import math
import pickle
from random import Random
from helper.client import Client                             


from torch.multiprocessing import Queue                 
import numpy as np

class clientManager(object):

    def __init__(self, mode, args): #, sample_seed=233):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode                                   
        self.filter_less = args.filter_less        
        self.filter_more = args.filter_more

        self.clients_exe_cost = {}

        self.ucbSampler = None                          

        if self.mode == 'oort':                            
            from oort import create_training_selector     
            self.ucbSampler = create_training_selector(args=args)
            
        self.feasibleClients = []                          
        self.rng = Random()
        self.rng.seed(233 + args.sample_seed)              
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None                             
        self.args = args                                  

        #cache online clients
        self.cur_online_clients = []                      
        #introduce a per client counter of avail periods equal to the length of deadline
        self.avail_counter = {}                        
        self.low_avail = 0

        if args.device_avail_file is not None:           
           
            with open(args.device_avail_file, 'rb') as fin:
                self.user_trace = pickle.load(fin)
            #set values for the random assignment from low avail clients   
            if self.args.random_behv == 2:                  
            #Behaviour heterogenity or not: 0: do not use behaviour trace - always available, 1: use behaviour trace, dynamic client availability
                self.sorted_user_ids = [item[0] for item in sorted(self.user_trace.items(), key=lambda item: item[1]['duration'])]  
                self.user_trace_len = len(self.user_trace) 

        
        
        self.client_event_queue = Queue() 
        self.average_loss = 0




  
    def getPriority(self,clientId, cur_time, time_window, cur_epoch, lookup_timeslots=2, deadline = 100):
  
        logging.info("--------------------------------------------------client_manager.py: getPriority start!------------------------------------------- ")
        priority=0

        availability = self.getAvailability(clientId, cur_time, time_window, lookup_timeslots, deadline)


        loss = self.Clients[self.getUniqueId(0, clientId)].round_loss
#        priority = availability*loss                                

#        priority = availability*loss*(self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a)

#        delta_a = 0
#        delta_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a +3.00
#        priority = availability*loss*(delta_a/4)

#        delta_a = 0
#        delta_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a +3.00
#        if delta_a < 0:
#            delta_a = -delta_a       
#        priority = availability*loss*((delta_a+1)/2)


#        delta_a = 0
#        delta_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a +3.00
#        if delta_a < 0:
#            delta_a = -delta_a       

        #cur_round = math.floor(cur_time/deadline)       

#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability*loss*((delta_a+1)/2)*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability*loss*((delta_a+1)/2)*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))


#        delta_a = 0
#        delta_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a +3.00

#        priority = availability*(delta_a/4)
        

#        delta_a = 0
#        delta_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a
#        if delta_a < 0:
#            delta_a = -delta_a       
#        priority = availability*loss*((delta_a+1)/2)



#        delta_a = 0
#        delta_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0, clientId)].l_a
#        if delta_a < 0:
#            delta_a = -delta_a       
        #cur_round = math.floor(cur_time/deadline)       

#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability*loss*((delta_a+1)/2)*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability*loss*((delta_a+1)/2)*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))




#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#            availability = 1
#            loss = 100000
#            average_a = 1
#            priority = availability*loss*average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))


#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >= 0:
#            priority = availability*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            if self.Clients[self.getUniqueId(0, clientId)].flag == 1:  
#                logging.info("--------------------------------------------------online again!------------------------------------------- ")
#                priority = availability*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#            else:       
#                logging.info("--------------------------------------------------first online!------------------------------------------- ")                                               #还没online过的
#                availability = 1
#                priority = availability*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
        availability_dot = 0
        availability_dot = (1+availability)/2
        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
        else:
            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        availability_dot = 0
#        availability_dot = (1+availability)/2
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a
#        else:
#            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a


#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        availability_1 = 0
#        availability_1 = (1+availability)/2

#        t_rate = 0
#        t_rate = (math.exp(1)/(math.exp(1)-1))*(1-math.exp(-0.001*cur_epoch))

#        availability_dot = 0
#        alpha = 0.7

#        availability_dot = alpha+(1-alpha)*t_rate*availability_1

#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        availability_dot = 0
#        b_e = 0.8
#        availability_dot = b_e + (1-b_e)*availability**(0.5)
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability_dot*loss*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        priority = loss


#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability*loss*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability*loss**(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))


#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        availability_dot = 0
#        availability_dot = (1+availability)/2
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability_dot*loss*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability_dot*loss*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))

##            priority = availability*loss**(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))



#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
##        availability_dot = 0
##        availability_dot = (1+availability)/2
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = loss*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = loss*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))


#        logging.info("--------------------------------------------------inital round priority!------------------------------------------- ")
#        availability_dot = 0
#        availability_dot = (1+availability)/2
#        if self.Clients[self.getUniqueId(0, clientId)].last_round >=0:
#            priority = availability_dot*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(self.Clients[self.getUniqueId(0, clientId)].last_round+1)))
#        else:
#            priority = availability_dot*self.Clients[self.getUniqueId(0, clientId)].average_a*(1+(math.log(cur_epoch, 10)+1)/(10*(-self.Clients[self.getUniqueId(0, clientId)].last_round+1)))


#        if loss == 0:         
#            priority = availability*loss
#        else:                 
#            priority = availability*loss            

        return priority               


    def getAvailability(self,clientId, cur_time, time_window, lookup_timeslots=2, deadline = 100):
        logging.info("--------------------------------------------------client_manager.py: getAvailability start!------------------------------------------- ")
        count = 0                            
        availability = 0                   
#        future_time = deadline*100           
        future_time = 5                     
        


        '''
        norm_time = cur_time % self.Clients[self.getUniqueId(0, clientId)].traces['finish_time']   
        availabilityPeriods=self.Clients[self.getUniqueId(0, clientId)].availabilityPeriods()    
        for period in availabilityPeriods:
            start, end = period                                     
            if end <= norm_time:                                   
                count += math.floor(norm_time - start) / deadline


        if count != 0:
#            rate = count / norm_time                             
            round = math.ceil(norm_time / deadline)
            if round < 1:
                round = 1
            rate = math.ceil(count / round)                   
            availability = 1 - np.exp(-rate * future_time)      

        '''

        norm_time = cur_time % self.Clients[self.getUniqueId(0, clientId)].traces['finish_time']   
        availabilityPeriods=self.Clients[self.getUniqueId(0, clientId)].availabilityPeriods()     
        his_window = 50    
#        norm_time      
        round = math.ceil(norm_time / deadline)   
        norm_time_start = 0
        if (round > his_window):
            norm_time_start = norm_time - his_window*deadline

           
        
        for period in availabilityPeriods:
            start, end = period                                     
            
            #mode1， 
#            if (start <= norm_time):                                
#                min_end = min(end, norm_time)
#                round_end = math.floor(min_end) / deadline
#                round_start = math.ceil(norm_time_start) / deadline
#                tem_count = round_end - round_start
#                if tem_count <= 0:
#                    tem_count = 0

#                count += tem_count     #            

                
            #mode2
            if (norm_time_start <= start <= norm_time) or (norm_time_start <= end <= norm_time):                  
                min_end = min(end, norm_time)
                norm_end = math.floor(min_end) / deadline
                max_start = max(start, norm_time_start)
                norm_start = math.ceil(max_start) / deadline
                tem_count = norm_end - norm_start
                if tem_count <= 0:
                    tem_count = 0

                count += tem_count     #                     
      
         

        
        if count != 0:
#            rate = count / norm_time                            
            round = math.ceil(norm_time / deadline)
            if round < 1:
                round = 1
            rate = math.ceil(count / round)                    
            availability = 1 - np.exp(-rate * future_time)        
                     



        return availability





    #get the count of availability periods divided into slots of the deadline
    def getPeriodCount(self, clientId, cur_time, deadline):       
        availabilityPeriods = self.Clients[self.getUniqueId(0, clientId)].availabilityPeriods()
        finishtime = self.Clients[self.getUniqueId(0, clientId)].traces['finish_time']
        norm_time = cur_time % finishtime                       
        index = 0
        for period in availabilityPeriods:
            start, end = period
            if norm_time < start:
                # logging.info('client {} period {} normtime {}'.format(clientId, period, norm_time))
                break
            index += 1                                         
        count = 0
        if index > 0:                                          
            v1, v2 = availabilityPeriods[index - 1]             
            count += int((v2 - v1 - norm_time) / deadline)       
            
        for i in range(index, len(availabilityPeriods)):        
            start, end = availabilityPeriods[i]
            duration_normed = int((end - start) / deadline)    
            if duration_normed > 0:
                count += duration_normed                        
             
        return count




    '''
    def registerClient(self, hostId, clientId, size, speed, duration=1):  
        uniqueId = self.getUniqueId(hostId, clientId)                      
        user_trace = None
        if self.user_trace and self.args.random_behv >= 0:                 
        #
        #Behaviour heterogenity or not: 0: do not use behaviour trace - always available, 1: use behaviour trace, dynamic client availability 
            if self.args.random_behv > 0:                                 
                if self.args.random_behv == 1:                      
                    # randomly set the user behaviour to the client        
                    user_trace = self.user_trace[self.rng.randint(1, len(self.user_trace))]  
                elif self.args.random_behv == 2:                           
                    u = self.rng.random()                                
                    index = int(self.user_trace_len * 0.1)                
                    if u < 1.0:
                        user_id = self.rng.choice(self.sorted_user_ids[:index])         
                        self.low_avail += 1                                          
                    else:                                                               
                        user_id = self.rng.choice(self.sorted_user_ids[index + 1:])     
                    user_trace = self.user_trace[user_id]                              

            else:  
                #fix an error thrown when the clientId from the dataset is more than 107K in user_trace (happend with stackoverflow)
                #for stackoverflow feasible clients {'total_feasible_clients': 281347, 'total_length': 41367564}
                # set the user behaviour based on client ID (sequential)
                cid = int(clientId) % len(self.user_trace)                              
                if cid in self.user_trace:                                        
                    user_trace = self.user_trace[cid]
                else:
                    cid=self.rng.randint(1, len(self.user_trace))                    
                    user_trace = self.user_trace[cid]
        self.Clients[uniqueId] = Client(hostId, clientId, speed, user_trace)            

        # remove clients                  
        if self.args.used_samples < 0 or (size >= self.filter_less and size <= self.filter_more):    
            self.feasibleClients.append(clientId)           
            #还将该clientId加入self.feasibleClients                                                
            self.feasible_samples += size             

            if self.mode == "oort":                        
                feedbacks = {'reward':min(size, self.args.local_steps*self.args.batch_size),
                            'duration':duration,
                            }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)
    '''


    
    def registerClient(self, hostId, clientId, size, speed, duration=1):    
        matchId = int(clientId) % len(self.user_trace)                    
        uniqueId = self.getUniqueId(hostId, clientId)                    
        user_trace = None
        if self.user_trace and self.args.random_behv >= 0:             
        #Behaviour heterogenity or not: 0: do not use behaviour trace - always available, 1: use behaviour trace, dynamic client availability 
            if self.args.random_behv > 0:                                   #dynamic availability
                if self.args.random_behv == 1:                      
                    # randomly set the user behaviour to the client        
                    #logging.info(f'Number of trace: {len(self.user_trace)}')  
                    #user_trace = self.user_trace[self.rng.randint(1, len(self.user_trace))]  
                    #user_trace = self.user_trace[self.rng.randint(1, len(self.user_trace))-1]   
                    user_trace = self.user_trace[matchId]  

                elif self.args.random_behv == 2:                          
                    u = self.rng.random()                                   
                    index = int(self.user_trace_len * 0.1)                  
                    if u < 1.0:
                        user_id = self.rng.choice(self.sorted_user_ids[:index])         
                        self.low_avail += 1                                             
                    else:                                                               
                        user_id = self.rng.choice(self.sorted_user_ids[index + 1:])     
                    user_trace = self.user_trace[user_id]                               

            else:    #0: do not use behaviour trace - always available
                #fix an error thrown when the clientId from the dataset is more than 107K in user_trace (happend with stackoverflow)
                #for stackoverflow feasible clients {'total_feasible_clients': 281347, 'total_length': 41367564}
                # set the user behaviour based on client ID (sequential)
                cid = int(clientId) % len(self.user_trace)                               
                if cid in self.user_trace:                                            
                    user_trace = self.user_trace[cid]
                else:
                    cid=self.rng.randint(1, len(self.user_trace))               
                    user_trace = self.user_trace[cid]
        
        elif self.user_trace and self.args.random_behv == -6:                            
            user_trace = self.user_trace[matchId]  
       
       
        self.Clients[uniqueId] = Client(hostId, clientId, speed, user_trace)         




        # remove clients                   
        if self.args.used_samples < 0 or (size >= self.filter_less and size <= self.filter_more):    
            self.feasibleClients.append(clientId)                                                          
            self.feasible_samples += size                  

            if self.mode == "oort":                       
                feedbacks = {'reward':min(size, self.args.local_steps*self.args.batch_size),
                            'duration':duration,
                            }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)



    def getAllClients(self):                              
        return self.feasibleClients

    def getAllClientsLength(self):                        
        return len(self.feasibleClients)

    def getClient(self, clientId):                       
        return self.Clients[self.getUniqueId(0, clientId)]

    #update client costs
    def updateClientCosts(self):                       
        self.clients_exe_cost = dict(sorted(self.clients_exe_cost.items(), key=lambda item: item[1][0]+item[1][1]))    
        if self.args.scale_sys_percent > 0:               
            count = len(self.clients_exe_cost)
            i = 0
            for c in self.clients_exe_cost:
                if i <= int(count * self.args.scale_sys_percent):
                    self.clients_exe_cost[c] = (self.clients_exe_cost[c][0] * self.args.scale_sys, self.clients_exe_cost[c][1] * self.args.scale_sys)
                self.ucbSampler.update_duration(c, self.clients_exe_cost[c][0] +  self.clients_exe_cost[c][1])
                i += 1
        logging.info(f'Execution costs are updated: {len(self.clients_exe_cost)}')

    def registerDuration(self, clientId, batch_size, upload_epoch, upload_size, download_size):
        clientId = self.getUniqueId(0, clientId)            
        exe_cost = self.Clients[clientId].getCompletionTime(batch_size=batch_size, upload_epoch=upload_epoch,
            upload_size=upload_size, download_size=download_size)     
        self.clients_exe_cost[clientId] = (exe_cost['computation'], exe_cost['communication'])
        if self.mode == "oort":
            self.ucbSampler.update_duration(clientId,  self.clients_exe_cost[clientId][0] + self.clients_exe_cost[clientId][1])

    def getCompletionList(self, percent, scale):
        client_exe_list = sorted([self.clients_exe_cost[c][0] + self.clients_exe_cost[c][1] for c in self.clients_exe_cost])
        if percent > 0:
            count = len(client_exe_list)                  
            i = 0
            for i in range(0, count):
                if i <= int(count * percent):              
                    client_exe_list[i] = client_exe_list[i] * scale
                i += 1
        return client_exe_list

    def getCompletionTime(self, clientId):                 
        clientId = self.getUniqueId(0, clientId)
        return {'computation': self.clients_exe_cost[clientId][0], 'communication': self.clients_exe_cost[clientId][1]}

    def registerSpeed(self, hostId, clientId, speed):      
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed             

    def registerScore(self, clientId, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward         
        if self.mode == "oort":                         
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }
            self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)
        self.registerClientScore(clientId, reward)               

    def registerClientScore(self, clientId, reward):       
        self.Clients[self.getUniqueId(0, clientId)].registerReward(reward)

    def getScore(self, hostId, clientId):                  
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):  
            client = self.Clients[clientId]
            clientInfo[client.clientId] = client.distance   
        return clientInfo

    def nextClientIdToRun(self, hostId):
        init_id = hostId - 1                            
        lenPossible = len(self.feasibleClients)            

        while True:
            clientId = str(self.feasibleClients[init_id])  
            csize = self.Clients[clientId].size             
            if self.args.used_samples < 0 or (csize >= self.filter_less and csize <= self.filter_more): 
                return int(clientId)
            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))    
        return init_id

    def getUniqueId(self, hostId, clientId):                
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientSampler(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size            

    def clientOnHost(self, clientIds, hostId):
        self.clientOnHosts[hostId] = clientIds                          

    def getCurrentClientIds(self, hostId):                                  
        return self.clientOnHosts[hostId]                               

    def getClientLenOnHost(self, hostId):
        return len(self.clientOnHosts[hostId])                              

    def getClientSize(self, clientId):                                     
        return self.Clients[self.getUniqueId(0, clientId)].size                   

    def getSampleRatio(self, clientId, hostId, even=False):
        totalSampleInTraining = 0.
        if not even:
            for key in self.clientOnHosts.keys():                         
                for client in self.clientOnHosts[key]:
                    uniqueId = self.getUniqueId(key, client)               
                    totalSampleInTraining += self.Clients[uniqueId].size   

            #1./len(self.clientOnHosts.keys())
            return float(self.Clients[self.getUniqueId(hostId, clientId)].size)/float(totalSampleInTraining)  
        else:
            for key in self.clientOnHosts.keys():
                totalSampleInTraining += len(self.clientOnHosts[key])       

            return 1./totalSampleInTraining                              


    
    def getFeasibleClients(self, cur_time):
        if self.user_trace is None:                                         
            clients_online = self.feasibleClients                           
        else:                                                                                                                         
            clients_online = [clientId for clientId in self.feasibleClients if self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)]

        logging.info(f"Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " + f"{len(self.feasibleClients)-len(clients_online)} clients offline")

        return clients_online


    
    def isClientActive(self, clientId, cur_time, time_window):       
        return self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time + time_window)


    '''
    def resampleClients(self, numOfClients, cur_time=0, time_window=0):
        priority_clients = []                                           
        remaining_clients = []                                         
        self.count += 1                                                  

        clients_online = self.getFeasibleClients(cur_time)                  
        self.cur_online_clients = clients_online                            
        

        if len(clients_online) <= numOfClients:                             
            return clients_online, priority_clients                      

        feasible_clients = clients_online                                   

        online = len(feasible_clients)                                     
        target_num = numOfClients  # max(numOfClients, int(0.25 * online))

        temp_feasible_clients = feasible_clients                           

        #select from the high priority with p=2
        # TODO: create a sub-set of feasible (online) with the ones that have priority = 2
        if self.args.avail_priority >= 1:                                   
            priority_vals = {}                                              
            for c in feasible_clients:                                      
                priority_vals[c] = self.getPriority(c, cur_time, time_window, lookup_timeslots=2)            
            priority_clients = [key for key, val in priority_vals.items() if val == 1]                 
                          
            remaining_clients = [key for key, val in priority_vals.items() if val == 0]                
            
            #apply probability - not 100% accuracy                  
            if self.args.avail_probability > 0:                            
                acc_num = int(len(priority_clients) * self.args.avail_probability)
                priority_clients = self.rng.sample(priority_clients, acc_num)     

                acc_num = int(len(remaining_clients) * self.args.avail_probability)
                remaining_clients = self.rng.sample(remaining_clients, acc_num)   

            #set the feasible to the list of clients
            temp_feasible_clients = [v for v in priority_clients]                 

            #fill the feasible list with random seleciton from remaining clients
            remain_num = target_num - len(temp_feasible_clients)                    
            if self.args.avail_priority == 1 and remain_num > 0:                  
                temp_feasible_clients.extend(self.rng.sample(remaining_clients, remain_num))   
                   

        selection_list = temp_feasible_clients         
        temp_pickled_clients = []
        if len(selection_list) <= target_num and self.args.avail_priority == 2: 
            temp_pickled_clients = selection_list
            target_num -= len(selection_list)            
            selection_list = remaining_clients
            selection_set = set(remaining_clients)
        else:
            selection_set = set(selection_list)         

        pickled_clients = None                          
        if self.mode == "oort" and self.count > 1:           
            pickled_clients = self.ucbSampler.select_participant(target_num, feasible_clients=selection_set)
        else:                                           
            self.rng.shuffle(selection_list)
            client_len = min(target_num, len(selection_list) - 1)   
            pickled_clients = selection_list[:client_len]           

        pickled_clients.extend(temp_pickled_clients)    

        logging.info(
            "==== PS Client Sampler - avail prio {} - window {}, online {}, target: {}:{} feasible {}:{}:{}, not high:{} high priority {}:{} ".format(
                self.args.avail_priority, time_window, online, numOfClients, target_num, len(temp_feasible_clients),
                len(pickled_clients), pickled_clients, [x for x in pickled_clients if x not in priority_clients], len(priority_clients), priority_clients))

        return pickled_clients, priority_clients        
    '''


    
    def resampleClients(self, numOfClients, cur_time=0, time_window=0, cur_epoch = 0):        
        priority_clients = []                                             
        remaining_clients = []                                           
        self.count += 1                                                  

        clients_online = self.getFeasibleClients(cur_time)                  
        self.cur_online_clients = clients_online                            


        if len(clients_online) <= numOfClients:                             
            return clients_online, priority_clients                         

        feasible_clients = clients_online                                   

        online = len(feasible_clients)                                      
        target_num = numOfClients  # max(numOfClients, int(0.25 * online))

        temp_feasible_clients = feasible_clients                            

        #select from the high priority with p=2
        # TODO: create a sub-set of feasible (online) with the ones that have priority = 2
        
        
        priority_mode = 1               
        
        if (self.args.avail_priority >= 1) and (priority_mode == 0):                                   
            priority_vals = {}                                              
            for c in feasible_clients:                                      
                priority_vals[c] = self.getPriority(c, cur_time, time_window, lookup_timeslots=2)              

            priority_clients = [key for key, val in priority_vals.items() if val == 1]                                
            remaining_clients = [key for key, val in priority_vals.items() if val == 0]                 

            #- apply probability - not 100% accuracy                  
            if self.args.avail_probability > 0:                            
                acc_num = int(len(priority_clients) * self.args.avail_probability)
                priority_clients = self.rng.sample(priority_clients, acc_num)     

                acc_num = int(len(remaining_clients) * self.args.avail_probability)
                remaining_clients = self.rng.sample(remaining_clients, acc_num)   

            #- set the feasible to the list of clients
            temp_feasible_clients = [v for v in priority_clients]                 

            #- fill the feasible list with random seleciton from remaining clients
            remain_num = target_num - len(temp_feasible_clients)                         
            if self.args.avail_priority == 1 and remain_num > 0:                             
                temp_feasible_clients.extend(self.rng.sample(remaining_clients, remain_num))   
                    

        selection_list = temp_feasible_clients           
        temp_pickled_clients = []
        if len(selection_list) <= target_num and self.args.avail_priority == 2:   
            temp_pickled_clients = selection_list
            target_num -= len(selection_list)            
            selection_list = remaining_clients
            selection_set = set(remaining_clients)
        else:
            selection_set = set(selection_list)         

        pickled_clients = None                          
        if self.mode == "oort" and self.count > 1:             
            pickled_clients = self.ucbSampler.select_participant(target_num, feasible_clients=selection_set)
        else:                                           
            self.rng.shuffle(selection_list)
            client_len = min(target_num, len(selection_list) - 1)   
            pickled_clients = selection_list[:client_len]           

        pickled_clients.extend(temp_pickled_clients)    



        if (self.args.avail_priority >= 1) and (priority_mode == 1):                                   
            logging.info("--------------------------------------------------client_manager.py: start new resampleClients!------------------------------------------- ")
            priority_vals = {}                                              
            for c in feasible_clients:                                     
#                priority_vals[c] = self.getPriority(c, cur_time, time_window, lookup_timeslots=2)             
                priority_vals[c] = self.getPriority(c, cur_time, time_window, cur_epoch, lookup_timeslots=2)              
           
#            priority_clients = [key for key, val in priority_vals.items()]                
            
            
            sorted_priority_vals = sorted(priority_vals.items(), key=lambda x: x[1], reverse=True)
#            sorted_priority_clients_list = [key for key, val in sorted_priority_vals.items()]
#            sorted_dict = {k: v for k, v in sorted_priority_clients_list}
            priority_clients = [v[0] for v in sorted_priority_vals]


            temp_feasible_clients = [v for v in priority_clients] 
            

        selection_list = temp_feasible_clients          
        temp_pickled_clients = []
        if len(selection_list) <= target_num and self.args.avail_priority == 2:  
            temp_pickled_clients = selection_list
            target_num -= len(selection_list)             
            selection_list = remaining_clients
            selection_set = set(remaining_clients)
        else:
            selection_set = set(selection_list)        

        pickled_clients = None                         
        if self.mode == "oort" and self.count > 1:           
            pickled_clients = self.ucbSampler.select_participant(target_num, feasible_clients=selection_set)
        else:                                           
            client_len = min(target_num, len(selection_list) - 1)  
            pickled_clients = selection_list[:client_len]          

        pickled_clients.extend(temp_pickled_clients)    


        if (self.args.avail_priority >= 1) and (priority_mode == 2):                                   
            logging.info("--------------------------------------------------client_manager.py: start marginal return test resampleClients!------------------------------------------- ")
            priority_vals = {}                                              
            for c in feasible_clients:                                      

                priority_vals[c] = self.Marginal_return_priority(c, cur_time)            
           
        
            sorted_priority_vals = sorted(priority_vals.items(), key=lambda x: x[1], reverse=True)
            priority_clients = [v[0] for v in sorted_priority_vals]

            temp_feasible_clients = [v for v in priority_clients] 
            

        selection_list = temp_feasible_clients          
        temp_pickled_clients = []

        selection_set = set(selection_list)           

        pickled_clients = None                       


        client_len = min(target_num, len(selection_list) - 1)   
        pickled_clients = selection_list[:client_len]          

        pickled_clients.extend(temp_pickled_clients)    


        logging.info(
            "==== PS Client Sampler - avail prio {} - window {}, online {}, target: {}:{} feasible {}:{}:{}, not high:{} high priority {}:{} ".format(
                self.args.avail_priority, time_window, online, numOfClients, target_num, len(temp_feasible_clients),
                len(pickled_clients), pickled_clients, [x for x in pickled_clients if x not in priority_clients], len(priority_clients), priority_clients))

        return pickled_clients, priority_clients        




    def Marginal_return_priority(self, c, cur_time):
        priority = 0
        acc_duration = 0
        index = 0

        trace = self.Clients[self.getUniqueId(0, c)].traces
        norm_time = cur_time % trace['finish_time']  

        availabilityPeriods = self.Clients[self.getUniqueId(0, c)].availabilityPeriods()   

        final_time = self.args.deadline * self.args.epochs

        for period in availabilityPeriods:
            start, end = period 
 
            if (norm_time <= end) and (start <= final_time):
                acc_duration = acc_duration + min(final_time, end) - max(norm_time, start)
                index = index + 1

        priority = acc_duration / (self.args.deadline)

        return priority 



    def getAllMetrics(self):
        if self.mode == "oort":                        
            return self.ucbSampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, clientId):              
        return self.ucbSampler.get_client_reward(clientId)

    def get_median_reward(self):
        if self.mode == 'oort':
            return self.ucbSampler.get_median_reward()
        return 0.                                     



    def updateLoss1(self, clientId, round_loss, c_a, previous_round_participant, last_round): 

        logging.info("--------------------------------------------------client_manager.py: start updateLoss1------------------------------------------- ")        
        if clientId in previous_round_participant:
            self.Clients[self.getUniqueId(0, clientId)].round_loss = round_loss          
            self.Clients[self.getUniqueId(0, clientId)].last_round = last_round          
            
            
            self.Clients[self.getUniqueId(0, clientId)].c_a = c_a   
            self.Clients[self.getUniqueId(0, clientId)].l_a = self.Clients[self.getUniqueId(0, clientId)].c_a   


            
#            self.Clients[self.getUniqueId(0, clientId)].a_count = self.Clients[self.getUniqueId(0, clientId)].a_count + (self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0,clientId)].l_a +3)/4

            # self.Clients[self.getUniqueId(0, clientId)].his_acc.append(c_a)                                                  
            # if len(self.Clients[self.getUniqueId(0, clientId)].his_acc) == 1:
            #     d_a = 0
            # elif (len(self.Clients[self.getUniqueId(0, clientId)].his_acc) > 1) and (len(self.Clients[self.getUniqueId(0, clientId)].his_acc) < self.args.his_beta):
            #     d_a = self.Clients[self.getUniqueId(0, clientId)].his_acc[-1] - self.Clients[self.getUniqueId(0, clientId)].his_acc[0]
            # elif (len(self.Clients[self.getUniqueId(0, clientId)].his_acc) >= self.args.his_beta):
            #     d_a = self.Clients[self.getUniqueId(0, clientId)].his_acc[-1] - self.Clients[self.getUniqueId(0, clientId)].his_acc[-self.args.his_beta] 

            
            d_a = self.Clients[self.getUniqueId(0, clientId)].c_a - self.Clients[self.getUniqueId(0,clientId)].l_a  
#            d_a = self.Clients[self.getUniqueId(0, clientId)].c_a    #new operation for marginal return test        

            if d_a >=0:
                self.Clients[self.getUniqueId(0, clientId)].a_count = self.Clients[self.getUniqueId(0, clientId)].a_count + (d_a +1)/2           
            else:
                self.Clients[self.getUniqueId(0, clientId)].a_count = self.Clients[self.getUniqueId(0, clientId)].a_count + (-1*d_a +1)/2           


            self.Clients[self.getUniqueId(0, clientId)].p_count = self.Clients[self.getUniqueId(0, clientId)].p_count + 1

            self.Clients[self.getUniqueId(0, clientId)].average_a = self.Clients[self.getUniqueId(0, clientId)].a_count / self.Clients[self.getUniqueId(0, clientId)].p_count            



    def updateLoss2(self, previous_round_participant, previous_round_online_clients, last_round): 
        logging.info("--------------------------------------------------client_manager.py: start updateLoss2------------------------------------------- ")
        average_loss = 0
        average_c_a = 0
        average_a = 0


        for clientId in previous_round_participant: 
            average_loss += self.Clients[self.getUniqueId(0, clientId)].round_loss
            average_c_a += self.Clients[self.getUniqueId(0, clientId)].c_a           
            average_a += self.Clients[self.getUniqueId(0, clientId)].average_a            
        average_loss = average_loss / len(previous_round_participant)
        average_c_a = average_c_a / len(previous_round_participant)           
        average_a = average_a / len(previous_round_participant)               
##        self.average_loss = average_loss
##        for clientId in self.feasibleClients:                               
        for clientId in previous_round_online_clients:                           
            self.Clients[self.getUniqueId(0, clientId)].flag = 1
            if clientId not in previous_round_participant:
            
                if self.Clients[self.getUniqueId(0, clientId)].last_round == -1:                        
#                    self.Clients[self.getUniqueId(0, clientId)].round_loss = self.average_loss            
                    self.Clients[self.getUniqueId(0, clientId)].round_loss = average_loss          
                    self.Clients[self.getUniqueId(0, clientId)].c_a = average_c_a                     
                    self.Clients[self.getUniqueId(0, clientId)].average_a = average_a                                          
                elif self.Clients[self.getUniqueId(0, clientId)].last_round != last_round:           
                    pass                                                                              


