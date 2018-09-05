# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:00:28 2018

Read from ROS
@author: Nitin Nataraj
"""

#import matplotlib
#matplotlib.use('Agg')
import rosbag
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import random
import argparse
pi = math.pi


#import matplotlib.pyplot as plt
precision = 0.1
sigma_hit = 0.25


def transform(odom_message, lm_msg):
    xr, yr,theta = odom_message[0], odom_message[1], odom_message[2]
    lx, ly = lm_msg[3], -lm_msg[1]

    
    xl1 = xr + lx * np.cos(theta) - ly * np.sin(theta)
    yl1  =  yr + lx * np.sin(theta) + ly*np.cos(theta)
    return (lm_msg[0],xl1,yl1)
def quart_to_euler(x,y,z,w):
    roll  = math.atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z);
    pitch = math.atan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z);
    yaw   =  math.asin(2*x*y + 2*z*w);
    return roll,pitch, yaw

landmarks = list(zip(pd.read_csv("landmarks.csv")['X'].tolist(), 
                     pd.read_csv("landmarks.csv")['Y'].tolist()))


class monte_carlo:
    def __init__(self,
                 grid_dims,
                 resultArray,
                 landmark_locs,
                 precision = 0.1,
                 num_particles = 100,
                 motion_noise = 0.25,
                 turn_noise = 0.1
                 ):
       
        self.resultArray = resultArray
        self.inv_precision = 1/precision
        self.grid_dims = [int(grid_dims[0]),int(grid_dims[1]), 
                          int(grid_dims[2]*self.inv_precision)+1]
        self.landmark_locs = landmark_locs #Dictionary containing (x,y) of actual landmarks
        self.pNoise = 0.8
        self.sigma_hit = 0.25
        self.__form_grid()
        self.num_particles = num_particles
        self.particles =[]
        self.motion_noise = float(motion_noise)
        self.turn_noise = float(turn_noise)
        
        
        
    
    def __form_grid(self):
        self.grid = np.ones(self.grid_dims)/(self.grid_dims[0]*self.grid_dims[1]*self.grid_dims[2])
        print("Created grid of size: ", self.grid_dims)  
    

    
    def motion_mc(self,startState, endState):
        #startState and endState are the control inputs
        new_grid = np.zeros(shape = self.grid.shape) + 0.00001
        new_grid = new_grid/sum(new_grid)
        ##Change the orientation
        x1, y1, theta1 = startState[0], startState[1], startState[2]
        x2, y2, theta2 = endState[0], endState[1], endState[2]
        orientation = math.atan2((x2-x1),(y2 -y1)) - theta1
        
        particles = []
        for i,tup in enumerate(self.particles):
            x = tup[0]
            y = tup[1]
            theta = tup[2]
            x_new  = x + (x2 - x1) + random.gauss(0.0, self.motion_noise)
            y_new = y + (y2-y1) +  random.gauss(0.0, self.motion_noise)
            theta_new = theta + orientation + random.gauss(0.0, self.turn_noise)
            particles.append((x_new,y_new, theta_new))
            #Place probability in new grid cell
            x,y,theta = int(x*self.inv_precision), int(y*self.inv_precision), int(theta*self.inv_precision)
            x_new , y_new, theta_new = int(x_new*self.inv_precision),int(y_new*self.inv_precision), \
                                        int(theta_new*self.inv_precision)
            
            ##Shift x_new to fit in positive axis
            x_new += 100
            x += 100
            #print("Trying to move from ", (x,y,theta)," to ", (x_new,y_new, theta_new))
            if (x_new >= 0 and x_new < self.grid.shape[0] and 
                y_new >= 0 and y_new < self.grid.shape[1] and
                theta_new >= 0 and theta_new < self.grid.shape[2] and
                x >= 0 and x < self.grid.shape[0] and 
                y >= 0 and y < self.grid.shape[1] and
                theta >= 0 and theta < self.grid.shape[2]):

                new_grid[x_new][y_new][theta_new] += self.grid[x][y][theta]  
        self.particles = particles
        self.grid = new_grid
        
        print(np.sum(new_grid))
    
    def euclidean_dist(self,start,end):
        #start and end are of type (x,y)
        return math.sqrt((start[0] - end[0]) **2 + (start[1] - end[1])**2)
    def beam(self,z_range, z_star):
        """
        Returns the corrected value of the range measurement
        z_range = sensor measurement at that time step
        z_star = point in state space being considered at that time step
        """
    
        exponent = (-0.5*(z_range-z_star)**2)/(self.sigma_hit**2)
        noise = (1/np.sqrt(2*pi*self.sigma_hit*self.sigma_hit))* np.exp(exponent)
        result = self.pNoise * noise + (1-self.pNoise) * (1/z_range)
        return result

    
    def observe_mc(self,newMsg):
        weights = []
        land_id = newMsg["landmark1"][0]
        xl = newMsg["landmark1"][1]
        yl = newMsg["landmark1"][2]
        for tup in self.particles:
            x = tup[0]
            y = tup[1]
            theta = tup[2]

            #For every point, check distance from landmark
            observed_dist = self.euclidean_dist((x,y),(xl,yl))
            actual_dist = self.euclidean_dist((x,y),self.landmark_locs[land_id])
            x_new,y_new,theta_new = int(x*self.inv_precision), int(y* self.inv_precision), \
                                    int(theta*self.inv_precision)
                                    
            #Shift x_new to positive axis
            x_new += 100
            
            wt = self.beam(observed_dist,actual_dist)
            weights.append(wt)
            if (x_new >= 0 and x_new < self.grid.shape[0] and 
                y_new >= 0 and y_new < self.grid.shape[1] and
                theta_new >= 0 and theta_new < self.grid.shape[2]):
                self.grid[x_new][y_new][theta_new] *= wt
                
        
        self.grid = self.grid/np.sum(self.grid)
        print(np.sum(self.grid))
        return weights

    def resample(self,weights):
        particles = []
        #Normalize weights
        norm_weights = weights/np.sum(weights)
        
        #Draw representative sample
        indices = [np.random.choice(np.arange(0, self.num_particles), 
                                    p=norm_weights) for i in range(self.num_particles)]

        for i in indices:
            particles.append(self.particles[i])
        assert self.num_particles == len(particles)
        
        #Update particles
        self.particles = particles
        self.num_particles = len(particles)
                
        print("There are now %d particles"%self.num_particles)
        
    def localize_mc(self,steps):
        grids = []
        f = open("stuff.txt",'a')
        #Generate random set of particles
        message_stream = get_message(self.resultArray)
        while True:
            startMsg = next(message_stream)
            if startMsg is not {}:
                break
        
        startState = (startMsg["msg"][0],startMsg["msg"][1],startMsg["msg"][2])
        
        #Generate random particles
        for i in range(0,self.num_particles):
            x = round(random.random(),5) * self.grid_dims[0] /(2*self.inv_precision)
            y = round(random.random(),5) * self.grid_dims[1] /self.inv_precision         
            orientation = round(random.random(),5) *self.grid_dims[2] /self.inv_precision
            self.particles.append((x,y,orientation))
        print("Particles created")
        
        for i in range(steps):
            print("Iteration %d"%i)
            while True:
                endMsg = next(message_stream)
                if endMsg is not {}:
                    break
            print(endMsg)
            endState = (endMsg["msg"][0],endMsg["msg"][1],endMsg["msg"][2])
            print("Moving")
            self.motion_mc(startState,endState)
            
            print("Observing")
            print("Calculating particle weights")
            weights = self.observe_mc(endMsg)
            
            print("Resampling")
            self.resample(weights)

            startState = endState
            ans = np.where(self.grid == np.amax(self.grid))
            print(ans)
            grids.append(self.grid)
            f.write(str(ans))
            startMsg = endMsg
        pickle.dump(grids, open("grids.pkl",'wb'),protocol = 2)
        return self.grid
            


            
        
        
            


def get_ros_samples(bagFile):
    bag = rosbag.Bag(bagFile)
    lm_msg_prev = (None,0.0,0.0)
    lm_msg_prev2 = (None,0.0,0.0)
    results = []
    count = 0
    for topic, msg, t in bag.read_messages(topics=['/odom', '/tag_detections']):
            
    
            if topic == "/odom": 
                
                #Flipping the x coordinate
                #y = -round(msg.pose.pose.position.x,5)
                #x = round(msg.pose.pose.position.y,5)
                x = round(msg.pose.pose.position.x,5)
                y = round(msg.pose.pose.position.y,5)
                q0 = msg.pose.pose.orientation.x
                q1 = msg.pose.pose.orientation.y
                q2 = msg.pose.pose.orientation.z
                q3 = msg.pose.pose.orientation.w
                _,_,yaw = quart_to_euler(q0,q1,q2,q3)
                
                odom_message = (x,y,yaw)
    
            if topic == "/tag_detections":
                if len(msg.detections) == 1:
                    #1 landmark
                    
                    landmark_id = msg.detections[0].id
                    if landmark_id != 18:
    
                        x = round(msg.detections[0].pose.pose.position.x,5)
                        y = round(msg.detections[0].pose.pose.position.y,5)
                        z = round(msg.detections[0].pose.pose.position.z,5)
                        q0 = msg.detections[0].pose.pose.orientation.x
                        q1 = msg.detections[0].pose.pose.orientation.y
                        q2 = msg.detections[0].pose.pose.orientation.z
                        q3 = msg.detections[0].pose.pose.orientation.w
                        _,_,yaw = quart_to_euler(q0,q1,q2,q3)
                        
                        ##Calculate landmark coordinates with respect to robot
                        lm_msg_cur = transform(odom_message, (landmark_id,x,y,z))
                        
                        if ((abs(lm_msg_cur[1] - lm_msg_prev[1]) >= precision) or
                             (abs(lm_msg_cur[2] - lm_msg_prev[2]) >= precision)):
                                count += 1
                                result = {}
                                result["msg"] = odom_message
                                result["landmark1"] = lm_msg_cur
                                result["landmark2"] = None
                                results.append(result)
                                lm_msg_prev = lm_msg_cur
                        
                        #Check for landmark 2
                        if len(msg.detections) == 2:
                            #2 landmark
                            
                            landmark_id_2 = msg.detections[1].id
                            if landmark_id_2 != 18:
                                x2 = round(msg.detections[1].pose.pose.position.x,5)
                                y2 = round(msg.detections[1].pose.pose.position.y,5)
                                z2 = round(msg.detections[1].pose.pose.position.z,5)
                                q02 = msg.detections[1].pose.pose.orientation.x
                                q12 = msg.detections[1].pose.pose.orientation.y
                                q22 = msg.detections[1].pose.pose.orientation.z
                                q32 = msg.detections[1].pose.pose.orientation.w
                                _,_,yaw2 = quart_to_euler(q02,q12,q22,q32)
                                
                                ##Calculate landmark coordinates with respect to robot
                                lm_msg_cur2 = transform(odom_message, (landmark_id_2,x2,y2,z2))
                                
                                if (((abs(lm_msg_cur2[1] - lm_msg_prev2[1]) >= precision) or
                                     (abs(lm_msg_cur2[2] - lm_msg_prev2[2]) >= precision)) or
                                    ((abs(lm_msg_cur2[1] - lm_msg_prev[1]) >= precision) or
                                     (abs(lm_msg_cur2[2] - lm_msg_prev[2]) >= precision))):
            
                                        results[-1]["landmark2"] = lm_msg_cur2
                                        lm_msg_prev2 = lm_msg_cur2
    print("Loaded %d ros samples"%len(results))
    return results

def get_message(resultArray):  
    for dic in resultArray:
        yield dic
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", help="Number of localization steps",type = int,required = True) 
    parser.add_argument("--num_particles", help="Number of particles",type = int,required = True) 
    args = parser.parse_args()
    steps = args.steps
    num_particles = args.num_particles
    #steps = 200
    #num_particles = 5000
    bagFile = "CSE668_1.bag"
    tr = get_ros_samples(bagFile)
    #Get actual landmark locations and put into a dictionary
    df_landmark = pd.read_csv("landmarks.csv")
    landmark_locs = {df_landmark['ID'][i]:(df_landmark['X'][i],df_landmark['Y'][i]) 
                        for i in range(len(df_landmark))}
    
    #a = read_from_ros2(rosbag.Bag(bagFile))
    mc = monte_carlo(grid_dims = [200,100,6.2], resultArray = tr,
                     landmark_locs = landmark_locs,num_particles=num_particles)
    grid = mc.localize_mc(steps)
    

    #Get x values of trajectory points.
    x = [dic["msg"][0] for dic in tr]
    y = [dic["msg"][1] for dic in tr]
    ids = [dic["landmark1"][0] for dic in tr]
    xl = [dic["landmark1"][1] for dic in tr]
    yl = [dic["landmark1"][2] for dic in tr]
    xl2 = [dic["landmark2"][1] if dic["landmark2"] is not None else None for dic in tr]
    yl2 = [dic["landmark2"][2] if dic["landmark2"] is not None else None for dic in tr] 
    ids2 = [dic["landmark2"][0] if dic["landmark2"] is not None else None for dic in tr] 
    
    #Saving to a dataframe for printing later
    df = pd.DataFrame(data = {"x":x,"y":y,"xl":xl,"yl":yl,"id":ids,"xl2":xl2,"yl2":yl2,"id2":ids2})
    df.to_csv("plotting.csv")

    

    