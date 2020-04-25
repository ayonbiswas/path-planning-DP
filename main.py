#!/usr/bin/env python
# coding: utf-8
import numpy as np
import gym
from utils import *
import os
from matplotlib import patches
import matplotlib.pyplot as plt
from visualise_tools import *

env_list = [
        "doorkey-5x5-normal.env",
        "doorkey-6x6-normal.env",
        "doorkey-8x8-normal.env",
        
        "doorkey-6x6-direct.env",
        "doorkey-8x8-direct.env",
        
        "doorkey-6x6-shortcut.env",
        "doorkey-8x8-shortcut.env"]

#computes policy for a given state and returns updated policy 
def compute_policy(state, policy,empty, door, goal, key, door_lock_state, key_keep_down_state, goal_nn):
    
    cost_fwd = 1
    cost_rot = 1
    door_cost = 1
    key_cost = 1
    
    KS = state[2]
    DS = state[3]
    cood = state[0]
    ori = state[1]
    val = policy[state]['val']
    lst = []
    if(state in door_lock_state):
        temp = (cood, ori, 1, 0)
        if(temp in policy ):
            if(val + door_cost < policy[temp]['val']):

                policy[temp] = {'val' : val+door_cost,'policy': [UD] }
                lst.append(temp)

            elif(val+door_cost == policy[temp]['val']):
                policy[temp]['policy'].append(UD)
                lst.append(temp)
        else:
            policy[temp] = {'val' : val+door_cost,'policy': [UD] }
        
            lst.append(temp)
        
    elif state in key_keep_down_state:
        temp = (cood, ori, 0, 0)
        if(temp in policy ):
            if(val + key_cost < policy[temp]['val']):

                policy[temp] = {'val' : val+key_cost,'policy': [PK] }
                lst.append(temp)

            elif(val+key_cost == policy[temp]['val']):
                policy[temp]['policy'].append(PK)
                lst.append(temp)
        else:
            policy[temp] = {'val' : val+key_cost,'policy': [PK] }
        

            lst.append(temp)
    nd = tuple(np.subtract(cood,ori))

    if(nd in empty or (nd == door and DS == 1 and KS == 0) or (nd == key and DS == 0 and KS == 1)):
        temp0  = (nd,ori,KS, DS)
        if(temp0 in policy ):
            if(val + cost_fwd < policy[temp0]['val']):

                policy[temp0] = {'val' : val+cost_fwd,'policy': [MF] }
                lst.append(temp0)

            elif(val+cost_fwd == policy[temp0]['val']):
                policy[temp0]['policy'].append(MF)
                lst.append(temp0)
        else:
            policy[temp0] = {'val' : val+cost_fwd,'policy': [MF] }
            lst.append(temp0)

    temp1 = (cood,ori_map[ori][TL],KS, DS)
    
    if(temp1 in policy ):
        if(val + cost_rot < policy[temp1]['val']):

            policy[temp1] = {'val' : val+cost_rot,'policy': [TL] }
            lst.append(temp1)

        elif(val+cost_rot == policy[temp1]['val']):
            policy[temp1]['policy'].append(TL)
            lst.append(temp1)
    else:
        policy[temp1] = {'val' : val+cost_rot,'policy': [TL] }
        lst.append(temp1)
        
    temp2 = (cood,ori_map[ori][TR],KS, DS)
    if(temp2 in policy ):
        if(val + cost_rot < policy[temp2]['val']):

            policy[temp2] = {'val' : val+cost_rot,'policy': [TR] }
            lst.append(temp2)

        elif(val+cost_rot == policy[temp2]['val']):
            policy[temp2]['policy'].append(TR)
            lst.append(temp2)
    else:
        policy[temp2] = {'val' : val+cost_rot,'policy': [TR] }

        lst.append(temp2)
        

    return policy , lst 


# takes a state, generates previous state and their policy and value for transitioning to this state
def inv_trans_func(state, policy,empty, door, goal, key, door_lock_state, key_keep_down_state,goal_nn):
    KS = state[2]
    DS = state[3]
    cood = state[0]
    ori = state[1]
    
    lst = []
    if(state[1] == '_'):
        lst = [((cood[0]+1,cood[1]),(-1,0),KS,DS),((cood[0]-1,cood[1]),(+1,0),KS,DS),
               ((cood[0],cood[1]+1),(0,-1),KS,DS),((cood[0],cood[1]-1),(0,1),KS,DS)]
        if(DS):
            lst = [x for x in lst if x[0] in empty or x[0] == door]
        elif(KS):
            lst = [x for x in lst if x[0] in empty or x[0] == key]
        else:
            lst = [x for x in lst if x[0] in empty]
                   
        for i in lst:
            policy[i] = {'val' : 1,'policy': [MF]}

    else:

        policy, lst = compute_policy(state, policy,empty, door, goal, key, door_lock_state, key_keep_down_state, goal_nn)
        
        
    return policy, lst

#returns optimal control seq
def doorkey_problem(env):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    #environment encoded matrix
    env_matrix = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0]
    
    #get empty, door, goal,key cell locations
    goal_ = np.where(env_matrix == 8)
    goal = (goal_[0][0],goal_[1][0])
    key_ = np.where(env_matrix == 5)
    key = (key_[0][0],key_[1][0])
    door_ = np.where(env_matrix == 4)
    door = (door_[0][0],door_[1][0])
    empty_ = np.where(env_matrix == 1)
    empty = list(zip(empty_[0],empty_[1]))
    
    #special states
    key_keep_down_state = [((key[0]+1,key[1]),(-1,0),1,0),((key[0]-1,key[1]),(+1,0),1,0),
                           ((key[0],key[1]+1),(0,-1),1,0),((key[0],key[1]-1),(0,1),1,0)]
    key_keep_down_state = [x for x in key_keep_down_state if x[0] in empty]
    
    door_lock_state = [((door[0]-1,door[1]),(1,0),0,1)]     
    
    goal_nn = [((goal[0]+1,goal[1]),(-1,0)),((goal[0]-1,goal[1]),(+1,0)),
               ((goal[0],goal[1]+1),(0,-1)),((goal[0],goal[1]-1),(0,1))]
    goal_nn = [x for x in goal_nn if x[0] in empty]
    
    goal_value_vs_T = {}
    for i in goal_nn:
        goal_value_vs_T[i[0]] = []
    
    door_value_vs_T = {door_lock_state[0][0]:[]}
    key_value_vs_T = {}
    for i in key_keep_down_state:
        key_value_vs_T[i[0]] = []    
    
    policy = {}
    frontier = [(goal,'_',0,0),(goal,'_',0,1),(goal,'_',1,0)]
    itr = 0
    T = env_matrix.shape[0]*env_matrix.shape[1]*16
    while True and itr < T:
        next_frontier = []
        while frontier:
            itr+=1
            x = frontier.pop(0)
            policy, lst= inv_trans_func(x,policy,empty, door, goal, key, door_lock_state, key_keep_down_state, goal_nn)
            next_frontier+=lst
            #tracking value of special states

            if door_lock_state[0][0] == x[0]:
                  door_value_vs_T [door_lock_state[0][0]] .append((itr, policy[x]['val']))

            for i in  key_keep_down_state:
                if( i[0] == x[0]):
                    key_value_vs_T[i[0]].append((itr, policy[x]['val']))
            

            for i in goal_nn:
                if( i[0] == x[0]):
                    goal_value_vs_T[i[0]].append((itr, policy[x]['val']))


        frontier= list(set(next_frontier))
    
        if(not frontier):
            break
    
    #generating control sequence from policy starting from the robot initial position
    next_state = (tuple(env.agent_pos),tuple(env.dir_vec),0,0) 
    
    control = []
    
    while (next_state[0] != goal ):
        u = policy[next_state]['policy'][0]
        control.append(u)
        next_state = forward(next_state, u)
    
    return policy, control, door_value_vs_T, goal_value_vs_T, key_value_vs_T ,itr, door, key

#computes policy, genrates plots for all envs
def main():
    for env in env_list:
        env_path = './envs/{}'.format(env)
        env, info = load_env(env_path) # load an environment
        policy, seq,door_value_vs_T, goal_value_vs_T, key_value_vs_T,itr_ ,door, key = doorkey_problem(env) # find the optimal action sequence
        env_name = env_path.split("/")[-1].split('.')[0]
        draw_gif_from_seq(seq, env, path='./gif/{}.gif'.format(env_name)) # draw a GIF & save

        door_val_keys = list(door_value_vs_T.keys())
        key_val_keys =list(key_value_vs_T.keys())
        goal_val_keys = list(goal_value_vs_T.keys())

        #plotting value vs T
        for j in door_val_keys:
            value_T_plotter(j,door_value_vs_T,env_name,'door',itr_)
        for j in key_val_keys:
            value_T_plotter(j,key_value_vs_T,env_name,'key',itr_)

        for j in goal_val_keys:
            value_T_plotter(j,goal_value_vs_T,env_name,'goal',itr_)

        #optimal seq
        optimal_seq = generate_seq(seq)
        print(optimal_seq)

        #generate policy and value plots
        ori_list  = list(ori_map.keys())
        for i in range(4):
            draw_policy_value(policy, ori_list[i], 0, 1,key, door, env_name, env_path)

        for i in range(4):
            draw_policy_value(policy, ori_list[i], 0, 0,key, door, env_name, env_path)

        for i in range(4):
            draw_policy_value(policy, ori_list[i], 1, 0,key, door, env_name, env_path)

if __name__ == '__main__':
    main()

