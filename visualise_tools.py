import numpy as np
import gym
from utils import *
import os
from matplotlib import patches
import matplotlib.pyplot as plt
from visualise_tools import *

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

#inverse rotation mapping
ori_map = {}
ori_map[(0,-1)] = {TL : (1,0) , TR : (-1,0)}
ori_map[(1,0)] = {TL : (0,1), TR : (0,-1)}
ori_map[(0,1)] = {TL : (-1,0), TR : (1,0)}
ori_map[(-1,0)] = {TL : (0,-1), TR : (0,1)} 

#rotation mapping
ori_map_forward = {}
ori_map_forward[(0,-1)] = {TL : (-1,0) , TR : (1,0)}
ori_map_forward[(1,0)] = {TL : (0,-1), TR : (0,1)}
ori_map_forward[(0,1)] = {TL : (1,0), TR : (-1,0)}
ori_map_forward[(-1,0)] = {TL : (0,1), TR : (0,-1)} 


#computes control seq
def forward(state, u):
    KS = state[2]
    DS = state[3]
    cood = state[0]
    ori = state[1]
    
    if(u == MF):
        cood =tuple(np.add(cood, ori))
    elif(u == TL or u == TR):
        ori = ori_map_forward[ori][u] 
    elif(u == PK):
        KS = 1
        
    else:
        DS = 1
        KS = 0
    return (cood,ori,KS,DS)


#function for drawing arrows for policy and writing numbers for values
def draw_policy_value(policy, ori, KS, DS, key, door,filename, env_path):
    env,info = load_env(env_path)
    states = policy.keys()
    states = [i for i in states if i[1] == ori and i[2] ==KS and i[3] ==DS]
    
    fig, ax = plt.subplots(1,figsize= (6,6))
    img = env.render('rgb_array', tile_size=32, highlight = False)
    x , y = env.agent_pos[0],env.agent_pos[1]
    rect = patches.Rectangle((32*x,32*y),32,32,linewidth=0,edgecolor='k',facecolor='k', fill = True)
    ax.add_patch(rect)
    if(KS or DS):
        if (KS):
            x, y = key[0], key [1]
        else:
            x, y = door[0], door[1]

        
        rect = patches.Rectangle((32*x,32*y),32,32,linewidth=0,edgecolor='k',facecolor='k', fill = True)
        ax.add_patch(rect)
    plt.imshow(img)

    for i in range(len(states)):

        x = states[i][0][0]
        y = states[i][0][1]
        val = policy[states[i]]['val']
        plt.text((x+.33)*32,(y+.65)*32, "{}".format(val), fontsize=22,color='w')
    plt.axis('off')
    if not os.path.exists('./plots/{}'.format(filename)):
        os.makedirs('./plots/{}'.format(filename))

    plt.savefig("./plots/{}/val_{}_{}_{}.png".format(filename, str(ori), KS, DS),bbox_inches='tight',pad_inches = 0)
    plt.close()
    
    fig, ax = plt.subplots(1,figsize= (6,6))
    
    img = env.render('rgb_array', tile_size=32, highlight = False)
    x , y = env.agent_pos[0],env.agent_pos[1]
    rect = patches.Rectangle((32*x,32*y),32,32,linewidth=0,edgecolor='k',facecolor='k', fill = True)
    ax.add_patch(rect)

    if(KS or DS):
        if (KS):
            x, y = key[0], key [1]
        else:
            x, y = door[0], door[1]

        
        rect = patches.Rectangle((32*x,32*y),32,32,linewidth=0,edgecolor='k',facecolor='k', fill = True)
        ax.add_patch(rect)

    plt.imshow(img)

    
    for i in range(len(states)):
        x = states[i][0][0]
        y = states[i][0][1]

        seq = policy[states[i]]['policy']
        for control in seq:

            if(control == 1):
                plt.plot((x+.5)*32,(y+.5)*32,marker=r'$\circlearrowleft$',ms=30,color = 'orange',markeredgewidth=0.7)
            elif(control == 2):
                plt.plot((x+.5)*32,(y+.5)*32,marker=r'$\circlearrowright$',ms=30,color = 'c',markeredgewidth = .7)

            elif(control == 0 or control == 3 or control == 4):
                if(control == 0):
                    color = 'r'
                elif(control == 3):
                    color = 'y'
                else:
                    color = 'g'
                if(ori == (0,1)):
                    plt.arrow((x+.5)*32,(y+.25)*32,0 ,12,width = 1,head_width=7, head_length=5, fc=color, ec=color)
                elif(ori == (1,0)):
                    plt.arrow((x+.25)*32,(y+.5)*32,12 , 0,width = 1,head_width=7, head_length=5, fc=color, ec=color)

                elif(ori == (0,-1)):
                    plt.arrow((x+.5)*32,(y+.75)*32,0 , -12,width = 1,head_width= 7, head_length=5, fc=color, ec=color)

                elif(ori == (-1,0)):
                    plt.arrow((x+.75)*32,(y+.5)*32,-12 , 0,width = 1,head_width=7, head_length=5, fc=color, ec=color)


    plt.axis('off')
    plt.savefig("./plots/{}/pol_{}_{}_{}.png".format(filename, str(ori), KS, DS),bbox_inches='tight',pad_inches = 0)
    plt.close()

#for plotting value vs T for special states
def value_T_plotter(key,dict_,filename, loc,itr):
    if not os.path.exists('./val_plots/{}'.format(filename)):
        os.makedirs('./val_plots/{}'.format(filename))
    val = dict_[key]
    y_axis = [20 for i in range(val[0][0])]
    for j in range(1,len(val)):
        y_axis += [val[j-1][1] for i in range(val[j][0]-val[j-1][0])]
    y_axis += [val[-1][1] for i in range(itr-val[-1][0])] 
    plt.figure()
    plt.plot(y_axis)
    plt.grid(True)
    plt.xlabel("Exploration steps, T",size = 16)
    plt.ylabel("Value",size = 16)
    plt.title("Value Vs T at {} : {}".format(loc,key),size = 18)
    plt.savefig("./val_plots/{}/{}_{}.png".format(filename,loc,key))
    plt.close()

#mapping for controls
def generate_seq(control):
    seq = []
    for i in control:
        if(i == 0):
            seq.append('MF')
        elif(i == 1):
            seq.append('TL')
        elif(i == 2):
            seq.append('TR')
        elif(i == 3):
            seq.append('PK')
        elif(i == 4):
            seq.append('UD')

    return seq