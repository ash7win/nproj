#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

from math import pi

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import os
data=pd.read_csv('data.csv')
data


# In[17]:


class Recommendation():
    
    def __init__(self, name):
        self.player_name = name
    
    def PlayerList(self):    
        data = pd.read_csv('data.csv')  
        data = data[data['Overall'] > 70] # Lower Overall
        attributes = ['Name','Nationality','Club','Age','Position','Overall','Potential','Preferred Foot','Value']
        abilities = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys',
                 'Dribbling','Curve','FKAccuracy','LongPassing','BallControl',
                 'Acceleration','SprintSpeed','Agility','Reactions','Balance',
                 'ShotPower','Jumping','Stamina','Strength','LongShots',
                 'Aggression','Interceptions','Positioning','Vision','Penalties',
                 'Composure','Marking','StandingTackle','SlidingTackle',
                 'GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']
        
        # Unit directional vector
        AbilitiesData = data[abilities]
        vec_length = np.sqrt(np.square(AbilitiesData).sum(axis=1))
        mat_abt = AbilitiesData.values.reshape(AbilitiesData.shape)
        
        for i in np.arange(AbilitiesData.shape[0]):
                mat_abt[i] = mat_abt[i,:]/vec_length[i]
                
        df_norm = pd.DataFrame(mat_abt, columns=abilities) 
    
        # Inner Product
        compared_player = df_norm[data['Name'] == self.player_name].iloc[0]
        data['Inner Product'] = df_norm.dot(compared_player)
        
        threshold_idp = 0.990 
        lower_potential = 88 # High potential
        substitutes = data[(data['Inner Product'] >= threshold_idp) & (data['Potential'] >= lower_potential)]
        
        if substitutes.shape[0] <= 1:
            print('There is no recommendation.')
            
        else:    
            substitutes = substitutes.sort_values(by=['Inner Product'], ascending=False)
            
            # Maximum of Player Recommendations = 5 players
            if substitutes.shape[0] > 6:
                substitutes = substitutes[0:6]
                
            substitutes = substitutes[attributes]
            substitutes.reset_index(drop=True)
            
            # Save the Scout list
            substitutes.to_csv('./scout_list.csv', index=False)
            
            standard_player = data[abilities][data.Name == self.player_name]
            
            for player_list in substitutes['Name'][1:]:
                
                add = data[abilities][data.Name == player_list]
                standard_player = standard_player.append([add])

            player_name = substitutes['Name'].values
            
            return standard_player, abilities, player_name
            


# In[18]:


def RadorChart(graph, abilities, player_name):
    len1 = graph.shape[0]
    len2 = graph.shape[1]
    temp = graph.values.reshape((len1, len2))
    
    tmp = pd.DataFrame(temp, columns = abilities)
    Attributes =list(tmp)
    AttNo = len(Attributes)
    
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True)
    
    colors = ['green', 'blue', 'red', 'black', 'gold', 'orange', 'lightskyblue', 'pink']
    
    for i in range(len1):
        values = tmp.iloc[i].tolist() #
        values += values [:1]
    
        angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
        angles += angles [:1]
        
        plt.xticks(angles[:-1],Attributes)
        ax.plot(angles, values, color = colors[i])
        ax.fill(angles, values, colors[i], alpha=0.1)
        plt.figtext(0.8, 0.18-0.022*i, player_name[i], color = colors[i], fontsize=12)
    
    plt.show()
    plt.savefig('RadarChart.png')


# In[21]:


Scouter = Recommendation('M. Verratti')
standard_player, abilities, player_name = Scouter.PlayerList()
RadorChart(standard_player, abilities, player_name)


# In[22]:


data[data['Age']>40]


# In[ ]:




