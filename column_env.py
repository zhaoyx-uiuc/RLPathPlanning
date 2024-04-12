import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Any
#from stable_baselines3 import DQN, PPO
from IPython import display
import copy
import torch
class RandomObstaclesEnv(gym.Env):
    def __init__(self, obstacle_x_count_range=(1,5), space_size=(30, 30), obs_dist=5, x_step_range=(6,9),y_step_range=(7,10),half_obstacle_size_range=(1,2),obs_format = 'array'):
        #space_size is the size of space array, so the first term is the number of rows (length on y), the second term is the number of columns (length on x)
        #x_step is the distance between the centers of two columns
        super(RandomObstaclesEnv, self).__init__()
        self.step_count=0
        self.obstacle_x_count_range = obstacle_x_count_range
        self.action_dict={'up':0,'down':1,'left':2,'right':3,'vertical':4, 'stop':5}
        self.x_length,self.y_length=space_size
        self.space_size=space_size
        self.obs_dist= obs_dist
        self.x_step_range=x_step_range
        self.y_step_range=y_step_range
        self.half_obstacle_size_range=half_obstacle_size_range
        self.obstacles = []
        self.rendered_frames = []
        self.state_recorder = []
        self.action_recorder = []
        self.action_space = gym.spaces.Discrete(6)  # Example action space (up, down, left, right,vertical)
        self.obs_format = 'array'
        if self.obs_format == 'array':
            self.observation_space = gym.spaces.Box(low=0,high=max(self.space_size[0],self.space_size[1]),shape=(4,space_size[0],space_size[1]))
        elif self.obs_format == 'dict':
            self.observation_space = gym.spaces.Dict(
                {
                'position': gym.spaces.MultiDiscrete([space_size[0],space_size[1]]),
                'map': gym.spaces.MultiBinary([3,space_size[0], space_size[1]])#gym.spaces.Dict({'map':map_space})  # Example observation space
                }
            )
        
        self.seed = 10000
        self.reset()
    
    
    
        a=1
        
    def reset(self): 
        np.random.seed(self.seed)
        self.obstacles = []
        self.half_obstacle_size=np.random.randint(self.half_obstacle_size_range[0],self.half_obstacle_size_range[1]) #half of the column size
        self.obstacle_size=2*self.half_obstacle_size
        obstacle_x_count=4#np.random.randint(self.obstacle_x_count_range[0],self.obstacle_x_count_range[1]) #the number verticle lines of columns
        x_step=np.random.randint(self.x_step_range[0],self.x_step_range[1]) #interval on x direction
        y_step=np.random.randint(self.y_step_range[0],self.y_step_range[1])
        x_edge=int((self.x_length-x_step*(obstacle_x_count-1)-self.obstacle_size)/2) #the distance from the most left surface to the left edge of the space
        obstacle_y_count=(self.y_length-self.obstacle_size)//(y_step)+1
        y_edge=int((self.y_length-y_step*(obstacle_y_count-1)-self.obstacle_size)/2) #distance from most up surface to the top edge of space
        for obstacle_x_idx in range(obstacle_x_count):
            
            obstacle_x=x_step*(obstacle_x_idx)+x_edge+self.half_obstacle_size #the x coordinate of the center of the obstacle_x_idx column

            obstacle_y=y_edge+self.half_obstacle_size            
            for obstacle_y_idx in range(obstacle_y_count):
                self.obstacles.append((obstacle_x, obstacle_y))
                obstacle_y+=y_step
        self.env,self.column_outline,self.surface_list=self.build_env()
        self.uninspected_surface_list=self.surface_list
        self.map=np.zeros((4,self.space_size[0], self.space_size[1]),dtype='float32')
        self.map[0,:,:]=self.pos_to_known_map(np.array([0,0]))
        self.map[1,:,:]=self.update_free(self.map[0,:,:])   
        self.position = np.array([0,0])  #x,y (or column, row)

        self.traj_record=np.zeros((self.space_size[0], self.space_size[1]),dtype=bool)
        self.traj_record[0,0]=1
        # Reset the environment state here if needed
        
        if self.obs_format == 'array':
            self.state = self.map.copy() #torch.tensor(self.map.copy())
            
        elif self.obs_format == 'dict':
            self.state = {'position':self.position, 'map':self.map[0:3]}

        #The first term is map, which is a w*h*3 array. 3 means known(1)/unknown(0), free/column, column inspected/not

        return self.state,{}
    
    def build_env(self):
        image = np.ones((self.y_length, self.x_length), dtype=bool)
        column_outline=np.ones((self.space_size[0], self.space_size[1]), dtype=bool)
        surface_list=[[],[],[],[]]
        for obstacle in self.obstacles:
            #take left top point as original point, x-axis is horizontal, y-axis is vertical
            x, y = obstacle
            image[y-self.half_obstacle_size:y+self.half_obstacle_size+1, x-self.half_obstacle_size:x+self.half_obstacle_size+1] = False  # Color obstacles in red
            column_outline[y-self.half_obstacle_size,x-self.half_obstacle_size:x+self.half_obstacle_size]=False #up
            column_outline[y+self.half_obstacle_size-1,x-self.half_obstacle_size:x+self.half_obstacle_size]=False #down            
            column_outline[y-self.half_obstacle_size:y+self.half_obstacle_size,x-self.half_obstacle_size]=False #left
            column_outline[y-self.half_obstacle_size:y+self.half_obstacle_size,x+self.half_obstacle_size-1]=False #right

            #surface_list records the endpoints coordinates (x,y) of the surface 
            surface_list[0].append(np.array([[x-self.half_obstacle_size,y-self.half_obstacle_size],[x+self.half_obstacle_size,y-self.half_obstacle_size]])) #up
            surface_list[1].append(np.array([[x-self.half_obstacle_size,y+self.half_obstacle_size],[x+self.half_obstacle_size,y+self.half_obstacle_size]])) #down
            surface_list[2].append(np.array([[x-self.half_obstacle_size,y-self.half_obstacle_size],[x-self.half_obstacle_size,y+self.half_obstacle_size]])) #left
            surface_list[3].append(np.array([[x+self.half_obstacle_size,y-self.half_obstacle_size],[x+self.half_obstacle_size,y+self.half_obstacle_size]])) #right

            #the four lists in surface_list represent the four types of faces facing left, right, up, down. Each surface is represented by a 2*2 array
        return image, column_outline, surface_list
    
    def pos_to_known_map(self,position):
        #input the position, output the known map from this position.
        result_array=np.zeros_like(self.map[0]).astype(bool)
        result_array[max(0,position[1]-self.obs_dist):position[1]+self.obs_dist+1,max(0,position[0]-self.obs_dist):position[0]+self.obs_dist+1]=1
        return result_array    
    # This is the old version of updating known map. Distance is calculated here to determine if a pixel is known. 
    # This cause problem when initializing the path, and also increase computational expense.
    '''
    def pos_to_known_map(self,position):
        #input the position, output the known map from this position.
        row_indices, col_indices = np.ogrid[:self.y_length, :self.x_length]
        # Calculate distances using vectorized operations
        x,y=position
        distance_square = (row_indices - y)**2 + (col_indices - x)**2 #The distance of each pixel to position, 300*300
        # Create the array based on the distance threshold
        result_array = (distance_square <= self.obs_dist**2).astype(bool)
        return result_array
    '''
    def update_free(self,known_map):
        #use known map to generate free map. The cells of obstacle in the enrironment are known in known map, but not free in free map
        new_free_map=self.env & known_map.astype(bool)
        return new_free_map

    def update_inspected_column(self, position):
        #update the map representing the inspected columns
        inspected_column_map=np.array(self.map[2,:,:])
        new_left_list=[]
        new_right_list=[]
        new_up_list=[]
        new_down_list=[]

        already_inspected=False  
        # record if there is any surface already inspected. 
        # #As the for loop is from the most top surface to the bottom, if there is any surface already inspected, the sight towards the remaining surfaces should be obstacled.
        for up_surface in self.surface_list[0]:            
            xmin=up_surface[0,0]
            xmax=up_surface[1,0]
            y=up_surface[0,1]
            if y-self.obs_dist<=position[1]<=y and xmin<=position[0]<=xmax and already_inspected==False:
                inspected_column_map[y,xmin:xmax+1]=True
                already_inspected==True
            else:
                new_up_list.append(up_surface)
        self.surface_list[0] = new_up_list

        for down_surface in self.surface_list[1]:            
            xmin=down_surface[0,0]
            xmax=down_surface[1,0]
            y=down_surface[0,1]
            if y<=position[1]<=y+self.obs_dist and xmin<=position[0]<=xmax:
                inspected_column_map[y,xmin:xmax+1]=True
            else:
                new_down_list.append(down_surface)     
        self.surface_list[1] = new_down_list

        for left_surface in self.surface_list[2]:
            x=left_surface[0,0]
            ymin=left_surface[0,1]
            ymax=left_surface[1,1]
            if x-self.obs_dist<=position[0]<=x and ymin<=position[1]<=ymax:
                inspected_column_map[ymin:ymax+1,x]=True
            else:
                new_left_list.append(left_surface)
            self.surface_list[2] = new_left_list    

        for right_surface in self.surface_list[3]:            
            x=right_surface[0,0]
            ymin=right_surface[0,1]
            ymax=right_surface[1,1]
            if x<=position[0]<=x+self.obs_dist and ymin<=position[1]<=ymax:
                inspected_column_map[ymin:ymax+1,x]=True
            else:
                new_right_list.append(right_surface)
            self.surface_list[3] = new_right_list       
        return inspected_column_map
    def get_coverage(self):
        return np.count_nonzero(self.map[0])/self.map[0].size
        
    def get_uninspected_surfaces(self):
        return len(self.surface_list[0])+len(self.surface_list[1])+len(self.surface_list[2])+len(self.surface_list[3])   
    def step(self, action):
        self.state_recorder.append(self.state)
        self.action_recorder.append(action)
        self.step_count+=1 

        penalty = 0  
        #take left top point as original point, x-axis is horizontal, y-axis is vertical
        if action == self.action_dict['up'] and self.position[1]!=0 : #'up'
            delta_position=np.array([0,-1])
        elif action == self.action_dict['down'] and self.position[1]!= self.y_length-1:#'down'
            delta_position=np.array([0,1])
        elif action == self.action_dict['left'] and self.position[0]!=0:#'left': 
            delta_position=np.array([-1,0])
        elif action == self.action_dict['right'] and self.position[0]!= self.x_length-1: #'right'
            delta_position=np.array([1,0])
        
        elif action == self.action_dict['vertical'] or action == self.action_dict['stop']: #'vertical'
            delta_position=np.array([0,0])
        
        else: #if at border
            delta_position=np.array([0,0])
            penalty = -10

        new_position=(self.position+delta_position).astype(int)
        self.traj_record[new_position[1],new_position[0]]=1
        known_map=self.pos_to_known_map(new_position) #known map represents if a cell is inspected. True means known, False means not inspected      
        new_known_map=self.map[0,:,:].astype(bool) | known_map #update the step known map to global known map
        new_known_pixels=new_known_map.sum()-self.map[0,:,:].sum()
        free_map=self.update_free(new_known_map) #True means the cell is free, False means the cell is column   
        self.map[0,:,:]=np.array(new_known_map)
        self.map[1,:,:]=np.array(free_map)
        if action == self.action_dict['stop']:
            penalty = 1
        if action == self.action_dict['vertical']:
            inspected_surface_map=self.update_inspected_column(new_position)
            new_inspected_columns=inspected_surface_map.sum()-self.map[2,:,:].sum()
            self.map[2,:,:]=inspected_surface_map
        else:
            new_inspected_columns=0
        self.map[3,:,:]=np.zeros_like(self.map[3,:,:])
        self.map[3,0,0]=new_position[0]
        self.map[3,0,1]=new_position[1]
        self.position = new_position
        if self.obs_format == 'array':
            self.state = self.map.copy()#torch.tensor(self.map.copy())
        elif self.obs_format == 'dict':
            self.state={'position':new_position, 'map':self.map[0:3]}
        self.reward=-1.0+new_known_pixels+new_inspected_columns*10+penalty
        terminated=True if action == self.action_dict['stop'] else False
        truncated=True if self.step_count>10000000 else False
        info={}
        self.rendered_frames.append([self.traj_record.copy(),self.map[0].copy(),self.map[1].copy(),self.map[2].copy()])
        return self.state, self.reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            display.clear_output(wait=True)
            fig, axs = plt.subplots(1, 3)   
            axs[0].imshow(self.traj_record)
            axs[1].imshow(np.array(self.map[0,:,:]))
            axs[2].imshow(np.array(self.map[2,:,:]))
            plt.show()
        elif mode == 'gif':
            np.save('state.npy', np.array(self.state_recorder))
            np.save('action.npy', np.array(self.action_recorder))

            fig, axs = plt.subplots(2,2)
            ims = []
            for i in range(len(self.rendered_frames)):
                im_traj = axs[0][0].imshow(self.rendered_frames[i][0])
                im_known = axs[0][1].imshow(self.rendered_frames[i][1])
                im_free = axs[1][0].imshow(self.rendered_frames[i][2])
                im_inspected = axs[1][1].imshow(self.rendered_frames[i][3])
                
                ims.append([im_traj, im_known, im_free, im_inspected])

            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                            repeat_delay=1000)
            '''
            for i in range(len(self.rendered_frames)):

                im_traj = axs.imshow(self.rendered_frames[i][0].astype(int), animated=True)
                #im_map = axs[1].imshow(self.rendered_frames[i][1], animated=True)
                #im_inspected = axs[2].imshow(self.rendered_frames[i][2], animated=True)
                if i == 0:
                    axs.imshow(self.rendered_frames[i][0].astype(int))  # show an initial one first
                ims.append([im_traj])
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
            '''
            plt.show()
            ani.save("animation.gif", writer='imagemagick', fps=10)
        else:
            # Implement other rendering modes if needed
            pass

    def greedy(self):
        reward_list=[]
        for key,action in self.action_dict.items():
            dummy = copy.deepcopy(self)
            state, reward, terminated, truncated, info=dummy.step(action = action)
            reward_list.append(reward)
        
        return reward_list.index(max(reward_list))

    def move(self):
        
        for i in range(8):
            self.step(action=self.action_dict['right'])      

        for i in range(20):
            #self.step(action=self.action_dict['down'])
            #self.step(action=self.action_dict['down'])
            self.step(action=self.action_dict['down'])

            self.step(action=self.action_dict['vertical'])
        plt.imshow(self.column_outline)
        plt.show()

    def greedy_move(self):
        while True:
            
            action = self.greedy()
            if action == 5: break
            self.step(action = action)
            
        self.render(mode='gif')
            #print(self.state['position'], action)

    def cal_metric(self):
        print('coverage',self.get_coverage(),'uninspected_surfaces',self.get_uninspected_surfaces())
    
    def ymoveto(self,y):
        if self.position[1]<y:
            while self.position[1]!=y:
                self.step(1)
        elif self.position[1]>y:
            while self.position[1]!=y:
                self.step(0)

    def xmoveto(self,x):
        if self.position[0]<x:
            while self.position[0]!=x:
                self.step(3)
        elif self.position[0]>x:
            while self.position[0]!=x:
                self.step(2)
                
    def yxmoveto(self,xy):
        x,y = xy
        self.ymoveto(y)
        self.xmoveto(x)

    def xymoveto(self,xy):
        x,y = xy
        self.xmoveto(x)
        self.ymoveto(y)
                
    def initial_rule(self):
        #move out of the corner
        ######################################
        for i in range(self.obs_dist):
            self.step(action = 1) #down
            self.step(action = 3) #right


        self.step(action = 1)
        last_action = 1
        count = 0
        while self.get_coverage()!=1:
            count +=1
            #move towards down
            ######################################                
            if last_action == 1:
                bottom_left_corner_x = self.position[0]-self.obs_dist
                if self.map[0,self.space_size[0]-1,bottom_left_corner_x]==0 or self.position[0]==self.space_size[0]-1:
                    self.step(action = 1)
                    last_action = 1
                else:
                    right_border = self.position[0]+self.obs_dist
                    self.step(action = 3)
                    last_action = 3
                    last_vertical_move = 1
                    
            #move towards right
            ######################################
            if last_action == 3:
                if self.position[0]-self.obs_dist<=right_border and self.position[0]!=self.space_size[0]-1:
                    self.step(action = 3)
                    last_action = 3
                else:
                    if last_vertical_move ==1:
                        self.step(action = 0)
                        last_action = 0
                    else:
                        self.step(action = 1)
                        last_action = 1
                                  
            #move towards up
            ######################################
            if last_action == 0:
                top_left_corner_x = self.position[0]-self.obs_dist
                if self.map[0,0,top_left_corner_x]==0 or self.position[0]==self.space_size[0]-1:
                    self.step(action = 0)
                    last_action = 0
                else:
                    right_border = self.position[0]+self.obs_dist
                    self.step(action = 3)
                    last_action = 3
                    last_vertical_move = 0
        ############################################
        #Start to inspect columns 
        ############################################
        a = self.position
        b = self.surface_list
        #sort the obstacles to the sequence of being inspected
        obstacle_list = self.obstacles
        classified_dict={}
        for item in obstacle_list:
            key = item[0]
            if key not in classified_dict:
                classified_dict[key] = []
            classified_dict[key].append(item)
        classified_obstacle_list = list(classified_dict.values())
        classified_obstacle_list.reverse()
        #each sub list denote a line of columns, right lines first. In each line, column at top first.     
        
        #sort the list to the sequence of inspecting obstacles
        if self.position[1] == self.obs_dist: #if the end point is at right top corner
            for line_idx in range(len(classified_obstacle_list)):
                if line_idx%2 ==1:
                    classified_obstacle_list[line_idx].reverse()
        if self.position[1] == self.space_size[1]-1-self.obs_dist: #if the end point is at right bottom corner
            for line_idx in range(len(classified_obstacle_list)):
                if line_idx%2 ==0:
                    classified_obstacle_list[line_idx].reverse()

        else:
            print('Error: final position not correct')
            return 0
        #start to inspect
        dist_to_center = 1 + self.half_obstacle_size #distance to the surface is 1. Change this number if necessary
        
        for line_idx in range(len(classified_obstacle_list)):
            for item_idx in range(len(classified_obstacle_list[line_idx])):
                inspect_coordinates_1 = (classified_obstacle_list[line_idx][item_idx][0]+dist_to_center,classified_obstacle_list[line_idx][item_idx][1]) #xax, y
                inspect_coordinates_2 = (classified_obstacle_list[line_idx][item_idx][0],classified_obstacle_list[line_idx][item_idx][1]+dist_to_center) #x, ymax
                inspect_coordinates_3 = (classified_obstacle_list[line_idx][item_idx][0]-dist_to_center,classified_obstacle_list[line_idx][item_idx][1]) #xin, y
                inspect_coordinates_4 = (classified_obstacle_list[line_idx][item_idx][0],classified_obstacle_list[line_idx][item_idx][1]-dist_to_center) #x, ymin
                self.yxmoveto(inspect_coordinates_1)
                self.step(action = 4)
                self.yxmoveto(inspect_coordinates_2)
                self.step(action = 4)
                self.xymoveto(inspect_coordinates_3)      
                self.step(action = 4) 
                self.yxmoveto(inspect_coordinates_4)
                self.step(action = 4)
                self.xymoveto(inspect_coordinates_1)
            if line_idx!= len(classified_obstacle_list)-1:
                if self.position[1]>=self.space_size[1]-self.position[1]:
                    self.yxmoveto((classified_obstacle_list[line_idx+1][0][0]+dist_to_center,classified_obstacle_list[line_idx][-1][1]+dist_to_center))
                else:
                    self.yxmoveto((classified_obstacle_list[line_idx+1][0][0]+dist_to_center,classified_obstacle_list[line_idx][-1][1]-dist_to_center))
        self.step(action = 5)
        
        self.render(mode='gif')
            
#env = RandomObstaclesEnv() 
#env.greedy_move()
#env.initial_rule()
#env.move()
'''
env = RandomObstaclesEnv()  
model = DQN("CnnPolicy", env, buffer_size = 10000, verbose=1,policy_kwargs=dict(normalize_images=False),device='cpu')
para=model.get_parameters()
#model = PPO("CnnPolicy", env, verbose=1,policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=40000, log_interval=10, progress_bar=True)
model.save("dqn_cartpole")


obs, info = env.reset()
terminated=False
action_list=[]

while terminated == False:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    action_list.append(action)
    #print(action)

print(action_list)
'''
#plt.imshow(obs[:,:,0])
#plt.savefig('output_image.png')


