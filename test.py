from column_env import RandomObstaclesEnv
env = RandomObstaclesEnv()
file_path = 'action_file'

# Initialize an empty 2D list
action_record = []

# Open the file for reading
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Split the line into elements (assuming space-separated elements)
        elements = line.strip().split()
        
        # Append the elements to the 2D list
        action_record.append(elements)
last_action = action_record[7357]

for action in last_action:
    env.rendered_frames.append([env.traj_record.copy(),env.map[0].copy(),env.map[1].copy(),env.map[2].copy()])
    env.step(action = int(action))
    
'''        
while True:
    env.rendered_frames.append([env.traj_record.copy(),env.map[0].copy(),env.map[1].copy(),env.map[2].copy()])
    action = env.greedy()
    if action == 5: break
    env.step(action = action)
'''   
env.render(mode='gif')

    