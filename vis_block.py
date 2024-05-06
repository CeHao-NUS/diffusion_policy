import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# load store

file_name = 'store.npy'
# file_name = 'store_wo.npy'

store = np.load(file_name, allow_pickle=True).item()
actions = store['action_pred'].squeeze()
action = store['action'].squeeze()
obss = store['obs'].squeeze()


WORKSPACE_BOUNDS = np.array(((0.15, -0.5), (0.7, 0.5)))

'''
OrderedDict([
('block_translation', (0.40976190152367453, -0.13544302678898584)), ('block_orientation', array([1.85069921])), [0:2] [2]
('block2_translation', (0.30783815449058405, -0.2651577247443393)), ('block2_orientation', array([2.88417522])), [3:5] [5]
('effector_translation', array([ 0.31309738, -0.37432996])),  [6:8] 
('effector_target_translation', array([ 0.31404954, -0.37327074])),  [8:10]
('target_translation', array([0.51980894, 0.20282992])), ('target_orientation', array([0.07968668])),   [10:12] [12]
('target2_translation', array([0.28418235, 0.19575233])), ('target2_orientation', array([0.01365312]))]) [13:15] [15]

'''

block_translation = obss[:, -1, 0:2]
block2_translation = obss[:, -1, 3:5]
effector_translation = obss[:, -1, 6:8]
effector_target_translation = obss[:, -1, 8:10]
target_translation = obss[:, -1, 10:12]
target2_translation = obss[:, -1, 13:15]


points = np.array([block_translation, block2_translation, target_translation, target2_translation, effector_translation, effector_target_translation])
labels = ['block0', 'block1', 'target0', 'target1', 'effector', 'effector_target']
colors = [ 'red', 'green', 'green', 'red', 'blue', 'black']
dots = ['o', 'o', 's', 's', '*', '*']



# Set up the figure and axis for plotting
fig, ax = plt.subplots()

# equal
ax.set_aspect('equal', 'box')

# base ee trajectory
cmap = plt.cm.viridis
color_idx = np.linspace(0, 1, effector_translation.shape[0])
for i in range(len(effector_translation) - 1):
    plt.plot(effector_translation[i:i+2, 0], effector_translation[i:i+2, 1], color=cmap(color_idx[i]))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))


line, = ax.plot([], [], lw=2)
# Initialize a list of plot objects for points, one for each set
point_plots = [ax.plot([], [], markersize=10, marker=dots[i], color=colors[i], label=labels[i])[0] for i in range(points.shape[0])]
text = ax.text(0.5, 0.9, '', transform=ax.transAxes, color='black')

plt.legend(loc='lower right')

# Initialize the plot frame
def init():
    ax.set_xlim(WORKSPACE_BOUNDS[0,0], WORKSPACE_BOUNDS[1,0])
    ax.set_xlim(0, 1)
    ax.set_ylim(WORKSPACE_BOUNDS[0,1], WORKSPACE_BOUNDS[1,1])
    return [line] + point_plots + [text]

# Update the plot for each frame
def update(frame):
    # add + current 
    line.set_data(actions[frame, :, 0] + effector_translation[frame,0], actions[frame, :, 1]+ effector_translation[frame,1])
    for i, point_plot in enumerate(point_plots):
        point_plot.set_data(points[i, frame, 0], points[i, frame, 1])

    text.set_text(f'Step {frame} \nAction: {np.round(action[frame]*1000, 2)}e-3')
    return [line] + point_plots + [text]

# Create the animation object
ani = FuncAnimation(fig, update, frames=range(actions.shape[0]), init_func=init, blit=True, interval=60)

# Save the animation
ani.save('data_animation.mp4', writer='ffmpeg')

# If you want to display the plot as well, uncomment the following line
# plt.show()
