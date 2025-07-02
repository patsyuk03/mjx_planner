import numpy as np

num_batch = 13

num_dof = 12

theta_init = np.tile(np.zeros(12), (num_batch, 1))
thetadot_init = np.tile(np.zeros(12), (num_batch, 1))
thetaddot_init = np.tile(np.zeros(12), (num_batch, 1))
thetadot_fin = np.zeros((num_batch, num_dof))
thetaddot_fin = np.zeros((num_batch, num_dof))

state_term = np.hstack((theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin))
state_term = np.asarray(state_term)

print(state_term[:, ::num_dof//2].shape)
print(state_term[:, num_dof//2::num_dof//2].shape)