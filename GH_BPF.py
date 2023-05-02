import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#import scipy.io

# Setting random seeds for reproducibility
np.random.seed(21)

# State equations function
def state_eqns(state, noise, F, G):
    mu,cov = noise
    return np.matmul(F,state)+np.matmul(np.random.multivariate_normal(mu,cov,1),G.T)

# Measurements equations function
def measurement_eqns(state, noise):

    mu,cov = noise
    return np.arctan(state[2]/state[0])+np.random.normal(mu,np.sqrt(cov))

# Prediction function
def predict(particles_prev, state_eqns, noise, F, G):
   
    num_particles = particles_prev.shape[0]
    particles_next = np.zeros_like(particles_prev)
    

    for i in range(num_particles):
   
        particle = state_eqns(particles_prev[i],noise, F, G)
        particles_next[i] = particle
    
    return particles_next

# Update function

def update(particles, weights, measurement, noise):

    mu, cov = noise
    num_particles = particles.shape[0]
    weights_updated = np.zeros_like(weights)
    
    for i in range(num_particles):

        expected_measurement = np.arctan(particles[i,2]/particles[i,0])
        likelihood =np.exp(-0.5 * np.power(measurement - expected_measurement,2) / cov)
        weights_updated[i] = likelihood
        
    # weights_updated += 1e-10      # avoid round-off to zero
    weights_sum=np.sum(weights_updated)
    weights_updated /= weights_sum
    return weights_updated


def resample(particles_trajectories, weights):

    num_particles = particles_trajectories.shape[0]
    aux_trajectories = np.zeros_like(particles_trajectories)

    c = np.cumsum(weights)
    c = np.insert(c, 0, 0)
    c[-1] = 1.0

    for i in range(num_particles):

        u = np.random.uniform()
        indices = np.where(c <= u)[0]  # find all indices where z <= x
        index = np.max(indices)  # largest index determines sample chosen
        
        if index == num_particles:
            index = num_particles - 1

        aux_trajectories[i,:,:] = particles_trajectories[index,:,:]
    
    

    return aux_trajectories

# Run the bootstrap particle filter
state_dimension=4       # state dimension
time = 100              # time horizon
num_particles = 5000    # number of particles

# Initialize weights
weights=(1/num_particles)*np.ones(num_particles)

# Initialize true states and measurements
F = np.array([[1,1, 0, 0],
              [0,1,0,0],
              [0,0,1,1],
              [0,0,0,1]])
G = np.array([[0.5, 0],
              [1, 0],
              [0,0.5],
              [0, 1]])
              
# Initialize state and measurement vector
mu_0 = np.array([0,0,0.4,-0.05])
cov_0 = np.diag([0.25,25e-6,0.09,1e-4])
true_states_0 = np.array([-0.05, 0.001, 0.7, -0.055])


mu_u = np.array([0,0])
cov_u = np.diag([1e-6,1e-6])
noise_u = [mu_u,cov_u]

mu_v = 0
cov_v = 25e-6
noise_v = [mu_v,cov_v]


true_states = np.empty((time+1, state_dimension))
measurements = np.empty((time, 1))
true_states[0] = true_states_0
    
for t in range(1, time+1):
    true_states[t] = state_eqns(true_states[t-1], noise_u, F,G)
    measurements[t-1] = measurement_eqns(true_states[t], noise_v)

#true_states = np.array(scipy.io.loadmat('xvec.mat')['xvec']).T
#measurements = np.array(scipy.io.loadmat('yvec.mat')['yvec'])

# Initialize particles

particles_0 = np.empty((num_particles, state_dimension))
particles_0 = np.random.multivariate_normal(mu_0,cov_0,num_particles)

particles = particles_0


# Initialize particles trajectories
particles_trajectories = np.empty((num_particles, 4, time+1))
particles_trajectories[:,:,0] = particles_0


 # Loop over time steps
for t in range(time):
        


        # Prediction step
        particles_pred = predict(particles, state_eqns, noise_u, F, G)

        # Trajectories augmentation 
        particles_trajectories[:,:,t+1] = particles_pred

        # Update step
        weights_updated =update(particles_pred, weights, measurements[t], noise_v)

        #
        particles = particles_pred

        # Resampling step
        particles_trajectories_updated = resample(particles_trajectories[:,:,0:t+2], weights_updated)
 
        # Updating trajectories
        particles_trajectories[:,:,0:t+2] = particles_trajectories_updated
        
        # Reseting weights
        weights = (1/num_particles)*np.ones(num_particles)

# mean of trajectories
mean_trajectories=np.mean(particles_trajectories,axis=0).T


plt.scatter(mean_trajectories[:,0], mean_trajectories[:,2], facecolors='none', edgecolors='b', label='estimated trajectory')
plt.scatter(true_states[:,0], true_states[:,2], edgecolors='r', facecolors='none', label='actual trajectory')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim([-0.5, 0.3])
plt.ylim([-6, 2])
plt.title('Bootstrap Particle Filter (BPF)')
plt.legend()
plt.show()
plt.savefig('plot_BPF.png')