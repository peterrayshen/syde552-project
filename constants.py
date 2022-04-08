# The coarse network structure and the time steps are dicated by the SHD dataset. 
nb_inputs  = 700
nb_hidden  = 200
nb_outputs = 20

time_step = 1e-3
nb_steps = 100
max_time = 1.4

batch_size = 256

tau_mem_readout = 40e-3
tau_syn = 20e-3

xp_type_tau_constant = 'xp_type_tau_constant'
xp_type_tau_uniform = 'xp_type_tau_uniform'
xp_type_tau_gauss = 'xp_type_tau_gauss'