from shapely import Polygon, make_valid, GeometryCollection, MultiPolygon

import numpy as np

from futils import generate_landscape, forward_pass_farsite
from tqdm import tqdm
from putils import calculate_max_area_geom, validate_geom



########## ENKF functions #########

def state_to_geom(state):
    return validate_geom(Polygon(zip(state[:-2:2], state[1:-2:2])))

def geom_to_state(geom, wx, wy):
    return_state = np.zeros((2*len(geom.exterior.coords[:-1])+2,1))
    return_state[:-2,0] = np.array(geom.exterior.coords[:-1]).reshape(2*len(geom.exterior.coords[:-1]), 1).flatten()
    return_state[-2,0] = wx
    return_state[-1,0] = wy
    return return_state

################# ALIGN GEOMS ####################
def make_ccw(geom):
    if not geom.exterior.is_ccw:
        return geom.reverse()
    
    return geom

def interpolate_perimeter(vertices, dnumber):
    # Changes the number of vertices of the given set of vertices
    # if len(vertices) == dnumber:
    #     return vertices
    
    vertices = np.array(vertices)
    step_len = np.sqrt(np.sum(np.diff(vertices, 1, 0)**2, 1)) # length of each side
    step_len = np.append([0], step_len)
    cumulative_len = np.cumsum(step_len)
    interpolation_loc = np.linspace(0, cumulative_len[-1], dnumber)
    X = np.interp(interpolation_loc, cumulative_len, vertices[:,0])
    Y = np.interp(interpolation_loc, cumulative_len, vertices[:,1])

    return list(zip(X,Y))

def align_vertices(interpolated_vertices):
    minroll_lst = []
    
    aligned_vertices = [interpolated_vertices[0]]
    for i in range(len(interpolated_vertices)-1):
        right_vertices = interpolated_vertices[i+1]

        # Cycle right_vertices
        l2perroll = []
        for roll in range(len(interpolated_vertices[i])-1):
            diff = aligned_vertices[0] - right_vertices
            diff2sum = (diff[:,0]**2 + diff[:,1]**2).sum()

            # Calculate diff^2 in
            l2perroll.append(diff2sum)

            right_vertices = np.roll(right_vertices,1, axis=0)

        minroll_lst.append(np.argmin(l2perroll))

    for i in range(len(interpolated_vertices)-1):
        aligned_vertices.append(np.roll(interpolated_vertices[i+1], minroll_lst[i], axis=0))
    
    return aligned_vertices


def interpolate_geom(geom, vertex_count):
    interpolated_geom = Polygon(interpolate_perimeter((geom.exterior.coords), vertex_count))
    if len(interpolated_geom.exterior.coords[:-1]) == vertex_count-1:
        interpolated_geom = Polygon(interpolate_perimeter((geom.exterior.coords), vertex_count+1))
    if len(interpolated_geom.exterior.coords[:-1]) == vertex_count+1:
        interpolated_geom = Polygon(interpolate_perimeter((geom.exterior.coords), vertex_count-1))

    return interpolated_geom

def interpolate_geoms(geoms, vertex_count):
        
    interpolated_geoms = []
    for geom in geoms:
        interpolated_geoms.append(interpolate_geom(geom, vertex_count))
        
    return interpolated_geoms

def align_geoms(geoms, vertex_count): 
    '''
        Will align all the geometries based on geoms[0]
    '''
    
    # Calculate interpolated vertices first
    interpolated_geoms = interpolate_geoms(geoms, vertex_count)
    
    interpolated_vertices = [make_ccw(interpolated_geoms[0]).exterior.coords[:-1]]
    for geom in interpolated_geoms[1:]:
        interpolated_vertices.append(make_ccw(geom).exterior.coords[:-1])

    # for vertices in align_vertices(np.array(interpolated_vertices)):
    #     poly = Polygon(vertices)
    #     geom = interpolate_geom(validate_geom(poly), vertex_count)
            
    
    return [Polygon(vertices) for vertices in align_vertices(np.array(interpolated_vertices))]

#############################    
    
def xy_to_state(x,y):
    ret = []
    for i in range(len(x)):
        ret.append(x[i])
        ret.append(y[i])

    return np.array(ret).reshape((2*len(x),1))    
    
    
def align_states(state_lst, vertex_count=None):
    if vertex_count is None:
        vertex_count = max(len(st) for st in state_lst)//2
    x0 = state_lst[0][:-2:2]
    y0 = state_lst[0][1:-2:2]
    x1 = state_lst[1][:-2:2]
    y1 = state_lst[1][1:-2:2]

    geom0 = make_ccw(Polygon(zip(x0,y0)))
    geom1 = make_ccw(Polygon(zip(x1,y1)))

    geom0, geom1 = align_geoms([geom0, geom1], vertex_count)
    x,y = geom0.exterior.coords.xy
    x0 = x.tolist()[:-1]
    y0 = y.tolist()[:-1]
    
    state0 = np.zeros((2*vertex_count + 2, 1))
    state0[:-2,0] = xy_to_state(x0, y0).flatten()
    state0[-2,0] = state_lst[0][-2]
    state0[-1,0] = state_lst[0][-1]
    
    x,y = geom1.exterior.coords.xy
    x1 = x.tolist()[:-1]
    y1 = y.tolist()[:-1]
    
    state1 = np.zeros_like(state0)
    state1[:-2,0] = xy_to_state(x1, y1).flatten()
    state1[-2,0] = state_lst[1][-2]
    state1[-1,0] = state_lst[1][-1]

    return [state0, state1]
    

    
def return_ws_wd(xk):
    wx = xk[-2,:]
    wy = xk[-1,:]

    ws = np.sqrt(wx**2 + wy**2)
    wd = np.fmod((180/np.pi)*np.arctan2(wy,wx) + 360,360)
    
    return ws,wd

def adjusted_state_EnKF_farsite(initial_state, observation_state, wssigma,
                        X, n_states, n_output, n_vertex, n_samples, rng, dt,
                        vsize, wsize, description,
                               dist_res, perim_res):

    xkhat_ensemble = np.zeros((n_states, n_samples))
    
    zkphat_ensemble = np.zeros((n_states, n_samples))
    xkphat_ensemble = np.zeros((n_states, n_samples))
    ykhat_ensemble = np.zeros((n_output, n_samples))
    
    Xs = np.linalg.cholesky(X)
    # For each sample
    zero_samples = []
    
    # Generate lcp for initial_state
    
    initial_geom = state_to_geom(initial_state)
    lcppath = generate_landscape(initial_geom, description=description)
    
    for s in tqdm(range(n_samples)):
    
        xkhat_ensemble[:,s:(s+1)] = initial_state + np.matmul(Xs, rng.normal(size=(n_states,1)))
        xkhat_ensemble[-2,s] = initial_state[-2] + rng.normal(0,scale=wssigma)
        xkhat_ensemble[-1,s] = initial_state[-1] + rng.normal(0,scale=wssigma)
    
        wx = xkhat_ensemble[-2,s]
        wy = xkhat_ensemble[-1,s]
        #####################
        # convert wx,wy to ws and wd
        ws = np.sqrt(wx**2 + wy**2)
        wd = np.fmod((180/np.pi)*np.arctan2(wy,wx) + 360,360)

        # Calculate the ensemble for the observations
        # ykhat_ensemble[:,s:(s+1)] = xy_to_state(*sample_xy(observation_state[::2], observation_state[1::2], rng, scale=vsize))
        ykhat_ensemble[:,s:(s+1)] = xkhat_ensemble[:,s:(s+1)] + rng.normal(0, scale=vsize, size=(n_output,1))
        ykhat_ensemble[-2,s] = xkhat_ensemble[-2,s] + rng.normal(0,scale=wssigma)
        ykhat_ensemble[-1,s] = xkhat_ensemble[-1,s] + rng.normal(0,scale=wssigma)
########################################
        forward_geom = forward_pass_farsite(state_to_geom(xkhat_ensemble[:,s:(s+1)]),
                                            params={'windspeed': int(ws),
                                                    'winddirection': int(wd),
                                                    'dt': dt},
                                            lcppath=lcppath, 
                                            description=description,
                                            dist_res=dist_res, perim_res=perim_res)
        # TODO:
        # implement forward ws and wd
        forward_wx = wx + rng.normal(0, scale=wssigma)
        forward_wy = wy + rng.normal(0, scale=wssigma)
    
        if forward_geom is None:
            zero_samples.append(s)
            continue
            
        forward_state = geom_to_state(forward_geom, forward_wx, forward_wy)
        aligned_states = align_states([initial_state, forward_state], vertex_count = n_vertex)

        # Randomness is not given here. It's given later. This is only for calculating the ensembles
#         zkphat_ensemble[:,s:(s+1)] = xy_to_state(aligned_states[1][::2], aligned_states[1][1::2])
        zkphat_ensemble[:,s:(s+1)] = aligned_states[1]
        
########################################
        

    # Calculate the mean of the non-zero ensembles
    zkphat_mean = zkphat_ensemble.sum(axis=1, keepdims=True)/(n_samples - len(zero_samples))
    
    # Fill in the zero samples with the mean
    for s in zero_samples:
        zkphat_ensemble[:,s:(s+1)] = zkphat_mean
    
    filled_counts = len(zero_samples)
    
    # zkphat_mean = zkphat_ensemble.mean(axis=1, keepdims=True)
    ykhat_mean = ykhat_ensemble.mean(axis=1, keepdims=True)
    
    # Calculate errors
    # zkphat_ensemble -= zkphat_mean
    # ykhat_ensemble -= ykhat_mean
    ezkphat_ensemble = np.zeros_like(zkphat_ensemble)
    for n in range(n_states):
        ezkphat_ensemble[n:(n+1),:] = zkphat_ensemble[n:(n+1),:] - zkphat_mean[n] + rng.normal(0, scale=wsize)  # + omega_k^j
    
    eykhat_ensemble = np.zeros_like(ykhat_ensemble)
    for n in range(n_output):
        eykhat_ensemble[n:(n+1),:] = ykhat_ensemble[n:(n+1),:] - ykhat_mean[n]
    
    Pzy = 1/n_samples*np.matmul(ezkphat_ensemble, eykhat_ensemble.T)
    
    Py = 1/n_samples*np.matmul(eykhat_ensemble, eykhat_ensemble.T)
    Pyinv = np.linalg.pinv(Py)

    inv_product = np.matmul(Py, Pyinv)
    if not np.allclose(inv_product, np.eye(n_output)):
        print('Inverse calculation is incorrect')
        display(inv_product)
        
    # warnings.warn('Not checking the inverse calculation')
    
    # Compute estimated Kalman gain based on correlations
    L_EnKF = np.matmul(Pzy, Pyinv)
    
    # compute mean valued state adjustment using measurement y(k)
    # yk = geom_to_state(observation['geometry'], n_states, nvertex)
    aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    yk = aligned_states[1]
    
    adjusted_state = zkphat_mean + np.matmul(L_EnKF, yk - ykhat_mean)
    
    # Compute the state adjustment ensembles to update state covariance matrix X
    for j in range(n_samples):
        xkphat_ensemble[:,j:(j+1)] = zkphat_ensemble[:,j:(j+1)] + np.matmul(L_EnKF, yk - ykhat_ensemble[:,j:(j+1)])
    
    # xkphat_ensemble -= xkphat_ensemble.mean(axis=1, keepdims=True)
    xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    exkphat_ensemble = np.zeros_like(xkphat_ensemble)
    for n in range(n_states):
        exkphat_ensemble[n:(n+1), :] = xkphat_ensemble[n:(n+1),:] - xkphat_mean[n]
    
    X = 1/n_samples*np.matmul(exkphat_ensemble, exkphat_ensemble.T) + 1e-10*np.eye(n_states)
    
    # Validate adjusted state and reinterpolate
    adjusted_geom = validate_geom(state_to_geom(adjusted_state))
    adjusted_state = geom_to_state(adjusted_geom, adjusted_state[-2,0], adjusted_state[-1,0])
    aligned_states = align_states([initial_state, adjusted_state], vertex_count=n_vertex)
    adjusted_state = aligned_states[1]
    
    return adjusted_state, X, zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, xkphat_ensemble

###################
# SAMPLING ########
###################
def sample_geometry(geom, rng, sigma=1):
    sampled_vertices = []
    
    # Choose a random direction
    theta = rng.uniform(0,2*np.pi)

    for (x,y) in geom.exterior.coords[:-1]:
        mu=0
        
        randx = rng.normal(mu, sigma)
        randy = rng.normal(mu, sigma)
        
#         # Choose a normal random radius based on the given sigma
#         radius = abs(random.gauss(mu, sigma))
        
#         # Calculate x and y distance for the random
#         randx = radius*np.cos(theta)
#         randy = radius*np.sin(theta)
        
        sampled_vertices.append((x+randx, y+randy))

    sampled_vertices = np.array(sampled_vertices)
    return Polygon(sampled_vertices)

def sample_windspeed(loc, sigma, rng):
    ws = rng.normal(loc, sigma)
    if ws < 0:
        ws = 0
    return ws
def sample_winddirection(loc, sigma, rng):
    return np.fmod(rng.normal(loc, sigma)+360, 360)