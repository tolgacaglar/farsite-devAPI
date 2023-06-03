import numpy as np
from shapely.geometry import MultiPolygon, Polygon
import datetime
import geopandas as gpd
import pandas as pd
import random

def get_vertices(geom):
    if isinstance(geom, MultiPolygon):
        geompoly = geom.geoms[0]
    elif isinstance(geom, Polygon):
        geompoly = geom

    return np.array((geompoly.exterior.coords))

def interpolate_perimeter(vertices, dnumber):
    # Changes the number of vertices of the given set of vertices
    if len(vertices) == dnumber:
        return vertices
    
    vertices = np.array(vertices)
    step_len = np.sqrt(np.sum(np.diff(vertices, 1, 0)**2, 1)) # length of each side
    step_len = np.append([0], step_len)
    cumulative_len = np.cumsum(step_len)
    interpolation_loc = np.linspace(0, cumulative_len[-1], dnumber)
    X = np.interp(interpolation_loc, cumulative_len, vertices[:,0])
    Y = np.interp(interpolation_loc, cumulative_len, vertices[:,1])

    return list(zip(X,Y))

def find_pairs(vertices_A, vertices_B):
    # Returns a list of quadruples: (AX, AY, BX, BY)
    
    number_of_vertices = max(len(vertices_A), len(vertices_B))
    
    vertices_A = interpolate_perimeter(vertices_A, number_of_vertices)
    vertices_B = interpolate_perimeter(vertices_B, number_of_vertices)
    
    # Find closest points to start with
    distance_to_zero = []
    for vertex_B in vertices_B:
        distance = np.sqrt((vertices_A[0][0] - vertex_B[0])**2 + 
                           (vertices_A[0][1] - vertex_B[1])**2)
        
        distance_to_zero.append(distance)
        
    minidx = np.argmin(distance_to_zero)
    
    return_list = []
    for i, vertex_A in enumerate(vertices_A):
        B_i = (i + minidx)%len(vertices_B)
        vertex_B = vertices_B[B_i]
        
        return_list.append((vertex_A[0], vertex_A[1], vertex_B[0], vertex_B[1]))
        
    return return_list

def align_perimeters(vertices_lst):
    # Reorders the vertices for each set in the vertices_lst
    
    # Calculate the max number of vertices to interpolate all perimeters to
    vertex_lengths = []
    for vertices in vertices_lst:
        vertex_lengths.append(len(vertices))

    number_of_vertices = 100
    
    # Interpolate all the perimeters
    interpolated_vertices = []
    for vertices in vertices_lst:
        interpolated_vertices.append(interpolate_perimeter(vertices, number_of_vertices))
    
    # Find the starting index for each vertex
    starting_indices = []
    # First one is zero
    starting_indices.append(0)
    for i in range(len(interpolated_vertices[1:])):
        vertices_A = interpolated_vertices[i]
        vertices_B = interpolated_vertices[i+1]
        
        start_idx = starting_indices[i]
        distance_to_zero = []
        for vertex_B in vertices_B:
            distance = np.sqrt((vertices_A[start_idx][0] - vertex_B[0])**2 + 
                               (vertices_A[start_idx][1] - vertex_B[1])**2)

            distance_to_zero.append(distance)
            
        starting_indices.append(np.argmin(distance_to_zero))
        
    return_list = []
    # Rotate each interpolated perimeter to match the indices of the vertices
    for (vix, vertices) in enumerate(interpolated_vertices):
        rotated_vertices = []
        for i, vertex in enumerate(vertices):
            v = vertices[(i+starting_indices[vix])%number_of_vertices]
            rotated_vertices.append(v)
        return_list.append(rotated_vertices)
        
    return np.array(return_list)
    
def calculate_trajectories(rotated_vertices_lst):
    number_of_vertices = len(rotated_vertices_lst[-1])
    
    trajectories_lst = []
    for vx in range(number_of_vertices):
        traj = []
        for i in range(len(rotated_vertices_lst)):
            vertices = rotated_vertices_lst[i]
            traj.append((vertices[vx][0], vertices[vx][1]))
        trajectories_lst.append(traj)
        
    return trajectories_lst

def calculate_vectors(rotated_vertices_lst):
    number_of_vertices = len(rotated_vertices_lst[-1])
    
    centroids_lst = []
    for vertices in rotated_vertices_lst:
        centroids_lst.append(np.mean(vertices, axis=0))
    
    vectors_lst = []
    for i in range(len(rotated_vertices_lst)):
        centroid = centroids_lst[i]
        
        vectors = []
        for vix in range(number_of_vertices):
            vertex = rotated_vertices_lst[i][vix]
            x = vertex[0] - centroid[0]
            y = vertex[1] - centroid[1]
            length = np.sqrt(x**2 + y**2)
            vectors.append((x/length, y/length))
            
        vectors_lst.append(vectors)
        
    return vectors_lst

##################################################
######### Observed perimeter uncertainties #######
##################################################
def observed_uncertainties(vectors, windspeed, winddirection, scale):
    windx = np.cos((90-winddirection)*np.pi/180)
    windy = np.sin((90-winddirection)*np.pi/180)
    return_vals = ((1 - np.array(vectors).dot(np.array([windx, windy]))))/4*scale
    
    return np.zeros_like(return_vals) + 50


def calculate_vectors_align(vertices_lst):
    rotated_vertices_lst = align_perimeters(vertices_lst)
    trajectories_lst = calculate_trajectories(rotated_vertices_lst)
    return calculate_vectors(rotated_vertices_lst)

def calculate_uncertainties_observed(vertices, windspeed, winddirection, scale=1):
    # Calculate centroid
    centroid = np.mean(vertices, axis=0)
    
    vectors = []
    for vertex in vertices:
        x = vertex[0] - centroid[0]
        y = vertex[1] - centroid[1]
        
        length = np.sqrt(x**2 + y**2)
        vectors.append((x/length, y/length))
        
    return observed_uncertainties(vectors, windspeed, winddirection, scale=scale)
    
    
def calculate_modified(observe_aligned, observed_velocity_aligned, observed_uncertainties,
                       next_observe_aligned, next_observed_velocity_aligned, P_kneg):
    dt = 20
    
    x_modified = []
    vx_modified = []
    y_modified = []
    vy_modified = []
    P_k = []
    P_kpred = []
    for i in range(len(observe_aligned)):
        x_measure = observe_aligned[i,0]
        y_measure = observe_aligned[i,1]
        vx_measure = observed_velocity_aligned[i,0]
        vy_measure = observed_velocity_aligned[i,1]
        dx_measure = observed_uncertainties[i]
        dy_measure = observed_uncertainties[i]
        dvx_measure = observed_uncertainties[i]
        dvy_measure = observed_uncertainties[i]

        x_measure_next = next_observe_aligned[i,0]
        y_measure_next = next_observe_aligned[i,1]
        vx_measure_next = next_observed_velocity_aligned[i,0]
        vy_measure_next = next_observed_velocity_aligned[i,1]
        # dx_measure_next = next_observed_uncertainties[i]
        # dy_measure_next = next_observed_uncertainties[i]
        # dvx_measure_next = next_observed_uncertainties[i]
        # dvy_measure_next = next_observed_uncertainties[i]

        # 4x1 matrix for initial values
        x_kneg = np.array([[x_measure], [vx_measure], [y_measure], [vy_measure]])

        # Linear Kalman model
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        ### One step Kalman filter
        # Prediction - Kalman result
        X_kpred = np.matmul(A, x_kneg)

        # Predicted process covariance matrix
        P_kpred.append(np.matmul(np.matmul(A, P_kneg[i]), A.transpose()))
        # Zero for the cross terms
        P_kpred[-1] = P_kpred[-1]*np.eye(4)

        # Calculate Kalman Gain
        H = np.eye(4)
        R = np.array([[dx_measure*dx_measure, 0, 0, 0],
                      [0, dvx_measure*dvx_measure, 0, 0],
                      [0, 0, dy_measure*dy_measure, 0],
                      [0, 0, 0, dvy_measure*dvy_measure]])

        K_upper = np.matmul(P_kpred[-1], H)
        K_lower = np.matmul(np.matmul(H, P_kpred[-1]), H.transpose()) + R

        K = np.zeros_like(H)
        for i in range(4):
            K[i,i] = K_upper[i,i]/K_lower[i,i]

        # New observation
        C = np.eye(4)
        x_kmeasure = [[x_measure_next], [vx_measure_next], [y_measure_next], [vy_measure_next]]
        Y_k = np.matmul(C, x_kmeasure)

        # Calculating modified state. This will be used to calculate the next prediction
        X_k = X_kpred + np.matmul(K, (Y_k - np.matmul(H, X_kpred)))

        # New process covariance matrix
        P_k.append(np.matmul(np.eye(4) - np.matmul(K, H), P_kpred[-1]))

        x_modified.append(X_k[0,0])
        vx_modified.append(X_k[1,0])
        y_modified.append(X_k[2,0])
        vy_modified.append(X_k[3,0])

    modified_vertices = []
    for x,y in zip(x_modified, y_modified):
        modified_vertices.append((x,y))
                       
    #TODO: Rearrange vertices for geometry construction
    return modified_vertices, P_k

def calculate_parameters(ignition, observe, model, winddirection, dt):
    #1- Rotate & align each vertices after interpolation
    ignition_aligned, observe_aligned, model_aligned = align_perimeters([ignition, observe, model])

    #2 Calculate uncertainties in observe


    ignition_uncertainties = calculate_uncertainties_observed(ignition_aligned, winddirection)
    # observed_uncertainties = calculate_uncertainties_observed(observe_aligned, winddirection)
    observed_velocity_aligned = (observe_aligned - ignition_aligned) / dt
    model_velocity_aligned = (model_aligned - ignition_aligned) / dt
    
    return ignition_aligned, model_velocity_aligned, model_aligned, observe_aligned, observed_velocity_aligned, ignition_uncertainties




############################################
#####  FINAL HELPERS AS OF 05142023 ########
############################################

class State:
    def __init__(self, geom):
        self.geom = geom
        
        # Initialize
        self.vertices = self.calculate_vertices()
        self.lengths = self.calculate_lengths()
    def calculate_vertices(self):
        geom = self.geom
        
        if isinstance(geom, MultiPolygon):
            geompoly = calculate_max_area_geom(geom)
        elif isinstance(geom, Polygon):
            geompoly = geom

        return np.array((geompoly.exterior.coords))
    
    def calculate_lengths(self):
        return np.sqrt((np.diff(self.vertices, axis=0)**2).sum(axis=1))
    
    def calculate_vector(self):
        # Returns column vector of the vertices (x0, y0, x1, y1, ...)
        return self.vertices.reshape(len(self.vertices)*2, 1)

def sample_geometry(current_state, uncertainties):
    
    maxlength = current_state.lengths.max()
    
    sampled_vertices = []
    
    # Choose a random direction
    theta = random.uniform(0,2*np.pi)

    for (x,y), sigma in zip(current_state.vertices, uncertainties):
        mu=0
        # randx = random.gauss(mu, sigma)
        # randy = random.gauss(mu, sigma)
        
        # Choose a normal random radius based on the given sigma
        radius = abs(random.gauss(mu, sigma))
        
        # Calculate x and y distance for the random
        randx = radius*np.cos(theta)
        randy = radius*np.sin(theta)
        
        sampled_vertices.append((x+randx, y+randy))

    sampled_vertices = np.array(sampled_vertices)
    # return Polygon(sampled_vertices).buffer(maxlength, join_style=1).buffer(-maxlength, join_style=1)
    return Polygon(sampled_vertices)

def calculate_max_area_geom(multigeom):
    max_area = 0
    max_area_idx = 0
    for ix, g in enumerate(multigeom.geoms):
        if g.area > max_area:
            max_area = g.area
            max_area_idx = ix
    return multigeom.geoms[max_area_idx]


def calculate(igniteidx, compareidx, usr, label, windspeed = 10, winddirection = 90, dt=datetime.timedelta(minutes=30)):
    lcpidx = '43b7f5db36994599861eec4849cc68fd'        # Index for Maria2019
    barrieridx = 'cb47616cd2dc4ccc8fd523bd3a5064bb'    # NoBarrier shapefile index

    # Generate df for the next reference ignition only to get the datetime
    filetype = 'Ignition'
    # objectid = str(usr.db.gdfignition.loc[igniteidx, 'objectid']) + '_simRef'
    filepath = f'/home/jovyan/farsite/inputs/maria_ignite/maria_{compareidx}'
    comparedatetime = usr.db.gdfignition.loc[igniteidx, 'datetime'] + dt
    description = 'Maria2019'

    gdfcompare = gpd.GeoDataFrame(index=[compareidx], data = {'filetype': filetype,
                                          'objectid': label,
                                          'filepath': filepath,
                                          'datetime': comparedatetime,
                                          'description': description,
                                          'geometry': None})

    usr.db.gdfignition = pd.concat([usr.db.gdfignition, gdfcompare])

    inputData = {'description': description,
                 'igniteidx'  : igniteidx,
                 'compareidx' : compareidx,
                 'lcpidx'     : lcpidx,
                 'barrieridx' : barrieridx,

                 'windspeed': windspeed, 'winddirection': winddirection,
                 'relhumid': 90, 'temperature': 20}

    mainapi = usr.calculatePerimeters(inputData)
    mainapi.run_farsite()

    # Collect the simulated geometry
    gdfsim = usr.db.gdfsimulation.iloc[-1]
    gdfsim_geom = gdfsim['geometry']
    ##########################################################################
    if isinstance(gdfsim_geom, MultiPolygon):
        gdfsim_geom = calculate_max_area_geom(gdfsim_geom)
    ###########################################################################

    # Update the ignition table with the simulated info
    usr.db.gdfignition.loc[compareidx, 'filepath'] = usr.db.gdfsimulation.iloc[-1]['filepath']
    usr.db.gdfignition.loc[compareidx, 'geometry'] = gdfsim_geom
    
    gpd.GeoDataFrame({'FID': [0], 'geometry':gdfsim_geom}, 
                 crs='EPSG:5070').to_file(gdfsim['filepath'])
    
    
    
def get_coordinates(geom):
    x,y = geom.exterior.coords.xy
    x = np.array(x)
    y = np.array(y)
    
    return x,y

def calculate_rms(geom1, geom2):
    xy1, xy2 = interpolate_geometries([geom1, geom2], vertex_count=100)
    xy1, xy2 = align_vertices([xy1, xy2])
    return np.sqrt(np.sum((xy1[:,0] - xy2[:,0])**2 + (xy1[:,1] - xy2[:,1])**2)/xy1.shape[0])
    

def calculate_area_diff(geom1, geom2):
    return (geom1.union(geom2) - geom1.intersection(geom2)).area

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

def interpolate_geometries(geoms, vertex_count = None):
    
    if vertex_count == None:
        vertex_count = 0
        for geom in geoms:
            if isinstance(geom, MultiPolygon):
                geom = calculate_max_area_geom(geom)

            if vertex_count < len(geom.exterior.coords):
                vertex_count = len(geom.exterior.coords)

    interpolated_vertices = []
    for geom in geoms:
        if isinstance(geom, MultiPolygon):
            geom = calculate_max_area_geom(geom)
        
        geom_state = State(geom)
        vertices = np.array(interpolate_perimeter(geom_state.calculate_vertices(), vertex_count))

        interpolated_vertices.append(vertices)
        
    return interpolated_vertices

def validate_geoms_matrix(X, aligned_geom):
    Xnew = np.zeros((2*aligned_geom.shape[0], X.shape[1]))

    for i in range(X.shape[1]):
        X_0 = X[:-2,i]
        x = X_0[::2]
        y = X_0[1::2]

        geom = Polygon(zip(x,y)).buffer(0)
        if isinstance(geom, MultiPolygon):
            geom = calculate_max_area_geom(geom)

        xx,yy = geom.exterior.xy
        xx = list(xx)
        yy = list(yy)

        geom = np.array(interpolate_perimeter(list(zip(xx, yy)), aligned_geom.shape[0]))
        geom = align_vertices([aligned_geom, geom])[1]

        Xnew[:,i] = geom.flatten()

    X = np.append(Xnew, X[-2:,:], axis=0)
    return X

def update_EnKF(Xt, Y, aligned_geom):
    nsamples = Y.shape[1]

    xt = Xt.mean(axis=1, keepdims=True)
    y = Y.mean(axis=1, keepdims=True)

    Ex = Xt - xt.repeat(nsamples, axis=1)
    Ey = Y - y.repeat(nsamples, axis=1)

    Py = 1/(nsamples)*np.matmul(Ey, Ey.T)
    Pxy = 1/(nsamples)*np.matmul(Ex, Ey.T)

    # max_Py = abs(Py).max()
    # max_Pxy = abs(Pxy).max()
    # Py /= max_Py
    # Pxy /= max_Pxy

    Py_inv = np.linalg.pinv(Py, hermitian=True)

    assert(np.allclose(np.matmul(Py_inv, Py), np.eye(Y.shape[0]))), 'Inverse calculation is incorrect'

    # K = np.matmul(Pxy, Py_inv)*(max_Pxy/max_Py)
    K = np.matmul(Pxy, Py_inv)

    # Note that Xt has additional +2 in it
    # Remove that with the matrix C
    C = np.eye(Y.shape[0], Xt.shape[0])
    #### Update the state ensemble
    X = Xt + np.matmul(K, (Y - np.matmul(C, Xt)))

    ### TODO ####
    # Fix invalid geometries

    X = validate_geoms_matrix(X, aligned_geom)

    return X


def fill_zeros(A, nonzerolen, nsamples):
    
    mu = A[:,:nonzerolen].mean(axis=1)
    stdev = A[:,:nonzerolen].std(axis=1)

    for i in range(nonzerolen, nsamples):
        # Calculate uncertainties for each point
        A[:,i] = np.random.normal(mu, stdev)
        
    return A

def create_ensemble_matrix(gdf, nsamples, vertex_count=20, aligned_geom=None, observed=False):

    geoms = gdf['geometry'].tolist()
    
    if not observed:
        wdlst = gdf['winddirection'].tolist()
        wslst = gdf['windspeed'].tolist()
    
    interpolated_vertices = interpolate_geometries(geoms, vertex_count=vertex_count)
    # Add first list of vertices from the state vector to align. align_vertices aligns all the perimeters w.r.t the first array
    if aligned_geom is not None:
        interpolated_vertices = [aligned_geom] + interpolated_vertices

    aligned_vertices = align_vertices(interpolated_vertices)
    if aligned_geom is not None:
        aligned_vertices = aligned_vertices[1:]
    else:
        aligned_geom = aligned_vertices[0]
    
    if vertex_count is None:
        vertex_count = aligned_vertices[0].shape[0]
    
    X = np.zeros((vertex_count*2 + 2*(not observed), nsamples))  # Two additional for each wd ans ws
    for i, vertices in enumerate(aligned_vertices):
        if observed:
            X[:,i] = vertices.flatten()
        else:
            X[:-2,i] = vertices.flatten()
        
        if not observed:
            X[-2,i] = wdlst[i]
            X[-1,i] = wslst[i]
        
    if not observed:
        X = fill_zeros(X, len(geoms), nsamples)
        X = validate_geoms_matrix(X, aligned_geom)
    
    return X, aligned_geom, vertex_count