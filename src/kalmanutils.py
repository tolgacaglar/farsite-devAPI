import numpy as np
from shapely.geometry import MultiPolygon, Polygon

def get_vertices(geom):
    if isinstance(geom, MultiPolygon):
        geompoly = geom.geoms[0]
    elif isinstance(geom, Polygon):
        geompoly = geom

    return list(geompoly.exterior.coords)

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
def observed_uncertainties(vectors, winddirection, scale):
    windx = np.cos((90-winddirection)*np.pi/180)
    windy = np.sin((90-winddirection)*np.pi/180)
    return ((1 - np.array(vectors).dot(np.array([windx, windy]))))/4*scale


def calculate_vectors_align(vertices_lst):
    rotated_vertices_lst = align_perimeters(vertices_lst)
    trajectories_lst = calculate_trajectories(rotated_vertices_lst)
    return calculate_vectors(rotated_vertices_lst)

def calculate_uncertainties_observed(vertices, winddirection, scale=1):
    # Calculate centroid
    centroid = np.mean(vertices, axis=0)
    
    vectors = []
    for vertex in vertices:
        x = vertex[0] - centroid[0]
        y = vertex[1] - centroid[1]
        
        length = np.sqrt(x**2 + y**2)
        vectors.append((x/length, y/length))
        
    return observed_uncertainties(vectors, winddirection, scale=scale)
    
    
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