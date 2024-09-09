import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion into a rotation matrix."""
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

def create_transformation_matrix(rotation, translation):
    """Create a homogeneous transformation matrix from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

cs_records = [
    [1.52387798135, 0.494631336551, 1.50932822144, 0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068],
    [1.70079118954, 0.0159456324149, 1.51095763913, 0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
    [1.5508477543, -0.493404796419, 1.49574800619, 0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485],
    [1.0148780988, -0.480568219723, 1.56239545128, 0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798],
    [0.0283260309358, 0.00345136761476, 1.57910346144, 0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578],
    [1.03569100218, 0.484795032713, 1.59097014818, 0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753]
]

lidar_cs_record = {'translation': [0.943713, 0.0, 1.84023], 
                   'rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]}
lidar_cs_rotation = quaternion_to_rotation_matrix(lidar_cs_record['rotation'])
lidar_cs_trans = create_transformation_matrix(lidar_cs_rotation, lidar_cs_record['translation'])

transformed_matrices = []

for cs_record in cs_records:
    translation, rotation = cs_record[:3], cs_record[3:]
    cam_cs_rotation = quaternion_to_rotation_matrix(rotation)
    cam_cs_trans = create_transformation_matrix(cam_cs_rotation, translation)
    
    # Compute lidar to camera transformation matrix
    lidar_to_cam_computed = np.linalg.inv(cam_cs_trans) @ lidar_cs_trans
    transformed_matrices.append(lidar_to_cam_computed)

# Provided lidar2cam matrices
lidar2cam_provided = [[[ 1.26780975e+03,  8.13738032e+02,  2.43565782e+01,
        -3.07586088e+02],
       [ 9.39695964e+00,  5.16006239e+02, -1.25659898e+03,
        -6.15437060e+02],
       [ 1.74362345e-03,  9.99809674e-01,  1.94312979e-02,
        -3.96409491e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]], [[ 1.36675788e+03, -6.11267650e+02, -2.95588776e+01,
        -4.93576551e+02],
       [ 4.01145984e+02,  3.03038137e+02, -1.25791187e+03,
        -7.22633315e+02],
       [ 8.36397107e-01,  5.48101238e-01,  4.99120155e-03,
        -5.92413395e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]], [[ 6.12767942e+01,  1.51578679e+03,  3.78716955e+01,
        -1.98612215e+02],
       [-3.87933008e+02,  3.08949456e+02, -1.26638280e+03,
        -6.64616893e+02],
       [-8.15932990e-01,  5.78019333e-01,  1.21246462e-02,
        -4.87402360e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]], [[-8.14579570e+02, -8.23830824e+02, -1.43755086e+01,
        -8.59604825e+02],
       [ 4.77835726e+00, -4.75469198e+02, -8.12930052e+02,
        -7.25445075e+02],
       [-6.52979313e-03, -9.99948026e-01, -7.82987300e-03,
        -1.03499340e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]], [[-1.14913597e+03,  9.41414788e+02,  8.13640214e+00,
        -6.20887836e+02],
       [-4.42313277e+02, -1.14305634e+02, -1.27024418e+03,
        -5.24086815e+02],
       [-9.48333235e-01, -3.15921268e-01, -2.92886870e-02,
        -4.42762891e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]], [[ 2.99627235e+02, -1.46441301e+03, -6.12385717e+01,
        -3.70908876e+02],
       [ 4.60821147e+02, -1.28958359e+02, -1.26830032e+03,
        -5.97897726e+02],
       [ 9.32921208e-01, -3.59544114e-01, -1.96481530e-02,
        -5.06698020e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]]

# Function to compare matrices
def matrices_are_similar(mat1, mat2, tolerance=1e-6):
    return np.allclose(mat1, mat2, atol=tolerance)

# Compare each computed matrix with the corresponding provided matrix
comparison_results = [matrices_are_similar(comp, prov) for comp, prov in zip(transformed_matrices, lidar2cam_provided)]

print(comparison_results)
