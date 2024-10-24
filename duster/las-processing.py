import laspy
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

def load_las_file(file_path):
    # Open the las file
    las = laspy.read(file_path)
    
    # Extract point cloud data
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    return points, las

def icp_registration(source_points, target_points, max_iterations=50, tolerance=0.001):
    """
    Iterative Closest Point registration algorithm
    """
    prev_error = 0
    for iteration in range(max_iterations):
        # Find nearest neighbors
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(target_points)
        distances, indices = neigh.kneighbors(source_points)
        
        # Get corresponding points
        corresponding_target_points = target_points[indices.ravel()]
        
        # Compute centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(corresponding_target_points, axis=0)
        
        # Center the point clouds
        centered_source = source_points - source_centroid
        centered_target = corresponding_target_points - target_centroid
        
        # Compute rotation
        H = np.dot(centered_source.T, centered_target)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure a right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = target_centroid - np.dot(R, source_centroid)
        
        # Apply transformation
        source_points = np.dot(R, source_points.T).T + t
        
        # Compute error
        current_error = np.mean(distances)
        if abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error
    
    return source_points, R, t

# Example usage
def main():
    # Load two LAS files
    source_points, source_las = load_las_file('source.las')
    target_points, target_las = load_las_file('target.las')
    
    # Perform registration
    aligned_points, rotation, translation = icp_registration(source_points, target_points)
    
    # Create a new LAS file with aligned points
    aligned_las = laspy.create(point_format=source_las.header.point_format, file_version=source_las.header.version)
    aligned_las.points = source_las.points
    
    # Update coordinates
    aligned_las.x = aligned_points[:, 0]
    aligned_las.y = aligned_points[:, 1]
    aligned_las.z = aligned_points[:, 2]
    
    # Write the aligned points to a new LAS file
    aligned_las.write('aligned.las')

if __name__ == "__main__":
    main()
