

class ICPAligner:
    """
    Implements Iterative Closest Point (ICP) algorithm to align two point clouds.
    Can use GPS poses as initial estimate.
    """
    
    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 1.0,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
    
    def gps_to_local_transform(self, pose1: Pose, pose2: Pose) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute initial transformation from GPS poses.
        Converts lat/lon to local ENU (East-North-Up) coordinates.
        
        :param pose1: Source GPS pose
        :param pose2: Target GPS pose
        :return: (R, t) - Initial rotation and translation estimate
        """
        # Convert GPS to local meters (simple approximation)
        # More accurate: use pyproj or utm library for proper conversion
        lat_to_m = 111320.0  # meters per degree latitude
        lon_to_m = 111320.0 * np.cos(np.radians(pose1.latitude))
        
        # Compute translation in local frame
        delta_lon = (pose2.longitude - pose1.longitude) * lon_to_m
        delta_lat = (pose2.latitude - pose1.latitude) * lat_to_m
        delta_alt = pose2.altitude - pose1.altitude
        
        t = np.array([delta_lon, delta_lat, delta_alt])
        
        # For now, assume no rotation (vehicle moves forward)
        # Could estimate heading from GPS trajectory
        R = np.eye(3)
        
        return R, t
    
    def align(self, 
              source: np.ndarray, 
              target: np.ndarray,
              initial_R: Optional[np.ndarray] = None,
              initial_t: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Align source point cloud to target using ICP.
        
        :param source: Source point cloud (N x 3)
        :param target: Target point cloud (M x 3)
        :param initial_R: Initial rotation estimate (3x3), defaults to identity
        :param initial_t: Initial translation estimate (3,), defaults to zero
        :return: Dictionary with alignment results
        """
        # Initialize transformation
        R = initial_R if initial_R is not None else np.eye(3)
        t = initial_t if initial_t is not None else np.zeros(3)
        
        # Apply initial transformation
        source_transformed = (R @ source.T).T + t
        prev_error = float('inf')
        
        # Build KD-tree for fast nearest neighbor search
        tree = cKDTree(target)
        
        for iteration in range(self.max_iterations):
            # Step 1: Find correspondences
            distances, indices = tree.query(source_transformed, k=1)
            
            # Filter by distance threshold
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 3:
                print(f"Warning: Too few correspondences ({np.sum(valid_mask)})")
                break
            
            source_matched = source_transformed[valid_mask]
            target_matched = target[indices[valid_mask]]
            
            # Step 2: Compute centroids
            source_centroid = np.mean(source_matched, axis=0)
            target_centroid = np.mean(target_matched, axis=0)
            
            # Step 3: Center the points
            source_centered = source_matched - source_centroid
            target_centered = target_matched - target_centroid
            
            # Step 4: Compute rotation using SVD
            H = source_centered.T @ target_centered
            U, _, Vt = np.linalg.svd(H)
            R_iter = Vt.T @ U.T
            
            # Ensure proper rotation
            if np.linalg.det(R_iter) < 0:
                Vt[-1, :] *= -1
                R_iter = Vt.T @ U.T
            
            # Step 5: Compute translation
            t_iter = target_centroid - R_iter @ source_centroid
            
            # Step 6: Apply incremental transformation
            source_transformed = (R_iter @ source_transformed.T).T + t_iter
            
            # Update cumulative transformation
            t = R_iter @ t + t_iter
            R = R_iter @ R
            
            # Step 7: Compute error
            error = np.mean(distances[valid_mask])
            
            # Check convergence
            if abs(prev_error - error) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations (error: {error:.6f}m)")
                break
            
            prev_error = error
        
        return {
            'R': R,
            't': t,
            'error': error,
            'iterations': iteration + 1,
            'transformation_matrix': self._to_transformation_matrix(R, t),
            'num_correspondences': np.sum(valid_mask)
        }
    
    def _to_transformation_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Convert R and t to 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod
    def load_pcd(pcd_file: str) -> np.ndarray:
        """Load a .pcd file and return points as numpy array (N x 3)."""
        pcd = o3d.io.read_point_cloud(pcd_file)
        return np.asarray(pcd.points)
    
    @staticmethod
    def visualize_alignment(source: np.ndarray, target: np.ndarray, 
                           R: np.ndarray, t: np.ndarray):
        """Visualize source, target, and aligned source point clouds."""
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        source_pcd.paint_uniform_color([1, 0, 0])  # Red
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        target_pcd.paint_uniform_color([0, 1, 0])  # Green
        
        source_aligned = (R @ source.T).T + t
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(source_aligned)
        aligned_pcd.paint_uniform_color([0, 0, 1])  # Blue
        
        o3d.visualization.draw_geometries(
            [source_pcd, target_pcd, aligned_pcd],
            window_name="ICP: Red=Source, Green=Target, Blue=Aligned"
        )


def align_consecutive_scans(dataset: LidarGnssDataset, 
                           scan_idx1: int, 
                           scan_idx2: int,
                           use_gps_init: bool = True) -> Dict[str, Any]:
    """
    Apply ICP to align two consecutive scans, optionally using GPS as initial estimate.
    
    :param dataset: LidarGnssDataset instance
    :param scan_idx1: Index of source scan
    :param scan_idx2: Index of target scan
    :param use_gps_init: Whether to use GPS poses for initialization
    :return: ICP alignment result
    """
    # Get scan info
    scan1_info = dataset[scan_idx1]
    scan2_info = dataset[scan_idx2]
    
    print(f"\nAligning scans {scan_idx1} -> {scan_idx2}")
    print(f"Source: {scan1_info['pcd_file']}")
    print(f"Target: {scan2_info['pcd_file']}")
    
    # Load point clouds
    source = ICPAligner.load_pcd(scan1_info['pcd_file'])
    target = ICPAligner.load_pcd(scan2_info['pcd_file'])
    
    print(f"Source points: {len(source)}, Target points: {len(target)}")
    
    # Get GPS-based initial estimate
    icp = ICPAligner(max_iterations=50, tolerance=1e-6, max_correspondence_distance=1.0)
    
    if use_gps_init:
        pose1 = scan1_info['pose']
        pose2 = scan2_info['pose']
        
        print(f"\nGPS poses:")
        print(f"  Pose 1: ({pose1.latitude:.6f}, {pose1.longitude:.6f}, {pose1.altitude:.2f})")
        print(f"  Pose 2: ({pose2.latitude:.6f}, {pose2.longitude:.6f}, {pose2.altitude:.2f})")
        
        R_init, t_init = icp.gps_to_local_transform(pose1, pose2)
        print(f"\nGPS-based initial translation: {t_init}")
        
        result = icp.align(source, target, initial_R=R_init, initial_t=t_init)
        print(f"\nICP with GPS initialization:")
    else:
        result = icp.align(source, target)
        print(f"\nICP without initialization:")
    
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final error: {result['error']:.6f} m")
    print(f"  Correspondences: {result['num_correspondences']}")
    print(f"  Translation: {result['t']}")
    
    # Visualize
    ICPAligner.visualize_alignment(source, target, result['R'], result['t'])
    
    return result


# Example usage
if __name__ == "__main__":
    print("Matched scans:", len(dataset))

    #Trajectory colored by t
    dataset.plot_trajectory(color_by_dt=True)
    # Compare ICP with and without GPS initialization
    if len(dataset) >= 2:
        print("\n" + "="*60)
        print("WITHOUT GPS initialization:")
        result_no_gps = align_consecutive_scans(dataset, 0, 1, use_gps_init=False)
        
        print("\n" + "="*60)
        print("WITH GPS initialization:")
        result_gps = align_consecutive_scans(dataset, 0, 1, use_gps_init=True)