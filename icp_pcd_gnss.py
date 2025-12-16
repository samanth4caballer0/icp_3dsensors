import os
import glob
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class LidarGnssDataset:
    """
      1) Load GNSS (fix.csv) and LiDAR (.pcd) data
      2) Match LiDAR scans to the nearest GNSS wrt time
      3) Provide scan + pose access via indexing
      4) Plot matched trajectory and time differences
    """

    def __init__(self, gps_readings_dir, lidar_pcd_dir, tolerance_ns):
    
        self.gps_readings_dir = gps_readings_dir
        self.lidar_pcd_dir = lidar_pcd_dir
        self.tolerance_ns = tolerance_ns

        self.gnss_df = self._load_gnss_files()          # Load GNSS data frame
        self.pcd_df = self._load_pcd_files()            # Load LiDAR PCD data frame
        self.matched = self._match_lidar_to_gnss()      # Match LiDAR and GNSS together

        # Keep only rows that actually got a match
        self.matched = self.matched.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Data loading 
    # ------------------------------------------------------------------ #
    def _load_gnss_files(self):
        # Read the CSV file
        gnss_readings_df = pd.read_csv(self.gps_readings_dir)
        
        # concatenate and convert to nanoseconds (to match LiDAR timestamps)
        gnss_readings_df["t_ns"] = gnss_readings_df["header_stamp_secs"] * 10**9 + gnss_readings_df["header_stamp_nsecs"] 
        
        # Sort readings by timestamp
        gnss_readings_df = gnss_readings_df.sort_values("t_ns").reset_index(drop=True)
        return gnss_readings_df

    def _load_pcd_files(self):
        # Find all .pcd files 
        pcd_files = sorted(glob.glob(os.path.join(self.lidar_pcd_dir, "*.pcd")))
        
        # Create dataframe with timestamps extracted from pcd filenames
        pcd_df = pd.DataFrame({"pcd_file": pcd_files})
        pcd_df["stamp_str"] = pcd_df["pcd_file"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        pcd_df["t_ns"] = pd.to_numeric(pcd_df["stamp_str"], errors="coerce")

        # Sort by timestamp
        pcd_df = pcd_df.sort_values("t_ns").reset_index(drop=True)
        return pcd_df

    def _match_lidar_to_gnss(self):
        # Match each LiDAR scan to the nearest GNSS fix in time using pandas.merge_asof        
        matched = pd.merge_asof(
            self.pcd_df.sort_values("t_ns"),
            self.gnss_df.sort_values("t_ns"),
            on="t_ns",
            direction="nearest",
            tolerance=self.tolerance_ns)
        return matched

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    
    def plot_trajectory(self):
        # Plot the trajectory of matched LiDAR–GNSS points
        plt.scatter(
            self.matched["longitude"],
            self.matched["latitude"],
            s=5,
        )
        plt.title("Matched LiDAR–GNSS Trajectory")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis("equal")
        plt.show()
    
    def visualize_global_map(self, merged):
        # Build and plot the merged global point cloud
        o3d.visualization.draw_geometries([merged])

    
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.matched)        # Number of matched LiDAR–GNSS pairs
    
    
    def get_scan_info(self, idx):       #Return basic info (path + pose metadata) about the scan+pose at index idx.
        row = self.matched.iloc[idx]

        pose = {
            "t_ns": int(row["t_ns"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "altitude": float(row.get("altitude", 0.0)),
        }

        return {
            "pcd_file": row["pcd_file"],
            "t_ns": int(row["t_ns"]),
            "pose": pose,
        }    
    
    def pcd_read(self, pcd_file):       # alternative to using o3d.io.read_point_cloud function
        """Reads an ASCII .pcd file, parses XYZ points after the DATA header into a NumPy array.
        Returns an Open3D PointCloud containing only those XYZ points"""
        with open(pcd_file, 'r') as f:
            lines = f.readlines()

            # Find where DATA starts
            data_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("DATA"):
                    data_idx = i + 1
                    break

        # Read points as numpy array
        points = []
        for line in lines[data_idx:]:
            vals = line.strip().split()
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            points.append([x, y, z])

        points = np.array(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd      
    
    def load_pcd(self, idx: int) -> o3d.geometry.PointCloud:
        """
        Load the LiDAR scan at index `idx` as an Open3D point cloud
        """
        info = self.get_scan_info(idx)  #input index of the scan to load
        pcd_path = info["pcd_file"]     
        pcd = self.pcd_read(pcd_path)   # load point cloud from .pcd file         
        return pcd                      
    
# -----------------------------main------------------------------- #

dataset = LidarGnssDataset(
    gps_readings_dir="parkolo1/fix.csv",
    lidar_pcd_dir="parkolo1/pcd",
    tolerance_ns=500_000_000,
)

# GNSS Trajectory
dataset.plot_trajectory()

# ------------------------------------------------------------------ #
# ICP-based global map merging
def merge_dataset_to_global_map_icp(
    dataset: LidarGnssDataset,
    voxel_size: 0.2,            #for downsampling scans 
    max_correspondence_distance: 1.0,   
    xyz_out_path,           #to save the merged global points to a plain-text .xyz file
    traj_out_path,          #to save ICP odometry trajectory translations (x y z per line)
    ):

    # ----------------------------
    # LLA -> ECEF -> ENU
    # ----------------------------
    def lla_to_ecef(lat_deg, lon_deg, alt_m):
        a = 6378137.0               # equatorial radius [m] (referenced from WGS84 world geodetic system)
        e_sq = 6.69437999014e-3     # first eccentricity squared

        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = a / np.sqrt(1.0 - e_sq * sin_lat**2)    # prime vertical radius of curvature

        x = (N + alt_m) * cos_lat * cos_lon
        y = (N + alt_m) * cos_lat * sin_lon
        z = (N * (1.0 - e_sq) + alt_m) * sin_lat

        return np.array([x, y, z])

    def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m):   
        # Origin in ECEF
        x0, y0, z0 = lla_to_ecef(lat0_deg, lon0_deg, alt0_m)
        dx, dy, dz = x - x0, y - y0, z - z0

        lat0 = np.deg2rad(lat0_deg)
        lon0 = np.deg2rad(lon0_deg)

        sin_lat0 = np.sin(lat0)
        cos_lat0 = np.cos(lat0)
        sin_lon0 = np.sin(lon0)
        cos_lon0 = np.cos(lon0)

        R = np.array([                  # ECEF to ENU rotation matrix
            [-sin_lon0,               cos_lon0,              0.0],
            [-sin_lat0 * cos_lon0,   -sin_lat0 * sin_lon0,   cos_lat0],
            [ cos_lat0 * cos_lon0,    cos_lat0 * sin_lon0,   sin_lat0],
        ])

        return R @ np.array([dx, dy, dz])  # (e, n, u)

    # ---------------------------- HERE STARTS THE ACTUAL PROCESSING ----------------------------
    
    # ENU positions for all scans (GNSS-only)
    # ----------------------------
    # Use first scan pose as ENU origin
    ref_info = dataset.get_scan_info(0)         #origin gnns pose from first scan
    ref_pose = ref_info["pose"]
    lat0, lon0, alt0 = ref_pose["latitude"], ref_pose["longitude"], ref_pose["altitude"]

    enu_positions = []
    for i in range(len(dataset)):
        info = dataset.get_scan_info(i)
        pose = info["pose"]
        x_ecef, y_ecef, z_ecef = lla_to_ecef(pose["latitude"], pose["longitude"], pose["altitude"])
        e, n, u = ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
        enu_positions.append(np.array([e, n, u], dtype=np.float64))
    enu_positions = np.stack(enu_positions, axis=0)  # (N, 3)

    # ----------------------------
    # ICP + transform accumulation
    # ----------------------------
    # Load first scan
    pcd0 = dataset.load_pcd(0)

    # INITIALIZATION - store global transforms and merged points
    T_global_list = [np.eye(4)]                     # 4×4 matrix that transforms scan i → global frame. First scan is identity (no transform)  
    global_points = [np.asarray(pcd0.points)]       # Store first scan's points (already in global frame)    
    trajectory_icp = [np.zeros(3)]                  # Trajectory starts at origin [0, 0, 0]
    registration = o3d.pipelines.registration       # Open3D module doing actual icp algorithm 
    prev_pcd = pcd0                                 # Initialize with first scan (no need to reload in loop)

    for i in range(1, len(dataset)):                # for each subsequent scan
        pcd_i = dataset.load_pcd(i)                 # Load current scan
        
        # GNSS-based initial guess between current (source) and previous (target)
        pos_prev = enu_positions[i - 1]
        pos_curr = enu_positions[i]

        # Transformation that maps source (current) -> target (previous) based on GNSS delta
        delta = pos_curr - pos_prev                 # ENU translation from previous (GNSS) to current scan. 
        trans_init = np.eye(4)          
        trans_init[:3, 3] = delta                   # Initial guess transformation matrix
        # print(f"Scan {i}: GNSS delta (ENU) = {delta}")
        
        # run ICP registration  
        icp_result = registration.registration_icp(
            source=pcd_i,                                               # Current scan (to be aligned)
            target=prev_pcd,                                            # Previous scan (reference)
            max_correspondence_distance=max_correspondence_distance,    
            init=trans_init,                                            # GNSS-based starting guess
            estimation_method=registration.TransformationEstimationPointToPoint(),      # Point-to-point ICP
        )

        T_prev_curr = icp_result.transformation                         # maps current -> previous
        
        if icp_result.inlier_rmse < 0.3:
            # Accumulate to global:
            T_prev_global = T_global_list[-1]                           # map previous scan to global transform = accumulated transform up to previous scan
            T_curr_global = T_prev_global @ T_prev_curr                 # current scan → previous scan (from ICP) @ previous scan → global (accumulated)
            T_global_list.append(T_curr_global)                         # Store current scan → global transform

            # Record translation component (odometry pose)
            trajectory_icp.append(T_curr_global[:3, 3])

            # Transform current (full resolution) scan to global and store points
            pts = np.asarray(pcd_i.points)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])        # (N,4) homogeneous[x,y,z,1]
            pts_global = (T_curr_global @ pts_h.T).T[:, :3]             # Transform to global frame
            global_points.append(pts_global)                    

            # Update previous scan for next iteration's ICP target
            prev_pcd = pcd_i
        else:
            # If rejected, repeat previous pose
            T_global_list.append(T_global_list[-1].copy())
            trajectory_icp.append(T_global_list[-1][:3, 3])

    # ----------------------------
    # Merge and return as Open3D PointCloud
    # ----------------------------
    merged_pts = np.vstack(global_points)

    # Save merged point cloud to file in xyz format
    if xyz_out_path:
        os.makedirs(os.path.dirname(xyz_out_path), exist_ok=True)
        # Write "x y z" per line
        with open(xyz_out_path, "w") as f:
            np.savetxt(f, merged_pts, fmt="%.6f")

    # Save ICP trajectory to file
    trajectory_icp = np.vstack(trajectory_icp)
    if traj_out_path:
        os.makedirs(os.path.dirname(traj_out_path), exist_ok=True)
        np.savetxt(traj_out_path, trajectory_icp, fmt="%.6f")

    # Return merged point cloud and trajectory
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
    return merged_pcd, trajectory_icp

# ----------------------------- Run ICP-based merging ------------------------------- #
merged_icp, traj_icp = merge_dataset_to_global_map_icp(
    dataset,
    xyz_out_path="parkolo1/outputs/merged_global.xyz",
    traj_out_path="parkolo1/outputs/trajectory_icp.txt",
    voxel_size=0.2,
    max_correspondence_distance=1.0)

dataset.visualize_global_map(merged_icp)
print("ICP trajectory shape:", traj_icp.shape)
