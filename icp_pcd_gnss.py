import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class Pose:
    """Simple container for a GNSS pose."""
    t_ns: int
    latitude: float
    longitude: float
    altitude: float


class LidarGnssDataset:
    """
      1) Load GNSS (fix.csv) and LiDAR (.pcd) data
      2) Match LiDAR scans to the nearest GNSS fix in time
      3) Provide scan + pose access via indexing
      4) Visualize matched trajectory and time differences
    """

    def __init__(
        self,
        gps_readings_dir: str,
        lidar_pcd_dir: str,
        tolerance_ns: int = 500_000_000,  # 0.5 seconds
    ) -> None:
        
        self.gps_readings_dir = gps_readings_dir
        self.lidar_pcd_dir = lidar_pcd_dir
        self.tolerance_ns = tolerance_ns

        self.gnss_df = self._load_gnss_files()          # Load GNSS data frame
        self.pcd_df = self._load_pcd_files()            # Load LiDAR PCD data frame
        self.matched = self._match_lidar_to_gnss()      # Match LiDAR and GNSS together

        # Keep only rows that actually got a match (no NaNs in key GNSS columns)
        self.matched = self.matched.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Internal loaders
    # ------------------------------------------------------------------ #
    def _load_gnss_files(self) -> pd.DataFrame:
        """ Loads GPS measurements from a CSV file into a pandas DataFrame"""
        # Read the CSV file
        gnss_readings_df = pd.read_csv(self.gps_readings_dir)
        
        # concatenate and convert to nanoseconds
        gnss_readings_df["t_ns"] = gnss_readings_df["header_stamp_secs"] * 10**9 + gnss_readings_df["header_stamp_nsecs"] 
        
        # Sort readings by timestamp
        gnss_readings_df = gnss_readings_df.sort_values("t_ns").reset_index(drop=True)
        return gnss_readings_df

    def _load_pcd_files(self) -> pd.DataFrame:
        """Scan pcd_dir for .pcd files and extract timestamp from filename"""
        # Find all .pcd files 
        pcd_files = sorted(glob.glob(os.path.join(self.lidar_pcd_dir, "*.pcd")))
        
        # Create DataFrame with timestamps extracted from filenames
        pcd_df = pd.DataFrame({"pcd_file": pcd_files})
        pcd_df["stamp_str"] = pcd_df["pcd_file"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        pcd_df["t_ns"] = pd.to_numeric(pcd_df["stamp_str"], errors="coerce")

        # Sort by timestamp
        pcd_df = pcd_df.sort_values("t_ns").reset_index(drop=True)
        return pcd_df

    def _match_lidar_to_gnss(self) -> pd.DataFrame:
        """
        Match each LiDAR scan to the nearest GNSS fix in time using pandas.merge_asof.
        """
        
        matched = pd.merge_asof(
            self.pcd_df.sort_values("t_ns"),
            self.gnss_df.sort_values("t_ns"),
            on="t_ns",
            direction="nearest",
            tolerance=self.tolerance_ns)
        return matched

    def _ensure_time_diff(self) -> None:
        """Compute time difference between LiDAR and GNSS in ms"""

        self.matched["gnss_t_ns"] = (
            self.matched["header_stamp_secs"] * 10**9
            + self.matched["header_stamp_nsecs"])
        
        self.matched["dt_ms"] = (self.matched["t_ns"] - self.matched["gnss_t_ns"]) / 1e6


    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    def plot_trajectory(self, color_by_dt: bool = False) -> None:
        """
        Plot the trajectory of matched LiDAR–GNSS points.
        """
        if len(self.matched) == 0:
            raise RuntimeError("No matched scans to plot.")

        if color_by_dt:
            self._ensure_time_diff()
            c = self.matched["dt_ms"]
            sc = plt.scatter(
                self.matched["longitude"],
                self.matched["latitude"],
                s=5,
                c=c,
            )
            plt.colorbar(sc, label="(LiDAR - GNSS) time difference in [ms]")
            plt.title("Matched LIDAR–GNSS Trajectory")
        else:
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
        """
        Build and visualize the merged global point cloud.
        """

        o3d.visualization.draw_geometries([merged])

    
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.matched)

    def get_scan_info(self, idx: int) -> Dict[str, Any]:
        """
        Return basic info about the scan+pose at index idx.
        Does NOT load the PCD file itself (just path + pose metadata).
        """
        row = self.matched.iloc[idx]

        pose = Pose(
            t_ns=int(row["t_ns"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            altitude=float(row.get("altitude", 0.0)),
        )

        return {
            "pcd_file": row["pcd_file"],
            "t_ns": int(row["t_ns"]),
            "pose": pose,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Alias for get_scan_info so you can do: dataset[i]"""
        return self.get_scan_info(idx)
    
                                                                                   
    def iter_scans(self):
        """Convenience generator over all scans."""
        for i in range(len(self)):
            yield self.get_scan_info(i)
    
    def pcd_read(self, pcd_file):
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
        Load the LiDAR scan at index `idx` as an Open3D point cloud.

        :param idx: Index into the matched dataset (0 <= idx < len(self))
        :return: open3d.geometry.PointCloud
        """
        info = self.get_scan_info(idx)
        pcd_path = info["pcd_file"]
        # pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = self.pcd_read(pcd_path)            
        return pcd
    
    
# ------------------------------------------------------------------ #

dataset = LidarGnssDataset(
    gps_readings_dir="parkolo1/fix.csv",
    lidar_pcd_dir="parkolo1/pcd",
    tolerance_ns=500_000_000,
)

# Trajectory colored by time difference
dataset.plot_trajectory(color_by_dt=True)


def merge_dataset_to_global_map_icp(
    dataset: LidarGnssDataset,
    voxel_size: float = 0.2,
    max_correspondence_distance: float = 1.0,
    xyz_out_path: Optional[str] = None,
    traj_out_path: Optional[str] = None,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Build a merged global point cloud from a LidarGnssDataset.

    - GNSS is used only to provide a trans_init (initial guess) for ICP
      between consecutive scans.
    - ICP refines the relative transform between current and previous scan.
    - Transforms are accumulated to build a global map.

    Parameters
    ----------
    dataset : LidarGnssDataset
        Dataset with matched LiDAR scans and GNSS poses.
    voxel_size : float
        Voxel size for downsampling scans used in ICP.
    max_correspondence_distance : float
        Max correspondence distance (meters) for ICP.
    xyz_out_path : Optional[str]
        If provided, saves the merged global points to a plain-text .xyz file.
    traj_out_path : Optional[str]
        If provided, saves ICP odometry trajectory translations (x y z per line).

    Returns
    -------
    merged_pcd, trajectory_icp
        trajectory_icp shape (M,3), M = number of accepted poses.
    """

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty, cannot build global map.")

    # ----------------------------
    # WGS84 helpers: LLA -> ECEF -> ENU
    # ----------------------------
    def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
        a = 6378137.0               # WGS84 semi-major axis [m]
        e_sq = 6.69437999014e-3     # first eccentricity squared

        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = a / np.sqrt(1.0 - e_sq * sin_lat**2)

        x = (N + alt_m) * cos_lat * cos_lon
        y = (N + alt_m) * cos_lat * sin_lon
        z = (N * (1.0 - e_sq) + alt_m) * sin_lat

        return np.array([x, y, z])

    def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m):
        # Origin in ECEF
        x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
        dx, dy, dz = x - x0, y - y0, z - z0

        lat0 = np.deg2rad(lat0_deg)
        lon0 = np.deg2rad(lon0_deg)

        sin_lat0 = np.sin(lat0)
        cos_lat0 = np.cos(lat0)
        sin_lon0 = np.sin(lon0)
        cos_lon0 = np.cos(lon0)

        R = np.array([
            [-sin_lon0,               cos_lon0,              0.0],
            [-sin_lat0 * cos_lon0,   -sin_lat0 * sin_lon0,   cos_lat0],
            [ cos_lat0 * cos_lon0,    cos_lat0 * sin_lon0,   sin_lat0],
        ])

        return R @ np.array([dx, dy, dz])  # (e, n, u)

    # ----------------------------
    # ENU positions for all scans (GNSS-only)
    # ----------------------------
    # Use first scan pose as ENU origin
    ref_info = dataset.get_scan_info(0)
    ref_pose = ref_info["pose"]
    lat0, lon0, alt0 = ref_pose.latitude, ref_pose.longitude, ref_pose.altitude

    enu_positions = []
    for i in range(len(dataset)):
        info = dataset.get_scan_info(i)
        pose = info["pose"]
        x_ecef, y_ecef, z_ecef = geodetic_to_ecef(pose.latitude, pose.longitude, pose.altitude)
        e, n, u = ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
        enu_positions.append(np.array([e, n, u], dtype=np.float64))
    enu_positions = np.stack(enu_positions, axis=0)  # (N, 3)

    # ----------------------------
    # ICP + transform accumulation
    # ----------------------------
    # Load first scan
    pcd0 = dataset.load_pcd(0)
    pcd0_down = pcd0.voxel_down_sample(voxel_size)

    # Global transform list: T_global[i] maps scan i -> global
    T_global_list = [np.eye(4)]
    global_points = [np.asarray(pcd0.points)]
    trajectory_icp = [np.zeros(3)]  # origin translation

    registration = o3d.pipelines.registration

    for i in range(1, len(dataset)):
        # Current scan
        prev_pcd_down = dataset.load_pcd(i - 1)
        pcd_i = dataset.load_pcd(i)
        if len(pcd_i.points) == 0:
            # skip empty scans
            T_global_list.append(T_global_list[-1].copy())
            trajectory_icp.append(trajectory_icp[-1])
            continue

        #pcd_i_down = pcd_i.voxel_down_sample(voxel_size)
        pcd_i_down = pcd_i
        # GNSS-based initial guess between current (source) and previous (target)
        pos_prev = enu_positions[i - 1]
        pos_curr = enu_positions[i]

        # Transformation that maps source (current) -> target (previous)
        # t = p_curr - p_prev (see derivation)
        delta = pos_curr - pos_prev
        trans_init = np.eye(4)
        trans_init[:3, 3] = delta

        # ICP: source = current, target = previous
        icp_result = registration.registration_icp(
            source=pcd_i_down,
            target=prev_pcd_down,
            max_correspondence_distance=max_correspondence_distance,
            init=trans_init,
            estimation_method=registration.TransformationEstimationPointToPoint(),
        )

        T_prev_curr = icp_result.transformation  # maps current -> previous
        if icp_result.inlier_rmse < 0.3:
            # Accumulate to global:
            # T_curr_global = T_prev_global @ T_prev_curr
            T_prev_global = T_global_list[-1]
            T_curr_global = T_prev_global @ T_prev_curr
            T_global_list.append(T_curr_global)

            # Record translation component (odometry pose)
            trajectory_icp.append(T_curr_global[:3, 3])

            # Transform current *full-resolution* scan to global and store points
            pts = np.asarray(pcd_i.points)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N,4)
            pts_global = (T_curr_global @ pts_h.T).T[:, :3]
            global_points.append(pts_global)

            # Update previous downsampled scan (always in local frame)
            prev_pcd_down = pcd_i_down
        else:
            # If rejected, repeat previous pose
            T_global_list.append(T_global_list[-1].copy())
            trajectory_icp.append(T_global_list[-1][:3, 3])

    # ----------------------------
    # Merge and return as Open3D PointCloud
    # ----------------------------
    merged_pts = np.vstack(global_points)

    if xyz_out_path:
        os.makedirs(os.path.dirname(xyz_out_path), exist_ok=True)
        # Write "x y z" per line
        with open(xyz_out_path, "w") as f:
            np.savetxt(f, merged_pts, fmt="%.6f")

    trajectory_icp = np.vstack(trajectory_icp)
    if traj_out_path:
        os.makedirs(os.path.dirname(traj_out_path), exist_ok=True)
        np.savetxt(traj_out_path, trajectory_icp, fmt="%.6f")

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
    return merged_pcd, trajectory_icp

merged_icp, traj_icp = merge_dataset_to_global_map_icp(
    dataset,
    xyz_out_path="parkolo1/outputs/merged_global.xyz",
    traj_out_path="parkolo1/outputs/trajectory_icp.txt",
)
dataset.visualize_global_map(merged_icp)
print("ICP trajectory shape:", traj_icp.shape)
