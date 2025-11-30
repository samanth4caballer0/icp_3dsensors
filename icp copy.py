import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import subprocess
import shutil


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
            plt.colorbar(sc, label="Δt (LiDAR - GNSS) [ms]")
            plt.title("Matched LiDAR–GNSS Trajectory (colored by Δt)")
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
    
    def load_pcd(self, idx: int) -> o3d.geometry.PointCloud:
        """
        Load the LiDAR scan at index `idx` as an Open3D point cloud.

        :param idx: Index into the matched dataset (0 <= idx < len(self))
        :return: open3d.geometry.PointCloud
        """
        info = self.get_scan_info(idx)
        pcd_path = info["pcd_file"]
        pcd = o3d.io.read_point_cloud(pcd_path)

        return pcd
# ------------------------------------------------------------------ #

dataset = LidarGnssDataset(
    gps_readings_dir="parkolo1/fix.csv",
    lidar_pcd_dir="parkolo1/pcd",
    tolerance_ns=500_000_000,
)

# Trajectory colored by time difference
dataset.plot_trajectory(color_by_dt=True)

#print the dataset trayectory


# Load the first matched scan as a point cloud
#idx = 0
#scan_info = dataset[idx]
# print("PCD file:", scan_info["pcd_file"])
# print("Pose (lat, lon, alt):", scan_info["pose"].latitude,
#       scan_info["pose"].longitude, scan_info["pose"].altitude)

# pcd0 = dataset.load_pcd(idx)
# o3d.visualization.draw_geometries([pcd0])           # Visualize it in an Open3D window


import numpy as np
import open3d as o3d

def merge_dataset_to_global_map(dataset: LidarGnssDataset) -> o3d.geometry.PointCloud:
    """
    Build a merged global point cloud from a LidarGnssDataset.

    - Uses GNSS latitude/longitude/altitude to compute a local ENU frame.
    - First scan pose is used as the ENU origin.
    - Assumes LiDAR frame is already aligned with ENU axes (translation only).

    Parameters
    ----------
    dataset : LidarGnssDataset
        Dataset with matched LiDAR scans and GNSS poses.

    Returns
    -------
    merged_pcd : o3d.geometry.PointCloud
        Global merged point cloud in ENU coordinates (meters).
    """

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty, cannot build global map.")

    # ----------------------------
    # WGS84 helpers: LLA -> ECEF -> ENU
    # ----------------------------
    def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
        """
        Convert geodetic coordinates (lat, lon, alt) to ECEF (x, y, z) in meters.
        WGS84 ellipsoid.
        """
        # WGS84 constants
        a = 6378137.0               # semi-major axis [m]
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
        """
        Convert ECEF (x, y, z) to ENU coordinates (e, n, u) w.r.t. an origin
        defined by (lat0, lon0, alt0).
        """
        # Origin in ECEF
        x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)

        # Translate so origin is at (0,0,0)
        dx = x - x0
        dy = y - y0
        dz = z - z0

        lat0 = np.deg2rad(lat0_deg)
        lon0 = np.deg2rad(lon0_deg)

        sin_lat0 = np.sin(lat0)
        cos_lat0 = np.cos(lat0)
        sin_lon0 = np.sin(lon0)
        cos_lon0 = np.cos(lon0)

        # ECEF -> ENU rotation matrix
        R = np.array([
            [-sin_lon0,               cos_lon0,              0.0],
            [-sin_lat0 * cos_lon0,   -sin_lat0 * sin_lon0,   cos_lat0],
            [ cos_lat0 * cos_lon0,    cos_lat0 * sin_lon0,   sin_lat0],
        ])

        enu = R @ np.array([dx, dy, dz])
        return enu  # (e, n, u)

    # ----------------------------
    # Prepare ENU origin (first scan)
    # ----------------------------
    ref_info = dataset.get_scan_info(0)
    ref_pose = ref_info["pose"]
    lat0 = ref_pose.latitude
    lon0 = ref_pose.longitude
    alt0 = ref_pose.altitude

    all_points = []

    # ----------------------------
    # For each scan: compute ENU translation and move points
    # ----------------------------
    for i in range(len(dataset)):
        info = dataset.get_scan_info(i)
        pose = info["pose"]

        # GNSS position of this scan
        lat = pose.latitude
        lon = pose.longitude
        alt = pose.altitude

        # ECEF of this scan
        x_ecef, y_ecef, z_ecef = geodetic_to_ecef(lat, lon, alt)

        # ENU translation of this scan w.r.t origin
        e, n, u = ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)
        translation = np.array([e, n, u], dtype=np.float64)

        # Load PCD and translate its points
        pcd = dataset.load_pcd(i)
        if len(pcd.points) == 0:
            continue  # skip empty scans

        pts = np.asarray(pcd.points, dtype=np.float64)
        pts_global = pts + translation  # broadcast translation

        all_points.append(pts_global)

    if not all_points:
        raise RuntimeError("No points found in any scan; cannot build global map.")

    # ----------------------------
    # Merge into a single Open3D point cloud
    # ----------------------------
    merged_pts = np.vstack(all_points)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)

    return merged_pcd
# Build global map in ENU frame
merged = merge_dataset_to_global_map(dataset)

# Visualize using your helper
dataset.visualize_global_map(merged)

def ensure_git_remote(remote_name: str, remote_url: str) -> None:
    """
    Ensure a git repository exists in the current directory and the given remote is set.
    If remote exists with different URL it will be updated.
    """
    # Init repo if missing
    if not os.path.isdir(".git"):
        subprocess.run(["git", "init"], check=True)
    # Get current remotes
    res = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    existing = {}
    for line in res.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            existing[parts[0]] = parts[1]
    if remote_name in existing:
        if existing[remote_name] != remote_url:
            subprocess.run(["git", "remote", "remove", remote_name], check=True)
            subprocess.run(["git", "remote", "add", remote_name, remote_url], check=True)
    else:
        subprocess.run(["git", "remote", "add", remote_name, remote_url], check=True)
    print(f"Remote '{remote_name}' -> {remote_url} configured.")

# Configure remote (edit URL as needed)
ensure_git_remote("origin", "https://github.com/your-user/your-repo.git")

# Optional first commit & push (uncomment when ready):
# subprocess.run(["git", "add", "."], check=True)
# subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
# subprocess.run(["git", "branch", "-M", "main"], check=True)
# subprocess.run(["git", "push", "-u", "origin", "main"], check=True)



