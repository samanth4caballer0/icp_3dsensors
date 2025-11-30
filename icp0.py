import os
import glob
from dataclasses import dataclass
from typing import Dict, Any
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
      5) Load individual scans as Open3D point clouds
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
        """Loads GPS measurements from a CSV file into a pandas DataFrame."""
        gnss_readings_df = pd.read_csv(self.gps_readings_dir)
        
        # concatenate and convert to nanoseconds
        gnss_readings_df["t_ns"] = (
            gnss_readings_df["header_stamp_secs"] * 10**9
            + gnss_readings_df["header_stamp_nsecs"]
        )
        
        # Sort readings by timestamp
        gnss_readings_df = gnss_readings_df.sort_values("t_ns").reset_index(drop=True)
        return gnss_readings_df

    def _load_pcd_files(self) -> pd.DataFrame:
        """Scan pcd_dir for .pcd files and extract timestamp from filename."""
        pcd_files = sorted(glob.glob(os.path.join(self.lidar_pcd_dir, "*.pcd")))
        if not pcd_files:
            raise FileNotFoundError(f"No .pcd files found in {self.lidar_pcd_dir}")
        
        # Create DataFrame with timestamps extracted from filenames
        pcd_df = pd.DataFrame({"pcd_file": pcd_files})
        pcd_df["stamp_str"] = pcd_df["pcd_file"].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )
        pcd_df["t_ns"] = pd.to_numeric(pcd_df["stamp_str"], errors="coerce")

        if pcd_df["t_ns"].isna().any():
            bad = pcd_df[pcd_df["t_ns"].isna()]["pcd_file"].tolist()
            raise ValueError(
                f"Some PCD filenames could not be parsed as integer timestamps: {bad}"
            )

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
            tolerance=self.tolerance_ns,
        )
        return matched

    def _ensure_time_diff(self) -> None:
        """Compute time difference between LiDAR and GNSS in ms."""
        self.matched["gnss_t_ns"] = (
            self.matched["header_stamp_secs"] * 10**9
            + self.matched["header_stamp_nsecs"]
        )
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
        """Alias for get_scan_info so you can do: dataset[i]."""
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

    def icp_between(
        self,
        idx_source: int,
        idx_target: int,
        voxel_size: float = 0.1,
        max_corr_dist: float = 0.5,
    ):
        """
        Run point-to-point ICP between two scans (source -> target).

        Args:
            idx_source: index of source scan (will be transformed).
            idx_target: index of target scan (reference).
            voxel_size: voxel size for downsampling (meters).
            max_corr_dist: max correspondence distance for ICP (meters).

        Returns:
            T : 4x4 np.ndarray, transform that maps source into target frame
            result : Open3D RegistrationResult object
        """
        # 1. Load point clouds
        source = self.load_pcd(idx_source)
        target = self.load_pcd(idx_target)

        # 2. Downsample for robustness and speed
        source_ds = source.voxel_down_sample(voxel_size)
        target_ds = target.voxel_down_sample(voxel_size)

        # 3. Remove non-finite points (if any)
        source_ds.remove_non_finite_points()
        target_ds.remove_non_finite_points()

        # 4. Initial guess (identity)
        init = np.eye(4)

        # 5. ICP (point-to-point)
        result = o3d.pipelines.registration.registration_icp(
            source_ds,
            target_ds,
            max_corr_dist,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        T = result.transformation  # 4x4 matrix
        return T, result


# ------------------------------------------------------------------ #
# ICP trajectory + map merging helpers (outside the class)
# ------------------------------------------------------------------ #

import copy
import numpy as np
import open3d as o3d

def debug_icp_pair(dataset, i, j, voxel_size=0.05, max_corr_dist=0.5):
    src = dataset.load_pcd(i)
    tgt = dataset.load_pcd(j)

    # Color original clouds
    src.paint_uniform_color([1, 0, 0])   # red
    tgt.paint_uniform_color([0, 1, 0])   # green

    print(f"Showing raw scans {i} (red) and {j} (green)")
    o3d.visualization.draw_geometries([src, tgt])

    # Run ICP
    T, res = dataset.icp_between(
        idx_source=i,
        idx_target=j,
        voxel_size=voxel_size,
        max_corr_dist=max_corr_dist,
    )
    print("ICP fitness:", res.fitness, "RMSE:", res.inlier_rmse)
    print("T =\n", T)

    # Transform source with ICP result
    src_aligned = copy.deepcopy(dataset.load_pcd(i))
    src_aligned.transform(T)
    src_aligned.paint_uniform_color([0, 0, 1])  # blue

    print(f"Showing ICP-aligned scan {i} (blue) vs target {j} (green)")
    o3d.visualization.draw_geometries([src_aligned, tgt])



def build_icp_trajectory(
    dataset: LidarGnssDataset,
    num_scans: int = None,
    voxel_size: float = 0.1,
    max_corr_dist: float = 0.5,
):
    """
    Compute a global pose for each scan using ICP odometry (chain of relative transforms).

    Returns:
        poses: list of 4x4 np.ndarray, one per scan in [0, num_scans)
               Pose of scan i is T_world_i.
    """
    if num_scans is None:
        num_scans = len(dataset)
    num_scans = min(num_scans, len(dataset))

    poses = []
    # Pose of scan 0 in world frame: identity
    T_world_0 = np.eye(4)
    poses.append(T_world_0)

    for i in range(1, num_scans):
        print(f"Running ICP between scans {i-1} (target) and {i} (source)...")

        # Transform from scan i -> scan i-1 (target frame)
        T_i_to_prev, result = dataset.icp_between(
            idx_source=i,
            idx_target=i - 1,
            voxel_size=voxel_size,
            max_corr_dist=max_corr_dist,
        )
        print(f"  fitness={result.fitness:.3f}, inlier_rmse={result.inlier_rmse:.3f}")

        # Compose: T_world_i = T_world_{i-1} * T_i_to_prev
        T_world_prev = poses[i - 1]
        T_world_i = T_world_prev @ T_i_to_prev
        poses.append(T_world_i)

    return poses


def merge_scans_to_map(
    dataset: LidarGnssDataset,
    poses,
    voxel_size: float = 0.1,
    color_each_scan: bool = False,
):
    """
    Merge all scans using given global poses.

    Args:
        dataset: LidarGnssDataset instance.
        poses: list of 4x4 transforms (T_world_i for each scan i).
        voxel_size: downsampling voxel size (meters).
        color_each_scan: if True, assign a random color per scan.

    Returns:
        merged: open3d.geometry.PointCloud with all scans merged.
        geoms: list of individual transformed point clouds (one per scan).
    """
    merged = o3d.geometry.PointCloud()
    geoms = []

    for i, T_world_i in enumerate(poses):
        pc = dataset.load_pcd(i)

        # Optional downsampling for speed and memory
        pc = pc.voxel_down_sample(voxel_size)

        # Transform into world frame
        pc.transform(T_world_i)

        if color_each_scan:
            pc.paint_uniform_color(np.random.rand(3))

        merged += pc
        geoms.append(pc)

    return merged, geoms

def merge_scans_to_map_with_time_z(
    dataset: LidarGnssDataset,
    poses,
    dz: float = 0.05,         # height step per scan
    voxel_size: float = 0.02,
):
    merged = o3d.geometry.PointCloud()
    geoms = []

    for i, T_world_i in enumerate(poses):
        pc = dataset.load_pcd(i)
        pts = np.asarray(pc.points)

        # Give each scan a different Z according to its index
        pts[:, 2] = i * dz
        pc.points = o3d.utility.Vector3dVector(pts)

        # Optional downsampling
        if voxel_size > 0:
            pc = pc.voxel_down_sample(voxel_size)

        # Transform by pose (XY motion still in the plane)
        pc.transform(T_world_i)

        # Color by scan index
        color = np.random.rand(3)
        pc.paint_uniform_color(color)

        merged += pc
        geoms.append(pc)

    return merged, geoms

# ------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    dataset = LidarGnssDataset(
        gps_readings_dir="parkolo1/fix.csv",
        lidar_pcd_dir="parkolo1/pcd",
        tolerance_ns=500_000_000,
    )

    print("Matched scans:", len(dataset))

    # Plot trajectory colored by time difference
    dataset.plot_trajectory(color_by_dt=True)

    ###################### DEBUGGING ##############################    
    # pc = dataset.load_pcd(0)
    # o3d.visualization.draw_geometries([pc])
    # print(dataset.pcd_df["pcd_file"].iloc[0])
    
    # with open(dataset.pcd_df["pcd_file"].iloc[0], "r") as f:
    #     for _ in range(30):
    #         print(f.readline().strip())

    # pc = dataset.load_pcd(0)
    # pts = np.asarray(pc.points)
    # print("Points shape:", pts.shape)
    # print("First 5 points:\n", pts[:5])
    
    # pc = dataset.load_pcd(0)
    # print("Num points:", np.asarray(pc.points).shape)
    # debug_icp_pair(dataset, 10, 11)

    # pc = dataset.load_pcd(0)
    # pts = np.asarray(pc.points)

    # print("Number of points:", pts.shape[0])
    # print("Min X,Y,Z:", pts.min(axis=0))
    # print("Max X,Y,Z:", pts.max(axis=0))
    ###############################################################

    
    # Load and visualize the first matched scan
    idx = 0
    scan_info = dataset[idx]
    print("PCD file:", scan_info["pcd_file"])
    print(
        "Pose (lat, lon, alt):",
        scan_info["pose"].latitude,
        scan_info["pose"].longitude,
        scan_info["pose"].altitude,
    )

    pcd0 = dataset.load_pcd(idx)
    o3d.visualization.draw_geometries([pcd0])

    # ---- ICP-based trajectory and merged map ----
    num_scans_to_use = 400  # start small; increase once it works

    poses = build_icp_trajectory(
        dataset,
        num_scans=num_scans_to_use,
        voxel_size=0.001,
        max_corr_dist=0.05,
    )

    merged_map, geoms = merge_scans_to_map(
        dataset,
        poses,
        voxel_size=0.01,
        color_each_scan=True,  # helpful to see overlap
    )
    print(f"Total .pcd files: {len(glob.glob(os.path.join('parkolo1/pcd', '*.pcd')))}")
    o3d.visualization.draw_geometries([merged_map])
    
    #print some poses for debugging
    for i in range(0, num_scans_to_use, 50):
        print(f"Pose of scan {i}:\n", poses[i])
    
    # Or:
    # o3d.visualization.draw_geometries(geoms)

