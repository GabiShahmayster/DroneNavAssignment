import dataclasses
import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from src.solution_utils import lightglue_to_opencv_matches, LightGlueResult, estimate_homography
from src.superpoint import SuperPoint
from src.lightglue import LightGlue
from utils import delta_latlon_to_meters


@dataclasses.dataclass
class DronePositionConfig:
    device: torch.device
    lightglue_match_threshold: float
    homography_inlier_threshold: int
    reprojection_inlier_threshold: float
    sat_image_altitude: float
    min_number_of_inliers: int
    min_number_of_matches: int
    save_results: bool
    enable_plot: bool
    output_path: Path

    @classmethod
    def build_default(cls):
        return DronePositionConfig(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                   lightglue_match_threshold=.1,
                                   homography_inlier_threshold=3,
                                   reprojection_inlier_threshold=8.0,
                                   sat_image_altitude=.0,
                                   min_number_of_inliers=10,
                                   min_number_of_matches=30,
                                   save_results=False,
                                   enable_plot=False,
                                   output_path=Path(__file__).parent / 'results')


def get_inliers(matching_results: LightGlueResult, inliers: List[int]) -> LightGlueResult:
    return LightGlueResult(drone_kpts=[matching_results.drone_kpts[i] for i in inliers],
                           sat_kpts=[matching_results.sat_kpts[i] for i in inliers],
                           matches=[matching_results.matches[i] for i in inliers])


class DronePosition:
    config: DronePositionConfig

    detector: SuperPoint
    matcher: LightGlue
    matching_results: Optional[LightGlueResult]
    drone_img: np.ndarray
    sat_img: np.ndarray

    H_drone_to_sat: Optional[np.ndarray]
    H_drone_to_sat_inliers: List[int]

    T_pixel_to_east_north: np.ndarray
    T_east_north_to_pixel: np.ndarray

    drone_pose_enu: np.ndarray
    waypoints: np.ndarray
    waypoints_drone: np.ndarray

    camera_mat: np.ndarray
    distCoeffs: Optional[np.ndarray]

    def __init__(self, config: DronePositionConfig = None):
        if config is None:
            config = DronePositionConfig.build_default()
        self.config = config
        if self.config.save_results:
            os.makedirs(self.config.output_path, exist_ok=True)

        # General
        self.drone_path = 'drone.png'
        self.sat_path = 'sat_40.70178897_-73.99330167.png'

        # Satellite info
        self.mpp = 0.905  # Meter per pixel of the satellite image
        self.sat_top_left = np.array([40.70178897, -73.99330167])  # (Latitude, Longitude) of the top left pixel

        # Drone's camera
        fx = fy = 920.  # 890
        cx = 645.
        cy = 323.
        self.camera_mat = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])

        # World waypoints
        self.waypoints = np.array([[40.69961509703899, -73.98978172432932],
                                   [40.699821453971865, -73.98976595014747],
                                   [40.70006040530168, -73.98974433372055],
                                   [40.70022484365462, -73.98973431293507]])

        self.init_params()

    def init_params(self):
        superpoint_config = {}
        lightglue_config = {"filter_threshold": self.config.lightglue_match_threshold}
        self.detector = SuperPoint(config=superpoint_config).to(self.config.device)
        self.matcher = LightGlue(features='superpoint', **lightglue_config).to(self.config.device)
        self.matching_results = None
        self.drone_img = None
        self.sat_img = None
        self.H_drone_to_sat = None
        self.H_drone_to_sat_inliers = None
        self.T_pixel_to_east_north = np.array([[self.mpp, .0],
                                               [.0, -self.mpp]])
        self.T_east_north_to_pixel = np.linalg.inv(self.T_pixel_to_east_north)
        self.drone_pose_enu = None
        self.distCoeffs = None

    def image_to_tensor(self, im: np.ndarray) -> torch.Tensor:
        # return BxCxHxW tensor
        assert len(im.shape) == 3, "image_to_tensor must be provided with BGR images"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(im / 255.).float()[None, None].to(self.config.device)

    def find_matches(self):
        """
        Find Matches between drone's frame and satellite image
        """

        """
        - using superpoint (pre-trained weights) for feature detection
        - using lightglue (pre-trained weights for superpoint) for feature matching, retain matches with a minimal score (config parameters)
        - convert lightglue torch tensors (optionally gpu) to opencv standard cv2.Keypoints/cv2.DMatch objects (cpu)
        - draw and save matches  
        """

        with torch.no_grad():
            self.drone_img = cv2.imread(self.drone_path)
            self.sat_img = cv2.imread(self.sat_path)
            drone_img: torch.Tensor = self.image_to_tensor(self.drone_img)
            sat_img: torch.Tensor = self.image_to_tensor(self.sat_img)
            drone_det = self.detector({"image": drone_img})
            sat_det = self.detector({"image": sat_img})
            lightglue_input_dict = {"image0": self.get_lightlue_input(drone_det, drone_img),
                                    "image1": self.get_lightlue_input(sat_det, sat_img)}
            res = self.matcher(lightglue_input_dict)

        if res['matches'][0].shape[0] < self.config.min_number_of_inliers:
            self.matching_results = None
            return
        else:
            self.matching_results = lightglue_to_opencv_matches(lightglue_input_dict, res)

        self.draw_matches()

    def find_footprint(self):
        """
        Find Footprint of drone's frame in the satellite image
        """

        """
        - using lighglue matches to estimate homography (opencv)
        - applying drone->sat homography to transform drone image corners to sat image, to obtain footprint  
        - plot and save
        """

        self.H_drone_to_sat, self.H_drone_to_sat_inliers = estimate_homography(matching_results=self.matching_results,
                                                                               threshold=self.config.homography_inlier_threshold,
                                                                               min_number_of_inliers=self.config.min_number_of_inliers)
        if self.H_drone_to_sat is None:
            return
        self.draw_footprint()

    def estimate_drone_position(self):
        """
        Estimates drone's position and orientation using PNP algorithm
        """


        """
        - define local navigation frame "enu" (tangential frame, xy axis are horizontal, z-axis is vertical). 
        - origin of enu frame is at the geographic position of the top-left pixel
        - axes of enu frame are: +x_enu = East, +y_enu = North, +z_enu = Up
        - enu frame will be the cartesian "world" reference frame for all world-cam transformations. 
        - camera pose is estimated w.r.t enu frame by solving camera pose using PnP  
        - PnP is provided with drone image points (keypoints detected by superpoint) and corresponding world points (3D positions, enu frame) 
        - world points are obtained by:
        -       drone image -> sat image -> world point
        - where:
        -       drone image -> sat image := 2d-2d transformation (homography) from drone image to sat image
        -       sat image -> world point := sat image pixels converted to meters (using self.mpp) and correct image y-axis sign (aligned with negative North) + zero altitide assumption (config param)  
        """

        # we will calculate the world coordinates of drone keypoints, using the drone<->satellite homography
        # world coordinates of satellite image keypoints are given by translating pixel offset to meters and 0 altitude assumption
        matching_inliers: LightGlueResult = get_inliers(self.matching_results, self.H_drone_to_sat_inliers)
        drone_kpts = np.array([np.array(kpt.pt) for kpt in matching_inliers.drone_kpts])
        sat_kpts = np.array([np.array(kpt.pt) for kpt in matching_inliers.sat_kpts])
        world_coords = np.vstack((self.T_pixel_to_east_north @ sat_kpts.T, self.config.sat_image_altitude * np.ones((1, len(matching_inliers.matches)))))
        world_coords = world_coords.T
        pixel_coords = np.expand_dims(drone_kpts, axis=1)
        res, rvec, tvec, mask = cv2.solvePnPRansac(objectPoints=world_coords,
                                                   imagePoints=pixel_coords,
                                                   cameraMatrix=self.camera_mat,
                                                   distCoeffs=self.distCoeffs,
                                                   reprojectionError=self.config.reprojection_inlier_threshold)
        if not res:
            self.cam_pose = None
            return

        T_world_to_cam = np.eye(4)
        T_world_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_world_to_cam[:3, 3] = tvec.squeeze()
        self.drone_pose_enu = np.linalg.inv(T_world_to_cam)
        self.draw_drone_on_sat_image()

    def project_waypoints(self):
        """
        project the waypoints (lat, lon) to drone's body frame using extrinsic matrix (R|T) from `estimate_drone_position`
        """


        """
        - transform waypoints to world frame (enu)
        - transform from world frame to drone-camera frame, using previouse estimated drone camera pose     
        """

        T_enu_to_drone = np.linalg.inv(self.drone_pose_enu)
        self.waypoints_drone = np.empty((3, self.waypoints.shape[0]))
        for idx, wpt in enumerate(self.waypoints):
            wpt_east_north = delta_latlon_to_meters(lat1=self.sat_top_left[0],
                                             lon1=self.sat_top_left[1],
                                             lat2=wpt[0],
                                             lon2=wpt[1])
            wpt_enu = np.array([wpt_east_north[0], wpt_east_north[1], self.config.sat_image_altitude, 1.0])
            wpt_drone = T_enu_to_drone @ wpt_enu
            self.waypoints_drone[:, idx] = wpt_drone[:3]

        if self.config.save_results:
            output_filename: str = str(self.config.output_path / 'waypoints_drone.txt')
            with open(output_filename, 'w') as f:
                np.savetxt(f, self.waypoints_drone.T, delimiter=',', fmt='%.1f')

    def draw_waypoints_extrinsic(self):
        """
        Draws waypoints (lat, lon) on drone's image using extrinsic matrix (R|T) from `estimate_drone_position`
        """

        """
        - project previously computed drone-camera frame waypoint positions, using opencv's projectPoints     
        """

        drone_image_with_waypoints_from_6dof: np.ndarray = np.array(self.drone_img)
        waypoints_drone = np.expand_dims(self.waypoints_drone.T, axis=0)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        wpts_image, _ = cv2.projectPoints(waypoints_drone, rvec, tvec, self.camera_mat, self.distCoeffs)
        for wpt in wpts_image.squeeze():
            p = (int(wpt[0]), int(wpt[1]))
            cv2.drawMarker(drone_image_with_waypoints_from_6dof, p, (255, 0, 127), cv2.MARKER_SQUARE, 10, 3)

        if self.config.save_results:
            output_filename: str = str(self.config.output_path / 'proj_waypoints_using_6dof.png')
            cv2.imwrite(output_filename, drone_image_with_waypoints_from_6dof)

        if self.config.enable_plot:
            cv2.imshow("projected waypoints using 6dof", drone_image_with_waypoints_from_6dof)
        cv2.waitKey(100)

    def draw_waypoints_homography(self):
        """
        Draws waypoints (lat, lon) on drone's image using homography
        """


        """
        - instead of projecting the waypoint world positions, we will apply the previously estimated homography directly to the waypoints pixel coords. (in the sat image)             
        """

        drone_image_with_waypoints_from_homograpy: np.ndarray = np.array(self.drone_img)
        waypoints_east_north = np.empty((2, self.waypoints.shape[0]))
        for idx, wpt in enumerate(self.waypoints):
            wpt_east_north = delta_latlon_to_meters(lat1=self.sat_top_left[0],
                                                    lon1=self.sat_top_left[1],
                                                    lat2=wpt[0],
                                                    lon2=wpt[1])
            waypoints_east_north[:, idx] = wpt_east_north
        waypoints_sat = self.T_east_north_to_pixel @ waypoints_east_north
        wpts_image = np.linalg.inv(self.H_drone_to_sat) @ np.vstack((waypoints_sat, np.ones((1, self.waypoints.shape[0]))))
        wpts_image = wpts_image[:2, :] / wpts_image[2, :]
        for wpt in wpts_image.T:
            p = (int(wpt[0]), int(wpt[1]))
            cv2.drawMarker(drone_image_with_waypoints_from_homograpy, p, (0, 0, 255), cv2.MARKER_SQUARE, 10, 3)

        if self.config.save_results:
            output_filename: str = str(self.config.output_path / 'proj_waypoints_using_homography.png')
            cv2.imwrite(output_filename, drone_image_with_waypoints_from_homograpy)

        if self.config.enable_plot:
            cv2.imshow("projected waypoints using homography", drone_image_with_waypoints_from_homograpy)
        cv2.waitKey(100)

    def get_lightlue_input(self, detector_output: Dict, image: torch.Tensor):
        """
        https://kornia.readthedocs.io/en/latest/feature.html
        keypoints: [B x M x 2]
        descriptors: [B x M x D]
        image: [B x C x H x W] or image_size: [B x 2]
        """
        im_size: torch.Tensor = torch.unsqueeze(torch.Tensor([image.shape[2], image.shape[3]]), dim=0)
        return {"keypoints": torch.unsqueeze(detector_output['keypoints'][0], dim=0),
                "descriptors": torch.unsqueeze(detector_output['descriptors'][0].T, dim=0),
                "image_size": im_size}

    def draw_matches(self):
        matches_img: np.ndarray = cv2.drawMatches(self.drone_img, self.matching_results.drone_kpts, self.sat_img, self.matching_results.sat_kpts, self.matching_results.matches, None)
        if self.config.enable_plot:
            cv2.imshow("matches", matches_img)
        if self.config.save_results:
            output_filename: str = str(self.config.output_path / 'matches.png')
            cv2.imwrite(output_filename, matches_img)
        cv2.waitKey(100)

    def draw_footprint(self):
        # transform image corners
        image_height, image_width, _ = self.drone_img.shape
        drone_corners: np.ndarray = np.empty((3, 4))
        idx = 0
        corners = [(.0, .0), (image_width, .0), (image_width, image_height), (.0, image_height)]
        for uv in corners:
            drone_corners[:, idx] = np.array([uv[0], uv[1], 1.0])
            idx += 1
        drone_corners_warp = self.H_drone_to_sat @ drone_corners
        drone_corners_warp = drone_corners_warp[:2, :] / drone_corners_warp[2, :]
        sat_img_with_footprint: np.ndarray = np.array(self.sat_img)
        CORNER_COLOR = (255, 0, 0)
        LINE_COLOR = (0, 255, 0)
        n_corners: int = len(corners)
        for corner_idx in range(n_corners):
            pt = tuple(drone_corners_warp[:, corner_idx].astype(int))
            next_pt = tuple(drone_corners_warp[:, (corner_idx + 1) % n_corners].astype(int))
            cv2.drawMarker(sat_img_with_footprint, pt, CORNER_COLOR, markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=3)
            cv2.line(sat_img_with_footprint, pt, next_pt, LINE_COLOR, thickness=3)

        if self.config.enable_plot:
            cv2.imshow("drone footprint", sat_img_with_footprint)
        
        if self.config.save_results:
            output_filename: str = str(self.config.output_path / 'drone_footprint.png')
            cv2.imwrite(output_filename, sat_img_with_footprint)
            
        cv2.waitKey(100)

    def draw_drone_on_sat_image(self):
        drone_on_sat_image = np.array(self.sat_img)
        drone_translation_east_north = self.drone_pose_enu[:2, 3]
        drone_position_pixels = self.T_east_north_to_pixel @ drone_translation_east_north
        dront_pixel = (int(drone_position_pixels[0]), int(drone_position_pixels[1]))
        DRONE_COLOR = (0, 0, 255)
        cv2.drawMarker(drone_on_sat_image, dront_pixel, DRONE_COLOR, cv2.MARKER_TRIANGLE_UP, markerSize=3, thickness=10)
        if self.config.enable_plot:
            cv2.imshow("drone on sat image", drone_on_sat_image)

        if self.config.save_results:
            output_filename: str = str(self.config.output_path / 'drone_position.png')
            cv2.imwrite(output_filename, drone_on_sat_image)
            
        cv2.waitKey(100)


if __name__ == '__main__':
    SAVE_RESULTS: bool = True
    ENABLE_PLOT: bool = False
    config: DronePositionConfig = DronePositionConfig(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                                      lightglue_match_threshold=.1,
                                                      homography_inlier_threshold=3,
                                                      reprojection_inlier_threshold=8,
                                                      sat_image_altitude=.0,
                                                      min_number_of_inliers=10,
                                                      min_number_of_matches=30,
                                                      save_results=SAVE_RESULTS,
                                                      enable_plot=ENABLE_PLOT,
                                                      output_path=Path('/tmp/assign/results'))
    drone_pos = DronePosition(config=config)

    """
    general: this exercise demonstrates two difference approaches to projecting geographic waypoints to the drone image:
    1. 2d<->2d registration between the geographic global reference-frame (squashed to a 2d orthophoto) and the drone image, using homography
    2. 3d->2d approach, which allows working with 3d geographic waypoints, at the cost of having to estimate camera extrinsic (6dof pose) and intrinsic (pinhole model) parameters, for the projection         
    """

    # Estimate drone's position
    drone_pos.find_matches()
    assert drone_pos.matching_results is not None, "could not find enough matches between sat and drone images, aborting!"
    drone_pos.find_footprint()
    assert drone_pos.H_drone_to_sat is not None, "cound not find homography between sat and drone images, aborting!"
    drone_pos.estimate_drone_position()

    # Convert the waypoints
    drone_pos.project_waypoints()
    drone_pos.draw_waypoints_extrinsic()
    drone_pos.draw_waypoints_homography()
