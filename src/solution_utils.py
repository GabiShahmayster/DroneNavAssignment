import dataclasses
import cv2
from typing import List, Dict, Optional

import numpy as np
import torch


@dataclasses.dataclass
class LightGlueResult:
    drone_kpts: List[cv2.KeyPoint]
    sat_kpts: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]

def get_opencv_kpt(pt, score) -> cv2.KeyPoint:
    kpt = cv2.KeyPoint()
    kpt.pt = tuple(pt)
    kpt.response = score
    return kpt

def lightglue_to_opencv_matches(input_dict: Dict, lightlue_output: Dict) -> LightGlueResult:
    matches: List[cv2.DMatch] = []
    query_kpts: List[cv2.KeyPoint] = []
    train_kpts: List[cv2.KeyPoint] = []
    matches_indices: np.ndarray = lightlue_output['matches'][0].detach().cpu().numpy()
    detected_kpts0: np.ndarray = input_dict['image0']['keypoints'][0].detach().cpu().numpy()
    detected_kpts1: np.ndarray = input_dict['image1']['keypoints'][0].detach().cpu().numpy()
    scores: np.ndarray = lightlue_output['scores'][0].detach().cpu().numpy()
    for idx, (m, score) in enumerate(zip(matches_indices, scores)):
        match: cv2.DMatch = cv2.DMatch()
        match.queryIdx = idx
        match.trainIdx = idx
        matches.append(match)
        query_kpts.append(get_opencv_kpt(detected_kpts0[m[0]], score))
        train_kpts.append(get_opencv_kpt(detected_kpts1[m[1]], score))
    return LightGlueResult(drone_kpts=query_kpts,
                           sat_kpts=train_kpts,
                           matches=matches)

def estimate_homography(matching_results: LightGlueResult, threshold: int, min_number_of_inliers: int) -> Optional[np.ndarray]:
    sat_pts: np.ndarray = np.array([np.array(k.pt) for k in matching_results.drone_kpts])
    drone_pts: np.ndarray = np.array([np.array(k.pt) for k in matching_results.sat_kpts])
    H, mask = cv2.findHomography(sat_pts, drone_pts, cv2.RANSAC, threshold)
    inliers = [int(i) for i in np.where(mask)[0]]
    if len(inliers) > min_number_of_inliers:
        return H, inliers
    else:
        return None, None