import math
import sys

import cv2
import numpy as np

from hive_annotator.entities.abstract_entity import AbstractEntity

sys.path.append('')

likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                   'LIKELY', 'VERY_LIKELY')
EYE_OPEN_THRESHOLD = 5
MOUTH_OPEN_THRESHOLD = 10


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class FaceLandmark(AbstractEntity):
    def __init__(self, face=None, **kwargs) -> None:
        """
        face from google API response
        """
        super().__init__(**kwargs)
        if face:
            self.face = face
            self.landmarks = face['landmark_3d_68']
            self.bbox = face.bbox

    def draw(self, mat: np.array) -> None:
        """
        Draw landmark on mat
        """
        for i, point in enumerate(self.landmarks):
            cv2.circle(mat, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
            cv2.putText(mat, str(i), (int(point[0]), int(
                point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        xmin, ymin, xmax, ymax = self.bbox.astype(int)

        cv2.rectangle(mat, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        rois = []
        rois.extend(self.eye_bounding_box())
        rois.append(self.mouth_bounding_box())
        for roi in rois:
            xmin, ymin, xmax, ymax = roi.astype(int)
            cv2.rectangle(mat, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        return mat

    def compute_eye_heights(self):
        pairs = np.array([[37, 41], [38, 40], [43, 47], [44, 46]])
        top_points = self.landmarks[pairs[:, 0]]
        bottom_points = self.landmarks[pairs[:, 1]]
        distance = np.linalg.norm(top_points - bottom_points, axis=1)
        return distance

    def is_eye_open(self):
        """
        Return true if eyes is open
        """
        heights = self.compute_eye_heights()
        print(heights)
        if np.mean(heights) < EYE_OPEN_THRESHOLD:
            return False
        return True

    def is_mouth_open(self):
        """
        Return true if mouth is open
        """
        lip_points = np.array([self.landmarks['UPPER_LIP'].to_array(),
                               self.landmarks['LOWER_LIP'].to_array()])
        mouth_height = np.linalg.norm(
            lip_points[0] - lip_points[1], axis=1)
        if mouth_height < MOUTH_OPEN_THRESHOLD:
            return False
        return True

    def eye_bounding_box(self, padding=10) -> tuple[np.array, np.array]:
        """
        Return bounding box of eyes
        :return: bounding box of eyes as numpy array ( xmin, ymin, xmax, ymax )
        """
        left_eye_points = self.landmarks[36:42]
        right_eye_points = self.landmarks[42:48]

        left_eye_bbox = np.array([np.min(left_eye_points[:, 0]) - padding,
                                  np.min(left_eye_points[:, 1]) - padding,
                                  np.max(left_eye_points[:, 0]) + padding,
                                  np.max(left_eye_points[:, 1]) + padding])

        right_eye_bbox = np.array([np.min(right_eye_points[:, 0]) - padding,
                                   np.min(right_eye_points[:, 1]) - padding,
                                   np.max(right_eye_points[:, 0]) + padding,
                                   np.max(right_eye_points[:, 1]) + padding])

        return left_eye_bbox, right_eye_bbox

    def mouth_bounding_box(self, padding=10):
        """
        Return bounding box of mouth
        :return: bounding box of mouth as numpy array ( xmin, ymin, xmax, ymax )
        """
        lip_points = self.landmarks[48:65]

        lip_bbox = np.array([np.min(lip_points[:, 0]) - padding,
                             np.min(lip_points[:, 1]) - padding,
                             np.max(lip_points[:, 0]) + padding,
                             np.max(lip_points[:, 1]) + padding])

        return lip_bbox

    def get_face_bounding_box(self):
        """
        Return bounding box of face
        """
        face_bbox = np.array([np.min(self.landmarks[:, 0]),
                              np.min(self.landmarks[:, 1]),
                              np.max(self.landmarks[:, 0]),
                              np.max(self.landmarks[:, 1])])
        return face_bbox

    def get_face_angle_2d(self):
        # get face yaw pitch roll
        xmin, ymin, xmax, ymax = self.get_face_bounding_box()
        size = (xmax - xmin, ymax - ymin)

        points_2D = np.array([
            self.landmarks[33][:2],  # Nose tip
            self.landmarks[8][:2],  # Chin
            self.landmarks[36][:2],  # Left eye corner
            self.landmarks[45][:2],  # Right eye corner
            self.landmarks[48][:2],  # Left mouth
            self.landmarks[54][:2]  # Right mouth
        ], dtype="double")

        points_3D = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth
            (150.0, -150.0, -125.0)  # Right mouth
        ])

        # Camera internals
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        center = (size[1] / 2, size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        success, rotation_vector, translation_vector = cv2.solvePnP(
            points_3D, points_2D, camera_matrix, dist_coeffs, flags=0)

        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        return yaw, pitch, roll

    def get_face_angle_3d(self):
        mean = np.mean(self.landmarks, axis=0)
        scaled = (self.landmarks /
                  np.linalg.norm(self.landmarks[42] - self.landmarks[39])) * 0.06
        # <- This is the scaled mean
        centered = scaled - np.mean(scaled, axis=0)
        rotationMatrix = np.empty((3, 3))
        rotationMatrix[0, :] = (centered[16] - centered[0]) / \
                               np.linalg.norm(centered[16] - centered[0])
        rotationMatrix[1, :] = (centered[8] - centered[27]) / \
                               np.linalg.norm(centered[8] - centered[27])
        rotationMatrix[2, :] = np.cross(
            rotationMatrix[0, :], rotationMatrix[1, :])
        invRot = np.linalg.inv(rotationMatrix)
        eulerAngles = rotationMatrixToEulerAngles(rotationMatrix)
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        return yaw, pitch, roll

    @property
    def area(self):
        """
        Return area of face
        """
        return float(self.get_face_bounding_box().prod())

    def get_mouth_height(self):
        """
        Return mouth height
        """
        mouth_bbox = self.mouth_bounding_box()
        return mouth_bbox[3] - mouth_bbox[1]

    def to_firebase(self):
        data = self.to_dict()
        data['bbox'] = data['bbox'].tolist()
        data['landmarks'] = data['landmarks'].flatten().tolist()
        return data

    @property
    def contour(self):
        """
        Face contour of face image
        """
        return self.landmarks
