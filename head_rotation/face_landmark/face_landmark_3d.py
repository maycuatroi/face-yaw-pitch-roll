from hive_annotator.entities.annotations.face_landmark.face_landmark import FaceLandmark


class FaceLandmark3D(FaceLandmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
