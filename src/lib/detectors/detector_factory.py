from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .multi_pose import MultiPoseDetector
from .landmark import LandmarkDetector

detector_factory = {
  'multi_pose': MultiPoseDetector,
  'landmark': LandmarkDetector
}
