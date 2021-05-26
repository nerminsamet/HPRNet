from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .multi_pose import MultiPoseTrainer
from .landmark import LandmarkTrainer

train_factory = {
  'multi_pose': MultiPoseTrainer,
  'landmark': LandmarkTrainer,
}
