from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.multi_pose import MultiPoseDataset
from .sample.landmark import LandmarkDataset


from src.lib.datasets.dataset.coco_hp import COCOHP
from src.lib.datasets.dataset.coco_body import COCOBODY

dataset_factory = {
  'coco_hp': COCOHP,
  'coco_body': COCOBODY
}

_sample_factory = {
  'multi_pose': MultiPoseDataset,
  'landmark': LandmarkDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
