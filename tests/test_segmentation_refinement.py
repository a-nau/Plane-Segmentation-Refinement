import logging
import unittest

from run_refinement import main

logger = logging.getLogger(__name__)


class TestSegmentationRefinement(unittest.TestCase):
    def test_segmentation_refinement(self):
        # Load data
        dir = "./input/0_dataset_027"
        config = "./config.yaml"

        result_dict = main(dir, config)

        # Check if we see an improvement on all masks
        i = 0
        while True:
            if f"ious_planercnn_masks_{i}" in result_dict.keys():
                self.assertTrue(result_dict[f"ious_planercnn_masks_{i}"] < result_dict[f"ious_refined_masks_{i}"])
                i += 1
            else:
                break
