"""Checks for the person-overlap rules. Run from the project root:
    ~/.conda/envs/animal-det/bin/python -m unittest web/test_rules.py
Needs no model and no Flask."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import Det, iou, person_overlap  # noqa: E402

PERSON = [Det("person", "person", 0.9, (100.0, 100.0, 200.0, 300.0))]


class PersonOverlapRules(unittest.TestCase):
    def test_centre_inside_person_is_dropped(self):
        self.assertTrue(person_overlap((150, 150, 190, 250), PERSON, 0.3, 0.7))

    def test_iou_rule_alone(self):
        box = (111, 84, 307, 257)  # IoU 0.35 with the person, centre outside
        self.assertTrue(person_overlap(box, PERSON, 0.3, 0.7))
        self.assertFalse(person_overlap(box, PERSON, 1.1, 1.1))  # rule switched off

    def test_mostly_contained_is_dropped(self):
        self.assertTrue(person_overlap((120, 120, 190, 290), PERSON, 0.3, 0.7))

    def test_cow_beside_person_is_kept(self):
        self.assertFalse(person_overlap((250, 120, 400, 300), PERSON, 0.3, 0.7))

    def test_small_edge_touch_is_kept(self):
        self.assertFalse(person_overlap((195, 280, 300, 400), PERSON, 0.3, 0.7))

    def test_iou_helper(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


if __name__ == "__main__":
    unittest.main()
