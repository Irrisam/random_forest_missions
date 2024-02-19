import unittest
from stable_random_forest import StatModels
import pandas as pd

class TestEvaluateMission(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("simple_mission_test.csv")
        self.mission_row = data.iloc[0].to_dict()
        stat_model_instance = StatModels(mission_values=self.mission_row, model_stats_check=False, confusion_matrix_check=False)
        self.stat_model_instance = stat_model_instance

    def test_evaluate_mission_output(self):
        #test normal output
        expected_output = "mission score: 3.74 and predicted count: 0"
        result = self.stat_model_instance.evaluate_mission()
        self.assertEqual(result, expected_output)
        
    def test_evaluate_mission_input(self):
        column_to_drop = "salary"
        if column_to_drop in self.stat_model_instance.mission_values:
            self.stat_model_instance.mission_values.pop(column_to_drop)
        expected_error = 'The feature names should match those that were passed during fit.\nFeature names seen at fit time, yet now missing:\n- salary\n'
        with self.assertRaises(ValueError) as context:
            self.stat_model_instance.evaluate_mission() 
        self.assertIn(expected_error, str(context.exception))

    def test_evaluate_mission_missing_values(self):
        self.stat_model_instance.mission_values = None
        with self.assertRaises(ValueError) as context:
            self.stat_model_instance.evaluate_mission()
        self.assertIn("KeyError", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
