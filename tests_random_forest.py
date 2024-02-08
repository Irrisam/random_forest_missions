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
        expected_output = "mission score: 4.01 and predicted count: 0"
        result = self.stat_model_instance.evaluate_mission()
        self.assertEqual(result, expected_output)

    #def test_evaluate_mission_input(self):
    #    stat_model_instance = StatModels(mission_values={}, model_stats_check=False, confusion_matrix_check=False)
    #    expected_error = "KeyError: "['announcement_id'] 
    #    with self.assertRaises(SystemExit) as context:
    #        stat_model_instance.evaluate_mission()
    #    self.assertIn(expected_error, str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
