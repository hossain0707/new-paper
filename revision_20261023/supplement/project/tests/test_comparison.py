"""Verify the new controls do not leak answers through schema or alter source data."""
import copy,json,unittest
from src.revision.comparison import contract_schema,messages_for

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.case={'values':{'observations':{'glucose_mgdl':110.,'hr_bpm':70.,'input_lag_minutes':0},
          'forecasts_mgdl':{'h30':120.,'h60':130.,'h90':140.},'rate_mgdl_per_min':1.,
          'study_score':25.,'shap_h30':[{'feature':'glucose_lag_0min','contribution_mgdl':3.}]}}

    def test_schema_does_not_reveal_numeric_answers(self):
        schema=contract_schema(self.case)
        self.assertEqual(schema['properties']['observations']['properties']['glucose_mgdl'],{'type':'number'})
        self.assertNotIn('110',json.dumps(schema))
        self.assertNotIn('const',json.dumps(schema))
        self.assertFalse(schema['additionalProperties'])

    def test_direction_factor_only_changes_supplied_condition(self):
        original=copy.deepcopy(self.case)
        a=messages_for(self.case,[],supplied_direction=False)
        b=messages_for(self.case,[],supplied_direction=True)
        self.assertNotIn('precomputed_direction',json.loads(a[-1]['content']))
        self.assertEqual(json.loads(b[-1]['content'])['precomputed_direction'],'rising')
        self.assertEqual(original,self.case)

    def test_examples_use_same_factor_as_test_case(self):
        messages=messages_for(self.case,[self.case],few_shot=True,supplied_direction=True)
        self.assertEqual(len(messages),4)
        self.assertEqual(json.loads(messages[1]['content'])['precomputed_direction'],'rising')
        self.assertNotIn('precomputed_direction',json.loads(messages[2]['content']))

if __name__=='__main__': unittest.main()
