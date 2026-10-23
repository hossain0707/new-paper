"""Regression tests for causal availability, clinical boundaries, and subject isolation."""
import unittest
import numpy as np
import pandas as pd
from src.evaluation.metrics import clarke_error_grid_zones
from src.llm.safety_filters import correct_tier, tier_consistency_check
from src.revision.data import _grid, build_samples
from src.revision.experiment import split_subjects,simplex_weights,weights
from src.revision.grounding import validate,expected_output,deliver
import json


class RevisionTests(unittest.TestCase):
    def test_clarke_danger_zones(self):
        for reference,prediction,zone in [(300,100,'D'),(150,20,'C'),(50,200,'E'),
                                          (200,50,'E'),(50,50,'A'),(100,110,'A')]:
            self.assertEqual(clarke_error_grid_zones([reference],[prediction])['counts'][zone],1)

    def test_tier_continuous_boundaries(self):
        for score,tier in [(39.99,1),(40,2),(59.3,2),(60,3),(79.99,3),(80,4),(94.99,4),(95,5),(100,5)]:
            self.assertEqual(correct_tier(score)['tier'],tier)
        self.assertFalse(tier_consistency_check('Tier 4',{'risk_score':101})['passed'])

    def test_grid_no_future_information(self):
        g=pd.DataFrame({'PtID':['p']*3,'ts':pd.to_datetime(['2020-01-01 00:00','2020-01-01 00:01','2020-01-01 00:16']),
                        'glucose_mgdl':[100.,999.,222.],'sensor':['s']*3})
        h=g.rename(columns={'glucose_mgdl':'hr_bpm'})
        result=_grid(g,h,'test')
        self.assertEqual(result.iloc[0].glucose_mgdl,100.)
        self.assertEqual(result.iloc[1].glucose_mgdl,999.)
        # Future perturbation cannot alter an already completed bin.
        g.loc[2,'glucose_mgdl']=333.
        self.assertEqual(_grid(g,h,'test').iloc[1].glucose_mgdl,999.)

    def test_horizon_alignment_and_missing_targets(self):
        d=pd.DataFrame({'PtID':['p']*30,'ts':pd.date_range('2020-01-01',periods=30,freq='15min'),
                        'glucose_mgdl':np.arange(30)+100.,'hr_bpm':80.,'source':'s','sensor':'s'})
        x,y,m,_=build_samples(d,min_samples=1)
        self.assertEqual(x.shape[1],9)
        np.testing.assert_allclose(y[0],[110,112,114])
        self.assertEqual(x[0,-1,0],108.)
        d.loc[10,'glucose_mgdl']=np.nan
        _,_,meta,_=build_samples(d,min_samples=1)
        self.assertNotIn(m.iloc[0].sample_id,meta.sample_id.tolist())

    def test_delayed_input_does_not_shift_targets(self):
        d=pd.DataFrame({'PtID':['p']*30,'ts':pd.date_range('2020-01-01',periods=30,freq='15min'),
                        'glucose_mgdl':np.arange(30)+100.,'hr_bpm':80.,'source':'s','sensor':'s'})
        x,y,m,_=build_samples(d,min_samples=1,input_lag_steps=1)
        self.assertEqual(x[0,-1,0],108.)
        self.assertEqual(m.iloc[0].glucose_mgdl,108.)
        np.testing.assert_allclose(y[0],[111,113,115])

    def test_subject_split_and_balancing(self):
        m=pd.DataFrame({'PtID':['a']*10+['b']*3+['c']*5+['d']*2})
        train,val,test=split_subjects(m,'a')
        self.assertFalse(set(train)&{val,test});self.assertNotEqual(val,test)
        self.assertEqual(set(train)|{val,test},set(m.PtID))
        w=weights(m.PtID.to_numpy())
        totals=pd.Series(w).groupby(m.PtID).sum()
        np.testing.assert_allclose(totals,totals.iloc[0])

    def test_simplex_can_reject_bad_member(self):
        truth=np.arange(10,dtype=float)
        preds=np.column_stack([truth,truth+20,truth-5])
        w=simplex_weights(preds/100,truth/100)
        self.assertAlmostEqual(w.sum(),1)
        self.assertTrue((w>=0).all())
        np.testing.assert_allclose(preds@w,truth,atol=1e-3)

    def test_typed_gate_has_no_vacuous_or_unit_swap_pass(self):
        case={'values':{'observations':{'glucose_mgdl':110.,'hr_bpm':70.,'input_lag_minutes':0},
                       'forecasts_mgdl':{'h30':120.,'h60':130.,'h90':140.},
                       'rate_mgdl_per_min':1.,'study_score':25.,
                       'shap_h30':[{'feature':'glucose_lag_0min','contribution_mgdl':3.}]}}
        valid=expected_output(case)
        self.assertTrue(validate(json.dumps(valid),case)['accepted'])
        self.assertFalse(validate('{}',case)['accepted'])
        valid['observations']['glucose_mgdl']=70.
        # Mutating an output must not mutate the trusted source.
        self.assertEqual(case['values']['observations']['glucose_mgdl'],110.)
        text=json.dumps(expected_output(case)).replace('110.0','70.0')
        self.assertEqual(deliver(text,case)['origin'],'template_fallback')


if __name__=='__main__':unittest.main()
