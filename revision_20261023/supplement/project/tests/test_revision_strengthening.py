"""Guard against window-count dominance in multi-patient validation."""
import numpy as np
from src.revision.strengthening import patient_mae,weighted_simplex


def test_patient_validation_can_reverse_window_weighted_ranking():
    subjects=np.array(['many']*100+['few'])
    truth=np.zeros((101,3))
    first=np.zeros_like(truth);first[-1]=18
    second=np.ones_like(truth);second[-1]=0
    assert np.abs(first-truth).mean()<np.abs(second-truth).mean()
    assert patient_mae(first,truth,subjects)==9
    assert patient_mae(second,truth,subjects)==.5


def test_patient_balanced_stack_matches_equal_patient_closed_form():
    subjects=np.array(['many']*100+['few'])
    prediction=np.tile([0.,10.],(101,1));truth=np.zeros(101);truth[-1]=10
    # Each patient supplies equal mass: mean of 0 and 10, irrespective of window count.
    w=weighted_simplex(prediction,truth,subjects)
    np.testing.assert_allclose(w,[.5,.5],atol=1e-6)
