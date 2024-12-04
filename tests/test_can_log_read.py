import os
from asammdf import MDF

from can_utils import mdf_can_log_utils
import can_utils.mdf_can_log_utils as utls
from proj_config.config import RAW_DATA_DIR, DBC_DIR, VEH_CONF_DIR


def test_mdf_get_trace():
    log_file1 = os.path.join(RAW_DATA_DIR,"vehicle_A_can_log_1.mf4");

    mdf1 = MDF(log_file1)
    mdf1_trace=utls.mdf_get_trace(mdf1)
    assert mdf1_trace.shape==(515834, 15), "output dataframe shape is not correct"

