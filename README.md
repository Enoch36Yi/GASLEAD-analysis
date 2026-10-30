# GASLEAD analysis

- Analysis code for compressed-air leak testing with IEPE, MEMS and AE sensors.
- Includes the August 6 rule, held-out cycle testing, paired randomization, time/order controls and channel stability checks.
- Install: `python -m pip install -r EI2_automation/requirements.txt` (Python 3.12).
- Verify: `python EI2_automation/ei2.py verify-package`; test: `python -m unittest discover -s EI2_automation -p test_ei2.py`.
- Prepare local inputs: `python EI2_automation/ei2.py prepare --out local_inputs`; enter the actual session date and file mapping before running.
- Raw data, measurement fixtures, photos, session records and analysis outputs stay local. The numerical code is unchanged; export details are in `EI2_automation/EXPORT_PROVENANCE.json`.
