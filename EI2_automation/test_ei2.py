"""Breaks caught: reversed directions, test leakage, wrong assignment space, bad mappings, silent data loss."""
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import ei2
except ModuleNotFoundError:
    ei2 = None


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ei2, 'The EI2 runner has not been implemented')

    def test_strict_threshold_and_forced_higher_on_reversed_data(self):
        rule = ei2.fit_rule([3,4,1,2], [0,0,1,1], direction='higher')
        self.assertEqual(rule['direction'], 'higher')
        self.assertEqual(rule['balanced_accuracy'], 0.5)
        self.assertEqual(ei2.predict([0.1,0.2,0.3], {'threshold':0.2,'direction':'higher'}).tolist(), [0,0,1])
        self.assertEqual(ei2.fit_rule([3,4,1,2], [0,0,1,1])['direction'], 'lower')

    def test_threshold_never_splits_ties(self):
        r = ei2.fit_rule([1,1,2,2], [0,1,0,1], 'higher')
        self.assertEqual(r['balanced_accuracy'], 0.5)

    def test_exact_randomization_uses_252_assignments_and_no_plus_one(self):
        positions = [1,-1,-1,-1,-1,-1,1,1,1,1]
        result, reference = ei2.randomization_test(np.array(positions,dtype=float), positions, 'A')
        self.assertEqual(len(reference), 252)
        self.assertAlmostEqual(result['p_one_sided_higher'], 1/252)
        self.assertEqual(result['observed_mean_difference'], 1.0)
        null, _ = ei2.randomization_test(np.zeros(10), positions, 'A')
        self.assertEqual(null['p_one_sided_higher'], 1.0)

    def test_stratified_randomization_uses_400_not_924(self):
        z=[-1,1,-1,-1,1,1,-1,1,-1,1,-1,1]
        result, ref=ei2.randomization_test(np.array(z,dtype=float),z,'B')
        self.assertEqual(len(ref),400)
        self.assertAlmostEqual(result['p_one_sided_higher'],1/400)

    def test_loco_holds_both_members_and_excludes_held_values(self):
        # Last pair has no influence on its own threshold: train midpoint is 1.5.
        p=ei2.loco([1,2,1,2,-100,100], [0,1,0,1,0,1], [1,1,2,2,3,3], 'higher')
        held=[x for x in p if x['cycle']==3]
        self.assertEqual([x['threshold'] for x in held],[1.5,1.5])
        self.assertEqual([x['n_training_files'] for x in held],[4,4])
        self.assertEqual(ei2.score([0,1,0,1,0,1],[x['prediction'] for x in p])['ba'],1.0)

    def test_nonpositive_log_values_not_dropped_or_pseudocounted(self):
        result,ref=ei2.randomization_from_pairs([0,2],[1,1],[1,-1],'A')
        self.assertFalse(result['available'])
        self.assertEqual(ref,[])

    def test_baseline_ac_iqr_uses_same_file_values(self):
        r=ei2.baseline([1,2,3,4])
        self.assertEqual(r['max_min'],4)
        self.assertEqual(r['niqr'],0.6)

    def test_manifest_rejects_wrong_state_and_duplicate_selected_paths(self):
        rows=ei2.schedule_rows('A')
        for i,r in enumerate(rows):r['relative_path']=f'f{i}.h5'
        ei2.validate_manifest(rows,'A')
        bad=[dict(r) for r in rows];bad[0]['state']='SEALED'
        with self.assertRaises(ValueError):ei2.validate_manifest(bad,'A')
        bad=[dict(r) for r in rows];bad[1]['relative_path']=bad[0]['relative_path']
        with self.assertRaises(ValueError):ei2.validate_manifest(bad,'A')

    def test_outside_raw_root_rejected_and_exclusions_need_reason(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):ei2.raw_path(Path(d),'../escape.h5')
        rows=ei2.schedule_rows('A')
        for i,r in enumerate(rows):r['relative_path']=f'f{i}.h5'
        rows.append({'use':'no','relative_path':'warmup.h5','exclusion_reason':''})
        with self.assertRaises(ValueError):ei2.validate_manifest(rows,'A')

    def test_output_cannot_be_created_in_frozen_authority_folder(self):
        with tempfile.TemporaryDirectory() as d:
            forbidden=Path(d)/'tables'/'test_output'
            with patch.object(ei2,'ROOT',Path(d)):
                with self.assertRaises(ValueError):ei2.new_output(forbidden)
            self.assertFalse(forbidden.exists())

    def test_folder_labels_must_agree_with_log_when_present(self):
        rows=ei2.schedule_rows('A')
        for i,r in enumerate(rows):r['relative_path']=f'f{i}.h5'
        rows[0]['relative_path']='no_leak/open_slot.h5'
        with self.assertRaises(ValueError):ei2.validate_manifest(rows,'A')

    def test_balanced_design_position_control_can_score_zero(self):
        rows=ei2.schedule_rows('A')
        y=[int(r['state']=='OPEN') for r in rows]
        pr=ei2.loco([int(r['position']) for r in rows],y,[int(r['cycle']) for r in rows])
        # Training imbalance reverses the prediction for each held pair.
        self.assertEqual(ei2.score(y,[r['prediction'] for r in pr])['ba'],0.0)

    def test_demo_writes_complete_tables_without_claiming_real_data(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'demo'
            ei2.demo(out)
            s=json.loads((out/'summary.json').read_text(encoding='utf8'))
            self.assertTrue(s['synthetic'])
            self.assertEqual(s['channels']['piezo']['locked_transfer']['ba'],1.0)
            self.assertEqual(s['channels']['piezo']['paired_randomization']['n_assignments'],252)
            self.assertAlmostEqual(s['channels']['piezo']['paired_randomization']['p_one_sided_higher'],1/252)
            self.assertEqual(len(ei2.read_csv(out/'paired_features.csv')),30)
            self.assertTrue((out/'paired_log10_differences.png').is_file())
            self.assertIn('SYNTHETIC',(out/'ANALYSIS_REVIEW.md').read_text(encoding='utf8'))

    def test_threshold_port_reproduces_frozen_code_on_synthetic_cases(self):
        self.assertTrue((ei2.HERE/'frozen_threshold.py').is_file(),'Frozen threshold oracle not yet snapshotted')
        import frozen_threshold
        rng=np.random.default_rng(20261014)
        for _ in range(80):
            x=rng.integers(0,6,20).astype(float);y=np.tile([0,1],10);rng.shuffle(y)
            self.assertEqual(ei2.fit_rule(x,y),frozen_threshold.fit_threshold(x,y))

    def test_inventory_keeps_bad_files_visible_and_raw_unchanged(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'raw';root.mkdir();p=root/'bad.h5'
            with h5py.File(p,'w') as f:
                f.create_group('metadata').attrs['sample_rate_hz']=2_000_000
                f.create_group('raw').create_dataset('piezo',data=np.array([1.,2.]))
            before=ei2.sha256(p);out=Path(d)/'inventory'
            ei2.inventory(root,out)
            q=json.loads((out/'technical_inventory.json').read_text(encoding='utf8'))
            self.assertEqual(len(q),1);self.assertTrue(q[0]['errors'])
            self.assertEqual(ei2.sha256(p),before)

    def test_hdf5_qc_checks_sample_counts_and_warns_rails_without_excluding(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5'
            with h5py.File(p,'w') as f:
                f.create_group('metadata').attrs['sample_rate_hz']=2_000_000
                g=f.create_group('raw')
                for c in ei2.CHANNELS:g.create_dataset(c,data=np.array([0.,5.,0.,0.]))
            q=ei2.inspect_h5(p,minimum_seconds=0)
            self.assertEqual(q['errors'],[])
            self.assertTrue(q['warnings'])
            with h5py.File(p,'a') as f:del f['raw/ae']
            self.assertTrue(ei2.inspect_h5(p,minimum_seconds=0)['errors'])

    def test_streamed_features_match_frozen_window_function(self):
        import h5py
        fs=2_000_000
        t=np.arange(fs)/fs
        x=(0.02*np.sin(2*np.pi*3000*t)+0.01*np.sin(2*np.pi*9000*t)).astype(float)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5'
            with h5py.File(p,'w') as f:
                f.create_group('metadata').attrs['sample_rate_hz']=fs
                g=f.create_group('raw')
                for c in ei2.CHANNELS:g.create_dataset(c,data=x)
            rows,window_rows=ei2.extract_h5(p)
            _,_,expected=ei2.frozen.periodogram_features(x,fs)
            for r in rows:
                self.assertAlmostEqual(r['feature_value'],expected[ei2.FEATURES[r['channel']].removesuffix('__median')],places=12)
                self.assertAlmostEqual(r['median_window_ac_rms'],np.sqrt(np.mean((x-x.mean())**2)),places=12)
                self.assertEqual(r['n_full_windows'],1)
            self.assertEqual(len(window_rows),3)

    def test_disconnected_ai3_is_not_analyzed_or_used_to_reject_files(self):
        import h5py
        fs=2_000_000;t=np.arange(fs)/fs;x=np.sin(2*np.pi*3000*t)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'open_input.h5'
            with h5py.File(p,'w') as f:
                f.create_group('metadata').attrs['sample_rate_hz']=fs;g=f.create_group('raw')
                for c in ['piezo','mems','ae']:g.create_dataset(c,data=x)
                g.create_dataset('contact_mic',data=np.array([float('nan'),5.0]))
            q=ei2.inspect_h5(p,minimum_seconds=0)
            self.assertEqual(q['errors'],[])
            self.assertNotIn('contact_mic',[r['channel'] for r in q['channels']])
            rows,windows=ei2.extract_h5(p)
            self.assertEqual({r['channel'] for r in rows},{'piezo','mems','ae'})
            self.assertNotIn('contact_mic',[r['channel'] for r in windows])

if __name__=='__main__':unittest.main(verbosity=2)
