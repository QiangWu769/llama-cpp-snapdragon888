"""Independent accounting regressions; no phone or SDK required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from summarize_intervals import IDS, QTIMER_HZ, SDK_TICK_NS, packets, summarize, summarize_union, calibration_active_regions, intersection_duration, sdk_timestamp_to_qtimer_ticks, analyze


def fixture(rows):
    metrics = []
    for index, (end, ticks, values) in enumerate(rows):
        end = int(float(round(end * QTIMER_HZ / 1e9)) * SDK_TICK_NS)
        for identifier in IDS:
            value = values.get(identifier, 0.0)
            metrics.append({'id': identifier, 'value': value, 'status': 'reported', 'source_timestamp': {'value': end},
                            'params': [3, value, 0, 0, index, 100, ticks, ticks * 1000 / QTIMER_HZ, 7, 0, 0, 4294967295]})
    return {'domains': {'npu': {'status': 'ok', 'dropped_sample_count': 0, 'truncated': False, 'metrics': metrics}}}


class Accounting(unittest.TestCase):
    def test_verified_sdk_inverse_removes_uptime_scale_bias(self):
        ticks=40413268475448
        sdk=int(float(ticks)*SDK_TICK_NS)
        self.assertEqual(sdk_timestamp_to_qtimer_ticks(sdk),ticks)
        self.assertGreater(ticks*1e9/QTIMER_HZ-sdk,100_000_000)
        with self.assertRaises(ValueError):sdk_timestamp_to_qtimer_ticks(1_000_000_000)

    def test_weighted_rates_preserve_zero(self):
        result = summarize(packets(fixture([(10**9, QTIMER_HZ, {4481: 80.}), (4*10**9, 3*QTIMER_HZ, {4481: 0.})])), 0, 4*10**9)
        self.assertEqual(result['metrics']['4481']['time_weighted_mean'], 20.)
        self.assertEqual(result['metrics']['4481']['reported_mcycles'], 80.)
        self.assertEqual(result['coverage_fraction'], 1.)

    def test_sdk_ratio_mean_not_reference_clock_ratio(self):
        rows = [(10**9, QTIMER_HZ, {4097: 1., 4182: 100., 4481: 1., 4480: 100.}),
                (4*10**9, 3*QTIMER_HZ, {4097: 100., 4182: 100., 4481: 25., 4480: 25.})]
        m = summarize(packets(fixture(rows)), 0, 4*10**9)['metrics']
        self.assertEqual(m['4480']['time_weighted_mean'], 43.75)
        self.assertEqual(m['4481']['reference_clock_equivalent_percent'], 19.)
        self.assertNotAlmostEqual(43.75, 100*m['4481']['reported_mcycles']/m['4097']['reported_mcycles'])

    def test_rounding_overlap_changes_integration_not_coverage(self):
        r = summarize(packets(fixture([(10**9, QTIMER_HZ, {4481: 10.}), (2*10**9, QTIMER_HZ+1, {4481: 10.})])), 0, 2*10**9)
        self.assertEqual(r['coverage_fraction'], 1.)
        self.assertEqual(r['covered_seconds'], 2.)
        self.assertGreater(r['reported_interval_seconds'], 2.)
        self.assertGreater(r['interval_overlap_seconds'], 0.)
        self.assertGreater(r['metrics']['4481']['reported_mcycles'], 20.)

    def test_union_removes_sampling_gaps_and_duplicates(self):
        records = packets(fixture([(i*10**9, QTIMER_HZ, {4481: float(i)}) for i in (1,2,3)]))
        r = summarize_union(records, [(0,10**9), (2*10**9,3*10**9), (0,10**9)])
        self.assertEqual(r['complete_intervals'], 2)
        self.assertEqual(r['window_seconds'], 2.)
        self.assertEqual(r['window_envelope_seconds'], 3.)
        self.assertEqual(r['metrics']['4481']['reported_mcycles'], 4.)

    def test_no_boundary_interpolation(self):
        r = summarize(packets(fixture([(10**9, QTIMER_HZ, {4481: 50.})])), 100, 10**9-1)
        self.assertEqual(r['complete_intervals'], 0)
        self.assertEqual(r['coverage_fraction'], 0)
        self.assertEqual(r['metrics'], {})

    def test_failures_and_nonfinite_metadata_rejected(self):
        original = fixture([(10**9,QTIMER_HZ,{})]); cases=[]
        for field,value in [('dropped_sample_count',1),('truncated',True),('invalid_sample_count',1),('stop_api_status',7)]:
            c=copy.deepcopy(original);c['domains']['npu'][field]=value;cases.append(c)
        for field in ['cleanup_failed','callback_release_error_count']:
            c=copy.deepcopy(original);c[field]=True;cases.append(c)
        for parameter,value in [(6,float('nan')),(6,0),(6,True),(7,float('nan')),(4,-1)]:
            c=copy.deepcopy(original)
            for metric in c['domains']['npu']['metrics']:metric['params'][parameter]=value
            cases.append(c)
        for c in cases:
            with self.subTest(case=c),self.assertRaises(ValueError):packets(c)

    def test_gaps_duplicates_and_missing_metrics_rejected(self):
        for kind in ['gap','duplicate','missing']:
            c=fixture([(10**9,QTIMER_HZ,{}),(2*10**9,QTIMER_HZ,{})]);m=c['domains']['npu']['metrics']
            if kind=='gap':
                for r in m[len(IDS):]:r['source_timestamp']['value']+=10000
            elif kind=='duplicate':m.append(copy.deepcopy(m[0]))
            else:m.pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):packets(c)

    def test_active_period_overlap_and_zero_duty(self):
        b={'qtimer_start':0,'qtimer_end':2*QTIMER_HZ,'active_qtimer':QTIMER_HZ//2,
           'periods':[{'index':i,'start_qtimer':i*QTIMER_HZ,'end_qtimer':(i+1)*QTIMER_HZ,'active_qtimer':QTIMER_HZ//4} for i in (0,1)]}
        a=calibration_active_regions(b)
        self.assertEqual(intersection_duration([(125000000,1125000000)],a),250000000)
        self.assertEqual(intersection_duration([(250000000,10**9)],a),0)
        self.assertEqual(intersection_duration([(0,2*10**9)]*2,a),500000000)
        for p in b['periods']:p['active_qtimer']=0
        b['active_qtimer']=0
        self.assertEqual(intersection_duration([(0,2*10**9)],calibration_active_regions(b)),0)

    def test_probe_clock_brackets_and_raw_capture_preserved(self):
        benchmark={'passed':True,'qtimer_hz':QTIMER_HZ,'host_cntfrq_hz':QTIMER_HZ,
                   'host_rpc_start_cntvct':0,'host_rpc_end_cntvct':4*QTIMER_HZ,
                   'qtimer_start':QTIMER_HZ,'qtimer_end':3*QTIMER_HZ,'active_qtimer':QTIMER_HZ,
                   'measured_active_region_percent':50.,
                   'periods':[{'index':i,'start_qtimer':(i+1)*QTIMER_HZ,
                              'end_qtimer':(i+2)*QTIMER_HZ,'active_qtimer':QTIMER_HZ//2} for i in (0,1)]}
        with tempfile.TemporaryDirectory() as path:
            d=Path(path);raw=json.dumps(fixture([(i*10**9,QTIMER_HZ,{}) for i in (1,2,3,4)]))
            (d/'capture.json').write_text(raw)
            (d/'stdout.json').write_text(json.dumps(benchmark))
            result=analyze(d)
            self.assertTrue(result['clock_alignment_checks']['dsp_qtimer_within_host_cntvct_rpc'])
            self.assertEqual(result['windows'][0]['covered_measured_active_region_percent'],50.)
            self.assertEqual((d/'capture.json').read_text(),raw)
            for field,value in [('host_rpc_end_cntvct',2*QTIMER_HZ),('host_cntfrq_hz',1),('passed',False)]:
                c=copy.deepcopy(benchmark);c[field]=value;(d/'stdout.json').write_text(json.dumps(c))
                with self.subTest(field=field),self.assertRaises(ValueError):analyze(d)


if __name__=='__main__':unittest.main()
