"""Verify the completion-wait correction cannot admit escaping device work."""
import hashlib
import types
from pathlib import Path
import json, tempfile, unittest
R = Path(__file__).resolve().parent

def load(name, path):
    value = json.loads(path.read_bytes())
    source = value['source_utf8'].encode('utf8')
    assert hashlib.sha256(source).hexdigest() == value['sha256']
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module
strict = load('strict', R / 'source-accounting-strict.json')
fixed = load('fixed', R / 'accounting-source.json')

def fixture():
    metadata = [{'ph': 'M', 'name': 'process_name', 'pid': 1, 'args': {'name': '[1] Runtime API IP:host pid:77'}}, {'ph': 'M', 'name': 'process_name', 'pid': 2, 'args': {'name': '[2] Compute on Device 0 IP:host pid:77'}}]

    def event(name, track, lo, hi, index):
        return {'ph': 'X', 'name': name, 'cat': 'HIP' if track == 1 else 'HIPOPS', 'pid': track, 'tid': 'Thread3', 'ts': str(lo / 1000), 'dur': str((hi - lo) / 1000), 'args': {'pid': 77, 'tid': 3, 'BeginNs': str(lo), 'EndNs': str(hi), 'DurationNs': hi - lo, 'index': index}}
    events = metadata + [event('hipModuleLaunchKernel', 1, 2000, 2200, 5), event('add_kernel', 2, 2400, 3400, 5), event('hipDeviceSynchronize', 1, 8000, 9050, 6)]
    worker = {'identity': {'pid': 77, 'host': 'host'}, 'route': 'fg', 'status': 'ok', 'records': [{'id': 'case', 'status': 'ok', 'correct': True, 'roi': [{'rep': 0, 'valid': True, 'tid': 3, 'start_epoch_ns': 1000, 'end_epoch_ns': 9000}]}]}
    return ({'traceEvents': events}, worker)

class SyncBoundaryTests(unittest.TestCase):

    def evaluate(self, mutate=lambda t, w: None):
        trace, worker = fixture()
        mutate(trace, worker)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'trace.json'
            q = root / 'worker.json'
            p.write_text(json.dumps(trace))
            q.write_text(json.dumps(worker))
            before = (p.read_bytes(), q.read_bytes())
            a = strict.analyze(p, q, expected_rois=1)
            b = fixed.analyze(p, q, expected_rois=1)
            self.assertEqual(before, (p.read_bytes(), q.read_bytes()))
            return (a, b)

    def test_only_completion_wait_changes_acceptance(self):
        a, b = self.evaluate()
        self.assertEqual(a['cases']['case']['rois'][0]['errors'], ['host activity crosses ROI boundary: 4'])
        self.assertEqual(b['status'], 'ok', b)
        sample = b['cases']['case']['rois'][0]
        self.assertEqual(sample['kernel_us'], 1.0)
        self.assertEqual(sample['kernel_count'], 1)
        self.assertEqual(sample['completion_sync_events'][0]['host_end_overhang_ns'], '50')
        self.assertEqual(a['cases']['case']['rois'][0]['links'], sample['links'])

    def test_previous_valid_result_is_bit_exact(self):

        def change(t, w):
            e = t['traceEvents'][-1]
            e['args'].update(EndNs='8950', DurationNs=950)
        a, b = self.evaluate(change)
        self.assertEqual(a['status'], 'ok', a)
        self.assertEqual(b['status'], 'ok', b)
        self.assertEqual({k: v for k, v in b['cases']['case']['rois'][0].items() if k != 'completion_sync_events'}, a['cases']['case']['rois'][0])

    def assert_invalid(self, mutate):
        a, b = self.evaluate(mutate)
        self.assertEqual(a['status'], 'error')
        self.assertEqual(b['status'], 'error', b)

    def test_launch_end_crossing_still_rejected(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][2]['args'].update(EndNs='9100', DurationNs=7100))

    def test_launch_start_crossing_still_rejected(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][2]['args'].update(BeginNs='900', DurationNs=1300))

    def test_device_end_crossing_still_rejected(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][3]['args'].update(EndNs='9100', DurationNs=6700))

    def test_left_crossing_sync_still_rejected(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][-1]['args'].update(BeginNs='900', DurationNs=8150))

    def test_foreign_thread_sync_still_rejected(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][-1]['args'].update(tid=4))

    def test_unrecognized_launch_cannot_become_a_wait(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][-1].update(name='hipLaunchKernel'))

    def test_missing_producer_still_rejected(self):
        self.assert_invalid(lambda t, w: t['traceEvents'][3]['args'].update(index=88))
if __name__ == '__main__':
    unittest.main()
