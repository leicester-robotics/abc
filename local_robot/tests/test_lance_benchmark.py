from local_robot.lance_benchmark import summarize


def test_timing_summary_reports_observed_latency_and_cpu():
    result = summarize([.01, .02, .03], cpu_seconds=.09)
    assert abs(result['samples_per_second'] - 50) < 1e-9
    assert abs(result['median_ms'] - 20) < 1e-9
    assert abs(result['cpu_percent_one_core'] - 150) < 1e-9
    assert result['p95_ms'] > 20


def test_all_sampled_indices_are_warm_before_timing():
    from local_robot.lance_benchmark import measure_loaders
    clock = [0.]
    class Dataset:
        def __init__(self): self.seen = set()
        def __getitem__(self, index):
            clock[0] += .001 if index in self.seen else .1
            self.seen.add(index)
    durations, cpu = measure_loaders({'a': Dataset(), 'b': Dataset()}, [0,1,2,3], 2,
                                     clock=lambda: clock[0], cpu_clock=lambda: clock[0])
    for values in durations.values():
        assert len(values) == 8
        assert all(abs(t-.001)<1e-9 for t in values)
