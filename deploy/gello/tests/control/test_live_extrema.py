import time
from unittest.mock import Mock

import numpy as np
import pytest

from deploy.gello.control import gello_cli


@pytest.mark.parametrize("finish_during_dropout", [False, True])
def test_extrema_survive_drop_and_enter_accepts_without_restarting(
    monkeypatch, finish_during_dropout
):
    first, second = np.zeros(7), np.zeros(7)
    first[6], second[6] = -0.3, 0.2
    bus = Mock(paced=True)
    bus.read.side_effect = [
        (first, time.monotonic()),
        OSError("missing reply"),
        (second, time.monotonic()),
    ]
    capture = gello_cli.Capture(bus, list(range(1, 8)))
    capture.accepted = {"home_raw": [0] * 7}
    capture.restore_accepted_travel()
    monkeypatch.setattr(gello_cli.sys, "stdin", Mock(isatty=lambda: True, readline=lambda: "\n"))
    polls = iter([False, False, True] if finish_during_dropout else [False, False, False, True])
    monkeypatch.setattr(
        gello_cli.select, "select", lambda *args: ([True] if next(polls) else [], [], [])
    )
    monkeypatch.setattr(gello_cli.time, "sleep", lambda _: None)
    capture.capture_limits([6], "trigger")
    assert capture.low[6] == -0.3
    assert capture.high[6] == (0 if finish_during_dropout else 0.2)
    assert bus.read.call_count == (2 if finish_during_dropout else 3)
    assert bus.reopen.call_count == (0 if finish_during_dropout else 1)
