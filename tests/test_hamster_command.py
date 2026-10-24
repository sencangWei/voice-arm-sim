"""Child ownership on cancellation; no model weights or real user jobs needed."""
import signal
import subprocess
from unittest.mock import Mock

import pytest

from vla.hamster_command import run_stage, save_report


def test_interruption_stops_owned_child(monkeypatch):
    child = Mock()
    child.poll.return_value = None
    child.wait.side_effect = [KeyboardInterrupt(), 0]
    monkeypatch.setattr('vla.hamster_command.subprocess.Popen', Mock(return_value=child))
    with pytest.raises(KeyboardInterrupt):
        run_stage(['local-model'], {}, 20)
    child.send_signal.assert_called_once_with(signal.SIGINT)
    child.kill.assert_not_called()


def test_failed_stage_is_not_reported_as_finished(monkeypatch):
    child = Mock()
    child.wait.return_value = 7
    child.poll.return_value = 7
    monkeypatch.setattr('vla.hamster_command.subprocess.Popen', Mock(return_value=child))
    with pytest.raises(subprocess.CalledProcessError):
        run_stage(['local-model'], {}, 20)
    child.send_signal.assert_not_called()


def test_progress_is_replaced_atomically(tmp_path):
    save_report(tmp_path, {'result': 'started'})
    save_report(tmp_path, {'result': 'interrupted'})
    assert 'interrupted' in (tmp_path / 'workflow.json').read_text()
    assert not (tmp_path / 'workflow.tmp').exists()
