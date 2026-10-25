"""Experiment launch ownership and overlap guards without creating a GUI."""
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vla.panel import ChinesePanel


def panel_stub():
    panel = ChinesePanel.__new__(ChinesePanel)
    panel.hybrid = panel.training = None
    panel.hybrid_output = None
    panel.hybrid_status = SimpleNamespace(set=Mock())
    panel.prompt = SimpleNamespace(get=lambda: '把酒瓶放进碗里')
    panel.lab = SimpleNamespace(state=lambda: {'running': False, 'manual': False})
    return panel


@pytest.mark.parametrize("limit", [0, 360])
def test_run_forwards_visible_step_limit(limit):
    panel = panel_stub()
    panel.step_limit = SimpleNamespace(get=lambda: str(limit))
    panel.send = Mock()
    panel.run_model()
    panel.send.assert_called_once_with("run", '把酒瓶放进碗里', limit)


def test_invalid_step_limit_does_not_start_model():
    panel = panel_stub()
    panel.step_limit = SimpleNamespace(get=lambda: "-1")
    panel.send = Mock()
    with pytest.raises(ValueError, match="0～10000"):
        panel.run_model()
    panel.send.assert_not_called()


def test_launch_uses_unique_output_real_policy_and_native_viewer(tmp_path, monkeypatch):
    panel = panel_stub()
    plans = tmp_path / 'evidence/hamster/trial-init0-protobuf'
    plans.mkdir(parents=True)
    (plans / 'command-01.json').write_text('{}')
    monkeypatch.setattr('vla.panel.ROOT', tmp_path)
    popen = Mock(return_value=SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr('vla.panel.subprocess.Popen', popen)
    panel.start_hybrid('01')
    command = popen.call_args.args[0]
    assert command[:4] == [str(tmp_path / '.venv-vla-sim/bin/python'), '-u', '-m', 'vla.hybrid']
    assert '--viewer' in command and '--policy-only' not in command
    assert popen.call_args.kwargs['start_new_session'] is True
    assert command[command.index('--plan')+1] == str(plans / 'command-01.json')
    assert not panel.hybrid_output.exists()  # No prefilled output that trips overwrite protection.
    assert panel.hybrid_output.with_suffix('.log').is_file()
    with pytest.raises(ValueError, match='仍在运行'):
        panel.start_hybrid('01')


def test_live_and_manual_sessions_block_experiment():
    panel = panel_stub()
    for state in ({'running': True, 'manual': False}, {'running': False, 'manual': True}):
        panel.lab.state = lambda: state
        with pytest.raises(ValueError, match='暂停当前模型'):
            panel.start_hybrid('01')


def test_fresh_instruction_is_forwarded_to_real_planner(tmp_path, monkeypatch):
    panel = panel_stub()
    monkeypatch.setattr('vla.panel.ROOT', tmp_path)
    popen = Mock(return_value=SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr('vla.panel.subprocess.Popen', popen)
    panel.start_hybrid('fresh')
    command = popen.call_args.args[0]
    assert command[3] == 'vla.hamster_command'
    assert command[command.index('--prompt')+1] == '把酒瓶放进碗里'
    assert '--plan' not in command


def test_stop_only_signals_process_group_owned_by_panel(monkeypatch):
    panel = panel_stub()
    panel.stop_hybrid()
    process = SimpleNamespace(poll=lambda: None, pid=123456)
    killpg = Mock()
    monkeypatch.setattr('vla.panel.os.killpg', killpg)
    panel.hybrid = process
    panel.stop_hybrid()
    killpg.assert_called_once_with(123456, signal.SIGINT)
    with pytest.raises(ValueError, match='停止分层实验'):
        panel.send('run')
