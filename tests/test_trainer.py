"""Unit tests for trainer/continuous_trainer.py."""

import os
import json
import time
import pytest
from unittest.mock import patch

import continuous_trainer


# ===== LearnState =====

class TestLearnState:
    def test_initial_state(self, tmp_data_dir):
        with patch.object(continuous_trainer, 'STATE_FILE',
                         os.path.join(tmp_data_dir, 'learn_state.json')):
            state = continuous_trainer.LearnState()
            assert state.total_steps == 0
            assert state.total_batches == 0
            assert state.last_train_time is None

    def test_save_and_load(self, tmp_data_dir):
        state_file = os.path.join(tmp_data_dir, 'learn_state.json')
        with patch.object(continuous_trainer, 'STATE_FILE', state_file), \
             patch.object(continuous_trainer, 'DATA_DIR', tmp_data_dir):
            state = continuous_trainer.LearnState()
            state.total_steps = 100
            state.total_batches = 5
            state.save()

            state2 = continuous_trainer.LearnState()
            assert state2.total_steps == 100
            assert state2.total_batches == 5

    def test_record_batch(self, tmp_data_dir):
        state_file = os.path.join(tmp_data_dir, 'learn_state.json')
        with patch.object(continuous_trainer, 'STATE_FILE', state_file), \
             patch.object(continuous_trainer, 'DATA_DIR', tmp_data_dir):
            state = continuous_trainer.LearnState()
            state.record_batch(25)
            assert state.total_steps == 25
            assert state.total_batches == 1
            assert state.last_train_time is not None

            state.record_batch(30)
            assert state.total_steps == 55
            assert state.total_batches == 2

    def test_load_corrupted_file(self, tmp_data_dir):
        state_file = os.path.join(tmp_data_dir, 'learn_state.json')
        with open(state_file, 'w') as f:
            f.write("not json{{{")

        with patch.object(continuous_trainer, 'STATE_FILE', state_file):
            state = continuous_trainer.LearnState()
            assert state.total_steps == 0


# ===== ChangeAccumulator =====

class TestChangeAccumulator:
    def test_empty_not_ready(self):
        acc = continuous_trainer.ChangeAccumulator()
        assert not acc.ready()

    def test_add_and_ready(self):
        acc = continuous_trainer.ChangeAccumulator()
        acc.add("/tmp/test.py")
        # Not ready immediately (debounce)
        assert not acc.ready()

    def test_ready_after_debounce(self):
        acc = continuous_trainer.ChangeAccumulator()
        acc.add("/tmp/test.py")
        # Simulate debounce passed
        acc.last_change_time = time.time() - continuous_trainer.DEBOUNCE_SECS - 1
        assert acc.ready()

    def test_consume_clears(self):
        acc = continuous_trainer.ChangeAccumulator()
        acc.add("/tmp/a.py")
        acc.add("/tmp/b.py")
        acc.last_change_time = time.time() - continuous_trainer.DEBOUNCE_SECS - 1

        files = acc.consume()
        assert len(files) == 2
        assert "/tmp/a.py" in files
        assert "/tmp/b.py" in files
        assert not acc.ready()  # cleared

    def test_deduplication(self):
        acc = continuous_trainer.ChangeAccumulator()
        acc.add("/tmp/test.py")
        acc.add("/tmp/test.py")
        acc.add("/tmp/test.py")
        acc.last_change_time = time.time() - continuous_trainer.DEBOUNCE_SECS - 1

        files = acc.consume()
        assert len(files) == 1


# ===== Step Calculation =====

class TestCalculateSteps:
    def test_minimum_steps(self):
        steps = continuous_trainer.calculate_steps(0)
        assert steps == continuous_trainer.MIN_STEPS

    def test_one_file(self):
        steps = continuous_trainer.calculate_steps(1)
        assert steps == continuous_trainer.MIN_STEPS + 2

    def test_many_files(self):
        steps = continuous_trainer.calculate_steps(100)
        assert steps == continuous_trainer.MAX_STEPS

    def test_scales_linearly(self):
        s1 = continuous_trainer.calculate_steps(5)
        s2 = continuous_trainer.calculate_steps(10)
        assert s2 > s1

    def test_capped_at_max(self):
        steps = continuous_trainer.calculate_steps(1000)
        assert steps == continuous_trainer.MAX_STEPS


# ===== Config Loading =====

class TestLoadWatchDirs:
    def test_from_config(self, tmp_data_dir):
        config_path = os.path.join(tmp_data_dir, "config.json")
        config = {
            "sources": [
                {"name": "Home", "path": os.path.expanduser("~"), "enabled": True},
                {"name": "Disabled", "path": "/tmp", "enabled": False},
            ]
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        with patch.object(continuous_trainer, 'CONFIG_FILE', config_path):
            dirs = continuous_trainer.load_watch_dirs()
            assert os.path.expanduser("~") in dirs
            assert "/tmp" not in dirs

    def test_fallback_no_config(self):
        with patch.object(continuous_trainer, 'CONFIG_FILE', '/nonexistent/config.json'):
            dirs = continuous_trainer.load_watch_dirs()
            assert isinstance(dirs, list)

    def test_fallback_invalid_json(self, tmp_data_dir):
        config_path = os.path.join(tmp_data_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write("invalid json{{{")

        with patch.object(continuous_trainer, 'CONFIG_FILE', config_path):
            dirs = continuous_trainer.load_watch_dirs()
            assert isinstance(dirs, list)


# ===== Daemon Status =====

class TestDaemonStatus:
    def test_stop_no_pid(self, tmp_data_dir, capsys):
        with patch.object(continuous_trainer, 'PID_FILE',
                         os.path.join(tmp_data_dir, 'learn.pid')):
            continuous_trainer.stop_daemon()
            captured = capsys.readouterr()
            assert "No learn daemon" in captured.out

    def test_stop_stale_pid(self, tmp_data_dir, capsys):
        pid_path = os.path.join(tmp_data_dir, 'learn.pid')
        with open(pid_path, 'w') as f:
            f.write("99999999")  # Non-existent PID

        with patch.object(continuous_trainer, 'PID_FILE', pid_path):
            continuous_trainer.stop_daemon()
            captured = capsys.readouterr()
            assert "not running" in captured.out.lower() or "stale" in captured.out.lower()
            assert not os.path.exists(pid_path)

    def test_status_not_running(self, tmp_data_dir, capsys):
        with patch.object(continuous_trainer, 'PID_FILE',
                         os.path.join(tmp_data_dir, 'learn.pid')), \
             patch.object(continuous_trainer, 'STATE_FILE',
                         os.path.join(tmp_data_dir, 'learn_state.json')):
            continuous_trainer.show_status()
            captured = capsys.readouterr()
            assert "NOT RUNNING" in captured.out
