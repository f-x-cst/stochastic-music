from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import stochastic_music  # noqa: E402

APP = str(Path(stochastic_music.__file__).with_name("app.py"))


def run_app():
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    return app


def test_app_loads_and_generates():
    app = run_app()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.get("download_button")) == 2
    assert len(app.get("audio")) == 1


def test_app_multi_voice_every_process_and_clock():
    app = run_app()
    app.sidebar.slider[2].set_value(2).run()  # number of voices
    app.sidebar.slider[3].set_value(0.5).run()  # correlation
    for key in stochastic_music.PROCESSES:
        app.selectbox(key="v0_process").set_value(key).run()
        assert not app.exception
    for rhythm in ["Poisson (random)", "Hawkes (bursts)", "Driven by a process"]:
        app.selectbox(key="v1_clock").set_value(rhythm).run()
        assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
