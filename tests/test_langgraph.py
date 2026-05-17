from langchain_kernelbox import get_session_id_from_config

def test_get_session_id_valid_config():
    config = {"configurable": {"thread_id": "thread_123"}}
    session_id = get_session_id_from_config(config)
    assert session_id == "thread_123"

def test_get_session_id_missing_thread_id():
    config = {"configurable": {"other_key": "value"}}
    session_id = get_session_id_from_config(config, default="fallback")
    assert session_id == "fallback"

def test_get_session_id_empty_config():
    config = {}
    session_id = get_session_id_from_config(config, default="fallback")
    assert session_id == "fallback"

def test_get_session_id_none_config():
    config = None
    session_id = get_session_id_from_config(config, default="fallback")
    assert session_id == "fallback"
