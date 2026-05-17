from langchain_core.runnables.config import RunnableConfig

def get_session_id_from_config(config: RunnableConfig, default: str = "default_session") -> str:
    """
    Extracts the thread_id from a LangGraph configuration to be used as a KernelBox session_id.
    
    Args:
        config (RunnableConfig): The LangGraph configuration object.
        default (str): The default session ID to use if thread_id is not found.
        
    Returns:
        str: The extracted thread_id or the default session ID.
    """
    if not config:
        return default
        
    configurable = config.get("configurable", {})
    return configurable.get("thread_id", default)
