import pytest
from unittest.mock import patch, MagicMock
from langchain_kernelbox import KernelBoxTool

class MockExecutionResult:
    def __init__(self, output="", stderr="", error=None, return_value=None, truncated=False):
        self.output = output
        self.stderr = stderr
        self.error = error
        self.return_value = return_value
        self.truncated = truncated

@patch("langchain_kernelbox.tool.get_or_create")
@patch("langchain_kernelbox.tool.execute")
def test_kernelbox_tool_sync(mock_execute, mock_get_or_create):
    # Setup mock
    mock_kernel = MagicMock()
    mock_get_or_create.return_value = mock_kernel
    mock_execute.return_value = MockExecutionResult(output="test output", return_value=42)
    
    # Run tool
    tool = KernelBoxTool(session_id="test_session")
    result = tool.run({"code": "print('test output')\n42"})
    
    # Assertions
    mock_get_or_create.assert_called_once_with("test_session")
    mock_execute.assert_called_once_with(mock_kernel, "print('test output')\n42", language="python", timeout=None)
    
    assert "STDOUT:\ntest output" in result
    assert "RETURN VALUE:\n42" in result

@pytest.mark.asyncio
@patch("langchain_kernelbox.tool.get_or_create")
@patch("langchain_kernelbox.tool.execute")
async def test_kernelbox_tool_async(mock_execute, mock_get_or_create):
    # Setup mock
    mock_kernel = MagicMock()
    mock_get_or_create.return_value = mock_kernel
    mock_execute.return_value = MockExecutionResult(stderr="error occurred")
    
    # Run tool
    tool = KernelBoxTool(session_id="test_async_session")
    result = await tool.arun({"code": "1 / 0"})
    
    # Assertions
    mock_get_or_create.assert_called_once_with("test_async_session")
    mock_execute.assert_called_once_with(mock_kernel, "1 / 0", language="python", timeout=None)
    
    assert "STDERR:\nerror occurred" in result

@patch("langchain_kernelbox.tool.get_or_create")
@patch("langchain_kernelbox.tool.execute")
def test_kernelbox_tool_no_output(mock_execute, mock_get_or_create):
    mock_kernel = MagicMock()
    mock_get_or_create.return_value = mock_kernel
    mock_execute.return_value = MockExecutionResult()
    
    tool = KernelBoxTool()
    result = tool.run({"code": "x = 10"})
    
    assert result == "Execution completed successfully without output."
