import uuid
import asyncio
from typing import Optional, Type, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from kernelbox import get_or_create, execute


class KernelBoxInput(BaseModel):
    """Schema for KernelBoxTool input."""
    code: str = Field(..., description="The Python code to execute in the stateful IPython kernel.")


class KernelBoxTool(BaseTool):
    """
    A LangChain Tool for executing Python code in a stateful IPython environment using KernelBox.
    """
    name: str = "python_repl_stateful"
    description: str = (
        "A stateful Python REPL tool. Use this to execute Python code. "
        "Variables, imports, and functions persist across executions within the same session. "
        "Input should be valid Python code."
    )
    args_schema: Type[BaseModel] = KernelBoxInput
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The KernelBox session ID.")

    def __init__(self, session_id: Optional[str] = None, **kwargs: Any):
        if session_id:
            kwargs["session_id"] = session_id
        super().__init__(**kwargs)

    def _format_result(self, result: Any) -> str:
        """Format the kernelbox ExecutionResult for the LLM."""
        out = []
        if getattr(result, "stdout", None):
            out.append(f"STDOUT:\n{result.stdout}")
        if getattr(result, "stderr", None):
            out.append(f"STDERR:\n{result.stderr}")
        if getattr(result, "return_value", None) is not None:
            out.append(f"RETURN VALUE:\n{result.return_value}")
        
        if not out:
            return "Execution completed successfully without output."
        return "\n\n".join(out)

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        kernel = get_or_create(self.session_id)
        result = execute(kernel, code)
        return self._format_result(result)

    async def _arun(
        self,
        code: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        loop = asyncio.get_running_loop()
        
        def run_sync():
            kernel = get_or_create(self.session_id)
            return execute(kernel, code)
            
        result = await loop.run_in_executor(None, run_sync)
        return self._format_result(result)
