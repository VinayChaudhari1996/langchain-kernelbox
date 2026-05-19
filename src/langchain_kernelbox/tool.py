import uuid
import asyncio
from typing import Optional, Type, Any, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from kernelbox import get_or_create, execute, destroy


class KernelBoxInput(BaseModel):
    """Schema for KernelBoxTool input."""
    code: str = Field(..., description="The code to execute in the stateful IPython kernel.")
    language: Literal["python", "bash"] = Field(
        default="python", 
        description="The language to execute. 'python' for python code, 'bash' for shell commands."
    )
    timeout: Optional[int] = Field(
        default=None, 
        description="Optional maximum execution time in seconds."
    )


class KernelBoxTool(BaseTool):
    """
    A LangChain Tool for executing Python code in a stateful IPython environment using KernelBox.
    """
    name: str = "python_repl_stateful"
    description: str = (
        "A stateful execution environment. Use this to execute Python code or bash commands. "
        "Variables, imports, and functions persist across executions within the same session. "
        "Can also execute bash by setting language='bash' and has optional timeout controls."
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
        if getattr(result, "output", None):
            out.append(f"STDOUT:\n{result.output}")
        if getattr(result, "stderr", None):
            out.append(f"STDERR:\n{result.stderr}")
            
        error = getattr(result, "error", None)
        if error is not None:
            ename = getattr(error, "ename", "")
            evalue = getattr(error, "evalue", "")
            traceback = "\n".join(getattr(error, "traceback", []))
            out.append(f"ERROR: {ename}: {evalue}\n{traceback}".strip())
            
        if getattr(result, "return_value", None) is not None:
            out.append(f"RETURN VALUE:\n{result.return_value}")
            
        if getattr(result, "truncated", False):
            out.append("\n[WARNING: Output was truncated due to length limits.]")
        
        if not out:
            return "Execution completed successfully without output."
        return "\n\n".join(out)

    def _run(
        self,
        code: str,
        language: Literal["python", "bash"] = "python",
        timeout: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        kernel = get_or_create(self.session_id)
        result = execute(kernel, code, language=language, timeout=timeout)
        return self._format_result(result)


    def destroy(self) -> None:
        """Destroy the kernelbox session and clean up resources."""
        destroy(self.session_id)
