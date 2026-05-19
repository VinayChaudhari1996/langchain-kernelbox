import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_kernelbox import KernelBoxTool
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)   

# Use the user's API key if present, otherwise default to a placeholder to prevent crash
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("Please set the GOOGLE_API_KEY environment variable in the .env file")
    exit(0)

@tool
def stateless_python_tool(code: str) -> str:
    """Executes Python code in a STATELESS environment. 
    Variables and imports DO NOT persist between executions.
    You MUST re-import modules and re-define variables every time you call this."""
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        # Empty globals and locals so it's truly stateless
        exec(code, {}, {})
        return mystdout.getvalue()
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout

async def run_benchmark_step_by_step(llm, tool, is_stateless: bool):
    messages = []
    
    if is_stateless:
        system_prompt = (
            "You are a python assistant. You have a STATELESS python environment. "
            "Variables DO NOT persist between steps. "
            "You MUST output the full cumulative code from all previous steps plus the new step. "
            "Output ONLY the python code, nothing else."
        )
    else:
        system_prompt = (
            "You are a python assistant. You have a STATEFUL python environment. "
            "Variables persist. "
            "Output ONLY the new python code for the current step, nothing else."
        )
        
    messages.append(SystemMessage(content=system_prompt))
    
    steps = [
        "Step 1: Set a=10 and print it.",
        "Step 2: Set b=a*2 and print it.",
        "Step 3: Set c=b*2 and print it.",
        "Step 4: Set d=c*2 and print it.",
        "Step 5: Set e=d*2 and print it.",
        "Step 6: Set f=e*2 and print it.",
        "Step 7: Set g=f*2 and print it.",
        "Step 8: Set h=g*2 and print it.",
        "Step 9: Set i=h*2 and print it.",
        "Step 10: Set j=i*2 and print it.",
    ]
    
    total_tokens = 0
    token_history = []
    
    for idx, step in enumerate(steps):
        messages.append(("human", step))
        
        # 1. Ask LLM for the code
        response = await llm.ainvoke(messages)
        
        step_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            step_tokens = response.usage_metadata.get("total_tokens", 0)
            
        total_tokens += step_tokens
        token_history.append(step_tokens)
            
        messages.append(response)
        
        # 2. Extract code
        content = response.content
        if isinstance(content, list):
            code = ""
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    code += block["text"] + "\n"
                elif isinstance(block, str):
                    code += block + "\n"
            code = code.strip()
        else:
            code = str(content).strip()
            
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:-1])
            
        # 3. Run tool
        print(f"  [Step {idx+1}] Executing generated code...")
        if is_stateless:
            tool_output = tool.invoke({"code": code})
        else:
            tool_output = tool.invoke({"code": code})
            
        # 4. Feed result back
        messages.append(("human", f"Tool Output:\n{tool_output}"))
        
    return total_tokens, token_history

async def main():
    print("Initializing Google Gemini Model (gemini-3.1-flash-lite)...")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            temperature=0, 
            top_k=1, 
            top_p=0.0,
            seed=42
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return
        
    kernelbox_tool = KernelBoxTool()
    
    print("\nRunning Stateful Benchmark (KernelBox)...")
    stateful_total, stateful_history = await run_benchmark_step_by_step(llm, kernelbox_tool, is_stateless=False)
    print(f"  Total Tokens:  {stateful_total}")
    
    print("\nRunning Stateless Benchmark...")
    stateless_total, stateless_history = await run_benchmark_step_by_step(llm, stateless_python_tool, is_stateless=True)
    print(f"  Total Tokens:  {stateless_total}")
    
    print("\n============================================================")
    print("Token Usage Benchmark: Stateless Tool vs KernelBox")
    print("============================================================")
    print(f"{'Step':<10} | {'Stateless Tokens':<22} | {'KernelBox Tokens':<20}")
    print("-" * 65)
    
    for i in range(len(stateful_history)):
        print(f"Step {i+1:<5} | {stateless_history[i]:<22} | {stateful_history[i]:<20}")
        
    print("-" * 65)
    print(f"{'TOTAL':<10} | {stateless_total:<22} | {stateful_total:<20}")
    print("============================================================")
    
    if stateless_total > 0 and stateful_total > 0:
        savings = (stateless_total - stateful_total) / stateless_total * 100
        print(f"\nConclusion: KernelBox saves ~{savings:.1f}% on LLM API costs!")

if __name__ == "__main__":
    asyncio.run(main())
