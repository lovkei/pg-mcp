"""MCP stdio 协议验证脚本。

向 pg-mcp 服务器发送 JSON-RPC 请求，等待响应并打印结果。
保持 stdin 打开直到收到响应，避免服务器提前关闭。

用法:
    OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=http://... OPENAI_MODEL=... \
        uv run python verify_mcp.py
"""

import json
import os
import subprocess
import sys
import time

cmd = [sys.executable, "-m", "pg_mcp"]
# stderr 不重定向：服务器的日志和崩溃堆栈直接打到终端，便于排查
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=None,
    text=True,
    env=os.environ.copy(),
)


def send(msg: dict) -> None:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def read_until_response(req_id: int | None, timeout: float = 60.0) -> dict | None:
    """读取 stdout 行，直到拿到匹配 req_id 的响应（跳过通知）。

    等待期间每 5 秒打印一个点：LLM 生成/校验需要时间，
    打点表示脚本仍在等响应，而不是卡死。
    """
    import selectors

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    dot_printed = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if dot_printed:
                print()
            return None
        events = sel.select(min(remaining, 5.0))
        if not events:
            print(".", end="", flush=True)
            dot_printed = True
            continue
        line = proc.stdout.readline()
        if not line:
            # EOF：服务器进程已退出（如 .env 里 key 非法会启动即崩）
            if dot_printed:
                print()
            return None
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == req_id:
            if dot_printed:
                print()
            return msg


# 1. initialize
send({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "verify-script", "version": "1.0"},
    },
})
resp = read_until_response(1)
print("=== initialize ===")
if resp is None:
    print("!! 服务器启动失败，请检查上方堆栈")
    proc.stdin.close()
    proc.terminate()
    sys.exit(1)
print("server:", resp["result"]["serverInfo"])

send({"jsonrpc": "2.0", "method": "notifications/initialized"})

# 2. tools/list
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
resp = read_until_response(2)
print("\n=== tools/list ===")
print("tools:", [t["name"] for t in resp["result"]["tools"]])

# 3. tools/call — 自然语言查询（LLM 生成 + 执行 + 结果校验，需要等待）
question = sys.argv[1] if len(sys.argv) > 1 else "数据库里有多少个用户？"
print(f"\n=== tools/call query: {question} ===")
print("等待 LLM 生成 SQL 并执行（两次 LLM 调用，通常 10~60 秒）…", flush=True)
t0 = time.monotonic()
send({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {"name": "query", "arguments": {"question": question, "return_type": "result"}},
})
resp = read_until_response(3, timeout=120)
print(f"耗时 {time.monotonic() - t0:.1f} 秒")
if resp is None:
    print("!! 未收到响应 —— 服务器可能启动失败，请检查：")
    print("   1) .env 的 OPENAI_API_KEY 是否为真实 sk- 开头的 key")
    print("   2) pg-mcp-db 容器是否在运行 (docker ps)")
elif "error" in resp:
    print("RPC error:", resp["error"])
else:
    data = json.loads(resp["result"]["content"][0]["text"])
    print("success:", data.get("success"))
    print("sql:", (data.get("generated_sql") or "").strip())
    if data.get("data"):
        print("rows:", data["data"].get("rows"))
        print("row_count:", data["data"].get("row_count"))
    print("confidence:", data.get("confidence"))
    if data.get("error"):
        print("error:", data["error"])

proc.stdin.close()
proc.terminate()
