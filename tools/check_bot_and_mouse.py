"""Check elevated bot + probe mouse while game is focused."""
import ctypes
import os
import re
import subprocess
import time
from ctypes import wintypes

import win32con
import win32gui
import win32process

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
advapi32 = ctypes.windll.advapi32
shell32 = ctypes.windll.shell32

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_debug_out",
    "check_bot_mouse.txt",
)


def elev(pid):
    h = kernel32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return None
    try:
        tok = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(h, 0x8, ctypes.byref(tok)):
            return None
        try:

            class TE(ctypes.Structure):
                _fields_ = [("TokenIsElevated", wintypes.DWORD)]

            te = TE()
            n = wintypes.DWORD()
            ok = advapi32.GetTokenInformation(
                tok, 20, ctypes.byref(te), ctypes.sizeof(te), ctypes.byref(n)
            )
            return bool(ok and te.TokenIsElevated)
        finally:
            kernel32.CloseHandle(tok)
    finally:
        kernel32.CloseHandle(h)


def get_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    p = POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    # Find python processes related to this project
    try:
        out = subprocess.check_output(
            [
                "wmic",
                "process",
                "where",
                'name="python.exe"',
                "get",
                "ProcessId,CommandLine,ExecutablePath",
                "/FORMAT:LIST",
            ],
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as e:
        out = ""
        log(f"wmic_err={e}")

    bots = []
    for block in out.split("\n\n"):
        b = block.strip()
        if not b:
            continue
        low = b.lower()
        if "src.main" in low or "maplestory-auto-ds" in low:
            bots.append(b)

    log("=== bot processes ===")
    if not bots:
        log("(none found)")
    for b in bots:
        log(b.replace("\r", ""))
        m = re.search(r"ProcessId=(\d+)", b)
        if m:
            pid = int(m.group(1))
            log(f"pid={pid} elevated={elev(pid)}")

    hwnd = win32gui.FindWindow(None, "冒险岛怀旧服")
    game_pid = None
    if hwnd:
        _, game_pid = win32process.GetWindowThreadProcessId(hwnd)
    log(f"game_hwnd={hwnd} game_pid={game_pid} elevated={elev(game_pid) if game_pid else None}")
    log(f"this_probe_admin={bool(shell32.IsUserAnAdmin())}")

    # Mouse probe (will fail if this probe is not admin while game is)
    if hwnd:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            log(f"fg_err={e}")
        time.sleep(0.35)
        target = win32gui.ClientToScreen(hwnd, (420, 320))
    else:
        b = get_pos()
        target = (b[0] + 80, b[1] + 50)

    before = get_pos()
    ret = user32.SetCursorPos(int(target[0]), int(target[1]))
    time.sleep(0.12)
    after = get_pos()
    miss = ((after[0] - target[0]) ** 2 + (after[1] - target[1]) ** 2) ** 0.5
    log(f"before={before} target={target} ret={ret} after={after} miss={miss:.1f}")
    ok = miss < 20
    log("MOUSE_PROBE " + ("PASS" if ok else "FAIL"))
    user32.SetCursorPos(int(before[0]), int(before[1]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
