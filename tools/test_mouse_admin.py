"""Quick admin mouse probe: move cursor onto the game window and verify."""
import ctypes
import os
import sys
import time

try:
    import win32con
    import win32gui
except ImportError:
    win32gui = None
    win32con = None

user32 = ctypes.windll.user32
shell32 = ctypes.windll.shell32

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_debug_out",
    "mouse_admin_test.txt",
)


def get_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    p = POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def main() -> int:
    lines = []
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def log(msg: str):
        print(msg)
        lines.append(msg)

    admin = bool(shell32.IsUserAnAdmin())
    log(f"bot_admin={admin}")
    if not admin:
        log("FAIL: not elevated")
        open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        return 2

    hwnd = 0
    if win32gui is not None:
        hwnd = win32gui.FindWindow(None, "冒险岛怀旧服") or 0
    log(f"game_hwnd={hwnd}")
    if hwnd and win32gui is not None:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            log(f"fg_err={e}")
        time.sleep(0.4)
        target = win32gui.ClientToScreen(hwnd, (400, 300))
    else:
        before = get_pos()
        target = (before[0] + 100, before[1] + 60)

    before = get_pos()
    log(f"before={before} target={target}")
    ret = user32.SetCursorPos(int(target[0]), int(target[1]))
    time.sleep(0.15)
    after = get_pos()
    dist = ((after[0] - target[0]) ** 2 + (after[1] - target[1]) ** 2) ** 0.5
    log(f"SetCursorPos_ret={ret} after={after} miss={dist:.1f}px")
    ok = dist < 20
    log("RESULT " + ("PASS" if ok else "FAIL"))
    user32.SetCursorPos(int(before[0]), int(before[1]))
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    time.sleep(0.5)
    sys.exit(code)
