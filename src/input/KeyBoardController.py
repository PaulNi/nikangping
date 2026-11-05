"""
Keyboard controller for sending movement commands.
Uses pyautogui for movement keys, pynput for jump/teleport (better game compat).
Runs in daemon thread, consumes commands from set_command().
"""
import time
import random
import threading

import pyautogui
from pynput.keyboard import Key, Controller as KBController

from src.utils.logger import logger
from src.utils.common import activate_game_window, get_game_window_title_by_token, is_windows

pyautogui.FAILSAFE = False
_pynput_kb = KBController()


def _resolve_pynput_key(key):
    """Resolve string key names like 'home'/'end'/'shift' to pynput Key constants."""
    if isinstance(key, str):
        normalized = key.strip().lower()
        aliases = {
            "shift": getattr(Key, "shift_l", Key.shift),
            "ctrl": getattr(Key, "ctrl_l", Key.ctrl),
            "alt": getattr(Key, "alt_l", Key.alt),
            "space": Key.space,
            "enter": Key.enter,
            "esc": Key.esc,
            "escape": Key.esc,
            "pagedown": Key.page_down,
            "page_down": Key.page_down,
            "page down": Key.page_down,
            "pgdn": Key.page_down,
            "pageup": Key.page_up,
            "page_up": Key.page_up,
            "pgup": Key.page_up,
        }
        if normalized in aliases:
            return aliases[normalized]
        if hasattr(Key, normalized):
            return getattr(Key, normalized)
    return key


def key_down(key):
    """Press and hold a key. Try pynput first, then pyautogui."""
    try:
        _pynput_kb.press(_resolve_pynput_key(key))
        return True
    except Exception:
        pass
    try:
        pyautogui.keyDown(key)
        return True
    except Exception:
        return False


def key_up(key):
    """Release a held key. Try pynput first, then pyautogui."""
    try:
        _pynput_kb.release(_resolve_pynput_key(key))
        return True
    except Exception:
        pass
    try:
        pyautogui.keyUp(key)
        return True
    except Exception:
        return False


def press_key(key: str, duration=0.2) -> bool:
    """Press and release a key. Returns True on success."""
    # Try pynput first (works better for many in-game keys).
    try:
        resolved_key = _resolve_pynput_key(key)
        _pynput_kb.press(resolved_key)
        time.sleep(duration)
        _pynput_kb.release(resolved_key)
        return True
    except Exception:
        pass

    # Fallback to pyautogui for keys that are not accepted by pynput/game.
    try:
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
        return True
    except Exception:
        return False


class KeyBoardController:
    """
    Sends movement commands to the game via pyautogui.
    Runs in a daemon thread, polling for new commands.

    Usage:
        kb = KeyBoardController(cfg)
        kb.set_command("left none none")   # walk left
        kb.set_command("none none jump")   # jump
        kb.set_command("none none none")   # stop
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # start_idle: wait for F1/F2 before combat
        self.is_enable = not bool((cfg.get("bot") or {}).get("start_idle", True))
        self.is_terminated = False
        self._active_profile = None
        self._detector = None  # optional MonsterDetector for nameplate reload

        # Key bindings
        key_cfg = cfg.get("key", {})
        self.key_jump = key_cfg.get("jump", "space")
        self.key_teleport = key_cfg.get("teleport", "e")
        self.key_directional_attack = key_cfg.get("directional_attack", "w")
        self.key_aoe_skill = key_cfg.get("aoe_skill", "q")
        self.key_pickup = key_cfg.get("pickup", "z")
        self.jump_hold_up_seconds = float(key_cfg.get("jump_hold_up_seconds", 2.0))
        self.attack_mode = cfg.get("bot", {}).get("attack", "directional")
        self._t_last_attack = 0.0
        self._aoe_next_cd = 0.0  # next A delay sampled in [cooldown_min, cooldown_max]
        self._t_last_pickup = 0.0
        self._t_last_walk_teleport = 0.0
        self._walk_teleport_next_cd = 0.0  # next Z delay sampled in [cooldown_min, cooldown_max]
        self._chase_active = False  # True only while flashing toward a monster
        self._attack_pause_until = 0.0
        self._face_direction_until = 0.0
        self._hold_up_until = 0.0
        self._t_last_face_tap = 0.0
        self._last_face_dir = None
        self._attack_count = 0            # 连续攻击计数
        self._force_walk_until = 0.0      # 连击后强制移动（朝向怪物）的截止时间
        # Buff skill
        buff_cfg = self.cfg.get("buff_skill", {})
        buff_keys = buff_cfg.get("keys", []) or []
        buff_cds = buff_cfg.get("cooldown", []) or []
        # 启动后先立刻轮询一轮 buff（用「已冷却完」初始化），之后再按各自冷却刷新
        now0 = time.time()
        self._t_last_buff_cast = []
        for i, _k in enumerate(buff_keys):
            cd = float(buff_cds[i]) if i < len(buff_cds) else 0.0
            self._t_last_buff_cast.append(now0 - cd)
        self._t_last_skill = 0.0
        self._buff_freeze_until = 0.0       # 放 buff 期间冻结（禁止移动）的截止时间
        self._alert_freeze_until = 0.0       # lie-detector freeze
        startup_delay = float(buff_cfg.get("startup_delay", 0.0))
        self._buff_ready_time = time.time() + startup_delay  # 首次 buff 允许时间（启动延迟）
        # 初始 buff 全部加完前，禁止移动/闪现/攻击（wait_before_combat）
        self._wait_before_combat = bool(buff_cfg.get("wait_before_combat", True)) and bool(buff_keys)
        self._initial_buff_pending = set(range(len(buff_keys))) if self._wait_before_combat else set()
        self._startup_combat_blocked = bool(self._initial_buff_pending)
        # Pet feed: tap-only, never freezes move/attack (no cast animation).
        self._t_last_pet_feed = 0.0
        self._pet_feed_ready_time = 0.0
        if buff_keys:
            logger.info(
                f"[Buff] 已加载 {len(buff_keys)} 个定时键 {buff_keys}，"
                f"{startup_delay:.1f}s 后开始按，冷却 {buff_cds}"
                + ("；加完后再开始闪现攻击" if self._startup_combat_blocked else "")
            )
        if not self.is_enable:
            # Idle until F1/F2: do not cast buffs / move until a profile starts
            self._startup_combat_blocked = False
            self._initial_buff_pending = set()
            logger.info("[KB] Idle on start — press F1 / F2 / F3 to begin")
        self._action_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._window_title = ""
        self._t_last_focus = 0.0
        if is_windows():
            token = str((cfg.get("game_window") or {}).get("title", "") or "")
            if token:
                self._window_title = get_game_window_title_by_token(token) or token

        self._pending_hp_potion = False
        self._pending_mp_potion = False

        # Force heal flag (set by HealthMonitor)
        self.is_need_force_heal = False

        # Current command state
        self.cmd_left_right = "none"  # "left", "right", "stop", "none"
        self._dir_override_token = 0  # bumped on F2 bound force flip
        self.cmd_up_down = "none"     # "up", "down", "stop", "none"
        self.cmd_action = "none"      # "jump", "teleport", "goal", "attack", "none", "add_hp", "add_mp"

        # Tracking to avoid redundant key_down/key_up
        self._last_left_right = "none"
        self._last_up_down = "none"
        self._last_action = "none"
        # Windows focus/one-shot keys often drop held arrows; re-assert periodically
        self._t_last_dir_reassert = 0.0
        # Was 0.25s — too frequent key_up/key_down made F2 feel stuttery.
        self._dir_reassert_interval = float(
            (cfg.get("directional_attack") or {}).get("dir_reassert_interval_s", 1.0)
        )
        self._focus_interval_s = 5.0  # do not Alt-nudge focus on every attack
        self._t_last_profile_toggle = 0.0
        self._profile_toggle_debounce = 0.45

        # FPS limit
        self._fps_limit = cfg["system"].get("fps_limit_main", 15)
        self._t_last = time.time()

        # Start daemon thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        state = "ENABLED" if self.is_enable else "IDLE (F1/F2/F3)"
        logger.info(f"[KeyBoardController] Started — {state}")

    def set_command(self, cmd_str):
        """Set movement command from string like 'left none jump'."""
        parts = cmd_str.strip().split()
        if len(parts) >= 3:
            self.cmd_left_right = parts[0]
            self.cmd_up_down = parts[1]
            self.cmd_action = parts[2]

    def _tap_direction(self, key: str, duration=0.04) -> bool:
        """Tap a direction briefly to turn without visibly walking."""
        with self._send_lock:
            if not key_down(key):
                return False
            time.sleep(duration)
            key_up(key)
        return True

    @staticmethod
    def _deep_update(dst, src):
        """In-place nested dict merge."""
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                KeyBoardController._deep_update(dst[k], v)
            else:
                dst[k] = v

    def set_detector(self, detector):
        """Wire MonsterDetector so profile switch can reload nameplate template."""
        self._detector = detector

    def apply_profile(self, profile_id: str) -> bool:
        """Merge profiles.<id> into live cfg and refresh buff/attack/name bindings."""
        profiles = self.cfg.get("profiles") or {}
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            logger.error(f"[KB] Unknown profile: {profile_id}")
            return False
        for key, val in profile.items():
            if key in ("label",):
                continue
            if isinstance(val, dict):
                if key not in self.cfg or not isinstance(self.cfg.get(key), dict):
                    self.cfg[key] = {}
                self._deep_update(self.cfg[key], val)
            else:
                self.cfg[key] = val

        key_cfg = self.cfg.get("key", {})
        self.key_jump = key_cfg.get("jump", self.key_jump)
        self.key_teleport = key_cfg.get("teleport", self.key_teleport)
        self.key_directional_attack = key_cfg.get("directional_attack", self.key_directional_attack)
        self.key_aoe_skill = key_cfg.get("aoe_skill", self.key_aoe_skill)
        self.key_pickup = key_cfg.get("pickup", self.key_pickup)
        self.attack_mode = self.cfg.get("bot", {}).get("attack", self.attack_mode)
        self._aoe_next_cd = 0.0
        self._walk_teleport_next_cd = 0.0
        self._t_last_attack = 0.0
        self._active_profile = profile_id
        da_live = self.cfg.get("directional_attack") or {}
        self._dir_reassert_interval = float(
            da_live.get("dir_reassert_interval_s", self._dir_reassert_interval)
        )

        if self._detector is not None and hasattr(self._detector, "reload_player_name"):
            try:
                self._detector.reload_player_name()
            except Exception as e:
                logger.warning(f"[KB] reload_player_name failed: {e}")
        if self._detector is not None and hasattr(self._detector, "reset_patrol_cycle"):
            try:
                self._detector.reset_patrol_cycle()
            except Exception:
                pass

        # Reload monster templates only when this profile enables detect.
        # F2 pre-monster: enable=false — skip luster_pixie load (assets stay on disk).
        md_prof = profile.get("monster_detect") if isinstance(profile.get("monster_detect"), dict) else {}
        md_want = bool((self.cfg.get("monster_detect") or {}).get("enable", False))
        if (
            self._detector is not None
            and md_want
            and hasattr(self._detector, "load_monster_images")
        ):
            try:
                self._detector.monsters_info = {}
                ok = self._detector.load_monster_images()
                logger.info(f"[KB] Reloaded monster templates ok={ok} "
                            f"keys={list(getattr(self._detector, 'monsters_info', {}).keys())}")
            except Exception as e:
                logger.warning(f"[KB] load_monster_images failed: {e}")
        elif self._detector is not None and not md_want:
            try:
                self._detector.monsters_info = {}
                self._detector.monsters = []
                logger.info("[KB] monster_detect off — cleared templates (no luster load)")
            except Exception:
                pass

        label = profile.get("label") or profile_id
        pn = (self.cfg.get("player_name") or {}).get("text", "?")
        buff = (self.cfg.get("buff_skill") or {}).get("keys", [])
        combo = (self.cfg.get("aoe_skill") or {}).get("attack_combo") or []
        gap = (self.cfg.get("aoe_skill") or {}).get("combo_gap", None)
        wt_on = bool((self.cfg.get("walk_teleport") or {}).get("enable", False))
        da = self.cfg.get("directional_attack") or {}
        lr = da.get("search_left_ratio")
        rr = da.get("search_right_ratio")
        prog = da.get("search_progress", "auto")
        er = da.get("search_edge_ratio", None)
        if lr is not None and rr is not None:
            zone = f"mid[{lr},{rr}]"
        elif str(prog).lower() in ("minimap", "yellow"):
            zone = f"yellow-edge(r={er})"
        elif str(prog).lower() in ("character", "screen", "name", "player"):
            zone = f"char-edge(r={er})"
        else:
            zone = "full-edge"
        md_on = bool((self.cfg.get("monster_detect") or {}).get("enable", False))
        mobs = (self.cfg.get("monster_detect") or {}).get("monsters", [])
        mid_zone = lr is not None and rr is not None
        steer = bool(da.get("search_mob_steer", False))
        station = bool(da.get("station_hold", False))
        aoe = self.cfg.get("aoe_skill") or {}
        cd_lo = aoe.get("cooldown_min", aoe.get("cooldown"))
        cd_hi = aoe.get("cooldown_max", aoe.get("cooldown"))
        logger.info(
            f"[KB] Profile '{profile_id}' ({label}) loaded | name={pn} | "
            f"buff={buff} | combo={combo or ['a']} gap={gap} | "
            f"atk={cd_lo}-{cd_hi}s | Z={'on' if wt_on else 'off'} | zone={zone} | "
            f"station={'on' if station else 'off'} | "
            f"mobs={'on' if md_on else 'off'}{mobs} steer={steer}"
        )
        # Dump live directional_attack keys so missing merge is obvious in logs.
        dump_keys = (
            "search_progress", "search_edge_ratio",
            "search_left_ratio", "search_right_ratio", "search_mob_steer",
            "search_mob_lead", "search_mob_sticky_frames", "search_dir_sticky_frames",
            "search_x_ema", "search_bottom_y_min_ratio", "search_edge_px",
            "search_edge_stall_frames", "search_stall_frames", "search_flip_frames",
        )
        logger.info(
            "[KB] directional_attack live | "
            + " ".join(f"{k}={da.get(k)!r}" for k in dump_keys)
        )
        # Fresh F2 session: allow immediate first mob steer / bound rehold.
        if self._detector is not None:
            try:
                self._detector._hunt_mob_steer_frame = 0
                self._detector._hunt_dir_change_frame = 0
                self._detector._hunt_smooth_x = None
                self._detector._hunt_progress_src = None
            except Exception:
                pass
            # F3: re-lock station center at current yellow-dot when (re)enabled
            try:
                if hasattr(self._detector, "reset_station_hold"):
                    self._detector.reset_station_hold()
            except Exception:
                pass
        return True

    def start_profile(self, profile_id: str):
        """F1/F2/F3 handler: same profile again toggles off; otherwise switch + enable."""
        now = time.time()
        # Prevent accidental double-tap (game/OS key repeat) from instantly disabling
        if (now - getattr(self, "_t_last_profile_toggle", 0.0)) < getattr(
            self, "_profile_toggle_debounce", 0.45
        ):
            logger.info(f"[KB] Profile {profile_id} tap ignored (debounce)")
            return
        self._t_last_profile_toggle = now
        if self.is_enable and self._active_profile == profile_id:
            self.disable()
            logger.info(f"[KB] Profile {profile_id} DISABLED")
            return
        # Switching profile / first start: release keys first
        if self.is_enable:
            self.disable()
        if not self.apply_profile(profile_id):
            return
        self.enable()
        logger.info(f"[KB] Profile {profile_id} ENABLED")

    def disable(self):
        """Stop AFK: clear cmds, abort combo, release every movement key.

        After this returns, the KB loop must NOT touch any keys so the user
        can walk freely. Do not re-call key_up/release_all from the idle path.
        """
        self.is_enable = False
        self._chase_active = False
        self.cmd_left_right = "none"
        self.cmd_up_down = "none"
        self.cmd_action = "none"
        self._buff_freeze_until = 0.0
        self._alert_freeze_until = 0.0
        self._attack_pause_until = 0.0
        self._force_walk_until = 0.0
        self._hold_up_until = 0.0
        self._face_direction_until = 0.0
        self._startup_combat_blocked = False
        self._initial_buff_pending = set()
        self.is_need_force_heal = False
        with self._action_lock:
            self._pending_hp_potion = False
            self._pending_mp_potion = False
        self._dir_override_token = int(getattr(self, "_dir_override_token", 0)) + 1
        self.release_all()
        # Extra hard release (pynput + pyautogui) — stuck holds after F2 stop
        for k in ("left", "right", "up", "down"):
            try:
                key_up(k)
            except Exception:
                pass
            try:
                pyautogui.keyUp(k)
            except Exception:
                pass
            try:
                _pynput_kb.release(getattr(Key, k, k))
            except Exception:
                pass
        self._last_left_right = "none"
        self._last_up_down = "none"
        self._last_action = "none"

    def enable(self):
        self.is_enable = True
        # Re-resolve HWND before focus — avoid Cursor/B站 title matches.
        try:
            self._refresh_window_title()
            cap = getattr(self._detector, "capture", None) if self._detector else None
            if cap is not None and hasattr(cap, "rebind_window_title"):
                cap.rebind_window_title()
                if getattr(cap, "window_title", ""):
                    self._window_title = cap.window_title
        except Exception:
            pass
        # Make sure keys go to the game, then ready for fresh holds
        try:
            self._ensure_game_focus(force=True)
        except Exception:
            pass
        self._invalidate_held_dirs()
        self._last_left_right = "none"
        # 开启挂机时，立刻准备再加一轮 buff
        buff_cfg = self.cfg.get("buff_skill", {})
        keys = buff_cfg.get("keys", []) or []
        cds = buff_cfg.get("cooldown", []) or []
        now = time.time()
        self._buff_ready_time = now + float(buff_cfg.get("startup_delay", 1.0))
        self._t_last_buff_cast = []
        for i, _k in enumerate(keys):
            cd = float(cds[i]) if i < len(cds) else 0.0
            self._t_last_buff_cast.append(now - cd)
        self._t_last_skill = 0.0
        self._wait_before_combat = bool(buff_cfg.get("wait_before_combat", True)) and bool(keys)
        self._initial_buff_pending = set(range(len(keys))) if self._wait_before_combat else set()
        self._startup_combat_blocked = bool(self._initial_buff_pending)
        self._walk_teleport_next_cd = 0.0
        self._aoe_next_cd = 0.0
        # Pet feed timer (profile may enable; no freeze)
        pet = self.cfg.get("pet_feed") or {}
        if bool(pet.get("enable", False)) and str(pet.get("key", "") or "").strip():
            interval = max(1.0, float(pet.get("interval_s", 900)))
            delay = max(0.0, float(pet.get("startup_delay_s", 2.0)))
            self._pet_feed_ready_time = now + delay
            # First tap soon after ready, then every interval_s.
            self._t_last_pet_feed = now + delay - interval
            logger.info(
                f"[Pet] feed '{pet.get('key')}' every {interval:.0f}s "
                f"(first in {delay:.1f}s, no freeze)"
                + (f" profile={self._active_profile}" if self._active_profile else "")
            )
        else:
            self._pet_feed_ready_time = 0.0
            self._t_last_pet_feed = 0.0
        if self._startup_combat_blocked:
            logger.info(
                f"[Buff] 开局先加 buff {keys}，再开始闪现攻击"
                + (f"（profile={self._active_profile}）" if self._active_profile else "")
            )

    def toggle(self):
        """Legacy toggle: if idle, start f1; if running, stop."""
        if self.is_enable:
            self.disable()
            logger.info("[KB] DISABLED")
        else:
            if (self.cfg.get("profiles") or {}).get("f1"):
                self.start_profile("f1")
            else:
                self.enable()
                logger.info("[KB] ENABLED")

    def release_all(self):
        """Release all held movement keys."""
        for key in ["left", "right"]:
            key_up(key)
        for key in ["up", "down"]:
            try:
                _pynput_kb.release(getattr(Key, key, key))
            except Exception:
                pass
        self._last_left_right = "none"
        self._last_up_down = "none"
        self._last_action = "none"
        self._hold_up_until = 0.0

    def stop(self):
        """Stop the controller and release all keys."""
        self.is_terminated = True
        self.release_all()
        logger.info("[KeyBoardController] Stopped")

    def queue_hp_potion(self):
        """Queue one HP potion action (consumed by keyboard thread)."""
        with self._action_lock:
            self._pending_hp_potion = True
        return True

    def queue_mp_potion(self):
        """Queue one MP potion action (consumed by keyboard thread)."""
        with self._action_lock:
            self._pending_mp_potion = True
        return True

    def _ensure_game_focus(self, force: bool = False) -> None:
        """Keep game window focused so simulated keys land in-game, not on NewMaple/cmd.

        Throttled: calling SetForeground/Alt-nudge on every S/A tap made the
        client stutter after F2. Only force-focus on enable or every few seconds.
        """
        if not is_windows() or not self._window_title:
            return
        now = time.time()
        interval = float(getattr(self, "_focus_interval_s", 5.0))
        if (not force) and (now - float(getattr(self, "_t_last_focus", 0.0)) < interval):
            return
        activate_game_window(self._window_title)
        self._t_last_focus = now

    def _invalidate_held_dirs(self):
        """Force next _apply_movement to re-press left/right/up/down.

        One-shot taps and focus restores (Alt nudge / SetForegroundWindow) often
        drop physically held arrow keys on Windows while our tracker still thinks
        they are held — character then stands still forever.

        Also physically release Left/Right here so a stuck opposite arrow after
        an F2 bound flip (left wall forever) cannot survive until the next apply.
        """
        try:
            key_up("left")
            key_up("right")
        except Exception:
            pass
        self._last_left_right = "none"
        self._last_up_down = "none"
        self._t_last_dir_reassert = 0.0

    def is_buff_frozen(self) -> bool:
        """True while casting buff or waiting for initial buffs — no move/attack."""
        if getattr(self, "_startup_combat_blocked", False):
            return True
        return time.time() < float(getattr(self, "_buff_freeze_until", 0.0))


    def pause_for_lie_alert(self, seconds: float = 3.0):
        """Release keys and freeze AFK actions while lie-detector dialog is on screen."""
        until = time.time() + max(0.5, float(seconds))
        self._alert_freeze_until = max(float(getattr(self, "_alert_freeze_until", 0.0)), until)
        self._buff_freeze_until = max(float(getattr(self, "_buff_freeze_until", 0.0)), until)
        self._attack_pause_until = max(float(getattr(self, "_attack_pause_until", 0.0)), until)
        self.cmd_left_right = "none"
        self.cmd_up_down = "none"
        self.cmd_action = "none"
        self._chase_active = False
        try:
            self.release_all()
        except Exception:
            pass

    def is_lie_alert_frozen(self) -> bool:
        return time.time() < float(getattr(self, "_alert_freeze_until", 0.0))

    def clear_lie_alert_freeze(self):
        """End AFK freeze from lie-detector so hang-up can resume immediately."""
        now = time.time()
        self._alert_freeze_until = 0.0
        # pause_for_lie_alert also bumps buff/attack freezes — drop those too.
        self._buff_freeze_until = min(
            float(getattr(self, "_buff_freeze_until", 0.0)), now
        )
        self._attack_pause_until = min(
            float(getattr(self, "_attack_pause_until", 0.0)), now
        )

    def request_dir_override(self, direction: str):
        """Instant F2 bound flip: set cmd, bump token (aborts combo), re-hold now.

        Detector calls this the same frame yellow hits the 10%/90% band so we
        do not wait for the current S->A combo sleep to finish (~0.2s+) or for
        sticky/invalidate-every-6-frames to catch up.

        During buff freeze / initial buff wait: only update cmd — never physically
        press arrows (would cancel the buff cast).
        """
        if direction not in ("left", "right"):
            return
        # AFK stopped (F1/F2 toggle off): never re-press arrows.
        if not getattr(self, "is_enable", False):
            try:
                key_up("left")
                key_up("right")
            except Exception:
                pass
            self._last_left_right = "none"
            return
        self.cmd_left_right = direction
        self._dir_override_token = int(getattr(self, "_dir_override_token", 0)) + 1
        # Do not re-hold movement while buff is casting or startup buffs pending.
        if getattr(self, "_startup_combat_blocked", False) or time.time() < float(
            getattr(self, "_buff_freeze_until", 0.0)
        ):
            try:
                key_up("left")
                key_up("right")
            except Exception:
                pass
            self._last_left_right = "none"
            return
        try:
            key_up("left")
            key_up("right")
        except Exception:
            pass
        try:
            key_down(direction)
        except Exception:
            pass
        self._last_left_right = direction
        self._last_face_dir = "face_left" if direction == "left" else "face_right"
        self._t_last_dir_reassert = time.time()
        self._face_direction_until = time.time() + 0.05

    def _press_key_serialized(self, key: str, duration=0.05, rehold_dirs: bool = True) -> bool:
        """Serialize one-shot key sends across keyboard and health-monitor threads.

        rehold_dirs=False for buff casts: do not schedule arrow re-hold (movement
        stays released for freeze_seconds so the cast is not cancelled).
        """
        # AFK off: never send any one-shot keys (attack/buff/teleport/potion).
        if not getattr(self, "is_enable", False):
            return False
        with self._send_lock:
            # Soft check only — never Alt-nudge on the attack hot path.
            self._ensure_game_focus(force=False)
            ok = press_key(key, duration)
            if rehold_dirs:
                # Re-hold directions after combat one-shots (S / A / Z / potions)
                self._invalidate_held_dirs()
            else:
                # Buff: keep arrows physically up
                try:
                    key_up("left")
                    key_up("right")
                except Exception:
                    pass
                self._last_left_right = "none"
                self._last_up_down = "none"
            return ok

    def press_hp_potion(self) -> bool:
        """Immediately send the HP potion key and pause attack briefly."""
        if not getattr(self, "is_enable", False):
            return False
        ok = self._press_key_serialized(self.cfg["key"].get("add_hp", "1"), 0.05)
        if ok:
            self._attack_pause_until = max(self._attack_pause_until, time.time() + 0.12)
            self._last_action = "add_hp"
        return ok

    def press_mp_potion(self) -> bool:
        """Immediately send the MP potion key and pause attack briefly."""
        if not getattr(self, "is_enable", False):
            return False
        ok = self._press_key_serialized(self.cfg["key"].get("add_mp", "2"), 0.05)
        if ok:
            self._attack_pause_until = max(self._attack_pause_until, time.time() + 0.12)
            self._last_action = "add_mp"
        return ok


    def _refresh_window_title(self) -> str:
        """Re-resolve game HWND title (skip Cursor/B站 etc.)."""
        token = str((self.cfg.get("game_window") or {}).get("title", "") or "")
        if not token:
            return self._window_title or ""
        title = get_game_window_title_by_token(token) or ""
        if title:
            if title != self._window_title:
                logger.info(f"[KB] game window → '{title}'")
            self._window_title = title
        return self._window_title or token

    def _poll_profile_hotkeys(self):
        """Fallback F1/F2/F3 via GetAsyncKeyState when game eats pynput events."""
        if not is_windows():
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # VK_F1=0x70, VK_F2=0x71, VK_F3=0x72; high bit set => currently down
            f1_down = bool(user32.GetAsyncKeyState(0x70) & 0x8000)
            f2_down = bool(user32.GetAsyncKeyState(0x71) & 0x8000)
            f3_down = bool(user32.GetAsyncKeyState(0x72) & 0x8000)
        except Exception:
            return
        # First sample only primes state — a key already held at boot must not
        # look like a rising edge (was auto-starting F2 on launch).
        if not bool(getattr(self, "_poll_profile_primed", False)):
            self._poll_f1_down = f1_down
            self._poll_f2_down = f2_down
            self._poll_f3_down = f3_down
            self._poll_profile_primed = True
            return
        prev1 = bool(getattr(self, "_poll_f1_down", False))
        prev2 = bool(getattr(self, "_poll_f2_down", False))
        prev3 = bool(getattr(self, "_poll_f3_down", False))
        self._poll_f1_down = f1_down
        self._poll_f2_down = f2_down
        self._poll_f3_down = f3_down
        # Rising edge only
        if f1_down and not prev1:
            try:
                self.start_profile("f1")
            except Exception:
                pass
        if f2_down and not prev2:
            try:
                self.start_profile("f2")
            except Exception:
                pass
        if f3_down and not prev3:
            try:
                if (self.cfg.get("profiles") or {}).get("f3"):
                    self.start_profile("f3")
            except Exception:
                pass

    def _poll_manual_turn(self):
        """While AFK: user Left/Right tap instantly flips patrol direction.

        Only the opposite of current hunt dir counts as a manual flip, so our own
        held arrow (same dir) does not false-trigger via GetAsyncKeyState.
        """
        if not self.is_enable or not is_windows():
            return
        # Don't steal keys during buff cast
        if self.is_buff_frozen():
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            left_down = bool(user32.GetAsyncKeyState(0x25) & 0x8000)   # VK_LEFT
            right_down = bool(user32.GetAsyncKeyState(0x27) & 0x8000)  # VK_RIGHT
        except Exception:
            return
        prev_l = bool(getattr(self, "_poll_left_down", False))
        prev_r = bool(getattr(self, "_poll_right_down", False))
        self._poll_left_down = left_down
        self._poll_right_down = right_down
        # Current intended patrol dir
        cur = None
        det = getattr(self, "_detector", None)
        if det is not None:
            cur = getattr(det, "_hunt_search_dir", None)
        if cur not in ("left", "right"):
            cur = self.cmd_left_right if self.cmd_left_right in ("left", "right") else None
        if cur not in ("left", "right"):
            cur = self._last_left_right if self._last_left_right in ("left", "right") else "right"
        # Debounce manual flips
        now = time.time()
        if now - float(getattr(self, "_t_last_manual_flip", 0.0)) < 0.25:
            return
        want = None
        if left_down and not prev_l and cur == "right":
            want = "left"
        elif right_down and not prev_r and cur == "left":
            want = "right"
        if want is None:
            return
        self._t_last_manual_flip = now
        self.force_patrol_dir(want, reason="manual_key")

    def force_patrol_dir(self, direction: str, reason: str = "manual"):
        """Force patrol + physical hold to left/right (user tap or bound flip)."""
        if direction not in ("left", "right"):
            return
        if not getattr(self, "is_enable", False):
            return
        det = getattr(self, "_detector", None)
        if det is not None:
            try:
                prev = getattr(det, "_hunt_search_dir", None)
                det._hunt_search_dir = direction
                det._hunt_dir_forced = True
                det._hunt_dir_change_frame = int(getattr(det, "frame_count", 0))
                det._hunt_search_last_reason = reason
                # Reset edge-stall so we don't immediately flip back
                det._hunt_edge_stall_frames = 0
            except Exception:
                prev = None
        else:
            prev = None
        self.request_dir_override(direction)
        logger.info(f"[KB] Manual flip {prev} -> {direction} ({reason})")

    def _run(self):
        """Daemon loop — applies left/right, up/down, action, pickup, and buff commands."""
        while not self.is_terminated:
            # Always poll F1/F2 (game focus often swallows pynput)
            try:
                self._poll_profile_hotkeys()
            except Exception:
                pass
            if self.is_enable:
                if self.is_lie_alert_frozen():
                    # lie-detector dialog: stop all AFK keys so user can answer manually
                    try:
                        if self._last_left_right != "none" or self._last_up_down != "none":
                            self.release_all()
                    except Exception:
                        pass
                else:
                    try:
                        self._poll_manual_turn()
                    except Exception:
                        pass
                    self._apply_movement()
                    self._apply_action()
                    self._apply_pickup()
                    self._apply_walk_teleport()
                    self._apply_buff()
                    self._apply_pet_feed()
            else:
                # Idle (F2/F1 off): do NOTHING to keys.
                # Continuous key_up/release_all here was fighting the user's held
                # arrows and made the character unable to move after toggle-off.
                # disable() already did a one-shot hard release.
                pass
            self._limit_fps()

    def _apply_pet_feed(self):
        """Tap pet-feed key on an interval. Never freezes move/attack."""
        if self.cfg.get("bot", {}).get("mode", "detect") != "hunt":
            return
        if self.is_buff_frozen() or self.is_lie_alert_frozen():
            return
        pet = self.cfg.get("pet_feed") or {}
        if not bool(pet.get("enable", False)):
            return
        key = str(pet.get("key", "") or "").strip()
        if not key:
            return
        now = time.time()
        ready = float(getattr(self, "_pet_feed_ready_time", 0.0) or 0.0)
        if ready > 0.0 and now < ready:
            return
        interval = max(1.0, float(pet.get("interval_s", 900)))
        last = float(getattr(self, "_t_last_pet_feed", 0.0) or 0.0)
        if last > 0.0 and (now - last) < interval:
            return
        hold = max(0.05, float(pet.get("press_duration", 0.08)))
        ok = self._press_key_serialized(key, hold, rehold_dirs=True)
        self._t_last_pet_feed = now
        logger.info(f"[Pet] feed '{key}' ok={ok} (every {interval:.0f}s, no freeze)")

    def _apply_buff(self):
        """自动加 buff：仅在 hunt 模式，按各 buff 键的冷却时间按下。"""
        if self.cfg.get("bot", {}).get("mode", "detect") != "hunt":
            return
        # 启动延迟：脚本启动后 startup_delay 秒内不执行 buff。
        if time.time() < self._buff_ready_time:
            return
        buff_cfg = self.cfg.get("buff_skill", {})
        keys = buff_cfg.get("keys", []) or []
        cooldowns = buff_cfg.get("cooldown", []) or []
        action_cd = float(buff_cfg.get("action_cooldown", 1.0))
        freeze_seconds = float(buff_cfg.get("freeze_seconds", 1.5))
        press_duration = float(buff_cfg.get("press_duration", 0.12))
        if not keys:
            return
        # 长度变化时重建计时器
        if len(self._t_last_buff_cast) != len(keys):
            now = time.time()
            self._t_last_buff_cast = [
                now - float(cooldowns[i] if i < len(cooldowns) else 0.0) for i in range(len(keys))
            ]
        t_now = time.time()
        for i, k in enumerate(keys):
            cd = float(cooldowns[i]) if i < len(cooldowns) else 0.0
            last = self._t_last_buff_cast[i] if i < len(self._t_last_buff_cast) else 0.0
            if t_now - last >= cd and t_now - self._t_last_skill >= action_cd:
                # 先松手并稍停，避免方向键/闪现吞掉 buff（尤其 shift/end）
                self.release_all()
                self.cmd_left_right = "none"
                self.cmd_up_down = "none"
                self.cmd_action = "none"
                self._chase_active = False
                # 施放前就冻结：挡住 detector 换向/攻击（含 request_dir_override）
                pre_freeze = max(freeze_seconds, hold_guess := (
                    max(press_duration, float(buff_cfg.get("special_press_duration", 0.28)))
                    + 0.2
                ))
                self._attack_pause_until = time.time() + pre_freeze
                self._buff_freeze_until = time.time() + pre_freeze
                time.sleep(0.15)
                key_name = str(k).strip().lower()
                hold = press_duration
                if key_name in ("shift", "end", "home", "ctrl", "alt", "pagedown", "page_down", "page down", "pgdn"):
                    hold = max(hold, float(buff_cfg.get("special_press_duration", 0.28)))
                ok = self._press_key_serialized(str(k), hold, rehold_dirs=False)
                # 施放后立刻再空一拍，保证下个 buff 前间隔够
                time.sleep(0.05)
                self.release_all()
                self.cmd_left_right = "none"
                self.cmd_up_down = "none"
                self.cmd_action = "none"
                self._t_last_buff_cast[i] = t_now
                self._t_last_skill = t_now
                # 放 buff 期间停止攻击 + 移动，人物冻结 freeze_seconds 秒（从按完起算）。
                self._attack_pause_until = time.time() + freeze_seconds
                self._buff_freeze_until = time.time() + freeze_seconds
                logger.info(
                    f"[Buff] 按下 '{k}' ok={ok}（冷却 {cd}s，按住 {hold}s，冻结 {freeze_seconds}s）"
                )
                # 初始轮：记一下已按过的 buff；全部加完后放开闪现/攻击
                if self._startup_combat_blocked and i in self._initial_buff_pending:
                    self._initial_buff_pending.discard(i)
                    if not self._initial_buff_pending:
                        # 等本技能冻结结束再开战，避免立刻 Z/A 打断末尾 buff
                        self._startup_combat_blocked = False
                        self._invalidate_held_dirs()
                        self._t_last_attack = 0.0
                        self._aoe_next_cd = 0.0
                        # 开战后尽快闪第一次，之后再按 ~5s 节奏
                        self._t_last_walk_teleport = 0.0
                        self._walk_teleport_next_cd = 0.0
                        logger.info("[Buff] 初始 buff 全部加完，开始闪现巡逻攻击")
                break

    def _apply_pickup(self):
        """Tap the pickup key periodically while the character is moving（仅 hunt 模式）。"""
        if self.cfg.get("bot", {}).get("mode", "detect") != "hunt":
            return
        pickup_cfg = self.cfg.get("pickup", {})
        if not pickup_cfg.get("enable", True):
            return
        moving = (
            self.cmd_left_right in ("left", "right")
            or self.cmd_up_down in ("up", "down")
        )
        if not moving:
            return
        pickup_cd = float(pickup_cfg.get("cooldown", 0.3))
        t_now = time.time()
        if t_now - self._t_last_pickup >= pickup_cd:
            self._press_key_serialized(self.key_pickup, 0.03)
            self._t_last_pickup = t_now
    def _apply_walk_teleport(self):
        """Flash (Z) while left/right moving when chase_active or only_when_chasing=false."""
        if self.cfg.get("bot", {}).get("mode", "detect") != "hunt":
            return
        if getattr(self, "_startup_combat_blocked", False):
            return
        wt = self.cfg.get("walk_teleport", {})
        if not wt.get("enable", False):
            return
        # Default: only flash when detector says we are chasing a monster
        if wt.get("only_when_chasing", True) and not self._chase_active:
            return
        if self.cmd_left_right not in ("left", "right"):
            return
        if time.time() < self._buff_freeze_until:
            return
        cd_min = float(wt.get("cooldown_min", wt.get("cooldown", 5.0)))
        cd_max = float(wt.get("cooldown_max", wt.get("cooldown", 5.0)))
        if cd_max < cd_min:
            cd_min, cd_max = cd_max, cd_min
        if self._walk_teleport_next_cd <= 0:
            self._walk_teleport_next_cd = random.uniform(cd_min, cd_max) if cd_max > cd_min else cd_min
        t_now = time.time()
        if t_now - self._t_last_walk_teleport >= self._walk_teleport_next_cd:
            self._press_key_serialized(self.key_teleport, 0.04)
            self._t_last_walk_teleport = t_now
            self._walk_teleport_next_cd = random.uniform(cd_min, cd_max) if cd_max > cd_min else cd_min


    def _apply_movement(self):
        """Apply left/right and up/down key state. Up/down uses pynput for game compatibility."""
        # 初始 buff 未加完：站着别动，等 buff 轮完再巡逻。
        if getattr(self, "_startup_combat_blocked", False):
            if self._last_left_right in ("left", "right"):
                key_up(self._last_left_right)
            if self._last_up_down in ("up", "down"):
                _pynput_kb.release(getattr(Key, self._last_up_down, self._last_up_down))
            self._last_left_right = "none"
            self._last_up_down = "none"
            return
        # 放 buff 冻结期间：释放所有方向键，禁止移动。
        if time.time() < self._buff_freeze_until:
            if self._last_left_right in ("left", "right"):
                key_up(self._last_left_right)
            if self._last_up_down in ("up", "down"):
                _pynput_kb.release(getattr(Key, self._last_up_down, self._last_up_down))
            self._last_left_right = "none"
            self._last_up_down = "none"
            return

        # Left/Right: support both walking (hold) and face-only taps.
        lr = self.cmd_left_right
        # 连击后强制移动：把「转身」临时变成「走」，朝怪物方向走一小段重新面向。
        if time.time() < self._force_walk_until and lr in ("face_left", "face_right"):
            lr = "left" if lr == "face_left" else "right"
        if lr in ("face_left", "face_right"):
            # Face commands: tap only when the facing direction changes,
            # so repeated face commands don't cause the character to creep.
            if self._last_left_right in ("left", "right"):
                key_up(self._last_left_right)

            if lr != self._last_face_dir:
                tap_duration = float(self.cfg.get("directional_attack", {}).get("face_tap_duration", 0.07))
                key = "left" if lr == "face_left" else "right"
                if self._tap_direction(key, duration=max(0.03, tap_duration)):
                    self._last_face_dir = lr
                    # 转身后多停一会儿再攻击，确保转身（可能被攻击动画吞掉）先完成。
                    self._face_direction_until = time.time() + tap_duration + 0.3
                    if time.time() - getattr(self, '_last_face_log', 0) >= 2.0:
                        logger.info(f"[KB] 转身 {lr} → 按 {key} 键 ({tap_duration:.2f}s)")
                        self._last_face_log = time.time()
            self._last_left_right = lr
        elif lr != self._last_left_right:
            # Always drop BOTH arrows on a turn so a stuck Left-into-wall
            # cannot leave Right unheld after F2 bound flip.
            key_up("left")
            key_up("right")
            if lr in ("left", "right"):
                key_down(lr)
                self._face_direction_until = time.time() + 0.09
                self._t_last_dir_reassert = time.time()
                # Walking turns the character: keep the face-direction cache
                # in sync so the next face_* command actually taps to turn.
                self._last_face_dir = "face_left" if lr == "left" else "face_right"
            elif lr == "stop":
                pass  # both already released above
            self._last_left_right = lr
        elif lr in ("left", "right"):
            # Same direction: periodically re-press in case OS dropped the hold.
            # Always release the opposite arrow first — a stuck Left after an
            # F2 left-bound flip-to-right leaves the char walking into the wall.
            now_ra = time.time()
            if now_ra - getattr(self, "_t_last_dir_reassert", 0.0) >= getattr(
                self, "_dir_reassert_interval", 0.25
            ):
                opp = "left" if lr == "right" else "right"
                try:
                    key_up(opp)
                except Exception:
                    pass
                key_down(lr)
                self._t_last_dir_reassert = now_ra

        # Up/Down (pynput — same as jump, better game compatibility)
        ud = self.cmd_up_down
        # 抓绳：跳跃后短时间内强制按住 up（除非正往下）。
        if time.time() < self._hold_up_until and ud != "down":
            ud = "up"
        if ud != self._last_up_down:
            if self._last_up_down in ("up", "down"):
                _pynput_kb.release(getattr(Key, self._last_up_down, self._last_up_down))
            if ud in ("up", "down"):
                _pynput_kb.press(getattr(Key, ud, ud))
            elif ud == "stop":
                _pynput_kb.release(Key.up)
                _pynput_kb.release(Key.down)
            self._last_up_down = ud

    def _jc_sleep(self, seconds: float, token0: int) -> bool:
        """Sleep in slices; return False if AFK stopped or direction flipped."""
        end = time.time() + max(0.0, float(seconds))
        while time.time() < end:
            try:
                self._poll_profile_hotkeys()
            except Exception:
                pass
            if not getattr(self, "is_enable", False):
                return False
            if self.is_lie_alert_frozen() or self.is_buff_frozen():
                return False
            if int(getattr(self, "_dir_override_token", 0)) != token0:
                return False
            time.sleep(min(0.04, max(0.0, end - time.time())))
        return True

    def _jc_rehold_dir(self):
        lr = self.cmd_left_right
        if lr in ("left", "right"):
            try:
                key_down(lr)
            except Exception:
                pass
            self._last_left_right = lr
            self._t_last_dir_reassert = time.time()

    def _execute_air_jump_attack(self, jump_key: str, skill_key: str, aoe_cfg: dict) -> bool:
        """
        Air jump timing: hold jump → air delay → hold skill → release jump
        → release gap → release skill. Direction arrow stays held.
        """
        air = max(0.0, float(aoe_cfg.get("jump_air_delay", 0.08)))
        skill_hold = max(0.02, float(aoe_cfg.get("skill_hold", 0.12)))
        rel_gap = max(0.0, float(aoe_cfg.get("skill_release_gap", 0.12)))
        token0 = int(getattr(self, "_dir_override_token", 0))

        def _kd(k: str):
            with self._send_lock:
                self._ensure_game_focus(force=False)
                key_down(k)

        def _ku(k: str):
            with self._send_lock:
                try:
                    key_up(k)
                except Exception:
                    pass

        try:
            _kd(jump_key)
            if not self._jc_sleep(air, token0):
                _ku(jump_key)
                self._jc_rehold_dir()
                return False
            _kd(skill_key)
            if not self._jc_sleep(skill_hold, token0):
                _ku(jump_key)
                _ku(skill_key)
                self._jc_rehold_dir()
                return False
            _ku(jump_key)
            if not self._jc_sleep(rel_gap, token0):
                _ku(skill_key)
                self._jc_rehold_dir()
                return False
            _ku(skill_key)
            self._jc_rehold_dir()
        except Exception:
            try:
                _ku(jump_key)
                _ku(skill_key)
            except Exception:
                pass
            self._jc_rehold_dir()
            return False
        return True

    def _do_jump_attack(self, aoe_cfg: dict) -> float:
        """One air-jump attack; returns next cooldown seconds."""
        combo = aoe_cfg.get("attack_combo") or ["s", "a"]
        jump_key = str(aoe_cfg.get("jump_key") or (combo[0] if combo else self.key_jump))
        skill_key = str(
            aoe_cfg.get("skill_key")
            or (combo[1] if len(combo) > 1 else self.key_aoe_skill)
        )
        ok = self._execute_air_jump_attack(jump_key, skill_key, aoe_cfg)
        cd_lo = float(aoe_cfg.get("jump_interval_min", aoe_cfg.get("cooldown_min", 0.35)))
        cd_hi = float(aoe_cfg.get("jump_interval_max", aoe_cfg.get("cooldown_max", 0.50)))
        if cd_hi < cd_lo:
            cd_lo, cd_hi = cd_hi, cd_lo
        if not ok:
            return max(0.15, cd_lo * 0.5)
        return random.uniform(cd_lo, cd_hi)

    def _apply_action(self):
        """Apply one-shot actions only once per command change."""
        # Force heal overrides normal action
        if self.is_need_force_heal:
            self.press_hp_potion()
            return

        # Potions are high priority and are serialized in this same thread.
        pending_hp = False
        pending_mp = False
        with self._action_lock:
            if self._pending_hp_potion:
                pending_hp = True
                self._pending_hp_potion = False
            elif self._pending_mp_potion:
                pending_mp = True
                self._pending_mp_potion = False

        if pending_hp:
            self.press_hp_potion()
            return
        if pending_mp:
            self.press_mp_potion()
            return

        # Attack is level-triggered: keep firing on cooldown while command stays "attack"/"aoe".
        if self.cmd_action in ("attack", "aoe"):
            if getattr(self, "_startup_combat_blocked", False):
                return
            if time.time() < self._attack_pause_until:
                return
            if time.time() < self._face_direction_until:
                return
            if self.cmd_action == "aoe" or self.attack_mode == "aoe_skill":
                atk_key = self.key_aoe_skill
                aoe_cfg = self.cfg.get("aoe_skill", {}) or {}
                style = str(aoe_cfg.get("attack_style", "") or "").strip().lower()
                cd_min = float(aoe_cfg.get("cooldown_min", aoe_cfg.get("cooldown", 0.5)))
                cd_max = float(aoe_cfg.get("cooldown_max", aoe_cfg.get("cooldown", 1.5)))
                if cd_max < cd_min:
                    cd_min, cd_max = cd_max, cd_min
                # First press or after reset: sample immediately usable delay
                if self._aoe_next_cd <= 0:
                    self._aoe_next_cd = random.uniform(cd_min, cd_max)
                atk_cd = self._aoe_next_cd
            else:
                atk_key = self.key_directional_attack
                aoe_cfg = {}
                style = ""
                atk_cd = self.cfg.get("directional_attack", {}).get("cooldown", 0.9)

            t_now = time.time()
            if t_now - self._t_last_attack >= float(atk_cd):
                aoe_cfg = self.cfg.get("aoe_skill", {}) or {}
                style = str(aoe_cfg.get("attack_style", "") or "").strip().lower()
                if style in ("jump_cruise", "jump_attack", "cruise"):
                    next_cd = self._do_jump_attack(aoe_cfg)
                    self._t_last_attack = time.time()
                    self._aoe_next_cd = float(next_cd)
                    self._attack_count += 1
                    return

                # Optional combo e.g. ["s","a"] with combo_gap between keys (legacy F2)
                combo = aoe_cfg.get("attack_combo") or []
                if isinstance(combo, (list, tuple)) and len(combo) > 0:
                    gap = float(aoe_cfg.get("combo_gap", 0.2))
                    hold = float(aoe_cfg.get("combo_press_duration", 0.05))  # S/A both short tap
                    token0 = int(getattr(self, "_dir_override_token", 0))
                    for i, ck in enumerate(combo):
                        # Bound flip mid-combo: abort remaining taps, keep new dir held.
                        if int(getattr(self, "_dir_override_token", 0)) != token0:
                            break
                        self._press_key_serialized(str(ck), hold)
                        if int(getattr(self, "_dir_override_token", 0)) != token0:
                            # request_dir_override already re-held the new arrow
                            break
                        # Jump/attack taps drop arrow holds on Windows; re-hold immediately
                        # so F2 does not drift left between S and A.
                        if not getattr(self, "is_enable", False):
                            break
                        lr_now = self.cmd_left_right
                        if lr_now in ("left", "right"):
                            key_down(lr_now)
                            self._last_left_right = lr_now
                            self._t_last_dir_reassert = time.time()
                        if i < len(combo) - 1 and gap > 0:
                            # Slice sleep so a bound flip / F2-off can abort within ~50ms
                            end = time.time() + gap
                            while time.time() < end:
                                try:
                                    self._poll_profile_hotkeys()
                                except Exception:
                                    pass
                                if not getattr(self, "is_enable", False):
                                    break
                                if int(getattr(self, "_dir_override_token", 0)) != token0:
                                    break
                                time.sleep(min(0.05, max(0.0, end - time.time())))
                else:
                    self._press_key_serialized(atk_key, 0.05)
                self._t_last_attack = t_now
                # Resample next attack delay in [cooldown_min, cooldown_max]
                if self.cmd_action == "aoe" or self.attack_mode == "aoe_skill":
                    aoe_cfg = self.cfg.get("aoe_skill", {}) or {}
                    cd_min = float(aoe_cfg.get("cooldown_min", 0.5))
                    cd_max = float(aoe_cfg.get("cooldown_max", 1.5))
                    if cd_max < cd_min:
                        cd_min, cd_max = cd_max, cd_min
                    self._aoe_next_cd = random.uniform(cd_min, cd_max)
                self._attack_count += 1
                if self.attack_mode != "aoe_skill":
                    interval = int(self.cfg.get("directional_attack", {}).get("attack_move_interval", 3))
                    if self._attack_count >= interval:
                        move_seconds = float(self.cfg.get("directional_attack", {}).get("attack_move_seconds", 0.3))
                        self._force_walk_until = time.time() + move_seconds
                        self._attack_count = 0
            return

        if self.cmd_action != self._last_action:
            self._last_action = self.cmd_action
            if self.cmd_action == "jump":
                time.sleep(0.05)
                self._press_key_serialized(self.key_jump)
                # 抓绳优化：按跳的同时按住 up 一段时间（向下跳除外）。
                if self.cmd_up_down != "down" and self.jump_hold_up_seconds > 0:
                    self._hold_up_until = time.time() + self.jump_hold_up_seconds
            elif self.cmd_action == "teleport":
                time.sleep(0.05)
                self._press_key_serialized(self.key_teleport)
            elif self.cmd_action == "attack_once":
                # One-shot single attack (ignores cooldown), then reset.
                self._press_key_serialized(self.key_directional_attack, 0.05)
                self.cmd_action = "none"
            elif self.cmd_action == "add_hp":
                self.press_hp_potion()
                self.cmd_action = "none"
            elif self.cmd_action == "add_mp":
                self.press_mp_potion()
                self.cmd_action = "none"

    def _limit_fps(self):
        """Cap the loop rate."""
        elapsed = time.time() - self._t_last
        target = 1.0 / self._fps_limit
        if elapsed < target:
            time.sleep(target - elapsed)
        self._t_last = time.time()
