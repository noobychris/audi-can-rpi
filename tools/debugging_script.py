#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined Audi test GUI:
- FIS display emulator (listens to vcan0 IDs 0x265 / 0x267 / 0x667 / 0x66B)
- RNS-E button image GUI (sends 0x461 frames)
- cansend preset/control GUI

Expected files in the same folder as this script:
- fis.png
- rns-e.png
"""

import os
import sys
import sysconfig
import subprocess
import shutil
import importlib
import binascii
import datetime
import time
import threading
from typing import Optional, List

# =============================================================================
# Bootstrap / packages
# =============================================================================

REQUIRED_PY = (3, 7)
VENV_PATH = os.path.expanduser("~/.venv-canbus")
VENV_PY = os.path.join(VENV_PATH, "bin", "python")


def _ensure_python_version():
    if sys.version_info < REQUIRED_PY:
        print(
            "Python {}.{}+ required. Current: {}.{}".format(
                REQUIRED_PY[0], REQUIRED_PY[1],
                sys.version_info[0], sys.version_info[1]
            )
        )
        sys.exit(1)


def _ensure_running_in_venv():
    if sys.executable != VENV_PY:
        os.environ["PYTHONNOUSERSITE"] = "1"
        print("Restarting inside virtual environment: {}".format(VENV_PY))
        os.execv(VENV_PY, [VENV_PY] + sys.argv)


def _prioritize_venv_site_packages(remove_system_site=True):
    paths = sysconfig.get_paths()
    venv_site = paths.get("purelib")
    venv_plat = paths.get("platlib") or venv_site

    for p in (venv_plat, venv_site):
        if p and p in sys.path:
            sys.path.remove(p)
        if p:
            sys.path.insert(0, p)

    if not remove_system_site:
        return

    new_path = []
    for p in sys.path:
        if (("site-packages" in p) or ("dist-packages" in p)) and p.startswith(
            ("/usr/lib/python3", "/usr/local/lib/python3")
        ):
            continue
        new_path.append(p)
    sys.path[:] = new_path


def ensure_pkg(mod_name: str, pip_name: Optional[str] = None, version_spec: Optional[str] = None):
    try:
        importlib.import_module(mod_name)
        return
    except ImportError:
        pass

    pkg = pip_name or mod_name
    req = "{}{}".format(pkg, version_spec or "")
    print("Installing {} into venv ...".format(req))
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--ignore-installed",
        "--no-user",
        req,
    ])
    importlib.invalidate_caches()
    importlib.import_module(mod_name)


def ensure_tkinter_system() -> bool:
    try:
        import tkinter  # noqa
        return True
    except Exception:
        print("tkinter fehlt. Bitte installieren: sudo apt-get install -y python3-tk")
        return False


def _bootstrap():
    _ensure_python_version()
    _ensure_running_in_venv()
    _prioritize_venv_site_packages(remove_system_site=True)
    if not ensure_tkinter_system():
        sys.exit(1)

    pillow_spec = "<10" if sys.version_info < (3, 8) else ""
    ensure_pkg("PIL", "pillow", pillow_spec)

    # python-can imports these dependencies at module import time.  When the
    # venv was created with access to system site-packages, pip may otherwise
    # incorrectly treat the system copies as sufficient even though they are
    # deliberately removed from sys.path above.
    ensure_pkg("typing_extensions", "typing-extensions")
    ensure_pkg("packaging", "packaging")
    ensure_pkg("can", "python-can")


_bootstrap()

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import can
from can import Notifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Layout fine tuning
IMAGE_GRID_GAP = 8          # black breathing room between image panels and white grid lines
MAIN_GRID_LINE_WIDTH = 3

# Hauptaufteilung links/rechts in festen Pixelbreiten.
# Wenn du LEFT_PANEL_WIDTH kleiner machst, wird das Gesamtfenster schmaler.
# RIGHT_PANEL_WIDTH bleibt dabei unverändert.
LEFT_PANEL_WIDTH = 760
RIGHT_PANEL_WIDTH = 545

WINDOW_HEIGHT = 900
WINDOW_MIN_HEIGHT = 700
WINDOW_MIN_WIDTH = 900

WINDOW_WIDTH = LEFT_PANEL_WIDTH + MAIN_GRID_LINE_WIDTH + RIGHT_PANEL_WIDTH

# Linkes Bild-Grid feinjustieren
# Je kleiner der Wert, desto weniger Höhe bekommt das jeweilige Bildpanel.
# Beispiel: FIS kleiner, RNS-E kleiner, mehr schwarzer Abstand/Reserve im linken Bereich.
LEFT_GRID_FIS_WEIGHT = 35
LEFT_GRID_RNSE_WEIGHT = 65

# Zusätzliche Skalierung der Bilder innerhalb ihrer Grid-Zellen.
# 1.0 = so groß wie möglich innerhalb der Zelle, 0.9 = 90% davon.
FIS_PANEL_SCALE = 0.98
RNSE_PANEL_SCALE = 0.98

# Mindesthöhe der Bildzeilen. 0 = komplett über Gewichtung steuern.
LEFT_GRID_FIS_MIN_HEIGHT = 0
LEFT_GRID_RNSE_MIN_HEIGHT = 0


# =============================================================================
# General CAN helpers
# =============================================================================

CAN_INTERFACE = "vcan0"


def ensure_vcan0():
    try:
        exists = subprocess.run(
            ["ip", "link", "show", CAN_INTERFACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ).returncode == 0

        if not exists:
            subprocess.run(["sudo", "ip", "link", "add", "dev", CAN_INTERFACE, "type", "vcan"])
        subprocess.run(["sudo", "ip", "link", "set", "up", CAN_INTERFACE])
        subprocess.run(["sudo", "ifconfig", CAN_INTERFACE, "txqueuelen", "1000"])
    except Exception as exc:
        print("Could not ensure {}: {}".format(CAN_INTERFACE, exc))


def has_cansend():
    return shutil.which("cansend") is not None


def run_cansend(frame: str, testmode: bool = False):
    if testmode:
        print("[TEST] cansend {} {}".format(CAN_INTERFACE, frame))
        return 0, "TEST"

    if not has_cansend():
        return 127, "cansend missing"

    try:
        res = subprocess.run(
            ["cansend", CAN_INTERFACE, frame],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode == 0:
            return 0, "OK"
        return res.returncode, (res.stderr or res.stdout or "").strip()
    except Exception as exc:
        return 1, str(exc)


def send_frame(frame: str):
    rc, msg = run_cansend(frame)
    if rc != 0:
        print("cansend failed: {} ({})".format(frame, msg))


# =============================================================================
# FIS display
# =============================================================================

FIS_BACKGROUND_IMAGE = "fis.png"

# FIS CAN IDs differ by vehicle platform:
# - Audi A4 8E / Audi R8 42: line 1 = 0x265, line 2 = 0x267
# - Audi A3 8L / 8P / Audi TT 8J: line 1 = 0x667, line 2 = 0x66B
#
# Internally, both variants are mapped to the existing display positions
# "265" (line 1) and "267" (line 2).
FIS_CAN_ID_TO_DISPLAY_LINE = {
    "265": "265",
    "667": "265",
    "267": "267",
    "66B": "267",
}

FIS_CAN_IDS = tuple(FIS_CAN_ID_TO_DISPLAY_LINE.keys())

# Schrift. Für echtes 8-Zeichen-Verhalten eine Monospace-Schrift verwenden.
FIS_FONT_NAME = "Liberation Mono"
FIS_FONT_SIZE = 42
FIS_FONT_MIN_SIZE = 8
FIS_FONT_WEIGHT = "bold"
FIS_TEXT_COLOR = "#efe8e0"

FIS_WINDOW_INITIAL_SCALE = 0.5

# Positionen bezogen auf die Originalgröße von fis.png.
FIS_TEXT_BASE_POS = {
    "265": {"x": 855.0, "y": 285.0},
    "267": {"x": 855.0, "y": 348.0},
}

FIS_SHOW_TEXT_GUIDES = False
FIS_DRAG_TEXT_LINES = False
FIS_PRINT_TEXT_POS_ON_DRAG = False
FIS_GUIDE_COLOR = "#ff3030"
FIS_GUIDE_WIDTH = 1
FIS_GUIDE_DASH = (4, 3)

# Zeichenraster. Bei Monospace reicht meist ein normaler String; hier bleiben
# feste Slots für left/right erhalten.
FIS_CHAR_COUNT = 8
FIS_CHAR_WIDTH = 32.0
FIS_CHAR_OFFSET_X = 0.0
FIS_CHAR_OFFSET_Y = 0.0
FIS_SHOW_CHARACTER_GRID = False
FIS_GRID_COLOR = "#ff6060"
FIS_GRID_WIDTH = 1
FIS_GRID_DASH = (2, 4)


def _audi_ascii_map():
    return {
        "01": "61", "02": "62", "03": "63", "04": "64", "05": "65",
        "06": "66", "07": "67", "08": "68", "09": "69", "0A": "6A",
        "0B": "6B", "0C": "6C", "0D": "6D", "0E": "6E", "0F": "6F",
        "10": "70",
        "91": "E4", "97": "F6", "99": "FC",
        "5F": "C4", "60": "D6", "61": "DC", "8D": "DF", "66": "5F",
        "AA": "A3", "BF": "A7", "A2": "A9", "B4": "B1", "B8": "B5",
        "B1": "B9", "BB": "BA", "83": "E8", "82": "E9",
    }


def decode_audi_fis_bytes(hex_content=""):
    """
    0x65 = echtes Leerzeichen.
    0x20 = Center-/Padding-Marker.

    Wenn 0x20 vorkommt: Marker entfernen, Modus center.
    Ohne 0x20: 8 Slots exakt erhalten.
    """
    mapping = _audi_ascii_map()
    byte_pairs = [hex_content[i:i + 2].upper() for i in range(0, len(hex_content), 2)]
    has_center_marker = any(pair == "20" for pair in byte_pairs)

    result_hex = ""
    for pair in byte_pairs:
        if pair == "20":
            if has_center_marker:
                continue
        if pair == "65":
            result_hex += "20"
        else:
            result_hex += mapping.get(pair, pair)

    ascii_text = bytes.fromhex(result_hex).decode("ISO-8859-1") if result_hex else ""
    mode = "center" if has_center_marker else "slots"

    if mode == "slots":
        ascii_text = ascii_text[:FIS_CHAR_COUNT].ljust(FIS_CHAR_COUNT)

    return ascii_text, mode


class FisPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.image_path = os.path.join(SCRIPT_DIR, FIS_BACKGROUND_IMAGE)
        if not os.path.exists(self.image_path):
            raise FileNotFoundError("FIS background image not found: {}".format(self.image_path))

        self.image = Image.open(self.image_path)
        self.base_w, self.base_h = self.image.size
        self.aspect = self.base_w / self.base_h
        self.render_x = 0
        self.render_y = 0
        self.render_w = self.base_w
        self.render_h = self.base_h

        self.bg_photo = ImageTk.PhotoImage(self.image)
        self.bg_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)
        self.canvas.image = self.bg_photo

        self.center_text_ids = {}
        self.slot_text_ids = {}
        self.guide_ids = {}
        self.grid_ids = {}
        self.text_ids = {}
        self.line_state = {
            "265": {"text": "", "mode": "center"},
            "267": {"text": "", "mode": "center"},
        }
        self.drag_state = {"tag": None, "last_x": 0, "last_y": 0}

        self._create_text_items()
        self.canvas.bind("<Configure>", self._resize_elements)

    def _font_for_width(self, width):
        size = max(FIS_FONT_MIN_SIZE, int(FIS_FONT_SIZE * width / self.base_w))
        return (FIS_FONT_NAME, size, FIS_FONT_WEIGHT)

    def _base_pos(self, tag):
        return (
            float(FIS_TEXT_BASE_POS[tag]["x"]),
            float(FIS_TEXT_BASE_POS[tag]["y"])
        )

    def _scaled_pos(self, tag, width, height):
        base_x, base_y = self._base_pos(tag)
        return (
            self.render_x + width * (base_x / self.base_w),
            self.render_y + height * (base_y / self.base_h),
        )

    def _slot_positions(self, tag, width, height):
        center_x, center_y = self._scaled_pos(tag, width, height)
        scale_x = width / self.base_w
        scale_y = height / self.base_h
        char_width = FIS_CHAR_WIDTH * scale_x
        offset_x = FIS_CHAR_OFFSET_X * scale_x
        offset_y = FIS_CHAR_OFFSET_Y * scale_y
        first_x = center_x - ((FIS_CHAR_COUNT - 1) * char_width / 2.0) + offset_x
        y = center_y + offset_y
        return [(first_x + i * char_width, y) for i in range(FIS_CHAR_COUNT)]

    def _center_char_positions(self, tag, width, height, text_len):
        center_x, center_y = self._scaled_pos(tag, width, height)
        scale_x = width / self.base_w
        scale_y = height / self.base_h
        char_width = FIS_CHAR_WIDTH * scale_x
        offset_x = FIS_CHAR_OFFSET_X * scale_x
        offset_y = FIS_CHAR_OFFSET_Y * scale_y
        text_len = max(0, min(FIS_CHAR_COUNT, int(text_len)))
        if text_len <= 0:
            return []
        first_x = center_x - ((text_len - 1) * char_width / 2.0) + offset_x
        y = center_y + offset_y
        return [(first_x + i * char_width, y) for i in range(text_len)]

    def _active_positions(self, tag, width, height):
        state = self.line_state.get(tag, {"text": "", "mode": "center"})
        if state.get("mode") == "center":
            return self._center_char_positions(tag, width, height, len(state.get("text", "")))
        return self._slot_positions(tag, width, height)

    def _set_line_coords(self, tag, width, height):
        x, y = self._scaled_pos(tag, width, height)
        self.canvas.coords(self.center_text_ids[tag], x, y)

        positions = self._active_positions(tag, width, height)
        for i, char_id in enumerate(self.slot_text_ids[tag]):
            if i < len(positions):
                self.canvas.coords(char_id, positions[i][0], positions[i][1])

        if tag in self.guide_ids:
            self.canvas.coords(self.guide_ids[tag], self.render_x, y, self.render_x + width, y)

        if tag in self.grid_ids:
            grid_positions = self._slot_positions(tag, width, height)
            for i, grid_id in enumerate(self.grid_ids[tag]):
                gx = grid_positions[i][0]
                self.canvas.coords(grid_id, gx, self.render_y, gx, self.render_y + height)

    def _create_text_items(self):
        width = self.base_w
        height = self.base_h
        current_font = self._font_for_width(width)

        for tag in ("265", "267"):
            item_tag = "line_" + tag
            x, y = self._scaled_pos(tag, width, height)

            if FIS_SHOW_TEXT_GUIDES:
                self.guide_ids[tag] = self.canvas.create_line(
                    0, y, width, y,
                    fill=FIS_GUIDE_COLOR,
                    width=FIS_GUIDE_WIDTH,
                    dash=FIS_GUIDE_DASH,
                    tags=(item_tag, "text_guide")
                )

            self.center_text_ids[tag] = self.canvas.create_text(
                x, y,
                text="",
                font=current_font,
                fill=FIS_TEXT_COLOR,
                anchor="center",
                justify="center",
                tags=(item_tag, "fis_text", "fis_center_text")
            )

            self.slot_text_ids[tag] = []
            for sx, sy in self._slot_positions(tag, width, height):
                char_id = self.canvas.create_text(
                    sx, sy,
                    text="",
                    font=current_font,
                    fill=FIS_TEXT_COLOR,
                    anchor="center",
                    justify="center",
                    state="hidden",
                    tags=(item_tag, "fis_text", "fis_slot_text")
                )
                self.slot_text_ids[tag].append(char_id)

            if FIS_SHOW_CHARACTER_GRID:
                self.grid_ids[tag] = []
                for sx, _sy in self._slot_positions(tag, width, height):
                    grid_id = self.canvas.create_line(
                        sx, 0, sx, height,
                        fill=FIS_GRID_COLOR,
                        width=FIS_GRID_WIDTH,
                        dash=FIS_GRID_DASH,
                        tags=(item_tag, "char_grid")
                    )
                    self.grid_ids[tag].append(grid_id)

            self.text_ids[tag] = self.center_text_ids[tag]
            self.canvas.tag_bind(item_tag, "<ButtonPress-1>", lambda event, t=tag: self._start_drag(event, t))
            self.canvas.tag_bind(item_tag, "<B1-Motion>", self._drag)
            self.canvas.tag_bind(item_tag, "<ButtonRelease-1>", self._stop_drag)

    def _display_fis_line(self, tag, text, mode):
        if tag not in self.center_text_ids or tag not in self.slot_text_ids:
            return

        self.canvas.itemconfig(self.center_text_ids[tag], text="", state="hidden")
        text = (text or "")[:FIS_CHAR_COUNT]
        self.line_state[tag] = {"text": text, "mode": mode}

        width = max(getattr(self, "render_w", self.canvas.winfo_width()), 1)
        height = max(getattr(self, "render_h", self.canvas.winfo_height()), 1)
        self._set_line_coords(tag, width, height)

        if mode == "center":
            for i, char_id in enumerate(self.slot_text_ids[tag]):
                if i < len(text):
                    self.canvas.itemconfig(char_id, text=text[i], state="normal")
                else:
                    self.canvas.itemconfig(char_id, text="", state="hidden")
        else:
            slot_text = text[:FIS_CHAR_COUNT].ljust(FIS_CHAR_COUNT)
            for i, char_id in enumerate(self.slot_text_ids[tag]):
                self.canvas.itemconfig(char_id, text=slot_text[i], state="normal")

    def update_from_can_hex(self, can_id, hex_content):
        display_line = FIS_CAN_ID_TO_DISPLAY_LINE.get(can_id)
        if display_line is None:
            return

        text, mode = decode_audi_fis_bytes(hex_content)
        text = text.upper()
        line_number = 1 if display_line == "265" else 2
        print("CAN ID {} -> FIS line {} |{}|".format(can_id, line_number, text))
        self.after(
            0,
            lambda: self._display_fis_line(display_line, text, mode)
        )

    def _resize_elements(self, event):
        canvas_w = max(event.width, 1)
        canvas_h = max(event.height, 1)

        new_w = canvas_w
        new_h = int(new_w / self.aspect)

        if new_h > canvas_h:
            new_h = canvas_h
            new_w = int(new_h * self.aspect)

        panel_scale = max(0.05, min(1.0, float(FIS_PANEL_SCALE)))
        new_w = max(1, int(new_w * panel_scale))
        new_h = max(1, int(new_h * panel_scale))

        self.render_w = new_w
        self.render_h = new_h
        self.render_x = max(0, (canvas_w - new_w) // 2)
        self.render_y = max(0, (canvas_h - new_h) // 2)

        resized_bg = self.image.resize((new_w, new_h), Image.LANCZOS)
        new_bg = ImageTk.PhotoImage(resized_bg)
        self.canvas.itemconfig(self.bg_item, image=new_bg)
        self.canvas.coords(self.bg_item, self.render_x, self.render_y)
        self.canvas.image = new_bg

        for tag in self.center_text_ids.keys():
            self._set_line_coords(tag, new_w, new_h)

        new_font = self._font_for_width(new_w)
        for text_id in self.center_text_ids.values():
            self.canvas.itemconfig(text_id, font=new_font)
        for char_ids in self.slot_text_ids.values():
            for char_id in char_ids:
                self.canvas.itemconfig(char_id, font=new_font)

    def _current_scale(self):
        width = max(getattr(self, "render_w", self.canvas.winfo_width()), 1)
        height = max(getattr(self, "render_h", self.canvas.winfo_height()), 1)
        return width / self.base_w, height / self.base_h

    def _move_text_line_to_base(self, tag, base_x, base_y):
        base_x = max(0, min(self.base_w, base_x))
        base_y = max(0, min(self.base_h, base_y))
        FIS_TEXT_BASE_POS[tag]["x"] = base_x
        FIS_TEXT_BASE_POS[tag]["y"] = base_y
        self._set_line_coords(tag, max(getattr(self, "render_w", self.canvas.winfo_width()), 1), max(getattr(self, "render_h", self.canvas.winfo_height()), 1))

    def _print_current_text_positions(self):
        if not FIS_PRINT_TEXT_POS_ON_DRAG:
            return
        print("\n# Neue Werte für FIS_TEXT_BASE_POS im Code:")
        print("FIS_TEXT_BASE_POS = {")
        for tag in ("265", "267"):
            base_x, base_y = self._base_pos(tag)
            print("    '{}': {{'x': {:.1f}, 'y': {:.1f}}},".format(tag, base_x, base_y))
        print("}\n")

    def _start_drag(self, event, tag):
        if not FIS_DRAG_TEXT_LINES:
            return
        self.drag_state["tag"] = tag
        self.drag_state["last_x"] = event.x
        self.drag_state["last_y"] = event.y

    def _drag(self, event):
        tag = self.drag_state.get("tag")
        if not FIS_DRAG_TEXT_LINES or not tag:
            return
        scale_x, scale_y = self._current_scale()
        dx_base = (event.x - self.drag_state["last_x"]) / scale_x
        dy_base = (event.y - self.drag_state["last_y"]) / scale_y
        self.drag_state["last_x"] = event.x
        self.drag_state["last_y"] = event.y
        base_x, base_y = self._base_pos(tag)
        self._move_text_line_to_base(tag, base_x + dx_base, base_y + dy_base)

    def _stop_drag(self, event):
        if self.drag_state.get("tag"):
            self._print_current_text_positions()
        self.drag_state["tag"] = None


# =============================================================================
# RNS-E button panel
# =============================================================================

RNSE_BACKGROUND_IMAGE = "rns-e.png"
RNSE_SHOW_BUTTON_FRAMES = False
RNSE_BASE_W, RNSE_BASE_H = 1147, 700


def create_canvas_button(canvas, x_ratio, y_ratio, w_ratio, h_ratio, can_id, press_message, release_message, repeat=True):
    button_state = {"pressed": False, "thread": None}

    def repeat_send():
        while button_state["pressed"]:
            send_frame("{}#{}".format(can_id, press_message))
            time.sleep(0.1)

    def on_press(event):
        if button_state["pressed"]:
            return
        button_state["pressed"] = True
        if repeat:
            thread = threading.Thread(target=repeat_send, daemon=True)
            thread.start()
            button_state["thread"] = thread
        else:
            send_frame("{}#{}".format(can_id, press_message))

    def on_release(event):
        button_state["pressed"] = False
        if release_message:
            for _ in range(5):
                send_frame("{}#{}".format(can_id, release_message))
                time.sleep(0.1)

    rect = canvas.create_rectangle(
        0, 0, 1, 1,
        outline="red" if RNSE_SHOW_BUTTON_FRAMES else "",
        width=2 if RNSE_SHOW_BUTTON_FRAMES else 0,
        fill=""
    )

    canvas.tag_bind(rect, "<ButtonPress-1>", on_press)
    canvas.tag_bind(rect, "<ButtonRelease-1>", on_release)
    return rect, x_ratio, y_ratio, w_ratio, h_ratio


class RnsePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)

        bg_path = os.path.join(SCRIPT_DIR, RNSE_BACKGROUND_IMAGE)
        if not os.path.exists(bg_path):
            raise FileNotFoundError("RNS-E background image not found: {}".format(bg_path))

        self.background_image = Image.open(bg_path).convert("RGBA")
        self.render_x = 0
        self.render_y = 0
        self.render_w = RNSE_BASE_W
        self.render_h = RNSE_BASE_H
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_item = self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")
        self.canvas.image = self.background_photo

        self.buttons = [
            # Drehknopf links
            create_canvas_button(self.canvas, 0.760, 0.450, 0.056, 0.103, "461", "373001004001", None, False),
            # Drehknopf rechts
            create_canvas_button(self.canvas, 0.885, 0.450, 0.056, 0.103, "461", "373001002001", None, False),
            # Hoch
            create_canvas_button(self.canvas, 0.765, 0.337, 0.080, 0.105, "461", "373001400000", "373004400000", True),
            # Runter
            create_canvas_button(self.canvas, 0.765, 0.562, 0.080, 0.105, "461", "373001800000", "373004800000", True),
            # Drehknopf drücken / OK
            create_canvas_button(self.canvas, 0.820, 0.448, 0.060, 0.110, "461", "373001001000", "373004001000", True),
            # RETURN
            create_canvas_button(self.canvas, 0.765, 0.680, 0.155, 0.070, "461", "373001000200", "373004000200", True),
            # Weiter / Next
            create_canvas_button(self.canvas, 0.849, 0.240, 0.080, 0.065, "461", "373001020000", "373004020000", True),
            # Zurück / Previous
            create_canvas_button(self.canvas, 0.764, 0.240, 0.080, 0.065, "461", "373001010000", "373004010000", True),
            # SETUP
            create_canvas_button(self.canvas, 0.672, 0.837, 0.080, 0.065, "461", "373001000100", "373004000100", True),
        ]

        self.canvas.bind("<Configure>", self._resize_elements)

    def _resize_elements(self, event):
        aspect = RNSE_BASE_W / RNSE_BASE_H
        new_w = event.width
        new_h = int(new_w / aspect)

        if new_h > event.height:
            new_h = event.height
            new_w = int(new_h * aspect)

        panel_scale = max(0.05, min(1.0, float(RNSE_PANEL_SCALE)))
        new_w = max(1, int(new_w * panel_scale))
        new_h = max(1, int(new_h * panel_scale))

        self.render_w = new_w
        self.render_h = new_h
        self.render_x = max(0, (event.width - new_w) // 2)
        self.render_y = max(0, (event.height - new_h) // 2)

        resized_bg = self.background_image.resize((new_w, new_h), Image.LANCZOS)
        new_bg = ImageTk.PhotoImage(resized_bg)
        self.canvas.itemconfig(self.background_item, image=new_bg)
        self.canvas.coords(self.background_item, self.render_x, self.render_y)
        self.canvas.image = new_bg

        for item, x, y, w, h in self.buttons:
            new_x = self.render_x + x * new_w
            new_y = self.render_y + y * new_h
            new_item_w = w * new_w
            new_item_h = h * new_h
            self.canvas.coords(item, new_x, new_y, new_x + new_item_w, new_y + new_item_h)

        for item, *_ in self.buttons:
            self.canvas.tag_raise(item)


# =============================================================================
# cansend GUI panel
# =============================================================================

ENABLE_TRANSLATION = True
TRANSLATION_LANG = "de"

TRANSLATIONS = {
    "de": {
        "time_date": "Zeit / Datum",
        "car_models": "Auto-Modelle",
        "lights": "Licht",
        "ignition": "Zündung",
        "radio_input": "Radio Quelle",
        "speed": "Geschwindigkeit",
        "rpm": "Drehzahl",
        "coolant": "Kühlmitteltemperatur",
        "outside_temp": "Außentemperatur",
        "manual_input": "Manuelle Eingabe (berechnet & sendet CAN-Frames)",
        "unit_kmh": "km/h",
        "unit_mph": "mph",
        "unit_celsius": "°C",
        "unit_fahrenheit": "°F",
        "unit_rpm": "rpm",
        "btn_send": "Senden",
        "btn_copy": "Kopieren",
        "btn_time_tick": "Di 25. Okt 2022 12:22",
        "btn_time_fixed": "So 24. Aug 2025 13:30",
        "btn_light_on": "Licht AN",
        "btn_light_off": "Licht AUS",
        "btn_radio_fm": "FM",
        "btn_radio_tv": "TV",
        "btn_ignition_off": "Zündung aus",
        "btn_key_removed": "Schlüssel gezogen",
        "btn_forward": "Vorwärts",
        "btn_reverse": "Rückwärts",
        "label_speed_input": "Geschwindigkeit",
        "label_rpm_input": "Drehzahl",
        "label_coolant_input": "Kühlmitteltemperatur",
        "label_outside_input": "Außentemperatur",
        "btn_compose_send": "Alle Felder senden",
        "error_title": "Fehler",
        "error_input_title": "Eingabefehler",
        "error_invalid_number": "Ungültige oder leere Zahl.",
        "copy_title": "Kopieren",
        "copy_ok": "Befehl in die Zwischenablage kopiert.",
    }
}


def t(key: str) -> str:
    return TRANSLATIONS.get(TRANSLATION_LANG, {}).get(key, key)


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def kmh_to_mph(kmh):
    return kmh * 0.621371


CMD_TIME_TICK = "623#0512221125102022"
CMD_TIME_FIXED = "623#0513300024082025"

CMD_LIGHT_OFF = "635#0A0000"
CMD_LIGHT_ON = "635#0A3800"
CMD_TV_MODE = "661#8301123700000000"
CMD_FM_MODE = "661#830112A000000000"
CMD_IGNITION_OFF = "271#11"
CMD_KEY_OUT = "271#10"

CMD_FORWARD = "351#00FD2710E47F7B10"
CMD_REVERSE = "351#020000E40B7F7B10"

SPEED_PRESETS = [
    {"frame": "351#000000E40B7F7B10", "kmh": 0, "mph": 0},
    {"frame": "351#00FD2710E47F7B10", "kmh": 51, "mph": 31},
    {"frame": "351#00FD55E40B7F7B10", "kmh": 110, "mph": 68},
    {"frame": "351#0F00ACA4FB7FFB10", "kmh": 220, "mph": 136},
]

RPM_PRESETS = [
    {"frame": "353#000000B80000", "label": "0 RPM"},
    {"frame": "353#00401FB80000", "label": "2000 RPM"},
    {"frame": "353#00803EB80000", "label": "4000 RPM"},
    {"frame": "353#00C05DB80000", "label": "6000 RPM"},
    {"frame": "353#009065B80000", "label": "6500 RPM"},
    {"frame": "353#00606DB80000", "label": "7000 RPM"},
    {"frame": "353#003075B80000", "label": "7500 RPM"},
    {"frame": "353#00007DB80000", "label": "8000 RPM"},
]

COOLANT_PRESETS_C = [
    {"frame": "353#000000380000", "c": -6},
    {"frame": "353#000000B80000", "c": 90},
    {"frame": "353#000000C60000", "c": 100},
    {"frame": "353#000000D30000", "c": 110},
    {"frame": "353#000000E00000", "c": 120},
    {"frame": "353#000000EE0000", "c": 130},
]

OUTSIDE_TEMP_PRESETS_C = [
    {"frame": "351#000000E40B500012", "c": -10.0},
    {"frame": "351#000000E40B7F7B12", "c": 13.5},
    {"frame": "351#000000E40BA00012", "c": 30.0},
]

CAR_MODEL_PRESETS = [
    {"frames": ["65F#0035C837E2574155", "65F#015A5A5A38453032", "65F#0241313238383831"], "label": "Audi A4 8E (2002)"},
    {"frames": ["65F#0000000000574155", "65F#015A5A5A38503544", "65F#0241303236303133"], "label": "Audi A3 8P (2013)"},
    {"frames": ["65F#0000000000545255", "65F#015A5A5A384A3737", "65F#0231303339323432"], "label": "Audi TT 8J (2007)"},
    {"frames": ["65F#0000000000575541", "65F#015A5A5A34323038", "65F#024E303036343538"], "label": "Audi R8 42 (2008)"},
]


def encode_speed_kmh(kmh: float) -> str:
    raw = int(round(float(kmh) * 200.0))
    raw = max(0, min(0xFFFF, raw))
    hex4 = "{:04X}".format(raw)
    HH, LL = hex4[:2], hex4[2:]
    return "351#00FD{}{}E47F7B10".format(HH, LL)


def encode_rpm(rpm: float) -> str:
    raw = int(round(float(rpm) * 4.0))
    raw = max(0, min(0xFFFF, raw))
    hex4 = "{:04X}".format(raw)
    high, low = hex4[:2], hex4[2:]
    return "353#00{}{}B80000".format(low, high)


def encode_coolant_c(c: float) -> str:
    val = int(round((float(c) + 48.0) / 0.75))
    val = max(0, min(255, val))
    return "353#000000{:02X}0000".format(val)


def encode_outside_c(temp_c: float) -> str:
    val = int(round((float(temp_c) + 50.0) * 2.0))
    val = max(0, min(255, val))
    return "351#000000E40B{:02X}0012".format(val)


def frame_to_id_and_bytes(frame: str):
    cid, payload_hex = frame.split("#", 1)
    payload_hex = payload_hex.strip()
    if len(payload_hex) % 2 != 0 or len(payload_hex) == 0:
        raise ValueError("Payload must have an even number of hex chars")
    return cid.upper().strip(), [int(payload_hex[i:i + 2], 16) for i in range(0, len(payload_hex), 2)]


def id_and_bytes_to_frame(cid: str, b: List[int]) -> str:
    return "{}#{}".format(cid, "".join("{:02X}".format(x) for x in b))


def ensure_len(b: List[int], n: int) -> List[int]:
    if len(b) < n:
        return b + [0] * (n - len(b))
    return b[:n]


# =============================================================================
# Cansend panel spacing / tuning
# =============================================================================

# Abstand zwischen der großen cansend-Überschrift und dem ersten Inhaltsblock.
CANSEND_MAIN_TITLE_CONTENT_GAP = 10

# Abstand zwischen Abschnittsüberschrift und jeweiligem Inhalt/Button-Raster.
CANSEND_SECTION_TITLE_CONTENT_GAP = 1

# Abstand vor dem Block "Manuelle Eingabe".
CANSEND_MANUAL_TOP_GAP = 20


class CansendPanel(ttk.Frame):
    # Ausgewogenes Layout mit variablen Abständen für Feintuning.
    def __init__(self, parent):
        super().__init__(parent)

        self.h1_font = ("TkDefaultFont", 11, "bold")
        self.title_font = ("TkDefaultFont", 13, "bold")
        self.manual_inputs = {}

        self.columnconfigure(0, weight=1)

        title_frame = ttk.Frame(self, padding=(4, 3, 4, 1))
        title_frame.pack(fill="x")
        ttk.Label(title_frame, text="Send CAN test messages", font=self.title_font).pack(anchor="w")
        if CANSEND_MAIN_TITLE_CONTENT_GAP > 0:
            ttk.Frame(self, height=CANSEND_MAIN_TITLE_CONTENT_GAP).pack(fill="x")

        self._add_section(t("time_date"), self._section_time)
        self._add_section(t("car_models"), self._section_car_models)
        self._add_section(t("lights"), self._section_lights)
        self._add_section(t("ignition"), self._section_ignition)
        self._add_section(t("radio_input"), self._section_radio)
        self._add_section(t("speed"), self._section_speed)
        self._add_section(t("rpm"), self._section_rpm)
        self._add_section(t("coolant"), self._section_coolant)
        self._add_section(t("outside_temp"), self._section_outside_temp)

        # Optische Trennung vor der manuellen Eingabe.
        if CANSEND_MANUAL_TOP_GAP > 0:
            ttk.Frame(self, height=CANSEND_MANUAL_TOP_GAP).pack(fill="x")
        self._add_section(t("manual_input"), self._section_manual_input)

    def _add_section(self, title, builder_fn):
        # White separator line between right-panel sections. This makes the
        # control panel follow the same visual grid as the left image area.
        if hasattr(self, "_section_count") and self._section_count > 0:
            tk.Frame(self, bg="white", height=1).pack(fill="x", padx=4, pady=(3, 2))
        self._section_count = getattr(self, "_section_count", 0) + 1

        frame = ttk.Frame(self, padding=(4, 1, 4, 1))
        frame.pack(fill="x", pady=0)
        ttk.Label(frame, text=title, font=self.h1_font).pack(anchor="w", pady=(0, CANSEND_SECTION_TITLE_CONTENT_GAP))
        builder_fn(frame)

    def _grid_buttons(self, parent, items, per_row, width=None):
        # Buttons fill the available right-panel width instead of forcing
        # fixed character widths that can push controls under the scrollbar.
        cols = per_row if per_row and per_row > 0 else max(1, len(items))
        g = ttk.Frame(parent)
        g.pack(fill="x", pady=(0, 1))
        for c in range(cols):
            g.columnconfigure(c, weight=1, uniform="btncols")
        for i, (text, cmd) in enumerate(items):
            r, c = divmod(i, cols)
            ttk.Button(g, text=text, command=cmd).grid(row=r, column=c, padx=2, pady=1, sticky="ew")

    def _send(self, frame):
        rc, _ = run_cansend(frame)
        if rc != 0:
            self.bell()

    def _send_frames(self, frames):
        for frame in frames:
            rc, _ = run_cansend(frame)
            if rc != 0:
                self.bell()
                break

    def _section_time(self, p):
        self._grid_buttons(p, [
            (t("btn_time_tick"), lambda: self._send(CMD_TIME_TICK)),
            (t("btn_time_fixed"), lambda: self._send(CMD_TIME_FIXED)),
        ], 2, 22)

    def _section_car_models(self, p):
        self._grid_buttons(p, [
            (m["label"], lambda frames=m["frames"]: self._send_frames(frames))
            for m in CAR_MODEL_PRESETS
        ], 2, 22)

    def _section_lights(self, p):
        self._grid_buttons(p, [
            (t("btn_light_on"), lambda: self._send(CMD_LIGHT_ON)),
            (t("btn_light_off"), lambda: self._send(CMD_LIGHT_OFF)),
        ], 2, 22)

    def _section_ignition(self, p):
        self._grid_buttons(p, [
            (t("btn_ignition_off"), lambda: self._send(CMD_IGNITION_OFF)),
            (t("btn_key_removed"), lambda: self._send(CMD_KEY_OUT)),
        ], 2, 22)

    def _section_radio(self, p):
        self._grid_buttons(p, [
            (t("btn_radio_fm"), lambda: self._send(CMD_FM_MODE)),
            (t("btn_radio_tv"), lambda: self._send(CMD_TV_MODE)),
        ], 2, 22)

    def _section_speed(self, p):
        items = []
        for preset in SPEED_PRESETS:
            label = "{} {} | {} {}".format(preset["kmh"], t("unit_kmh"), preset["mph"], t("unit_mph"))
            items.append((label, lambda f=preset["frame"]: self._send(f)))
        items.append((t("btn_forward"), lambda: self._send(CMD_FORWARD)))
        items.append((t("btn_reverse"), lambda: self._send(CMD_REVERSE)))
        self._grid_buttons(p, items, 2, 22)

    def _section_rpm(self, p):
        self._grid_buttons(p, [(p_["label"], lambda f=p_["frame"]: self._send(f)) for p_ in RPM_PRESETS], 4, 10)

    def _section_coolant(self, p):
        items = []
        for preset in COOLANT_PRESETS_C:
            label = "{}{} | {}{}".format(preset["c"], t("unit_celsius"), int(round(c_to_f(preset["c"]))), t("unit_fahrenheit"))
            items.append((label, lambda f=preset["frame"]: self._send(f)))
        self._grid_buttons(p, items, 3, 14)

    def _section_outside_temp(self, p):
        items = []
        for preset in OUTSIDE_TEMP_PRESETS_C:
            label = "{:.1f}{} | {:.1f}{}".format(
                preset["c"], t("unit_celsius"),
                c_to_f(preset["c"]), t("unit_fahrenheit")
            )
            items.append((label, lambda f=preset["frame"]: self._send(f)))
        self._grid_buttons(p, items, 3, 14)

    def _section_manual_input(self, p):
        grid = ttk.Frame(p)
        grid.pack(fill="x", pady=(2, 0))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=0)
        grid.columnconfigure(2, weight=0)
        grid.columnconfigure(3, weight=1)
        grid.columnconfigure(4, weight=0)
        grid.columnconfigure(5, weight=0)

        def make_row(row, label_key, unit_text, build_fn, key_name, conv_fn=None, conv_suffix=""):
            ttk.Label(grid, text=t(label_key)).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
            entry = ttk.Entry(grid, width=6, justify="right")
            entry.grid(row=row, column=1, sticky="w", pady=1)
            ttk.Label(grid, text=unit_text).grid(row=row, column=2, sticky="w", padx=(5, 5))

            conv_var = tk.StringVar(value="—")
            ttk.Label(grid, textvariable=conv_var, foreground="#666", width=10, anchor="w").grid(
                row=row, column=3, sticky="w", padx=(6, 4)
            )

            def update_conv():
                txt = entry.get().strip().replace(",", ".")
                if not txt:
                    conv_var.set("—")
                    return
                try:
                    val = float(txt)
                except ValueError:
                    conv_var.set("—")
                    return
                if conv_fn:
                    conv_var.set("≈ {:.1f} {}".format(conv_fn(val), conv_suffix))
                else:
                    conv_var.set("")

            def build_frame_from_entry():
                val_txt = entry.get().strip().replace(",", ".")
                if not val_txt:
                    return None
                try:
                    val = float(val_txt)
                except ValueError:
                    return None
                return build_fn(val)

            def on_send():
                f = build_frame_from_entry()
                if not f:
                    messagebox.showerror(t("error_input_title"), t("error_invalid_number"))
                    return
                self._send(f)

            def on_copy():
                f = build_frame_from_entry()
                if not f:
                    messagebox.showerror(t("error_input_title"), t("error_invalid_number"))
                    return
                cmd = "cansend {} {}".format(CAN_INTERFACE, f)
                self.clipboard_clear()
                self.clipboard_append(cmd)
                self.update_idletasks()
                messagebox.showinfo(t("copy_title"), t("copy_ok"))

            entry.bind("<KeyRelease>", lambda e: update_conv())
            entry.bind("<Return>", lambda e: on_send())
            ttk.Button(grid, text=t("btn_send"), command=on_send, width=8).grid(row=row, column=4, sticky="ew", padx=(6, 2), pady=1)
            ttk.Button(grid, text=t("btn_copy"), command=on_copy, width=8).grid(row=row, column=5, sticky="ew", padx=(2, 0), pady=1)

            self.manual_inputs[key_name] = {"entry": entry, "get_frame": build_frame_from_entry, "update_conv": update_conv}
            update_conv()

        make_row(0, "label_speed_input", t("unit_kmh"), encode_speed_kmh, "speed",
                 conv_fn=lambda v: kmh_to_mph(v), conv_suffix=t("unit_mph"))
        make_row(1, "label_rpm_input", t("unit_rpm"), encode_rpm, "rpm")
        make_row(2, "label_coolant_input", t("unit_celsius"), encode_coolant_c, "coolant",
                 conv_fn=lambda v: c_to_f(v), conv_suffix=t("unit_fahrenheit"))
        make_row(3, "label_outside_input", t("unit_celsius"), encode_outside_c, "outside",
                 conv_fn=lambda v: c_to_f(v), conv_suffix=t("unit_fahrenheit"))

        ttk.Button(grid, text=t("btn_compose_send"), command=self._compose_and_send_combined).grid(
            row=4, column=0, columnspan=6, sticky="ew", padx=(0, 0), pady=(4, 0)
        )

    def _compose_and_send_combined(self):
        f_speed = self.manual_inputs["speed"]["get_frame"]()
        f_outside = self.manual_inputs["outside"]["get_frame"]()
        f_rpm = self.manual_inputs["rpm"]["get_frame"]()
        f_coolant = self.manual_inputs["coolant"]["get_frame"]()

        combined_351 = None
        if f_speed and f_outside:
            _, b_speed = frame_to_id_and_bytes(f_speed)
            _, b_outside = frame_to_id_and_bytes(f_outside)
            b = ensure_len(b_speed, 8)
            b[5] = ensure_len(b_outside, 8)[5]
            combined_351 = id_and_bytes_to_frame("351", b)
        elif f_speed:
            combined_351 = f_speed
        elif f_outside:
            combined_351 = f_outside

        combined_353 = None
        if f_rpm and f_coolant:
            _, b_rpm = frame_to_id_and_bytes(f_rpm)
            _, b_coolant = frame_to_id_and_bytes(f_coolant)
            b = ensure_len(b_rpm, 6)
            b[3] = ensure_len(b_coolant, 6)[3]
            combined_353 = id_and_bytes_to_frame("353", b)
        elif f_rpm:
            combined_353 = f_rpm
        elif f_coolant:
            combined_353 = f_coolant

        sent_any = False
        for frame in (combined_351, combined_353):
            if frame:
                self._send(frame)
                sent_any = True

        if not sent_any:
            messagebox.showinfo("cansend gui", t("error_invalid_number"))


# =============================================================================
# Main application
# =============================================================================

class CombinedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audi CAN Test GUI")
        self.geometry("{}x{}".format(WINDOW_WIDTH, WINDOW_HEIGHT))
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        try:
            style = ttk.Style(self)
            style.theme_use("clam")
            style.configure("TButton", padding=(3, 1))
        except tk.TclError:
            pass

        self.notifier = None
        self.bus = None

        # Main layout: left = FIS + RNS-E, white grid divider, right = controls.
        # Links/rechts haben feste Breiten. Dadurch wird das Fenster schmaler,
        # wenn LEFT_PANEL_WIDTH reduziert wird; rechts wird nicht automatisch breiter.
        self.columnconfigure(0, weight=0, minsize=LEFT_PANEL_WIDTH)
        self.columnconfigure(1, weight=0, minsize=MAIN_GRID_LINE_WIDTH)
        self.columnconfigure(2, weight=0, minsize=RIGHT_PANEL_WIDTH)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg="black", width=LEFT_PANEL_WIDTH)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)
        # Image rows with a visible white separator in between. Each panel keeps
        # its own aspect ratio and centers the rendered image inside its cell.
        left.rowconfigure(0, weight=LEFT_GRID_FIS_WEIGHT, minsize=LEFT_GRID_FIS_MIN_HEIGHT)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=LEFT_GRID_RNSE_WEIGHT, minsize=LEFT_GRID_RNSE_MIN_HEIGHT)
        left.columnconfigure(0, weight=1)

        self.fis_panel = FisPanel(left)
        self.fis_panel.grid(
            row=0, column=0, sticky="nsew",
            padx=IMAGE_GRID_GAP, pady=(IMAGE_GRID_GAP, IMAGE_GRID_GAP)
        )

        tk.Frame(left, bg="white", height=MAIN_GRID_LINE_WIDTH).grid(row=1, column=0, sticky="ew")

        self.rnse_panel = RnsePanel(left)
        self.rnse_panel.grid(
            row=2, column=0, sticky="nsew",
            padx=IMAGE_GRID_GAP, pady=(IMAGE_GRID_GAP, IMAGE_GRID_GAP)
        )

        tk.Frame(self, bg="white", width=MAIN_GRID_LINE_WIDTH).grid(row=0, column=1, sticky="ns")

        right_outer = ttk.Frame(self, width=RIGHT_PANEL_WIDTH)
        right_outer.grid(row=0, column=2, sticky="nsew")
        right_outer.grid_propagate(False)

        # Scrollable right control panel, with auto-hidden scrollbar.
        # On the normal 900px high window it should fit without a visible mini scrollbar;
        # on smaller windows the scrollbar appears automatically.
        self.right_canvas = tk.Canvas(right_outer, highlightthickness=0, width=RIGHT_PANEL_WIDTH)
        self.right_scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=self._on_right_yscroll)
        self.right_canvas.pack(side="left", fill="both", expand=True)

        self.cansend_frame = CansendPanel(self.right_canvas)
        self.right_window = self.right_canvas.create_window((0, 0), window=self.cansend_frame, anchor="nw")
        self.cansend_frame.bind("<Configure>", self._on_right_frame_configure)
        self.right_canvas.bind("<Configure>", self._on_right_canvas_configure)

        self._start_can_listener()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_right_frame_configure(self, event):
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        self._update_right_scrollbar_visibility()

    def _on_right_canvas_configure(self, event):
        self.right_canvas.itemconfig(self.right_window, width=event.width)
        self._update_right_scrollbar_visibility()

    def _on_right_yscroll(self, first, last):
        self.right_scrollbar.set(first, last)
        self._update_right_scrollbar_visibility()

    def _update_right_scrollbar_visibility(self):
        bbox = self.right_canvas.bbox("all")
        if not bbox:
            return
        content_h = bbox[3] - bbox[1]
        canvas_h = max(self.right_canvas.winfo_height(), 1)
        needs_scroll = content_h > canvas_h + 8

        if needs_scroll:
            if not getattr(self, "_right_scrollbar_visible", False):
                self.right_scrollbar.pack(side="right", fill="y")
                self._right_scrollbar_visible = True
        else:
            if getattr(self, "_right_scrollbar_visible", False):
                self.right_scrollbar.pack_forget()
                self._right_scrollbar_visible = False
            self.right_canvas.yview_moveto(0)

    def _start_can_listener(self):
        can_filters = [
            dict(can_id=int(can_id, 16), can_mask=0x7FF, extended=False)
            for can_id in FIS_CAN_IDS
        ]
        try:
            self.bus = can.interface.Bus(
                CAN_INTERFACE,
                interface="socketcan",
                bitrate=100000,
                can_filters=can_filters,
                receive_own_messages=False
            )
            self.notifier = Notifier(self.bus, [self._read_on_canbus])
        except Exception as exc:
            messagebox.showwarning("CAN", "CAN listener konnte nicht gestartet werden:\n{}".format(exc))

    def _read_on_canbus(self, msg):
        can_id = "{:X}".format(msg.arbitration_id)
        hex_content = binascii.hexlify(msg.data).decode("ascii").upper()
        if can_id in FIS_CAN_ID_TO_DISPLAY_LINE:
            self.fis_panel.update_from_can_hex(can_id, hex_content)

    def _on_close(self):
        try:
            if self.notifier:
                self.notifier.stop()
        except Exception:
            pass

        try:
            if self.bus:
                self.bus.shutdown()
        except Exception:
            pass

        self.destroy()


def main():
    ensure_vcan0()
    app = CombinedApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
