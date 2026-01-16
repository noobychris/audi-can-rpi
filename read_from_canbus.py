#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Project on GitHub: https://github.com/noobychris/audi-can-rpi
script_version = "v0.9.6"

# If you have any trouble with the script, you can enable LOGGING_OUTPUT to get more information's about exceptions.
# The messages will be saved in the script/logs folder with date, name of the script.
# Example: 2023-02-01_read_from_canbus_errors.log

ENABLE_LOGGING = False  # or False, default before reading feature file
show_can_messages_in_logs = False

FEATURE_FILE = "feature_set.txt"

# --- 1. Default features without ENABLE_LOGGING ---
default_features = {
    "can_interface": 'can0',
    "welcome_message_1st_line": 'WELCOME',
    "welcome_message_2nd_line": 'USER',
    "send_on_canbus": True,
    "only_send_if_radio_is_in_tv_mode": False,
    "activate_rnse_tv_input": False,
    "tv_input_format": 'NTSC',
    "show_label": False,
    "toggle_fis1": 6,
    "toggle_fis2": 7,
    "scroll_type": 'oem_style',
    "read_and_set_time_from_dashboard": True,
    "control_pi_by_rns_e_buttons": True,
    "read_mfsw_buttons": False,
    "send_values_to_dashboard": True,
    "toggle_values_by_rnse_longpress": True,
    "reversecamera_by_reversegear": False,
    "reversecamera_by_down_longpress": False,
    "reversecamera_guidelines": True,
    "reversecamera_turn_off_delay": 5,
    "shutdown_by_ignition_off": False,
    "shutdown_by_pulling_key": False,
    "shutdown_type": 'gently',
    "initial_day_night_mode": 'night',
    "change_dark_mode_by_car_light": True,
    "send_api_mediadata_to_dashboard": True,
    "send_to_api_gauges": True,
    "lower_speed": 0,
    "upper_speed": 100,
    "export_speed_measurements_to_file": True,
    "speed_unit": 'km/h',
    "temp_unit": '°C',
    "show_can_messages_in_logs": False
}

import asyncio, binascii, configparser, contextlib, importlib, importlib.util, inspect, io, logging, os, re, shutil
import socket, struct, subprocess, sys, sysconfig, tempfile, textwrap, threading, time, traceback, zipfile, json
from asyncio import get_running_loop
from ctypes.util import find_library
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Union, Tuple


# ---------------------------
# Status / Configuration
# ---------------------------
stop_flag = False
shutdown_script = False
api_is_connected = False

backend = None
tmset = None
car_model_set = None
carmodel = ''
Client = None
version = None
PackageNotFoundError = None
camera = None

# ---------------------------
# Button states
# ---------------------------
press_mfsw = 0
up = 0
down = 0
select = 0
back = 0

nextbtn = 0
prev = 0
setup = 0
gear = 0
light_status = 0
elapsed_time = 0.0

# ---------------------------
# Projection states
# ---------------------------
ProjectionState = None
ProjectionSource = None

# ---------------------------
# FIS / CAN data
# ---------------------------
FIS1 = '265'
FIS2 = '267'

speed = 0
rpm = 0
coolant = 0
outside_temp = ""

last_speed = None
last_outside_temp = None
last_rpm = None
last_coolant = None

rpm_counter = 0
coolant_counter = 0
speed_counter = 0
outside_temp_counter = 0

last_msg_635 = ''
last_msg_271_2C3 = ''

playing = ''
position = ''
source = ''
title = ''
artist = ''
album = ''
state = None
duration = ''

begin1 = -1
end1 = 7
begin2 = -1
end2 = 7

pause_fis1 = False
pause_fis2 = False
light_set = False
script_started = False
deactivate_overwrite_dis_content = False

can_functional = None
guidelines_set = False
camera_active = None

measure_done = 0
data = ''
last_data = ''
drop1 = 0
start_time = None

# ---------------------------
# System metrics / processes
# ---------------------------
cpu_load = 0
cpu_temp = 0
cpu_freq_mhz = 0
candump_process = None

# ---------------------------
# Internal caches
# ---------------------------
_cached_metadata = None
tv_mode_active = 1

# Threading / async
lock = threading.Lock()
stop_completed_event = threading.Event()
tasks, remote_task = [], None


async def init_async_primitives():
    """Create asyncio primitives once a loop is running (Py3.7-safe)."""
    global remote_server_shutdown_event
    if remote_server_shutdown_event is None:
        remote_server_shutdown_event = asyncio.Event()

class ContextualFormatter(logging.Formatter):
    def format(self, record):
        # Only replace if funcName is the default value
        if record.funcName in {"<module>", "run", "main", "unknown"}:
            # Walk through the stack until we find the actual calling function
            for frame_info in inspect.stack():
                frame = frame_info.frame
                code = frame.f_code
                if code.co_filename == record.pathname and code.co_name not in {"format", "emit"}:
                    record.funcName = code.co_name
                    record.lineno = frame_info.lineno
                    break
        return super().format(record)

class ThreadNameFilter(logging.Filter):
    def filter(self, record):
        if "can.notifier" in record.threadName:
            record.threadName = "can_notifier"
        elif record.threadName.startswith("ThreadPoolExecutor-"):
            record.threadName = "API_EventHandler"
        return True

def setup_logging():
    global logger, log_file, log_filename
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = ContextualFormatter("%(asctime)s | %(levelname)-7s | %(lineno)4d | %(funcName)-23s | %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S,%f")

    now = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    path = os.path.dirname(os.path.abspath(__file__))
    filename = str(os.path.basename(__file__)).rstrip('.py')
    log_dir = f'{path}/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_filename = f'{log_dir}{now}_{filename}_errors.log'
    if ENABLE_LOGGING:
        file_handler = logging.FileHandler(log_filename, mode='a', delay=True)
        formatter = logging.Formatter(
            '%(asctime)-22s | %(levelname)-7s | %(lineno)-4d | %(funcName)-21s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)-22s | %(levelname)-7s | %(lineno)-4d | %(funcName)-21s | %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logging()
logger.addFilter(ThreadNameFilter())


async def write_header_to_log_file(log_filename):
    header = f"{'TIME':<23} | {'LEVEL':<7} | {'LINE':<4} | {'FUNCTION':<21} | {'MESSAGE'}"
    message_start_index = header.rfind("|")
    if message_start_index != -1:
        separator_line = ''.join("|" if c == "|" else "-" for c in header)
        separator_line += "-" * 80
    else:
        separator_line = "-" * (len(header) + 80)

    print(header)
    print(separator_line)

    if ENABLE_LOGGING:
        _write_to_file(log_filename, header, separator_line)


def _write_to_file(log_filename, header, separator_line):
    with open(log_filename, 'a') as log_file:
        log_file.write('\n')
        log_file.write(header + '\n')
        log_file.write(separator_line + '\n')


def _print_to_console(header, separator_line):
    print()
    print()
    print(header)
    print(separator_line)


class AsyncDualOutput:
    def __init__(self, file):
        self.file = file
        self.loop = asyncio.get_event_loop()

    async def write(self, message):
        self.loop = asyncio.get_running_loop()
        await self.loop.run_in_executor(None, self.file.write, message)
        await self.loop.run_in_executor(None, sys.__stdout__.write, message)

    async def flush(self):
        self.loop = asyncio.get_running_loop()
        await self.loop.run_in_executor(None, self.file.flush)
        await self.loop.run_in_executor(None, sys.__stdout__.flush)



def load_features():
    """Load features from FEATURE_FILE or use defaults, update globals, and log all changes."""
    global ENABLE_LOGGING

    user_features = {}

    # 1. Read feature file if it exists
    if os.path.exists(FEATURE_FILE):
        if ENABLE_LOGGING:
            logger.info(f"Feature file '{FEATURE_FILE}' found, loading user settings.")
            logger.info("")
        with open(FEATURE_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key in default_features:  # ENABLE_LOGGING nicht mehr hier
                        # Convert value safely: True/False, int, float, or string
                        if val.lower() in ("true", "false"):
                            val = val.lower() == "true"
                        elif val.isdigit():
                            val = int(val)
                        elif val.replace(".", "", 1).isdigit():
                            val = float(val)
                        else:
                            val = val.strip("'\"")
                        user_features[key] = val
                        # Log each loaded feature nicely
                        if ENABLE_LOGGING:
                            logger.info(f"   • {key} = {val}")
                except Exception as e:
                    if ENABLE_LOGGING:
                        logger.warning(f"Error reading line: '{line}' -> {e}")
    else:
        if ENABLE_LOGGING:
            logger.info(f"Feature file '{FEATURE_FILE}' not found, using default values.")

    # 2. Merge defaults with user-defined values
    features = default_features.copy()
    features.update(user_features)

    # 3. Set variables in the script's namespace
    globals().update(features)
    if ENABLE_LOGGING:
        logger.info("")
        logger.info("All features were exposed as global variables.")

    # 4. Write the complete feature file back (ohne ENABLE_LOGGING)
    with open(FEATURE_FILE, "w") as f:
        for k, v in features.items():
            if isinstance(v, str):
                v = f"'{v}'"
            f.write(f"{k} = {v}\n")

    if ENABLE_LOGGING:
        logger.info(f"Feature file '{FEATURE_FILE}' updated with all current values.")
        logger.info("")



# --- robust venv bootstrap (ensure py>=3.11, then (re)build venv, then restart) ---
import os, sys, shutil, subprocess

REQUIRED_PY = (3, 11)
VENV_PATH   = os.path.expanduser("~/.venv-canbus")
VENV_PY     = os.path.join(VENV_PATH, "bin", "python")

def _in_venv():
    return hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix

def _find_python311():
    p = shutil.which("python3.11")
    if p: return p
    for path in ("/usr/local/bin/python3.11", "/usr/bin/python3.11"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    try:
        out = subprocess.check_output(["bash", "-lc", "command -v python3.11 || true"],
                                      universal_newlines=True)
        cand = (out or "").strip()
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    except Exception:
        pass
    return None

def _exec_clean(py_path):
    env = os.environ.copy()
    # alte venv-Einflüsse entfernen
    venv_dir = env.get("VIRTUAL_ENV")
    if venv_dir:
        venv_bin = os.path.join(venv_dir, "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep)
            if os.path.abspath(p) != os.path.abspath(venv_bin)
        )
        env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    for extra in ("/usr/local/bin", "/usr/bin"):
        if extra not in env.get("PATH", ""):
            env["PATH"] = (env.get("PATH", "") + (os.pathsep if env.get("PATH") else "") + extra)
    os.execvpe(py_path, [py_path] + sys.argv, env)

# Schritt 1: Vor ALLEM sicherstellen, dass wir unter >=3.11 laufen (ohne venv!)
SKIP_VENV_BOOTSTRAP = False
if sys.version_info < REQUIRED_PY:
    py311 = _find_python311()
    if py311:
        print("🔁 Restart with {} (detach old venv)…".format(py311), flush=True)
        _exec_clean(py311)
    else:
        # 3.11 noch nicht installiert – das erledigt später dein ensure_*()
        # Wichtig: jetzt NICHT in die alte venv springen!
        SKIP_VENV_BOOTSTRAP = True
        print("⬇️ Python < 3.11 and python3.11 not found yet – skipping old venv for this run.", flush=True)

# Schritt 2: Nur wenn nicht übersprungen → venv (neu) bauen und hineinstarten
def _venv_is_modern():
    if not os.path.exists(VENV_PY):
        return False
    try:
        out = subprocess.check_output(
            [VENV_PY, "-c", "import sys; print(sys.version_info[:2])"],
            universal_newlines=True
        ).strip()
        # akzeptiere 3.11, 3.12, 3.13 …
        return out.startswith("(3, 11)") or out.startswith("(3, 12)") or out.startswith("(3, 13)")
    except Exception:
        return False

if not SKIP_VENV_BOOTSTRAP and not _in_venv():
    if not _venv_is_modern():
        # alte/falsche venv weg und neu erstellen auf Basis des JETZT laufenden Interpreters
        try:
            shutil.rmtree(VENV_PATH)
        except Exception:
            pass
        print("📦 Creating virtual environment at {} …".format(VENV_PATH), flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", VENV_PATH])
        subprocess.call([VENV_PY, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    print("🔁 Restarting the script inside venv ({})…".format(VENV_PATH), flush=True)
    print()
    os.execv(VENV_PY, [VENV_PY] + sys.argv)
# --- end bootstrap ---


# ===== HELFER: CPU-HW-Min/Max + Meta-Aufbau + Hello-Antwort =====
import os, json, asyncio, time

def _read_int(path: str):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def _read_cpuinfo_minmax_khz():
    bases = [
        "/sys/devices/system/cpu/cpufreq/policy0",
        "/sys/devices/system/cpu/cpu0/cpufreq",
    ]
    for base in bases:
        if os.path.isdir(base):
            min_khz = _read_int(os.path.join(base, "cpuinfo_min_freq"))
            max_khz = _read_int(os.path.join(base, "cpuinfo_max_freq"))
            if min_khz is not None and max_khz is not None:
                return min_khz, max_khz
    return None, None

def _khz_to_mhz(x): return round(x / 1000) if isinstance(x, int) else None

def build_initial_meta(logger=None, enable_logging=True):
    """
    Baut den _meta-Block wie in update_to_api:
    - cpu_freq_mhz: Hardware-Min/Max (aus cpuinfo_*), in MHz
    - cpu_temp / outside_temp / coolant: {unit}
    - speed: {units, current_unit}
    - speed_measure: {lower_speed, upper_speed, unit}
    Greift auf deine globalen Variablen zurück: temp_unit, speed_unit, lower_speed, upper_speed
    """
    min_khz, max_khz = _read_cpuinfo_minmax_khz()
    min_mhz, max_mhz = _khz_to_mhz(min_khz), _khz_to_mhz(max_khz)

    meta = {
        "cpu_freq_mhz": {"min_mhz": min_mhz, "max_mhz": max_mhz},
        "cpu_temp": temp_unit,
        "outside_temp": temp_unit,
        "coolant": temp_unit,
        "speed": speed_unit,
        "speed_measure": {
            "lower_speed": lower_speed,
            "upper_speed": upper_speed,
            "unit": speed_unit,  # hier bleibt es sinnvoll, weil mehrere Infos
        },
    }

    if enable_logging and logger:
        logger.info("")
        logger.info("🔧 Built initial meta dict:")
        for k, v in meta.items():
            if isinstance(v, dict):
                # format dict-Inhalt ohne { }
                inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
                logger.info("   - %s: %s", k, inner)
            else:
                logger.info("   - %s: %s", k, v)
        logger.info("")
    return meta


async def _send_meta_to_ws(ws, meta_cache: dict, logger=None, enable_logging=True):
    """Schickt {_meta: ...} an genau diesen Client und loggt das Ereignis."""
    try:
        await ws.send(json.dumps({"_meta": meta_cache}, separators=(",", ":")))
        if enable_logging and logger:
            logger.info("✅ Sent initial _meta to the connected client.")
    except Exception:
        if enable_logging and logger:
            logger.warning("⚠️ Failed to send initial _meta to a new client.", exc_info=True)



class HudiyDashboardHub:
    def __init__(self, host="127.0.0.1", port=8765, logger=None, enable_logging=True):
        self.host = host
        self.port = port
        self._server = None
        self._clients = set()
        self._loop = None
        self.started = False
        self.logger = logger
        self.enable_logging = enable_logging
        self._meta_cache = {}  # bleibt: sammelt letzte Metadaten pro Key

    async def start(self):
        if self.started:
            return
        if WS_SERVER is None:
            raise RuntimeError("Package 'websockets' fehlt. Installiere mit: pip install websockets")
        self._loop = asyncio.get_running_loop()
        self._server = await WS_SERVER.serve(self._handler, self.host, self.port)
        self.started = True
        if self.enable_logging and self.logger:
            self.logger.info("Hudiy Websocket hub started at ws://%s:%d", self.host, self.port)

    async def stop(self):
        if not self.started:
            return
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._clients.clear()
        self.started = False
        if self.enable_logging and self.logger:
            self.logger.info("WS hub stopped")

    async def _handler(self, ws):
        """
        Hello-/Meta-Handshake:
        - Browser sendet {type:"request_meta", page, href, title} nach onopen
          (Kompatibilität: {type:"hello"} funktioniert weiterhin)
        - Wir loggen die Quelle und antworten *diesem* ws mit {_meta: ...}
        - Danach normale Broadcasts via set()/update()/set_with_meta()
        """
        self._clients.add(ws)
        try:
            async for msg in ws:
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue

                if not isinstance(obj, dict):
                    continue

                msg_type = obj.get("type", "").lower()

                if msg_type in ("request_meta", "hello"):
                    page = str(obj.get("page") or "").strip()
                    href = str(obj.get("href") or "").strip()
                    title = str(obj.get("title") or "").strip()

                    # schöneres Logging
                    if self.enable_logging and self.logger:
                        if page or title or href:
                            self.logger.info("")
                            self.logger.info(
                                '🌐 HTML page "%s" (%s) connected via WebSocket – requesting metadata…',
                                title or "(no title)",
                                page or "(no path)",
                            )
                            self.logger.info("")
                        else:
                            self.logger.info("")
                            self.logger.info("🌐 WebSocket client connected – requesting metadata…")
                            self.logger.info("")


                    # Metas bauen (loggt intern den Dict-Inhalt)
                    fresh_meta = build_initial_meta(
                        logger=self.logger,
                        enable_logging=self.enable_logging
                    )

                    # Cache zusammenführen (falls vorher Metas via set_with_meta() ergänzt wurden)
                    for k, v in fresh_meta.items():
                        prev = self._meta_cache.get(k)
                        if isinstance(v, dict) and isinstance(prev, dict):
                            # beide dict → zusammenführen
                            self._meta_cache[k] = {**prev, **v}
                        else:
                            # ansonsten einfach überschreiben (z. B. String, Zahl …)
                            self._meta_cache[k] = v

                    # Nur an diesen Client senden (loggt Erfolg/Fehler)
                    await _send_meta_to_ws(
                        ws, self._meta_cache,
                        logger=self.logger, enable_logging=self.enable_logging
                    )

                # (optional: weitere Client-Kommandos hier behandeln …)

        finally:
            self._clients.discard(ws)

    async def _broadcast_json(self, obj: dict):
        if not self._clients:
            return
        data = json.dumps(obj)
        await asyncio.gather(*(c.send(data) for c in list(self._clients)), return_exceptions=True)

    def set(self, key: str, value):
        if self.started and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_json({"key": key, "value": value, "t": time.time()}),
                self._loop
            )

    def set_with_meta(self, key: str, value, meta: dict):
        # Cache aktuell halten, damit der nächste "hello"-Client alles bekommt
        if isinstance(meta, dict):
            self._meta_cache[key] = {**self._meta_cache.get(key, {}), **meta}
        if self.started and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_json({"key": key, "value": value, "meta": meta or {}, "t": time.time()}),
                self._loop
            )

    def update(self, **kwargs):
        if not (self.started and self._loop):
            return
        for k, v in kwargs.items():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_json({"key": k, "value": v, "t": time.time()}),
                self._loop
            )


# global hub handle
hudiy_ws_hub = None  # type: HudiyDashboardHub | None

async def ensure_ws_hub_started(logger=None, ENABLE_LOGGING=True, backend=None, base_dir="/home/pi/scripts"):
    """
    Create/start the WS hub if needed (idempotent) and ensure ONE static web server
    is running with a backend-specific docroot (no HTML creation here).

    - Hudiy:    docroot = <base_dir>/hudiy_api/html_files
    - OpenAuto: docroot = <base_dir>/openauto_api
    """
    # 1) Static web server for the current backend
    try:
        b = (backend or "").strip().lower()
    except Exception:
        b = ""

    if b == "hudiy":
        docroot = os.path.join(base_dir, "hudiy_api")
    else:
        docroot = os.path.join(base_dir, "openauto_api")

    try:
        os.makedirs(docroot, exist_ok=True)
    except Exception as e:
        if ENABLE_LOGGING and logger:
            logger.warning("Could not ensure docroot (%s): %s", docroot, e)


    global hudiy_ws_hub
    if WS_SERVER is None:
        if ENABLE_LOGGING and logger:
            logger.error("Package 'websockets' nicht installiert. Run: pip install websockets")
        return
    if hudiy_ws_hub is None:
        hudiy_ws_hub = HudiyDashboardHub(logger=logger, enable_logging=ENABLE_LOGGING)
    if not hudiy_ws_hub.started:
        await hudiy_ws_hub.start()


# Ensure script is run with Python 3
if sys.version_info < (3, 0):
    logger.info("🚫 This script requires Python 3. Please run it using 'python3 script.py'")
    sys.exit(1)



async def handle_exception(exception, fallback_message=None):
    tb = traceback.extract_tb(exception.__traceback__)
    origin = tb[-1] if tb else None
    func_name = origin.name if origin else "unknown"
    line_number = origin.lineno if origin else 0
    error_message = f"{type(exception).__name__} at line {line_number} in {func_name}(): {exception}"
    exc_info_tuple = (type(exception), exception, exception.__traceback__)
    error_record = logger.makeRecord(name=logger.name, level=logging.ERROR,
                                     fn=origin.filename if origin else "<unknown>",
                                     lno=line_number, msg=error_message, args=(), exc_info=exc_info_tuple,
                                     func=func_name, extra=None)
    logger.handle(error_record)
    if fallback_message:
        info_record = logger.makeRecord(name=logger.name, level=logging.INFO,
                                        fn=origin.filename if origin else "<unknown>",
                                        lno=line_number + 1, msg=fallback_message, args=(), exc_info=None,
                                        func=func_name, extra=None)
        logger.handle(info_record)



def handle_errors(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        global loop
        try:
            loop = asyncio.get_running_loop()
            return await func(*args, **kwargs)
        except ImportError as e:
            await loop.run_in_executor(None, logger.error, f"ImportError in {func.__name__}: {e}", {"exc_info": True})
            return "import_error"
        except Exception as e:
            await loop.run_in_executor(None, logger.error, f"Exception occurred in {func.__name__}: {e}",
                                       {"exc_info": True})
            return "error"

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            logger.error(f"ImportError in {func.__name__}: {e}", exc_info=True)
            return "import_error"
        except Exception as e:
            logger.error(f"Exception occurred in {func.__name__}: {e}", exc_info=True)
            return "error"

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


@handle_errors
async def detect_installs():
    global openauto_ok, hudiy_ok, backend

    async def _read_text(path: Path):
        """Kleine Textdatei asynchron (threaded) lesen, ohne aiofiles."""
        if not path.is_file():
            return None
        try:
            return (await asyncio.to_thread(path.read_text, encoding="utf-8")).strip()
        except Exception:
            return None

    async def _read_version_from_log(path: Path, regex=r"version:\s*([0-9][\w.\-+]*)"):
        """Version per Regex aus Log holen, asynchron (threaded) und robustes Decoding."""
        if not path.is_file():
            return None
        try:
            data = await asyncio.to_thread(path.read_bytes)
            text = data.decode("utf-8", errors="ignore")
            matches = re.findall(regex, text or "", flags=re.IGNORECASE)
            return matches[-1] if matches else None
        except Exception:
            return None

    def _binary_ok(path: Path) -> bool:
        return path.is_file() and os.access(path, os.X_OK)

    home = Path.home()

    # HUDIY paths
    hudiy_folder_path = home / ".hudiy"
    hudiy_binary_path = hudiy_folder_path / "share" / "hudiy"
    hudiy_version_file_path = hudiy_folder_path / "share" / "version.txt"

    # OpenAuto paths
    openauto_folder_path = home / ".openauto"
    openauto_binary_path = Path("/usr/local/bin/autoapp")
    openauto_log_path = openauto_folder_path / "cache" / "openauto.log"

    # Existence checks
    hudiy_folder_found = hudiy_folder_path.is_dir()
    hudiy_binary_found = _binary_ok(hudiy_binary_path)
    openauto_folder_found = openauto_folder_path.is_dir()
    openauto_binary_found = _binary_ok(openauto_binary_path)

    if 'ENABLE_LOGGING' in globals() and ENABLE_LOGGING:
        logger.info("")
        logger.info("%s folder %s", hudiy_folder_path, "found" if hudiy_folder_found else "NOT found")
        logger.info("%s binary %s", hudiy_binary_path, "found and executable" if hudiy_binary_found else "NOT found or not executable")
        logger.info("%s folder %s", openauto_folder_path, "found" if openauto_folder_found else "NOT found")
        logger.info("%s binary %s", openauto_binary_path, "found and executable" if openauto_binary_found else "NOT found or not executable")

    # Versions (parallel ohne aiofiles)
    hudiy_version_task = _read_text(hudiy_version_file_path) if (hudiy_folder_found and hudiy_binary_found) else asyncio.sleep(0, result=None)
    openauto_version_task = _read_version_from_log(openauto_log_path) if (openauto_folder_found and openauto_binary_found) else asyncio.sleep(0, result=None)
    hudiy_version, openauto_version = await asyncio.gather(hudiy_version_task, openauto_version_task)

    hudiy_ok = hudiy_folder_found and hudiy_binary_found
    openauto_ok = openauto_folder_found and openauto_binary_found

    logger.info("")
    if hudiy_ok:
        global hudiy_ws_hub
        backend = "Hudiy"
        logger.info(f"{backend} found with version %s", hudiy_version or "unknown")
        hudiy_ws_hub = HudiyDashboardHub(logger=logger, enable_logging=ENABLE_LOGGING)
    elif openauto_ok:
        backend = "OpenAuto"
        logger.info(f"{backend} found with version %s", openauto_version or "unknown")
    else:
        logger.info(f"{backend} not found")
    logger.info("")

    if 'ENABLE_LOGGING' in globals() and ENABLE_LOGGING:
        logger.info(f"hudiy_ok: {hudiy_ok}")
        logger.info(f"openauto_ok: {openauto_ok}")

    return {
        "hudiy_ok": hudiy_ok,
        "hudiy_version": hudiy_version,
        "openauto_ok": openauto_ok,
        "openauto_version": openauto_version,
    }


@handle_errors
def is_python_too_old(min_version=(3, 13, 5)):
    return sys.version_info < min_version


@handle_errors
async def run_command(cmd: str, log_output: bool = False, check: bool = False) -> dict:
    """Run shell command, capture stdout/stderr, optional logging and check."""
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="ignore")
    err = (err_b or b"").decode(errors="ignore")

    if log_output and out:
        for line in out.strip().splitlines():
            logger.info(line)
    if log_output and err:
        for line in err.strip().splitlines():
            logger.warning(line)

    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{err or out}")

    return {"returncode": proc.returncode, "stdout": out, "stderr": err}


@handle_errors
def logger_prompt(level, func_name, lineno, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    level_str = level.upper().ljust(7)
    func_str = func_name.ljust(20)
    sys.stdout.write(f"{timestamp:<23} | {level_str:<7} | {lineno:<4} | {func_str:<21} | {message}")
    # header = f"{'TIME':<23} | {'LEVEL':<7} | {'LINE':<4} | {'FUNCTION':<21} | {'MESSAGE'}"
    sys.stdout.flush()
    return input()


# ---------------------------
# Installer für Python 3.11.2
# ---------------------------
@handle_errors
async def install_python_3_11_2():
    logger.info("🚀 Starting Python 3.11.2 installation (this may take 20–30 minutes)...")

    # Optional: störende Prozesse beenden (aus deinem 3.13.5-Code übernommen)
    logger.info("🔍 Checking if 'hudiy'/'autoapp' is running...")
    await run_command(
        "sudo pkill -f autoapp && echo '✅ autoapp was closed.' || echo 'ℹ️ autoapp was not active.'; "
        "sudo pkill -f hudiy && echo '✅ hudiy was closed.' || echo 'ℹ️ hudiy was not active.'",
        log_output=True
    )

    commands = [
        ("Updating package list...", "sudo apt update"),
        ("Installing build dependencies...", "sudo apt install -y build-essential libssl-dev zlib1g-dev "
                                             "libncurses5-dev libbz2-dev libreadline-dev libsqlite3-dev "
                                             "wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev "
                                             "libxmlsec1-dev libffi-dev liblzma-dev"),
        ("Downloading Python 3.11.2 source...",
         "cd /usr/src && sudo wget -q https://www.python.org/ftp/python/3.11.2/Python-3.11.2.tgz"),
        ("Extracting archive...", "cd /usr/src && sudo tar xzf Python-3.11.2.tgz"),
        ("Configuring build...", "cd /usr/src/Python-3.11.2 && sudo ./configure --enable-optimizations"),
        ("Compiling Python (this may take a while)...", "cd /usr/src/Python-3.11.2 && sudo make -j$(nproc)"),
        # altinstall -> installiert /usr/local/bin/python3.11 (nicht python3.11.2)
        ("Installing Python 3.11.2...", "cd /usr/src/Python-3.11.2 && sudo make altinstall"),
    ]

    for step_msg, cmd in commands:
        logger.info(f"➡️ {step_msg}")
        if not await run_command(cmd, log_output=False):
            logger.error("❌ Installation failed at step: %s", step_msg)
            sys.exit(1)

    logger.info("⬆️ Upgrading pip for Python 3.11...")
    # Nutze absoluten Pfad – unabhängig vom PATH/Alias/venv
    pip_upgrade_cmds = [
        "/usr/local/bin/python3.11 -m pip install --upgrade pip",
        "/usr/bin/python3.11 -m pip install --upgrade pip",
    ]
    ok = False
    for cmd in pip_upgrade_cmds:
        if await run_command(cmd, log_output=False):
            ok = True
            break
    if not ok:
        logger.warning("⚠️ Failed to upgrade pip for Python 3.11 (will continue).")
    else:
        logger.info("✅ pip for Python 3.11 upgraded successfully!")

    logger.info("🎉 Python 3.11.2 installed successfully!")

# ---------------------------
# Neustart unter 3.11 (robust)
# ---------------------------
@handle_errors
def restart_with_python_3_11():
    """
    Restart this script with python3.11, even if we're currently inside an old 3.7 venv.
    - Find python3.11 via PATH, common absolute paths, or a login-like shell.
    - Strip current venv from PATH and unset VIRTUAL_ENV/PYTHONHOME/PYTHONPATH.
    - Exec the current script under python3.11.
    """
    import shutil, subprocess, os, sys

    candidates = []

    # 1) PATH lookup (may fail if venv masks /usr/local/bin)
    p = shutil.which("python3.11")
    if p:
        candidates.append(p)

    # 2) Well-known locations after `make altinstall`
    for path in ("/usr/local/bin/python3.11", "/usr/bin/python3.11"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            candidates.append(path)

    # 3) Ask a login-like shell (in case PATH differs)
    if not candidates:
        try:
            proc = subprocess.run(
                ["bash", "-lc", "command -v python3.11 || true"],
                check=False, capture_output=True, text=True
            )
            shell_path = (proc.stdout or "").strip()
            if shell_path and os.path.isfile(shell_path) and os.access(shell_path, os.X_OK):
                candidates.append(shell_path)
        except Exception:
            pass

    if not candidates:
        logger.error(
            "⚠️ python3.11 not found. Likely installed to /usr/local/bin but masked by venv PATH.\n"
            "Add /usr/local/bin to PATH or call this script with /usr/local/bin/python3.11 explicitly."
        )
        sys.exit(1)

    python311 = candidates[0]

    # --- Clean environment from current venv influence ---
    env = os.environ.copy()

    # Drop venv bin path segments
    venv_dir = env.get("VIRTUAL_ENV")
    if venv_dir:
        venv_bin = os.path.join(venv_dir, "bin")
        path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if os.path.abspath(p) != os.path.abspath(venv_bin)]
        env["PATH"] = os.pathsep.join(path_parts)
        # Unset venv markers
        env.pop("VIRTUAL_ENV", None)

    # Also unset other python env that could pin to wrong stdlib/site-packages
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    # Ensure system bins are visible
    for extra in ("/usr/local/bin", "/usr/bin"):
        if extra not in env.get("PATH", ""):
            env["PATH"] = env.get("PATH", "") + (os.pathsep if env.get("PATH") else "") + extra

    logger.info(f"🔁 Restarting script using {python311} (venv detached)…")
    os.execvpe(python311, [python311] + sys.argv, env)


@handle_errors
def _find_python311() -> Optional[str]:
    import shutil, subprocess, os
    p = shutil.which("python3.11")
    if p:
        return p
    for path in ("/usr/local/bin/python3.11", "/usr/bin/python3.11"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    try:
        proc = subprocess.run(
            ["bash", "-lc", "command -v python3.11 || true"],
            check=False, capture_output=True, text=True
        )
        cand = (proc.stdout or "").strip()
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    except Exception:
        pass
    return None


@handle_errors
def _exec_clean(py_path: str):
    """Restart this script with py_path, removing effects of any active old venv."""
    import os
    env = os.environ.copy()
    venv_dir = env.get("VIRTUAL_ENV")
    if venv_dir:
        venv_bin = os.path.join(venv_dir, "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep)
            if os.path.abspath(p) != os.path.abspath(venv_bin)
        )
        env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    for extra in ("/usr/local/bin", "/usr/bin"):
        if extra not in env.get("PATH", ""):
            env["PATH"] = (env.get("PATH", "") + (os.pathsep if env.get("PATH") else "") + extra)
    os.execvpe(py_path, [py_path] + sys.argv, env)


@handle_errors
def restart_with_python_3_11():
    """Find python3.11 and exec into it; exit with error if not found."""
    py311 = _find_python311()
    if not py311:
        logger.error(
            "⚠️ python3.11 not found – ensure it exists (typically /usr/local/bin/python3.11). "
            "Also make sure /usr/local/bin is in PATH for login shells."
        )
        sys.exit(1)
    logger.info(f"🔁 Restarting script using {py311} …")
    _exec_clean(py311)

# ---------------------------
# Orchestrierung
# ---------------------------
@handle_errors
async def ensure_python_3_11_2():
    """
    Ensure current interpreter is >= 3.11.2.
    If older: install Python 3.11.2 from source and restart this process under python3.11.
    """
    target = (3, 11, 2)
    if sys.version_info >= target:
        if ENABLE_LOGGING:
            logger.info("✅ Python version is recent enough; no action required.")
        return

    # Wir sind zu alt -> optional Nachfrage (kannst du auch weglassen)
    try:
        from platform import python_version
    except Exception:
        python_version = lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    logger.warning(f"⚠️ Your current Python version is too old: {python_version()}")
    response = logger_prompt(
        level="warning",
        func_name=inspect.currentframe().f_code.co_name,
        lineno=inspect.currentframe().f_lineno + 1,
        message="❓ Do you want to automatically install Python 3.11.2 now? [y/N]: "
    ).strip().lower()

    if response != "y":
        logger.info("🚫 Script aborted. Please install Python ≥ 3.11.2 manually.")
        sys.exit(1)

    # Installieren
    await install_python_3_11_2()

    # Neu starten unter python3.11 (ohne Aliase/ohne alte venv)
    restart_with_python_3_11()  # no return


path = os.path.dirname(os.path.abspath(__file__))
script_filename = str(os.path.basename(__file__))
script_fullpath = os.path.realpath(__file__)

@handle_errors
async def start_script():
    logger.info("")
    logger.info(f"Script is starting...")
    logger.info("")
    python_path = sys.executable
    python_version = sys.version_info
    python_version_str = f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"

    loop = asyncio.get_event_loop()
    logger.info(f"Python interpreter: {python_version_str} ({python_path})")
    logger.info("Script version: %s (%s)", script_version, script_fullpath)
    logger.info(f"LOGGING_ENABLED: {ENABLE_LOGGING}")
    logger.info(f"Detected backend: {backend}")

    logger.info("")
    if ENABLE_LOGGING:

        config_vars = {
            "can_interface": can_interface,
            "welcome_message_1st_line": welcome_message_1st_line,
            "welcome_message_2nd_line": welcome_message_2nd_line,
            "send_on_canbus": send_on_canbus,
            "only_send_if_radio_is_in_tv_mode": only_send_if_radio_is_in_tv_mode,
            "activate_rnse_tv_input": activate_rnse_tv_input,
            "tv_input_format": tv_input_format,
            "show_label": show_label,
            "toggle_fis1": toggle_fis1,
            "toggle_fis2": toggle_fis2,
            "scroll_type": scroll_type,
            "read_and_set_time_from_dashboard": read_and_set_time_from_dashboard,
            "control_pi_by_rns_e_buttons": control_pi_by_rns_e_buttons,
            "read_mfsw_buttons": read_mfsw_buttons,
            "send_values_to_dashboard": send_values_to_dashboard,
            "toggle_values_by_rnse_longpress": toggle_values_by_rnse_longpress,
            "reversecamera_by_reversegear": reversecamera_by_reversegear,
            "reversecamera_by_down_longpress": reversecamera_by_down_longpress,
            "reversecamera_guidelines": reversecamera_guidelines,
            "reversecamera_turn_off_delay": reversecamera_turn_off_delay,
            "shutdown_by_ignition_off": shutdown_by_ignition_off,
            "shutdown_by_pulling_key": shutdown_by_pulling_key,
            "shutdown_type": shutdown_type,
            "initial_day_night_mode": initial_day_night_mode,
            "change_dark_mode_by_car_light": change_dark_mode_by_car_light,
            "send_api_mediadata_to_dashboard": send_api_mediadata_to_dashboard,
            "send_to_api_gauges": send_to_api_gauges,
            "lower_speed": lower_speed,
            "upper_speed": upper_speed,
            "export_speed_measurements_to_file": export_speed_measurements_to_file,
            "speed_unit": speed_unit,
            "temp_unit": temp_unit
        }

        logger.info("")
        logger.info("✅ Current Configuration:")
        for name, value in config_vars.items():
            logger.info("   • %s = %s", name, value)
        logger.info("")



server_socket = None
server_running = True

@handle_errors
async def remote_control(host='127.0.0.1', port=23456):
    global server_socket

    try:
        if server_socket:
            logger.info("Old server socket exists – closing...")
            server_socket.close()
            await server_socket.wait_closed()
            server_socket = None

        server = await asyncio.start_server(handle_client, host, port)
        server_socket = server
        if ENABLE_LOGGING:
            logger.info("")
            logger.info(f"✅ Remote control server started on {host}:{port}")
            logger.info("")


        # 🆕 Create an explicit server task.
        serve_task = asyncio.create_task(server.serve_forever())

        # Wait for shutdown signal
        await remote_server_shutdown_event.wait()
        logger.info("🛑 Shutdown signal received for remote_control")

        # Close server
        server.close()
        await server.wait_closed()

        # cancle server task
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            logger.info("✋ serve_forever task cancelled cleanly.")

    except asyncio.CancelledError:
        logger.info("⚠️ Task remote_control cancelled.")
        raise
    except Exception as e:
        logger.error(f"❌ Error in remote_control: {e}", exc_info=True)
    finally:
        if server_socket:
            try:
                server_socket.close()
                await server_socket.wait_closed()
                logger.info("⚠️ Remote server socket closed.")
            except Exception as e:
                logger.warning(f"❌ Error closing socket: {e}")
            server_socket = None
        remote_server_shutdown_event.clear()  # <– important!


@handle_errors
async def handle_client(reader, writer):
    global stop_flag
    try:
        data = await reader.read(1024)
        command = data.decode().strip()

        if command == 'stop_script':
            logger.info("Stop command received.")
            asyncio.create_task(stop_script())

        elif command == 'kill_script':
            logger.info("Kill command received.")
            asyncio.create_task(kill_script())

        else:
            logger.warning(f"Unknown command received: {command}")
            # Optional: still respond for unknown command
            # writer.write(b"Unknown command\n")
            # await writer.drain()

    except asyncio.CancelledError:
        logger.info("handle_client task stopped.")
    except Exception as e:
        logger.error(f"Error handling client: {e}", exc_info=True)
    finally:
        writer.close()
        await writer.wait_closed()


@handle_errors
async def get_other_pids(script_name: str):
    import psutil
    current_pid = os.getpid()
    parent_pid = os.getppid()
    other_pids = []


    for proc in psutil.process_iter(['pid', 'ppid', 'cmdline']):
        try:
            pid = proc.info['pid']
            ppid = proc.info['ppid']
            cmdline = proc.info['cmdline']

            # Eigene PID und Parent-PID ignorieren
            if pid in (current_pid, parent_pid):
                continue

            # Nur Python-Prozesse mit dem Skriptnamen
            if cmdline and script_name in " ".join(cmdline) and "python" in cmdline[0]:
                other_pids.append(pid)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return other_pids

@handle_errors
async def send_command(host, port, command):
    try:
        reader, writer = await asyncio.open_connection(host, port)
        logger.info(f"Sending command to {host}:{port}: {command}")
        writer.write(command.encode())
        await writer.drain()

    except Exception as e:
        logger.error("Error sending command", exc_info=True)
    finally:
        if 'writer' in locals():
            writer.close()
            await writer.wait_closed()


@handle_errors
async def wait_for_stop(script_name, timeout=10, kill_after_timeout=False):
    for sec in range(timeout):
        pids = await get_other_pids(script_name)
        if not pids:
            logger.info("Other running script instance has stopped.")
            return True
        logger.info(f"[{sec + 1}/{timeout}] Still running PIDs: {pids}")
        await asyncio.sleep(1)

    logger.warning(f"Script instance did not stop after {timeout} seconds.")

    if kill_after_timeout:
        pids = await get_other_pids(script_name)
        if pids:
            logger.warning(f"Forcing shutdown of stuck script instance(s): {pids}")
            for pid in pids:
                await run_command(f"kill -9 {pid}")
            await asyncio.sleep(1)
            return True
    return False


@handle_errors
async def stop_other_instance(script_name):
    pids = await get_other_pids(script_name)
    if not pids:
        if ENABLE_LOGGING:
            logger.info("✅ No other script instances running.")
        return False

    logger.info(f"⚠️ Detected other running script instance(s): {pids} → sending 'stop_script'")
    await send_command("localhost", 23456, "stop_script")

    if await wait_for_stop(script_name):
        logger.info("✅ Other running script instance(s) shut down gracefully.")
        logger.info("")
        return False

    logger.warning("⚠️ No response from other running script instance(s) – sending SIGTERM...")
    for pid in pids:
        await run_command(f"kill -15 {pid}")
    await asyncio.sleep(1)

    remaining = await get_other_pids(script_name)
    for pid in remaining:
        logger.warning(f"⚠️ PID {pid} still running – sending SIGKILL")
        await run_command(f"kill -9 {pid}")
    return True


@handle_errors
async def is_script_running(script_name):
    try:
        return await stop_other_instance(script_name)
    except Exception as e:
        logger.error("❌ Error checking for running script instances", exc_info=True)
        return False


@handle_errors
async def ensure_importlib(logger=None):
    global _cached_metadata
    if _cached_metadata:
        return _cached_metadata

    try:
        from importlib.metadata import version, PackageNotFoundError
        _cached_metadata = (version, PackageNotFoundError)
        return _cached_metadata
    except ImportError:
        pass

    # Check if the backport module is installed
    if importlib.util.find_spec("importlib_metadata") is None:
        if logger:
            logger.warning("⚠️ Backport 'importlib_metadata' missing. Attemping to install it...")
        else:
            logger.info("⚠️ Backport 'importlib_metadata' missing. Attemping to install it...")

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--user", "importlib-metadata",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if logger:
                logger.error(f"❌ Installation from importlib-metadata failed: {error_msg}")
            raise RuntimeError("❌ Installation from 'importlib-metadata' failed")

        if logger:
            logger.info("✅ importlib-metadata successfully installed.")

    # It should now be available
    try:
        from importlib_metadata import version, PackageNotFoundError
        _cached_metadata = (version, PackageNotFoundError)
        if logger and ENABLE_LOGGING:
            logger.info("Using importlib_metadata (backport)")
        return _cached_metadata
    except ImportError:
        raise RuntimeError("❌ Could not import 'importlib_metadata' after installation.")


@handle_errors
async def check_can_utils():
    from packaging import version
    import shutil
    import os

    if ENABLE_LOGGING:
        logger.info("")
        logger.info("🔍 Checking can-utils installation...")

    MIN_REQUIRED_VERSION = "v2025.01"

    def is_version_outdated(current: str, required: str) -> bool:
        try:
            if current.startswith("v") and "-" in current:
                current_clean = current.split("-")[0].lstrip("v")
            else:
                current_clean = current.lstrip("v")
            required_clean = required.lstrip("v")
            return version.parse(current_clean) < version.parse(required_clean)
        except Exception as e:
            logger.warning(f"⚠️ Version parsing failed: {e}")
            return True

    # --- Cron-kompatible Suche nach binaries ---
    CAN_UTILS_NAMES = ["candump", "cansend"]
    CAN_UTILS_PATHS = ["/usr/local/bin", "/usr/bin", "/bin", "/sbin", "/usr/sbin"]

    def find_can_util(bin_name):
        path = shutil.which(bin_name)
        if path:
            return path
        for p in CAN_UTILS_PATHS:
            candidate = os.path.join(p, bin_name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    candump_path = find_can_util("candump")
    cansend_path = find_can_util("cansend")
    candump_ok = candump_path is not None
    cansend_ok = cansend_path is not None

    if candump_ok and cansend_ok:
        if ENABLE_LOGGING:
            logger.info(f"✅ candump found at: {candump_path}")
            logger.info(f"✅ cansend found at: {cansend_path}")
    else:
        logger.info("ℹ️ candump and/or cansend not found in PATH. Will check/install can-utils.")

    current_version = None
    needs_upgrade = True

    # --- 1. Check APT installation ---
    version_output = await run_command("dpkg -s can-utils | grep Version", log_output=False)
    if version_output["stdout"]:
        try:
            current_version = version_output["stdout"].split(":")[1].strip()
            logger.warning(f"✅ can-utils installed via APT. Current version: {current_version}")
            logger.info("🧹 Removing APT version of can-utils...")
            await run_command("sudo apt remove -y can-utils", log_output=False)
        except Exception:
            logger.warning("⚠️ Unexpected format in version output. Proceeding with upgrade.")
    elif os.path.isdir("/usr/src/can-utils/.git"):
        git_version = await run_command("cd /usr/src/can-utils && git describe --tags", log_output=False)
        if git_version["stdout"]:
            current_version = git_version["stdout"].strip()
            if ENABLE_LOGGING:
                logger.info(f"✅ can-utils version: {current_version}")
                logger.info("")
            if candump_ok and cansend_ok:
                if ENABLE_LOGGING:
                    logger.info(f"✅ can-utils installed from Git. Current version: {current_version}")
                if not is_version_outdated(current_version, MIN_REQUIRED_VERSION):
                    if ENABLE_LOGGING:
                        logger.info("✅ Git version is up-to-date and binaries are present. No installation required.")
                    needs_upgrade = False
                else:
                    logger.warning("⚠️ Git version is outdated. Reinstallation required.")
            else:
                logger.warning("⚠️ Git repo exists, but binaries missing. Will reinstall.")
        else:
            logger.warning("⚠️ Git repo exists, but version info missing. Will reinstall.")
    else:
        logger.warning("ℹ️ can-utils not found. Proceeding with installation.")

    # --- 2. Installation via Git/CMake falls nötig ---
    if needs_upgrade:
        logger.info(f"⬇️ Installing latest version from GitHub via CMake... (target: {MIN_REQUIRED_VERSION})")

        await run_command("sudo rm -rf /usr/src/can-utils", log_output=False)
        await run_command("cd /usr/src && sudo git clone https://github.com/linux-can/can-utils.git", log_output=False)
        await run_command("git config --global --add safe.directory /usr/src/can-utils", log_output=False)

        await run_command("sudo apt install -y cmake build-essential", log_output=False)

        await run_command("sudo mkdir -p /usr/src/can-utils/build", log_output=False)
        await run_command("cd /usr/src/can-utils/build && sudo cmake ..", log_output=False)
        await run_command("cd /usr/src/can-utils/build && sudo make", log_output=False)
        await run_command("cd /usr/src/can-utils/build && sudo make install", log_output=False)

        new_version = await run_command("cd /usr/src/can-utils && git describe --tags", log_output=False)
        if new_version["stdout"]:
            logger.info(f"✅ can-utils successfully installed. New version: {new_version['stdout'].strip()}")
            logger.info("")
        else:
            logger.warning("⚠️ can-utils installed, but version could not be verified.")




@handle_errors
def pep668_active() -> bool:
    """Debian/Bookworm: 'externally-managed environment' aktiv?"""
    platlib = sysconfig.get_paths().get('platlib') or ''
    base = platlib.split("/site-packages")[0]
    return os.path.exists(os.path.join(base, "EXTERNALLY-MANAGED"))


async def ensure_packaging(logger=None, enable_logging=True):
    """
    Stellt asynchron sicher, dass 'packaging' verfügbar ist.
    Gibt (Requirement, Version) zurück.
    """
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        return Requirement, Version
    except ImportError:
        if enable_logging and logger: logger.info("'packaging' missing – installing via pip…")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--upgrade", "packaging",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"pip install packaging failed: {err.decode(errors='ignore').strip()}")
        if enable_logging and logger: logger.info("✔ 'packaging' installed")
        from packaging.requirements import Requirement
        from packaging.version import Version
        return Requirement, Version


async def packaging_globals(logger=None, enable_logging=True):
    """Lädt Requirement/Version und schreibt sie in globals()."""
    Requirement, Version = await ensure_packaging(logger, enable_logging)
    globals()["Requirement"] = Requirement
    globals()["Version"] = Version
    if enable_logging and logger: logger.info("🔧 packaging globals ready: Requirement, Version")



@handle_errors
async def _pip_install(python_exe: str, packages: List[str]) -> bool:
    if not packages:
        return True

    # Install
    cmd = f'"{python_exe}" -m pip install ' + " ".join(packages)
    res = await run_command(cmd, log_output=False)
    if res["returncode"] != 0:
        logger.error("❌ pip install failed: %s", (res["stderr"] or res["stdout"]).strip())
        return False

    # Versionsfunktion holen
    version, PackageNotFoundError = await ensure_importlib(logger)

    # Paketnamen robust aus Requirement-Strings extrahieren
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None  # sollte durch deine packaging-first-Installation vorhanden sein

    logger.info("")
    logger.info("✅ pip packages installed:")
    for spec in packages:
        # Name aus Spec holen (handles ~=, ==, extras, etc.)
        if Requirement is not None:
            try:
                name = Requirement(spec).name
            except Exception:
                name = spec
        else:
            # grober Fallback, falls packaging wider Erwarten fehlt
            name = (
                spec.split("==")[0]
                    .split("~=")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split(">")[0]
                    .split("<")[0]
                    .strip()
            )

        try:
            ver = version(name)  # echte installierte Version als String
            logger.info("    • %s %s", name, ver)
        except PackageNotFoundError:
            # sehr selten (z.B. exotische Direct-References ohne Name)
            logger.info("    • %s (version unknown)", name)

    return True


@handle_errors
async def _apt_install(packages: List[str]) -> bool:

    if not packages:
        return True
    await run_command("sudo apt update -y", log_output=False)
    res = await run_command("sudo apt install -y " + " ".join(packages), log_output=False)
    if res["returncode"] != 0:
        logger.error("❌ apt install failed: %s", (res["stderr"] or res["stdout"]).strip())
        return False

    logger.info("")
    logger.info("✅ apt packages installed:")
    for pkg in packages:
        logger.info("    • %s (system package, version not checked)", pkg.strip())
    return True


@handle_errors
async def python_modules() -> Tuple[Dict[str, str], List[str]]:
    """
    Scannt die 'required' Module (inkl. backend-spezifischer) und liefert:
      - installed_modules: {name -> version}
      - missing_modules:   ["pkg", "pkg~=x.y", ...] (inkl. Versionsanforderung)
    """
    from packaging.version import Version
    from packaging.requirements import Requirement

    version, PackageNotFoundError = await ensure_importlib(logger)


    base_required = [
        "aioconsole",
        "aiofiles",
        "requests",
        "protobuf~=3.19",   # >=3.19,<4
        "google",
        "python-can",
        "psutil",
        "python-uinput",
        "packaging",
    ]

    required = list(base_required)

    backend_norm = (backend or "").strip().lower()
    if backend_norm == "hudiy":
        required += ["websockets", "websocket-client~=1.8"]
    if backend_norm == "openauto":
        # Wir verwenden pypng statt Pillow
        required += ["pypng"]

    # Deduplizieren unter Beibehaltung der Reihenfolge
    seen = set()
    required = [r for r in required if not (r in seen or seen.add(r))]

    installed_modules: Dict[str, str] = {}
    missing_modules: List[str] = []

    for req_str in required:

        req = Requirement(req_str)
        name = req.name
        try:
            inst_ver = version(name)
            installed_modules[name] = inst_ver

            if req.specifier and (Version(inst_ver) not in req.specifier):
                missing_modules.append(str(req))
        except PackageNotFoundError:
            missing_modules.append(str(req))

    return installed_modules, missing_modules


@handle_errors
async def install_missing(missing_modules: List[str],
                          installed_modules: Optional[dict] = None) -> Dict[str, str]:
    """
    Pip-only installer:
      - Installs every item in `missing_modules` using pip in the current interpreter/venv.
      - No apt calls. No warnings. No special-casing.
      - Extends/returns `installed_modules` with resolved versions.
    """
    version, PackageNotFoundError = await ensure_importlib(logger)

    if installed_modules is None:
        installed_modules = {}

    missing_modules = [m.strip() for m in (missing_modules or []) if m and m.strip()]
    if not missing_modules:
        return installed_modules

    ok = await _pip_install(sys.executable, missing_modules)
    if not ok:
        sys.exit(1)

    for spec in missing_modules:
        base = spec.split("==")[0].strip()
        try:
            installed_modules[base] = version(base)
        except PackageNotFoundError:
            installed_modules[base] = "(pip-installed)"

    return installed_modules



@handle_errors
async def modules_inst_import():
    """
    1) scan -> 2) ggf. installieren -> 3) erneut scannen -> 4) importieren -> 5) EIN Block Logging.
    Verhindert doppelte 'Successfully imported modules:' Header.
    """
    installed, missing = await python_modules()

    if missing:
        if ENABLE_LOGGING:
            logger.warning("")
            logger.warning("⚠️  Missing modules (fetched for install):")
            for m in missing:
                logger.warning("   • %s", m)

        # Installiere fehlende Module (deine bestehende Funktion)
        await install_missing(missing)

        # Danach NOCHMAL scannen, damit auch frisch installierte (z. B. pypng) gelistet/importiert werden
        installed, missing_after = await python_modules()
        if missing_after:
            # Falls etwas immer noch fehlt, einmalig warnen
            if ENABLE_LOGGING:
                logger.warning("")
                logger.warning("⚠️  Still missing after install:")
                for m in missing_after:
                    logger.warning("   • %s", m)

    # Jetzt genau EINMAL importieren und loggen
    import_modules(installed)

# ---- Aliases für die WS-Bibliotheken (nicht überschreiben!) ----
WS_SERVER = None   # für 'websockets' (Server)
WS_CLIENT = None   # für 'websocket' aus 'websocket-client' (Client)

@handle_errors
def import_modules(installed_modules: Dict[str, str]) -> None:
    """
    Importiert die gelisteten Module (Mapping: Paketname -> Version) und loggt diese EINMAL gesammelt.
    """
    if ENABLE_LOGGING:
        logger.info("")
        logger.info("✅ Successfully imported modules:")

    for mod, ver in installed_modules.items():
        try:
            # Paketname -> Import-Name mappen
            if mod == "aiofiles":
                global aiofiles; import aiofiles
            elif mod == "requests":
                global requests; import requests
            elif mod == "python-can":
                global can, Notifier; import can; from can import Notifier
            elif mod == "psutil":
                global psutil; import psutil
            elif mod == "google":
                global google; import google
            elif mod == "protobuf":
                global protobuf; import google.protobuf as protobuf
            elif mod == "aioconsole":
                global aioconsole; import aioconsole
            elif mod == "python-uinput":
                global uinput; import uinput
            elif mod == "packaging":
                global packaging; import packaging

            # Backend-spezifische Mappings
            backend_norm = (backend or "").strip().lower()
            if backend_norm == "openauto":
                if mod == "pypng":
                    global png; import png
            elif backend_norm == "hudiy":
                if mod == "websockets":
                    global WS_SERVER; import websockets as _ws; WS_SERVER = _ws
                elif mod == "websocket-client":
                    global WS_CLIENT; import websocket as _wsc; WS_CLIENT = _wsc

            if ENABLE_LOGGING:
                logger.info("   • %s %s", mod, ver)

        except ImportError as e:
            logger.warning("⚠️  Failed to import %s (installed version: %s): %s", mod, ver, e)

    if ENABLE_LOGGING:
        logger.info("")


@handle_errors
async def uinput_permissions():
    if control_pi_by_rns_e_buttons:
        try:
            result = await run_command("stat /dev/uinput", log_output=False)
            if "0666" not in result["stdout"]:
                logger.warning("⚠️ Permissions for /dev/uinput are incorrect.")
                logger.info("Setting correct permissions...")
                await run_command("sudo modprobe uinput", log_output=False)
                await run_command("sudo chmod 666 /dev/uinput", log_output=False)
                result = await run_command("stat /dev/uinput", log_output=False)
                if "0666" in result["stdout"]:
                    logger.info("✅ Permissions successfully set.")
                    await import_uinput()
                else:
                    logger.error("❌ Failed to set permissions for /dev/uinput.")
                    return False
            else:
                if ENABLE_LOGGING:
                    logger.info("")
                    logger.info("✅ Permissions for /dev/uinput are correct.")
                    logger.info("")
                await import_uinput()
        except Exception as error:
            await handle_exception(error, "❌ Couldn't check uinput permissions.")


# Check if the raspberry pi is in powersave mode (pi will stick at 600MHz frequency).
# If that is the case, set the powermode/scaling mode to "ondemand" so the cpu can change its frequency dynamicly.
# To change this permanently, you can add "cpufreq.default_governor=ondemand" at the end of the file "/boot/cmdline.txt"

@handle_errors
async def set_powerplan():
    governor_path = "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
    target_governor = "ondemand"
    try:
        async with aiofiles.open(governor_path, mode="r") as f:
            current_governor = (await f.read()).strip()
        if current_governor == "powersave":
            proc = await asyncio.create_subprocess_exec(
                "sudo", "tee", governor_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate(input=target_governor.encode())
            if proc.returncode == 0:
                logger.info(f"Powersave mode detected. Changing from '{current_governor}' to '{target_governor}' "
                            f"(dynamic CPU frequency mode).")
            else:
                logger.error(f"Failed to set governor: {stderr.decode().strip()}")
    except Exception as e:
        logger.error("Error while checking/setting power plan to 'ondemand':", exc_info=True)

@handle_errors
async def import_uinput():
    global events, device, uinput
    try:
        events = (
            uinput.KEY_1, uinput.KEY_2, uinput.KEY_UP, uinput.KEY_DOWN, uinput.KEY_LEFT, uinput.KEY_RIGHT,
            uinput.KEY_ENTER, uinput.KEY_ESC, uinput.KEY_F2, uinput.KEY_B, uinput.KEY_N, uinput.KEY_V,
            uinput.KEY_F12, uinput.KEY_M, uinput.KEY_X, uinput.KEY_C, uinput.KEY_LEFTCTRL, uinput.KEY_H, uinput.KEY_T,
            uinput.KEY_O
        )
        device = uinput.Device(events)
    except ImportError as error:
        await handle_exception(error, "Couldn't import uinput.")
    except Exception as error:
        await handle_exception(error, "An error occurred while initializing uinput.")


can_filters = [dict(can_id=0x271, can_mask=0x7FF, extended=False),
               dict(can_id=0x2C3, can_mask=0x7FF, extended=False),
               dict(can_id=0x351, can_mask=0x7FF, extended=False),
               dict(can_id=0x353, can_mask=0x7FF, extended=False),
               dict(can_id=0x35B, can_mask=0x7FF, extended=False),
               dict(can_id=0x461, can_mask=0x7FF, extended=False),
               dict(can_id=0x5C3, can_mask=0x7FF, extended=False),
               dict(can_id=0x602, can_mask=0x7FF, extended=False),
               dict(can_id=0x623, can_mask=0x7FF, extended=False),
               dict(can_id=0x635, can_mask=0x7FF, extended=False),
               dict(can_id=0x65F, can_mask=0x7FF, extended=False),
               dict(can_id=0x661, can_mask=0x7FF, extended=False)]


REQUIRED_LINES = [
    "dtparam=spi=on",
    "dtoverlay=mcp2515-can0,oscillator=16000000,interrupt=25",
    "dtoverlay=spi-bcm2835-overlay"
]


@handle_errors
async def check_pican2_3_config():

    # Detect config path
    try:
        with open("/etc/os-release") as f:
            codename = next((l.split("=")[1].strip().strip('"') for l in f if l.startswith("VERSION_CODENAME=")), None)
    except:
        codename = None
    logger.info(f"Raspberry Pi OS: {codename}")
    if codename in ("bookworm", "trixie"):
        CONFIG_FILE = "/boot/firmware/config.txt"
    else:
        CONFIG_FILE = "/boot/config.txt"

    if not os.path.exists(CONFIG_FILE):
        CONFIG_FILE = "/boot/firmware/config.txt" if os.path.exists("/boot/firmware/config.txt") else "/boot/config.txt"

    if not os.path.exists(CONFIG_FILE):
        logger.error("❌ Config file not found! Expected /boot/config.txt or /boot/firmware/config.txt")
        return

    # Read config
    result = await run_command(f"sudo cat {CONFIG_FILE}", log_output=False)
    if result["returncode"] != 0:
        logger.error(f"❌ Error reading {CONFIG_FILE}: {result['stderr']}")
        return

    config_lines = set(line.strip() for line in result["stdout"].splitlines() if line.strip())
    missing = [line for line in REQUIRED_LINES if line not in config_lines]
    if not missing:
        logger.info(f"✅ PiCAN2/3 config is correct in {CONFIG_FILE}")
        return

    logger.warning(f"⚠️ Missing lines in {CONFIG_FILE}:")
    for line in missing:
        logger.warning(f"  ➤ {line}")

    if (await aioconsole.ainput("Add these lines now? (yes/no): ")).strip().lower() not in ("yes", "y"):
        logger.warning("❌ Aborted.")
        return

    # Backup with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup = f"{CONFIG_FILE}.bak_{timestamp}"
    if (await run_command(f"sudo cp -a {CONFIG_FILE} {backup}", log_output=False))["returncode"] == 0:
        logger.info(f"🧰 Backup created: {backup}")
    else:
        logger.warning("⚠️ Could not create backup.")

    # Append missing lines
    newline = "\n"
    cmd = f"sudo bash -c 'echo -e \"{newline}#PiCAN2/3 Settings{newline}{newline.join(missing)}\" >> {CONFIG_FILE}'"
    if (await run_command(cmd, log_output=False))["returncode"] == 0:
        logger.info(f"✅ Missing lines added to {CONFIG_FILE}")
    else:
        logger.error(f"❌ Failed to write to {CONFIG_FILE}")
        return

    if (await aioconsole.ainput("Reboot now? (yes/no): ")).strip().lower() in ("yes", "y"):
        logger.info("🔄 Rebooting...")
        await run_command("sudo reboot", log_output=False)
    else:
        logger.warning("ℹ️ Please reboot manually.")



@handle_errors
async def test_can_interface():
    """Tests and initializes the CAN interface.."""
    global bus, send_on_canbus, can_functional
    bus = None  # Ensure that bus is initialized
    can_functional = False

    try:
        if can_interface == 'vcan0':
            result = await run_command("ip link show vcan0", log_output=False)
            if result["stderr"]:
                logger.warning("⚠️ vcan0 does not exist. Creating vcan0...")
                result = await run_command("sudo ip link add dev vcan0 type vcan", log_output=False)
                if result["stderr"]:
                    logger.error(f"❌ Failed to create vcan0: {result['stderr']}")
                    return
            logger.info("✅ vcan0 created successfully.")
            result = await run_command("sudo ip link set up vcan0", log_output=False)
            if result["stderr"]:
                logger.error(f"❌ Failed to bring vcan0 up: {result['stderr']}")
                return
            logger.info("✅ vcan0 interface is up.")
            try:
                bus = can.interface.Bus(
                    can_interface,
                    interface='socketcan',
                    bitrate=100000,
                    can_filters=can_filters,
                    receive_own_messages=False
                )
                logger.info("✅ CAN-Interface 'vcan0' found and opened.")
                result = await run_command(f'sudo ifconfig {can_interface} txqueuelen 1000', log_output=False)
                if result["stderr"]:
                    logger.error(f"❌ Failed to set txqueuelen for vcan0: {result['stderr']}")
                    send_on_canbus = False
                    return
            except can.CanError as e:
                logger.error(f"❌ Failed to initialize CAN-Bus on {can_interface}. Error: {e}")
                return
        else:
            if not os.path.exists(f'/sys/class/net/{can_interface}/operstate'):
                logger.warning(f"⚠️ Interface {can_interface} does not exist. Maybe it was not installed properly?")
                await check_pican2_3_config()
                return
            async with aiofiles.open(f'/sys/class/net/{can_interface}/operstate', mode='r') as f:
                can_network_state = (await f.read()).strip()
            if can_network_state != 'up':
                logger.warning(f"⚠️ {can_interface} is down, trying to bring it up...")
                result = await run_command(
                    f'sudo /sbin/ip link set {can_interface} up type can restart-ms 1000 bitrate 100000',
                    log_output=False
                )
                if result["stderr"]:
                    logger.error(f"❌ Failed to bring {can_interface} up: {result['stderr']}")
                    return
                result = await run_command(f'sudo ifconfig {can_interface} txqueuelen 1000', log_output=False)
                if result["stderr"]:
                    logger.error(f"❌ Failed to set txqueuelen for {can_interface}: {result['stderr']}")
                    return
            try:
                bus = can.interface.Bus(
                    can_interface,
                    interface='socketcan',
                    bitrate=100000,
                    can_filters=can_filters,
                    receive_own_messages=False
                )
                logger.info(f"✅ CAN-Interface '{can_interface}' found and opened.")
            except can.CanError as e:
                logger.error(f"❌ Failed to initialize CAN-Bus on {can_interface}. Error: {e}")
                return
        if bus is not None:
            received_message = bus.recv(timeout=1.0)  # Timeout in seconds (non-blocking)
            if received_message is not None or can_interface == 'vcan0':
                logger.info("✅ CAN message received. CAN-Bus seems to be working.")
                logger.info("")
                can_functional = True
            else:
                logger.warning("⚠️ No CAN message received. Disabling CAN-Bus communication.")
        else:
            logger.error("❌ CAN-Bus initialization failed, no bus object available.")
    except FileNotFoundError:
        logger.error("File not found!", exc_info=True)
    except Exception as e:
        logger.error(f"❌ Error while testing the CAN interface: {e}", exc_info=True)
    finally:
        if not can_functional and bus is None:
            logger.error("❌ Failed to initialize CAN-Bus. Disabling CAN-Bus features.")
            send_on_canbus = False


@handle_errors
async def check_camera():
    """
    Check if the legacy camera stack is enabled and a camera is detected.
    Uses `vcgencmd get_camera` and interprets:
      - 'supported=1 detected=1' → OK
      - 'supported=1 detected=0' → enabled but no camera detected
      - otherwise               → likely disabled in raspi-config
    """
    global reversecamera_by_reversegear, reversecamera_by_down_longpress
    try:
        process = await asyncio.create_subprocess_exec(
            "vcgencmd", "get_camera",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error("Error while checking camera status: %s", (stderr or b"").decode().strip())
            return False

        result = (stdout or b"").decode().strip()
        if "supported=1 detected=1" in result:
            logger.info("Camera is activated and detected.")
            return True
        elif "supported=1 detected=0" in result:
            logger.warning("Camera is activated but NOT detected.")
            reversecamera_by_reversegear = False
            reversecamera_by_down_longpress = False
            return False
        else:
            logger.warning("Camera appears disabled in Raspberry Pi Config (raspi-config). Please enable the legacy camera.")
            return False
    except Exception:
        logger.error("Error while checking Camera status.", exc_info=True)
        return False

cam = None               # Singleton-Instanz
CAM_WITH_OVERLAY = False # wird bei init gesetzt

@handle_errors
def cam_init(reversecamera_guidelines: bool, overlay_png="/home/pi/overlay.png"):
    """Einmaliger Warmstart: mit Overlay, wenn reversecamera_guidelines=True (und PNG existiert)."""
    global cam, CAM_WITH_OVERLAY
    CAM_WITH_OVERLAY = bool(reversecamera_guidelines) and os.path.isfile(overlay_png)

    cam = Cam(
        overlay_png=(overlay_png if CAM_WITH_OVERLAY else None),
        device="/dev/video0",
        display=":0",
        width=800, height=480, fps=30,
        input_format="yuyv422"   # ggf. "mjpeg"
    )

    warm_variant = "overlay" if CAM_WITH_OVERLAY else "base"
    cam.start(visible=False, warm_variant=warm_variant)


@handle_errors
async def send_can_message(arb_id, data_content):
    def get_call_chain(max_depth=5, stop_at_system=True):
        stack = inspect.stack()
        call_chain = []

        for frame in stack[2:2 + max_depth]:
            func = frame.function
            lineno = frame.lineno

            # Stop when we reach the framework
            if stop_at_system and func in {
                "_run", "run_forever", "run_until_complete", "<module>", "_run_once"
            }:
                break

            call_chain.append(f"{func} (line {lineno})")

        return " ← ".join(call_chain)

    try:
        message = can.Message(arbitration_id=arb_id, data=data_content, is_extended_id=False)
        loop = get_running_loop()
        if send_on_canbus and can_functional:
            await loop.run_in_executor(None, bus.send, message)
            if ENABLE_LOGGING and show_can_messages_in_logs:
                data_hex = "".join(f"{byte:02X}" for byte in data_content)
                caller_info = get_call_chain()
                logger.info(
                    f"CAN-Message with CAN-ID: {arb_id:X} Message: {data_hex} sent ← from {caller_info}"
                )
    except can.CanError as e:
        logger.error(f"Failed to send CAN message: {e}")

@handle_errors
async def overwrite_dis():
    global send_on_canbus, deactivate_overwrite_dis_content, bus, task_overwrite_dis_content

    if not bus:
        logger.error("❌ CAN-Bus is not initialized. Aborting overwrite_dis.")
        return

    msg_activate = can.Message(arbitration_id=0x665, data=[0x03, 0x00], is_extended_id=False)
    msg_deactivate = can.Message(arbitration_id=0x665, data=[0x01, 0x00], is_extended_id=False)

    try:
        task_overwrite_dis_content = bus.send_periodic(msg_activate, 1.00)
        if task_overwrite_dis_content is None:
            logger.error("❌ Failed to start periodic message for overwrite_dis.")
            return

        if ENABLE_LOGGING:
            logger.info("🔁 Started periodic overwrite_dis task.")

        while not stop_flag:
            # State check: are we allowed to send?
            if only_send_if_radio_is_in_tv_mode:
                allow_send = toggle_fis1 != 0 and toggle_fis2 != 0 and tv_mode_active != 0
            else:
                allow_send = toggle_fis1 != 0 and toggle_fis2 != 0

            # Set state on change
            if allow_send and not send_on_canbus:
                send_on_canbus = True
                deactivate_overwrite_dis_content = False
                logger.info("✅ CAN sending re-enabled (toggle conditions met).")
            elif not allow_send and send_on_canbus:
                send_on_canbus = False
                deactivate_overwrite_dis_content = True
                logger.info("🚫 CAN sending disabled (toggle conditions unmet).")

            # change/send dates
            if not deactivate_overwrite_dis_content:
                task_overwrite_dis_content.modify_data([msg_activate])
            elif deactivate_overwrite_dis_content and (not pause_fis1 or not pause_fis2):
                task_overwrite_dis_content.modify_data([msg_deactivate])

            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        if ENABLE_LOGGING:
            logger.info("🛑 overwrite_dis task was cancelled.")
    except Exception as e:
        logger.error(f"🔥 Error in overwrite_dis: {e}", exc_info=True)



tv_input_task = None


@handle_errors
async def send_tv_input():
    global tv_input_task

    if send_on_canbus and can_functional and activate_rnse_tv_input:
        base_data = [0x12, 0x31, 0x41, 0x56, 0x20, 0x31]
        tv_format_prefix = 0x89 if tv_input_format == "NTSC" else 0x81
        msg = can.Message(
            arbitration_id=0x602,
            data=[tv_format_prefix] + base_data,
            is_extended_id=False
        )
        try:
            tv_input_task = bus.send_periodic(msg, 0.50)  # Starts periodic sending
            logger.info("Started periodic TV input activation message.")
        except asyncio.CancelledError:
            if ENABLE_LOGGING:
                logger.info("task send_tv_input was stopped.")
        except can.CanError as e:
            logger.error(f"Error while sending TV input message: {e}")


@handle_errors
async def align_center(fis='', content=''):
    if len(content) > 8:
        content = content[:8]
    content = content.encode('iso-8859-1', errors='ignore').hex().upper()
    content = await convert_audi_ascii(content)
    length = len(content)
    if length < 16:
        content = '2020202020202020'[:16 - length] + content
    content = list(bytearray.fromhex(content))
    fis = int(fis, 16)
    await send_can_message(fis, content)


@handle_errors
async def align_right(fis='', content=''):
    if len(content) > 8:
        content = content[:8]
    content = content.encode('iso-8859-1', errors='ignore').hex().upper()
    content = await convert_audi_ascii(content)
    length = len(content)
    if length < 16:
        content = '6565656565656565'[:16 - length] + content
    content = list(bytearray.fromhex(content))
    fis = int(fis, 16)
    await send_can_message(fis, content)


@handle_errors
async def align_left(fis='', content=''):
    if len(content) > 8:
        content = content[:8]
    content = content.encode('iso-8859-1', errors='ignore').hex().upper()
    content = await convert_audi_ascii(content)
    length = len(content)
    if length < 16:
        content = content + '6565656565656565'[:16 - length]
    content = list(bytearray.fromhex(content))
    fis = int(fis, 16)
    await send_can_message(fis, content)


@handle_errors
async def clear_content(FIS):
    clear1 = '6565656565656565'
    clear1 = list(bytearray.fromhex(clear1))
    fis = int(FIS, 16)
    await send_can_message(fis, clear1)


HEX_TO_AUDI_ASCII = {
    '61': '01', '62': '02', '63': '03', '64': '04', '65': '05',
    '66': '06', '67': '07', '68': '08', '69': '09', '6A': '0A',
    '6B': '0B', '6C': '0C', '6D': '0D', '6E': '0E', '6F': '0F',
    '70': '10', 'B0': 'BB', 'E4': '91', 'F6': '97', 'FC': '99',
    'C4': '5F', 'D6': '60', 'DC': '61', 'DF': '8D', '5F': '66',
    'A3': 'AA', 'A7': 'BF', 'A9': 'A2', 'B1': 'B4', 'B5': 'B8',
    'B9': 'B1', 'BA': 'BB', 'C8': '83', 'E8': '83',
    # Add more mappings here...
    # Default value for space character
    "20": "65"
}

FORMULA_TO_KEY = {
    "getPidValue(4)": "cpu_load",
    "getPidValue(5)": "cpu_temp",
    "getPidValue(6)": "cpu_freq_mhz",
    "getPidValue(0)": "speed",
    "getPidValue(7)": "speed_measure",
    "getPidValue(3)": "outside_temp",
    "getPidValue(1)": "rpm",
    "getPidValue(2)": "coolant",
}


@handle_errors
async def convert_audi_ascii(content=''):
    return ''.join(HEX_TO_AUDI_ASCII.get(content[i:i + 2], content[i:i + 2]) for i in range(0, len(content), 2))


@handle_errors
async def welcome_message():
    global script_started

    if send_on_canbus and can_functional and not script_started:
        await align_center(FIS1, welcome_message_1st_line)
        await align_center(FIS2, welcome_message_2nd_line)
        await asyncio.sleep(3)
        script_started = True
        await clear_content(FIS1)
        await clear_content(FIS2)
        if show_label:
            await align_center(FIS1, value_of_toggle_fis2)

# ------------------------------------------------------------
# EventHandler – Subscriptions nur EINMAL pro Verbindung (Guard)
# ------------------------------------------------------------

def define_event_handler_class():
    class EventHandler(ClientEventHandler):
        def __init__(self, client, main_loop):
            self.client = client
            self.main_loop = main_loop
            self.read_cpu_task = None
            self._subs_sent = False  # ✅ Guard pro Verbindung
            super().__init__()
            global openauto_ok, hudiy_ok

        # Optional: an passender Stelle in __init__ o.Ä.:
        # self._is_day_mode = None  # unbekannt beim Start
        @handle_errors
        def update_color_scheme_from_hudiy(self, dark_enabled: bool) -> None:
            """
            Kannst du aus deinem Message-Handler aufrufen, wenn Hudiy ein
            ColorScheme-Update liefert. Dann kennen wir den aktuellen Modus.
            dark_enabled=True  -> Night-Mode
            dark_enabled=False -> Day-Mode
            """
            try:
                self._is_day_mode = not bool(dark_enabled)
                if ENABLE_LOGGING:
                    logger.info("Color scheme update: is_day_mode=%s", self._is_day_mode)
            except Exception as exc:
                if ENABLE_LOGGING:
                    logger.warning("Failed to apply color scheme update: %s", exc)

        @handle_errors
        def _resolve_day_mode_for_toggle(self) -> bool:

            if getattr(self, "_is_day_mode", None) is not None:
                return self._is_day_mode

            # If you want an initial system/app default, change this to False for Night.
            return True

        @handle_errors
        def on_hello_response(self, client, message):
            global openauto_ok, hudiy_ok, backend, speed, outside_temp, rpm, coolant, cpu_temp, cpu_load, cpu_freq_mhz
            self.client = client  # ✅ aktiven Client übernehmen

            logger.info("")
            if backend == "OpenAuto":
                logger.info(f"✅ Received hello response from {backend} API")
                logger.info(f"openauto version: {message.oap_version.major}.{message.oap_version.minor}")
                logger.info(f"api version: {message.api_version.major}.{message.api_version.minor}")
                if message.api_version.minor == 1:
                    logger.warning("⚠️  API reports version 1.1, but GitHub release claims 1.2. Possibly outdated constant in proto file.")
                logger.info("")
            elif backend == "Hudiy":
                logger.info(f"✅ Received hello response from {backend} API")
                logger.info(f"hudiy version: {message.app_version.major}.{message.app_version.minor}")
                logger.info(f"api version: {message.api_version.major}.{message.api_version.minor}")
                if send_to_api_gauges:
                    asyncio.run_coroutine_threadsafe(
                        ensure_ws_hub_started(logger=logger, ENABLE_LOGGING=ENABLE_LOGGING),
                        self.main_loop
                    )

        # ✅ Subscriptions NUR EINMAL pro Verbindung
            if send_api_mediadata_to_dashboard and not self._subs_sent:
                try:
                    set_status_subscriptions = api.SetStatusSubscriptions()
                    set_status_subscriptions.subscriptions.append(api.SetStatusSubscriptions.Subscription.MEDIA)
                    set_status_subscriptions.subscriptions.append(api.SetStatusSubscriptions.Subscription.PROJECTION)
                    self.client.send(
                        api.MESSAGE_SET_STATUS_SUBSCRIPTIONS, 0,
                        set_status_subscriptions.SerializeToString()
                    )
                    self._subs_sent = True
                    if ENABLE_LOGGING:
                        logger.info("📨 Sent status subscriptions (MEDIA, PROJECTION).")
                except Exception:
                    logger.warning("Failed to send subscription message to API.", exc_info=True)
            elif self._subs_sent and ENABLE_LOGGING:
                logger.info("ℹ️ Subscriptions already sent for this connection; skipping.")

            # read_cpu Task starten/verwaltet wie gehabt
            if send_to_api_gauges:
                if self.read_cpu_task is None:
                    if ENABLE_LOGGING:
                        logger.info("Starting new read_cpu task")
                    self.read_cpu_task = asyncio.run_coroutine_threadsafe(
                        self.read_cpu(temp_unit), self.main_loop
                    )
                elif self.read_cpu_task.done():
                    logger.info("Replacing finished read_cpu task")
                    self.read_cpu_task = asyncio.run_coroutine_threadsafe(
                        self.read_cpu(temp_unit), self.main_loop
                    )
                else:
                    if ENABLE_LOGGING:
                        logger.info("read_cpu task is already running – not restarting.")

            # 🌗 Initial Day/Night Mode (bei dir aktuell auskommentiert)
            if initial_day_night_mode == "day":
                self.send_day_night('day')
                pass
            else:
                self.send_day_night('night')
                pass

        @handle_errors
        def on_media_status(self, client, message):
            global playing, position, source
            old_position = position
            playing, position, source = message.is_playing, message.position_label, message.source

            if send_on_canbus and can_functional:
                if (toggle_fis1 == 4 or toggle_fis2 == 4) and position != old_position:
                    if toggle_fis1 == 4 and not show_label and not pause_fis1:
                        tasks.append(self.main_loop.create_task(media_to_dis1(), name="media_status_to_dis1"))
                    if toggle_fis2 == 4 and not pause_fis2:
                        tasks.append(self.main_loop.create_task(media_to_dis2(), name="media_status_to_dis2"))

            if ENABLE_LOGGING:
                logger.info("")
                logger.info(f"playing:     {playing}")
                logger.info(f"position:    {position}")
                logger.info(f"source:      {source}")
                logger.info("")

        @handle_errors
        def on_media_metadata(self, client, message):
            global title, artist, album, duration
            old_title, old_artist, old_album, old_duration = title, artist, album, duration
            title, artist, album, duration = message.title, message.artist, message.album, message.duration_label

            if send_on_canbus and can_functional:
                if (title, artist, album, duration) != (old_title, old_artist, old_album, old_duration):
                    if toggle_fis1 in (1, 2, 3, 5) and not show_label and not pause_fis1:
                        tasks.append(self.main_loop.create_task(media_to_dis1(), name="media_metadata_to_dis1"))
                    if toggle_fis2 in (1, 2, 3, 5) and not pause_fis2:
                        tasks.append(self.main_loop.create_task(media_to_dis2(), name="media_metadata_to_dis2"))

            if ENABLE_LOGGING:
                logger.info(f"title:       {title}")
                logger.info(f"artist:      {artist}")
                logger.info(f"album:       {album}")
                logger.info(f"duration:    {duration}")
                logger.info("")

        @handle_errors
        def on_projection_status(self, client, message):
            global ProjectionState, ProjectionSource, ProjectionStatus
            if openauto_ok:
                ProjectionState, ProjectionSource = message.state, message.source
                if ENABLE_LOGGING:
                    logger.info(f"Projection status, state: {ProjectionState}, source: {ProjectionSource}")
            elif hudiy_ok:
                ProjectionStatus = message.active  # ✅ bool
                if ENABLE_LOGGING:
                    logger.info(f"ProjectionStatus: {ProjectionStatus}")

        FORMULA_TO_KEY = {
            "getPidValue(4)": "cpu_load",
            "getPidValue(5)": "cpu_temp",
            "getPidValue(6)": "cpu_freq_mhz",
            "getPidValue(0)": "speed",
            "getPidValue(7)": "speed_measure",
            "getPidValue(3)": "outside_temp",
            "getPidValue(1)": "rpm",
            "getPidValue(2)": "coolant",
        }

        @handle_errors
        def update_to_api(self, formula, variable, variable_name="variable"):
            """
            - OpenAuto: unverändert – injizieren
            - Hudiy: inkrementelles WS-Push; optional Meta je nach Key
            """
            global backend, hudiy_ws_hub

            try:
                hub = hudiy_ws_hub

                if backend == "OpenAuto":
                    msg = api.ObdInjectGaugeFormulaValue()
                    msg.formula = str(formula)
                    msg.value = float(variable)
                    self.client.send(api.MESSAGE_OBD_INJECT_GAUGE_FORMULA_VALUE, 0, msg.SerializeToString())
                    if ENABLE_LOGGING:
                        logger.info("Push value to OpenAuto API (gauges): %s = %s", formula, variable)

                elif backend == "Hudiy":
                    key = FORMULA_TO_KEY.get(str(formula))
                    if key is None:
                        if ENABLE_LOGGING:
                            logger.warning("Hudiy: unknown formula '%s', not sending to dashboard.", formula)
                        return
                    else:
                        hub.set(key, variable)

                        if ENABLE_LOGGING:
                            logger.info("Push to Hudiy Websocket hub: %s = %s", key, variable)


            except Exception as e:
                logger.error("update_to_api failed: %s", e, exc_info=True)

        @handle_errors
        def outside_to_api(self, outside_temp_int):
            try:
                print(f"backend {backend}")
                if backend == "OpenAuto":
                    if ENABLE_LOGGING:
                        logger.info(f"Sending outside temperature : {outside_temp_int}{temp_unit} to API")
                    inject_temperature_sensor_value = api.InjectTemperatureSensorValue()
                    inject_temperature_sensor_value.value = outside_temp_int  # integer
                    serialized_data = inject_temperature_sensor_value.SerializeToString()
                    has_transport = (
                        hasattr(self.client, "_socket") and self.client._socket is not None
                    ) or (
                        hasattr(self.client, "_websocket") and self.client._websocket is not None
                    )
                    if api_is_connected and has_transport:
                        self.client.send(api.MESSAGE_INJECT_TEMPERATURE_SENSOR_VALUE, 0, serialized_data)

            except Exception as e:
                logger.error(f"Failed to send outside temperature '{outside_temp_int}' to API: {e}")
                raise

        @handle_errors
        def send_day_night(self, mode: str) -> None:
            try:
                # --- Determine / toggle mode ----------------------------------------
                if mode == "toggle":
                    base_is_day = self._resolve_day_mode_for_toggle()
                    is_day_mode = not base_is_day
                elif mode in ("day", "night"):
                    is_day_mode = (mode == "day")
                else:
                    raise ValueError("Invalid mode. Use 'day', 'night' or 'toggle'.")

                # Update cache
                self._is_day_mode = is_day_mode

                # --- Check if transport is available --------------------------------
                has_transport = (
                                        getattr(self.client, "_socket", None) is not None
                                ) or (
                                        getattr(self.client, "_websocket", None) is not None
                                )
                if not (api_is_connected and has_transport):
                    if ENABLE_LOGGING:
                        logger.warning("Client not connected – skipping send_day_night.")
                    return

                # --- Send depending on backend --------------------------------------
                if backend == "Hudiy":
                    # Hudiy expects SetDarkMode(enabled)
                    # Mapping: day -> enabled=False, night -> enabled=True
                    enabled = not is_day_mode
                    if ENABLE_LOGGING:
                        logger.info("Sending dark mode to Hudiy: enabled=%s", enabled)

                    msg = api.SetDarkMode()
                    msg.enabled = enabled
                    self.client.send(api.MESSAGE_SET_DARK_MODE, 0, msg.SerializeToString())


                elif backend == "OpenAuto":
                    # OAP: SetDayNight (Flags invertiert zu is_day_mode)
                    if ENABLE_LOGGING:
                        logger.info(
                            "Sending day/night mode to OpenAuto: %s",
                            "Day" if is_day_mode else "Night",
                        )
                    msg = api.SetDayNight()
                    msg.android_auto_night_mode = not is_day_mode
                    msg.oap_night_mode = not is_day_mode
                    self.client.send(api.MESSAGE_SET_DAY_NIGHT, 0, msg.SerializeToString())

                else:
                    logger.warning("Unknown backend '%s'. Skipping send_day_night().", backend)

            except Exception as e:
                logger.error(
                    "Failed to send day/night mode '%s' to %s API: %s",
                    mode,
                    backend,
                    e,
                    exc_info=True,
                )
                raise

        @handle_errors
        async def read_cpu(self, temp_unit):
            global cpu_load, cpu_temp, cpu_freq_mhz

            if ENABLE_LOGGING:
                logger.info("read_cpu started")
            while not stop_flag:
                if send_to_api_gauges or 9 in (toggle_fis1, toggle_fis2):
                    try:
                        cpu_load = min(round(psutil.cpu_percent()), 99)
                        if send_to_api_gauges and api_is_connected and self.client:
                            self.update_to_api("getPidValue(4)", cpu_load, "cpu_load")

                        temps = psutil.sensors_temperatures().get("cpu_thermal", [])
                        if temps:
                            cpu_temp = int(round(temps[0].current))
                            if temp_unit == '°F':
                                cpu_temp = round(cpu_temp * 1.8 + 32)
                            data = f'{cpu_load:02d}% {cpu_temp:02d}{temp_unit}'
                            if send_on_canbus and can_functional:
                                if toggle_fis1 == 9 and not show_label and not pause_fis1:
                                    await align_center(FIS1, data)
                                if toggle_fis2 == 9 and not pause_fis2:
                                    await align_center(FIS2, data)
                            if send_to_api_gauges and api_is_connected and self.client:
                                self.update_to_api("getPidValue(5)", cpu_temp, "cpu_temp")


                        if send_to_api_gauges:
                            cpu_freq = psutil.cpu_freq()
                            if cpu_freq:
                                cpu_freq_mhz = int(round(cpu_freq.current))
                                if api_is_connected and self.client:
                                    self.update_to_api("getPidValue(6)", cpu_freq_mhz, "cpu_freq_mhz")

                        await asyncio.sleep(3.0)
                    except asyncio.CancelledError:
                        if ENABLE_LOGGING:
                            logger.info("Task read_cpu was stopped.")
                        break
                    except Exception as e:
                        logger.error(f"Unexpected error in CPU monitor task: {e}", exc_info=True)

    return EventHandler


# --------------------------------------------------------------------
# Installer & Importer
# --------------------------------------------------------------------
@handle_errors
async def _install_from_github(
    base_dir: Union[str, Path], logger,
    owner: str, repo: str,
    subdir_parts: Tuple[str, ...],
    files: Tuple[str, ...],
    pkg_root_name: str,
    prefer_release: bool = True,):
    """
    Lädt `files` aus <repo>/<subdir_parts...> (erst Release, sonst main)
    nach <base>/<pkg_root_name>/common und importiert anschließend 'common'.
    Gibt (api_module, ClientClass, ClientEventHandlerClass, api_root_str) zurück.
    """
    base = Path(base_dir)
    api_root = base / pkg_root_name
    common_dir = api_root / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    (api_root / "__init__.py").touch(exist_ok=True)
    (common_dir / "__init__.py").touch(exist_ok=True)

    # 1) URL bestimmen: Release -> main
    def latest_zip(owner, repo):
        if prefer_release:
            try:
                r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/latest",
                                 headers={"User-Agent": "installer"}, timeout=15)
                # nachher (Py 3.7-kompatibel)
                if r.ok:
                    data_tmp = r.json()
                    tag = data_tmp.get("tag_name") if isinstance(data_tmp, dict) else None
                    if tag:
                        return f"https://github.com/{owner}/{repo}/archive/refs/tags/{tag}.zip", f"release:{tag}"
                    return f"https://github.com/{owner}/{repo}/archive/refs/tags/{tag}.zip", f"release:{tag}"
            except Exception:
                pass
        return f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip", "branch:main"

    zip_url, label = latest_zip(owner, repo)
    logger.info("%s: downloading %s (%s)", repo, zip_url, label)

    # 2) Zip laden
    r = requests.get(zip_url, headers={"User-Agent": "installer"}, timeout=60)
    r.raise_for_status()
    content = r.content

    # 3) Entpacken & gesuchte Unterstruktur finden
    with tempfile.TemporaryDirectory(prefix=f"{repo}_", dir=str(api_root)) as tmp:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(tmp)

        want_suffix = "/".join(subdir_parts)
        found = None
        for root, dirs, filelist in os.walk(tmp):
            if all(f in filelist for f in files):
                norm = root.replace("\\", "/")
                if norm.endswith(want_suffix):
                    found = Path(root); break
        if not found:
            raise RuntimeError(f"{repo}: Subfolder {'/'.join(subdir_parts)} with {files} not found.")

        # 4) Dateien kopieren (vorher alte entfernen)
        for fname in files:
            dst = common_dir / fname
            try:
                if dst.exists(): dst.unlink()
            except Exception:
                pass
            shutil.copy2(found / fname, dst)

    logger.info("%s: copy to %s: %s", repo, common_dir, ", ".join(files))

    # 5) Import-Pfad sauber setzen (Kollisionen vermeiden)
    for m in list(sys.modules.keys()):
        if m == "common" or m.startswith("common."):
            del sys.modules[m]
    for p in list(sys.path):
        if p.endswith("/openauto_api") or p.endswith("/openauto_api/common") \
           or p.endswith("/hudiy_api") or p.endswith("/hudiy_api/common"):
            try: sys.path.remove(p)
            except ValueError: pass
    sys.path.insert(0, str(api_root))
    importlib.invalidate_caches()

    # 6) Module laden
    api_module = importlib.import_module("common.Api_pb2")
    client_mod = importlib.import_module("common.Client")
    ClientClass = getattr(client_mod, "Client")
    ClientEventHandlerClass = getattr(client_mod, "ClientEventHandler")
    return api_module, ClientClass, ClientEventHandlerClass, str(api_root)

# ---- Dünne Wrapper ----
@handle_errors
async def _install_openauto_api(base_dir: Union[str, Path], logger):
    return await _install_from_github(
        base_dir, logger,
        owner="bluewave-studio", repo="openauto-pro-api",
        subdir_parts=("api_examples", "python", "common"),
        files=("Api_pb2.py", "Client.py", "Message.py"),
        pkg_root_name="openauto_api",
        prefer_release=True,  # prefer latest release
    )

@handle_errors
async def _install_hudiy_api(base_dir: Union[str, Path], logger):
    return await _install_from_github(
        base_dir, logger,
        owner="wiboma", repo="hudiy",
        subdir_parts=("examples", "api", "python", "common"),
        files=("Api_pb2.py", "Client.py", "Message.py"),
        pkg_root_name="hudiy_api",
        prefer_release=True,
    )

 #--- Helper: Paket-Root in sys.path eintragen (damit "import hudiy_api.*" klappt)
@handle_errors
def _add_pkg_root_to_syspath(api_root: Path):
    # api_root ist z.B. /home/pi/scripts/hudiy_api
    pkg_root = api_root.parent  # -> /home/pi/scripts
    # alte Einträge entfernen, dann vorn eintragen
    for p in list(sys.path):
        if p == str(pkg_root):
            sys.path.remove(p)
    sys.path.insert(0, str(pkg_root))

@handle_errors
async def _ensure_backend_api(backend: str):
    """
    Importiert vorhandene API oder installiert sie (nur HUDIY von hier aus, OAP wie gehabt).
    Liefert (api_module, ClientClass, ClientEventHandlerClass, api_root_path).
    """
    base_dir = Path(__file__).resolve().parent

    def _purge_common():
        for mod in list(sys.modules.keys()):
            if mod == "common" or mod.startswith("common."):
                del sys.modules[mod]

    def _use_api_root(api_root: Path):
        # alte Backend-Pfade raus
        for p in list(sys.path):
            if p.endswith("/openauto_api") or p.endswith("/openauto_api/common") \
               or p.endswith("/hudiy_api") or p.endswith("/hudiy_api/common"):
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass
        # nur der Root-Ordner des gewählten Backends
        sys.path.insert(0, str(api_root))
        importlib.invalidate_caches()

    #backend = "OpenAuto"

    if backend == "Hudiy":
        api_root = base_dir / "hudiy_api"
        _purge_common()
        _use_api_root(api_root)
        try:
            api_module = importlib.import_module("common.Api_pb2")
            client_mod = importlib.import_module("common.Client")

            # Nur loggen, wenn ENABLE_LOGGING aktiv ist
            if ENABLE_LOGGING:
                common_path_api = Path(api_module.__file__).resolve().parent
                common_path_client = Path(client_mod.__file__).resolve().parent
                logger.info("HUDIY: imported common from %s", common_path_api)
                if common_path_api != common_path_client:
                    logger.warning(
                        "HUDIY: Api_pb2 and Client come from different dirs: %s vs %s",
                        common_path_api, common_path_client
                    )

            ClientClass = getattr(client_mod, "Client")
            ClientEventHandlerClass = getattr(client_mod, "ClientEventHandler")
            return api_module, ClientClass, ClientEventHandlerClass, str(api_root)

        except Exception:
            # Installieren & erneut verwenden
            api_module, ClientClass, ClientEventHandlerClass, api_path = await _install_hudiy_api(base_dir, logger)
            _purge_common()
            _use_api_root(Path(api_path))

            if ENABLE_LOGGING:
                try:
                    common_path_api = Path(api_module.__file__).resolve().parent
                except Exception:
                    common_path_api = Path(api_path) / "common"
                logger.info(
                    "HUDIY: installed and imported common from %s (api_root=%s)",
                    common_path_api, api_path
                )

            return api_module, ClientClass, ClientEventHandlerClass, api_path


    elif backend == "OpenAuto":
        api_root = base_dir / "openauto_api"
        _purge_common()
        _use_api_root(api_root)
        try:
            api_module = importlib.import_module("common.Api_pb2")
            client_mod = importlib.import_module("common.Client")
            # 🔎 Nur loggen, wenn aktiviert
            if ENABLE_LOGGING:
                common_path_api = Path(api_module.__file__).resolve().parent
                common_path_client = Path(client_mod.__file__).resolve().parent
                logger.info("OpenAuto: imported common from %s", common_path_api)
                if common_path_api != common_path_client:
                    logger.warning(
                        "OpenAuto: Api_pb2 and Client come from different dirs: %s vs %s",
                        common_path_api, common_path_client
                    )
            ClientClass = getattr(client_mod, "Client")
            ClientEventHandlerClass = getattr(client_mod, "ClientEventHandler")
            return api_module, ClientClass, ClientEventHandlerClass, str(api_root)
        except Exception:
            # Fallback: installieren & erneut verwenden
            api_module, ClientClass, ClientEventHandlerClass, api_path = await _install_openauto_api(base_dir, logger)
            _purge_common()
            _use_api_root(Path(api_path))
            if ENABLE_LOGGING:
                try:
                    common_path_api = Path(api_module.__file__).resolve().parent
                except Exception:
                    common_path_api = Path(api_path) / "common"
                logger.info(
                    "OpenAuto: installed and imported common from %s (api_root=%s)",
                    common_path_api, api_path
                )
            return api_module, ClientClass, ClientEventHandlerClass, api_path


# --------------------------------------------------------------------
# Deine vorhandene Einbindung
# --------------------------------------------------------------------
@handle_errors
async def check_import_api():
    """
    Entscheidet HUDIY vs OpenAuto, stellt API bereit (Install+protoc wenn nötig)
    und importiert Api_pb2 + Client in die Globals.
    """
    global change_dark_mode_by_car_light, send_api_mediadata_to_dashboard, send_to_api_gauges
    global api, Client, ClientEventHandler, EventHandler
    global hudiy_ok, openauto_ok

    # Welche Features brauchen die API?
    if not (change_dark_mode_by_car_light or send_api_mediadata_to_dashboard or send_to_api_gauges):
        return

    # Backend-Auswahl: HUDIY bevorzugen, sonst OpenAuto
    backend = "Hudiy" if hudiy_ok else ("OpenAuto" if openauto_ok else None)
    if not backend:
        # nichts erkannt -> Features deaktivieren
        send_api_mediadata_to_dashboard = False
        change_dark_mode_by_car_light = False
        send_to_api_gauges = False
        return

    try:
        api, Client, ClientEventHandler, _ = await _ensure_backend_api(backend)
        EventHandler = define_event_handler_class()
    except Exception as error:
        await handle_exception(error, f"{backend} API not found or failed to install. Disabling {backend} API features.")
        send_api_mediadata_to_dashboard = False
        change_dark_mode_by_car_light = False
        send_to_api_gauges = False


# weitere Imports wie bei dir (api, Client, EventHandler, logger, ...)

# ------------------------------------------------------------
# Verbindung/Loop – sauberes Shutdown + richtiges wait_for_message-Handling
# ------------------------------------------------------------
@handle_errors
async def api_connection(event: asyncio.Event = None):
    """
    Connect to OAP/Hudiy API, poll messages, and keep the TCP session alive by
    sending MESSAGE_PING periodically. Subscriptions are sent once per connect
    in EventHandler.on_hello_response().
    """
    global client, api_is_connected, stop_flag, event_handler
    global openauto_ok, hudiy_ok, backend, can_functional, send_api_mediadata_to_dashboard

    RECONNECT_DELAY = 10.0
    WAIT_TIMEOUT = 2.0
    HEARTBEAT_INT = 6.0

    async def _heartbeat_task(_client):
        while api_is_connected and not stop_flag:
            try:
                _client.send(api.MESSAGE_PING, 0, b"")
                #logger.info("PING to API sent")
            except (OSError, RuntimeError, Exception) as e:
                logger.info("Heartbeat failed (%s). Exiting heartbeat loop.", e)
                return
            await asyncio.sleep(HEARTBEAT_INT)

    while not stop_flag:
        heartbeat = None
        future = None

        try:
            client = Client("read_from_canbus.py")

            loop = asyncio.get_running_loop()
            EventHandler = define_event_handler_class()
            event_handler = EventHandler(client, loop)
            client.set_event_handler(event_handler)

            logger.info("")

            # Verbinden (blocking) im Thread
            await loop.run_in_executor(None, client.connect, '127.0.0.1', 44405)
            logger.info("")
            logger.info(f"✅ Successfully connected to {backend} API.")

            await asyncio.sleep(0.3)
            api_is_connected = True

            if event:
                event.set()

            # Subscriptions werden jetzt ausschließlich im on_hello_response verschickt

            # Heartbeat starten
            heartbeat = asyncio.create_task(_heartbeat_task(client))

            # Nachrichten-Schleife
            while not stop_flag and api_is_connected:
                future = loop.run_in_executor(None, client.wait_for_message)
                try:
                    can_continue = await asyncio.wait_for(future, timeout=WAIT_TIMEOUT)
                    future = None  # abgeschlossen
                    if not can_continue:
                        logger.info("Server requested termination (BYEBYE).")
                        break
                except asyncio.TimeoutError:
                    future = None
                    continue
                except Exception as e:
                    logger.info("Peer closed connection or error in wait_for_message: %s", e)
                    break

        except asyncio.CancelledError:
            logger.info("❌ api_connection was cancelled.")
            break

        except struct.error as se:
            logger.error("struct.error in receive: %s", se)
            api_is_connected = False
            if event_handler and getattr(event_handler, "read_cpu_task", None):
                with contextlib.suppress(Exception):
                    event_handler.read_cpu_task.cancel()
            await asyncio.sleep(RECONNECT_DELAY)

        except ConnectionRefusedError:
            logger.warning(f"⚠️ {backend} API is not running or unreachable.")
            if not can_functional:
                stop_flag = True
                logger.warning("⚠️ No CAN-BUS and no API available. Stopping script...")
                asyncio.create_task(stop_script())
            else:
                api_is_connected = False
                await asyncio.sleep(RECONNECT_DELAY)

        except Exception:
            logger.error("❌ Unexpected error in api_connection", exc_info=True)
            api_is_connected = False
            await asyncio.sleep(RECONNECT_DELAY)

        finally:
            # --- GRACEFUL SHUTDOWN ---
            was_connected = api_is_connected
            api_is_connected = False  # sofort runter, verhindert neue send()s

            # 1) Heartbeat zuerst stoppen
            if heartbeat:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError, OSError, RuntimeError):
                    await heartbeat

            # 2) Falls noch ein wait_for_message-Future existiert, kurz auslaufen lassen
            if future and not future.done():
                with contextlib.suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(future), timeout=0.2)
                future = None

            # 3) Sauber disconnecten (sendet BYEBYE intern und schließt sofort)
            if was_connected:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, client.disconnect)
                    logger.info("")
                    logger.info(f"✅ Successfully disconnected from {backend} API.")
                    logger.info("")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    logger.debug("Peer already closed; ignoring disconnect error.")
                except Exception:
                    logger.warning(f"❌ Error while disconnecting from {backend} API", exc_info=True)

            # 4) Handler-Tasks aufräumen
            if event_handler and getattr(event_handler, "read_cpu_task", None):
                with contextlib.suppress(Exception):
                    event_handler.read_cpu_task.cancel()


@handle_errors
async def oap_units_check(temp_unit, speed_unit, lower_speed, upper_speed):
    config_path = "/home/pi/.openauto/config/openauto_obd_gauges.ini"
    if ENABLE_LOGGING:
        logger.info("📄 Checking existence of OBD gauge configuration file...")

    if not await asyncio.to_thread(os.path.exists, config_path):
        if ENABLE_LOGGING:
            logger.warning(f"❌ Config file {config_path} not found at expected location. Aborting.")
        return

    if ENABLE_LOGGING:
        logger.info("✅ File found. Analyzing content for unit consistency...")

    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    await asyncio.to_thread(config.read, config_path)

    modified = False

    for section in config.sections():
        if section.startswith("ObdGauge_"):
            label = config.get(section, "Label", fallback="")

            # Speed conversion
            if "km/h" in label and speed_unit == "mph":
                logger.info(f"🔄 Updating speed label in [{section}] to mph")
                config.set(section, "Label", label.replace("km/h", "mph"))
                for key in ("MaxValue", "MaxLimit", "MinValue", "MinLimit"):
                    old = float(config.get(section, key))
                    new = old / 1.60934
                    config.set(section, key, f"{new:.2f}")
                modified = True

            elif "mph" in label and speed_unit == "km/h":
                logger.info(f"🔄 Updating speed label in [{section}] to km/h")
                config.set(section, "Label", label.replace("mph", "km/h"))
                for key in ("MaxValue", "MaxLimit", "MinValue", "MinLimit"):
                    old = float(config.get(section, key))
                    new = old * 1.60934
                    config.set(section, key, f"{new:.2f}")
                modified = True

            # Temperature conversion
            if "°C" in label and temp_unit == "°F":
                logger.info(f"🌡️ Updating temperature label in [{section}] to °F")
                config.set(section, "Label", label.replace("°C", "°F"))
                for key in ("MinValue", "MaxValue", "MinLimit", "MaxLimit"):
                    old = float(config.get(section, key))
                    new = old * 1.8 + 32
                    config.set(section, key, f"{new:.2f}")
                modified = True

            elif "°F" in label and temp_unit == "°C":
                logger.info(f"🌡️ Updating temperature label in [{section}] to °C")
                config.set(section, "Label", label.replace("°F", "°C"))
                for key in ("MinValue", "MaxValue", "MinLimit", "MaxLimit"):
                    old = float(config.get(section, key))
                    new = (old - 32) / 1.8
                    config.set(section, key, f"{new:.2f}")
                modified = True

            # Acceleration gauge label update (e.g. 0-100 (s))
            if section == "ObdGauge_7":
                expected_label = f"{lower_speed}-{upper_speed} (s)"
                current_label = config.get(section, "Label", fallback="").strip()
                if current_label != expected_label:
                    logger.info(f"🏁 Updating acceleration label in [{section}] to '{expected_label}'")
                    config.set(section, "Label", expected_label)
                    modified = True
                else:
                    if ENABLE_LOGGING:
                        logger.info(f"✅ Acceleration label in [{section}] is already correct: '{current_label}'")

    if modified:
        # Backup config only if modifications occur
        backup_path = config_path + ".bak"
        await asyncio.to_thread(shutil.copy2, config_path, backup_path)
        if ENABLE_LOGGING:
            logger.info(f"🛡️ Backup created at: {backup_path}")

        if ENABLE_LOGGING:
            logger.info("💾 Writing updated configuration back to file...")
        await asyncio.to_thread(write_config_file, config, config_path)

        logger.info("♻️ Restarting OpenAuto to apply changes...")
        await asyncio.create_subprocess_exec("pkill", "-f", "autoapp")

        await asyncio.sleep(2)

        openauto_cmd = (
            "setsid bash -c 'DISPLAY=:0 stdbuf -o0 /usr/local/bin/autoapp >> /home/pi/.openauto/cache/openauto.log 2>&1'"
        )
        await asyncio.create_subprocess_shell(openauto_cmd, cwd="/home/pi")
    else:
        if ENABLE_LOGGING:
            logger.info("ℹ️ Units already match current settings – no changes needed.")


@handle_errors
def _port_in_use(host: str, port: int) -> bool:
    """Return True if TCP port is already in use on host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0

@handle_errors
def ensure_node_http_server_installed(
    logger=None,
    ENABLE_LOGGING: bool = True,
    auto_install: bool = True,
) -> bool:
    """
    Check if 'http-server' (node-http-server) is available.
    Optionally try to install it via apt (best effort).
    Returns True if available; otherwise False.
    """
    if shutil.which("http-server"):
        if ENABLE_LOGGING and logger:
            logger.info("Node http-server is already installed.")
        return True

    if not auto_install:
        if ENABLE_LOGGING and logger:
            logger.warning("Node http-server not found and auto-install is disabled.")
        return False

    if ENABLE_LOGGING and logger:
        logger.info("Attempting installation: sudo apt install -y node-opener node-http-server …")
    try:
        subprocess.run(["sudo", "apt", "update", "-y"], check=False)
        subprocess.run(["sudo", "apt", "install", "-y", "node-opener", "node-http-server"], check=False)
    except Exception as e:
        if ENABLE_LOGGING and logger:
            logger.warning("Apt installation failed: %s", e)

    if shutil.which("http-server"):
        if ENABLE_LOGGING and logger:
            logger.info("Node http-server installed successfully.")
        return True

    if ENABLE_LOGGING and logger:
        logger.warning("Node http-server still not available.")
    return False


@handle_errors
def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    try:
        import requests  # type: ignore
        r = requests.get(url, headers={"User-Agent": "hudiy-setup"}, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        # fallback to stdlib
        import urllib.request, urllib.error
        req = urllib.request.Request(url, headers={"User-Agent": "hudiy-setup"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@handle_errors
def _download_if_missing(url: str, dest: Path, logger=None, ENABLE_LOGGING: bool = True) -> bool:
    """
    Download url -> dest only if dest missing. Returns True if file present after call.
    """
    if dest.exists() and dest.stat().st_size > 0:
        if ENABLE_LOGGING and logger:
            logger.info("Found: %s", dest)
        return True
    try:
        data = _fetch_bytes(url)
        _ensure_dir(dest.parent)
        with open(dest, "wb") as f:
            f.write(data)
        if ENABLE_LOGGING and logger:
            logger.info("Downloaded: %s -> %s", url, dest)
        return True
    except Exception as e:
        if ENABLE_LOGGING and logger:
            logger.warning("Could not download %s: %s", url, e)
        return False

def _iter_tree(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        yield p

@handle_errors
def ensure_hudiy_js_tree(
    base_dir: Union[str, Path],
    logger=None,
    ENABLE_LOGGING: bool = True,
    send_to_api_gauges: bool = False,
    auto_download_common: bool = True,
    create_data_dir: bool = True,
) -> Path:

    base_dir = Path(base_dir).expanduser().resolve()
    api_root = base_dir / "hudiy_api"
    js_root = api_root / "html_files"
    common = js_root / "common"

    # 1) Make sure folders exist
    _ensure_dir(js_root)
    if create_data_dir:
        _ensure_dir(js_root / "data")
    _ensure_dir(common)

    if ENABLE_LOGGING and logger:
        logger.info("Hudiy JS root: %s", js_root)

    # 2) Ensure 'common' core files (download ONLY if missing)
    if auto_download_common:
        RAW = "https://raw.githubusercontent.com/wiboma/hudiy/main/examples/api/js/common"
        needed = {
            "protobuf.min.js": f"{RAW}/protobuf.min.js",
            "hudiy_client.js": f"{RAW}/hudiy_client.js",
            "Api.proto":       f"{RAW}/Api.proto",
        }
        for fname, url in needed.items():
            _download_if_missing(url, common / fname, logger=logger, ENABLE_LOGGING=ENABLE_LOGGING)
    else:
        if ENABLE_LOGGING and logger:
            logger.info("auto_download_common=False — expecting 'common' files to be already present in: %s", common)

    # 3) Helpful hints
    if ENABLE_LOGGING and logger:
        missing = [f for f in ("protobuf.min.js", "hudiy_client.js", "Api.proto") if not (common / f).exists()]
        if missing:
            logger.warning("Missing files in '%s': %s", common, ", ".join(missing))

    return js_root

@handle_errors
def write_config_file(config, path):
    with open(path, "w") as configfile:
        config.write(configfile)


@handle_errors
async def toggle_camera():
    global reversecamera_by_reversegear, reversecamera_by_down_longpress, camera_active, camera, backend, client, api

    if not (reversecamera_by_reversegear or reversecamera_by_down_longpress):
        return

    try:
        if backend == "Hudiy":
            action = "camera_web" if camera_active else "go_home"
            payload = api.DispatchAction(action=action).SerializeToString()
            # Direkter synchroner Aufruf – schnell und unkritisch
            if api_is_connected:
                client.send(api.MESSAGE_DISPATCH_ACTION, 0, payload)

        elif backend == "OpenAuto":
            # optionaler Kontextwechsel, kann auch weggelassen werden
            await asyncio.sleep(0)
            if camera_active:
                cam.hide()
            else:
                cam.show()


        else:
            logger.warning("Unknown backend: %r", backend)

    except Exception:
        logger.error("Error while toggling the reverse camera's livestream", exc_info=True)
        logger.info("Problem while toggling reverse camera detected - disabling reverse camera feature")
        reversecamera_by_reversegear = False
        reversecamera_by_down_longpress = False


# --------- Direkt in dein Script einfügen (z.B. oben bei den Imports) ---------
import os, shlex, subprocess, time

class WarmFF:
    """
    ffplay-Vorschau (Fullscreen, optional PNG-Overlay) einmal starten
    und per show()/hide() ein-/ausblenden – ohne den Prozess zu beenden.
    """

    @handle_errors
    def __init__(self, overlay_png=None, device="/dev/video0", display=":0",
                 width=800, height=480, fps=30, input_format="yuyv422",
                 window_title="FFCAM"):
        self.overlay_png = overlay_png
        self.device = device
        self.display = display
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.input_format = input_format  # "yuyv422" oder "mjpeg"
        self.window_title = window_title  # eindeutiger Fenstertitel
        self.proc = None
        self.win_id = None
        self._fs_set = False  # beim ersten show() genau einmal 'f' senden

    @handle_errors
    def _env(self):
        e = os.environ.copy()
        e["DISPLAY"] = self.display
        return e

    @handle_errors
    def _env_with_pos(self, x=None, y=None):
        e = self._env()
        if x is not None and y is not None:
            # ffplay (SDL) nimmt diese Startposition
            e["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        return e

    @handle_errors
    def _offscreen_coords(self):
        # robust: Desktop-Größe holen und ein Stück daneben starten
        try:
            w, h = subprocess.check_output(
                ["xdotool", "getdisplaygeometry"], env=self._env()
            ).decode().strip().split()
            return (int(w) + 200, int(h) + 200)
        except Exception:
            return (8000, 8000)  # Fallback

    @handle_errors
    def _set_taskbar_visible(self, show: bool):
        if not self.win_id:
            return
        env = self._env()
        if show:
            subprocess.run(["wmctrl", "-i", "-r", self.win_id, "-b", "remove,skip_taskbar,skip_pager"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["wmctrl", "-i", "-r", self.win_id, "-b", "add,skip_taskbar,skip_pager"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @handle_errors
    def _build_cmd(self, tiny: bool = False):
        # kein -fs hier; Fullscreen toggeln wir später per 'f'
        caps = (
            f"-f video4linux2 -input_format {self.input_format} "
            f"-video_size {self.width}x{self.height} -framerate {self.fps}"
        )
        size = "-x 1 -y 1 " if tiny else ""
        base = (
            f'ffplay -hide_banner -loglevel error -noborder '
            f'-fflags nobuffer -flags low_delay -framedrop '
            f'{size}-window_title "{self.window_title}" '
            f'{caps} -i {self.device} '
        )

        w, h = self.width, self.height
        # fülle den Bildschirm vollständig (cover): erst vergrößern, dann crop
        sc = f"scale={w}:{h}:flags=fast_bilinear:force_original_aspect_ratio=increase,crop={w}:{h}"

        if self.overlay_png:
            # Kompatibel mit älteren ffmpeg: KEIN format=auto, klassische Labels
            vf = f'"[in]{sc}[v1];movie={self.overlay_png}[ol];[v1][ol]overlay=0:0[out]"'
            return f"{base}-vf {vf}"
        else:
            return f'{base}-vf "{sc}"'

    @handle_errors
    def _capture_window_id(self):
        self.win_id = None
        env = self._env()
        # 1) per Fenstertitel
        try:
            out = subprocess.check_output(
                ["xdotool", "search", "--name", self.window_title],
                env=env, stderr=subprocess.DEVNULL
            ).decode().strip().splitlines()
            if out:
                self.win_id = out[-1]; return
        except Exception:
            pass
        # 2) Fallback per PID
        if self.proc and self.proc.poll() is None:
            for args in (["xdotool", "search", "--onlyvisible", "--pid", str(self.proc.pid)],
                         ["xdotool", "search", "--pid", str(self.proc.pid)]):
                try:
                    out = subprocess.check_output(args, env=env, stderr=subprocess.DEVNULL).decode().strip().splitlines()
                    if out:
                        self.win_id = out[-1]; return
                except Exception:
                    pass

    @handle_errors
    def start(self, visible: bool = True):
        """ffplay starten. Bei visible=False: offscreen, 1x1, gemappt (warm & unsichtbar)."""
        # Läuft schon?
        if self.proc and self.proc.poll() is None:
            self.show() if visible else self.hide()
            return

        tiny = not visible
        cmd = self._build_cmd(tiny=tiny)  # <-- KEIN -fs hier
        env = self._env_with_pos(*self._offscreen_coords()) if tiny else self._env()
        self.proc = subprocess.Popen(shlex.split(cmd), env=env)
        self._fs_set = False

        # bis 2 s WID holen
        for _ in range(20):
            time.sleep(0.1)
            self._capture_window_id()
            if self.win_id:
                break

        # Warmstart versteckt: offscreen + 1x1 (gemappt lassen!)
        if not visible and self.win_id:
            env2 = self._env()
            offx, offy = self._offscreen_coords()
            subprocess.run(["xdotool", "windowmove", self.win_id, str(offx), str(offy)], env=env2,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["xdotool", "windowsize", self.win_id, "1", "1"], env=env2,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @handle_errors
    def show(self):
        # Falls Prozess/WID fehlen: sichtbar neu starten
        if not self.proc or self.proc.poll() is not None or not self.win_id:
            try:
                self.stop()
            except Exception:
                pass
            cmd = self._build_cmd(tiny=False)
            self.proc = subprocess.Popen(shlex.split(cmd), env=self._env())
            self._fs_set = False
            for _ in range(20):
                time.sleep(0.05)
                self._capture_window_id()
                if self.win_id: break
        if not self.win_id:
            return

        env = self._env()
        # Bildschirmgröße
        try:
            W, H = subprocess.check_output(["xdotool", "getdisplaygeometry"], env=env).decode().strip().split()
        except Exception:
            W, H = str(self.width), str(self.height)
        wid = self.win_id

        # sichtbar machen & bildfüllend ziehen
        subprocess.run(["xdotool", "windowmap", wid], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xdotool", "windowmove", wid, "0", "0"], env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        subprocess.run(["xdotool", "windowsize", wid, W, H], env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        # genau EINMAL ffplay-Fullscreen toggeln (Taste 'f'), dann merken
        if not self._fs_set:
            time.sleep(0.05)  # kleine Wartezeit, damit SDL den Key sicher nimmt
            subprocess.run(["xdotool", "key", "--window", wid, "f"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._fs_set = True

        subprocess.run(["xdotool", "windowraise", wid], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xdotool", "windowactivate", wid], env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    @handle_errors
    def hide(self):
        if not self.win_id:
            self._capture_window_id()
        if not self.win_id:
            return
        env = self._env()
        subprocess.run(["xdotool", "windowunmap", self.win_id], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @handle_errors
    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.win_id = None

    @handle_errors
    def _set_taskbar_visible(self, show: bool):
        if not self.win_id:
            return
        env = self._env()
        if show:
            subprocess.run(["wmctrl", "-i", "-r", self.win_id, "-b", "remove,skip_taskbar,skip_pager"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["wmctrl", "-i", "-r", self.win_id, "-b", "add,skip_taskbar,skip_pager"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Cam:
    """
    Hält genau EINE ffplay-Instanz aktiv (BASE ODER OVERLAY).
    Beim Umschalten wird ggf. die laufende Instanz gestoppt und die andere gestartet.
    So gibt es keinen Doppelzugriff auf /dev/video0.
    """

    @handle_errors
    def __init__(self, overlay_png=None, device="/dev/video0", display=":0",
                 width=800, height=480, fps=30, input_format="yuyv422"):
        self.device = device
        self.display = display
        self.width, self.height, self.fps = width, height, fps
        self.input_format = input_format
        self.overlay_png = overlay_png
        self.cur = None  # aktive WarmFF-Instanz (base oder over)

    @handle_errors
    def _make_base(self):
        return WarmFF(
            overlay_png=None, device=self.device, display=self.display,
            width=self.width, height=self.height, fps=self.fps,
            input_format=self.input_format, window_title="FFCAM-BASE"
        )

    @handle_errors
    def _make_overlay(self):
        return WarmFF(
            overlay_png=self.overlay_png, device=self.device, display=self.display,
            width=self.width, height=self.height, fps=self.fps,
            input_format=self.input_format, window_title="FFCAM-OVERLAY"
        )

    # in der Klasse Cam ersetzen:
    @handle_errors
    def start(self, visible: bool = False, warm_variant: str = "base"):
        """
        Eine Instanz warm starten (BASE oder OVERLAY).
        warm_variant: "base" oder "overlay"
        """
        if self.cur:
            self.cur.stop()
            self.cur = None

        use_overlay = (warm_variant == "overlay" and self.overlay_png is not None)
        inst = self._make_overlay() if use_overlay else self._make_base()

        inst.start(visible=visible)  # visible=False -> offscreen/1x1 gemappt (Warmstart)
        self.cur = inst

    @handle_errors
    def _switch_to(self, want_overlay: bool):
        # laufende Instanz immer freigeben, bevor eine neue startet
        if self.cur:
            self.cur.stop()
            self.cur = None
        inst = self._make_overlay() if want_overlay else self._make_base()
        inst.start(visible=True)  # jetzt wirklich starten & zeigen
        self.cur = inst

    @handle_errors
    def show(self):
        if self.overlay_png is None:
            # kein Overlay vorgesehen → einfach Base starten/zeigen
            if not self.cur:
                self._switch_to(False)
            else:
                # falls gerade Overlay läuft, umschalten
                if self.cur.overlay_png is not None:
                    self._switch_to(False)
                else:
                    self.cur.show()
        else:
            # Overlay vorhanden, aber hier explizit "ohne Overlay" anzeigen
            if not self.cur or self.cur.overlay_png is not None:
                self._switch_to(False)
            else:
                self.cur.show()

    @handle_errors
    def show_overlay(self):
        if self.overlay_png is None:
            # kein Overlay definiert → fallback auf show()
            return self.show()
        if not self.cur or self.cur.overlay_png is None:
            self._switch_to(True)
        else:
            self.cur.show()

    @handle_errors
    def hide(self):
        if self.cur:
            self.cur.hide()

    @handle_errors
    def stop(self):
        if self.cur:
            self.cur.stop()
            self.cur = None



@handle_errors
async def read_on_canbus(message):
    canid = message.arbitration_id
    msg = binascii.hexlify(message.data).decode('ascii').upper()
    canid_print = str(hex(message.arbitration_id).lstrip('0x').upper())
    if script_started:
        if show_can_messages_in_logs:
            logger.info(f"CAN-Message with CAN-ID: {canid_print} Message: {msg} received")
        callbacks = {
            0x271: process_canid_271_2C3,
            0x2C3: process_canid_271_2C3,
            0x351: process_canid_351,
            0x353: process_canid_353_35B,
            0x35B: process_canid_353_35B,
            0x461: process_canid_461,
            0x5C3: process_canid_5C3,
            0x602: process_canid_602,
            0x623: process_canid_623,
            0x635: process_canid_635,
            0x65F: process_canid_65F,
            0x661: process_canid_661,
        }
        callback = callbacks.get(canid)
        if callback:
            await callback(msg)


@handle_errors
async def process_canid_271_2C3(msg):
    global last_msg_271_2C3, shutdown_script
    if msg != last_msg_271_2C3:
        if msg[0:2] != last_msg_271_2C3[0:2]:
            if shutdown_by_ignition_off:
                if msg[0:2] == '11':
                    if shutdown_type == 'instant':
                        logger.info("Ignition off message detected - system will shutdown now!")
                        await run_command("sudo shutdown -h now", log_output=False)
                    elif shutdown_type == 'gently':
                        logger.info(
                            "Ignition off message detected. The system will stop the script and shutdown gently!")
                        shutdown_script = True
                        asyncio.create_task(stop_script())
            elif shutdown_by_pulling_key:
                if msg[0:2] == '10':
                    if shutdown_type == 'instant':
                        logger.info("Pulling key message detected - system will shutdown now!")
                        await run_command("sudo shutdown -h now", log_output=False)
                    elif shutdown_type == 'gently':
                        logger.info(
                            "Pulling key message detected. The system will stop the script and shutdown gently!")
                        shutdown_script = True
                        asyncio.create_task(stop_script())
        last_msg_271_2C3 = msg



@handle_errors
async def process_canid_351(msg):  # handler as EventHandler-Instance
    global gear, speed_counter, speed, outside_temp_counter, outside_temp, last_speed, last_outside_temp, elapsed_time
    global reversecamera_by_reversegear, reversecamera_by_down_longpress, reversecamera_guidelines, overlay
    global speed_measure_to_api, measure_done, start_time, elapsed_time_formatted, drop1, last_data, outside_temp_int

    speed_frequency = 2
    outside_temp_frequency = 10


    if reversecamera_by_reversegear:
        if msg[0:2] == '00' and gear == 1:
            gear = 0
            logger.info("Forward gear is engaged - stopping the reverse camera with a "
                        f"{reversecamera_turn_off_delay}-second delay.")
            await asyncio.sleep(reversecamera_turn_off_delay)
            try:
                if backend == "Hudiy":
                    if api_is_connected:
                        client.send(api.MESSAGE_DISPATCH_ACTION, 0, api.DispatchAction(action="go_home").SerializeToString())
                elif backend == "OpenAuto":
                    if reversecamera_guidelines:
                        cam.hide()
            except Exception as e:
                logger.error("Error while stopping the reverse camera's livestream.", exc_info=True)
                reversecamera_by_reversegear = False
                reversecamera_by_down_longpress = False
        elif msg[0:2] == '02' and gear == 0:
            gear = 1
            logger.info("Reverse gear engaged - starting the reverse camera")
            try:
                if backend == "Hudiy":
                    if reversecamera_guidelines:
                        if api_is_connected:
                            client.send(api.MESSAGE_DISPATCH_ACTION, 0, api.DispatchAction(action="camera_web_with_lines").SerializeToString())
                    else:
                        if api_is_connected:
                            client.send(api.MESSAGE_DISPATCH_ACTION, 0, api.DispatchAction(action="camera_web").SerializeToString())
                elif backend == "OpenAuto":
                    if reversecamera_guidelines and cam.overlay_png:
                        cam.show_overlay()
                    else:
                        cam.show()


            except Exception as e:
                logger.error("Error while starting the reverse camera's livestream", exc_info=True)
                reversecamera_by_reversegear = False
                reversecamera_by_down_longpress = False
    if 6 in (toggle_fis1, toggle_fis2) or 10 in (toggle_fis1, toggle_fis2) or send_to_api_gauges:
        speed_counter += 1
        if speed_counter % speed_frequency == 0 or last_speed is None or 10 in (toggle_fis1, toggle_fis2):

            # Rohwert einmal aus dem Frame ziehen
            raw = int(msg[4:6] + msg[2:4], 16)
            base_kmh = raw / 200.0

            # Einheit anwenden
            if speed_unit == 'km/h':
                speed = int(base_kmh)
            elif speed_unit == 'mph':
                speed = int(base_kmh * 0.621371)
            else:
                speed = int(base_kmh)  # Fallback

            # --- Änderung atomar feststellen & übernehmen (keine awaits hier!) ---
            prev = last_speed
            changed = (prev is None) or (speed != prev)
            if changed:
                last_speed = speed  # sofort setzen, bevor wir awaiten oder loggen

                if ENABLE_LOGGING:
                    logger.info("Speed has changed from %s to %s %s", prev, speed, speed_unit)

                # Push an Hudiy/OpenAuto nur bei Änderung
                if send_to_api_gauges and api_is_connected:
                    event_handler.update_to_api("getPidValue(0)", speed, "speed")

            # --- FIS-Anzeige aktualisieren (darf awaiten), aber nur bei Änderung ---
            if changed and send_on_canbus and can_functional and send_values_to_dashboard:
                data = f'{speed} {speed_unit}'
                if toggle_fis1 == 6 and not show_label and not pause_fis1:
                    await align_right(FIS1, data)
                if toggle_fis2 == 6 and not pause_fis2:
                    await align_right(FIS2, data)

            # --- 0–100-Messung ---
            if 10 in (toggle_fis1, toggle_fis2):
                if float(speed) > float(lower_speed):
                    if start_time is None:
                        start_time = time.time()

                    if measure_done == 0:
                        elapsed_time = time.time() - start_time
                        data = f"{elapsed_time:05.2f} s"
                        speed_measure_to_api = float(f"{elapsed_time:.2f}")

                    elif measure_done == 1:
                        # Anzeige bleibt beim letzten Wert
                        data = f"{elapsed_time:05.2f} s"
                        speed_measure_to_api = float(f"{elapsed_time:.2f}")

                    if float(speed) >= float(upper_speed) and measure_done == 0:
                        measure_done = 1
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        elapsed_time_formatted_print = f"{elapsed_time:05.2f}"

                        if export_speed_measurements_to_file:
                            start_time_formatted = datetime.fromtimestamp(start_time).strftime("%H:%M:%S.%f")[:-5]
                            end_time_formatted = datetime.fromtimestamp(end_time).strftime("%H:%M:%S.%f")[:-5]

                            range_label = f"{lower_speed}-{upper_speed} {speed_unit}"
                            result_message = "{:<10} | {:>12} | {:<10} - {:<10} | {:>6} seconds\n".format(
                                time.strftime("%Y/%m/%d"),
                                range_label,
                                start_time_formatted,
                                end_time_formatted,
                                elapsed_time_formatted_print
                            )

                            with open("speed_measurements.txt", "a") as file:
                                file.write(result_message)
                        logger.info("")
                        logger.info(
                            "The time to accelerate from %s-%s %s took %s seconds.",
                            lower_speed, upper_speed, speed_unit, elapsed_time_formatted_print
                        )
                        logger.info("")
                        data = f"{elapsed_time:05.2f} s"
                        speed_measure_to_api = float(f"{elapsed_time:.2f}")
                        start_time = None

                else:
                    # Reset wenn nicht über lower_speed
                    start_time = None
                    measure_done = 0
                    data = '00.00 s'
                    speed_measure_to_api = 0.00



                if data != last_data:
                    if 10 in (toggle_fis1, toggle_fis2):
                        drop1 += 1
                        if drop1 == 3 or measure_done or start_time is None:
                            if send_to_api_gauges:
                                event_handler.update_to_api("getPidValue(7)", speed_measure_to_api, "speed_measure")
                            if toggle_fis1 == 10 and not show_label and not pause_fis1:
                                await align_right(FIS1, data)
                            if toggle_fis2 == 10 and not pause_fis2:
                                await align_right(FIS2, data)
                            drop1 = 0
                    last_data = data
    if (11 in (toggle_fis1, toggle_fis2) or send_to_api_gauges) and carmodel == '8E':
        outside_temp_counter += 1
        if outside_temp_counter % outside_temp_frequency == 0 or last_outside_temp is None:
            if temp_unit == '°C':
                outside_temp = float(int(msg[10:12], 16) / 2 - 50)
            if temp_unit == '°F':
                outside_temp = int(msg[10:12], 16) / 2 - 50
                outside_temp = float(outside_temp * 1.8 + 32)
            if send_on_canbus and can_functional and send_values_to_dashboard and outside_temp != last_outside_temp:
                if toggle_fis1 == 11 and not show_label and not pause_fis1:
                    data = f'{outside_temp}{temp_unit}'
                    await align_right(FIS1, data)
                if toggle_fis2 == 11 and not pause_fis2:
                    data = f'{outside_temp}{temp_unit}'
                    await align_right(FIS2, data)
                if ENABLE_LOGGING:
                    logger.info("Outside-Temp has changed from %s to %s %s", last_outside_temp, outside_temp, temp_unit)
            if send_to_api_gauges and api_is_connected and outside_temp is not None:
                event_handler.update_to_api("getPidValue(3)", outside_temp, "outside_temp")
                if backend == "OpenAuto":
                    outside_temp_api = int(int(msg[10:12], 16) / 2 - 50)
                    event_handler.outside_to_api(outside_temp_api) # outside_temp for OpenAuto headline (temperatrue inject)
            last_outside_temp = outside_temp


@handle_errors
async def process_canid_353_35B(msg):
    global rpm_counter, rpm, coolant_counter, coolant, last_rpm, last_coolant
    rpm_frequency = 2
    coolant_frequency = 5

    if 7 in (toggle_fis1, toggle_fis2) or send_to_api_gauges:
        rpm_counter += 1
        if rpm_counter % rpm_frequency == 0 or last_rpm is None:
            rpm = int(msg[4:6] + msg[2:4], 16) / 4
            rpm = int(rpm)
            if send_on_canbus and can_functional and send_values_to_dashboard and rpm != last_rpm:
                if toggle_fis1 == 7 and not show_label and not pause_fis1:
                    data = f'{rpm} RPM'
                    await align_right(FIS1, data)
                if toggle_fis2 == 7 and not pause_fis2:
                    data = f'{rpm} RPM'
                    await align_right(FIS2, data)
                if ENABLE_LOGGING:
                    logger.info("RPM has changed from %s to %s rpm", last_rpm, rpm)
            if send_to_api_gauges and rpm != last_rpm and api_is_connected:
                event_handler.update_to_api("getPidValue(1)", rpm, "rpm")
            last_rpm = rpm
    if 8 in (toggle_fis1, toggle_fis2) or send_to_api_gauges:
        coolant_counter += 1
        if coolant_counter % coolant_frequency == 0 or last_coolant is None:
            if temp_unit == '°C':
                coolant = int(msg[6:8], 16) * 0.75 - 48
            elif temp_unit == '°F':
                coolant = int(msg[6:8], 16) * 0.75 - 48
                coolant = int(coolant * 1.8 + 32)
            coolant = int(coolant)
            if send_on_canbus and can_functional and send_values_to_dashboard and coolant != last_coolant:
                if toggle_fis1 == 8 and not show_label and not pause_fis1:
                    data = f'{coolant}{temp_unit} W'
                    await align_right(FIS1, data)
                if toggle_fis2 == 8 and not pause_fis2:
                    data = f'{coolant}{temp_unit} W'
                    await align_right(FIS2, data)
                if ENABLE_LOGGING:
                    logger.info("Coolant has changed from %s to %s %s", last_coolant, coolant, temp_unit)
            if send_to_api_gauges and api_is_connected:
                event_handler.update_to_api("getPidValue(2)", coolant, "coolant")
            last_coolant = coolant


@handle_errors
async def process_canid_461(msg):
    global up, down, select, back, nextbtn, prev, setup
    global toggle_fis1, toggle_fis2, pause_fis1, pause_fis2, camera_active

    if control_pi_by_rns_e_buttons:
        if msg == '373001004001':
            if ENABLE_LOGGING:
                logger.info("SHORT-Press of RNS-E Button detected: WHEEL left | Keyboard: 1 | OpenAuto: Scroll left | HUDIY: Scroll left")
            device.emit(uinput.KEY_1, 1)
            device.emit(uinput.KEY_1, 0)
        elif msg == '373001002001':
            if ENABLE_LOGGING:
                logger.info("SHORT-Press of RNS-E Button detected: Wheel right | Keyboard: 2 | OpenAuto: Scroll right | HUDIY: Scroll right")
            device.emit(uinput.KEY_2, 1)
            device.emit(uinput.KEY_2, 0)
        elif msg == '373001400000':  # RNS-E: up button pressed
            up += 1
        elif msg == '373004400000' and up > 0:  # RNS-E: up button released
            if up <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: UP | Keyboard: UP arrow | OpenAuto: Navigate up | HUDIY: Navigate up")
                device.emit(uinput.KEY_UP, 1)
                device.emit(uinput.KEY_UP, 0)
            elif 4 < up <= 16:
                if backend == "Hudiy":
                    if ENABLE_LOGGING:
                        logger.info(
                            "LONG-Press of RNS-E Button detected: UP | Keyboard: T | HUDIY: Toggle input focus")
                    device.emit(uinput.KEY_T, 1)
                    device.emit(uinput.KEY_T, 0)
                elif backend == "OpenAuto":
                    if ENABLE_LOGGING:
                        logger.info("LONG-Press of RNS-E Button detected: UP  | Keyboard: CTRL+F3 | OpenAuto: Toggle application")
                    device.emit(uinput.KEY_LEFTCTRL, 1)
                    device.emit(uinput.KEY_F3, 1)
                    device.emit(uinput.KEY_LEFTCTRL, 0)
                    device.emit(uinput.KEY_F3, 0)
            elif up > 16:
                    if toggle_values_by_rnse_longpress:
                        if not show_label:
                            up = 0
                            if ENABLE_LOGGING:
                                logger.info(
                                    "VERY LONG-Press of RNS-E Button detected: UP | HUDIY: Optional: toggle dis/fis 1. line values")
                            if (send_on_canbus and can_functional) or toggle_fis1 == 0:
                                pause_fis1 = True
                                await block_show_value1()
                                await media_to_dis1()
            up = 0
        elif msg == '373001800000':  # RNS-E: down button pressed
            down += 1
        elif msg == '373004800000' and down > 0:  # RNS-E: down button released
            if down <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: DOWN | Keyboard: DOWN arrow | OpenAuto: Navigate Down | HUDIY: Navigate Down")
                device.emit(uinput.KEY_DOWN, 1)
                device.emit(uinput.KEY_DOWN, 0)
            elif 4 < down <= 16:
                if ENABLE_LOGGING:
                    logger.info("LONG-Press of RNS-E Button detected: DOWN | Keyboard: O | OpenAuto: End phone call | HUDIY: End phone call")
                device.emit(uinput.KEY_O, 1)
                device.emit(uinput.KEY_O, 0)
            elif down > 16:
                if toggle_values_by_rnse_longpress:
                    down = 0
                    if ENABLE_LOGGING:
                        logger.info("VERY LONG-Press of RNS-E Button detected: DOWN | HUDIY: Optional: toggle dis/fis 2. line values")
                    if (send_on_canbus and can_functional) or toggle_fis2 == 0:
                        pause_fis2 = True
                        await block_show_value2()
                        await media_to_dis2()
            down = 0
        elif msg == '373001001000':  # RNS-E: wheel pressed
            select += 1
        elif msg == '373004001000' and select > 0:  # RNS-E: wheel released
            if select <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: WHEEL press | Keyboard: ENTER | OpenAuto: Select | HUDIY: Select")
                device.emit(uinput.KEY_ENTER, 1)
                device.emit(uinput.KEY_ENTER, 0)
            elif select > 4:
                if ENABLE_LOGGING:
                    logger.info("LONG-Press of RNS-E Button detected: WHEEL press | Keyboard: B | OpenAuto: Toggle play/pause | HUDIY: Toggle play/pause")
                device.emit(uinput.KEY_B, 1)
                device.emit(uinput.KEY_B, 0)
            select = 0
        elif msg == '373001000200':  # RNS-E: return button pressed
            back += 1
        elif msg == '373004000200' and back > 0:  # RNS-E: return button released
            if back <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: RETURN | Keyboard: ESC | OpenAuto: Back | HUDIY: Back")
                device.emit(uinput.KEY_ESC, 1)
                device.emit(uinput.KEY_ESC, 0)
            elif 4 < back <= 50:
                if backend == "Hudiy":
                    if ENABLE_LOGGING:
                        logger.info("LONG-Press of RNS-E Button detected: RETURN | Keyboard: H | HUDIY: Hudiy Home")
                    device.emit(uinput.KEY_H, 1)
                    device.emit(uinput.KEY_H, 0)
                elif backend == "OpenAuto":
                    if ENABLE_LOGGING:
                        logger.info("LONG-Press of RNS-E Button detected: RETURN | Keyboard: F12 | OpenAuto: Bring OAP to front")
                    device.emit(uinput.KEY_F12, 1)
                    device.emit(uinput.KEY_F12, 0)
            elif back > 50:
                if ENABLE_LOGGING:
                    logger.info("VERY LONG-Press of RNS-E Button detected: RETURN | OpenAuto: Shutdown Raspberry Pi | HUDIY: Shutdown Raspberry Pi")
                command = 'sudo shutdown -h now'
                result = await run_command("sudo shutdown -h now", log_output=False)
                if result["stderr"]:
                    logger.error(f"Failed to shutdown Raspberry Pi: {result['stderr']}")
            back = 0
        elif msg == '373001020000':  # RNS-E: next track button pressed
            nextbtn += 1
        elif msg == '373004020000' and nextbtn > 0:  # RNS-E: next track button released
            if nextbtn <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: >| (next) | Keyboard: N | OpenAuto: Next track | HUDIY: Next track")
                device.emit(uinput.KEY_N, 1)
                device.emit(uinput.KEY_N, 0)
            elif nextbtn > 4:
                if ENABLE_LOGGING:
                    logger.info("LONG-Press of RNS-E Button detected: >| (next) | Keyboard: Right arrow | OpenAuto: - | HUDIY: Right")
                device.emit(uinput.KEY_RIGHT, 1)
                device.emit(uinput.KEY_RIGHT, 0)
            nextbtn = 0
        elif msg == '373001010000':  # RNS-E: previous track button pressed
            prev += 1
        elif msg == '373004010000' and prev > 0:  # RNS-E: previous track button released
            if prev <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: |< (previous) | Keyboard: V | OpenAuto: Previous track | HUDIY: Previous track")
                device.emit(uinput.KEY_V, 1)
                device.emit(uinput.KEY_V, 0)
            elif 4 < prev <= 16:
                if ENABLE_LOGGING:
                    logger.info("LONG-Press of RNS-E Button detected: |< (previous) | Keyboard: Left arrow | OpenAuto: - | HUDIY: Left")
                device.emit(uinput.KEY_LEFT, 1)
                device.emit(uinput.KEY_LEFT, 0)
            elif prev > 16:
                if ENABLE_LOGGING:
                    logger.info("VERY LONG-Press of RNS-E Button detected: |< (previous) | Keyboard: - | OpenAuto: toggle_camera | HUDIY: toggle_camera ")
                if reversecamera_by_down_longpress:
                    camera_active = not camera_active
                    await toggle_camera()

            prev = 0
        elif msg == '373001000100':  # RNS-E: setup button pressed
            setup += 1
        elif msg == '373004000100' and setup > 0:  # RNS-E: setup button released
            if setup <= 4:
                if ENABLE_LOGGING:
                    logger.info("SHORT-Press of RNS-E Button detected: SETUP | Keyboard: M | OpenAuto: Voice command | HUDIY: Voice command")
                device.emit(uinput.KEY_M, 1)
                device.emit(uinput.KEY_M, 0)
            elif 4 < setup <= 16:
                if ENABLE_LOGGING:
                    logger.info("LONG-Press of RNS-E Button detected: SETUP | Keyboard: F2 (OAP) / API (Hudiy) | OpenAuto: Toggle night mode AA/general | HUDIY: Toggle night mode AA/general")
                if backend == "Hudiy":
                    event_handler.send_day_night("toggle")
                elif backend == "OpenAuto":
                    device.emit(uinput.KEY_F2, 1)
                    device.emit(uinput.KEY_F2, 0)
            elif setup > 16:
                setup = 0
                if ENABLE_LOGGING:
                    logger.info("VERY LONG-Press of RNS-E Button detected: SETUP |  OpenAuto: Toggle candump | HUDIY: Toggle candump")
                await candump()
            setup = 0


@handle_errors
async def process_canid_5C3(msg):
    global press_mfsw, nextbtn, prev

    if read_mfsw_buttons:
        if tv_mode_active == 1:
            if (carmodel == '8E' and msg == '3904') or \
                    (carmodel in ['8P', '8J'] and msg == '390B'):
                device.emit(uinput.KEY_1, 1)
                device.emit(uinput.KEY_1, 0)
                press_mfsw = 0
            elif (carmodel == '8E' and msg == '3905') or \
                    (carmodel in ['8P', '8J'] and msg == '390C'):
                device.emit(uinput.KEY_2, 1)
                device.emit(uinput.KEY_2, 0)
                press_mfsw = 0
            elif carmodel in ['8E', '8P', '8J'] and msg == '3908':
                press_mfsw += 1
            elif (msg in ['3900', '3A00']) and press_mfsw > 0:
                if press_mfsw == 1:
                    device.emit(uinput.KEY_ENTER, 1)
                    device.emit(uinput.KEY_ENTER, 0)
                    press_mfsw = 0
                elif press_mfsw >= 2:
                    device.emit(uinput.KEY_ESC, 1)
                    device.emit(uinput.KEY_ESC, 0)
                    press_mfsw = 0
            elif msg == '3900' and press_mfsw == 0:
                nextbtn = 0
                prev = 0


tv_input_activation_detected = False


@handle_errors
async def process_canid_602(msg):
    global tv_input_activation_detected
    if msg.startswith('091230') or msg.startswith('811230'):
        if not tv_input_activation_detected:
            logger.info("tv input message detected")
            tv_input_activation_detected = True
            if tv_input_task is not None:
                tv_input_task.stop()


@handle_errors
async def process_canid_623(msg):
    global tmset

    if read_and_set_time_from_dashboard and tmset is None:
        command = ['sudo', 'date', f"{msg[10:12]}{msg[8:10]}{msg[2:4]}{msg[4:6]}{msg[12:16]}.{msg[6:8]}"]
        logger.info('Setting system date with command: %s', ' '.join(command))
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if stdout:
                logger.info('Date command output: %s', stdout.decode().strip())
            if stderr:
                logger.error('Error output: %s', stderr.decode().strip())
            if process.returncode == 0:
                tmset = True
            else:
                logger.error('Failed to set date with return code: %d', process.returncode)
        except Exception as e:
            logger.error('Unexpected error while setting date: %s', str(e))


last_msg_635 = None
light_status = None

@handle_errors
async def process_canid_635(msg):
    global light_status, last_msg_635

    # Check if this is the first received message
    first_message = last_msg_635 is None

    if first_message or msg != last_msg_635:
        # Extract byte 2 (light status from CAN)
        light_value = int(msg[2:4], 16)

        # Determine new status (0 = off, 1 = on)
        if light_value > 0:
            new_light_status = 1
        else:
            new_light_status = 0

        # If first run or status has changed -> set mode
        if first_message or (new_light_status != light_status):
            if change_dark_mode_by_car_light and api_is_connected:
                if new_light_status == 1:
                    mode = "night"
                else:
                    mode = "day"

                logger.info(f"Light status changed: Setting {mode} mode immediately.")
                event_handler.send_day_night(mode)

        # Update cached values
        light_status = new_light_status
        last_msg_635 = msg



@handle_errors
async def process_canid_65F(msg):
    global car_model_set, carmodel, FIS1, FIS2

    if msg[0:2] == '01' and car_model_set is None:
        carmodel = bytes.fromhex(msg[8:12]).decode()
        carmodelyear = await translate_caryear(bytes.fromhex(msg[14:16]).decode())

        # handle US version model number "FM" of the Audi A3 8P as "8P" model
        if carmodel == "FM":
            carmodel = "8P"

        car_models = {
            '8E': ('Audi A4', '265', '267'),
            '8J': ('Audi TT', '667', '66B'),
            '8L': ('Audi A3', '667', '66B'),
            '8P': ('Audi A3', '667', '66B'),
            '42': ('Audi R8', '265', '267'),
        }
        model_info = car_models.get(carmodel[0:2], ('unknown car model', 'unknown', 'unknown'))
        carmodelfull, FIS1, FIS2 = model_info
        logger.info("")
        logger.info('The car model and car model year were successfully read from the CAN-Bus.')
        logger.info('CAR = %s %s %s', carmodelfull, carmodel, carmodelyear)
        logger.info('FIS1 = %s / FIS2 = %s', FIS1, FIS2)
        logger.info("")
        car_model_set = True


@handle_errors
async def process_canid_661(msg):
    global tv_mode_active, send_on_canbus, deactivate_overwrite_dis_content
    if msg in ['8101123700000000', '8301123700000000']:
        if tv_mode_active == 0:
            device.emit(uinput.KEY_X, 1)
            device.emit(uinput.KEY_X, 0)
            logger.info('RNS-E is (back) in TV mode - play media - Keyboard: "X" - Hudiy/OpenAuto: "play"')
            tv_mode_active = 1
            if only_send_if_radio_is_in_tv_mode:
                send_on_canbus = True
                deactivate_overwrite_dis_content = False
    else:
        if tv_mode_active == 1:
            device.emit(uinput.KEY_C, 1)
            device.emit(uinput.KEY_C, 0)
            logger.info('RNS-E is not in TV mode (anymore) - pause media - Keyboard: "C" - Hudiy/OpenAuto: "pause"')
            tv_mode_active = 0
            if only_send_if_radio_is_in_tv_mode:
                send_on_canbus = False
                deactivate_overwrite_dis_content = True


async def toggle_fis2_label():
    global value_of_toggle_fis2
    value_of_toggle_fis2 = {
        0: '',
        1: 'TITLE',
        2: 'ARTIST',
        3: 'ALBUM',
        4: 'POSITION',
        5: 'DURATION',
        6: 'SPEED',
        7: 'RPM',
        8: 'COOLANT',
        9: 'CPU/TEMP',
        10: f'{lower_speed}-{upper_speed}',
        11: 'OUTSIDE',
        12: 'CLEAR',
        13: 'DISABLE'
    }.get(toggle_fis2, None)


#asyncio.run(toggle_fis2_label())



# Globals (einmalig oben definieren)
candump_proc: asyncio.subprocess.Process | None = None
candump_lock = asyncio.Lock()

@handle_errors
async def candump():
    """Toggle candump start/stop (robuste, nicht blockierende Variante)."""
    global candump_proc, pause_fis1, pause_fis2

    async with candump_lock:
        # Läuft schon?
        if candump_proc and (candump_proc.returncode is None):
            # --- STOP ---
            logger.info("Stopping candump now")
            try:
                candump_proc.terminate()  # SIGTERM
                try:
                    await asyncio.wait_for(candump_proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.info("candump did not terminate in time, killing...")
                    candump_proc.kill()
                    await candump_proc.wait()

                rc = candump_proc.returncode
                # 0 (normal), 130 (SIGINT), 143 (SIGTERM) → kein Fehler
                if rc not in (0, 130, 143):
                    logger.error("candump exit code after stop: %s", rc)
                else:
                    logger.info("candump stopped (return code: %s)", rc)
            finally:
                candump_proc = None

            # HUDIY/FIS Feedback
            if send_on_canbus and can_functional:
                pause_fis1 = True; pause_fis2 = True
                await clear_content(FIS1); await clear_content(FIS2)
                await align_center(FIS1, "CANDUMP"); await align_center(FIS2, "STOP")
                await asyncio.sleep(2)
                await clear_content(FIS1); await clear_content(FIS2)
                pause_fis1 = False; pause_fis2 = False

            return

        # --- START ---
        logger.info("Starting candump now")
        now = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        log_dir = f"{path}/candumps"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f"{log_dir}/{now}-candump-{can_interface}.txt"

        command = ['sudo', 'candump', can_interface, '-tA']
        try:
            # Logfile im Append/Write öffnen und Handle behalten solange der Prozess lebt
            log_file = open(log_file_path, 'w')
            candump_proc = await asyncio.create_subprocess_exec(
                *command, stdout=log_file, stderr=log_file
            )
            logger.info("candump started (pid=%s), writing to %s", candump_proc.pid, log_file_path)

            # Watcher-Task: wartet im Hintergrund auf Prozessende und räumt auf
            async def _watch():
                nonlocal log_file
                rc = await candump_proc.wait()
                try:
                    log_file.flush()
                finally:
                    log_file.close()
                # 0/130/143 → normal; sonst Fehler
                if rc in (0, 130, 143):
                    logger.info("candump exited (return code: %s)", rc)
                else:
                    logger.error("candump exited with error (return code: %s)", rc)
                # Prozess ist vorbei → Handle löschen
                # Lock bewusst nicht hier benutzen, um Deadlocks zu vermeiden
                # Kleiner Schutz: nur löschen, wenn noch identisch
                if 'candump_proc' in globals():
                    # Achtung: könnte zwischenzeitlich neu gestartet worden sein
                    # → nur None setzen, wenn es derselbe war
                    pass

            # Task feuern (nicht awaiten!)
            asyncio.create_task(_watch())

        except Exception as e:
            logger.error('Unexpected error while starting candump: %s', str(e))
            # Aufräumen
            if candump_proc and candump_proc.returncode is None:
                try:
                    candump_proc.kill()
                    await candump_proc.wait()
                except Exception:
                    pass
            candump_proc = None
            return

        # HUDIY/FIS Feedback
        if send_on_canbus and can_functional:
            pause_fis1 = True; pause_fis2 = True
            await clear_content(FIS1); await clear_content(FIS2)
            await align_center(FIS1, "CANDUMP"); await align_center(FIS2, "START")
            await asyncio.sleep(3)
            await clear_content(FIS1); await clear_content(FIS2)
            pause_fis1 = False; pause_fis2 = False



@handle_errors
async def block_show_value1():
    global toggle_fis1, pause_fis1, begin1, end1, last_speed, last_rpm, last_coolant, last_outside_temp, pause_fis2

    if send_on_canbus and can_functional:
        await clear_content(FIS1)
    toggle_fis1 += 1
    data = {
        0: '', 1: 'TITLE', 2: 'ARTIST', 3: 'ALBUM', 4: 'POSITION', 5: 'DURATION',
        6: 'SPEED', 7: 'RPM', 8: 'COOLANT', 9: 'CPU/TEMP',
        10: f'{lower_speed}-{upper_speed}', 11: 'OUTSIDE', 12: 'CLEAR', 13: 'DISABLE'
    }.get(toggle_fis1, None)
    logger.info(f"toggle_fis1 changed to {toggle_fis1} ({data})")
    if data:
        if deactivate_overwrite_dis_content:
            await asyncio.sleep(2)
        if send_on_canbus and can_functional:
            await align_center(FIS1, data)
            begin1 = -1
            end1 = 7
            await asyncio.sleep(2)
            await clear_content(FIS1)
    if data == 'SPEED':
        last_speed = None
    elif data == 'RPM':
        last_rpm = None
    elif data == 'COOLANT':
        last_coolant = None
    elif data == 'OUTSIDE':
        last_outside_temp = None
    elif data == 'DISABLE':
        toggle_fis1 = 0
        if send_on_canbus and can_functional:
            await align_center(FIS1, data)
            pause_fis2 = True
            await asyncio.sleep(2)
            await clear_content(FIS1)
            await clear_content(FIS2)
            await asyncio.sleep(1)
            pause_fis2 = False
    pause_fis1 = False


@handle_errors
async def block_show_value2():
    global toggle_fis2, pause_fis2, begin2, end2, last_speed, last_rpm, last_coolant, last_outside_temp, pause_fis1

    if send_on_canbus and can_functional:
        await clear_content(FIS2)
    toggle_fis2 += 1
    data = {
        0: '', 1: 'TITLE', 2: 'ARTIST', 3: 'ALBUM', 4: 'POSITION', 5: 'DURATION',
        6: 'SPEED', 7: 'RPM', 8: 'COOLANT', 9: 'CPU/TEMP',
        10: f'{lower_speed}-{upper_speed}', 11: 'OUTSIDE', 12: 'CLEAR', 13: 'DISABLE'
    }.get(toggle_fis2, None)
    logger.info(f"toggle_fis2 changed to {toggle_fis2} ({data})")

    if data:
        if deactivate_overwrite_dis_content:
            await asyncio.sleep(2)
        if send_on_canbus and can_functional and not show_label:
            await align_center(FIS2, data)
            begin2 = -1
            end2 = 7
            await asyncio.sleep(2)
            await clear_content(FIS2)
            pause_fis2 = False
        elif send_on_canbus and can_functional and show_label:
            await toggle_fis2_label()
    if data == 'SPEED':
        last_speed = None
    elif data == 'RPM':
        last_rpm = None
    elif data == 'COOLANT':
        last_coolant = None
    elif data == 'OUTSIDE':
        last_outside_temp = None
    elif data == 'DISABLE':
        #toggle_fis2 = 0
        if send_on_canbus and can_functional:
            await align_center(FIS2, data)
            pause_fis1 = True
            await asyncio.sleep(2)
            await clear_content(FIS1)
            await clear_content(FIS2)
            await asyncio.sleep(1)
            pause_fis1 = False
            toggle_fis2 = 0
        else:
            toggle_fis2 = 0
    pause_fis2 = False


scroll_task_fis1 = None
scroll_task_fis2 = None
scrolling_active_fis1 = False
scrolling_active_fis2 = False


@handle_errors
async def media_to_dis1():
    global title, artist, album, position, duration, playing, last_data, scrolling_active_fis1
    rule1 = {1: title, 2: artist, 3: album, 4: position, 5: duration}.get(toggle_fis1, "")
    try:
        scrolling_active_fis1 = False
        await start_scrolling(rule1, "FIS1")
    except Exception as e:
        logger.error(f"Error in media_to_dis1: {e}")


last_value_of_toggle_fis2 = None


@handle_errors
async def media_to_dis2():
    global title, artist, album, position, duration, playing, last_data, scrolling_active_fis2, last_value_of_toggle_fis2
    rule2 = {1: title, 2: artist, 3: album, 4: position, 5: duration}.get(toggle_fis2, "")
    try:
        if show_label and value_of_toggle_fis2 != last_value_of_toggle_fis2:
            await align_center(FIS1, value_of_toggle_fis2)
            last_value_of_toggle_fis2 = value_of_toggle_fis2
        scrolling_active_fis2 = False
        await start_scrolling(rule2, "FIS2")
    except Exception as e:
        logger.error(f"Error in media_to_dis2: {e}")


async def start_scrolling(rule, display_type):
    global scroll_task_fis1, scroll_task_fis2, max_length, scroll_type
    max_length = 8
    wait_time = 3
    delay = 0.25
    try:
        if display_type == "FIS1":
            if scroll_task_fis1 and not scroll_task_fis1.done():
                scroll_task_fis1.cancel()
            if scroll_type == "scroll":
                scroll_task_fis1 = asyncio.create_task(_scroll_text(rule, wait_time, delay, FIS1), name="scroll_text_fis1")
            elif scroll_type == "oem_style":
                scroll_task_fis1 = asyncio.create_task(_scroll_oem_style(rule, wait_time, FIS1), name="scroll_oem_fis1")
            await scroll_task_fis1
        elif display_type == "FIS2":
            if scroll_task_fis2 and not scroll_task_fis2.done():
                scroll_task_fis2.cancel()
            if scroll_type == "scroll":
                scroll_task_fis2 = asyncio.create_task(_scroll_text(rule, wait_time, delay, FIS2), name="scroll_text_fis2")
            elif scroll_type == "oem_style":
                scroll_task_fis2 = asyncio.create_task(_scroll_oem_style(rule, wait_time, FIS2), name="scroll_oem_style_fis2")
            await scroll_task_fis2
    except asyncio.CancelledError:
        pass


async def _scroll_text(rule1, wait_time, delay, display):
    text_length = len(rule1)
    text_with_padding = ' ' * max_length + rule1 + ' ' * max_length
    reset_scroll = True
    current_rule = None

    while True:
        if (display == FIS1 and toggle_fis1 not in (1, 2, 3, 4, 5)) or (
                display == FIS2 and toggle_fis2 not in (1, 2, 3, 4, 5)):
            await (clear_content(FIS1) if display == FIS1 else clear_content(FIS2))
            return
        while (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
            await asyncio.sleep(0.5)
        if rule1 != current_rule:
            current_rule = rule1
            last_data = None
            reset_scroll = True
        if text_length > max_length:
            start_index = max_length
            end_index = text_length + max_length
            if reset_scroll:
                await align_center(display, text_with_padding[start_index:start_index + max_length])
                await asyncio.sleep(wait_time)
                reset_scroll = False
            i = start_index
            while i < end_index:
                if (display == FIS1 and toggle_fis1 not in (1, 2, 3, 4, 5)) or (
                        display == FIS2 and toggle_fis2 not in (1, 2, 3, 4, 5)):
                    await (clear_content(FIS1) if display == FIS1 else clear_content(FIS2))
                    return
                while (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    reset_scroll = True
                    await asyncio.sleep(0.5)
                if not reset_scroll:
                    data = text_with_padding[i:i + max_length]
                    if data != last_data:
                        await align_center(display, data)
                        last_data = data
                    await asyncio.sleep(delay)
                    i += 1
            reset_scroll = True
        else:
            if rule1 != last_data:
                if (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    await asyncio.sleep(0.5)
                    continue
                await align_center(display, rule1)
                last_data = rule1
                await asyncio.sleep(wait_time)
            else:
                await align_center(display, rule1)
            last_data = rule1
            await asyncio.sleep(0.95)


async def _scroll_oem_style(rule1, wait_time, display):
    last_data_fis1 = None
    last_data_fis2 = None
    current_rule_fis1 = None
    current_rule_fis2 = None
    segments = textwrap.wrap(rule1, max_length)
    segment_index1 = 0
    reset_scroll = True

    while True:
        if (display == FIS1 and toggle_fis1 not in (1, 2, 3, 4, 5)) or (
                display == FIS2 and toggle_fis2 not in (1, 2, 3, 4, 5)):
            await (clear_content(FIS1) if display == FIS1 else clear_content(FIS2))
            return
        while (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
            await asyncio.sleep(0.5)
        if rule1 != (current_rule_fis1 if display == FIS1 else current_rule_fis2):
            if display == FIS1:
                current_rule_fis1 = rule1
                segment_index1 = 0
            else:
                current_rule_fis2 = rule1
                segment_index1 = 0
            reset_scroll = True
        if segments:
            segment = segments[segment_index1]
            if reset_scroll:
                await align_center(display, segment)
                await asyncio.sleep(wait_time)
                reset_scroll = False
            if segment != (last_data_fis1 if display == FIS1 else last_data_fis2):
                if (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    await asyncio.sleep(0.5)
                    continue
                await align_center(display, segment)
                if display == FIS1:
                    last_data_fis1 = segment
                else:
                    last_data_fis2 = segment
            await asyncio.sleep(wait_time)
            segment_index1 = (segment_index1 + 1) % len(segments)
        else:
            if rule1 != (last_data_fis1 if display == FIS1 else last_data_fis2):
                if (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    await asyncio.sleep(0.5)
                    continue
                await align_center(display, rule1)
                if display == FIS1:
                    last_data_fis1 = rule1
                else:
                    last_data_fis2 = rule1
                await asyncio.sleep(wait_time)
            else:
                await align_center(display, rule1)
            if display == FIS1:
                last_data_fis1 = rule1
            else:
                last_data_fis2 = rule1
            await asyncio.sleep(0.95)


@handle_errors
async def send_to_dis(speed_unit, temp_unit, display):
    global script_started, speed_measure_to_api, elapsed_time_formatted, show_label
    global toggle_fis1, toggle_fis2, pause_fis1, pause_fis2  # <-- WICHTIG
    sleep_values, start_time, measure_done, data, last_data, drop = 0.5, None, 0, '', '', 0

    if ENABLE_LOGGING:
        logger.info(f"Task send_to_dis ({display}) was started.")
    while not stop_flag:
        fis_mapping = {
            FIS1: (FIS1, toggle_fis1, pause_fis1),
            FIS2: (FIS2, toggle_fis2, pause_fis2),
        }
        FIS, toggle_fis, pause_fis = fis_mapping.get(display, (None, None, None))
        try:
            if stop_flag:
                break
            if not (send_on_canbus and can_functional and script_started):
                await asyncio.sleep(0.5)
                continue
            if pause_fis:
                await asyncio.sleep(0.5)
                continue
            if display == FIS1 and show_label:
                await asyncio.sleep(1)
                continue
            if (not send_api_mediadata_to_dashboard and toggle_fis in (1, 2, 3, 4, 5)) or (
                    not send_values_to_dashboard and toggle_fis in (6, 7, 8, 9, 10, 11, 12)):
                sleep_values = 2.0
                data = 'DISABLED'
                await align_right(FIS, data)
                await asyncio.sleep(sleep_values)
                continue
            else:
                await asyncio.sleep(sleep_values)

            if toggle_fis == 12:
                data, sleep_values = '', 2.0
            if data != last_data:
                if toggle_fis == 10:
                    drop += 1
                    if drop == 3 or measure_done or start_time is None:
                        if send_to_api_gauges:
                            event_handler.update_to_api("getPidValue(7)", speed_measure_to_api, "speed_measure")
                        await align_right(FIS, data)
                        drop = 0
                else:
                    await align_right(FIS, data)
            last_data = data
            await asyncio.sleep(sleep_values)
        except asyncio.CancelledError:
            if ENABLE_LOGGING:
                logger.info(f"Task send_to_dis ({display}) was stopped.")
            break


# using a translation table from the VIN decoding list:
# https://www.nininet.de/deutsch/Fahrgestellnummer-entschluesseln.php
async def translate_caryear(carmodelyear):
    """Translates the model year of a vehicle into the corresponding year."""
    translation_table = {
        'V': 1997, 'W': 1998, 'X': 1999, 'Y': 2000,
        '1': 2001, '2': 2002, '3': 2003, '4': 2004,
        '5': 2005, '6': 2006, '7': 2007, '8': 2008,
        '9': 2009, 'A': 2010, 'B': 2011, 'C': 2012,
        'D': 2013, 'E': 2014, 'F': 2015, 'G': 2016,
        'H': 2017, 'J': 2018, 'K': 2019, 'L': 2020,
        'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
        'S': 2025, 'T': 2026,
    }
    return translation_table.get(carmodelyear, None)


stop_script_running = False
remote_server_shutdown_event = None

async def stop_script():
    global stop_script_running, api_is_connected, stop_flag, event_handler, server_socket, remote_task, remote_server_shutdown_event

    if stop_script_running:
        return
    stop_script_running = True

    # Hilfsfunktion: robusten Namen für ein Task-Objekt ermitteln
    def _task_label(t: "asyncio.Task"):
        # Versuche zuerst den Namen der Coroutine (__name__), dann Task.get_name(), sonst "task"
        coro = getattr(t, "_coro", None)
        name = getattr(coro, "__name__", None)
        if name:
            return name
        if hasattr(t, "get_name"):
            try:
                return t.get_name()
            except Exception:
                pass
        return "task"

    try:
        logger.info("Stopping script...")

        if ENABLE_LOGGING:
            for task in asyncio.all_tasks():
                logger.info(f"Found running task: {_task_label(task)}")

        # 🛑 first set stop_flag
        if ENABLE_LOGGING:
            logger.info("setting stop_flag and wait 2 seconds to gently close running tasks")
        stop_flag = True
        await asyncio.sleep(2)

        if ENABLE_LOGGING:
            for task in asyncio.all_tasks():
                logger.info(f"Found running task: {_task_label(task)}")

        # Remote-Control sauber herunterfahren
        if remote_task:
            logger.info("🚦 Triggering shutdown of remote_control")
            if remote_server_shutdown_event:
                remote_server_shutdown_event.set()

            try:
                await asyncio.wait_for(remote_task, timeout=5)
                logger.info("✅ remote_control_task stopped cleanly.")
            except asyncio.TimeoutError:
                logger.warning("⚠️ remote_control_task did not stop in time.")
            except asyncio.CancelledError:
                logger.warning("⚠️ remote_control_task was force cancelled.")
            except Exception as e:
                logger.error(f"❌ remote_control_task stop error: {e}", exc_info=True)

        # 🎯 relevante Tasks einsammeln (ohne auf _coro zuzugreifen)
        wanted = {"read_cpu", "receive_messages", "handle_client", "remote_control"}
        current = asyncio.current_task()
        tasks = []
        for task in asyncio.all_tasks():
            if task is current:
                continue
            if _task_label(task) in wanted:
                tasks.append(task)

        # Cancel & warten
        if tasks:
            for task in tasks:
                task.cancel()
            for task in tasks:
                lbl = _task_label(task)
                try:
                    await task
                    logger.info(f"✅ Task {lbl} was stopped.")
                except asyncio.CancelledError:
                    logger.info(f"✋ Task {lbl} was cancelled.")
                except Exception as e:
                    logger.error(f"❌ Error while waiting for task {lbl}: {e}", exc_info=True)
        else:
            logger.info("ℹ️ No matching tasks were running or all already stopped.")

        # 🔌 disconnect API
        if api_is_connected:
            try:
                loop = asyncio.get_running_loop()
                # client.disconnect ist vermutlich blockierend → Executor
                await loop.run_in_executor(None, client.disconnect)
                api_is_connected = False
                logger.info(f"Successfully disconnected from {backend} API.")
            except Exception as e:
                logger.error(f"Error while disconnecting from {backend} API.", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Task stop_script was cancelled.")
        raise
    finally:
        stop_script_running = False


@handle_errors
async def kill_script():
    current_pid = os.getpid()
    logger.info(f"Killing current script instance with PID: {current_pid}")
    command = f'sudo kill -15 {current_pid}'
    result = await run_command(command, log_output=False)
    if result["stderr"]:
        logger.error("Failed to kill the script: %s", result["stderr"])
    else:
        logger.info("Script killed successfully.")


notifier_started = False


@handle_errors
async def get_can_messages():
    global bus, notifier, notifier_started
    if notifier_started:
        logger.warning("CAN-Notifier is already running.")
        return
    notifier_started = True

    async def message_callback(msg):
        await read_on_canbus(msg)

    try:
        loop = asyncio.get_running_loop()
        notifier = can.Notifier(bus, [message_callback], loop=loop)
        if ENABLE_LOGGING:
            logger.info("CAN-Notifier started.")
            logger.info("")
        while not stop_flag:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        if ENABLE_LOGGING:
            logger.info("Task get_can_messages was stopped.")
    finally:
        if notifier:
            notifier.stop()
            if ENABLE_LOGGING:
                logger.info("CAN-Notifier stopped.")



def unwrap_coroutine(coro):
    """
    Tries to get the original coroutine function from wrapped coroutines.
    """
    base = coro
    while True:
        func = getattr(base, "__wrapped__", None)
        if func is None:
            break
        base = func

    name = getattr(base, "__name__", str(base))
    qualname = getattr(base, "__qualname__", name)
    return name, qualname




async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if not tasks:
        logger.info("No running tasks to stop.")
        return
    logger.info(f"Stopping {len(tasks)} running tasks...")
    for task in tasks:
        task.cancel()
    await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    logger.info("All tasks have been stopped.")


async def main():
    global camera, stop_flag, script_started, version, PackageNotFoundError, remote_task

    try:
        await init_async_primitives()  # <<< HIER EINFÜGEN
        #general startup tasks:
        #write header to console
        await write_header_to_log_file(log_filename)


        load_features()
        await toggle_fis2_label()

        #check if script is running with python3.13.5. If not, start install_python_3_13_5(), set_python3_alias(), restart_with_python_3_13_5()
        await ensure_python_3_11_2()
        #check if openauto or hudiy is installed/used
        await detect_installs()
        #console output "Script is starting..."
        await start_script()
        #check if the script is already running. If so, kill the old istance(s)
        await is_script_running(script_filename)
        #check importlib.metadata import with fallback
        version, PackageNotFoundError = await ensure_importlib(logger)
        #check for missing python3 (pip) packages. If missing, install them.

        await packaging_globals(logger=logger, enable_logging=ENABLE_LOGGING)

        await modules_inst_import()

        #check if can-utils ist already installed. If installed, check the version. If outdated remove and instal vi git.
        await check_can_utils()
        #check the uinput permissions, so that simulated keyboards controlls can work (Left, Right, Enter...)
        await uinput_permissions()
        #check the cpu powerplan and set to "ondemand" if it's in powersave mode
        await set_powerplan()

        #check if OpenAuto Pro API files are already downloaded. If not, download, extract and import.
        await check_import_api()
        #check if the units (km/h / rpm and °C / °F) are set correctly in openauto_obd_gauges.ini for OAP Dashboard view.
        if backend == "OpenAuto":
            await oap_units_check(temp_unit, speed_unit, lower_speed, upper_speed)
            if reversecamera_by_reversegear or reversecamera_by_down_longpress:
                cam_init(reversecamera_guidelines)  # genau ein Warmstart, abhängig vom Flag

        #start the remote control task to shutdown other running scripts on startup or via network controll website
        remote_task = asyncio.create_task(remote_control(), name="remote_control")
        tasks.append(remote_task)
        #test the can-interface and see if can-messages are getting received
        await test_can_interface()
        if send_on_canbus and can_functional:
            await welcome_message()

        script_started = True

        # conditional tasks
        #start api connection if api features are enabled
        if send_to_api_gauges or (
                (send_api_mediadata_to_dashboard or change_dark_mode_by_car_light) and bus is not None
        ):
            tasks.append(asyncio.create_task(api_connection(), name="receive_messages"))

        if send_to_api_gauges and backend == "Hudiy":
            # Use your chosen base
            base_dir = Path("/home/pi/scripts")
            js_root = ensure_hudiy_js_tree(
                base_dir,
                logger=logger,
                ENABLE_LOGGING=ENABLE_LOGGING,
                send_to_api_gauges=send_to_api_gauges,  # will list files if True
                auto_download_common=True,  # download 'common' only if missing
                create_data_dir=False  # useful if you write data/live.json
            )


        # if can-bus is running, start reading messages
        if bus and can_functional:
            tasks.append(asyncio.create_task(get_can_messages(), name="get_can_messages"))
            # initialise reversecamera if user has enabled this feature
            #activate rns-e tv input if user has enabled this feature
            if activate_rnse_tv_input and send_on_canbus:
                tasks.append(asyncio.create_task(send_tv_input(), name="send_tv_input"))
            #if user has allowed sending on can-bus, check wich features are enabled
            if send_on_canbus:
                #if user has allowed sending values (speed, rpm,...) or sending mediadata (title, artist,...) to dashboard, start the function to overwrite dis/fis via hands free channel
                if send_values_to_dashboard or send_api_mediadata_to_dashboard:
                    tasks.append(asyncio.create_task(overwrite_dis(), name="overwrite_dis"))
                tasks.append(asyncio.create_task(send_to_dis(speed_unit, temp_unit, FIS1), name="send_to_dis_1"))
                tasks.append(asyncio.create_task(send_to_dis(speed_unit, temp_unit, FIS2), name="send_to_dis_2"))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"Task '{task.get_name()}' raised: {result}")

    except asyncio.CancelledError:
        logger.info("Task main was cancelled.")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in main(): {e}", exc_info=True)

    finally:
        asyncio.create_task(stop_script())

        if shutdown_script:
            logger.info("Initiating system shutdown...")
            await run_command("sudo shutdown -h now", log_output=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())  # starts the script
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping script...")
    except asyncio.CancelledError:
        logger.info("Main coroutine cancelled during shutdown.")
        pass
