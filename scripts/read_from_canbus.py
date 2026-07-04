#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Project on GitHub: https://github.com/noobychris/audi-can-rpi
script_version = "v1.0"

# If you have any trouble with the script, you can enable LOGGING_OUTPUT to get more information's about exceptions.
# The messages will be saved in the script/logs folder with date, name of the script.
# Example: 2023-02-01_read_from_canbus_errors.log


from dataclasses import dataclass
from typing import List, Iterable, Dict, Union, Tuple, TYPE_CHECKING, Optional, cast
try:
    from typing import Protocol
except ImportError:
    # Python 3.7 has no typing.Protocol. Runtime fallback is sufficient here;
    # Protocol is only used for the optional stop() type declaration below.
    Protocol = object
import binascii, configparser, contextlib, importlib, importlib.util, inspect, io, logging, os, re, shutil
import socket, struct, subprocess, sys, sysconfig, tempfile, textwrap, threading, time, traceback, zipfile
import json, asyncio, shlex
from datetime import datetime
from functools import wraps, partial
from pathlib import Path


# --- Python 3.7+ compatible venv bootstrap ---

MIN_SUPPORTED_PYTHON = (3, 7)
MAX_TESTED_PYTHON = (3, 13)
CURRENT_PYTHON_VERSION = (sys.version_info.major, sys.version_info.minor)

VENV_PATH = os.path.expanduser("~/.venv-canbus")
VENV_PY = os.path.join(VENV_PATH, "bin", "python")


def _in_venv():
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def _venv_python_version():
    """Return the venv interpreter's (major, minor), or None if unusable."""
    if not os.path.isfile(VENV_PY) or not os.access(VENV_PY, os.X_OK):
        return None
    try:
        output = subprocess.check_output(
            [
                VENV_PY,
                "-c",
                "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))",
            ],
            universal_newlines=True,
        ).strip()
        major, minor = output.split(".", 1)
        return int(major), int(minor)
    except Exception:
        return None


def _venv_matches_current_python():
    return _venv_python_version() == CURRENT_PYTHON_VERSION


def _bootstrap_venv():
    if CURRENT_PYTHON_VERSION < MIN_SUPPORTED_PYTHON:
        print(
            "🚫 Unsupported Python version {}.{}. Python 3.7 or newer is required.".format(
                sys.version_info.major,
                sys.version_info.minor,
            ),
            flush=True,
        )
        sys.exit(1)

    if CURRENT_PYTHON_VERSION > MAX_TESTED_PYTHON:
        print(
            "⚠️ Python {}.{} is newer than the latest tested version {}.{}. "
            "The script will continue, but package compatibility is not guaranteed.".format(
                CURRENT_PYTHON_VERSION[0],
                CURRENT_PYTHON_VERSION[1],
                MAX_TESTED_PYTHON[0],
                MAX_TESTED_PYTHON[1],
            ),
            flush=True,
        )

    if _in_venv():
        return

    if not _venv_matches_current_python():
        if os.path.isdir(VENV_PATH):
            print(
                "♻️ Existing virtual environment uses a different Python version; "
                "rebuilding {} for Python {}.{} ...".format(
                    VENV_PATH,
                    sys.version_info.major,
                    sys.version_info.minor,
                ),
                flush=True,
            )
            shutil.rmtree(VENV_PATH)
        else:
            print(
                "📦 Creating virtual environment at {} for Python {}.{} ...".format(
                    VENV_PATH,
                    sys.version_info.major,
                    sys.version_info.minor,
                ),
                flush=True,
            )

        try:
            subprocess.check_call(
                [sys.executable, "-m", "venv", "--system-site-packages", VENV_PATH]
            )
        except subprocess.CalledProcessError:
            print(
                "🚫 Could not create the virtual environment. Install the matching "
                "python3-venv package for this OS and try again.",
                flush=True,
            )
            raise

        # Keep bootstrap tools compatible with the interpreter. Pip will select
        # versions whose Python-Requires metadata matches Python 3.7/3.11/3.13.
        subprocess.call(
            [VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"]
        )

    print(
        "🔁 Restarting the script inside {} (Python {}.{})...".format(
            VENV_PATH,
            sys.version_info.major,
            sys.version_info.minor,
        ),
        flush=True,
    )
    print()
    os.execv(VENV_PY, [VENV_PY] + sys.argv)


_bootstrap_venv()

# --- end bootstrap ---

BASE_PATH = Path(__file__).resolve().parent
logs_root = BASE_PATH / "logs"
FEATURE_FILE = str(BASE_PATH / "features.conf")
SPEED_MEASURE_LOG_FILE = logs_root / "speed_measurements.txt"

# ---------------------------
# Paths / script metadata
# ---------------------------
path = os.path.dirname(os.path.abspath(__file__))
script_filename = str(os.path.basename(__file__))
script_fullpath = os.path.realpath(__file__)


# ---------------------------
# Status / Configuration
# ---------------------------
welcome_active = False
stop_flag = False
shutdown_script = False
script_started = False
api_is_connected = False
send_to_dis_tasks_started = False

backend = None
tmset = None
car_model_set = None
carmodel = ''

version = None
PackageNotFoundError = None

# ---------------------------
# FIS / display configuration
# ---------------------------
FIS1 = '265'
FIS2 = '267'
pause_fis1 = True
pause_fis2 = True
light_set = False
deactivate_overwrite_dis_content = False
overwrite_dis_hold_visible_until = 0.0
overwrite_dis_reactivation_guard = False

task_send_fis1 = None
task_send_fis2 = None

begin1 = -1
end1 = 7
begin2 = -1
end2 = 7

# ---------------------------
# FIS sender state
# ---------------------------
fis1_pending = None   # (text, align)
fis2_pending = None

fis1_last_sent = None
fis2_last_sent = None

fis1_update_event: Optional[asyncio.Event] = None
fis2_update_event: Optional[asyncio.Event] = None

current_task_fis1 = None
current_task_fis2 = None

# ---------------------------
# CAN send timing
# ---------------------------
speed_send_interval = 0.2          # seconds
rpm_send_interval = 0.2            # seconds
coolant_send_interval = 1.0        # seconds
outside_temp_send_interval = 1.0  # seconds
speed_measure_send_interval = 0.3  # seconds
cpu_read_interval = 3.0  # seconds

last_speed_send_time = 0.0
last_rpm_send_time = 0.0
last_coolant_send_time = 0.0
last_outside_temp_send_time = 0.0
last_speed_measure_send_time = 0.0

last_speed_api_send_time = 0.0
last_rpm_api_send_time = 0.0
last_coolant_api_send_time = 0.0
last_outside_temp_api_send_time = 0.0

last_sent_speed = None
last_sent_rpm = None
last_sent_coolant = None
last_sent_outside_temp = None

pending_speed_display = None
pending_speed_api = None
pending_outside_temp_display = None
pending_outside_temp_api = None
pending_rpm_display = None
pending_rpm_api = None
pending_coolant_display = None
pending_coolant_api = None

# ---------------------------
# Live CAN values
# ---------------------------
speed = 0
rpm = 0
coolant = 0
outside_temp = ""

last_speed = None
last_rpm = None
last_coolant = None
last_outside_temp = None
last_msg_271_2C3 = None

last_msg_635 = None
light_status = None

can_functional = None
guidelines_set = False
camera_active = False
gear = 0

# ---------------------------
# Vehicle model / VIN state
# ---------------------------
vin_parts = {}
vin_set = False
vin_display = None

model_info_set = False
carmodelyear_cache = None
carmodelfull_cache = None

# ---------------------------
# Camera / TV state
# ---------------------------
CAM_WITH_OVERLAY = False
tv_input_activation_detected = False
cam: Optional["Cam"] = None
reverse_camera_off_task: Optional[asyncio.Task] = None

# ---------------------------
# Websocket / API state
# ---------------------------
hudiy_ws_hub: Optional["HudiyDashboardHub"] = None
ws_server = None
ws_client = None

# ---------------------------
# Media / projection state
# ---------------------------
playing = ''
position = ''
source = ''
title = ''
artist = ''
album = ''
duration = ''

state = None
ProjectionState = None
ProjectionSource = None
tv_mode_active = 1

last_media_status_perf: Optional[float] = None
last_media_position_change_perf: Optional[float] = None

media_position_task: Optional[asyncio.Task] = None
media_position_base_seconds: Optional[int] = None
media_position_anchor_monotonic: Optional[float] = None
media_position_last_displayed_seconds: Optional[int] = None
media_position_last_real_seconds: Optional[int] = None

# ---------------------------
# Scroll / async helper state
# ---------------------------
scroll_task_fis1: Optional[asyncio.Task] = None
scroll_task_fis2: Optional[asyncio.Task] = None
scrolling_active_fis1 = False
scrolling_active_fis2 = False
last_value_of_toggle_fis2 = None
candump_proc: Optional[asyncio.subprocess.Process] = None
candump_lock: Optional[asyncio.Lock] = None
cpu_task: Optional[asyncio.Task] = None

# ---------------------------
# Measurement state
# ---------------------------
elapsed_time = 0.0
measure_done = 0
data = None
last_data = ''
start_time = None
speed_measure_to_api = 0.00
# Hudiy speed_measure control protocol:
#   -1.0 = disabled/disarm
#    0.0 = reset/ready at lower_speed
#   -2.0 = start local gauge animation
#   >0.0 = final result; stop animation and show final value
hudiy_speed_measure_animation_started = False
# Only arm/start a 0-lower -> upper measurement after the car was actually at/below lower_speed.
# Prevents invalid short measurements when switching toggle_fis1/2 to speed_measure while already driving.
speed_measure_armed = False

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

# ---------------------------
# System metrics / processes
# ---------------------------
cpu_load = 0
cpu_temp = 0
cpu_freq_mhz = 0

# ---------------------------
# Network / server state
# ---------------------------
server_socket = None
HTTP_CONTROL_PORT = 23456
remote_task: Optional[asyncio.Task] = None
remote_server_shutdown_event: Optional[asyncio.Event] = None
notifier = None
can_reader: Optional["can.AsyncBufferedReader"] = None
can_reader_task = None
bus: Optional["can.BusABC"] = None

# ---------------------------
# Threading / async flags
# ---------------------------
lock = threading.Lock()
stop_completed_event = threading.Event()
tasks = []
background_tasks = []

# ---------------------------
# Internal runtime flags
# ---------------------------
stop_script_running = False
notifier_started = False

# ---------------------------
# Internal caches
# ---------------------------
_cached_metadata = None

# ---------------------------
# Desk-Setup setting: For normal users please let this "False"
# ---------------------------
AUTO_FALLBACK_TO_VCAN = False
AUTOSEND_CAR_MODEL = False


if TYPE_CHECKING:
    can_interface: str
    send_on_canbus: bool
    only_send_if_radio_is_in_tv_mode: bool
    show_label: bool
    toggle_fis1: int
    toggle_fis2: int
    lower_speed: int
    upper_speed: int
    export_speed_measurements_to_file: bool
    scroll_type: str
    speed_unit: str
    temp_unit: str
    welcome_message_1st_line: str
    welcome_message_2nd_line: str
    read_and_set_time_from_dashboard: bool
    control_pi_by_rns_e_buttons: bool
    read_mfsw_buttons: bool
    send_values_to_dashboard: bool
    toggle_values_by_rnse_longpress: bool
    initial_day_night_mode: str
    change_dark_mode_by_car_light: bool
    send_api_mediadata_to_dashboard: bool
    send_to_api_gauges: bool
    reversecamera_by_reversegear: bool
    reversecamera_guidelines: bool
    reversecamera_turn_off_delay: int
    reversecamera_turn_off_speed: int
    reversecamera_by_prev_longpress: bool
    reversecamera_video_pipeline: str
    shutdown_via_can: str
    shutdown_type: str
    activate_rnse_tv_input: bool
    tv_input_format: str
    ENABLE_LOGGING: bool
    show_can_messages_in_logs: bool


class StoppableTask(Protocol):
    def stop(self) -> None:
        ...

    def modify_data(self, messages) -> None:
        ...


class Shutdownable(Protocol):
    def shutdown(self) -> None:
        ...


tv_input_task: Optional[StoppableTask] = None
task_overwrite_dis_content: Optional[StoppableTask] = None


@dataclass
class Features:
    can_interface: str = "can0"
    send_on_canbus: bool = True
    only_send_if_radio_is_in_tv_mode: bool = False
    show_label: bool = False
    toggle_fis1: int = 6
    toggle_fis2: int = 7
    lower_speed: int = 0
    upper_speed: int = 100
    export_speed_measurements_to_file: bool = True
    scroll_type: str = "oem_style"
    speed_unit: str = "km/h"
    temp_unit: str = "°C"
    welcome_message_1st_line: str = "WELCOME"
    welcome_message_2nd_line: str = "USER"
    read_and_set_time_from_dashboard: bool = True
    control_pi_by_rns_e_buttons: bool = True
    read_mfsw_buttons: bool = False
    send_values_to_dashboard: bool = True
    toggle_values_by_rnse_longpress: bool = True
    initial_day_night_mode: str = "night"
    change_dark_mode_by_car_light: bool = True
    send_api_mediadata_to_dashboard: bool = True
    send_to_api_gauges: bool = True
    reversecamera_by_reversegear: bool = False
    reversecamera_guidelines: bool = True
    reversecamera_turn_off_delay: int = 5
    reversecamera_turn_off_speed: int = 0  # 0 = disabled; speed unit follows speed_unit
    reversecamera_by_prev_longpress: bool = False
    reversecamera_video_pipeline: str = "yuyv_30"
    shutdown_via_can: str = "False"
    shutdown_type: str = "gently"
    activate_rnse_tv_input: bool = False
    tv_input_format: str = "NTSC"
    ENABLE_LOGGING: bool = False
    show_can_messages_in_logs: bool = False


# ------------------------------------------------------------
# Defaults zuerst ins globale Namespace
# ------------------------------------------------------------
_defaults = Features()
globals().update(vars(_defaults))
features = _defaults

# Früher Minimal-Logger, damit "logger" nie undefined ist
logger = logging.getLogger(__name__)
if not logger.handlers:
    _early_console_handler = logging.StreamHandler(sys.stdout)
    _early_console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_early_console_handler)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_value(val: str):
    val = val.strip()

    if len(val) >= 2 and ((val[0] == "'" and val[-1] == "'") or (val[0] == '"' and val[-1] == '"')):
        return val[1:-1]

    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False

    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def make_trace(func_name: str, lineno: int) -> str:
    return f"{func_name} (line {lineno})"


def extend_trace(trace: Optional[str], func_name: str, lineno: int) -> str:
    current = make_trace(func_name, lineno)
    return f"{trace} -> {current}" if trace else current


def _caller_trace(skip: Optional[set] = None) -> Optional[str]:
    if not ENABLE_LOGGING:
        return None
    skip = set(skip or ())
    skip.update({"_caller_trace", "make_trace", "extend_trace", "set_fis", "set_fis1", "set_fis2"})
    try:
        for frame in inspect.stack()[1:]:
            func = frame.function
            if func in skip:
                continue
            return make_trace(func, frame.lineno)
    except Exception:
        return None
    return None


class ThreadNameFilter(logging.Filter):
    def filter(self, record):
        if "can.notifier" in record.threadName:
            record.threadName = "can_notifier"
        elif record.threadName.startswith("ThreadPoolExecutor-"):
            record.threadName = "API_EventHandler"
        elif record.threadName.startswith("API_Receiver"):
            record.threadName = "API_Receiver"
        return True


class ContextualFormatter(logging.Formatter):
    def format(self, record):
        # Avoid inspect.stack(), which allocates FrameInfo objects and is costly
        # when detailed CAN logging is enabled on a Raspberry Pi. Walk live
        # frames only for generic logging call sites.
        if record.funcName in {"<module>", "run", "main", "unknown"}:
            get_frame = getattr(sys, "_getframe", None)
            if get_frame is None:
                return super().format(record)
            frame = get_frame()
            try:
                while frame is not None:
                    code = frame.f_code
                    if (
                        code.co_filename == record.pathname
                        and code.co_name not in {"format", "emit"}
                    ):
                        record.funcName = code.co_name
                        record.lineno = frame.f_lineno
                        break
                    frame = frame.f_back
            finally:
                del frame
        return super().format(record)


# ------------------------------------------------------------
# Feature validation
# ------------------------------------------------------------
def validate_features(feature_values: Features) -> None:
    """Validate user-provided feature values before exposing them as globals.

    Invalid configuration is rejected at startup with a precise error instead of
    failing later inside a CAN/API callback.
    """
    errors = []

    def require_bool(name: str) -> None:
        if not isinstance(getattr(feature_values, name), bool):
            errors.append("{} must be true or false".format(name))

    def require_choice(name: str, choices) -> None:
        value = getattr(feature_values, name)
        if value not in choices:
            errors.append(
                "{}={!r} is invalid; expected one of: {}".format(
                    name, value, ", ".join(repr(choice) for choice in choices)
                )
            )

    for name in (
        "send_on_canbus",
        "only_send_if_radio_is_in_tv_mode",
        "show_label",
        "export_speed_measurements_to_file",
        "read_and_set_time_from_dashboard",
        "control_pi_by_rns_e_buttons",
        "read_mfsw_buttons",
        "send_values_to_dashboard",
        "toggle_values_by_rnse_longpress",
        "change_dark_mode_by_car_light",
        "send_api_mediadata_to_dashboard",
        "send_to_api_gauges",
        "reversecamera_by_reversegear",
        "reversecamera_guidelines",
        "reversecamera_by_prev_longpress",
        "activate_rnse_tv_input",
        "ENABLE_LOGGING",
        "show_can_messages_in_logs",
    ):
        require_bool(name)

    for name in ("toggle_fis1", "toggle_fis2"):
        value = getattr(feature_values, name)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 13:
            errors.append("{} must be an integer from 1 to 13".format(name))

    require_choice("scroll_type", ("scroll", "oem_style"))
    require_choice("speed_unit", ("km/h", "mph"))
    require_choice("temp_unit", ("°C", "°F"))
    require_choice("initial_day_night_mode", ("day", "night"))
    require_choice("tv_input_format", ("NTSC", "PAL"))
    require_choice("shutdown_type", ("gently", "instant"))
    require_choice("shutdown_via_can", ("False", "ignition_off", "pulled_key"))
    require_choice("reversecamera_video_pipeline", ("yuyv_30", "mjpg_60"))

    if not isinstance(feature_values.can_interface, str) or not re.fullmatch(
        r"[A-Za-z0-9_.:-]+", feature_values.can_interface
    ):
        errors.append("can_interface contains invalid characters")

    for name in ("lower_speed", "upper_speed", "reversecamera_turn_off_delay", "reversecamera_turn_off_speed"):
        value = getattr(feature_values, name)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append("{} must be an integer".format(name))

    if isinstance(feature_values.lower_speed, int) and feature_values.lower_speed < 0:
        errors.append("lower_speed must be >= 0")
    if isinstance(feature_values.upper_speed, int) and isinstance(feature_values.lower_speed, int):
        if feature_values.upper_speed <= feature_values.lower_speed:
            errors.append("upper_speed must be greater than lower_speed")
    if isinstance(feature_values.reversecamera_turn_off_delay, int) and feature_values.reversecamera_turn_off_delay < 0:
        errors.append("reversecamera_turn_off_delay must be >= 0")
    if isinstance(feature_values.reversecamera_turn_off_speed, int) and feature_values.reversecamera_turn_off_speed < 0:
        errors.append("reversecamera_turn_off_speed must be >= 0")

    if errors:
        raise ValueError("Invalid features.conf:\n  - " + "\n  - ".join(errors))


# ------------------------------------------------------------
# Features laden
# ------------------------------------------------------------
def load_features(feature_file: str, logger=None, enable_logging=True) -> Features:
    """Load known settings from features.conf.

    Existing files are read only and keep comments, formatting and unknown keys.
    A new file containing all defaults is created only when the file does not yet exist.
    Runtime changes of toggle_fis1/toggle_fis2 are persisted separately by
    update_toggle_in_features_conf().
    """
    loaded_features = Features()
    feature_path = Path(feature_file)

    if feature_path.exists():
        if enable_logging and logger:
            logger.info("Feature file '%s' found, loading user settings.\n", feature_path)

        parse_errors = []
        with feature_path.open(encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    if "=" not in line:
                        raise ValueError("missing '=' separator")
                    key, val = line.split("=", 1)
                    key = key.strip()
                    if not key:
                        raise ValueError("empty setting name")
                    val = parse_value(val)

                    if hasattr(loaded_features, key):
                        setattr(loaded_features, key, val)
                        if enable_logging and logger:
                            logger.info("   • %s = %s", key, val)
                    elif enable_logging and logger:
                        logger.info("Ignoring unknown feature key: %s", key)
                except Exception as exc:
                    parse_errors.append(
                        "line {} ({!r}): {}".format(
                            line_number, raw_line.rstrip("\r\n"), exc
                        )
                    )

        if parse_errors:
            raise ValueError(
                "Could not parse features.conf:\n  - " + "\n  - ".join(parse_errors)
            )
    else:
        if enable_logging and logger:
            logger.info("Feature file '%s' not found, creating it with defaults.", feature_path)

        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with feature_path.open("w", encoding="utf-8") as f:
            for key, value in vars(loaded_features).items():
                serialized = repr(value) if isinstance(value, str) else str(value)
                f.write("{} = {}\n".format(key, serialized))

        if enable_logging and logger:
            logger.info("Feature file '%s' created.\n", feature_path)

    validate_features(loaded_features)
    globals().update(vars(loaded_features))
    return loaded_features


# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
def setup_logging():
    global logger, log_file, log_filename, FIS_LOG_FILE, logs_root

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Alte Handler entfernen, damit nichts doppelt geschrieben wird
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    formatter = ContextualFormatter(
        "%(asctime)s,%(msecs)03d | %(levelname)-7s | %(lineno)4d | %(funcName)-21s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    now = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    # Gemeinsamer logs Ordner
    logs_root.mkdir(parents=True, exist_ok=True)

    # Script-Log → logs/script
    script_log_dir = logs_root / "script"
    script_log_dir.mkdir(exist_ok=True)

    filename = Path(__file__).stem
    log_filename = script_log_dir / f"{now}_{filename}.log"

    # FIS-Log → logs/fis_debug_history
    fis_log_dir = logs_root / "fis_debug_history"
    fis_log_dir.mkdir(exist_ok=True)

    FIS_LOG_FILE = fis_log_dir / f"{now}_fis_debug_history.log"

    # logging/open arbeiten zuverlässiger mit Strings
    log_filename = str(log_filename)
    FIS_LOG_FILE = str(FIS_LOG_FILE)

    # Konsole immer
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Datei nur wenn ENABLE_LOGGING aktiv
    if ENABLE_LOGGING:
        file_handler = logging.FileHandler(log_filename, delay=True, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ------------------------------------------------------------
# Header fürs Script-Log
# ------------------------------------------------------------
async def write_header_to_log_file(log_filename):
    header = f"{'TIME':<23} | {'LEVEL':<7} | {'LINE':<4} | {'FUNCTION':<21} | {'MESSAGE'}"
    message_start_index = header.rfind("|")
    if message_start_index != -1:
        separator_line = "".join("|" if c == "|" else "-" for c in header)
        separator_line += "-" * 80
    else:
        separator_line = "-" * (len(header) + 80)

    print(header)
    print(separator_line)

    if ENABLE_LOGGING:
        _write_to_file(log_filename, header, separator_line)


def _write_to_file(log_filename, header, separator_line):
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("\n")
        log_file.write(header + "\n")
        log_file.write(separator_line + "\n")


def _print_to_console(header, separator_line):
    print()
    print()
    print(header)
    print(separator_line)


# ------------------------------------------------------------
# Python-Version früh prüfen
# ------------------------------------------------------------
if sys.version_info < (3, 0):
    print("🚫 This script requires Python 3. Please run it using 'python3 script.py'")
    sys.exit(1)


# ------------------------------------------------------------
# Feature loading / logging bootstrap
# ------------------------------------------------------------
# Load features silently first so ENABLE_LOGGING is known before setup_logging().
# The visible feature-load summary is logged later inside start_script(), after
# the normal console/file logger and the table header are active.
features = load_features(FEATURE_FILE, enable_logging=False)
globals().update(vars(features))

logger = setup_logging()
logger.addFilter(ThreadNameFilter())

TOGGLE_MAP = {
    1: 'TITLE',
    2: 'ARTIST',
    3: 'ALBUM',
    4: 'POSITION',
    5: 'DURATION',
    6: 'SPEED',
    7: 'RPM',
    8: 'COOLANT',
    9: 'CPU/TEMP',
    10: lambda: f'{lower_speed}-{upper_speed}',
    11: 'OUTSIDE',
    12: 'BLANK',
    13: 'DISABLE'
}


async def async_to_thread(func, *args, **kwargs):
    """Run a blocking callable without blocking the asyncio event loop."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)

    loop = asyncio.get_running_loop()
    bound_call = partial(func, *args, **kwargs)

    def _run_bound_call(_unused):
        return bound_call()

    return await loop.run_in_executor(None, _run_bound_call, None)


def create_task_compat(coro, name=None, loop=None):
    """Create a task with names on 3.8+, and emulate names on Python 3.7."""
    if loop is None:
        loop = asyncio.get_running_loop()

    if sys.version_info >= (3, 8):
        return loop.create_task(coro, name=name)

    task = loop.create_task(coro)
    try:
        task._compat_name = name or "task"
    except Exception:
        pass
    return task


def task_name_compat(task):
    getter = getattr(task, "get_name", None)
    if getter is not None:
        try:
            return getter()
        except Exception:
            pass
    return getattr(task, "_compat_name", "task")


def _task_done_cleanup(task: asyncio.Task):
    global tasks

    with contextlib.suppress(ValueError):
        tasks.remove(task)

    try:
        exc = task.exception()
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            logger.warning("Task '%s' ended with exception: %s", task_name_compat(task), exc, exc_info=True)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.warning("Could not inspect finished task '%s'", task_name_compat(task), exc_info=True)


def track_task(coro, name: str) -> asyncio.Task:
    global tasks

    task = create_task_compat(coro, name=name)
    tasks.append(task)
    task.add_done_callback(_task_done_cleanup)
    return task


def fire_and_forget(loop, coro, name: str) -> asyncio.Task:
    global background_tasks

    task = create_task_compat(coro, name=name, loop=loop)
    background_tasks.append(task)

    def _done(t: asyncio.Task):
        global background_tasks
        with contextlib.suppress(ValueError):
            background_tasks.remove(t)
        try:
            exc = t.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                logger.warning("Task '%s' failed: %s", task_name_compat(t), exc, exc_info=True)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                "Could not inspect task '%s'",
                task_name_compat(t),
                exc_info=True
            )
    task.add_done_callback(_done)
    return task


async def init_async_primitives():
    global remote_server_shutdown_event, fis1_update_event, fis2_update_event, candump_lock

    if remote_server_shutdown_event is None:
        remote_server_shutdown_event = asyncio.Event()

    if fis1_update_event is None:
        fis1_update_event = asyncio.Event()

    if fis2_update_event is None:
        fis2_update_event = asyncio.Event()

    if candump_lock is None:
        candump_lock = asyncio.Lock()


class AsyncDualOutput:
    def __init__(self, file):
        self.file = file
        self.loop = asyncio.get_event_loop()

    async def write(self, message):
        self.loop = asyncio.get_running_loop()
        await async_to_thread(self.file.write, message)
        await async_to_thread(sys.__stdout__.write, message)

    async def flush(self):
        self.loop = asyncio.get_running_loop()
        await async_to_thread(self.file.flush)
        await async_to_thread(sys.__stdout__.flush)


# ===== HELFER: CPU-HW-Min/Max + Meta-Aufbau + Hello-Antwort =====

def _read_int(path: str):
    try:
        with open(path) as f:
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
    Baut den _meta-Block:
    - cpu_freq_mhz: Hardware-Min/Max (aus cpuinfo_*), in MHz
    - cpu_temp / outside_temp / coolant: unit
    - speed: current unit
    - speed_measure: lower_speed, upper_speed, unit
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
            "unit": speed_unit,
        },
    }

    if enable_logging and logger:
        logger.info("")
        logger.info("🔧 Built initial meta dict:")
        for k, v in meta.items():
            if isinstance(v, dict):
                inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
                logger.info("   - %s: %s", k, inner)
            else:
                logger.info("   - %s: %s", k, v)
        logger.info("")

    return meta


def _current_dashboard_values_snapshot(value_cache: Optional[dict] = None) -> dict:
    """
    Baut einen Initial-Snapshot aus WS-Cache + zuletzt dekodierten CAN-Werten.

    Wichtig: Gerade coolant/outside_temp ändern sich oft lange nicht. Wenn der erste
    CAN-Wert vor API/WS-Verbindung kam, darf die HTML trotzdem beim Verbinden den
    aktuellen Wert bekommen.
    """
    values = dict(value_cache or {})

    try:
        current_speed = last_speed
        current_rpm = last_rpm
        current_coolant = last_coolant
        current_outside_temp = last_outside_temp

        if current_speed is not None:
            values["speed"] = current_speed
        if current_rpm is not None:
            values["rpm"] = current_rpm
        if current_coolant is not None:
            values["coolant"] = current_coolant
        if current_outside_temp is not None:
            values["outside_temp"] = current_outside_temp

        # CPU-Werte dürfen 0 sein; sie werden ohnehin zyklisch aktualisiert.
        values["cpu_load"] = cpu_load
        values["cpu_temp"] = cpu_temp
        values["cpu_freq_mhz"] = cpu_freq_mhz

        # Speed-Measure: keine alten Finalwerte als Initialwert erzwingen.
        # Der Kanal wird über 0 / -2 / final / -1 live gesteuert.
        if values.get("speed_measure") is None:
            values["speed_measure"] = 0
    except Exception:
        pass

    return values


async def _send_initial_state_to_ws(ws, meta_cache: dict, value_cache: dict, logger=None, enable_logging=True):
    """Schickt {_meta: ..., values: ...} an genau diesen Client und loggt das Ereignis."""
    try:
        payload = {
            "_meta": meta_cache or {},
            "values": _current_dashboard_values_snapshot(value_cache),
            "t": time.time(),
        }

        await ws.send(json.dumps(payload, separators=(",", ":")))

        if enable_logging and logger:
            logger.info(
                "✅ Sent initial state to the connected client (%d meta entries, %d values).",
                len(payload["_meta"]),
                len(payload["values"])
            )
            logger.info("   📊 Initial values:")
            for k, v in payload["values"].items():
                if v is None:
                    logger.info("      - %s: <no value yet>", k)
                else:
                    logger.info("      - %s: %s", k, v)

    except Exception:
        if enable_logging and logger:
            logger.warning("⚠️ Failed to send initial state to a new client.", exc_info=True)


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
        self._meta_cache = {}
        self._value_cache = {
            "speed": None,
            "rpm": None,
            "coolant": None,
            "outside_temp": None,
            "cpu_load": None,
            "cpu_temp": None,
            "cpu_freq_mhz": None,
            "speed_measure": None,
        }

    async def start(self):
        if self.started:
            return
        if ws_server is None:
            raise RuntimeError("Package 'websockets' fehlt. Installiere mit: pip install websockets")
        self._loop = asyncio.get_running_loop()
        self._server = await websockets.serve(self._handler, self.host, self.port)
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

    async def _handler(self, ws, _path=None) -> None:
        """
        Hello-/Meta-Handshake:
        - Browser sendet {type:"request_meta", page, href, title} nach onopen
        - Wir loggen die Quelle
        - Wir senden diesem Client direkt _meta + aktuelle Werte
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

                    fresh_meta = build_initial_meta(
                        logger=self.logger,
                        enable_logging=self.enable_logging
                    )

                    for k, v in fresh_meta.items():
                        prev = self._meta_cache.get(k)
                        if isinstance(v, dict) and isinstance(prev, dict):
                            self._meta_cache[k] = {**prev, **v}
                        else:
                            self._meta_cache[k] = v

                    await _send_initial_state_to_ws(
                        ws,
                        self._meta_cache,
                        self._value_cache,
                        logger=self.logger,
                        enable_logging=self.enable_logging
                    )
        finally:
            self._clients.discard(ws)

    async def _broadcast_json(self, obj: dict):
        if not self._clients:
            return
        data = json.dumps(obj)
        await asyncio.gather(*(c.send(data) for c in list(self._clients)), return_exceptions=True)

    def set(self, key: str, value):
        self._value_cache[key] = value

        if self.started and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_json({"key": key, "value": value, "t": time.time()}),
                self._loop
            )

    def set_with_meta(self, key: str, value, meta: dict):
        self._value_cache[key] = value

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
            self._value_cache[k] = v
            asyncio.run_coroutine_threadsafe(
                self._broadcast_json({"key": k, "value": v, "t": time.time()}),
                self._loop
            )


async def ensure_ws_hub_started(logger=None, ENABLE_LOGGING=True, backend=None, base_dir=None):
    """
    Create/start the WS hub if needed (idempotent) and ensure ONE static web server
    is running with a backend-specific docroot (no HTML creation here).

    - Hudiy:    docroot = <base_dir>/hudiy_api
    - OpenAuto: docroot = <base_dir>/openauto_api
    """
    # Default: Verzeichnis dieses Skripts
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir).resolve()

    try:
        b = (backend or "").strip().lower()
    except Exception:
        b = ""

    if b == "hudiy":
        docroot = base_dir / "hudiy_api"
    else:
        docroot = base_dir / "openauto_api"

    try:
        docroot.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if ENABLE_LOGGING and logger:
            logger.warning("Could not ensure docroot (%s): %s", docroot, e)

    global hudiy_ws_hub
    if ws_server is None:
        if ENABLE_LOGGING and logger:
            logger.error("Package 'websockets' nicht installiert. Run: pip install websockets")
        return

    if hudiy_ws_hub is None:
        hudiy_ws_hub = HudiyDashboardHub(logger=logger, enable_logging=ENABLE_LOGGING)

    if not hudiy_ws_hub.started:
        await hudiy_ws_hub.start()


async def handle_exception(exception, fallback_message=None):
    tb = traceback.extract_tb(exception.__traceback__)
    origin = tb[-1] if tb else None
    func_name = origin.name if origin else "unknown"
    line_number = origin.lineno if origin else 0
    error_message = f"{type(exception).__name__} at line {line_number} in {func_name}(): {exception}"
    exc_info_tuple = (type(exception), exception, exception.__traceback__)
    error_record = logger.makeRecord(name=logger.name, level=logging.ERROR,
                                     fn=origin.filename if origin else "<unknown>", lno=line_number, msg=error_message,
                                     args=(), exc_info=exc_info_tuple, func=func_name)
    logger.handle(error_record)
    if fallback_message:
        info_record = logger.makeRecord(name=logger.name, level=logging.INFO,
                                        fn=origin.filename if origin else "<unknown>", lno=line_number + 1,
                                        msg=fallback_message, args=(), exc_info=None, func=func_name)
        logger.handle(info_record)


def log_and_reraise(func):
    """Log failures from critical functions and preserve normal exception flow."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            logger.exception("Critical failure in %s", func.__name__)
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Critical failure in %s", func.__name__)
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def handle_errors(func):
    """Log the complete traceback and preserve the original exception.

    Returning sentinel strings such as ``"error"`` changed declared return types
    and often caused misleading follow-up exceptions. Callers now receive either
    the normal result or the original exception.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Exception occurred in %s", func.__name__)
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Exception occurred in %s", func.__name__)
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def log_callback_errors(func):
    """Log API callback failures without tearing down the API connection.

    Infrastructure and startup functions still propagate exceptions. Individual
    API events are isolated so one malformed message cannot stop message receipt.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("API callback failed in %s", func.__name__)
            return None

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("API callback failed in %s", func.__name__)
            return None

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def run_in_daemon_thread(loop, func):
    """Run one blocking callable in a daemon thread and expose an asyncio Future.

    The daemon flag ensures a defective third-party blocking recv() cannot keep
    the Python interpreter alive after all shutdown attempts have completed.
    """
    future = loop.create_future()

    def runner():
        try:
            result = func()
        except BaseException as exc:
            # Bind the exception as a default argument. Python clears exception
            # variables after the except block, so a plain closure would fail.
            def set_exception(error=exc):
                if not future.done():
                    future.set_exception(error)
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(set_exception)
        else:
            def set_result(value=result):
                if not future.done():
                    future.set_result(value)
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(set_result)

    thread = threading.Thread(
        target=runner,
        name="API_Receiver",
        daemon=True,
    )
    thread.start()
    return future, thread


@log_and_reraise
async def run_command(cmd: str, log_output: bool = False, check: bool = False) -> dict:
    """Run a shell command asynchronously and return stdout, stderr and return code."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
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
        raise RuntimeError(
            "Command failed ({}): {}\n{}".format(
                proc.returncode,
                cmd,
                err or out,
            )
        )

    return {
        "returncode": proc.returncode,
        "stdout": out,
        "stderr": err,
    }


@handle_errors
async def detect_installs():
    global openauto_ok, hudiy_ok, backend

    async def _read_text(path: Path):
        """Kleine Textdatei asynchron (threaded) lesen, ohne aiofiles."""
        if not path.is_file():
            return None
        try:
            return (await async_to_thread(
                partial(path.read_text, encoding="utf-8")
            )).strip()
        except Exception:
            return None

    async def _read_version_from_log(path: Path, regex=r"version:\s*([0-9][\w.\-+]*)"):
        """Version per Regex aus Log holen, asynchron (threaded) und robustes Decoding."""
        if not path.is_file():
            return None
        try:
            data = await async_to_thread(path.read_bytes)
            text = data.decode(errors="ignore")
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
        logger.info("%s folder %s", hudiy_folder_path, "found" if hudiy_folder_found else "NOT found")
        logger.info("%s binary %s", hudiy_binary_path,
                    "found and executable" if hudiy_binary_found else "NOT found or not executable")
        logger.info("%s folder %s", openauto_folder_path, "found" if openauto_folder_found else "NOT found")
        logger.info("%s binary %s", openauto_binary_path,
                    "found and executable" if openauto_binary_found else "NOT found or not executable")

    # Versions (parallel ohne aiofiles)
    hudiy_version_task = _read_text(hudiy_version_file_path) if (
            hudiy_folder_found and hudiy_binary_found) else asyncio.sleep(0, result=None)
    openauto_version_task = _read_version_from_log(openauto_log_path) if (
            openauto_folder_found and openauto_binary_found) else asyncio.sleep(0, result=None)
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

    return {
        "hudiy_ok": hudiy_ok,
        "hudiy_version": hudiy_version,
        "openauto_ok": openauto_ok,
        "openauto_version": openauto_version,
    }


@handle_errors
def log_feature_file_state():
    """Log a visible feature.conf load summary after the final logger/header is active."""
    try:
        feature_path = Path(FEATURE_FILE).resolve()
        if feature_path.exists():
            logger.info("Feature file loaded and applied from: %s", feature_path)
            logger.info("Feature file loaded without rewriting existing content: %s", feature_path)
        else:
            logger.warning("Feature file not found, defaults are active: %s", feature_path)

        logger.info("Feature values are listed below in Current Configuration.")
        logger.info("")
    except Exception:
        logger.warning("Could not log feature file state.", exc_info=True)


def _safe_nonnegative_int(value, default=0) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


@handle_errors
def validate_reversecamera_turn_off_settings():
    """
    Validate reverse-camera automatic turn-off settings loaded from features.conf.

    Rules:
      - reversecamera_turn_off_delay <= 0 means timer disabled.
      - reversecamera_turn_off_speed <= 0 means speed cut-off disabled.
      - If reversecamera_by_reversegear is enabled, at least one of both must be > 0.
      - If both are 0, reversecamera_by_reversegear is disabled for safety.
    """
    global reversecamera_by_reversegear
    global reversecamera_turn_off_delay, reversecamera_turn_off_speed

    delay = _safe_nonnegative_int(reversecamera_turn_off_delay)
    speed_cutoff = _safe_nonnegative_int(reversecamera_turn_off_speed)

    if delay != reversecamera_turn_off_delay:
        logger.warning(
            "Invalid reversecamera_turn_off_delay=%r. Normalizing to %s.",
            reversecamera_turn_off_delay,
            delay
        )
        reversecamera_turn_off_delay = delay

    if speed_cutoff != reversecamera_turn_off_speed:
        logger.warning(
            "Invalid reversecamera_turn_off_speed=%r. Normalizing to %s.",
            reversecamera_turn_off_speed,
            speed_cutoff
        )
        reversecamera_turn_off_speed = speed_cutoff

    if reversecamera_by_reversegear and delay <= 0 and speed_cutoff <= 0:
        logger.warning(
            "Invalid reverse camera configuration: reversecamera_by_reversegear is enabled, "
            "but both reversecamera_turn_off_delay and reversecamera_turn_off_speed are 0. "
            "Disabling reversecamera_by_reversegear for safety."
        )
        reversecamera_by_reversegear = False
        return False

    if reversecamera_by_reversegear:
        if delay > 0 and speed_cutoff > 0:
            logger.info(
                "Reverse camera automatic turn-off: timer=%s s, speed cut-off=%s %s. "
                "Speed cut-off can stop the camera before the timer expires.",
                delay,
                speed_cutoff,
                speed_unit
            )
        elif delay > 0:
            logger.info(
                "Reverse camera automatic turn-off: timer=%s s, speed cut-off disabled.",
                delay
            )
        elif speed_cutoff > 0:
            logger.info(
                "Reverse camera automatic turn-off: timer disabled, speed cut-off=%s %s.",
                speed_cutoff,
                speed_unit
            )

    return True


@handle_errors
async def start_script():
    logger.info("")
    logger.info(f"Script is starting...")
    logger.info("")
    python_path = sys.executable
    python_version = sys.version_info
    python_version_str = f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"

    asyncio.get_event_loop()
    logger.info(f"Python interpreter: {python_version_str} ({python_path})")
    logger.info("Script version: %s (%s)", script_version, script_fullpath)
    logger.info(f"CAN-Interface: {can_interface}")
    logger.info(f"LOGGING_ENABLED: {ENABLE_LOGGING}")
    logger.info("")

    log_feature_file_state()
    validate_reversecamera_turn_off_settings()

    # check if openauto or hudiy is installed/used
    await detect_installs()
    logger.info(f"Detected backend: {backend}")

    logger.info("")
    if ENABLE_LOGGING:

        config_vars = {
            "can_interface": can_interface,
            "send_on_canbus": send_on_canbus,
            "only_send_if_radio_is_in_tv_mode": only_send_if_radio_is_in_tv_mode,
            "show_label": show_label,
            "toggle_fis1": toggle_fis1,
            "toggle_fis2": toggle_fis2,
            "lower_speed": lower_speed,
            "upper_speed": upper_speed,
            "export_speed_measurements_to_file": export_speed_measurements_to_file,
            "scroll_type": scroll_type,
            "speed_unit": speed_unit,
            "temp_unit": temp_unit,
            "welcome_message_1st_line": welcome_message_1st_line,
            "welcome_message_2nd_line": welcome_message_2nd_line,
            "read_and_set_time_from_dashboard": read_and_set_time_from_dashboard,
            "control_pi_by_rns_e_buttons": control_pi_by_rns_e_buttons,
            "read_mfsw_buttons": read_mfsw_buttons,
            "send_values_to_dashboard": send_values_to_dashboard,
            "toggle_values_by_rnse_longpress": toggle_values_by_rnse_longpress,
            "initial_day_night_mode": initial_day_night_mode,
            "change_dark_mode_by_car_light": change_dark_mode_by_car_light,
            "send_api_mediadata_to_dashboard": send_api_mediadata_to_dashboard,
            "send_to_api_gauges": send_to_api_gauges,
            "reversecamera_by_reversegear": reversecamera_by_reversegear,
            "reversecamera_guidelines": reversecamera_guidelines,
            "reversecamera_turn_off_delay": reversecamera_turn_off_delay,
            "reversecamera_turn_off_speed": reversecamera_turn_off_speed,
            "reversecamera_by_prev_longpress": reversecamera_by_prev_longpress,
            "reversecamera_video_pipeline": reversecamera_video_pipeline,
            "shutdown_via_can": shutdown_via_can,
            "shutdown_type": shutdown_type,
            "activate_rnse_tv_input": activate_rnse_tv_input,
            "tv_input_format": tv_input_format,
            "ENABLE_LOGGING": ENABLE_LOGGING,
            "show_can_messages_in_logs": show_can_messages_in_logs
        }

        logger.info("")
        logger.info("✅ Current Configuration:")
        for name, value in config_vars.items():
            logger.info("   • %s = %s", name, value)
        logger.info("")


def _http_response(status_code=200, body="", content_type="application/json; charset=utf-8"):
    reasons = {
        200: "OK",
        204: "No Content",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
    }
    reason = reasons.get(status_code, "OK")
    body_bytes = body.encode() if isinstance(body, str) else body
    headers = [
        f"HTTP/1.1 {status_code} {reason}",
        f"Content-Type: {content_type}",
        f"Content-Length: {len(body_bytes)}",
        "Connection: close",
        "Access-Control-Allow-Origin: *",
        "Access-Control-Allow-Methods: POST, OPTIONS",
        "Access-Control-Allow-Headers: Content-Type",
        "",
        "",
    ]
    return "\r\n".join(headers).encode() + body_bytes


def _parse_http_request(raw_bytes):
    head, _, body = raw_bytes.partition(b"\r\n\r\n")
    header_lines = head.decode("utf-8", errors="replace").split("\r\n")
    request_line = header_lines[0] if header_lines else ""
    parts = request_line.split()
    if len(parts) < 2:
        raise ValueError("Invalid HTTP request line")

    method, path = parts[0], parts[1]
    headers = {}
    for line in header_lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    content_length = int(headers.get("content-length", "0") or "0")
    if content_length > len(body):
        raise ValueError("Incomplete HTTP body")

    return method.upper(), path, headers, body[:content_length]


@handle_errors
async def remote_control(host='127.0.0.1', port=HTTP_CONTROL_PORT):
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
        serve_task = create_task_compat(server.serve_forever(), name="remote_control_server")

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
            fire_and_forget(asyncio.get_running_loop(), stop_script(), "stop_script_from_remote")

        elif command == 'kill_script':
            logger.info("Kill command received.")
            fire_and_forget(asyncio.get_running_loop(), kill_script(), "kill_script_from_remote")

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

    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            pid = proc.info['pid']
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
    writer = None

    try:
        reader, writer = await asyncio.open_connection(host, port)
        logger.info(f"Sending command to {host}:{port}: {command}")
        writer.write(command.encode())
        await writer.drain()

    except Exception:
        logger.error("Error sending command", exc_info=True)

    finally:
        if writer is not None:
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
    except Exception:
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
            logger.warning("⚠️ Backport 'importlib_metadata' missing. Attempting to install it...")
        else:
            print(
                "⚠️ Backport 'importlib_metadata' missing. Attempting to install it...",
                flush=True,
            )

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "importlib-metadata",
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


def _read_os_codename() -> Optional[str]:
    try:
        with open("/etc/os-release", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VERSION_CODENAME="):
                    return line.split("=", 1)[1].strip().strip('"').lower()
    except Exception:
        pass

    return None


async def _repair_buster_apt_sources() -> bool:
    """
    Repair obsolete Raspbian Buster mirror entries.

    Only Buster is modified. Bookworm, Trixie and other releases remain
    untouched.
    """
    codename = _read_os_codename()

    if codename != "buster":
        return True

    source_paths = [Path("/etc/apt/sources.list")]
    source_dir = Path("/etc/apt/sources.list.d")

    if source_dir.is_dir():
        source_paths.extend(source_dir.glob("*.list"))

    obsolete_sources = (
        "http://raspbian.raspberrypi.org/raspbian",
        "https://raspbian.raspberrypi.org/raspbian",
    )
    replacement = "http://legacy.raspbian.org/raspbian"

    files_to_update = []

    for source_path in source_paths:
        if not source_path.is_file():
            continue

        try:
            content = source_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "Could not read APT source file %s: %s",
                source_path,
                exc,
            )
            continue

        updated_content = content

        for obsolete_source in obsolete_sources:
            updated_content = updated_content.replace(
                obsolete_source,
                replacement,
            )

        if updated_content != content:
            files_to_update.append(
                (source_path, content, updated_content)
            )

    if not files_to_update:
        if ENABLE_LOGGING:
            logger.info(
                "Raspbian Buster APT source already uses the legacy repository."
            )
        return True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.warning(
        "Obsolete Raspbian Buster repository detected. "
        "Switching it to legacy.raspbian.org."
    )

    for source_path, original_content, updated_content in files_to_update:
        backup_path = Path(
            "{}.bak_{}".format(source_path, timestamp)
        )

        try:
            write_result = await run_command(
                "sudo cp -a {} {}".format(
                    shlex.quote(str(source_path)),
                    shlex.quote(str(backup_path)),
                )
            )

            if write_result["returncode"] != 0:
                logger.error(
                    "Could not create backup of %s: %s",
                    source_path,
                    (
                        write_result["stderr"]
                        or write_result["stdout"]
                    ).strip(),
                )
                return False

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
            ) as temporary_file:
                temporary_file.write(updated_content)
                temporary_path = temporary_file.name

            try:
                install_result = await run_command(
                    "sudo install -m 0644 {} {}".format(
                        shlex.quote(temporary_path),
                        shlex.quote(str(source_path)),
                    )
                )
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_path)

            if install_result["returncode"] != 0:
                logger.error(
                    "Could not update APT source file %s: %s",
                    source_path,
                    (
                        install_result["stderr"]
                        or install_result["stdout"]
                    ).strip(),
                )
                return False

            logger.info(
                "Updated Buster APT source: %s "
                "(backup: %s)",
                source_path,
                backup_path,
            )

        except Exception:
            logger.exception(
                "Failed to repair Buster APT source file %s",
                source_path,
            )
            return False

    return True


@log_and_reraise
async def check_can_utils() -> bool:
    """Ensure the native SocketCAN command-line tools are available.

    can-utils is independent of the Python interpreter. The distribution package
    is sufficient for the classic candump/cansend functionality used here, so an
    existing APT installation is kept instead of being replaced by a Git build.
    """
    if ENABLE_LOGGING:
        logger.info("")
        logger.info("🔍 Checking can-utils installation...")

    candump_path = shutil.which("candump")
    cansend_path = shutil.which("cansend")

    if candump_path and cansend_path:
        if ENABLE_LOGGING:
            logger.info("✅ candump found at: %s", candump_path)
            logger.info("✅ cansend found at: %s", cansend_path)
            logger.info("")
        return True

    missing_tools = []
    if not candump_path:
        missing_tools.append("candump")
    if not cansend_path:
        missing_tools.append("cansend")

    logger.warning(
        "⚠️ Missing can-utils tool(s): %s. Installing the distribution package via APT...",
        ", ".join(missing_tools),
    )

    if not await _apt_install(["can-utils"]):
        return False

    candump_path = shutil.which("candump")
    cansend_path = shutil.which("cansend")
    if not candump_path or not cansend_path:
        logger.error(
            "❌ can-utils installation completed, but required tools are still missing "
            "(candump=%r, cansend=%r).",
            candump_path,
            cansend_path,
        )
        return False

    logger.info("✅ candump installed at: %s", candump_path)
    logger.info("✅ cansend installed at: %s", cansend_path)
    logger.info("")
    return True


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
        if enable_logging and logger:
            logger.info("'packaging' missing – installing via pip…")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--upgrade",
            ("packaging<24" if CURRENT_PYTHON_VERSION == (3, 7) else "packaging"),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"pip install packaging failed: {err.decode(errors='ignore').strip()}")
        if enable_logging and logger:
            logger.info("✔ 'packaging' installed")
        from packaging.requirements import Requirement
        from packaging.version import Version
        return Requirement, Version


async def packaging_globals(logger=None, enable_logging=True):
    """Lädt Requirement/Version und schreibt sie in globals()."""
    Requirement, Version = await ensure_packaging(logger, enable_logging)
    globals()["Requirement"] = Requirement
    globals()["Version"] = Version
    if enable_logging and logger:
        logger.info("🔧 packaging globals ready: Requirement, Version")


@log_and_reraise
async def _pip_install(python_exe: str, packages: List[str]) -> bool:
    if not packages:
        return True

    # Pass each requirement directly to pip. This avoids shell interpretation of
    # characters such as '<' and '>' in version constraints.
    proc = await asyncio.create_subprocess_exec(
        python_exe,
        "-m",
        "pip",
        "install",
        *packages,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout = (stdout_b or b"").decode(errors="ignore")
    stderr = (stderr_b or b"").decode(errors="ignore")

    if proc.returncode != 0:
        logger.error("❌ pip install failed: %s", (stderr or stdout).strip())
        return False

    version, PackageNotFoundError = await ensure_importlib(logger)

    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None

    logger.info("")
    logger.info("✅ pip packages installed:")
    for spec in packages:
        if Requirement is not None:
            try:
                name = Requirement(spec).name
            except Exception:
                name = spec
        else:
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
            ver = version(name)
            logger.info("    • %s %s", name, ver)
        except PackageNotFoundError:
            logger.info("    • %s (version unknown)", name)

    return True


@log_and_reraise
async def _apt_install(packages: List[str]) -> bool:
    if not packages:
        return True

    if not await _repair_buster_apt_sources():
        logger.error(
            "APT sources could not be prepared for this Raspberry Pi OS version."
        )
        return False

    update_result = await run_command(
        "sudo apt-get update",
        log_output=True,
    )

    if update_result["returncode"] != 0:
        logger.error(
            "❌ apt-get update failed: %s",
            (
                update_result["stderr"]
                or update_result["stdout"]
            ).strip(),
        )
        return False

    install_result = await run_command(
        "sudo apt-get install -y " + " ".join(
            shlex.quote(package) for package in packages
        ),
        log_output=True,
    )

    if install_result["returncode"] != 0:
        logger.error(
            "❌ apt-get install failed: %s",
            (
                install_result["stderr"]
                or install_result["stdout"]
            ).strip(),
        )
        return False

    logger.info("")
    logger.info("✅ APT packages installed:")

    for package in packages:
        logger.info("    • %s", package)

    return True


def get_python_can_requirement() -> str:
    """Return a compatible, flexible python-can 4.x requirement."""
    if CURRENT_PYTHON_VERSION >= (3, 9):
        return "python-can>=4.6.1,<5"
    if CURRENT_PYTHON_VERSION >= (3, 7):
        return "python-can>=4.2.2,<4.3"
    raise RuntimeError(
        "Unsupported Python version for python-can: {}.{}".format(
            CURRENT_PYTHON_VERSION[0],
            CURRENT_PYTHON_VERSION[1],
        )
    )


@log_and_reraise
async def python_modules() -> Tuple[Dict[str, str], List[str]]:
    """
    Scannt die 'required' Module (inkl. backend-spezifischer) und liefert:
      - installed_modules: {name -> version}
      - missing_modules:   ["pkg", "pkg~=x.y", ...] (inkl. Versionsanforderung)
    """
    from packaging.version import Version
    from packaging.requirements import Requirement

    version, PackageNotFoundError = await ensure_importlib(logger)

    python_can_requirement = get_python_can_requirement()

    # Python 3.7/3.8 need upper bounds for packages whose newer releases have
    # dropped those interpreters. Python 3.9+ lets pip select current compatible
    # releases through each package's Requires-Python metadata.
    if CURRENT_PYTHON_VERSION < (3, 9):
        base_required = [
            "aioconsole<0.7",
            "aiofiles<24",
            "requests<3",
            "protobuf~=3.19",  # Intentionally kept on protobuf 3.x: both downloaded
            # OpenAuto Pro and Hudiy Api_pb2 bindings are generated for this API generation.
            python_can_requirement,
            "psutil<6",
            "python-uinput",
            "packaging<24",
            "importlib-metadata<7",
            "websockets<11",
        ]
    else:
        base_required = [
            "aioconsole",
            "aiofiles",
            "requests",
            "protobuf~=3.19",  # Intentionally kept on protobuf 3.x: both downloaded
            # OpenAuto Pro and Hudiy Api_pb2 bindings are generated for this API generation.
            python_can_requirement,
            "psutil",
            "python-uinput",
            "packaging",
            "websockets",
        ]

    required = list(base_required)

    backend_norm = (backend or "").strip().lower()
    if backend_norm == "hudiy":
        if CURRENT_PYTHON_VERSION < (3, 9):
            required += ["websocket-client<1.7"]
        else:
            required += ["websocket-client~=1.8"]
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


@log_and_reraise
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

    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None

    for spec in missing_modules:
        if Requirement is not None:
            try:
                package_name = Requirement(spec).name
            except Exception:
                package_name = spec
        else:
            package_name = (
                spec.split("==")[0]
                .split("~=")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split(">")[0]
                .split("<")[0]
                .strip()
            )
        try:
            installed_modules[package_name] = version(package_name)
        except PackageNotFoundError:
            installed_modules[package_name] = "(pip-installed)"

    return installed_modules


@log_and_reraise
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
            logger.error("")
            logger.error("❌ Python packages are still missing or incompatible after installation:")
            for m in missing_after:
                logger.error("   • %s", m)
            raise RuntimeError(
                "Required Python packages could not be installed: {}".format(
                    ", ".join(missing_after)
                )
            )

    # Jetzt genau EINMAL importieren und loggen
    import_modules(installed)


@log_and_reraise
def import_modules(installed_modules: Dict[str, str]) -> None:
    """
    Importiert die gelisteten Module (Mapping: Paketname -> Version) und loggt diese EINMAL gesammelt.
    """
    global aiofiles, requests, can, Notifier, psutil, protobuf
    global aioconsole, uinput, packaging, png, websockets, ws_server, ws_client

    if ENABLE_LOGGING:
        logger.info("")
        logger.info("✅ Successfully imported modules:")

    backend_norm = (backend or "").strip().lower()

    for mod, ver in installed_modules.items():
        try:
            if mod == "aiofiles":
                import aiofiles
            elif mod == "requests":
                import requests
            elif mod == "python-can":
                import can
                from can import Notifier
                if ENABLE_LOGGING:
                    logger.info(
                        "   ↳ python-can loaded version=%s from=%s",
                        getattr(can, "__version__", ver),
                        getattr(can, "__file__", "unknown"),
                    )
            elif mod == "psutil":
                import psutil
                if ENABLE_LOGGING:
                    logger.info(
                        "   ↳ psutil loaded version=%s from=%s",
                        getattr(psutil, "__version__", ver),
                        getattr(psutil, "__file__", "unknown"),
                    )
            elif mod == "protobuf":
                import google.protobuf as protobuf
                if ENABLE_LOGGING:
                    logger.info(
                        "   ↳ protobuf loaded version=%s from=%s",
                        getattr(protobuf, "__version__", ver),
                        getattr(protobuf, "__file__", "unknown"),
                    )
            elif mod == "aioconsole":
                import aioconsole
            elif mod == "python-uinput":
                import uinput
            elif mod == "packaging":
                import packaging
            elif mod == "websockets":
                import websockets
                ws_server = websockets
                if ENABLE_LOGGING:
                    logger.info(
                        "   ↳ websockets loaded version=%s from=%s",
                        getattr(websockets, "__version__", ver),
                        getattr(websockets, "__file__", "unknown"),
                    )
            elif mod == "websocket-client":
                import websocket as ws_client
                if ENABLE_LOGGING:
                    logger.info(
                        "   ↳ websocket-client loaded version=%s from=%s",
                        getattr(ws_client, "__version__", ver),
                        getattr(ws_client, "__file__", "unknown"),
                    )
            elif mod == "pypng" and backend_norm == "openauto":
                import png

            if ENABLE_LOGGING:
                logger.info("   • %s %s", mod, ver)

        except ImportError as e:
            raise RuntimeError(
                "Failed to import {} (installed version: {}): {}".format(mod, ver, e)
            ) from e

    if ENABLE_LOGGING:
        if "can" in globals():
            logger.info(
                "   ↳ python-can runtime: version=%s, path=%s",
                getattr(can, "__version__", "unknown"),
                getattr(can, "__file__", "unknown"),
            )
        if "protobuf" in globals():
            logger.info(
                "   ↳ protobuf runtime: version=%s, path=%s",
                getattr(protobuf, "__version__", "unknown"),
                getattr(protobuf, "__file__", "unknown"),
            )
        if "websockets" in globals():
            logger.info(
                "   ↳ websockets runtime: version=%s, path=%s",
                getattr(websockets, "__version__", "unknown"),
                getattr(websockets, "__file__", "unknown"),
            )
        logger.info("")


@handle_errors
async def uinput_permissions():
    if control_pi_by_rns_e_buttons:
        try:
            result = await run_command("stat /dev/uinput")
            if "0666" not in result["stdout"]:
                logger.warning("⚠️ Permissions for /dev/uinput are incorrect.")
                logger.info("Setting correct permissions...")
                await run_command("sudo modprobe uinput")
                await run_command("sudo chmod 666 /dev/uinput")
                result = await run_command("stat /dev/uinput")
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
    governor_dir = Path("/sys/devices/system/cpu/cpufreq/policy0")
    governor_path = governor_dir / "scaling_governor"
    available_path = governor_dir / "scaling_available_governors"

    if not governor_path.is_file():
        if ENABLE_LOGGING:
            logger.info("CPU governor control is unavailable; leaving the current power plan unchanged.")
        return

    current_governor = (await async_to_thread(governor_path.read_text)).strip()
    if current_governor != "powersave":
        return

    available = []
    if available_path.is_file():
        available = (await async_to_thread(available_path.read_text)).split()

    target_governor = None
    for candidate in ("ondemand", "schedutil"):
        if not available or candidate in available:
            target_governor = candidate
            break

    if target_governor is None:
        logger.warning(
            "CPU governor is powersave, but neither ondemand nor schedutil is available (%s).",
            ", ".join(available) or "unknown",
        )
        return

    proc = await asyncio.create_subprocess_exec(
        "sudo", "tee", str(governor_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate(input=target_governor.encode())
    if proc.returncode == 0:
        logger.info(
            "Powersave mode detected. Changing CPU governor from '%s' to '%s'.",
            current_governor,
            target_governor,
        )
    else:
        logger.error("Failed to set CPU governor: %s", stderr.decode(errors="ignore").strip())


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
               dict(can_id=0x5C0, can_mask=0x7FF, extended=False),
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
            codename = next(
                (
                    line.split("=", 1)[1].strip().strip('"')
                    for line in f
                    if line.startswith("VERSION_CODENAME=")
                ),
                None
            )
    except Exception:
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
    result = await run_command(f"sudo cat {CONFIG_FILE}")
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
    if (await run_command(f"sudo cp -a {CONFIG_FILE} {backup}"))["returncode"] == 0:
        logger.info(f"🧰 Backup created: {backup}")
    else:
        logger.warning("⚠️ Could not create backup.")

    # Append missing lines
    newline = "\n"
    cmd = f"sudo bash -c 'echo -e \"{newline}#PiCAN2/3 Settings{newline}{newline.join(missing)}\" >> {CONFIG_FILE}'"
    if (await run_command(cmd))["returncode"] == 0:
        logger.info(f"✅ Missing lines added to {CONFIG_FILE}")
    else:
        logger.error(f"❌ Failed to write to {CONFIG_FILE}")
        return

    if (await aioconsole.ainput("Reboot now? (yes/no): ")).strip().lower() in ("yes", "y"):
        logger.info("🔄 Rebooting...")
        await run_command("sudo reboot")
    else:
        logger.warning("ℹ️ Please reboot manually.")


def validate_can_interface_name(interface_name: str) -> str:
    """Validate a Linux network-interface name before using it in commands."""
    value = str(interface_name).strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", value):
        raise ValueError("Invalid CAN interface name: {!r}".format(interface_name))
    return value


@log_and_reraise
async def test_can_interface():
    """Tests and initializes the CAN interface."""
    global bus, send_on_canbus, can_functional, can_interface

    can_interface = validate_can_interface_name(can_interface)
    bus = None
    can_functional = False

    async def setup_vcan0() -> bool:
        global bus, can_interface

        result = await run_command("ip link show vcan0")
        if result["returncode"] != 0:
            logger.warning("⚠️ vcan0 does not exist. Creating vcan0...")
            result = await run_command("sudo ip link add dev vcan0 type vcan")
            if result["returncode"] != 0:
                logger.error(f"❌ Failed to create vcan0: {result['stderr'] or result['stdout']}")
                return False

        logger.info("✅ vcan0 created successfully.")

        result = await run_command("sudo ip link set up vcan0")
        if result["returncode"] != 0:
            logger.error(f"❌ Failed to bring vcan0 up: {result['stderr'] or result['stdout']}")
            return False

        logger.info("✅ vcan0 interface is up.")

        try:
            bus = can.interface.Bus(
                "vcan0",
                interface='socketcan',
                bitrate=100000,
                can_filters=can_filters,
                receive_own_messages=False
            )
            can_interface = "vcan0"
            logger.info("✅ CAN-Interface 'vcan0' found and opened.")

        except can.CanError as e:
            logger.error(f"❌ Failed to initialize CAN-Bus on vcan0. Error: {e}")
            bus = None
            return False

        await run_command("sudo ifconfig vcan0 txqueuelen 1000")
        return True

    async def setup_real_can() -> bool:
        global bus

        if not os.path.exists(f'/sys/class/net/{can_interface}/operstate'):
            logger.warning(f"⚠️ Interface {can_interface} does not exist.")
            await check_pican2_3_config()
            return False

        async with aiofiles.open(f'/sys/class/net/{can_interface}/operstate', mode='r') as f:
            state = (await f.read()).strip()

        if state != 'up':
            logger.warning(f"⚠️ {can_interface} is down, trying to bring it up...")
            result = await run_command(
                f'sudo /sbin/ip link set {can_interface} up type can restart-ms 1000 bitrate 100000'
            )
            if result["returncode"] != 0:
                logger.error(f"❌ Failed to bring {can_interface} up.")
                return False

        await run_command(f'sudo ifconfig {can_interface} txqueuelen 1000')

        try:
            bus = can.interface.Bus(
                can_interface,
                interface='socketcan',
                bitrate=100000,
                can_filters=can_filters,
                receive_own_messages=False
            )
            logger.info(f"✅ CAN-Interface '{can_interface}' found and opened.")
            return True
        except can.CanError as e:
            logger.error(f"❌ Failed to initialize CAN-Bus on {can_interface}. Error: {e}")
            bus = None
            return False

    async def has_traffic(timeout: float = 1.0) -> bool:
        global bus

        if bus is None:
            return False

        msg = await async_to_thread(bus.recv, timeout)
        return msg is not None

    try:
        if can_interface != "vcan0":
            if not await setup_real_can():
                return

            if await has_traffic():
                logger.info("✅ CAN message received. CAN-Bus seems to be working.")
                can_functional = True
                return

            if AUTO_FALLBACK_TO_VCAN:
                logger.info("Desk-Setup: No can0 activity -> Fallback to vcan0 (for Desk-Setup tests only).")

                if bus is not None:
                    try:
                        cast("can.BusABC", bus).shutdown()
                    except Exception:
                        pass
                bus = None

                if await setup_vcan0():
                    can_functional = True
                    logger.info("✅ vcan0 fallback initialized successfully.")
                else:
                    logger.warning("⚠️ vcan0 fallback failed.")
            else:
                logger.warning("⚠️ No CAN message received. Disabling CAN-Bus communication.")

        else:
            if await setup_vcan0():
                can_functional = True

    except Exception as e:
        logger.error(f"❌ Error while testing the CAN interface: {e}", exc_info=True)

    finally:
        if not can_functional:
            logger.error("❌ Failed to initialize CAN-Bus. Disabling CAN-Bus features.")
            send_on_canbus = False


async def start_send_to_dis():
    global send_to_dis_tasks_started, task_send_fis1, task_send_fis2

    if send_to_dis_tasks_started:
        return

    if not model_info_set:
        return

    task_send_fis1 = track_task(send_to_dis(FIS1), "send_to_dis_1")
    task_send_fis2 = track_task(send_to_dis(FIS2), "send_to_dis_2")
    send_to_dis_tasks_started = True

    if ENABLE_LOGGING:
        logger.info("send_to_dis tasks started with FIS1=%s and FIS2=%s", FIS1, FIS2)


@handle_errors
def get_reversecamera_pipeline_settings():
    """
    Select GStreamer input settings for the USB CVBS grabber.

    Supported values in features.conf:
      reversecamera_video_pipeline = 'yuyv_30'
      reversecamera_video_pipeline = 'mjpeg_60'
    """
    mode = str(reversecamera_video_pipeline).strip().lower()

    pipelines = {
        # Stable low-latency raw mode.
        # USB grabber: YUYV/YUY2 720x480 @ 30 fps.
        "yuyv_30": {
            "input_format": "yuyv422",
            "fps": 30,
            "capture_width": 720,
            "capture_height": 480,
            "output_width": 800,
            "output_height": 480,
            "label": "YUYV/YUY2 720x480 @ 30 fps",
        },

        # Smoother / less-tearing candidate.
        # USB grabber: MJPEG 720x480 @ 60 fps.
        "mjpeg_60": {
            "input_format": "mjpeg",
            "fps": 60,
            "capture_width": 720,
            "capture_height": 480,
            "output_width": 800,
            "output_height": 480,
            "label": "MJPEG 720x480 @ 60 fps",
        },
    }

    selected = pipelines.get(mode)

    if selected is None:
        logger.warning(
            "Unknown reversecamera_video_pipeline=%r. Falling back to yuyv_30.",
            reversecamera_video_pipeline,
        )
        selected = pipelines["yuyv_30"]

    if ENABLE_LOGGING:
        logger.info(
            "Reverse camera pipeline selected: %s "
            "(input_format=%s, capture=%sx%s, output=%sx%s, fps=%s)",
            selected["label"],
            selected["input_format"],
            selected["capture_width"],
            selected["capture_height"],
            selected["output_width"],
            selected["output_height"],
            selected["fps"],
        )

    return selected


@handle_errors
def cam_init(reversecamera_guidelines: bool, overlay_png=None):
    """Initial warm-start for the local reverse camera."""
    global cam, CAM_WITH_OVERLAY

    if overlay_png is None:
        overlay_png = str(BASE_PATH / "overlay.png")

    CAM_WITH_OVERLAY = bool(reversecamera_guidelines) and os.path.isfile(overlay_png)
    if ENABLE_LOGGING:
        logger.info(
            "Reverse camera overlay: %s (%s)",
            "enabled" if CAM_WITH_OVERLAY else "disabled",
            overlay_png
        )
    pipeline = get_reversecamera_pipeline_settings()

    cam = Cam(
        overlay_png=(overlay_png if CAM_WITH_OVERLAY else None),
        capture_width=pipeline["capture_width"],
        capture_height=pipeline["capture_height"],
        output_width=pipeline["output_width"],
        output_height=pipeline["output_height"],
        fps=pipeline["fps"],
        input_format=pipeline["input_format"],
    )

    warm_variant = "overlay" if CAM_WITH_OVERLAY else "base"
    cam.start(warm_variant=warm_variant)


@handle_errors
async def delayed_reverse_camera_off():
    global gear, reverse_camera_off_task
    global reversecamera_by_reversegear, reversecamera_by_prev_longpress

    this_task = asyncio.current_task()
    try:
        delay = _safe_nonnegative_int(reversecamera_turn_off_delay)
        if delay <= 0:
            logger.info("Reverse camera turn-off timer is disabled; delayed stop task exits.")
            return

        logger.info(
            "Reverse camera turn-off task started. Waiting %s seconds.",
            delay
        )

        await asyncio.sleep(delay)

        if gear != 0:
            logger.info("Reverse camera turn-off skipped because reverse gear is active again.")
            return

        logger.info("Reverse camera turn-off delay expired - stopping local camera.")
        await local_camera_action(show=False, force_stop=True)
        logger.info("Reverse camera turn-off completed.")

    except asyncio.CancelledError:
        if ENABLE_LOGGING:
            logger.info("delayed_reverse_camera_off cancelled.")
        raise
    except Exception:
        logger.error("Error while stopping the reverse camera's livestream.", exc_info=True)
        reversecamera_by_reversegear = False
        reversecamera_by_prev_longpress = False
    finally:
        if reverse_camera_off_task is this_task:
            reverse_camera_off_task = None


@handle_errors
async def read_cpu_loop():
    global cpu_load, cpu_temp, cpu_freq_mhz, event_handler

    if ENABLE_LOGGING:
        logger.info("read_cpu_loop started")
    while not stop_flag:
        try:
            if send_to_api_gauges or 9 in (toggle_fis1, toggle_fis2):
                cpu_load = min(round(psutil.cpu_percent()), 99)
                if send_to_api_gauges and api_is_connected and event_handler is not None:
                    event_handler.update_to_api("getPidValue(4)", cpu_load)
                temps = psutil.sensors_temperatures().get("cpu_thermal", [])
                if temps:
                    cpu_temp = int(round(temps[0].current))
                    if temp_unit == '°F':
                        cpu_temp = round(cpu_temp * 1.8 + 32)
                    data = f'{cpu_load:02d}% {cpu_temp:02d}{temp_unit}'
                    if send_on_canbus and can_functional:
                        if toggle_fis1 == 9 and not show_label and not pause_fis1:
                            set_fis1(
                                data,
                                "center",
                                trace=make_trace("read_cpu_loop", inspect.currentframe().f_lineno + 1) if ENABLE_LOGGING else None
                            )
                        if toggle_fis2 == 9 and not pause_fis2:
                            set_fis2(
                                data,
                                "center",
                                trace=make_trace("read_cpu_loop", inspect.currentframe().f_lineno + 1) if ENABLE_LOGGING else None
                            )
                    if send_to_api_gauges and api_is_connected and event_handler is not None:
                        event_handler.update_to_api("getPidValue(5)", cpu_temp)
                if send_to_api_gauges:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq:
                        cpu_freq_mhz = int(round(cpu_freq.current))
                        if api_is_connected and event_handler is not None:
                            event_handler.update_to_api("getPidValue(6)", cpu_freq_mhz)
            await asyncio.sleep(cpu_read_interval)
        except asyncio.CancelledError:
            if ENABLE_LOGGING:
                logger.info("Task read_cpu_loop was stopped.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in CPU monitor task: {e}", exc_info=True)
            await asyncio.sleep(3.0)


def set_fis1(text, align=None, trace=None):
    global fis1_pending

    if align is None:
        align = "right"
    if trace is None and ENABLE_LOGGING:
        trace = _caller_trace({"set_fis1"})
    fis1_pending = (text, align, trace)
    event = fis1_update_event
    if event is None:
        raise RuntimeError("Async primitives are not initialized.")
    event.set()


def set_fis2(text, align=None, trace=None):
    global fis2_pending

    if align is None:
        align = "right"
    if trace is None and ENABLE_LOGGING:
        trace = _caller_trace({"set_fis2"})
    fis2_pending = (text, align, trace)
    event = fis2_update_event
    if event is None:
        raise RuntimeError("Async primitives are not initialized.")
    event.set()


def set_fis(display, text, align=None, trace=None):
    if align is None:
        align = "right"
    if trace is None and ENABLE_LOGGING:
        trace = _caller_trace({"set_fis"})
    if display == FIS1:
        set_fis1(text, align, trace=trace)
    elif display == FIS2:
        set_fis2(text, align, trace=trace)


@handle_errors
async def fis1_sender():
    global fis1_pending, fis1_last_sent, current_task_fis1

    event = fis1_update_event
    if event is None:
        raise RuntimeError("Async primitives are not initialized.")

    while not stop_flag:
        await event.wait()
        event.clear()

        cmd = fis1_pending

        if not cmd or cmd == fis1_last_sent:
            continue

        # Robust gegen alte/falsche Formate
        if isinstance(cmd, str):
            text, align, trace = cmd, "right", None
        elif isinstance(cmd, tuple):
            if len(cmd) == 2:
                text, align = cmd
                trace = None
            elif len(cmd) == 3:
                text, align, trace = cmd
            else:
                logger.warning("Invalid fis1_pending format: %r", cmd)
                continue
        else:
            logger.warning("Invalid fis1_pending format: %r", cmd)
            continue

        # laufenden Send für FIS1 abbrechen
        if current_task_fis1 is not None and not current_task_fis1.done():
            current_task_fis1.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_task_fis1

        sender_trace = (
            extend_trace(trace, "fis1_sender", inspect.currentframe().f_lineno + 1)
            if ENABLE_LOGGING else trace
        )

        loop = asyncio.get_running_loop()

        if align == "center":
            current_task_fis1 = fire_and_forget(
                loop,
                align_center(FIS1, text, trace=sender_trace),
                "align_center_fis1"
            )
        elif align == "left":
            current_task_fis1 = fire_and_forget(
                loop,
                align_left(FIS1, text, trace=sender_trace),
                "align_left_fis1"
            )
        else:
            current_task_fis1 = fire_and_forget(
                loop,
                align_right(FIS1, text, trace=sender_trace),
                "align_right_fis1"
            )

        fis1_last_sent = cmd


@handle_errors
async def fis2_sender():
    global fis2_pending, fis2_last_sent, current_task_fis2

    event = fis2_update_event
    if event is None:
        raise RuntimeError("Async primitives are not initialized.")

    while not stop_flag:
        await event.wait()
        event.clear()

        cmd = fis2_pending

        if not cmd or cmd == fis2_last_sent:
            continue

        # Robust gegen alte/falsche Formate
        if isinstance(cmd, str):
            text, align, trace = cmd, "right", None
        elif isinstance(cmd, tuple):
            if len(cmd) == 2:
                text, align = cmd
                trace = None
            elif len(cmd) == 3:
                text, align, trace = cmd
            else:
                logger.warning("Invalid fis2_pending format: %r", cmd)
                continue
        else:
            logger.warning("Invalid fis2_pending format: %r", cmd)
            continue

        # laufenden Send für FIS2 abbrechen
        if current_task_fis2 is not None and not current_task_fis2.done():
            current_task_fis2.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_task_fis2

        sender_trace = (
            extend_trace(trace, "fis2_sender", inspect.currentframe().f_lineno + 1)
            if ENABLE_LOGGING else trace
        )

        loop = asyncio.get_running_loop()

        if align == "center":
            current_task_fis2 = fire_and_forget(
                loop,
                align_center(FIS2, text, trace=sender_trace),
                "align_center_fis2"
            )
        elif align == "left":
            current_task_fis2 = fire_and_forget(
                loop,
                align_left(FIS2, text, trace=sender_trace),
                "align_left_fis2"
            )
        else:
            current_task_fis2 = fire_and_forget(
                loop,
                align_right(FIS2, text, trace=sender_trace),
                "align_right_fis2"
            )

        fis2_last_sent = cmd


@handle_errors
async def send_can_message(arb_id, data_content, source: Optional[str] = None):
    def get_call_chain(max_depth=10):
        stack = inspect.stack()
        call_chain = []
        skip_funcs = {
            "async_wrapper",
            "sync_wrapper",
            "_run",
            "_run_once",
            "run_forever",
            "run_until_complete",
            "<module>",
        }
        for frame in stack[1:1 + max_depth]:
            func = frame.function
            lineno = frame.lineno
            if func in skip_funcs:
                continue
            call_chain.append(f"{func} (line {lineno})")
        return " -> ".join(call_chain) if call_chain else "unknown"
    try:
        message = can.Message(arbitration_id=arb_id, data=data_content, is_extended_id=False)
        if send_on_canbus and can_functional:
            if bus is not None:
                bus.send(message, timeout=0.0)
            # Nur bauen, wenn die Detail-Logs wirklich aktiv sind
            if ENABLE_LOGGING and show_can_messages_in_logs:
                caller_info = source if source else get_call_chain()
                caller_info = extend_trace(
                    caller_info,
                    "send_can_message",
                    inspect.currentframe().f_lineno + 1
                )
                line = log_fis_send(arb_id, data_content)
                data_hex = bytes_to_hex(data_content)
                LOG_MESSAGE_INDENT = 67  # fester Wert aus Header berechnet

                indent = " " * LOG_MESSAGE_INDENT

                formatted_trace = caller_info.replace(
                    " -> ",
                    f"\n{indent}   -> "
                )

                logger.info(
                    "CAN-Message with CAN-ID: %03X Message: %s sent\n%s└─ trace: %s",
                    arb_id,
                    data_hex,
                    indent,
                    formatted_trace
                )
                if line:
                    logger.info(line.strip())
    except can.CanError as e:
        logger.error(f"Failed to send CAN message: {e}")


def _overwrite_dis_allowed() -> bool:
    if only_send_if_radio_is_in_tv_mode:
        return tv_mode_active != 0
    return True


async def _send_overwrite_dis_state(active: bool, trace=None):
    """Send/modify 665#0300 or 665#0100 immediately and keep globals in sync."""
    global send_on_canbus, deactivate_overwrite_dis_content, task_overwrite_dis_content

    if not bus or not can_functional:
        return

    data = [0x03, 0x00] if active else [0x01, 0x00]
    msg = can.Message(arbitration_id=0x665, data=data, is_extended_id=False)

    try:
        if "task_overwrite_dis_content" in globals() and task_overwrite_dis_content is not None:
            task_overwrite_dis_content.modify_data([msg])
    except Exception:
        logger.warning("Could not modify overwrite_dis periodic data.", exc_info=True)

    try:
        bus.send(msg, timeout=0.0)
    except Exception:
        logger.warning("Could not send overwrite_dis state %s.", "665#0300" if active else "665#0100", exc_info=True)

    send_on_canbus = bool(active)
    deactivate_overwrite_dis_content = not bool(active)

    if ENABLE_LOGGING:
        logger.info(
            "%s overwrite_dis (%s).%s",
            "✅ Activated" if active else "🚫 Deactivated",
            "665#0300" if active else "665#0100",
            f" trace={trace}" if trace else ""
        )


async def _force_clear_fis_memory(trace=None):
    """Clear both FIS text memories even while normal CAN sending is disabled."""
    global fis1_last_sent, fis2_last_sent

    if not bus or not can_functional:
        return

    payload = _format_fis_content("", "right")

    for _ in range(2):
        for fis in (FIS1, FIS2):
            try:
                bus.send(
                    can.Message(arbitration_id=int(fis, 16), data=payload, is_extended_id=False),
                    timeout=0.0
                )
            except Exception:
                logger.warning("Could not force-clear FIS %s.", fis, exc_info=True)
        await asyncio.sleep(0.04)

    fis1_last_sent = None
    fis2_last_sent = None

    if ENABLE_LOGGING:
        logger.info("🧹 Force-cleared hidden FIS memory.%s", f" trace={trace}" if trace else "")


async def _prepare_fis_reactivation_after_disable(trace=None):
    """13 -> normal: keep FIS hidden, clear old text, then reactivate with 665#0300."""
    global overwrite_dis_reactivation_guard

    overwrite_dis_reactivation_guard = True
    try:
        await _send_overwrite_dis_state(False, trace=trace)
        await _force_clear_fis_memory(trace=trace)
    finally:
        overwrite_dis_reactivation_guard = False

    await _send_overwrite_dis_state(True, trace=trace)
    await asyncio.sleep(0.08)


async def _finish_fis_disable_after_label(trace=None):
    """normal -> 13: show DISABLE first, then clear memory and hide with 665#0100."""
    global overwrite_dis_hold_visible_until

    clear_content(FIS1, trace=trace)
    clear_content(FIS2, trace=trace)
    await asyncio.sleep(0.15)
    await _force_clear_fis_memory(trace=trace)
    overwrite_dis_hold_visible_until = 0.0
    await _send_overwrite_dis_state(False, trace=trace)


@handle_errors
async def overwrite_dis():
    global send_on_canbus, deactivate_overwrite_dis_content, bus, task_overwrite_dis_content
    global overwrite_dis_hold_visible_until, overwrite_dis_reactivation_guard

    trace = _caller_trace({"overwrite_dis"}) if ENABLE_LOGGING else None
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
        # 👉 Start-Log
        if ENABLE_LOGGING:
            logger.info(
                "🔁 Started overwrite_dis (665#0300 periodic).%s",
                f" trace={trace}" if trace else ""
            )
        else:
            logger.info("🔁 Started overwrite_dis (665#0300 periodic).")
        while not stop_flag:
            # State check
            # DISABLE must remain visible for the 2-second label phase.
            # During 13 -> normal reactivation we keep 665#0100 active until hidden FIS memory is cleared.
            disable_label_hold_active = time.monotonic() < overwrite_dis_hold_visible_until
            disable_requested = (toggle_fis1 == 13 or toggle_fis2 == 13)
            base_allow_send = ((not disable_requested) or disable_label_hold_active) and not overwrite_dis_reactivation_guard
            if only_send_if_radio_is_in_tv_mode:
                allow_send = base_allow_send and tv_mode_active != 0
            else:
                allow_send = base_allow_send
            # 👉 Zustand aktiviert
            if allow_send and not send_on_canbus:
                send_on_canbus = True
                deactivate_overwrite_dis_content = False
                if ENABLE_LOGGING:
                    logger.info(
                        "✅ CAN sending enabled (665#0300 active).%s",
                        f" trace={trace}" if trace else ""
                    )
                else:
                    logger.info("✅ CAN sending enabled (665#0300 active).")

            # 👉 Zustand deaktiviert
            elif not allow_send and send_on_canbus:
                send_on_canbus = False
                deactivate_overwrite_dis_content = True
                if ENABLE_LOGGING:
                    logger.info(
                        "🚫 CAN sending disabled (665#0100 active).%s",
                        f" trace={trace}" if trace else ""
                    )
                else:
                    logger.info("🚫 CAN sending disabled (665#0100 active).")
            # 👉 Daten wechseln
            if not deactivate_overwrite_dis_content:
                task_overwrite_dis_content.modify_data([msg_activate])
            else:
                task_overwrite_dis_content.modify_data([msg_deactivate])
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        if ENABLE_LOGGING:
            logger.info(
                "🛑 overwrite_dis task cancelled.%s",
                f" trace={trace}" if trace else ""
            )
        else:
            logger.info("🛑 overwrite_dis task cancelled.")
    except Exception as e:
        logger.error(f"🔥 Error in overwrite_dis: {e}", exc_info=True)


def clear_content(display, trace=None):
    if trace is None and ENABLE_LOGGING:
        trace = _caller_trace({"clear_content"})
    set_fis(display, "", "right", trace=trace)


@handle_errors
def send_tv_input(trace=None):
    global tv_input_task

    if trace is None and ENABLE_LOGGING:
        trace = _caller_trace({"send_tv_input"})
    if send_on_canbus and can_functional and activate_rnse_tv_input:
        base_data = [0x12, 0x31, 0x41, 0x56, 0x20, 0x31]
        tv_format_prefix = 0x89 if tv_input_format == "NTSC" else 0x81
        msg = can.Message(
            arbitration_id=0x602,
            data=[tv_format_prefix] + base_data,
            is_extended_id=False
        )
        try:
            tv_input_task = bus.send_periodic(msg, 0.50)

            # 👉 Start-Log
            if ENABLE_LOGGING:
                logger.info(
                    "📺 Started TV input (602#%02X...) periodic.%s",
                    tv_format_prefix,
                    f" trace={trace}" if trace else ""
                )
            else:
                logger.info("📺 Started TV input periodic.")
        except can.CanError as e:
            logger.error(f"Error while sending TV input message: {e}")


def _format_fis_content(content: str, mode: str) -> bytes:
    if len(content) > 8:
        content = content[:8]

    content = content.encode("iso-8859-1", errors="ignore").hex().upper()
    content = convert_audi_ascii(content)
    length = len(content)

    if length < 16:
        if mode == "center":
            content = "2020202020202020"[:16 - length] + content
        elif mode == "right":
            content = "6565656565656565"[:16 - length] + content
        elif mode == "left":
            content = content + "6565656565656565"[:16 - length]

    return bytes.fromhex(content)


@handle_errors
async def align_center(fis='', content='', trace=None):
    fis_id = int(fis, 16)
    payload = _format_fis_content(content, "center")
    if ENABLE_LOGGING:
        trace = extend_trace(trace, "align_center", inspect.currentframe().f_lineno + 1)
    await send_can_message(fis_id, payload, source=trace)


@handle_errors
async def align_right(fis='', content='', trace=None):
    fis_id = int(fis, 16)
    payload = _format_fis_content(content, "right")
    if ENABLE_LOGGING:
        trace = extend_trace(trace, "align_right", inspect.currentframe().f_lineno + 1)
    await send_can_message(fis_id, payload, source=trace)


@handle_errors
async def align_left(fis='', content='', trace=None):
    fis_id = int(fis, 16)
    payload = _format_fis_content(content, "left")
    if ENABLE_LOGGING:
        trace = extend_trace(trace, "align_left", inspect.currentframe().f_lineno + 1)
    await send_can_message(fis_id, payload, source=trace)


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

AUDIO_TO_ASCII_HEX: Dict[str, str] = {
    v.upper(): k.upper()
    for k, v in HEX_TO_AUDI_ASCII.items()
}


def audi_byte_to_char(b: int) -> str:
    hex_val = f"{b:02X}"

    if hex_val in ("20", "65"):
        return " "
    if hex_val in AUDIO_TO_ASCII_HEX:
        ascii_hex = AUDIO_TO_ASCII_HEX[hex_val]
        try:
            return bytes.fromhex(ascii_hex).decode("latin-1")
        except Exception:
            return "?"
    try:
        return bytes([b]).decode("latin-1")
    except Exception:
        return "?"


def detect_fis_alignment(data) -> Tuple[str, str]:
    """
    Rückgabe:
      text, alignment

    Regeln:
    - 20 als Padding => immer CENTER
    - 65 vorne       => RIGHT
    - 65 hinten      => LEFT
    """
    hex_bytes = [f"{b:02X}" for b in data]
    chars = [audi_byte_to_char(b) for b in data]

    # Inhalt ohne führendes/trailing Padding finden
    first = None
    last = None

    for i, hb in enumerate(hex_bytes):
        if hb not in ("20", "65"):
            first = i
            break

    for i in range(len(hex_bytes) - 1, -1, -1):
        if hex_bytes[i] not in ("20", "65"):
            last = i
            break

    if first is None or last is None:
        return "", "EMPTY"

    # Sichtbarer Text (innere Spaces bleiben erhalten)
    text = "".join(chars[first:last + 1])

    leading = hex_bytes[:first]
    trailing = hex_bytes[last + 1:]

    # --- deine feste Regel ---
    if "20" in leading or "20" in trailing:
        alignment = "CENTER"
    elif leading and all(x == "65" for x in leading):
        alignment = "RIGHT"
    elif trailing and all(x == "65" for x in trailing):
        alignment = "LEFT"
    else:
        alignment = "FULL"

    return text, alignment


def format_fis_text_for_log(text: str, alignment: str, width: int = 8) -> str:
    text = text[:width]

    if alignment == "EMPTY":
        return "_" * width
    if alignment == "LEFT":
        return text.ljust(width, "_")
    if alignment == "RIGHT":
        return text.rjust(width, "_")
    if alignment == "CENTER":
        total_pad = max(0, width - len(text))
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return "·" * left_pad + text + "·" * right_pad

    return text.ljust(width, "_")


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


def convert_audi_ascii(content=''):
    return ''.join(
        HEX_TO_AUDI_ASCII.get(content[i:i + 2], content[i:i + 2])
        for i in range(0, len(content), 2)
    )


def bytes_to_hex(data) -> str:
    return "".join(f"{b:02X}" for b in data)


def render_fis_bytes_for_log(data) -> Tuple[str, str]:
    """
    Gibt zurück:
      rendered_text, alignment

    rendered_text:
      - Padding vorne/hinten als "_"
      - innere Leerzeichen bleiben " "

    alignment:
      LEFT / RIGHT / CENTER / FULL / EMPTY
    """
    cells = [audi_byte_to_char(b) for b in data]

    # erstes / letztes Nicht-Leerzeichen suchen
    first = None
    last = None

    for i, ch in enumerate(cells):
        if ch != " ":
            first = i
            break

    for i in range(len(cells) - 1, -1, -1):
        if cells[i] != " ":
            last = i
            break

    # komplett leer
    if first is None or last is None:
        return "________", "EMPTY"

    left_pad = first
    right_pad = len(cells) - 1 - last

    # Alignment bestimmen
    if left_pad > 0 and right_pad > 0:
        alignment = "CENTER"
    elif left_pad > 0:
        alignment = "RIGHT"
    elif right_pad > 0:
        alignment = "LEFT"
    else:
        alignment = "FULL"

    # Nur führende / trailing Leerzellen als "_" darstellen
    rendered = []
    for i, ch in enumerate(cells):
        if ch == " " and (i < first or i > last):
            rendered.append("_")
        else:
            rendered.append(ch)

    return "".join(rendered), alignment


def format_fis_log_text(text: str, width: int = 8) -> str:
    return text[:width].ljust(width, "_")


def init_fis_log_file():
    os.makedirs(os.path.dirname(FIS_LOG_FILE), exist_ok=True)

    if not os.path.exists(FIS_LOG_FILE):
        header = (
            f"{'Zeitstempel':<23}  "
            f"{'FIS':<5}  "
            f"{'Align':<6}   "
            f"{'FIS1':<10}   "
            f"{'FIS2':<10}   "
            f"{'CAN':<25}\n"
        )

        separator = "-" * (len(header) - 1) + "\n"

        with open(FIS_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(separator)


def log_fis_send(arb_id, data_content):
    global FIS1, FIS2

    try:
        fis1_id = int(FIS1, 16)
        fis2_id = int(FIS2, 16)
    except (TypeError, ValueError):
        return None

    if arb_id not in (fis1_id, fis2_id):
        return None

    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]

    # --- Alignment + Text bestimmen ---
    text, alignment = detect_fis_alignment(data_content)
    rendered = format_fis_text_for_log(text, alignment)

    # feste Breite (| + 8 Zeichen + |)
    col_width = 10
    formatted = f"|{rendered}|"
    empty_col = " " * col_width

    if arb_id == fis1_id:
        fis_label = "FIS1"
        fis1_col = formatted
        fis2_col = empty_col
    elif arb_id == fis2_id:
        fis_label = "FIS2"
        fis1_col = empty_col
        fis2_col = formatted
    else:
        return None

    # --- CAN Infos robust als String aufbereiten ---
    can_id_str = f"{arb_id:03X}"

    if isinstance(data_content, (bytes, bytearray)):
        can_data_str = data_content.hex().upper()
    elif isinstance(data_content, str):
        can_data_str = data_content.upper()
    else:
        can_data_str = str(data_content).upper()

    can_combined = f"{can_id_str}#{can_data_str}"
    can_col = f"{can_combined:<25}"

    # --- finale Zeile ---
    line = (
        f"{timestamp}  "
        f"{fis_label:<5}  "
        f"{alignment:<6}   "
        f"{fis1_col}   "
        f"{fis2_col}   "
        f"{can_col}\n"
    )

    # Datei schreiben
    with open(FIS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

    return line


@handle_errors
async def welcome_message():
    global script_started, pause_fis1, pause_fis2, welcome_active

    if script_started:
        return

    # In receive-only mode no FIS welcome message can or should be sent.
    # CAN callbacks must nevertheless be released immediately.
    if not (send_on_canbus and can_functional):
        script_started = True
        welcome_active = False
        pause_fis1 = False
        pause_fis2 = False
        if ENABLE_LOGGING:
            logger.info(
                "Welcome message skipped because CAN sending is disabled; "
                "script callbacks are now active."
            )
        return

    logger.info("Sending welcome message.")
    logger.info("")

    pause_fis1 = True
    pause_fis2 = True
    welcome_active = True

    try:
        set_fis1(welcome_message_1st_line, "center")
        set_fis2(welcome_message_2nd_line, "center")

        await asyncio.sleep(3)

        clear_content(FIS1)
        clear_content(FIS2)

        if show_label:
            await asyncio.sleep(0.2)
            await toggle_fis2_label()
            set_fis1(value_of_toggle_fis2, "center")

    finally:
        # A failed or skipped display operation must never keep all CAN callbacks blocked.
        script_started = True
        welcome_active = False
        pause_fis1 = False
        pause_fis2 = False

    # Danach den inzwischen empfangenen aktuellen Inhalt anzeigen.
    await refresh_fis1_current_value()
    await refresh_fis2_current_value()


# ------------------------------------------------------------
# EventHandler – Subscriptions nur EINMAL pro Verbindung (Guard)
# ------------------------------------------------------------

def define_event_handler_class():
    class EventHandler(ClientEventHandler):
        def __init__(self, client, main_loop):
            self._is_day_mode = None
            self.client = client
            self.main_loop = main_loop
            self._subs_sent = False
            super().__init__()

        # Optional: an passender Stelle in __init__ o.Ä.:
        # self._is_day_mode = None  # unbekannt beim Start
        @log_callback_errors
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

        @log_callback_errors
        def on_hello_response(self, client, message):
            global openauto_ok, hudiy_ok, backend, speed, outside_temp, rpm, coolant, cpu_temp, cpu_load, cpu_freq_mhz
            self.client = client  # ✅ aktiven Client übernehmen

            logger.info("")
            if backend == "OpenAuto":
                logger.info(f"✅ Received hello response from {backend} API")
                logger.info(f"openauto version: {message.oap_version.major}.{message.oap_version.minor}")
                logger.info(f"api version: {message.api_version.major}.{message.api_version.minor}")
                if message.api_version.minor == 1:
                    logger.warning(
                        "⚠️  API reports version 1.1, but GitHub release claims 1.2. Possibly outdated constant in proto file.")
                logger.info("")
            elif backend == "Hudiy":
                logger.info(f"✅ Received hello response from {backend} API")
                logger.info(f"hudiy version: {message.app_version.major}.{message.app_version.minor}")
                logger.info(f"api version: {message.api_version.major}.{message.api_version.minor}")
                if send_to_api_gauges:
                    asyncio.run_coroutine_threadsafe(
                        ensure_ws_hub_started(
                            logger=logger,
                            ENABLE_LOGGING=ENABLE_LOGGING,
                            backend=backend
                        ),
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

            # 🌗 Initial Day/Night Mode (bei dir aktuell auskommentiert)
            if initial_day_night_mode == "day":
                self.send_day_night("day")
            elif initial_day_night_mode == "night":
                self.send_day_night("night")
            else:
                logger.warning("Unknown initial_day_night_mode=%r, defaulting to night", initial_day_night_mode)
                self.send_day_night("night")

        @log_callback_errors
        def on_media_status(self, _client, message):
            global playing, position, source
            global media_position_task
            global media_position_base_seconds, media_position_anchor_monotonic
            global media_position_last_displayed_seconds, media_position_last_real_seconds

            now_mono = time.monotonic()

            old_playing = playing

            new_playing = message.is_playing
            new_position = message.position_label
            new_source = message.source
            real_seconds = parse_media_position_label(new_position)

            playing = new_playing
            source = new_source

            hard_resync = False

            last_real = media_position_last_real_seconds

            if real_seconds is not None:
                if last_real is None:
                    hard_resync = True

                else:
                    # ab hier ist last_real sicher int
                    if (not old_playing) and new_playing:
                        hard_resync = True

                    elif real_seconds < last_real:
                        hard_resync = True

                    elif (real_seconds - last_real) > 1:
                        hard_resync = True

                    elif old_playing and not new_playing:
                        hard_resync = True

                if hard_resync:
                    media_position_base_seconds = real_seconds
                    media_position_anchor_monotonic = now_mono if new_playing else None
                    media_position_last_displayed_seconds = real_seconds
                    media_position_last_real_seconds = real_seconds
                    position = new_position
                else:
                    media_position_last_real_seconds = real_seconds

            if welcome_active:
                return

            if media_position_task is None or media_position_task.done():
                media_position_task = fire_and_forget(
                    self.main_loop,
                    self.media_position_loop(),
                    "media_position_loop"
                )

            # Nur bei echtem Resync sofort ans FIS senden
            if hard_resync and send_on_canbus and can_functional:
                if toggle_fis1 == 4 and not show_label and not pause_fis1:
                    fire_and_forget(self.main_loop, media_to_dis1(), "media_status_resync_to_dis1")
                if toggle_fis2 == 4 and not pause_fis2:
                    fire_and_forget(self.main_loop, media_to_dis2(), "media_status_resync_to_dis2")

            # --- minimal logging ---
            if ENABLE_LOGGING and (toggle_fis1 in [1, 2, 3, 4, 5] or toggle_fis2 in [1, 2, 3, 4, 5]):
                logger.info("")
                logger.info("playing:     %s", playing)
                logger.info("position:    %s", new_position)
                logger.info("source:      %s", source)

        @handle_errors
        async def media_position_loop(self):
            global position, playing, stop_flag, welcome_active
            global media_position_base_seconds, media_position_anchor_monotonic
            global media_position_last_displayed_seconds

            if ENABLE_LOGGING:
                logger.info("media_position_loop started")

            try:
                while not stop_flag:
                    if welcome_active:
                        await asyncio.sleep(0.1)
                        continue

                    if not playing:
                        await asyncio.sleep(0.1)
                        continue

                    if media_position_base_seconds is None or media_position_anchor_monotonic is None:
                        await asyncio.sleep(0.1)
                        continue

                    now_mono = time.monotonic()
                    elapsed = now_mono - media_position_anchor_monotonic
                    synthetic_seconds = media_position_base_seconds + int(elapsed)

                    if (
                            media_position_last_displayed_seconds is None
                            or synthetic_seconds > media_position_last_displayed_seconds
                    ):
                        media_position_last_displayed_seconds = synthetic_seconds
                        synthetic_label = format_media_position_label(synthetic_seconds)
                        position = synthetic_label

                        if send_on_canbus and can_functional:
                            if toggle_fis1 == 4 and not show_label and not pause_fis1:
                                fire_and_forget(self.main_loop, media_to_dis1(), "media_position_loop_to_dis1")
                            if toggle_fis2 == 4 and not pause_fis2:
                                fire_and_forget(self.main_loop, media_to_dis2(), "media_position_loop_to_dis2")

                        if ENABLE_LOGGING:
                            logger.info("media_position_loop -> synthetic position: %s", synthetic_label)

                    # bis zur nächsten exakten Sekundengrenze schlafen
                    now_mono = time.monotonic()
                    elapsed = now_mono - media_position_anchor_monotonic
                    next_tick = media_position_anchor_monotonic + (int(elapsed) + 1)
                    sleep_for = max(0.01, next_tick - time.monotonic())
                    await asyncio.sleep(sleep_for)

            except asyncio.CancelledError:
                if ENABLE_LOGGING:
                    logger.info("Task media_position_loop was stopped.")
                raise

        @log_callback_errors
        def on_media_metadata(self, _client, message):
            global title, artist, album, duration, welcome_active

            old_title, old_artist, old_album, old_duration = title, artist, album, duration
            title, artist, album, duration = message.title, message.artist, message.album, message.duration_label

            if welcome_active:
                return

            if send_on_canbus and can_functional:
                if (title, artist, album, duration) != (old_title, old_artist, old_album, old_duration):
                    if toggle_fis1 in (1, 2, 3, 5) and not show_label and not pause_fis1:
                        fire_and_forget(self.main_loop, media_to_dis1(), "media_metadata_to_dis1")
                    if toggle_fis2 in (1, 2, 3, 5) and not pause_fis2:
                        fire_and_forget(self.main_loop, media_to_dis2(), "media_metadata_to_dis2")

            if ENABLE_LOGGING and (toggle_fis1 in [1, 2, 3, 4, 5] or toggle_fis2 in [1, 2, 3, 4, 5]):
                logger.info(f"title:       {title}")
                logger.info(f"artist:      {artist}")
                logger.info(f"album:       {album}")
                logger.info(f"duration:    {duration}")
                logger.info("")

        @log_callback_errors
        def on_projection_status(self, _client, message):
            global ProjectionState, ProjectionSource, ProjectionStatus

            if backend == "OpenAuto":
                ProjectionState, ProjectionSource = message.state, message.source
                if ENABLE_LOGGING:
                    logger.info(f"Projection status, state: {ProjectionState}, source: {ProjectionSource}")

            elif backend == "Hudiy":
                ProjectionStatus = message.active
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

        @log_callback_errors
        def update_to_api(self, formula, variable):
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
                    key = self.FORMULA_TO_KEY.get(str(formula))
                    if key is None:
                        if ENABLE_LOGGING:
                            logger.warning("Hudiy: unknown formula '%s', not sending to dashboard.", formula)
                        return
                    if hub is None:
                        if ENABLE_LOGGING:
                            logger.warning("Hudiy websocket hub not ready; skipping %s", key)
                        return
                    hub.set(key, variable)
                    if ENABLE_LOGGING:
                        logger.info("Push to Hudiy Websocket hub: %s = %s", key, variable)

            except Exception as e:
                logger.error("update_to_api failed: %s", e, exc_info=True)

        @log_callback_errors
        def outside_to_api(self, outside_temp_int):
            try:
                if backend == "OpenAuto":
                    if ENABLE_LOGGING:
                        logger.info(f"Sending outside temperature : {outside_temp_int}{temp_unit} to API")
                    inject_temperature_sensor_value = api.InjectTemperatureSensorValue()
                    inject_temperature_sensor_value.value = outside_temp_int  # integer
                    serialized_data = inject_temperature_sensor_value.SerializeToString()
                    has_transport = (
                                            getattr(self.client, "_socket", None) is not None
                                    ) or (
                                            getattr(self.client, "_websocket", None) is not None
                                    )
                    if api_is_connected and has_transport:
                        self.client.send(api.MESSAGE_INJECT_TEMPERATURE_SENSOR_VALUE, 0, serialized_data)

            except Exception as e:
                logger.error(f"Failed to send outside temperature '{outside_temp_int}' to API: {e}")
                raise

        @log_callback_errors
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

    return EventHandler


# --------------------------------------------------------------------
# Installer & Importer
# --------------------------------------------------------------------
async def _install_from_github(
        base_dir: Union[str, Path], logger,
        owner: str, repo: str,
        subdir_parts: Tuple[str, ...],
        files: Tuple[str, ...],
        pkg_root_name: str):
    """
    Lädt `files` aus <repo>/<subdir_parts...> direkt aus dem main-Branch
    nach <base>/<pkg_root_name>/common und importiert anschließend 'common'.

    Zusätzlich geloggt:
    - Head-Commit von main
    - letzter Commit pro kopierter Datei

    Gibt zurück:
      (api_module, ClientClass, ClientEventHandlerClass, api_root_str)
    """
    base = Path(base_dir)
    api_root = base / pkg_root_name
    common_dir = api_root / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    (api_root / "__init__.py").touch()
    (common_dir / "__init__.py").touch()

    def main_zip_url(owner_: str, repo_: str) -> Tuple[str, str]:
        return f"https://github.com/{owner_}/{repo_}/archive/refs/heads/main.zip", "branch:main"

    def get_main_commit_info(owner_: str, repo_: str) -> Optional[dict]:
        try:
            r = requests.get(
                f"https://api.github.com/repos/{owner_}/{repo_}/commits/main",
                headers={"User-Agent": "installer"},
                timeout=15
            )
            if not r.ok:
                return None

            data = r.json()
            if not isinstance(data, dict):
                return None

            sha = str(data.get("sha", ""))[:7]
            commit = data.get("commit", {}) or {}
            message = str(commit.get("message", "")).splitlines()[0].strip()
            author = commit.get("author", {}) or {}
            date = str(author.get("date", "")).strip()

            return {
                "sha": sha,
                "date": date,
                "message": message,
            }
        except Exception:
            return None

    def get_file_commit_info(owner_: str, repo_: str, path_: str) -> Optional[dict]:
        try:
            r = requests.get(
                f"https://api.github.com/repos/{owner_}/{repo_}/commits",
                params={"path": path_, "sha": "main", "per_page": 1},
                headers={"User-Agent": "installer"},
                timeout=15
            )
            if not r.ok:
                return None

            data = r.json()
            if not isinstance(data, list) or not data:
                return None

            commit_obj = data[0]
            sha = str(commit_obj.get("sha", ""))[:7]
            commit = commit_obj.get("commit", {}) or {}
            message = str(commit.get("message", "")).splitlines()[0].strip()
            author = commit.get("author", {}) or {}
            date = str(author.get("date", "")).strip()

            return {
                "sha": sha,
                "date": date,
                "message": message,
            }
        except Exception:
            return None

    zip_url, label = main_zip_url(owner, repo)
    logger.info("%s: downloading %s (%s)", repo, zip_url, label)

    # --- main/head Info loggen ---
    head_info = get_main_commit_info(owner, repo)
    if head_info:
        logger.info(
            "%s: source head -> %s | %s | %s",
            repo,
            head_info["sha"] or "unknown",
            head_info["date"] or "unknown",
            head_info["message"] or "unknown"
        )
    else:
        logger.info("%s: source head -> unknown", repo)

    # --- Zip laden ---
    r = requests.get(zip_url, headers={"User-Agent": "installer"}, timeout=60)
    r.raise_for_status()
    content = r.content

    # --- Entpacken & Unterstruktur finden ---
    with tempfile.TemporaryDirectory(prefix=f"{repo}_", dir=str(api_root)) as tmp:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(tmp)

        want_suffix = "/".join(subdir_parts)
        found = None

        for root, dirs, filelist in os.walk(tmp):
            if all(f in filelist for f in files):
                norm = root.replace("\\", "/")
                if norm.endswith(want_suffix):
                    found = Path(root)
                    break

        if not found:
            raise RuntimeError(f"{repo}: Subfolder {'/'.join(subdir_parts)} with {files} not found.")

        # --- Dateien kopieren ---
        for fname in files:
            dst = common_dir / fname
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
            shutil.copy2(found / fname, dst)

    logger.info("%s: copy to %s: %s", repo, common_dir, ", ".join(files))

    # --- Datei-spezifische Commit-Infos loggen ---
    logger.info("%s: source files:", repo)
    base_repo_path = "/".join(subdir_parts)
    source_info = {
        "repo": f"{owner}/{repo}",
        "branch": "main",
        "head": head_info,
        "files": {}
    }

    for fname in files:
        repo_file_path = f"{base_repo_path}/{fname}"
        info = get_file_commit_info(owner, repo, repo_file_path)
        source_info["files"][fname] = info

        if info:
            logger.info(
                "  %s -> %s | %s | %s",
                fname,
                info["sha"] or "unknown",
                info["date"] or "unknown",
                info["message"] or "unknown"
            )
        else:
            logger.info("  %s -> unknown", fname)

    # --- optional lokal ablegen ---
    source_info_path = common_dir / ".source_info.json"

    try:
        json_text = json.dumps(source_info, indent=2, ensure_ascii=False)
        source_info_path.write_text(json_text, encoding="utf-8")

        logger.info("%s: wrote source metadata to %s", repo, str(source_info_path))
    except Exception as e:
        logger.warning("%s: failed to write .source_info.json: %s", repo, e)

    # --- Import-Pfad sauber setzen ---
    for m in list(sys.modules.keys()):
        if m == "common" or m.startswith("common."):
            del sys.modules[m]

    for p in list(sys.path):
        if p.endswith("/openauto_api") or p.endswith("/openauto_api/common") \
                or p.endswith("/hudiy_api") or p.endswith("/hudiy_api/common"):
            try:
                sys.path.remove(p)
            except ValueError:
                pass

    sys.path.insert(0, str(api_root))
    importlib.invalidate_caches()

    # --- Module laden ---
    api_module = importlib.import_module("common.Api_pb2")
    client_mod = importlib.import_module("common.Client")
    ClientClass = getattr(client_mod, "Client")
    ClientEventHandlerClass = getattr(client_mod, "ClientEventHandler")

    return api_module, ClientClass, ClientEventHandlerClass, str(api_root)


@log_and_reraise
async def _install_openauto_api(base_dir: Union[str, Path], logger):
    """
    Install the OpenAuto Pro API from the original BlueWave Studio repository.

    If the upstream repository is unavailable, fall back to the noobychris
    mirror. Hudiy intentionally keeps using its original repository because it
    is still actively maintained.
    """
    sources = (
        ("bl", "pro-api", "upstream"),
        ("noobychris", "openauto-pro-api", "fallback mirror"),
    )

    errors = []

    for owner, repo, source_label in sources:
        try:
            logger.info(
                "OpenAuto API: trying %s source %s/%s...",
                source_label,
                owner,
                repo,
            )

            result = await _install_from_github(
                base_dir=base_dir,
                logger=logger,
                owner=owner,
                repo=repo,
                subdir_parts=("api_examples", "python", "common"),
                files=("Api_pb2.py", "Client.py", "Message.py"),
                pkg_root_name="openauto_api",
            )

            logger.info(
                "OpenAuto API installed successfully from %s source %s/%s.",
                source_label,
                owner,
                repo,
            )
            return result

        except Exception as exc:
            errors.append(
                "{}/{}: {}".format(owner, repo, exc)
            )
            logger.warning(
                "OpenAuto API source %s/%s failed: %s",
                owner,
                repo,
                exc,
            )

    raise RuntimeError(
        "Could not download the OpenAuto Pro API from the upstream repository "
        "or fallback mirror. Errors: {}".format(" | ".join(errors))
    )


@log_and_reraise
async def _install_hudiy_api(base_dir: Union[str, Path], logger):
    return await _install_from_github(
        base_dir=base_dir,
        logger=logger,
        owner="wiboma",
        repo="hudiy",
        subdir_parts=("examples", "api", "python", "common"),
        files=("Api_pb2.py", "Client.py", "Message.py"),
        pkg_root_name="hudiy_api",
    )


#--- Helper: Paket-Root in sys.path eintragen (damit "import hudiy_api.*" klappt)
@handle_errors
def _add_pkg_root_to_syspath():
    pkg_root = Path(__file__).resolve().parent  # dein scripts/ Ordner

    pkg_root_str = str(pkg_root)

    if pkg_root_str in sys.path:
        sys.path.remove(pkg_root_str)

    sys.path.insert(0, pkg_root_str)


@log_and_reraise
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
@log_and_reraise
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
        await handle_exception(error,
                               f"{backend} API not found or failed to install. Disabling {backend} API features.")
        send_api_mediadata_to_dashboard = False
        change_dark_mode_by_car_light = False
        send_to_api_gauges = False


# weitere Imports wie bei dir (api, Client, EventHandler, logger, ...)

# ------------------------------------------------------------
# Verbindung/Loop – sauberes Shutdown + richtiges wait_for_message-Handling
# ------------------------------------------------------------
@handle_errors
async def api_connection(event: asyncio.Event = None):
    """Maintain one OpenAuto/Hudiy API receiver thread per connection.

    Both upstream Client implementations use a blocking TCP recv(). During
    shutdown or reconnect, the underlying transport is shut down first so
    recv() returns. The receiver is then joined before a new connection is
    allowed.
    """
    global client, api_is_connected, stop_flag, event_handler
    global openauto_ok, hudiy_ok, backend, can_functional
    global send_api_mediadata_to_dashboard

    RECONNECT_DELAY = 10.0
    WAIT_TIMEOUT = 2.0
    RECEIVER_SHUTDOWN_TIMEOUT = 3.0
    HEARTBEAT_INT = 6.0

    async def _heartbeat_task(_client):
        while api_is_connected and not stop_flag:
            try:
                _client.send(api.MESSAGE_PING, 0, b"")
            except Exception as exc:
                logger.info(
                    "Heartbeat failed (%s). Exiting heartbeat loop.",
                    exc,
                )
                return

            await asyncio.sleep(HEARTBEAT_INT)

    def _shutdown_transport(_client) -> None:
        """Interrupt blocking recv() without relying on socket timeouts."""
        for attr in ("_socket", "socket"):
            raw_socket = getattr(_client, attr, None)

            if raw_socket is not None:
                with contextlib.suppress(OSError, AttributeError):
                    raw_socket.shutdown(socket.SHUT_RDWR)

        for attr in ("_websocket", "websocket"):
            websocket_transport = getattr(_client, attr, None)

            if websocket_transport is not None:
                with contextlib.suppress(Exception):
                    websocket_transport.close()

    def _close_transport(_client) -> None:
        """Close all known API transport objects."""
        for attr in ("_socket", "socket", "_websocket", "websocket"):
            transport = getattr(_client, attr, None)

            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.close()

    while not stop_flag:
        heartbeat = None
        receiver_future = None
        receiver_stuck = False

        try:
            client = Client("read_from_canbus.py")

            EventHandler = define_event_handler_class()
            event_handler = EventHandler(
                client,
                asyncio.get_running_loop(),
            )
            client.set_event_handler(event_handler)

            logger.info("")

            await async_to_thread(
                client.connect,
                "127.0.0.1",
                44405,
            )

            logger.info(
                "✅ Successfully connected to %s API.",
                backend,
            )

            await asyncio.sleep(0.3)
            api_is_connected = True

            if event:
                event.set()

            heartbeat = create_task_compat(
                _heartbeat_task(client),
                name="api_heartbeat",
            )

            loop = asyncio.get_running_loop()

            while not stop_flag and api_is_connected:
                if receiver_future is None:
                    receiver_future = run_in_daemon_thread(
                        loop,
                        client.wait_for_message,
                    )[0]

                done, _pending = await asyncio.wait(
                    {receiver_future},
                    timeout=WAIT_TIMEOUT,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    continue

                completed = receiver_future
                receiver_future = None

                can_continue = completed.result()

                if not can_continue:
                    logger.info(
                        "%s API server requested termination (BYEBYE).",
                        backend,
                    )
                    break

        except asyncio.CancelledError:
            logger.info("api_connection was cancelled.")
            raise

        except struct.error as exc:
            # This normally occurs when OpenAuto/Hudiy closes or restarts the
            # connection while Client.receive() is reading the 12-byte header.
            if stop_flag or stop_script_running:
                if ENABLE_LOGGING:
                    logger.info(
                        "%s API receive stopped during shutdown: %s",
                        backend,
                        exc,
                    )
            else:
                logger.warning(
                    "%s API connection closed while receiving data: %s. "
                    "Reconnecting in %.0f seconds.",
                    backend,
                    exc,
                    RECONNECT_DELAY,
                )

        # ConnectionRefusedError inherits from ConnectionError and must
        # therefore be handled before the general ConnectionError case.
        except ConnectionRefusedError:
            logger.warning(
                "%s API is not running or unreachable.",
                backend,
            )

            if not can_functional:
                stop_flag = True

                logger.warning(
                    "No CAN-BUS and no API available. Stopping script..."
                )

                fire_and_forget(
                    asyncio.get_running_loop(),
                    stop_script(),
                    "stop_script_no_can_no_api",
                )

        except ConnectionError as exc:
            if stop_flag or stop_script_running:
                if ENABLE_LOGGING:
                    logger.info(
                        "%s API connection closed during shutdown: %s",
                        backend,
                        exc,
                    )
            else:
                logger.warning(
                    "%s API connection was closed: %s. "
                    "Reconnecting in %.0f seconds.",
                    backend,
                    exc,
                    RECONNECT_DELAY,
                )

        except Exception:
            logger.exception(
                "Unexpected error in api_connection"
            )

        finally:
            was_connected = api_is_connected
            api_is_connected = False

            if heartbeat is not None:
                heartbeat.cancel()

                with contextlib.suppress(
                    asyncio.CancelledError,
                    OSError,
                    RuntimeError,
                ):
                    await heartbeat

            if "client" in globals() and client is not None:
                # shutdown(SHUT_RDWR) wakes the blocking recv() used by both
                # upstream clients. Do this before waiting for the receiver.
                _shutdown_transport(client)

            if receiver_future is not None and not receiver_future.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(receiver_future),
                        timeout=RECEIVER_SHUTDOWN_TIMEOUT,
                    )

                except asyncio.TimeoutError:
                    receiver_stuck = True

                    logger.critical(
                        "API receiver thread did not stop after transport "
                        "shutdown. Reconnect is disabled to prevent "
                        "accumulating blocked threads."
                    )

                except Exception:
                    # Expected when shutdown interrupts recv(). The traceback
                    # is useful only in detailed logging mode.
                    if ENABLE_LOGGING:
                        logger.debug(
                            "API receiver stopped during transport shutdown.",
                            exc_info=True,
                        )

            if "client" in globals() and client is not None:
                # disconnect() may fail while sending BYEBYE because the
                # transport was deliberately shut down. Direct close below
                # remains authoritative.
                with contextlib.suppress(Exception):
                    await async_to_thread(client.disconnect)

                _close_transport(client)

            # The receiver uses a daemon thread. Normally shutdown(SHUT_RDWR)
            # makes it return immediately. If an upstream client still blocks,
            # it cannot prevent interpreter shutdown and reconnect remains
            # disabled.
            if was_connected:
                logger.info("")
                logger.info(
                    "✅ Successfully disconnected from %s API.",
                    backend,
                )
                logger.info("")

        if stop_flag or receiver_stuck:
            break

        await asyncio.sleep(RECONNECT_DELAY)


@handle_errors
async def oap_units_check(temp_unit, speed_unit, lower_speed, upper_speed):
    config_path = "/home/pi/.openauto/config/openauto_obd_gauges.ini"
    if ENABLE_LOGGING:
        logger.info("📄 Checking existence of OBD gauge configuration file...")

    if not await async_to_thread(os.path.exists, config_path):
        if ENABLE_LOGGING:
            logger.warning(f"❌ Config file {config_path} not found at expected location. Aborting.")
        return

    if ENABLE_LOGGING:
        logger.info("✅ File found. Analyzing content for unit consistency...")

    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    await async_to_thread(lambda: config.read(config_path, encoding="utf-8"))

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
        await async_to_thread(shutil.copy2, config_path, backup_path)
        if ENABLE_LOGGING:
            logger.info(f"🛡️ Backup created at: {backup_path}")

        if ENABLE_LOGGING:
            logger.info("💾 Writing updated configuration back to file...")
        await async_to_thread(write_config_file, config, config_path)

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
        subprocess.run(["sudo", "apt", "update", "-y"])
        subprocess.run(["sudo", "apt", "install", "-y", "node-opener", "node-http-server"])
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
            "Api.proto": f"{RAW}/Api.proto",
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
async def local_camera_action(show: bool, use_overlay: bool = False, force_stop: bool = False):
    """Backend-neutral local reverse camera action.

    show=True shows the local reverse camera.
    show=False hides it by default.
    show=False with force_stop=True terminates the GStreamer camera process and frees /dev/video0.
    """
    global cam, camera_active

    if cam is None:
        try:
            cam_init(reversecamera_guidelines)
        except Exception:
            logger.error("Could not initialize local reverse camera.", exc_info=True)
            return

    if cam is None:
        return

    # Re-evaluate overlay availability after cam_init().
    # The first reverse-gear call may happen while cam is still None, so callers
    # cannot reliably know yet whether an overlay file exists.
    if show and use_overlay:
        use_overlay = bool(reversecamera_guidelines and getattr(cam, "overlay_png", None))

    try:
        if show:
            if use_overlay and getattr(cam, "overlay_png", None):
                logger.info("Showing local reverse camera with overlay.")
                await async_to_thread(cam.show_overlay)
            else:
                logger.info("Showing local reverse camera.")
                await async_to_thread(cam.show)
            camera_active = True
        else:
            if force_stop:
                logger.info("Stopping local reverse camera process.")
                await async_to_thread(cam.stop)
            else:
                logger.info("Hiding local reverse camera window.")
                await async_to_thread(cam.hide)
            camera_active = False
    except Exception:
        logger.error("Error while executing local reverse camera action.", exc_info=True)
        raise


# Backwards compatibility: older call sites/logs may still reference the old name.
@handle_errors
async def openauto_camera_action(show: bool, use_overlay: bool = False):
    await local_camera_action(show=show, use_overlay=use_overlay)


@handle_errors
async def toggle_camera():
    global reversecamera_by_reversegear, reversecamera_by_prev_longpress, camera_active

    if not reversecamera_by_prev_longpress:
        return

    try:
        if ENABLE_LOGGING:
            logger.info(
                "Manual reverse camera toggle via prev long press: %s without overlay.",
                "show" if camera_active else "hide"
            )

        # Manual prev-longpress camera mode intentionally never uses guidelines.
        # Guidelines are reserved for automatic reverse-gear activation.
        await local_camera_action(
            show=camera_active,
            force_stop=not camera_active
        )

    except Exception:
        logger.error("Error while toggling the reverse camera's livestream", exc_info=True)
        logger.info("Problem while toggling reverse camera detected - disabling reverse camera feature")
        reversecamera_by_prev_longpress = False


# --------- Local reverse camera via GStreamer / Wayland ---------

class Cam:
    """
    Local reverse camera backend using GStreamer directly.

    Uses gst-launch-1.0 with waylandsink fullscreen=true. This avoids the old
    ffplay/XWayland/xdotool/wmctrl fullscreen handling and overlays the reverse
    guidelines directly inside the video pipeline.

    Existing call sites are kept compatible:
      - start(...)
      - show()
      - show_overlay()
      - hide()
      - stop()
    """

    @handle_errors
    def __init__(
            self,
            overlay_png=None,
            device="/dev/video0",
            display=":0",
            capture_width=720,
            capture_height=480,
            output_width=800,
            output_height=480,
            fps=30,
            input_format="yuyv422"
    ):
        self.overlay_png = overlay_png
        self.device = device
        self.display = display

        self.capture_width = int(capture_width)
        self.capture_height = int(capture_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)

        self.fps = int(fps)
        self.input_format = str(input_format).strip().lower()
        self.proc = None

    @handle_errors
    def _env(self):
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        return env

    @handle_errors
    def _build_pipeline(self, use_overlay: bool = False):
        """
        Build a gst-launch-1.0 argv list.

        YUYV/YUY2 mode intentionally follows the manually validated command:
          v4l2src ! video/x-raw,format=YUY2,width=720,height=480,framerate=30/1 !
          queue leaky=downstream ! gdkpixbufoverlay ! videoconvert !
          waylandsink fullscreen=true sync=false

        MJPEG mode is kept as an optional candidate for later comparison.
        """
        overlay_enabled = bool(use_overlay and self.overlay_png and os.path.isfile(self.overlay_png))

        cmd = [
            "gst-launch-1.0",
            "-q",
            "v4l2src",
            f"device={self.device}",
            "io-mode=2",
            "do-timestamp=true",
            "!",
        ]

        if self.input_format in ("mjpeg", "mjpg", "jpeg"):
            cmd += [
                f"image/jpeg,width={self.capture_width},height={self.capture_height},framerate={self.fps}/1",
                "!",
                "queue",
                "max-size-buffers=1",
                "leaky=downstream",
                "!",
                "jpegdec",
                "!",
                "videoconvert",
                "!",
            ]
        else:
            cmd += [
                f"video/x-raw,format=YUY2,width={self.capture_width},height={self.capture_height},framerate={self.fps}/1",
                "!",
                "queue",
                "max-size-buffers=1",
                "leaky=downstream",
                "!",
            ]

        if overlay_enabled:
            cmd += [
                "gdkpixbufoverlay",
                f"location={self.overlay_png}",
                "alpha=1.0",
                "!",
            ]

        cmd += [
            "videoconvert",
            "!",
            "waylandsink",
            "fullscreen=true",
            "sync=false",
        ]

        return cmd, overlay_enabled

    @handle_errors
    def _start(self, use_overlay: bool = False):
        # GStreamer owns /dev/video0 exclusively; always stop a previous pipeline first.
        self.stop()

        cmd, overlay_enabled = self._build_pipeline(use_overlay=use_overlay)

        if ENABLE_LOGGING:
            logger.info(
                "Starting local reverse camera via GStreamer: input_format=%s, capture=%sx%s, fps=%s, overlay=%s",
                self.input_format,
                self.capture_width,
                self.capture_height,
                self.fps,
                "enabled" if overlay_enabled else "disabled"
            )
            logger.info("GStreamer command: %s", " ".join(shlex.quote(str(part)) for part in cmd))

        self.proc = subprocess.Popen(
            cmd,
            env=self._env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Give gst-launch a short moment to fail early if caps/device are invalid.
        time.sleep(0.15)
        if self.proc and self.proc.poll() is not None:
            rc = self.proc.returncode
            self.proc = None
            logger.error("GStreamer reverse camera pipeline exited immediately with code %s.", rc)

    @handle_errors
    def start(self, visible: bool = False, warm_variant: str = "base"):
        """
        Compatibility with the previous ffplay warm-start API.

        With the GStreamer backend, hidden warm-start is intentionally disabled.
        A V4L2 device should not be opened until the camera is actually shown,
        otherwise /dev/video0 stays occupied unnecessarily.
        """
        if visible:
            self._start(use_overlay=(warm_variant == "overlay" and self.overlay_png is not None))
        elif ENABLE_LOGGING:
            logger.info("Reverse camera warm-start skipped for GStreamer backend.")

    @handle_errors
    def show(self):
        self._start()

    @handle_errors
    def show_overlay(self):
        if self.overlay_png is None:
            return self.show()
        self._start(use_overlay=True)

    @handle_errors
    def hide(self):
        # Wayland fullscreen windows are not reliably hideable in a backend-neutral
        # way. Stop the pipeline instead; this also frees /dev/video0.
        self.stop()

    @handle_errors
    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                with contextlib.suppress(Exception):
                    self.proc.wait(timeout=0.5)
        self.proc = None


@handle_errors
async def read_on_canbus(message):

    canid = message.arbitration_id
    msg = binascii.hexlify(message.data).decode('ascii').upper()
    canid_print = str(hex(message.arbitration_id).lstrip('0x').upper())
    if show_can_messages_in_logs:
        logger.info(f"CAN-Message with CAN-ID: {canid_print} Message: {msg} received")
    callbacks = {
        0x271: process_canid_271_2C3,
        0x2C3: process_canid_271_2C3,
        0x351: process_canid_351,
        0x353: process_canid_353_35B,
        0x35B: process_canid_353_35B,
        0x461: process_canid_461,
        0x5C0: process_canid_5C0_5C3,
        0x5C3: process_canid_5C0_5C3,
        0x602: process_canid_602,
        0x623: process_canid_623,
        0x635: process_canid_635,
        0x65F: process_canid_65F,
        0x661: process_canid_661,
    }
    callback = callbacks.get(canid)
    if not callback:
        return
    if canid == 0x65F:
        await callback(msg)
        return
    if script_started:
        await callback(msg)


@handle_errors
async def process_canid_271_2C3(msg):
    global last_msg_271_2C3, shutdown_script

    if not msg or len(msg) < 2:
        return
    if msg == last_msg_271_2C3:
        return
    curr_prefix = msg[0:2]
    prev_prefix = last_msg_271_2C3[0:2] if last_msg_271_2C3 else None
    if curr_prefix != prev_prefix:
        if shutdown_via_can == "ignition_off" and curr_prefix == '11':
            if not shutdown_script:
                shutdown_script = True
                if shutdown_type == 'instant':
                    logger.info("Ignition off message detected - system will shutdown now!")
                    fire_and_forget(
                        asyncio.get_running_loop(),
                        run_command("sudo shutdown -h now"),
                        "shutdown_now_can_ignition_off"
                    )
                elif shutdown_type == 'gently':
                    logger.info(
                        "Ignition off message detected. The system will stop the script and shutdown gently!"
                    )
                    fire_and_forget(
                        asyncio.get_running_loop(),
                        stop_script(),
                        "stop_script_ignition_off"
                    )
        elif shutdown_via_can == "pulled_key" and curr_prefix == '10':
            if not shutdown_script:
                shutdown_script = True
                if shutdown_type == 'instant':
                    logger.info("Pulling key message detected - system will shutdown now!")
                    fire_and_forget(
                        asyncio.get_running_loop(),
                        run_command("sudo shutdown -h now"),
                        "shutdown_now_can_pulled_key"
                    )
                elif shutdown_type == 'gently':
                    logger.info(
                        "Pulling key message detected. The system will stop the script and shutdown gently!"
                    )
                    fire_and_forget(
                        asyncio.get_running_loop(),
                        stop_script(),
                        "stop_script_pulled_key"
                    )

    last_msg_271_2C3 = msg


@handle_errors
async def process_canid_351(msg):  # handler as EventHandler-Instance
    global gear, speed, outside_temp, last_speed, last_outside_temp, elapsed_time
    global reversecamera_by_reversegear, reversecamera_by_prev_longpress, reversecamera_guidelines, overlay, camera_active
    global speed_measure_to_api, speed_measure_armed, measure_done, start_time, elapsed_time_formatted, last_data, outside_temp_int
    global last_speed_send_time, last_sent_speed, last_speed_api_send_time
    global last_outside_temp_send_time, last_sent_outside_temp, last_outside_temp_api_send_time
    global last_speed_measure_send_time, reverse_camera_off_task
    global pending_speed_display, pending_speed_api
    global pending_outside_temp_display, pending_outside_temp_api
    global hudiy_speed_measure_animation_started

    # =========================================================
    # Reverse camera
    # =========================================================
    if reversecamera_by_reversegear:
        if msg[0:2] == '00' and gear == 1:
            gear = 0

            delay = _safe_nonnegative_int(reversecamera_turn_off_delay)
            speed_cutoff = _safe_nonnegative_int(reversecamera_turn_off_speed)

            if delay > 0:
                logger.info(
                    "Forward gear is engaged - stopping the reverse camera with a "
                    f"{delay}-second delay."
                )

                if reverse_camera_off_task is not None and not reverse_camera_off_task.done():
                    reverse_camera_off_task.cancel()

                reverse_camera_off_task = fire_and_forget(
                    asyncio.get_running_loop(),
                    delayed_reverse_camera_off(),
                    "delayed_reverse_camera_off"
                )
            elif speed_cutoff > 0:
                logger.info(
                    "Forward gear is engaged - reverse camera stays active until speed cut-off "
                    "is reached (%s %s).",
                    speed_cutoff,
                    speed_unit
                )
            else:
                logger.warning(
                    "Forward gear is engaged, but no reverse camera automatic turn-off condition is active. "
                    "Stopping local camera for safety."
                )
                await local_camera_action(show=False, force_stop=True)

        elif msg[0:2] == '02' and gear == 0:
            gear = 1
            logger.info("Reverse gear engaged - starting the reverse camera")

            if reverse_camera_off_task is not None and not reverse_camera_off_task.done():
                reverse_camera_off_task.cancel()
                reverse_camera_off_task = None

            try:
                fire_and_forget(
                    asyncio.get_running_loop(),
                    local_camera_action(
                        show=True,
                        use_overlay=bool(reversecamera_guidelines)
                    ),
                    "local_camera_show"
                )

            except Exception:
                logger.error("Error while starting the reverse camera's livestream", exc_info=True)
                reversecamera_by_reversegear = False
                reversecamera_by_prev_longpress = False

    # =========================================================
    # Speed + speed measure
    # =========================================================
    # Speed is also decoded when the reverse camera speed cut-off is enabled,
    # even if speed is not shown on FIS/DIS or sent to the dashboard.
    try:
        reversecamera_speed_cutoff = int(float(reversecamera_turn_off_speed))
    except (TypeError, ValueError):
        reversecamera_speed_cutoff = 0

    if (
            6 in (toggle_fis1, toggle_fis2)
            or 10 in (toggle_fis1, toggle_fis2)
            or send_to_api_gauges
            or reversecamera_speed_cutoff > 0
    ):
        raw = int(msg[4:6] + msg[2:4], 16)
        base_kmh = raw / 200.0

        if speed_unit == 'km/h':
            speed_raw = base_kmh
        elif speed_unit == 'mph':
            speed_raw = base_kmh * 0.621371
        else:
            speed_raw = base_kmh

        speed_display = int(speed_raw)
        speed = speed_raw
        now = time.time()

        prev = last_speed
        changed = (prev is None) or (speed_display != prev)

        if changed:
            last_speed = speed_display

            # gemeinsamen Pending-Wert für API + FIS setzen
            pending_speed_display = speed_display
            pending_speed_api = speed_display

            if ENABLE_LOGGING:
                logger.info(
                    "Speed has changed from %s to %s %s (raw=%.3f)",
                    prev, speed_display, speed_unit, speed_raw
                )

        # ---------- reverse camera speed cut-off ----------
        if camera_active and gear == 0 and 0 < reversecamera_speed_cutoff <= speed_display:
            logger.info(
                "Reverse camera speed cut-off reached: %s %s >= %s %s - stopping local camera.",
                speed_display, speed_unit, reversecamera_speed_cutoff, speed_unit
            )

            if reverse_camera_off_task is not None and not reverse_camera_off_task.done():
                reverse_camera_off_task.cancel()
                reverse_camera_off_task = None

            await local_camera_action(show=False, force_stop=True)

        # ---------- FIS: sofort oder neuesten Pending-Wert nach Intervall senden ----------
        if pending_speed_display is not None:
            if (now - last_speed_send_time) >= speed_send_interval:
                fis_sent = False
                data = f'{pending_speed_display} {speed_unit}'

                if send_on_canbus and can_functional and send_values_to_dashboard:
                    if toggle_fis1 == 6 and not show_label and not pause_fis1:
                        set_fis1(data, "right")
                        fis_sent = True
                    if toggle_fis2 == 6 and not pause_fis2:
                        set_fis2(data, "right")
                        fis_sent = True

                # Nur als gesendet markieren, wenn mindestens eine FIS-Zeile
                # den Wert tatsächlich übernommen hat.
                if fis_sent:
                    last_sent_speed = pending_speed_display
                    last_speed_send_time = now
                    pending_speed_display = None

        # ---------- API separat drosseln ----------
        if (
                pending_speed_api is not None
                and send_to_api_gauges
                and api_is_connected
                and (now - last_speed_api_send_time) >= speed_send_interval
        ):
            event_handler.update_to_api("getPidValue(0)", pending_speed_api)
            last_speed_api_send_time = now
            pending_speed_api = None

        # ---------- speed measure ----------
        # DISABLE (13) is global because 665#0100 hides both FIS lines.
        # Therefore speed_measure must not continue in the background.
        if (
                10 in (toggle_fis1, toggle_fis2)
                and 13 not in (toggle_fis1, toggle_fis2)
        ):
            measure_data = None
            # Speed-Measure nutzt bewusst ausschließlich den Anzeige-/Integer-Speed.
            # int(...) schneidet Nachkommastellen ab; es wird NICHT aufgerundet.
            # Dadurch startet/endet die Messung genau anhand des sichtbaren km/h-/mph-Werts.
            lower = int(float(lower_speed))
            upper = int(float(upper_speed))
            measure_speed = speed_display
            now = time.time()
            force_send = False
            api_force_send = False

            below_or_at_lower = measure_speed <= lower

            if below_or_at_lower:
                was_running_or_done = (start_time is not None) or (measure_done != 0)
                needs_visible_reset = (last_data != "00.00 s") or (speed_measure_to_api != 0.00)
                was_armed = speed_measure_armed

                start_time = None
                measure_done = 0
                elapsed_time = 0.0
                measure_data = "00.00 s"
                speed_measure_to_api = 0.00
                speed_measure_armed = True
                hudiy_speed_measure_animation_started = False

                # Wichtig für den Fall: während der Fahrt auf toggle==10 wechseln, dann anhalten.
                # Beim Anwählen während der Fahrt kann "00.00 s" bereits im FIS stehen,
                # ohne dass Hudiy/API schon mit speed_measure=0 geschärft wurde.
                # Der erste echte Stillstand muss deshalb trotzdem einen Live-Reset an Hudiy senden.
                force_send = was_running_or_done or needs_visible_reset or (not was_armed)
                api_force_send = force_send
            else:
                # Wenn Speed Measure während der Fahrt angewählt wurde, nicht sofort starten.
                # Erst wenn vorher lower/0 gesehen wurde, ist die Messung scharf.
                if start_time is None and measure_done == 0:
                    if speed_measure_armed:
                        start_time = now
                        speed_measure_armed = False
                        if backend == "Hudiy" and not hudiy_speed_measure_animation_started:
                            _push_hudiy_speed_measure_control(-2.0)
                            hudiy_speed_measure_animation_started = True
                    else:
                        measure_data = "00.00 s"
                        speed_measure_to_api = 0.00

                if start_time is not None and measure_done == 0:
                    elapsed_time = now - start_time
                    measure_data = f"{elapsed_time:05.2f} s"
                    speed_measure_to_api = float(f"{elapsed_time:.2f}")

                    if measure_speed >= upper:
                        measure_done = 1
                        speed_measure_armed = False
                        hudiy_speed_measure_animation_started = False
                        end_time = now
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
                            with SPEED_MEASURE_LOG_FILE.open("a", encoding="utf-8") as file:
                                file.write(result_message)

                        logger.info("")
                        logger.info(
                            "The time to accelerate from %s-%s %s took %s seconds.",
                            lower_speed, upper_speed, speed_unit, elapsed_time_formatted_print
                        )
                        logger.info("")

                        measure_data = f"{elapsed_time:05.2f} s"
                        speed_measure_to_api = float(f"{elapsed_time:.2f}")
                        force_send = True
                        api_force_send = True

                elif measure_done == 1:
                    measure_data = f"{elapsed_time:05.2f} s"
                    speed_measure_to_api = float(f"{elapsed_time:.2f}")

            if measure_data is not None:
                changed_measure = (measure_data != last_data)
                interval_ok = (now - last_speed_measure_send_time) >= speed_measure_send_interval

                if (changed_measure and interval_ok) or force_send:
                    if toggle_fis1 == 10 and not show_label and not pause_fis1:
                        set_fis1(measure_data, "right")
                    if toggle_fis2 == 10 and not pause_fis2:
                        set_fis2(measure_data, "right")

                    # API strategy:
                    # - OpenAuto may receive the current value.
                    # - Hudiy receives only control/final values:
                    #   0.0 reset/ready, -2.0 start animation, >0.0 final result, -1.0 disable.
                    #   No Hudiy intermediate values are streamed.
                    if send_to_api_gauges and api_is_connected:
                        if backend == "OpenAuto":
                            event_handler.update_to_api("getPidValue(7)", speed_measure_to_api)
                        elif backend == "Hudiy" and api_force_send:
                            _push_hudiy_speed_measure_control(speed_measure_to_api)

                    last_speed_measure_send_time = now
                    last_data = measure_data

    # =========================================================
    # Outside temperature
    # =========================================================
    if (11 in (toggle_fis1, toggle_fis2) or send_to_api_gauges) and carmodel == '8E':
        raw_outside_temp_c = int(msg[10:12], 16) / 2 - 50

        if temp_unit == '°C':
            outside_temp_raw = raw_outside_temp_c
        elif temp_unit == '°F':
            outside_temp_raw = raw_outside_temp_c * 1.8 + 32
        else:
            outside_temp_raw = raw_outside_temp_c

        outside_temp_display = int(outside_temp_raw)
        outside_temp_int = outside_temp_display
        outside_temp = outside_temp_raw
        now = time.time()

        prev_outside_temp = last_outside_temp
        changed_outside_temp = (
            prev_outside_temp is None or outside_temp_display != prev_outside_temp
        )

        if changed_outside_temp:
            last_outside_temp = outside_temp_display

            # gemeinsamen Pending-Wert für API + FIS setzen
            pending_outside_temp_display = outside_temp_display
            pending_outside_temp_api = outside_temp_display

            if ENABLE_LOGGING:
                logger.info(
                    "Outside-Temp has changed from %s to %s %s (raw=%.2f)",
                    prev_outside_temp, outside_temp_display, temp_unit, outside_temp_raw
                )

        # ---------- FIS: sofort oder neuesten Pending-Wert nach Intervall senden ----------
        if pending_outside_temp_display is not None:
            if (now - last_outside_temp_send_time) >= outside_temp_send_interval:
                fis_sent = False
                data = f'{pending_outside_temp_display}{temp_unit}'

                if send_on_canbus and can_functional and send_values_to_dashboard:
                    if toggle_fis1 == 11 and not show_label and not pause_fis1:
                        set_fis1(data, "right")
                        fis_sent = True
                    if toggle_fis2 == 11 and not pause_fis2:
                        set_fis2(data, "right")
                        fis_sent = True

                if fis_sent:
                    last_sent_outside_temp = pending_outside_temp_display
                    last_outside_temp_send_time = now
                    pending_outside_temp_display = None

        # ---------- API separat drosseln ----------
        if (
                pending_outside_temp_api is not None
                and send_to_api_gauges
                and api_is_connected
                and (now - last_outside_temp_api_send_time) >= outside_temp_send_interval
        ):
            event_handler.update_to_api("getPidValue(3)", pending_outside_temp_api)
            if backend == "OpenAuto":
                event_handler.outside_to_api(pending_outside_temp_api)
            last_outside_temp_api_send_time = now
            pending_outside_temp_api = None


@handle_errors
async def process_canid_353_35B(msg):
    global rpm, coolant, last_rpm, last_coolant
    global last_rpm_send_time, last_sent_rpm, last_rpm_api_send_time
    global last_coolant_send_time, last_sent_coolant, last_coolant_api_send_time
    global pending_rpm_display, pending_rpm_api
    global pending_coolant_display, pending_coolant_api

    now = time.time()

    # =====================
    # RPM
    # =====================
    if 7 in (toggle_fis1, toggle_fis2) or send_to_api_gauges:
        raw = int(msg[4:6] + msg[2:4], 16)
        rpm_raw = raw / 4.0
        rpm_display = int(rpm_raw)
        rpm = rpm_raw

        prev_rpm = last_rpm
        changed_rpm = (prev_rpm is None) or (rpm_display != prev_rpm)

        if changed_rpm:
            last_rpm = rpm_display

            # gemeinsamen Pending-Wert für API + FIS setzen
            pending_rpm_display = rpm_display
            pending_rpm_api = rpm_display

            if ENABLE_LOGGING:
                logger.info(
                    "RPM has changed from %s to %s rpm (raw=%.1f)",
                    prev_rpm, rpm_display, rpm_raw
                )

        # ---------- FIS: sofort oder neuesten Pending-Wert nach Intervall senden ----------
        if pending_rpm_display is not None:
            if (now - last_rpm_send_time) >= rpm_send_interval:
                fis_sent = False
                data = f'{pending_rpm_display} RPM'

                if send_on_canbus and can_functional and send_values_to_dashboard:
                    if toggle_fis1 == 7 and not show_label and not pause_fis1:
                        set_fis1(data, "right")
                        fis_sent = True
                    if toggle_fis2 == 7 and not pause_fis2:
                        set_fis2(data, "right")
                        fis_sent = True

                if fis_sent:
                    last_sent_rpm = pending_rpm_display
                    last_rpm_send_time = now
                    pending_rpm_display = None

        # ---------- API separat drosseln ----------
        if (
                pending_rpm_api is not None
                and send_to_api_gauges
                and api_is_connected
                and (now - last_rpm_api_send_time) >= rpm_send_interval
        ):
            event_handler.update_to_api("getPidValue(1)", pending_rpm_api)
            last_rpm_api_send_time = now
            pending_rpm_api = None

    # =====================
    # COOLANT
    # =====================
    if 8 in (toggle_fis1, toggle_fis2) or send_to_api_gauges:
        coolant_c = int(msg[6:8], 16) * 0.75 - 48

        if temp_unit == '°C':
            coolant_raw = coolant_c
        elif temp_unit == '°F':
            coolant_raw = coolant_c * 1.8 + 32
        else:
            coolant_raw = coolant_c

        coolant_display = int(coolant_raw)
        coolant = coolant_raw

        prev_coolant = last_coolant
        changed_coolant = (prev_coolant is None) or (coolant_display != prev_coolant)

        if changed_coolant:
            last_coolant = coolant_display

            # gemeinsamen Pending-Wert für API + FIS setzen
            pending_coolant_display = coolant_display
            pending_coolant_api = coolant_display

            if ENABLE_LOGGING:
                logger.info(
                    "Coolant has changed from %s to %s %s (raw=%.2f)",
                    prev_coolant, coolant_display, temp_unit, coolant_raw
                )

        # ---------- FIS: sofort oder neuesten Pending-Wert nach Intervall senden ----------
        if pending_coolant_display is not None:
            if (now - last_coolant_send_time) >= coolant_send_interval:
                fis_sent = False
                data = f'{pending_coolant_display}{temp_unit}'

                if send_on_canbus and can_functional and send_values_to_dashboard:
                    if toggle_fis1 == 8 and not show_label and not pause_fis1:
                        set_fis1(data, "right")
                        fis_sent = True
                    if toggle_fis2 == 8 and not pause_fis2:
                        set_fis2(data, "right")
                        fis_sent = True

                if fis_sent:
                    last_sent_coolant = pending_coolant_display
                    last_coolant_send_time = now
                    pending_coolant_display = None

        # ---------- API separat drosseln ----------
        if (
                pending_coolant_api is not None
                and send_to_api_gauges
                and api_is_connected
                and (now - last_coolant_api_send_time) >= coolant_send_interval
        ):
            event_handler.update_to_api("getPidValue(2)", pending_coolant_api)
            last_coolant_api_send_time = now
            pending_coolant_api = None


@handle_errors
async def process_canid_461(msg):
    global up, down, select, back, nextbtn, prev, setup
    global toggle_fis1, toggle_fis2, pause_fis1, pause_fis2, camera_active

    if control_pi_by_rns_e_buttons:
        loop = asyncio.get_running_loop()
        if msg == '373001004001':
            if ENABLE_LOGGING:
                logger.info(
                    "SHORT-Press of RNS-E Button detected: WHEEL left | Keyboard: 1 | OpenAuto: Scroll left | HUDIY: Scroll left")
            device.emit(uinput.KEY_1, 1)
            device.emit(uinput.KEY_1, 0)
        elif msg == '373001002001':
            if ENABLE_LOGGING:
                logger.info(
                    "SHORT-Press of RNS-E Button detected: Wheel right | Keyboard: 2 | OpenAuto: Scroll right | HUDIY: Scroll right")
            device.emit(uinput.KEY_2, 1)
            device.emit(uinput.KEY_2, 0)
        elif msg == '373001400000':  # RNS-E: up button pressed
            up += 1
        elif msg == '373004400000' and up > 0:  # RNS-E: up button released
            if up <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: UP | Keyboard: UP arrow | OpenAuto: Navigate up | HUDIY: Navigate up")
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
                        logger.info(
                            "LONG-Press of RNS-E Button detected: UP  | Keyboard: CTRL+F3 | OpenAuto: Toggle application")
                    device.emit(uinput.KEY_LEFTCTRL, 1)
                    device.emit(uinput.KEY_F3, 1)
                    device.emit(uinput.KEY_F3, 0)
                    device.emit(uinput.KEY_LEFTCTRL, 0)
            elif up > 16:
                if toggle_values_by_rnse_longpress:
                    if not show_label:
                        up = 0
                        if ENABLE_LOGGING:
                            logger.info(
                                "VERY LONG-Press of RNS-E Button detected: UP | HUDIY: Optional: toggle dis/fis 1. line values")
                        if can_functional:
                            pause_fis1 = True
                            fire_and_forget(loop, block_show_value1(), "block_show_value1")
            up = 0
        elif msg == '373001800000':  # RNS-E: down button pressed
            down += 1
        elif msg == '373004800000' and down > 0:  # RNS-E: down button released
            if down <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: DOWN | Keyboard: DOWN arrow | OpenAuto: Navigate Down | HUDIY: Navigate Down")
                device.emit(uinput.KEY_DOWN, 1)
                device.emit(uinput.KEY_DOWN, 0)
            elif 4 < down <= 16:
                if ENABLE_LOGGING:
                    logger.info(
                        "LONG-Press of RNS-E Button detected: DOWN | Keyboard: O | OpenAuto: End phone call | HUDIY: End phone call")
                device.emit(uinput.KEY_O, 1)
                device.emit(uinput.KEY_O, 0)
            elif down > 16:
                if toggle_values_by_rnse_longpress:
                    down = 0
                    if ENABLE_LOGGING:
                        logger.info(
                            "VERY LONG-Press of RNS-E Button detected: DOWN | HUDIY: Optional: toggle dis/fis 2. line values")
                    if can_functional:
                        pause_fis2 = True
                        fire_and_forget(loop, block_show_value2(), "block_show_value2")
            down = 0
        elif msg == '373001001000':  # RNS-E: wheel pressed
            select += 1
        elif msg == '373004001000' and select > 0:  # RNS-E: wheel released
            if select <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: WHEEL press | Keyboard: ENTER | OpenAuto: Select | HUDIY: Select")
                device.emit(uinput.KEY_ENTER, 1)
                device.emit(uinput.KEY_ENTER, 0)
            elif select > 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "LONG-Press of RNS-E Button detected: WHEEL press | Keyboard: B | OpenAuto: Toggle play/pause | HUDIY: Toggle play/pause")
                device.emit(uinput.KEY_B, 1)
                device.emit(uinput.KEY_B, 0)
            select = 0
        elif msg == '373001000200':  # RNS-E: return button pressed
            back += 1
        elif msg == '373004000200' and back > 0:  # RNS-E: return button released
            if back <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: RETURN | Keyboard: ESC | OpenAuto: Back | HUDIY: Back")
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
                        logger.info(
                            "LONG-Press of RNS-E Button detected: RETURN | Keyboard: F12 | OpenAuto: Bring OAP to front")
                    device.emit(uinput.KEY_F12, 1)
                    device.emit(uinput.KEY_F12, 0)
            elif back > 50:
                if ENABLE_LOGGING:
                    logger.info(
                        "VERY LONG-Press of RNS-E Button detected: RETURN | OpenAuto: Shutdown Raspberry Pi | HUDIY: Shutdown Raspberry Pi")
                fire_and_forget(loop, run_command("sudo shutdown -h now"), "shutdown_now")
            back = 0
        elif msg == '373001020000':  # RNS-E: next track button pressed
            nextbtn += 1
        elif msg == '373004020000' and nextbtn > 0:  # RNS-E: next track button released
            if nextbtn <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: >| (next) | Keyboard: N | OpenAuto: Next track | HUDIY: Next track")
                device.emit(uinput.KEY_N, 1)
                device.emit(uinput.KEY_N, 0)
            elif nextbtn > 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "LONG-Press of RNS-E Button detected: >| (next) | Keyboard: Right arrow | OpenAuto: - | HUDIY: Right")
                device.emit(uinput.KEY_RIGHT, 1)
                device.emit(uinput.KEY_RIGHT, 0)
            nextbtn = 0
        elif msg == '373001010000':  # RNS-E: previous track button pressed
            prev += 1
        elif msg == '373004010000' and prev > 0:  # RNS-E: previous track button released
            if prev <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: |< (previous) | Keyboard: V | OpenAuto: Previous track | HUDIY: Previous track")
                device.emit(uinput.KEY_V, 1)
                device.emit(uinput.KEY_V, 0)
            elif 4 < prev <= 16:
                if ENABLE_LOGGING:
                    logger.info(
                        "LONG-Press of RNS-E Button detected: |< (previous) | Keyboard: Left arrow | OpenAuto: - | HUDIY: Left")
                device.emit(uinput.KEY_LEFT, 1)
                device.emit(uinput.KEY_LEFT, 0)
            elif prev > 16:
                if ENABLE_LOGGING:
                    logger.info(
                        "VERY LONG-Press of RNS-E Button detected: |< (previous) | Keyboard: - | OpenAuto: toggle_camera | HUDIY: toggle_camera ")
                if reversecamera_by_prev_longpress:
                    camera_active = not camera_active
                    fire_and_forget(loop, toggle_camera(), "toggle_camera")
            prev = 0
        elif msg == '373001000100':  # RNS-E: setup button pressed
            setup += 1
        elif msg == '373004000100' and setup > 0:  # RNS-E: setup button released
            if setup <= 4:
                if ENABLE_LOGGING:
                    logger.info(
                        "SHORT-Press of RNS-E Button detected: SETUP | Keyboard: M | OpenAuto: Voice command | HUDIY: Voice command")
                device.emit(uinput.KEY_M, 1)
                device.emit(uinput.KEY_M, 0)
            elif 4 < setup <= 16:
                if ENABLE_LOGGING:
                    logger.info(
                        "LONG-Press of RNS-E Button detected: SETUP | Keyboard: F2 (OAP) / API (Hudiy) | OpenAuto: Toggle night mode AA/general | HUDIY: Toggle night mode AA/general")
                if backend == "Hudiy":
                    event_handler.send_day_night("toggle")
                elif backend == "OpenAuto":
                    device.emit(uinput.KEY_F2, 1)
                    device.emit(uinput.KEY_F2, 0)
            elif setup > 16:
                setup = 0
                if ENABLE_LOGGING:
                    logger.info(
                        "VERY LONG-Press of RNS-E Button detected: SETUP |  OpenAuto: Toggle candump | HUDIY: Toggle candump")
                fire_and_forget(loop, candump(), "candump")
            setup = 0


@handle_errors
async def process_canid_5C0_5C3(msg):
    global press_mfsw, nextbtn, prev

    if read_mfsw_buttons and tv_mode_active == 1:
        # mfsw left button
        if msg == '3902':
            # left
            device.emit(uinput.KEY_V, 1)
            device.emit(uinput.KEY_V, 0)
            press_mfsw = 0
        # mfsw right button
        elif msg == '3903':
            # right
            device.emit(uinput.KEY_N, 1)
            device.emit(uinput.KEY_N, 0)
            press_mfsw = 0
        # mode button
        #if msg == '3A1C':
        # right
        #device.emit(uinput.KEY_N, 1)
        #device.emit(uinput.KEY_N, 0)
        #press_mfsw = 0
        # mfsw scroll wheel up
        elif msg in ('3904', '390B'):
            # left
            device.emit(uinput.KEY_1, 1)
            device.emit(uinput.KEY_1, 0)
            press_mfsw = 0
        # mfsw scroll wheel down
        elif msg in ('3905', '390C'):
            # right
            device.emit(uinput.KEY_2, 1)
            device.emit(uinput.KEY_2, 0)
            press_mfsw = 0
        # mfsw scroll wheel pressed
        elif msg == '3908':
            press_mfsw += 1
        #mfsw mode button - couting messages
        elif msg in ('3900', '3A00') and press_mfsw > 0:
            if press_mfsw == 1:
                device.emit(uinput.KEY_ENTER, 1)
                device.emit(uinput.KEY_ENTER, 0)
            elif press_mfsw >= 2:
                device.emit(uinput.KEY_ESC, 1)
                device.emit(uinput.KEY_ESC, 0)
            press_mfsw = 0
        # reset nextbtn and prev because MFSW wheel up/down may also trigger CAN 461 prev/next logic
        elif msg in ('3900', '3A00') and press_mfsw == 0:
            nextbtn = 0
            prev = 0
            press_mfsw = 0


@handle_errors
async def process_canid_602(msg):
    global tv_input_activation_detected, tv_input_task

    if not (msg.startswith('091230') or msg.startswith('811230')):
        return
    if tv_input_activation_detected:
        return
    logger.info("tv input message detected")
    tv_input_activation_detected = True
    if tv_input_task is not None:
        tv_input_task.stop()
        tv_input_task = None


@handle_errors
async def process_canid_623(msg):
    global tmset

    if not read_and_set_time_from_dashboard:
        return
    if tmset is not None:
        return
    if len(msg) < 16:
        logger.error("CAN 623 message too short: %r", msg)
        return
    tmset = False
    fire_and_forget(
        asyncio.get_running_loop(),
        set_time_from_can(msg),
        "set_system_time_from_can"
    )


@handle_errors
async def process_canid_635(msg):
    global light_status, last_msg_635
    if len(msg) < 4:
        return
    first_message = last_msg_635 is None
    if not (first_message or msg != last_msg_635):
        return
    light_value = int(msg[2:4], 16)
    new_light_status = 1 if light_value > 0 else 0
    if first_message or (new_light_status != light_status):
        if change_dark_mode_by_car_light and api_is_connected:
            mode = "night" if new_light_status == 1 else "day"
            logger.info("Light status changed: Setting %s mode immediately.", mode)
            event_handler.send_day_night(mode)
    light_status = new_light_status
    last_msg_635 = msg


@handle_errors
async def process_canid_65F(msg):
    global car_model_set, carmodel, FIS1, FIS2
    global vin_parts, vin_set, vin_display
    global model_info_set, carmodelyear_cache, carmodelfull_cache

    # -------- READ VIN FROM CAN --------
    if len(msg) >= 16 and not vin_set:
        frame_idx = msg[0:2]  # 00 / 01 / 02
        if frame_idx in ('00', '01', '02'):
            # Frame 00: read only the last 3 bytes (skip padding)
            if frame_idx == '00':
                hex_part = msg[10:16]  # bytes 5..7 => 57 41 55
            else:
                hex_part = msg[2:16]  # bytes 1..7
            ascii_part = bytes.fromhex(hex_part).decode(errors='ignore').strip('\x00')
            vin_parts[int(frame_idx, 16)] = ascii_part
        # Assemble VIN only when all fragments are present
        if 0 in vin_parts and 1 in vin_parts and 2 in vin_parts:
            vin = vin_parts[0] + vin_parts[1] + vin_parts[2]
            visible = vin[:10] if len(vin) >= 10 else vin
            masked = "*" * max(0, len(vin) - 10)
            vin_display = visible + masked
            vin_set = True
    # -------- READ MODEL / YEAR FROM FRAME 01 --------
    if len(msg) >= 16 and msg[0:2] == '01' and not model_info_set:
        carmodel = bytes.fromhex(msg[8:12]).decode(errors='ignore').strip('\x00')
        carmodelyear_cache = translate_caryear(
            bytes.fromhex(msg[14:16]).decode(errors='ignore').strip('\x00')
        )
        # Handle US version model number "FM" of the Audi A3 8P as "8P" model
        if carmodel == "FM":
            carmodel = "8P"
        car_models = {
            '8E': ('Audi A4', '265', '267'),
            '8J': ('Audi TT', '667', '66B'),
            '8L': ('Audi A3', '667', '66B'),
            '8P': ('Audi A3', '667', '66B'),
            '42': ('Audi R8', '265', '267'),
        }
        carmodelfull_cache, FIS1, FIS2 = car_models.get(
            carmodel[0:2],
            ('unknown car model', '265', '267')
        )
        model_info_set = True
        await start_send_to_dis()
        if not script_started and not welcome_active:
            fire_and_forget(asyncio.get_running_loop(), welcome_message(), "welcome_message")

    # -------- LOG ONLY WHEN BOTH VIN + MODEL ARE READY --------
    if car_model_set is None and vin_set and model_info_set:
        logger.info("")
        logger.info("The car model, model year and VIN were successfully read from the CAN-Bus.")
        logger.info("CAR = %s %s %s", carmodelfull_cache, carmodel, carmodelyear_cache)
        logger.info("VIN: %s", vin_display)
        logger.info("FIS1 = %s", FIS1)
        logger.info("FIS2 = %s", FIS2)
        logger.info("")

        car_model_set = True


@handle_errors
async def process_canid_661(msg):
    global tv_mode_active, send_on_canbus, deactivate_overwrite_dis_content

    TV_MODE_MSGS = {'8101123700000000', '8301123700000000'}
    is_tv = msg in TV_MODE_MSGS
    # --- TV MODE ACTIVATED ---
    if is_tv and tv_mode_active == 0:
        device.emit(uinput.KEY_X, 1)
        device.emit(uinput.KEY_X, 0)
        if ENABLE_LOGGING:
            logger.info(
                'RNS-E is (back) in TV mode - play media - Keyboard: "X" - Hudiy/OpenAuto: "play"'
            )
        tv_mode_active = 1
        if only_send_if_radio_is_in_tv_mode:
            send_on_canbus = True
            deactivate_overwrite_dis_content = False
        return
    # --- TV MODE DEACTIVATED ---
    if not is_tv and tv_mode_active == 1:
        device.emit(uinput.KEY_C, 1)
        device.emit(uinput.KEY_C, 0)
        if ENABLE_LOGGING:
            logger.info(
                'RNS-E is not in TV mode (anymore) - pause media - Keyboard: "C" - Hudiy/OpenAuto: "pause"'
            )
        tv_mode_active = 0
        if only_send_if_radio_is_in_tv_mode:
            send_on_canbus = False
            deactivate_overwrite_dis_content = True


async def toggle_fis2_label():
    global value_of_toggle_fis2
    value_of_toggle_fis2 = {
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
        12: 'BLANK',
        13: 'DISABLE'
    }.get(toggle_fis2, None)


#asyncio.run(toggle_fis2_label())


@handle_errors
async def set_time_from_can(msg):
    global tmset

    if len(msg) < 16:
        logger.error("CAN 623 message too short: %r", msg)
        return
    command = [
        "sudo",
        "date",
        f"{msg[10:12]}{msg[8:10]}{msg[2:4]}{msg[4:6]}{msg[12:16]}.{msg[6:8]}"
    ]
    logger.info("Setting system date with command: %s", " ".join(command))
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if stdout:
            logger.info("Date command output: %s", stdout.decode().strip())
        if stderr:
            logger.error("Error output: %s", stderr.decode().strip())
        if process.returncode == 0:
            tmset = True
        else:
            tmset = None
            logger.error("Failed to set date with return code: %d", process.returncode)
    except Exception as e:
        tmset = None
        logger.error("Unexpected error while setting date: %s", str(e))


@handle_errors
async def candump():
    """Toggle candump start/stop (robuste, nicht blockierende Variante)."""
    global candump_proc, pause_fis1, pause_fis2

    async with candump_lock:
        # Läuft schon?
        if candump_proc and (candump_proc.returncode is None):
            # --- STOP ---
            proc = candump_proc
            logger.info("Stopping candump now")
            try:
                proc.terminate()  # SIGTERM
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.info("candump did not terminate in time, killing...")
                    proc.kill()
                    await proc.wait()
                rc = proc.returncode
                # 0 (normal), 130 (SIGINT), 143 (SIGTERM) → kein Fehler
                if rc not in (0, 130, 143):
                    logger.error("candump exit code after stop: %s", rc)
                else:
                    logger.info("candump stopped (return code: %s)", rc)
            finally:
                # Nur löschen, wenn global noch auf genau diesen Prozess zeigt
                if candump_proc is proc:
                    candump_proc = None
            # HUDIY/FIS Feedback
            if send_on_canbus and can_functional:
                pause_fis1 = True
                pause_fis2 = True
                set_fis1("CANDUMP", "center")
                set_fis2("STOP", "center")
                await asyncio.sleep(2)
                clear_content(FIS1)
                clear_content(FIS2)
                pause_fis1 = False
                pause_fis2 = False
                # Aktuelle Anzeige sofort wiederherstellen
                await refresh_fis1_current_value()
                await refresh_fis2_current_value()
            return

        # --- START ---
        logger.info("Starting candump now")
        now = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        log_dir = f"{logs_root}/candumps"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f"{log_dir}/{now}-candump-{can_interface}.txt"

        command = ['candump', can_interface, '-tA']
        try:
            # Logfile öffnen und Handle behalten solange der Prozess lebt
            log_file = open(log_file_path, 'w')
            proc = await asyncio.create_subprocess_exec(
                *command, stdout=log_file, stderr=log_file
            )
            candump_proc = proc
            logger.info("candump started (pid=%s), writing to %s", proc.pid, log_file_path)
            # Watcher-Task: wartet im Hintergrund auf Prozessende und räumt auf

            async def _watch():
                global candump_proc
                nonlocal log_file
                rc = await proc.wait()
                try:
                    log_file.flush()
                finally:
                    log_file.close()
                # 0/130/143 → normal; sonst Fehler
                if rc in (0, 130, 143):
                    logger.info("candump exited (return code: %s)", rc)
                else:
                    logger.error("candump exited with error (return code: %s)", rc)
                # Prozess ist vorbei → Handle nur löschen, wenn es noch derselbe ist
                if candump_proc is proc:
                    candump_proc = None
            # Task feuern (nicht awaiten!)
            fire_and_forget(asyncio.get_running_loop(), _watch(), "candump_watch")
        except Exception as e:
            logger.error("Unexpected error while starting candump: %s", str(e), exc_info=True)
            # Aufräumen
            try:
                if 'log_file' in locals() and not log_file.closed:
                    log_file.close()
            except Exception:
                pass
            try:
                if 'proc' in locals() and proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            except Exception:
                pass
            if candump_proc is locals().get('proc'):
                candump_proc = None
            return

        # HUDIY/FIS Feedback
        if send_on_canbus and can_functional:
            pause_fis1 = True
            pause_fis2 = True
            set_fis1("CANDUMP", "center")
            set_fis2("START", "center")
            await asyncio.sleep(3)
            clear_content(FIS1)
            clear_content(FIS2)
            pause_fis1 = False
            pause_fis2 = False
            # Aktuelle Anzeige sofort wiederherstellen
            await refresh_fis1_current_value()
            await refresh_fis2_current_value()


def _reset_speed_measure_state(send_hudiy_disarm: bool = False):
    global start_time, measure_done, elapsed_time, last_data
    global speed_measure_to_api, speed_measure_armed, last_speed_measure_send_time
    global hudiy_speed_measure_animation_started

    start_time = None
    measure_done = 0
    elapsed_time = 0.0
    last_data = None
    speed_measure_to_api = 0.00
    speed_measure_armed = False
    hudiy_speed_measure_animation_started = False
    last_speed_measure_send_time = 0.0

    if send_hudiy_disarm:
        _push_hudiy_speed_measure_control(-1.0)


def _push_hudiy_speed_measure_control(value: float) -> None:
    if (
        send_to_api_gauges
        and api_is_connected
        and backend == "Hudiy"
        and event_handler is not None
    ):
        event_handler.update_to_api("getPidValue(7)", value)


def _push_openauto_speed_measure_value(value: float) -> None:
    if (
        send_to_api_gauges
        and api_is_connected
        and backend == "OpenAuto"
        and event_handler is not None
    ):
        event_handler.update_to_api("getPidValue(7)", value)


@handle_errors
async def refresh_fis1_current_value():
    global last_sent_speed, last_sent_rpm, last_sent_coolant, last_sent_outside_temp
    global last_speed_send_time, last_rpm_send_time, last_coolant_send_time, last_outside_temp_send_time
    global last_speed, last_rpm, last_coolant, last_outside_temp
    global start_time, measure_done, elapsed_time, last_data, speed_measure_to_api, speed_measure_armed, last_speed_measure_send_time
    global hudiy_speed_measure_animation_started
    global pending_speed_display, pending_speed_api
    global pending_rpm_display, pending_rpm_api
    global pending_coolant_display, pending_coolant_api
    global pending_outside_temp_display, pending_outside_temp_api

    if not (send_on_canbus and can_functional):
        return

    if toggle_fis1 in (1, 2, 3, 4, 5):
        if not show_label and not pause_fis1:
            await media_to_dis1()
        return

    if toggle_fis1 == 6:
        last_sent_speed = None
        last_speed_send_time = 0.0
        last_speed = None
        pending_speed_display = None
        pending_speed_api = None
        return

    if toggle_fis1 == 7:
        last_sent_rpm = None
        last_rpm_send_time = 0.0
        last_rpm = None
        pending_rpm_display = None
        pending_rpm_api = None
        return

    if toggle_fis1 == 8:
        last_sent_coolant = None
        last_coolant_send_time = 0.0
        last_coolant = None
        pending_coolant_display = None
        pending_coolant_api = None
        return

    if toggle_fis1 == 9:
        if not show_label:
            try:
                cpu_percent = psutil.cpu_percent()
                cpu_load_local = min(round(cpu_percent if cpu_percent is not None else 0), 99)
            except Exception:
                cpu_load_local = 0

            temps_all = psutil.sensors_temperatures() or {}
            temps = temps_all.get("cpu_thermal", [])
            if temps:
                cpu_temp_local = int(round(temps[0].current))
                unit = temp_unit if temp_unit else "°C"
                if unit == '°F':
                    cpu_temp_local = round(cpu_temp_local * 1.8 + 32)
                data_now = f'{cpu_load_local:02d}% {cpu_temp_local:02d}{unit}'
                set_fis1(data_now, "center")
        return

    if toggle_fis1 == 10:
        # Re-selecting speed_measure must not start a measurement while already driving.
        # The next valid run is armed by process_canid_351 only after speed <= lower_speed.
        start_time = None
        measure_done = 0
        elapsed_time = 0.0
        speed_measure_to_api = 0.00
        speed_measure_armed = False
        hudiy_speed_measure_animation_started = False
        last_data = None
        last_speed_measure_send_time = 0.0
        return

    if toggle_fis1 == 11:
        if toggle_fis2 != 10:
            start_time = None
            measure_done = 0
            elapsed_time = 0.0
            last_data = None
            last_speed_measure_send_time = 0.0
            speed_measure_to_api = 0.00

        last_sent_outside_temp = None
        last_outside_temp_send_time = 0.0
        last_outside_temp = None
        pending_outside_temp_display = None
        pending_outside_temp_api = None
        return

    if toggle_fis1 == 12:  # BLANK
        clear_content(FIS1)
        return

    if toggle_fis1 == 13:  # DISABLE
        clear_content(FIS1)
        return


@handle_errors
async def refresh_fis2_current_value():
    global last_sent_speed, last_sent_rpm, last_sent_coolant, last_sent_outside_temp
    global last_speed_send_time, last_rpm_send_time, last_coolant_send_time, last_outside_temp_send_time
    global last_speed, last_rpm, last_coolant, last_outside_temp
    global start_time, measure_done, elapsed_time, last_data, speed_measure_to_api, speed_measure_armed, last_speed_measure_send_time
    global hudiy_speed_measure_animation_started
    global pending_speed_display, pending_speed_api
    global pending_rpm_display, pending_rpm_api
    global pending_coolant_display, pending_coolant_api
    global pending_outside_temp_display, pending_outside_temp_api

    if not (send_on_canbus and can_functional):
        return

    if toggle_fis2 in (1, 2, 3, 4, 5):
        if not pause_fis2:
            await media_to_dis2()
        return

    if toggle_fis2 == 6:
        last_sent_speed = None
        last_speed_send_time = 0.0
        last_speed = None
        pending_speed_display = None
        pending_speed_api = None
        return

    if toggle_fis2 == 7:
        last_sent_rpm = None
        last_rpm_send_time = 0.0
        last_rpm = None
        pending_rpm_display = None
        pending_rpm_api = None
        return

    if toggle_fis2 == 8:
        last_sent_coolant = None
        last_coolant_send_time = 0.0
        last_coolant = None
        pending_coolant_display = None
        pending_coolant_api = None
        return

    if toggle_fis2 == 9:
        try:
            cpu_percent = psutil.cpu_percent()
            cpu_load_local = min(round(cpu_percent if cpu_percent is not None else 0), 99)
        except Exception:
            cpu_load_local = 0

        temps_all = psutil.sensors_temperatures() or {}
        temps = temps_all.get("cpu_thermal", [])
        if temps:
            cpu_temp_local = int(round(temps[0].current))
            unit = temp_unit if temp_unit else "°C"
            if unit == '°F':
                cpu_temp_local = round(cpu_temp_local * 1.8 + 32)
            data_now = f'{cpu_load_local:02d}% {cpu_temp_local:02d}{unit}'
            set_fis2(data_now, "center")
        return

    if toggle_fis2 == 10:
        # Re-selecting speed_measure must not start a measurement while already driving.
        # The next valid run is armed by process_canid_351 only after speed <= lower_speed.
        start_time = None
        measure_done = 0
        elapsed_time = 0.0
        speed_measure_to_api = 0.00
        speed_measure_armed = False
        hudiy_speed_measure_animation_started = False
        last_data = None
        last_speed_measure_send_time = 0.0
        return

    if toggle_fis2 == 11:
        if toggle_fis1 != 10:
            start_time = None
            measure_done = 0
            elapsed_time = 0.0
            last_data = None
            last_speed_measure_send_time = 0.0
            speed_measure_to_api = 0.00

        last_sent_outside_temp = None
        last_outside_temp_send_time = 0.0
        last_outside_temp = None
        pending_outside_temp_display = None
        pending_outside_temp_api = None
        return

    if toggle_fis2 == 12:  # BLANK
        clear_content(FIS2)
        return

    if toggle_fis2 == 13:  # DISABLE
        clear_content(FIS2)
        return


@handle_errors
async def block_show_value1():
    global toggle_fis1, pause_fis1, begin1, end1, pause_fis2
    global overwrite_dis_hold_visible_until

    old_toggle_fis1 = toggle_fis1
    was_disabled = deactivate_overwrite_dis_content or not send_on_canbus or old_toggle_fis1 == 13 or toggle_fis2 == 13

    if toggle_fis1 >= 13:
        toggle_fis1 = 1
    else:
        toggle_fis1 += 1

    if toggle_fis1 == 10:
        # Reset immediately when selected, before the 2-second label display.
        # Otherwise stale speed_measure_armed=True can start a run while already driving.
        _reset_speed_measure_state()

    elif toggle_fis1 == 13 and toggle_fis2 == 10:
        # DISABLE is global: stop an active speed measurement on the other line
        # immediately and disarm the Hudiy speed-measure channel once.
        _reset_speed_measure_state(send_hudiy_disarm=True)

    elif old_toggle_fis1 == 10 and toggle_fis1 != 10 != toggle_fis2:
        _reset_speed_measure_state(send_hudiy_disarm=True)

    data = TOGGLE_MAP[toggle_fis1]
    if callable(data):
        data = data()
    logger.info(f"toggle_fis1 changed to {toggle_fis1} ({data})")

    trace = make_trace("block_show_value1", inspect.currentframe().f_lineno + 1) if ENABLE_LOGGING else None

    # 13 -> normal: clear hidden FIS memory BEFORE 665#0300 can reveal old DISABLE.
    if was_disabled and toggle_fis1 != 13 and toggle_fis2 != 13 and _overwrite_dis_allowed():
        await _prepare_fis_reactivation_after_disable(trace=trace)

    # normal -> 13: keep 665#0300 active while the DISABLE label is visible.
    if toggle_fis1 == 13:
        overwrite_dis_hold_visible_until = time.monotonic() + 2.4

    # Kurze Umschaltanzeige im FIS1
    if data and send_on_canbus and can_functional:
        set_fis1(data, "center", trace=trace)
        begin1 = -1
        end1 = 7
        await asyncio.sleep(2)

        if toggle_fis1 == 13:
            pause_fis2 = True
            await _finish_fis_disable_after_label(trace=trace)
            pause_fis2 = False
        else:
            clear_content(FIS1, trace=trace)

    pause_fis1 = False

    # Aktuellen Inhalt passend zum Toggle sofort wiederherstellen.
    # After reactivation refresh both lines because 665#0100 hid both lines.
    if toggle_fis1 != 13:
        await refresh_fis1_current_value()
        if was_disabled:
            await refresh_fis2_current_value()

    update_toggle_features("toggle_fis1", toggle_fis1)


@handle_errors
async def block_show_value2():
    global toggle_fis2, pause_fis2, begin2, end2, pause_fis1
    global overwrite_dis_hold_visible_until
    global value_of_toggle_fis2, last_value_of_toggle_fis2

    old_toggle_fis2 = toggle_fis2

    was_disabled = (
        deactivate_overwrite_dis_content
        or not send_on_canbus
        or old_toggle_fis2 == 13
        or toggle_fis1 == 13
    )

    if toggle_fis2 >= 13:
        toggle_fis2 = 1
    else:
        toggle_fis2 += 1

    if toggle_fis2 == 10:
        # Reset immediately when selected, before the 2-second label display.
        # Otherwise stale speed_measure_armed=True can start a run while already driving.
        _reset_speed_measure_state()

    elif toggle_fis2 == 13 and toggle_fis1 == 10:
        # DISABLE is global: stop an active speed measurement on the other line
        # immediately and disarm the Hudiy speed-measure channel once.
        _reset_speed_measure_state(send_hudiy_disarm=True)

    elif old_toggle_fis2 == 10 and toggle_fis1 != 10 != toggle_fis2:
        _reset_speed_measure_state(send_hudiy_disarm=True)

    data = TOGGLE_MAP[toggle_fis2]
    if callable(data):
        data = data()

    logger.info(
        "toggle_fis2 changed to %s (%s)",
        toggle_fis2,
        data
    )

    trace = (
        make_trace(
            "block_show_value2",
            inspect.currentframe().f_lineno + 1
        )
        if ENABLE_LOGGING
        else None
    )

    # 13 -> normal:
    # Clear hidden FIS memory before 665#0300 can reveal old content.
    if (
        was_disabled
        and toggle_fis1 != 13
        and toggle_fis2 != 13
        and _overwrite_dis_allowed()
    ):
        await _prepare_fis_reactivation_after_disable(trace=trace)

    # normal -> 13:
    # Keep 665#0300 active while the DISABLE label is visible.
    if toggle_fis2 == 13:
        overwrite_dis_hold_visible_until = time.monotonic() + 2.4

    if data and send_on_canbus and can_functional:
        if not show_label:
            # Ohne permanente Labels wird der gewählte Toggle kurz in FIS2 angezeigt.
            set_fis2(
                data,
                "center",
                trace=trace
            )

            begin2 = -1
            end2 = 7

            await asyncio.sleep(2)

            if toggle_fis2 == 13:
                pause_fis1 = True
                try:
                    await _finish_fis_disable_after_label(trace=trace)
                finally:
                    pause_fis1 = False
            else:
                clear_content(
                    FIS2,
                    trace=trace
                )

        else:
            # Permanentes Label für den aktuellen FIS2-Wert berechnen.
            await toggle_fis2_label()

            # Das Label gehört in die obere FIS-Zeile.
            # Dies wird jetzt für alle Toggle-Werte 1–13 direkt ausgeführt,
            # nicht nur indirekt über die Media-Funktionen für 1–5.
            if value_of_toggle_fis2:
                set_fis1(
                    value_of_toggle_fis2,
                    "center",
                    trace=trace
                )
                last_value_of_toggle_fis2 = value_of_toggle_fis2

            if toggle_fis2 == 13:
                await asyncio.sleep(2)

                pause_fis1 = True
                try:
                    await _finish_fis_disable_after_label(trace=trace)
                finally:
                    pause_fis1 = False

    pause_fis2 = False

    # Aktuellen Inhalt passend zum Toggle sofort wiederherstellen.
    # Nach einer Reaktivierung beide Zeilen aktualisieren,
    # weil 665#0100 zuvor beide Zeilen ausgeblendet hatte.
    if toggle_fis2 != 13:
        await refresh_fis2_current_value()

        if was_disabled:
            await refresh_fis1_current_value()

    update_toggle_features(
        "toggle_fis2",
        toggle_fis2
    )


def update_toggle_features(toggle_key: str, value: int) -> None:
    """Update only toggle_fis1 or toggle_fis2 in the existing features.conf.

    All comments, formatting and unrelated/unknown settings remain untouched.
    If the key is missing, it is appended once.
    """
    if toggle_key not in {"toggle_fis1", "toggle_fis2"}:
        raise ValueError("Unsupported toggle key: {}".format(toggle_key))

    config_path = Path(FEATURE_FILE)
    if not config_path.exists():
        logger.warning("Could not update %s: %s does not exist.", toggle_key, config_path)
        return

    try:
        content = config_path.read_text(encoding="utf-8")
        pattern = r"(^\s*{}\s*=\s*)[^#\r\n]*(\s*(?:#.*)?$)".format(re.escape(toggle_key))
        replacement = r"\g<1>{}\g<2>".format(int(value))
        new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)

        if count == 0:
            separator = "" if not content or content.endswith("\n") else "\n"
            new_content = content + separator + "{} = {}\n".format(toggle_key, int(value))

        if new_content != content:
            # Atomic replacement without keeping a backup file. A power loss cannot
            # leave the existing features.conf partially written.
            temp_path = config_path.with_name(config_path.name + ".tmp")
            try:
                temp_path.write_text(new_content, encoding="utf-8")
                os.replace(str(temp_path), str(config_path))
            finally:
                with contextlib.suppress(OSError):
                    if temp_path.exists():
                        temp_path.unlink()
            logger.info("Updated %s to %s in %s", toggle_key, value, config_path)
    except Exception:
        logger.exception("Could not update %s in %s", toggle_key, config_path)
        raise


@handle_errors
async def media_to_dis1():
    global title, artist, album, position, duration, scrolling_active_fis1

    rule1 = {
        1: title,
        2: artist,
        3: album,
        4: position,
        5: duration
    }.get(toggle_fis1, "")

    try:
        scrolling_active_fis1 = False
        await start_scrolling(rule1, "FIS1")
    except Exception as e:
        logger.error(f"Error in media_to_dis1: {e}")


@handle_errors
async def media_to_dis2():
    global title, artist, album, position, duration
    global scrolling_active_fis2, last_value_of_toggle_fis2, value_of_toggle_fis2

    rule2 = {
        1: title,
        2: artist,
        3: album,
        4: position,
        5: duration
    }.get(toggle_fis2, "")

    try:
        if show_label:
            await toggle_fis2_label()

            if value_of_toggle_fis2 != last_value_of_toggle_fis2:
                if send_on_canbus and can_functional and not pause_fis1:
                    set_fis1(value_of_toggle_fis2, "center")
                last_value_of_toggle_fis2 = value_of_toggle_fis2

        scrolling_active_fis2 = False
        await start_scrolling(rule2, "FIS2")

    except Exception as e:
        logger.error(f"Error in media_to_dis2: {e}")


@handle_errors
async def start_scrolling(rule, display_type):
    global scroll_task_fis1, scroll_task_fis2, max_length, scroll_type

    max_length = 8
    wait_time = 3
    delay = 0.25
    loop = asyncio.get_running_loop()

    try:
        rule = "" if rule is None else str(rule)

        if display_type == "FIS1":
            if scroll_task_fis1 is not None and not scroll_task_fis1.done():
                scroll_task_fis1.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await scroll_task_fis1
                scroll_task_fis1 = None

            # kurze Texte direkt setzen, nicht scrollen
            if len(rule) <= max_length:
                if not show_label and not pause_fis1:
                    set_fis1(rule, "center")
                return

            if scroll_type == "scroll":
                scroll_task_fis1 = fire_and_forget(
                    loop,
                    _scroll_text(rule, wait_time, delay, FIS1),
                    "scroll_text_fis1"
                )
            elif scroll_type == "oem_style":
                scroll_task_fis1 = fire_and_forget(
                    loop,
                    _scroll_oem_style(rule, wait_time, FIS1),
                    "scroll_oem_fis1"
                )
            return

        elif display_type == "FIS2":
            if scroll_task_fis2 is not None and not scroll_task_fis2.done():
                scroll_task_fis2.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await scroll_task_fis2
                scroll_task_fis2 = None

            # kurze Texte direkt setzen, nicht scrollen
            if len(rule) <= max_length:
                if not pause_fis2:
                    set_fis2(rule, "center")
                return

            if scroll_type == "scroll":
                scroll_task_fis2 = fire_and_forget(
                    loop,
                    _scroll_text(rule, wait_time, delay, FIS2),
                    "scroll_text_fis2"
                )
            elif scroll_type == "oem_style":
                scroll_task_fis2 = fire_and_forget(
                    loop,
                    _scroll_oem_style(rule, wait_time, FIS2),
                    "scroll_oem_style_fis2"
                )
            return

    except asyncio.CancelledError:
        pass


async def _scroll_text(rule1, wait_time, delay, display):
    text_length = len(rule1)
    text_with_padding = ' ' * max_length + rule1 + ' ' * max_length
    reset_scroll = True
    current_rule = None
    last_data = None

    while True:
        if (display == FIS1 and toggle_fis1 not in (1, 2, 3, 4, 5)) or (
                display == FIS2 and toggle_fis2 not in (1, 2, 3, 4, 5)):
            (clear_content(FIS1) if display == FIS1 else clear_content(FIS2))
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
                set_fis(display, text_with_padding[start_index:start_index + max_length], "center")
                await asyncio.sleep(wait_time)
                reset_scroll = False
            i = start_index
            while i < end_index:
                if (display == FIS1 and toggle_fis1 not in (1, 2, 3, 4, 5)) or (
                        display == FIS2 and toggle_fis2 not in (1, 2, 3, 4, 5)):
                    (clear_content(FIS1) if display == FIS1 else clear_content(FIS2))
                    return
                while (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    reset_scroll = True
                    await asyncio.sleep(0.5)
                if not reset_scroll:
                    data = text_with_padding[i:i + max_length]
                    if data != last_data:
                        set_fis(display, data, "center")
                        last_data = data
                    await asyncio.sleep(delay)
                    i += 1
            reset_scroll = True
        else:
            if rule1 != last_data:
                if (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    await asyncio.sleep(0.5)
                    continue
                set_fis(display, rule1, "center")
                await asyncio.sleep(wait_time)
            else:
                set_fis(display, rule1, "center")
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
            (clear_content(FIS1) if display == FIS1 else clear_content(FIS2))
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
                set_fis(display, segment, "center")
                await asyncio.sleep(wait_time)
                reset_scroll = False
            if segment != (last_data_fis1 if display == FIS1 else last_data_fis2):
                if (display == FIS1 and pause_fis1) or (display == FIS2 and pause_fis2):
                    await asyncio.sleep(0.5)
                    continue
                set_fis(display, segment, "center")
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
                set_fis(display, rule1, "center")
                if display == FIS1:
                    last_data_fis1 = rule1
                else:
                    last_data_fis2 = rule1
                await asyncio.sleep(wait_time)
            else:
                set_fis(display, rule1, "center")
            if display == FIS1:
                last_data_fis1 = rule1
            else:
                last_data_fis2 = rule1
            await asyncio.sleep(0.95)


@handle_errors
async def send_to_dis(display):
    global script_started, speed_measure_to_api, elapsed_time_formatted, show_label
    global toggle_fis1, toggle_fis2, pause_fis1, pause_fis2

    sleep_values, start_time, measure_done, data, last_data, drop = 0.5, None, 0, '', '', 0

    if ENABLE_LOGGING:
        logger.info(f"Task send_to_dis ({display}) was started.")

    while not stop_flag:
        fis_mapping = {
            FIS1: (FIS1, toggle_fis1, pause_fis1),
            FIS2: (FIS2, toggle_fis2, pause_fis2),
        }

        mapping = fis_mapping.get(display)
        if mapping is None:
            logger.warning("Unknown display passed to send_to_dis: %r", display)
            return

        FIS, toggle_fis, pause_fis = mapping

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
                set_fis(FIS, data, "right")

                await asyncio.sleep(sleep_values)
                continue
            else:
                await asyncio.sleep(sleep_values)

            # Speed Measure (toggle 10) is handled exclusively in process_canid_351().
            # This generic loop must not send stale FIS/API values for it.
            if toggle_fis == 10:
                await asyncio.sleep(0.1)
                continue

            if toggle_fis == 12:
                data, sleep_values = '', 2.0
            if data != last_data:
                set_fis(FIS, data, "right")
            last_data = data
            await asyncio.sleep(sleep_values)

        except asyncio.CancelledError:
            if ENABLE_LOGGING:
                logger.info(f"Task send_to_dis ({display}) was stopped.")
            break


def parse_media_position_label(label):
    try:
        if not label:
            return None

        parts = str(label).strip().split(":")
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
        return None
    except Exception:
        return None


def format_media_position_label(total_seconds):
    total_seconds = max(0, int(total_seconds))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"


# using a translation table from the VIN decoding list:
# https://www.nininet.de/deutsch/Fahrgestellnummer-entschluesseln.php
def translate_caryear(carmodelyear):
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
    return translation_table.get(carmodelyear)


@handle_errors
async def inject_65f_for_desk_setup():
    if not AUTOSEND_CAR_MODEL:
        return
    if model_info_set:
        return

    logger.info(
        "Desk setup active on %s -> injecting default 65F model data internally.",
        can_interface
    )

    await process_canid_65F("0035C837E2574155")
    await process_canid_65F("015A5A5A38453032")
    await process_canid_65F("0241313238383831")


@handle_errors
async def cancel_background_tasks():
    global background_tasks

    current = asyncio.current_task()

    running = [
        t for t in background_tasks
        if t is not None
        and not t.done()
        and t is not current
    ]

    if not running:
        background_tasks = [
            t for t in background_tasks
            if t is not None
            and not t.done()
            and t is not current
        ]
        return

    logger.info("Cancelling %d background task(s)...", len(running))

    for task in running:
        try:
            logger.info("Cancelling background task: %s", task_name_compat(task))
            task.cancel()
        except Exception:
            logger.warning("Could not cancel background task", exc_info=True)

    results = await asyncio.gather(*running, return_exceptions=True)

    for task, result in zip(running, results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            logger.warning(
                "Background task '%s' ended with exception during shutdown: %s",
                task_name_compat(task),
                result,
                exc_info=True
            )

    background_tasks = [
        t for t in background_tasks
        if t is not None
        and not t.done()
        and t is not current
    ]


async def stop_script():
    global stop_script_running, api_is_connected, stop_flag, event_handler, server_socket, remote_task, remote_server_shutdown_event, tasks, media_position_task
    global task_overwrite_dis_content, tv_input_task, notifier, notifier_started, bus

    if stop_script_running:
        logger.info("stop_script already running - waiting for completion.")
        await async_to_thread(stop_completed_event.wait)
        return

    stop_completed_event.clear()
    stop_script_running = True
    current = asyncio.current_task()

    # If stop_script itself was started via fire_and_forget(), remove it from
    # background_tasks so it can never cancel/wait on itself during shutdown.
    if current in background_tasks:
        with contextlib.suppress(ValueError):
            background_tasks.remove(current)

    def _task_label(t: "asyncio.Task"):
        if hasattr(t, "get_name"):
            try:
                return task_name_compat(t)
            except Exception:
                pass
        coro = getattr(t, "_coro", None)
        name = getattr(coro, "__name__", None)
        return name or "task"

    try:
        logger.info("Stopping script...")

        # Pause media before shutting down tasks/API, while uinput and HUDIY/OpenAuto
        # are still available.
        try:
            if "device" in globals() and "uinput" in globals():
                if ENABLE_LOGGING:
                    logger.info("Sending media pause keypress: C")
                device.emit(uinput.KEY_C, 1)
                device.emit(uinput.KEY_C, 0)
                await asyncio.sleep(0.2)
            else:
                logger.warning("uinput device not initialized; skipping media pause keypress.")
        except Exception:
            logger.warning("Could not send media pause keypress.", exc_info=True)

        try:
            if cam is not None:
                logger.info("Stopping local reverse camera during script shutdown.")
                await async_to_thread(cam.stop)
        except Exception:
            logger.warning("Could not stop local reverse camera during script shutdown.", exc_info=True)

        # Stop python-can periodic senders before shutting down the notifier/bus.
        for task_name in ("task_overwrite_dis_content", "tv_input_task"):
            periodic_task = globals().get(task_name)
            if periodic_task is not None:
                try:
                    periodic_task.stop()
                    logger.info("Stopped periodic CAN task: %s", task_name)
                except Exception:
                    logger.warning("Could not stop periodic CAN task: %s", task_name, exc_info=True)
                globals()[task_name] = None

        if ENABLE_LOGGING:
            for task in asyncio.all_tasks():
                logger.info(f"Found running task: {_task_label(task)}")

        if ENABLE_LOGGING:
            logger.info("setting stop_flag and wait 2 seconds to gently close running tasks")
        stop_flag = True
        await asyncio.sleep(2)

        if remote_task:
            if remote_task.done():
                if ENABLE_LOGGING:
                    logger.info("ℹ️ remote_control task was already stopped.")
            else:
                logger.info("🚦 Triggering shutdown of remote_control")
                if remote_server_shutdown_event:
                    remote_server_shutdown_event.set()
                try:
                    await asyncio.wait_for(remote_task, timeout=5)
                    logger.info("✅ remote_control_task stopped cleanly.")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ remote_control_task did not stop in time.")
                except asyncio.CancelledError:
                    logger.info("ℹ️ remote_control_task was already cancelled.")
                except Exception as e:
                    logger.error(f"❌ remote_control_task stop error: {e}", exc_info=True)

        if media_position_task is not None and not media_position_task.done():
            media_position_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await media_position_task
            media_position_task = None

        tracked_tasks = [task for task in list(tasks) if task is not current and not task.done()]

        if tracked_tasks:
            for task in tracked_tasks:
                if task is remote_task:
                    continue
                task.cancel()

            for task in tracked_tasks:
                if task is remote_task:
                    continue
                lbl = _task_label(task)
                try:
                    await task
                    logger.info(f"✅ Task {lbl} was stopped.")
                except asyncio.CancelledError:
                    logger.info(f"✋ Task {lbl} was cancelled.")
                except Exception as e:
                    logger.error(f"❌ Error while waiting for task {lbl}: {e}", exc_info=True)
        else:
            logger.info("ℹ️ No tracked tasks were running or all already stopped.")

        # --- background (fire_and_forget) tasks stoppen ---
        await cancel_background_tasks()
        tasks = [t for t in tasks if not t.done()]

        if api_is_connected:
            try:
                await async_to_thread(client.disconnect)
                api_is_connected = False
                logger.info(f"Successfully disconnected from {backend} API.")
            except Exception:
                logger.error(f"Error while disconnecting from {backend} API.", exc_info=True)

        active_notifier = notifier
        if active_notifier is not None:
            try:
                cast(StoppableTask, active_notifier).stop()
                logger.info("CAN notifier stopped during shutdown.")
            except Exception:
                logger.warning("Could not stop CAN notifier during shutdown.", exc_info=True)
            notifier = None

        active_bus = bus
        if active_bus is not None:
            try:
                cast(Shutdownable, active_bus).shutdown()
                logger.info("CAN bus shut down cleanly.")
            except Exception:
                logger.warning("Could not shut down CAN bus cleanly.", exc_info=True)
            bus = None

    except asyncio.CancelledError:
        logger.info("Task stop_script was cancelled.")
        raise
    except Exception:
        logger.exception("Unexpected error during stop_script")
        raise
    finally:
        notifier_started = False
        stop_completed_event.set()


@handle_errors
async def kill_script():
    current_pid = os.getpid()
    logger.info(f"Killing current script instance with PID: {current_pid}")
    command = f'sudo kill -15 {current_pid}'
    result = await run_command(command)
    if result["stderr"]:
        logger.error("Failed to kill the script: %s", result["stderr"])
    else:
        logger.info("Script killed successfully.")


@handle_errors
async def can_reader_loop():
    global can_reader

    reader = can_reader
    if reader is None:
        logger.warning("CAN reader not initialized.")
        return
    try:
        while not stop_flag:
            msg = await reader.get_message()
            if msg is None:
                continue
            try:
                await read_on_canbus(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to process CAN message ID=0x%03X data=%s; continuing with the next message.",
                    msg.arbitration_id,
                    bytes(msg.data).hex().upper(),
                )
    except asyncio.CancelledError:
        if ENABLE_LOGGING:
            logger.info("Task can_reader_loop was stopped.")
        raise


@handle_errors
async def get_can_messages():
    global bus, notifier, notifier_started, can_reader, can_reader_task

    if notifier_started:
        logger.warning("CAN-Notifier is already running.")
        return

    notifier_started = True
    notifier = None
    can_reader = None
    can_reader_task = None

    try:
        loop = asyncio.get_running_loop()
        # python-can async listener
        can_reader = can.AsyncBufferedReader()
        # Notifier verteilt nur noch an den Reader
        notifier = can.Notifier(bus, [can_reader], loop=loop)
        # Ein einziger Worker verarbeitet alle Nachrichten nacheinander
        can_reader_task = track_task(can_reader_loop(), "can_reader_loop")
        if ENABLE_LOGGING:
            logger.info("CAN-Notifier started.")
            logger.info("CAN AsyncBufferedReader started.")
            logger.info("")
        while not stop_flag:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        if ENABLE_LOGGING:
            logger.info("Task get_can_messages was stopped.")
        raise
    finally:
        if can_reader_task and not can_reader_task.done():
            can_reader_task.cancel()
            try:
                await can_reader_task
            except asyncio.CancelledError:
                pass
        active_notifier = notifier
        if active_notifier is not None:
            cast(StoppableTask, active_notifier).stop()
            if ENABLE_LOGGING:
                logger.info("CAN-Notifier stopped.")
        can_reader = None
        can_reader_task = None
        notifier = None
        notifier_started = False


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
    global stop_flag, script_started, version, PackageNotFoundError, remote_task, features, pause_fis1, pause_fis2, logger, cpu_task

    try:
        await init_async_primitives()
        await write_header_to_log_file(log_filename)
        init_fis_log_file()
        await toggle_fis2_label()
        #console output "Script is starting..."
        await start_script()
        #check if the script is already running. If so, kill the old istance(s)
        await is_script_running(script_filename)
        #check importlib.metadata import with fallback
        version, PackageNotFoundError = await ensure_importlib(logger)
        #check for missing python3 (pip) packages. If missing, install them.
        await packaging_globals(logger=logger, enable_logging=ENABLE_LOGGING)
        await modules_inst_import()
        # can-utils is a native SocketCAN package and independent of Python.
        if not await check_can_utils():
            raise RuntimeError("can-utils could not be installed or verified")
        #check the uinput permissions, so that simulated keyboards controlls can work (Left, Right, Enter...)
        await uinput_permissions()
        #check the cpu powerplan and set to "ondemand" if it's in powersave mode
        await set_powerplan()
        if cpu_task is None or cpu_task.done():
            cpu_task = track_task(read_cpu_loop(), "read_cpu_loop")
        #test the can-interface and see if can-messages are getting received
        await test_can_interface()

        # Receive-only mode has no welcome-message phase. Release CAN callbacks
        # immediately instead of waiting for model detection / a FIS transmission.
        if can_functional and not send_on_canbus:
            script_started = True
            pause_fis1 = False
            pause_fis2 = False
            if ENABLE_LOGGING:
                logger.info(
                    "Receive-only CAN mode active: script_started=True; "
                    "CAN callbacks are enabled without a welcome message."
                )

        if bus and can_functional:
            track_task(get_can_messages(), "get_can_messages")

            if send_on_canbus:
                if send_values_to_dashboard or send_api_mediadata_to_dashboard:
                    track_task(overwrite_dis(), "overwrite_dis")

                if activate_rnse_tv_input:
                    send_tv_input()

                track_task(fis1_sender(), "fis1_sender")
                track_task(fis2_sender(), "fis2_sender")

            await inject_65f_for_desk_setup()
        #check if OpenAuto Pro API files are already downloaded. If not, download, extract and import.
        await check_import_api()
        #check if the units (km/h / rpm and °C / °F) are set correctly in openauto_obd_gauges.ini for OAP Dashboard view.
        if backend == "OpenAuto":
            await oap_units_check(temp_unit, speed_unit, lower_speed, upper_speed)

        if reversecamera_by_reversegear or reversecamera_by_prev_longpress:
            cam_init(reversecamera_guidelines)  # backend-neutraler lokaler Kamera-Warmstart
        #start the remote control task to shutdown other running scripts on startup or via network controll website
        remote_task = track_task(remote_control(), "remote_control")
        # conditional tasks
        #start api connection if api features are enabled
        if send_to_api_gauges or (
                (send_api_mediadata_to_dashboard or change_dark_mode_by_car_light) and bus is not None
        ):
            track_task(api_connection(), "receive_messages")
        if send_to_api_gauges and backend == "Hudiy":
            # Use your chosen base
            base_dir = Path(path)
            ensure_hudiy_js_tree(
                base_dir,
                logger=logger,
                ENABLE_LOGGING=ENABLE_LOGGING,
                create_data_dir=False
            )
        # Supervise all long-running tasks. The task list can grow later (for
        # example when the vehicle model starts the FIS periodic senders), so it
        # is re-snapshotted after every wake-up. Any unexpected task termination
        # is treated as a service failure and triggers the central shutdown.
        while not stop_flag:
            monitored = [
                task for task in list(tasks)
                if task is not None and not task.done()
            ]
            if not monitored:
                raise RuntimeError("No long-running tasks remain active")

            done, _pending = await asyncio.wait(
                monitored,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue

            for task in done:
                if stop_flag:
                    break
                if task.cancelled():
                    raise RuntimeError(
                        "Critical task '{}' was cancelled unexpectedly".format(
                            task_name_compat(task)
                        )
                    )
                exception = task.exception()
                if exception is not None:
                    raise RuntimeError(
                        "Critical task '{}' failed".format(task_name_compat(task))
                    ) from exception
                raise RuntimeError(
                    "Critical task '{}' stopped unexpectedly".format(
                        task_name_compat(task)
                    )
                )

    except asyncio.CancelledError:
        logger.info("Task main was cancelled.")
        raise

    except Exception:
        logger.exception("Unexpected error in main()")
        raise

    finally:
        await stop_script()
        if shutdown_script:
            logger.info("Initiating system shutdown...")
            result = await run_command(
                "sudo shutdown -h now",
                log_output=True
            )
            logger.info(
                "Shutdown command finished with return code %s",
                result["returncode"]
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())  # starts the script
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping script...")
    except asyncio.CancelledError:
        logger.info("Main coroutine cancelled during shutdown.")
        pass
