#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
import threading
import re
import logging
import importlib
from pathlib import Path
from datetime import datetime
from time import sleep

VENV_PATH = os.path.expanduser("~/.venv-canbus")
VENV_PYTHON = os.path.join(VENV_PATH, "bin", "python3")

# ===== Logging Config =====
ENABLE_LOGGING = True
LOG_TO_CONSOLE = True

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = Path(__file__).resolve().parent
logs_root = base_path / "logs"


SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


class StripAnsiFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return ANSI_ESCAPE_RE.sub('', message)


def setup_logging():
    if not ENABLE_LOGGING:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        return

    logs_root.mkdir(parents=True, exist_ok=True)

    # Unterordner pro Script
    script_log_dir = logs_root / SCRIPT_NAME
    script_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = script_log_dir / f"{now}_{SCRIPT_NAME}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_formatter = StripAnsiFormatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.handlers.clear()
    werkzeug_logger.propagate = False
    werkzeug_logger.disabled = True

    logging.info("")
    logging.info("=== Script start ===")
    logging.info("Script: %s", os.path.abspath(__file__))
    logging.info("Log file: %s", log_file)
    logging.info("Python: %s", sys.executable)


setup_logging()

# Prüfen, ob wir schon in der venv laufen
if not hasattr(sys, "real_prefix") and (sys.prefix != VENV_PATH):
    if os.path.exists(VENV_PYTHON):
        logging.info("Not in venv, restarting with %s ...", VENV_PYTHON)
        os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)
    else:
        logging.error("Virtualenv not found at %s", VENV_PATH)
        sys.exit(1)


def ensure_package(module_name, pip_name=None):
    """
    Import a Python module or install the matching pip package into the active venv.
    """
    if pip_name is None:
        pip_name = module_name

    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        logging.warning("Python package '%s' not found. Installing...", pip_name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                pip_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            logging.info(result.stdout.strip())
        if result.stderr:
            logging.info(result.stderr.strip())

        # Verify import after installation.
        importlib.import_module(module_name)
        logging.info("Package '%s' installed successfully.", pip_name)

    except subprocess.CalledProcessError as e:
        logging.error("Failed to install Python package '%s'.", pip_name)
        if e.stdout:
            logging.error("pip stdout: %s", e.stdout.strip())
        if e.stderr:
            logging.error("pip stderr: %s", e.stderr.strip())
        sys.exit(1)
    except ModuleNotFoundError:
        logging.exception(
            "Package '%s' was installed, but module '%s' still cannot be imported.",
            pip_name,
            module_name,
        )
        sys.exit(1)


REQUIRED_PACKAGES = {
    "flask": "flask",
    "uinput": "python3-uinput",
}

for module, package in REQUIRED_PACKAGES.items():
    ensure_package(module, package)

# --- ab hier: der eigentliche Server-Code ---
from flask import Flask, send_from_directory, make_response, request, jsonify
import uinput

app = Flask(__name__)
HTML_FOLDER = os.path.expanduser("~/scripts/hudiy_api/html_files")

# --- uinput lazy init / retry ---
device = None
device_lock = threading.Lock()


def debug_uinput_state():
    paths = ["/dev/uinput", "/dev/input/uinput"]
    logging.info("")

    ("DEBUG: uid=%s euid=%s", os.getuid(), os.geteuid())
    for p in paths:
        exists = os.path.exists(p)
        logging.info("DEBUG: %s exists=%s", p, exists)
        if exists:
            try:
                st = os.stat(p)
                logging.info(
                    "DEBUG: %s mode=%s uid=%s gid=%s",
                    p,
                    oct(st.st_mode & 0o777),
                    st.st_uid,
                    st.st_gid
                )
            except Exception as e:
                logging.exception("DEBUG: stat failed for %s: %s", p, e)


def create_uinput_device():
    debug_uinput_state()
    return uinput.Device([uinput.KEY_H, uinput.KEY_T])


def get_uinput_device(force_recreate=False):
    global device

    with device_lock:
        if force_recreate:
            device = None

        if device is not None:
            return device

        try:
            device = create_uinput_device()
            logging.info("uinput device initialized")
            logging.info("")
            return device
        except Exception as e:
            logging.exception("Failed to initialize uinput device: %s", e)
            device = None
            return None


def emit_key(key):
    global device

    dev = get_uinput_device()
    if dev is None:
        return False, "uinput device not available"

    try:
        dev.emit(key, 1)
        sleep(0.05)
        dev.emit(key, 0)
        return True, None
    except Exception as e:
        logging.exception("emit failed, recreating device: %s", e)

    dev = get_uinput_device(force_recreate=True)
    if dev is None:
        return False, "uinput device recreate failed"

    try:
        dev.emit(key, 1)
        sleep(0.05)
        dev.emit(key, 0)
        return True, None
    except Exception as e:
        logging.exception("emit failed after recreate: %s", e)
        return False, str(e)


@app.after_request
def log_response(response):
    logging.info(
        '%s "%s %s %s" %s',
        request.remote_addr,
        request.method,
        request.path,
        request.environ.get("SERVER_PROTOCOL"),
        response.status_code
    )
    return response


@app.route("/favicon.ico")
def favicon():
    return "", 204


# --- Serve other files with caching disabled ---
@app.route("/", defaults={"filename": "index.html"})
@app.route("/<path:filename>")
def serve_file(filename):
    response = make_response(send_from_directory(HTML_FOLDER, filename))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/hudiy_home")
def hudiy_home():
    ok, err = emit_key(uinput.KEY_H)
    ref = request.referrer
    logging.info("%s requested /hudiy_home from %s", request.remote_addr, ref)

    if not ok:
        logging.error("/hudiy_home failed: %s", err)
        return jsonify({"success": False, "error": err}), 500

    return jsonify({"success": True, "message": "H (Hudiy Home) sent via uinput"}), 200


@app.route("/toggle_hudiy_focus")
def toggle_hudiy_focus():
    ok, err = emit_key(uinput.KEY_T)
    ref = request.referrer
    logging.info("%s requested /toggle_hudiy_focus from %s", request.remote_addr, ref)

    if not ok:
        logging.error("/toggle_hudiy_focus failed: %s", err)
        return jsonify({"success": False, "error": err}), 500

    return jsonify({"success": True, "message": "T (toggle hudiy input focus) sent via uinput"}), 200



# --- Reverse Camera command to read_from_canbus.py ---
def send_can_script_command(command, host="127.0.0.1", port=23456):
    """
    Sends a small TCP command to read_from_canbus.py.

    Required in read_from_canbus.py handle_client():
        elif command == 'toggle_revcam_no_lines':
            fire_and_forget(
                asyncio.get_running_loop(),
                toggle_camera(with_lines=False),
                "toggle_camera_no_lines_from_remote"
            )
    """
    try:
        with socket.create_connection((host, port), timeout=1.0) as sock:
            sock.sendall(command.encode("utf-8"))
        return True, None
    except Exception as e:
        logging.exception("Failed to send command '%s' to CAN script: %s", command, e)
        return False, str(e)


@app.route("/revcam_toggle")
def revcam_toggle():
    logging.info("%s requested /revcam_toggle from %s", request.remote_addr, request.referrer)

    ok, err = send_can_script_command("toggle_revcam_no_lines")
    if not ok:
        return jsonify({"success": False, "error": err}), 500

    return jsonify({
        "success": True,
        "message": "Reverse camera toggle requested"
    }), 200


# --- Power Endpoints ---
@app.route("/shutdown")
def shutdown():
    logging.info("%s requested /shutdown", request.remote_addr)
    subprocess.Popen(["sudo", "shutdown", "-h", "now"])
    return "Shutdown command sent!", 200


@app.route("/reboot")
def reboot():
    logging.info("%s requested /reboot", request.remote_addr)
    subprocess.Popen(["sudo", "reboot"])
    return "Reboot command sent!", 200


@app.route("/close_hudiy")
def close_hudiy():
    logging.info("%s requested /close_hudiy", request.remote_addr)
    subprocess.Popen(["pkill", "-f", "hudiy"])
    return "Close hudiy command sent!", 200


# Lautstärke setzen
@app.route("/set_volume")
def set_volume():
    level = request.args.get("level")
    if level is None:
        return jsonify({"error": "No level provided"}), 400

    try:
        vol = int(level)
    except ValueError:
        return jsonify({"error": "Invalid level"}), 400

    vol = max(0, min(100, vol))
    wp_value = vol / 100.0

    try:
        subprocess.run(
            ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{wp_value:.2f}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info("%s requested /set_volume?level=%s -> backend=wpctl", request.remote_addr, vol)
        return jsonify({"success": True, "backend": "wpctl", "level": vol})
    except Exception:
        pass

    try:
        subprocess.run(
            ["amixer", "sset", "Digital", f"{vol}%"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info("%s requested /set_volume?level=%s -> backend=amixer", request.remote_addr, vol)
        return jsonify({"success": True, "backend": "amixer", "level": vol})
    except Exception as e:
        logging.exception("/set_volume failed: %s", e)
        return jsonify({"error": str(e)}), 500


# Read volume
@app.route("/get_volume")
def get_volume():
    try:
        result = subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
            capture_output=True,
            text=True,
            check=True
        )
        m = re.search(r"Volume:\s*([0-9]*\.?[0-9]+)", result.stdout)
        if m:
            level = int(round(float(m.group(1)) * 100))
            level = max(0, min(100, level))
            return jsonify({"level": level, "backend": "wpctl"})
    except Exception:
        pass

    for control in ("Master", "Digital"):
        try:
            result = subprocess.run(
                ["amixer", "get", control],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                continue

            m = re.search(r"\[(\d+)%]", result.stdout)
            if m:
                return jsonify({"level": int(m.group(1)), "backend": f"amixer:{control}"})
        except Exception:
            continue

    return jsonify({"level": 50, "backend": "fallback"})


def run_game(cmd):
    env = os.environ.copy()
    env["DISPLAY"] = ":0"
    subprocess.Popen(cmd, env=env)


# --- Game Endpoints ---
@app.route("/start_emulationstation")
def start_emulationstation():
    run_game(["es-de"])
    return "EmulationStation launched!", 200


@app.route("/start_audi_dash")
def start_audi_dash():
    run_game([VENV_PYTHON, "/home/pi/scripts/audi_dash/audi_dash.py"])
    return "Audi Dash launched!", 200

# --- Run Flask server ---
if __name__ == "__main__":
    logging.info("Starting Flask server on 127.0.0.1:44408")
    logging.info("")
    app.run(host="127.0.0.1", port=44408, debug=False)
