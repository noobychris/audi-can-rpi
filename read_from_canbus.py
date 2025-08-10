#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-

#  Enable here the functions you want to use
#########################################################
#
can_interface = 'can0'  # set can interface - default 'can0'
#
welcome_message_1st_line = 'WELCOME'  # Welcome message 1st line on script startup. Max 8 digits.
welcome_message_2nd_line = 'USER'  # Welcome message 2nd line on script startup. Max 8 digits.
#
send_on_canbus = True  # True/False = the script will send/not send on canbus.
only_send_if_radio_is_in_tv_mode = True  # True/False = script will just send if rns-e is in tv mode.
#
activate_rnse_tv_input = False  # Send the TV input activation message if you don't have an IMA.
tv_input_format = 'NTSC'  # Select if you know your video input format. 'PAL' or 'NTSC'
#
# Dashboard dis/fis 1st and 2nd Line                    #
show_label = False  # True = 1st line dis/fis shows label from 2nd lines value.
toggle_fis1 = 1  # check the numbers below this box     # 1st line dis/fis - disabled if show_label = True
toggle_fis2 = 6  # check the numbers below this box     # 2nd line dis/fis
scroll_type = 'oem_style'  # "scroll" or "oem_style"    # oem_style: show 8 digits and switch to the next 8 digits
#
read_and_set_time_from_dashboard = True  # Read the date and the time from the dashboard.
control_pi_by_rns_e_buttons = True  # Use rns-e buttons to control the raspberry pi.
send_values_to_dashboard = True  # Send speed, rpm, coolant, pi cpu usage and temp to dashboard
toggle_values_by_rnse_longpress = True  # Toggle values with up/down longpress
#
reversecamera_by_reversegear = False  # Use hdmi to csi shield to connect a reversecamera.
reversecamera_by_down_longpress = False  #
reversecamera_guidelines = False  # Show guidelines and put a .png file into the script folder.e
reversecamera_turn_off_delay = 5  # Delay to turn off the reversecamera in seconds.
#
shutdown_by_ignition_off = True  # Shutdown the raspberry pi, if the ignition went off.
shutdown_by_pulling_key = False  # Shutdown the raspberry pi, if the key got pulled.
shutdown_type = 'gently'  # 'gently' waits for stopping all threads / 'instant' shuts downn the pi instantly
#
# OAP API FEATURES !requires OpenAuto Pro 15 or higher! #
initial_day_night_mode = 'night'  # set here the mode you want to set by default."day"/"night"
change_dark_mode_by_car_light = True  # read cars light state on/off to change oap and aa day/night.
send_oap_api_mediadata_to_dashboard = True  # Send oap api mediadata (title, artist, etc.) to dashboard
send_to_oap_gauges = True  # obdinject speed etc to oap api
#
# Speed measure 0-100 km/h etc.                         #
lower_speed = 0  # speed to start acceleration measurement (example: 0 km/h)
upper_speed = 100  # speed to stop acceleration measurement (example: 100 km/h)
export_speed_measurements_to_file = True  #
#
speed_unit = 'km/h'  # 'km/h' or 'mph'
temp_unit = '°C'  # '°C' or '°F'
#
ENABLE_LOGGING = False  #
show_can_messages_in_logs = False
#
#########################################################

# 1 = title, 2 = artist, 3 = album, 4 = song position, 5 = song duration, 6 = speed, 7 = rpm, 8 = coolant
# 9 = cpu/temp, 10 = speed measure, 11 = outside temp, # 12 = blank line, 13 = disable sending to dis/fis


# If you have any trouble with the script, you can enable LOGGING_OUTPUT to get more information's about exceptions.
# The messages will be saved in the script/logs folder with date, name of the script.
# Example: 2023-02-01_read_from_canbus_errors.log

import sys, os, logging, threading, time, binascii, textwrap, asyncio, importlib.util
import zipfile, io, shutil, inspect, traceback, configparser
from datetime import datetime
from functools import wraps
from inspect import unwrap

stop_flag, tmset, car_model_set, carmodel, shutdown_script, Client, version, PackageNotFoundError = False, None, None, '', False, None, None, None
press_mfsw, up, down, select, back = 0, 0, 0, 0, 0
nextbtn, prev, setup, gear, light_status = 0, 0, 0, 0, 0
ProjectionState, ProjectionSource = None, None
lock = threading.Lock()
stop_completed_event = threading.Event()
tasks = []
remote_task = None

# Ensure script is run with Python 3
if sys.version_info < (3, 0):
    logger.info("🚫 This script requires Python 3. Please run it using 'python3 script.py'")
    sys.exit(1)


class ThreadNameFilter(logging.Filter):
    def filter(self, record):
        if "can.notifier" in record.threadName:
            record.threadName = "can_notifier"
        elif record.threadName.startswith("ThreadPoolExecutor-"):
            record.threadName = "OAP_EventHandler"
        return True


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


def is_python_too_old(min_version=(3, 13, 5)):
    return sys.version_info < min_version


async def run_command(command: str, log_output: bool = True):
    """Executes a shell command asynchronously and optionally logs the output."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if log_output:
        if stdout:
            for line in stdout.decode().splitlines():
                logger.info(line)
        if stderr:
            for line in stderr.decode().splitlines():
                logger.warning(line)

    return {
        "stdout": stdout.decode().strip(),
        "stderr": stderr.decode().strip(),
        "returncode": process.returncode
    }


def logger_prompt(level, func_name, lineno, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    level_str = level.upper().ljust(7)
    func_str = func_name.ljust(20)
    sys.stdout.write(f"{timestamp:<23} | {level_str:<7} | {lineno:<4} | {func_str:<21} | {message}")
    # header = f"{'TIME':<23} | {'LEVEL':<7} | {'LINE':<4} | {'FUNCTION':<21} | {'MESSAGE'}"
    sys.stdout.flush()
    return input()


async def install_python_3_13_5():
    logger.info("🚀 Starting Python 3.13.5 installation (this may take 20–30 minutes)...")

    # ⛔ Terminate any running instance of autoapp if applicable
    logger.info("🔍 	Checking if 'autoapp' is running...")
    await run_command("sudo pkill -f autoapp && echo '✅ autoapp was closed.' || echo 'ℹ️ autoapp was not active.'",
                      log_output=True)

    commands = [
        ("Updating package list...", "sudo apt update"),
        ("Installing build dependencies...", "sudo apt install -y build-essential libssl-dev zlib1g-dev "
                                             "libncurses5-dev libbz2-dev libreadline-dev libsqlite3-dev "
                                             "wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev "
                                             "libxmlsec1-dev libffi-dev liblzma-dev"),
        ("Downloading Python 3.13.5 source...",
         "cd /usr/src && sudo wget https://www.python.org/ftp/python/3.13.5/Python-3.13.5.tgz"),
        ("Extracting archive...", "cd /usr/src && sudo tar xzf Python-3.13.5.tgz"),
        ("Configuring build...", "cd /usr/src/Python-3.13.5 && sudo ./configure --enable-optimizations"),
        ("Compiling Python (this may take a while)...", "cd /usr/src/Python-3.13.5 && sudo make -j$(nproc)"),
        ("Installing Python 3.13.5...", "cd /usr/src/Python-3.13.5 && sudo make altinstall")
    ]

    for step_msg, cmd in commands:
        logger.info(f"➡️ {step_msg}")
        if not await run_command(cmd, log_output=False):
            logger.error("❌ Installation failed at step: %s", step_msg)
            sys.exit(1)

    logger.info("⬆️ Upgrading pip for Python 3.13.5...")
    if not await run_command("python3.13.5 -m pip install --upgrade pip", log_output=False):
        logger.warning("⚠️ Failed to upgrade pip.")
    else:
        logger.info("✅ pip for Python 3.13.5 upgraded successfully!")

    logger.info("🎉 Python 3.13.5 installed successfully!")


async def set_python3_alias():
    bashrc = os.path.expanduser("~/.bashrc")
    alias_lines = [
        "alias python3='python3.13'",
        "alias pip3='pip3.13'"
    ]

    already_set = False
    if os.path.exists(bashrc):
        with open(bashrc, "r") as f:
            content = f.read()
            already_set = any("alias python3='python3.13'" in line for line in content.splitlines())

    if not already_set:
        with open(bashrc, "a") as f:
            f.write("\n# Use Python 3.13 as default\n")
            for line in alias_lines:
                f.write(line + "\n")
        logger.info("✅ Alias python3 → python3.13 added to ~/.bashrc.")

        if os.environ.get("SHELL") and sys.stdin.isatty():
            logger.info("🔄 Applying alias now with: source ~/.bashrc")
            await run_command("source ~/.bashrc", log_output=False)

        logger.info("ℹ️  If the alias doesn't work immediately, open a new terminal or run:")
        logger.info("    source ~/.bashrc")
    else:
        logger.info("ℹ️  Alias already exists in ~/.bashrc.")


def restart_with_python_3_13_5():
    python3135 = shutil.which("python3.13.5")
    if not python3135:
        logger.error("⚠️ python3.13.5 not found – something went wrong during installation.")
        sys.exit(1)

    logger.info(f"🔁 Restarting script using {python3135} ...\n")
    os.execv(python3135, [python3135] + sys.argv)


async def ensure_python_3_13_5():
    """
    Checks if current Python version is >= 3.13.5, otherwise installs Python 3.13.5,
    sets up alias, and restarts script.
    """
    from platform import python_version
    if is_python_too_old((3, 13, 5)):
        logger.warning(f"⚠️ Your current Python version is too old: {python_version()}")
        response = logger_prompt(
            level="warning",
            func_name=inspect.currentframe().f_code.co_name,
            lineno=inspect.currentframe().f_lineno + 1,
            message="❓ Do you want to automatically install Python 3.13.5 now? [y/N]: "
        ).strip().lower()

        if response == "y":
            await install_python_3_13_5()
            await set_python3_alias()
            restart_with_python_3_13_5()
        else:
            logger.info("🚫 Script aborted. Please install Python ≥ 3.13.5 manually.")
            sys.exit(1)


async def script_starting():
    logger.info("")
    logger.info("Script is starting...")
    logger.info("")


server_socket = None
server_running = True
async def remote_control(host='localhost', port=12345):
    global server_socket

    try:
        if server_socket:
            logger.info("Old server socket exists – closing...")
            server_socket.close()
            await server_socket.wait_closed()
            server_socket = None

        server = await asyncio.start_server(handle_client, host, port)
        server_socket = server
        logger.info(f"Remote control server started on {host}:{port}")

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
                logger.info("✅ Remote server socket closed.")
            except Exception as e:
                logger.warning(f"⚠️ Error closing socket: {e}")
            server_socket = None
        remote_server_shutdown_event.clear()  # <– important!


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




path = os.path.dirname(os.path.abspath(__file__))
script_filename = str(os.path.basename(__file__))

async def run_command(cmd, log_output=False):
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    result = {"stdout": stdout.decode().strip(), "stderr": stderr.decode().strip(), "returncode": proc.returncode}
    if log_output:
        logger.warning(f"{cmd} → {result}")
    return result

async def get_other_pids(script_name):
    current = os.getpid()
    result = await run_command(f'pgrep -fa "{script_name}"')
    return [int(line.split()[0]) for line in result["stdout"].splitlines() if "python" in line and int(line.split()[0]) != current]

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


async def stop_other_instance(script_name):
    pids = await get_other_pids(script_name)
    if not pids:
        if ENABLE_LOGGING:
            logger.info("No other script instances running.")
        return False

    logger.info(f"Detected other running script instance(s): {pids} → sending 'stop_script'")
    await send_command("localhost", 12345, "stop_script")

    if await wait_for_stop(script_name):
        logger.info("Other running script instance(s) shut down gracefully.")
        logger.info("")
        return False

    logger.warning("No response from other running script instance(s) – sending SIGTERM...")
    for pid in pids:
        await run_command(f"kill -15 {pid}")
    await asyncio.sleep(1)

    remaining = await get_other_pids(script_name)
    for pid in remaining:
        logger.warning(f"PID {pid} still running – sending SIGKILL")
        await run_command(f"kill -9 {pid}")
    return True

async def is_script_running(script_name):
    try:
        return await stop_other_instance(script_name)
    except Exception as e:
        logger.error("Error checking for running script instances", exc_info=True)
        return False

FIS1, FIS2, speed, rpm, coolant, playing, position, source, title, artist, album, state = '265', '267', 0, 0, 0, '', '', '', '', '', '', None
duration, begin1, end1, begin2, end2, tv_mode_active, outside_temp = '', -1, 7, -1, 7, 1, ""
pause_fis1, pause_fis2, light_set, script_started, deactivate_overwrite_dis_content = False, False, False, False, False
can_functional, guidelines_set, camera_active, cpu_load, cpu_temp, cpu_freq_mhz = None, False, None, 0, 0, 0

_cached_metadata = None


async def ensure_importlib_metadata(logger=None):
    global _cached_metadata
    if _cached_metadata:
        return _cached_metadata

    try:
        from importlib.metadata import version, PackageNotFoundError
        _cached_metadata = (version, PackageNotFoundError)
        if logger:
            logger.info("Using importlib.metadata (stdlib)")
        return _cached_metadata
    except ImportError:
        pass

    # Check if the backport module is installed
    if importlib.util.find_spec("importlib_metadata") is None:
        if logger:
            logger.warning("Backport 'importlib_metadata' missing. Attemping to install it...")
        else:
            logger.info("Backport 'importlib_metadata' missing. Attemping to install it...")

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--user", "importlib-metadata",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if logger:
                logger.error(f"Installation from importlib-metadata failed: {error_msg}")
            raise RuntimeError("Installation from 'importlib-metadata' failed")

        if logger:
            logger.info("importlib-metadata successfully installed.")

    # It should now be available
    try:
        from importlib_metadata import version, PackageNotFoundError
        _cached_metadata = (version, PackageNotFoundError)
        if logger:
            logger.info("Using importlib_metadata (backport)")
        return _cached_metadata
    except ImportError:
        raise RuntimeError("Could not import 'importlib_metadata' after installation.")


async def check_python():
    python_path = sys.executable
    python_version = sys.version_info
    python_version_str = f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"

    if ENABLE_LOGGING:
        loop = asyncio.get_event_loop()
    logger.info(f"Python interpreter: {python_version_str} ({python_path})")
    logger.info("")


async def check_can_utils():
    from packaging import version
    if ENABLE_LOGGING:
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

    current_version = None
    needs_upgrade = True

    # 🔎 Check if candump AND cansend are in the path

    candump_path = shutil.which("candump")
    cansend_path = shutil.which("cansend")
    candump_ok = candump_path is not None
    cansend_ok = cansend_path is not None

    if candump_ok and cansend_ok:
        if ENABLE_LOGGING:
            logger.info(f"✅ candump found at: {candump_path}")
            logger.info(f"✅ cansend found at: {cansend_path}")
    else:
        logger.info("ℹ️ candump and/or cansend not found in PATH. Will check/install can-utils.")

    # 1. Check APT installation
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


async def python_modules():
    required = {
        "aioconsole",
        "aiofiles",
        "requests",
        "picamera",
        "protobuf:3.20",
        "google",
        "python-can",
        "psutil",
        "python-uinput",
        "Pillow",
        "packaging",
    }

    installed_modules = {}
    missing_modules = []

    for module in required:
        if ":" in module:
            mod_name, required_version = module.split(":")
        else:
            mod_name, required_version = module, None

        try:
            installed_version = version(mod_name)
            installed_modules[mod_name] = installed_version
            if required_version and installed_version.split(".")[:2] != required_version.split(".")[:2]:
                missing_modules.append(f"{mod_name}=={required_version}")
        except PackageNotFoundError:
            if required_version:
                missing_modules.append(f"{mod_name}=={required_version}")
            else:
                missing_modules.append(mod_name)

    if installed_modules and ENABLE_LOGGING:
        logger.info("")
        logger.info("✅ Installed Python3 (pip) Modules:")
        for mod, ver in installed_modules.items():
            logger.info("   • %s %s", mod, ver)

    if missing_modules:
        logger.warning("⚠️  Missing modules:")
        for mod in missing_modules:
            logger.warning("   • %s", mod)
    else:
        logger.info("✅ All required python3 (pip) modules are installed.")
        logger.info("")

    return installed_modules, missing_modules


async def install_missing(missing_modules):
    if missing_modules:
        logger.info("📦 Installing missing modules...")
        logger.info("")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                *missing_modules,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                installed = []
                for line in stdout.decode().splitlines():
                    if line.startswith("Successfully installed"):
                        installed = line.strip().split("Successfully installed ")[1].split()
                        break

                if installed:
                    logger.info("✅ Successfully installed:")
                    for mod in installed:
                        logger.info("   • %s", mod)
                else:
                    logger.info("✅ Modules installed (could not parse list).")

                logger.info("🔁 Restarting the script...\n\n")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                logger.error(f"❌ Failed to install modules: {stderr.decode().strip()}")

        except Exception as error:
            logger.error(f"❌ Exception during module installation: {error}")


def import_modules(installed_modules: dict):
    if ENABLE_LOGGING:
        logger.info("✅ Successfully imported modules:")
    for mod, ver in installed_modules.items():
        try:
            if mod == "aiofiles":
                global aiofiles
                import aiofiles
            elif mod == "requests":
                global requests
                import requests
            elif mod == "python-can":
                global can, Notifier
                import can
                from can import Notifier
            elif mod == "psutil":
                global psutil
                import psutil
            elif mod == "Pillow":
                global Image
                from PIL import Image
            elif mod == "picamera":
                global picamera
                import picamera
            elif mod == "google":
                global google
                import google
            elif mod == "protobuf":
                global protobuf
                import google.protobuf as protobuf
            elif mod == "aioconsole":
                global aioconsole
                import aioconsole
            elif mod == "python-uinput":
                global uinput
                import uinput
            if ENABLE_LOGGING:
                logger.info("   • %s %s", mod, ver)


        except ImportError as e:
            logger.warning("⚠️  Failed to import %s (installed version: %s): %s", mod, ver, e)
    if ENABLE_LOGGING:
        logger.info("")


async def uinput_permissions():
    if control_pi_by_rns_e_buttons:
        try:
            result = await run_command("stat /dev/uinput", log_output=False)
            if "0666" not in result["stdout"]:
                logger.warning("Permissions for /dev/uinput are incorrect.")
                logger.info("Setting correct permissions...")
                await run_command("sudo modprobe uinput", log_output=False)
                await run_command("sudo chmod 666 /dev/uinput", log_output=False)
                result = await run_command("stat /dev/uinput", log_output=False)
                if "0666" in result["stdout"]:
                    logger.info("Permissions successfully set.")
                    await import_uinput()
                else:
                    logger.error("Failed to set permissions for /dev/uinput.")
                    return False
            else:
                if ENABLE_LOGGING:
                    logger.info("Permissions for /dev/uinput are correct.")
                    logger.info("")
                await import_uinput()
        except Exception as error:
            await handle_exception(error, "Couldn't check uinput permissions.")


# Check if the raspberry pi is in powersave mode (pi will stick at 600MHz frequency).
# If that is the case, set the powermode/scaling mode to "ondemand" so the cpu can change its frequency dynamicly.
# To change this permanently, you can add "cpufreq.default_governor=ondemand" at the end of the file "/boot/cmdline.txt"

async def set_powerplan():
    import aiofiles
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


async def import_uinput():
    global events, device, uinput
    try:
        import uinput
        events = (
            uinput.KEY_1, uinput.KEY_2, uinput.KEY_UP, uinput.KEY_DOWN, uinput.KEY_LEFT, uinput.KEY_RIGHT,
            uinput.KEY_ENTER, uinput.KEY_ESC, uinput.KEY_F2, uinput.KEY_B, uinput.KEY_N, uinput.KEY_V,
            uinput.KEY_F12, uinput.KEY_M, uinput.KEY_X, uinput.KEY_C, uinput.KEY_LEFTCTRL
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
CONFIG_FILE = "/boot/config.txt"
REQUIRED_LINES = [
    "dtparam=spi=on",
    "dtoverlay=mcp2515-can0,oscillator=16000000,interrupt=25",
    "dtoverlay=spi-bcm2835-overlay"
]


async def check_pican2_3_config():
    import aioconsole

    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"❌ Error: {CONFIG_FILE} not found!")
        return
    result = await run_command(f"sudo cat {CONFIG_FILE}", log_output=False)
    if result["returncode"] != 0:
        logger.error(f"❌ Error reading {CONFIG_FILE}: {result['stderr']}")
        return
    config_lines = result["stdout"].splitlines()
    missing_lines = [line for line in REQUIRED_LINES if line not in config_lines]
    if not missing_lines:
        logger.info("✅ The PiCAN2/3 configuration is already correct.")
        return
    logger.warning("⚠️ The depending lines for PiCAN2/3 board are missing in /boot/config.txt:")
    for line in missing_lines:
        logger.warning(f"  ➤ {line}")
    choice = (await aioconsole.ainput("Would you like to add these lines now? (yes/no): ")).strip().lower()
    if choice not in ["yes", "y"]:
        logger.warning("❌ Aborted. The file was not modified.")
        return
    newline = "\n"
    command = f"sudo bash -c 'echo -e \"{newline}#PiCAN2/3 Settings{newline}{newline.join(missing_lines)}\" >> {CONFIG_FILE}'"
    await run_command(command, log_output=False)
    if result["returncode"] != 0:
        logger.error(f"❌ Error writing to {CONFIG_FILE}: {result['stderr']}")
        return
    logger.info("✅ The missing lines have been added to /boot/config.txt.")
    reboot_choice = (await aioconsole.ainput(
        "🔄 A reboot is required to detect the PiCAN2/3 board. Do you want to reboot now? (yes/no): ")).strip().lower()
    if reboot_choice in ["yes", "y"]:
        logger.info("🔄 Rebooting the system now...")
        await run_command("sudo reboot", log_output=False)
    else:
        logger.warning("ℹ️ Please reboot manually for the changes to take effect.")


async def test_can_interface():
    """Tests and initializes the CAN interface.."""
    global bus, send_on_canbus, can_functional
    bus = None  # Ensure that bus is initialized
    can_functional = False

    try:
        if can_interface == 'vcan0':
            result = await run_command("ip link show vcan0", log_output=False)
            if result["stderr"]:
                logger.warning("vcan0 does not exist. Creating vcan0...")
                result = await run_command("sudo ip link add dev vcan0 type vcan", log_output=False)
                if result["stderr"]:
                    logger.error(f"Failed to create vcan0: {result['stderr']}")
                    return
            logger.info("vcan0 created successfully, because it was not existing.")
            result = await run_command("sudo ip link set up vcan0", log_output=False)
            if result["stderr"]:
                logger.error(f"Failed to bring vcan0 up: {result['stderr']}")
                return
            logger.info("vcan0 interface is up.")
            try:
                bus = can.interface.Bus(
                    can_interface,
                    interface='socketcan',
                    bitrate=100000,
                    can_filters=can_filters,
                    receive_own_messages=False
                )
                logger.info("CAN-Interface 'vcan0' found and opened.")
                result = await run_command(f'sudo ifconfig {can_interface} txqueuelen 1000', log_output=False)
                if result["stderr"]:
                    logger.error(f"Failed to set txqueuelen for vcan0: {result['stderr']}")
                    send_on_canbus = False
                    return
            except can.CanError as e:
                logger.error(f"Failed to initialize CAN-Bus on {can_interface}. Error: {e}")
                return
        else:
            if not os.path.exists(f'/sys/class/net/{can_interface}/operstate'):
                logger.warning(f"Interface {can_interface} does not exist. Maybe it was not installed properly?")
                await check_pican2_3_config()
                return
            async with aiofiles.open(f'/sys/class/net/{can_interface}/operstate', mode='r') as f:
                can_network_state = (await f.read()).strip()
            if can_network_state != 'up':
                logger.warning(f"{can_interface} is down, trying to bring it up...")
                result = await run_command(
                    f'sudo /sbin/ip link set {can_interface} up type can restart-ms 1000 bitrate 100000',
                    log_output=False
                )
                if result["stderr"]:
                    logger.error(f"Failed to bring {can_interface} up: {result['stderr']}")
                    return
                result = await run_command(f'sudo ifconfig {can_interface} txqueuelen 1000', log_output=False)
                if result["stderr"]:
                    logger.error(f"Failed to set txqueuelen for {can_interface}: {result['stderr']}")
                    return
            try:
                bus = can.interface.Bus(
                    can_interface,
                    interface='socketcan',
                    bitrate=100000,
                    can_filters=can_filters,
                    receive_own_messages=False
                )
                logger.info(f"CAN-Interface '{can_interface}' found and opened.")
            except can.CanError as e:
                logger.error(f"Failed to initialize CAN-Bus on {can_interface}. Error: {e}")
                return
        if bus is not None:
            received_message = bus.recv(timeout=1.0)  # Timeout in seconds (non-blocking)
            if received_message is not None or can_interface == 'vcan0':
                logger.info("CAN message received. CAN-Bus seems to be working.")
                logger.info("")
                can_functional = True
            else:
                logger.warning("No CAN message received. Disabling CAN-Bus communication.")
        else:
            logger.error("CAN-Bus initialization failed, no bus object available.")
    except FileNotFoundError:
        logger.error("File not found!", exc_info=True)
    except Exception as e:
        logger.error(f"Error while testing the CAN interface: {e}", exc_info=True)
    finally:
        if not can_functional and bus is None:
            logger.error("Failed to initialize CAN-Bus. Disabling CAN-Bus features.")
            send_on_canbus = False


@handle_errors
async def check_camera():
    global reversecamera_by_reversegear, reversecamera_by_down_longpress
    try:
        process = await asyncio.create_subprocess_exec(
            "vcgencmd", "get_camera", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error("Error while checking camera status: %s", stderr.decode())
            return False
        result = stdout.decode()
        if "supported=1 detected=1" in result:
            logger.info("PiCamera is activated and detected.")
            return True
        elif "supported=1 detected=0" in result:
            logger.warning("PiCamera (in Raspberry Pi Config) is activated but NOT detected.")
            reversecamera_by_reversegear = False
            reversecamera_by_down_longpress = False
            return False
        else:
            logger.warning("PiCamera is deactivated (in Raspberry Pi Config), but reversecamera features are enabled.")
            logger.warning("Please activate PiCamera via Raspberry Pi Config (sudo raspi-config)")
            return False
    except Exception:
        logger.error("Error while checking PiCamera status.")
        return False


@handle_errors
async def initialize_reverse_camera():
    global reversecamera_by_reversegear, reversecamera_by_down_longpress
    try:
        from picamera import PiCamera
    except ImportError as error:
        logger.error("Couldn't import PiCamera. Reverse camera features will be disabled.")
        await handle_exception(error, "Couldn't import PiCamera.")
        return None
    try:
        camera_detected = await check_camera()
        if camera_detected:
            camera = PiCamera()
            logger.info("PiCamera connection successfully established")
            return camera
    except Exception as e:
        logger.error("Couldn't open PiCamera. Disabling reverse camera features now.", exc_info=True)
    reversecamera_by_reversegear = False
    reversecamera_by_down_longpress = False
    return None


async def add_overlay_async(camera):
    overlay_image_path = f'{path}/lines.png'
    loop = asyncio.get_running_loop()
    img = await loop.run_in_executor(None, Image.open, overlay_image_path)
    pad = Image.new('RGBA', (
        ((img.size[0] + 31) // 32) * 32,
        ((img.size[1] + 15) // 16) * 16,
    ))
    pad.paste(img, (0, 0), img)
    overlay = camera.add_overlay(pad.tobytes(), size=img.size, format='rgba', layer=3, alpha=128)
    overlay.fullscreen = True
    overlay.layer = 3
    return overlay


from asyncio import get_running_loop


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


@handle_errors
async def convert_audi_ascii(content=''):
    return ''.join(HEX_TO_AUDI_ASCII.get(content[i:i + 2], content[i:i + 2]) for i in range(0, len(content), 2))


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


def define_event_handler_class():
    class EventHandler(ClientEventHandler):
        def __init__(self, client, main_loop):
            self.client = client
            self.main_loop = main_loop
            self.read_cpu_task = None
            super().__init__()

        @handle_errors
        def on_hello_response(self, client, message):
            self.client = client  # ✅ important – take over the new client!

            logger.info("")
            logger.info("Received OpenAuto Pro hello response from OAP API")
            logger.info(f"oap version: {message.oap_version.major}.{message.oap_version.minor}")
            logger.info(f"api version: {message.api_version.major}.{message.api_version.minor}")
            if message.api_version.minor == 1:
                logger.warning(
                    "⚠️ API reports version 1.1, but GitHub release claims 1.2. Possibly outdated constant in proto file.")
            logger.info("")

            set_status_subscriptions = oap_api.SetStatusSubscriptions()
            if send_oap_api_mediadata_to_dashboard:
                set_status_subscriptions.subscriptions.append(
                    oap_api.SetStatusSubscriptions.Subscription.MEDIA)
                set_status_subscriptions.subscriptions.append(
                    oap_api.SetStatusSubscriptions.Subscription.PROJECTION
                )
            try:
                self.client.send(oap_api.MESSAGE_SET_STATUS_SUBSCRIPTIONS, 0, set_status_subscriptions.SerializeToString())
            except Exception as e:
                logger.warning("Failed to send subscription message to OAP API.", exc_info=True)

            if send_to_oap_gauges:
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

            # 🌗 Initial Day/Night Mode
            if initial_day_night_mode == "day":
                self.send_day_night('day')
            else:
                self.send_day_night('night')

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
            global ProjectionState, ProjectionSource
            ProjectionState, ProjectionSource = message.state, message.source
            if ENABLE_LOGGING:
                logger.info(f"Projection status, state: {ProjectionState}, source: {ProjectionSource}")

        @handle_errors
        def update_to_api(self, formula, variable, variable_name="variable"):
            global oap_api_is_connected

            try:
                if not oap_api_is_connected or not hasattr(self.client, '_socket') or self.client._socket is None:
                    logger.warning("Client not connected – skipping update_to_api.")
                    return
                caller = inspect.stack()[2]
                caller_info = f"{caller.function} (line {caller.lineno})"
                logger.info(
                    f"Sending formula: {formula} Variable: {variable_name} Value: {variable} ← from {caller_info}")

                # logger.info(f"Sending formula: {formula} Variable: {variable_name} Value: {variable}")
                msg = oap_api.ObdInjectGaugeFormulaValue()
                msg.formula = formula
                msg.value = variable

                try:
                    if oap_api_is_connected and hasattr(self.client, '_socket') and self.client._socket is not None:
                        if ProjectionState in (1, 3) and ProjectionSource == 0 and tv_mode_active == 1:
                            self.client.send(oap_api.MESSAGE_OBD_INJECT_GAUGE_FORMULA_VALUE, 0, msg.SerializeToString())
                        else:
                            logger.info(f"OpenAuto Pro is not in foreground or rns-e is not in tv_mode, skipping sending to oap api (exept outside temperature for 8E only).")
                except BrokenPipeError as e:
                    logger.error(f"Broken pipe from {caller_info}: {e}")
                    # logger.error(f"Broken pipe: {e}")
                    oap_api_is_connected = False
                    self.client = None
                    self.read_cpu_task.cancel()
                    logger.info("stopping read_cpu task")
                    try:
                        # Try to disconnect, but catch another BrokenPipeError
                        try:
                            self.client.disconnect()
                        except BrokenPipeError as e2:
                            logger.error(f"Broken pipe after disconnect from {caller_info}: {e}")
                            # logger.warning(f"Ignored BrokenPipeError during disconnect: {e2}")
                        except Exception as e2:
                            logger.warning(f"Exception during disconnect: {e2}")
                    finally:
                        logger.warning("Lost connection to OAP API – will trigger reconnect.")

            except Exception as e:
                logger.error(f"update_to_api failed: {e}", exc_info=True)

        @handle_errors
        def outside_to_oap_api(self, outside_temp_int):
            try:
                if ENABLE_LOGGING:
                    logger.info(f"Sending outside temperature : {outside_temp_int}{temp_unit} to API")
                inject_temperature_sensor_value = oap_api.InjectTemperatureSensorValue()
                inject_temperature_sensor_value.value = outside_temp_int  # set outside temperature as integer
                serialized_data = inject_temperature_sensor_value.SerializeToString()
                if oap_api_is_connected and hasattr(self.client, '_socket') and self.client._socket is not None:
                    self.client.send(oap_api.MESSAGE_INJECT_TEMPERATURE_SENSOR_VALUE, 0, serialized_data)
            except Exception as e:
                logger.error(f"Failed to send outside temperature '{outside_temp_int}' to API: {e}")
                raise

        @handle_errors
        def send_day_night(self, mode):
            try:
                if mode not in ['day', 'night']:
                    raise ValueError("Invalid mode. Use 'day' or 'night'.")
                is_day_mode = True if mode == 'day' else False
                if ENABLE_LOGGING:
                    logger.info(f"Sending day/night mode: {'Day' if is_day_mode else 'Night'} to API")
                set_day_night = oap_api.SetDayNight()
                set_day_night.android_auto_night_mode = not is_day_mode
                set_day_night.oap_night_mode = not is_day_mode
                serialized_data = set_day_night.SerializeToString()
                if oap_api_is_connected and hasattr(self.client, '_socket') and self.client._socket is not None:
                    self.client.send(oap_api.MESSAGE_SET_DAY_NIGHT, 0, serialized_data)

            except Exception as e:
                logger.error(f"Failed to send day/night mode '{mode}' to API: {e}")
                raise

        @handle_errors
        async def read_cpu(self, temp_unit):
            global cpu_load, cpu_temp, cpu_freq_mhz
            global last_cpu_temp, last_cpu_freq_mhz, last_cpu_load
            last_cpu_temp = None
            last_cpu_freq_mhz = None
            last_cpu_load = None
            logger.info("read_cpu started")
            while not stop_flag:

                if send_to_oap_gauges or 9 in (toggle_fis1, toggle_fis2):
                    try:
                        cpu_load = min(round(psutil.cpu_percent()), 99)
                        if send_to_oap_gauges and cpu_load != last_cpu_load and oap_api_is_connected and self.client:
                            self.update_to_api("getPidValue(4)", cpu_load, "cpu_load")
                            last_cpu_load = cpu_load
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
                            if send_to_oap_gauges and cpu_temp != last_cpu_temp and oap_api_is_connected and self.client:
                                self.update_to_api("getPidValue(5)", cpu_temp, "cpu_temp")
                                last_cpu_temp = cpu_temp
                        if send_to_oap_gauges:
                            cpu_freq = psutil.cpu_freq()
                            if cpu_freq:
                                cpu_freq_mhz = int(round(cpu_freq.current))
                                if cpu_freq_mhz != last_cpu_freq_mhz and oap_api_is_connected and self.client:
                                    self.update_to_api("getPidValue(6)", cpu_freq_mhz, "cpu_freq_mhz")
                                    last_cpu_freq_mhz = cpu_freq_mhz
                        await asyncio.sleep(3.0)
                    except asyncio.CancelledError:
                        if ENABLE_LOGGING:
                            logger.info("Task read_cpu was stopped.")
                        break
                    except Exception as e:
                        logger.error(f"Unexpected error in CPU monitor task: {e}", exc_info=True)

    return EventHandler


async def check_import_oap_api():
    global change_dark_mode_by_car_light, send_oap_api_mediadata_to_dashboard, send_to_oap_gauges

    if change_dark_mode_by_car_light or send_oap_api_mediadata_to_dashboard or send_to_oap_gauges:
        try:
            import common.Api_pb2
            from common import Client as OapClient

            global oap_api, Client, ClientEventHandler, EventHandler
            oap_api = common.Api_pb2
            Client = OapClient.Client
            ClientEventHandler = OapClient.ClientEventHandler

            EventHandler = define_event_handler_class()
        except ModuleNotFoundError as error:
            await handle_exception(error, "OAP API files not found. Trying to download and install them from github")

            # auto-download & install OAP API files
            repo = "bluewave-studio/openauto-pro-api"
            api_url = f"https://api.github.com/repos/{repo}/releases/latest"
            set_permissions_command = f"sudo chmod -R 775 {path}"

            try:
                result = await run_command(set_permissions_command, log_output=False)
                if result["stderr"]:
                    logger.error("Error setting permissions: %s", result["stderr"])

                response = requests.get(api_url)
                if response.status_code == 200:
                    release_info = response.json()
                    version_tag = release_info.get("tag_name", "unknown")
                    latest_release_url = release_info["zipball_url"]
                    response_zip = requests.get(latest_release_url)
                    if response_zip.status_code == 200:
                        with zipfile.ZipFile(io.BytesIO(response_zip.content)) as z:
                            z.extractall()
                        extracted_items = os.listdir(path)
                        matching_folders = [
                            item for item in extracted_items
                            if item.startswith("bluewave-studio-openauto-pro-api")
                        ]
                        if matching_folders:
                            source_folder = matching_folders[0]
                            # Remove target folder if it is already there
                            for folder in ("assets", "common"):
                                full_path = os.path.join(path, folder)
                                if os.path.exists(full_path):
                                    logger.warning(f"⚠️ Removing existing folder: {full_path}")
                                    shutil.rmtree(full_path)
                            shutil.move(f"{path}/{source_folder}/api_examples/python/common", path)
                            shutil.move(f"{path}/{source_folder}/api_examples/python/assets", path)
                            shutil.rmtree(f"{path}/{source_folder}")
                            logger.info(
                                f"Latest release (version {version_tag}) successfully downloaded and installed.")
                            print()
                            print()
                            os.execv(sys.executable, ['python3'] + sys.argv)
                        else:
                            logger.error("Could not find expected folder in extracted files.")
                    else:
                        logger.error("Failed to download ZIP archive.")
                else:
                    logger.error("Failed to fetch release info from GitHub API.")
            except Exception as e:
                await handle_exception(e, "An error occurred while handling OAP API files.")
        except ImportError as error:
            await handle_exception(error, "OAP import failed – disabling OAP features.")
            # Disable OAP features to continue running
            send_oap_api_mediadata_to_dashboard = False
            change_dark_mode_by_car_light = False
            send_to_oap_gauges = False


playing, position, source = None, None, None
oap_api_is_connected = False

event_handler = None  # 🔁 define globally, as high up in the script as possible

async def oap_api_con(event: asyncio.Event = None):
    global client, oap_api_is_connected, stop_flag, event_handler

    import struct
    reconnect_delay = 10  # Wait seconds on connection errors

    while not stop_flag:
        try:
            client = Client("media data example")

            # ⏬ Create EventHandler-Instanz globally
            event_handler = EventHandler(client, asyncio.get_running_loop())
            client.set_event_handler(event_handler)

            await asyncio.get_running_loop().run_in_executor(None, client.connect, '127.0.0.1', 44405)

            logger.info("")
            logger.info("Successfully connected to OpenAuto Pro API.")
            await asyncio.sleep(0.5)
            oap_api_is_connected = True

            if event:
                event.set()

            # 📨 Message loop with timeout & error handling
            while not stop_flag and oap_api_is_connected:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, client.wait_for_message)

                try:
                    await asyncio.wait_for(future, timeout=2.0)
                except asyncio.TimeoutError:
                    continue  # check regularly if stop_flag is set
                except Exception as e:
                    logger.warning(f"⚠️ wait_for_message error: {e}")
                    break

        except asyncio.CancelledError:
            logger.info("❗ oap_api_con wurde abgebrochen.")
            break

        except struct.error as se:
            logger.error(f"struct.error in receive: {se}")
            oap_api_is_connected = False
            if event_handler:
                event_handler.read_cpu_task.cancel()
            await asyncio.sleep(reconnect_delay)

        except ConnectionRefusedError:
            logger.warning("OpenAuto Pro API is not running or unreachable.")
            if not can_functional:
                stop_flag = True
                logger.warning("No CAN-BUS and no OAP-API available. Stopping script...")
                asyncio.create_task(stop_script())
            else:
                oap_api_is_connected = False
                await asyncio.sleep(reconnect_delay)

        except Exception as e:
            logger.error("Unexpected error in oap_api_con", exc_info=True)
            oap_api_is_connected = False
            await asyncio.sleep(reconnect_delay)

        finally:
            if oap_api_is_connected:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, client.disconnect)
                    logger.info("")
                    logger.info("Successfully disconnected from OAP API.")
                    logger.info("")
                except Exception as e:
                    logger.warning("Error while disconnecting from OAP API", exc_info=True)
            oap_api_is_connected = False



async def oap_units_check(temp_unit, speed_unit, lower_speed, upper_speed):
    config_path = "/home/pi/.openauto/config/openauto_obd_gauges.ini"
    if ENABLE_LOGGING:
        logger.info("📄 Checking existence of OBD gauge configuration file...")

    if not await asyncio.to_thread(os.path.exists, config_path):
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

def write_config_file(config, path):
    with open(path, "w") as configfile:
        config.write(configfile)



@handle_errors
async def toggle_camera():
    global reversecamera_by_reversegear, reversecamera_by_down_longpress, camera_active, camera
    if reversecamera_by_reversegear or reversecamera_by_down_longpress:
        try:
            if camera_active:
                await asyncio.sleep(0)  # Optional: Kontextwechsel erzwingen
                camera.start_preview()
            else:
                await asyncio.sleep(0)
                camera.stop_preview()
        except Exception:
            logger.error("Error while toggling the reverse camera's livestream", exc_info=True)
            logger.info("Problem while toggling reverse camera detected - disabling reverse camera feature")
            reversecamera_by_reversegear = False
            reversecamera_by_down_longpress = False


last_msg_635, last_msg_271_2C3, candump_process = '', '', None
last_speed, last_outside_temp, last_rpm, last_coolant = None, None, None, None
rpm_counter, coolant_counter, speed_counter, outside_temp_counter = 0, 0, 0, 0


@handle_errors
async def read_on_canbus(message):
    if script_started:
        canid = message.arbitration_id
        msg = binascii.hexlify(message.data).decode('ascii').upper()
        if ENABLE_LOGGING:
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


measure_done = 0
data = ''
last_data = ''
drop1 = 0
start_time = None


@handle_errors
async def process_canid_351(msg):  # handler as EventHandler-Instance
    global gear, speed_counter, speed, outside_temp_counter, outside_temp, last_speed, last_outside_temp
    global reversecamera_by_reversegear, reversecamera_by_down_longpress, reversecamera_guidelines, overlay
    speed_frequency = 2
    outside_temp_frequency = 10
    global speed_measure_to_api, measure_done, start_time, elapsed_time_formatted, drop1

    if reversecamera_by_reversegear:
        if msg[0:2] == '00' and gear == 1:
            gear = 0
            logger.info("Forward gear is engaged - stopping the reverse camera with a "
                        f"{reversecamera_turn_off_delay}-second delay.")
            await asyncio.sleep(reversecamera_turn_off_delay)
            try:
                if reversecamera_guidelines:
                    camera.remove_overlay(overlay)
                camera.stop_preview()
            except Exception as e:
                logger.error("Error while stopping the reverse camera's livestream.", exc_info=True)
                reversecamera_by_reversegear = False
                reversecamera_by_down_longpress = False
        elif msg[0:2] == '02' and gear == 0:
            gear = 1
            logger.info("Reverse gear engaged - starting the reverse camera")
            try:
                camera.start_preview()
                if reversecamera_guidelines:
                    overlay = await add_overlay_async(camera)
            except Exception as e:
                logger.error("Error while starting the reverse camera's livestream", exc_info=True)
                reversecamera_by_reversegear = False
                reversecamera_by_down_longpress = False
    if 6 in (toggle_fis1, toggle_fis2) or 10 in (toggle_fis1, toggle_fis2) or send_to_oap_gauges:
        speed_counter += 1
        if speed_counter % speed_frequency == 0 or last_speed is None or 10 in (toggle_fis1, toggle_fis2):
            if speed_unit == 'km/h':
                speed = int(int(msg[4:6] + msg[2:4], 16) / 200)
            elif speed_unit == 'mph':
                speed = int(int(msg[4:6] + msg[2:4], 16) / 200 * 0.621371)
            if send_on_canbus and can_functional and send_values_to_dashboard and speed != last_speed:
                if toggle_fis1 == 6 and not show_label and not pause_fis1:
                    data = f'{speed} {speed_unit}'
                    await align_right(FIS1, data)
                if toggle_fis2 == 6 and not pause_fis2:
                    data = f'{speed} {speed_unit}'
                    await align_right(FIS2, data)
                if ENABLE_LOGGING:
                    logger.info("Speed has changed from %s to %s %s", last_speed, speed, speed_unit)
            if send_to_oap_gauges and speed != last_speed and oap_api_is_connected and speed is not None:
                event_handler.update_to_api("getPidValue(0)", speed, "speed")
            last_speed = speed
            if 10 in (toggle_fis1, toggle_fis2):
                if int(speed) > int(lower_speed):
                    if start_time is None:
                        start_time = time.time()
                    if measure_done == 0:
                        elapsed_time = time.time() - start_time
                        data = "{:.1f}".format(elapsed_time).zfill(4) + "0 s"
                        speed_measure_to_api = float(data.split()[0])
                    if measure_done == 1:
                        print_measure_result1 = elapsed_time_formatted.zfill(4)
                        data = f"{print_measure_result1}0 s"
                        speed_measure_to_api = float(print_measure_result1)
                    if int(speed) >= int(upper_speed) and measure_done == 0:
                        measure_done = 1
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        elapsed_time_formatted = "{:.1f}".format(elapsed_time)
                        if export_speed_measurements_to_file:
                            start_time_formatted = datetime.fromtimestamp(start_time).strftime("%H:%M:%S.%f")[
                                                   :-5]  # Removes microseconds (rounded to hundredths)
                            end_time_formatted = datetime.fromtimestamp(end_time).strftime("%H:%M:%S.%f")[:-5]
                            elapsed_time_formatted_print = "{:05.2f}".format(float(elapsed_time_formatted))
                            result_message = "{} - {}-{} {} - {}0 - {}0  : {} seconds\n".format(
                                time.strftime("%d.%m.%Y"),
                                lower_speed,
                                upper_speed,
                                speed_unit,
                                start_time_formatted,
                                end_time_formatted,
                                elapsed_time_formatted_print
                            )
                            with open('speed_measurements.txt', 'a') as file:
                                file.write(result_message)
                        logger.info(
                            f"The time to accelerate from {lower_speed}-{upper_speed} {speed_unit} took {elapsed_time_formatted_print} seconds.")
                        start_time = None
                else:
                    start_time = None
                    measure_done = 0
                    data = '00.00 s'
                    speed_measure_to_api = float(0.00)
                if data != last_data:
                    if 10 in (toggle_fis1, toggle_fis2):
                        drop1 += 1
                        if drop1 == 3 or measure_done or start_time is None:
                            if send_to_oap_gauges:
                                event_handler.update_to_api("getPidValue(7)", speed_measure_to_api, "speed_measure")
                            if toggle_fis1 == 10 and not show_label and not pause_fis1:
                                await align_right(FIS1, data)
                            if toggle_fis2 == 10 and not pause_fis2:
                                await align_right(FIS2, data)
                            drop1 = 0
    if (11 in (toggle_fis1, toggle_fis2) or send_to_oap_gauges) and carmodel == '8E':
        outside_temp_counter += 1
        if outside_temp_counter % outside_temp_frequency == 0 or last_outside_temp is None:
            #sending temp in °C to API (head line, not dashboard) because OAP Settings has a switch for °C and °F
            outside_temp_api = int(int(msg[10:12], 16) / 2 - 50)
            if temp_unit == '°C':
                outside_temp = float(int(msg[10:12], 16) / 2 - 50)
            elif temp_unit == '°F':
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
            if send_to_oap_gauges and outside_temp != last_outside_temp and oap_api_is_connected and outside_temp is not None:
                outside_temp_int = int(outside_temp)
                event_handler.update_to_api("getPidValue(3)", outside_temp_int, "outside_temp")
            last_outside_temp = outside_temp
            if send_to_oap_gauges and carmodel == '8E' and outside_temp is not None:
                event_handler.outside_to_oap_api(outside_temp_api)


@handle_errors
async def process_canid_353_35B(msg):
    global rpm_counter, rpm, coolant_counter, coolant, last_rpm, last_coolant
    rpm_frequency = 2
    coolant_frequency = 10

    if 7 in (toggle_fis1, toggle_fis2) or send_to_oap_gauges:
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
            if send_to_oap_gauges and rpm != last_rpm and oap_api_is_connected:
                event_handler.update_to_api("getPidValue(1)", rpm, "rpm")
            last_rpm = rpm
    if 8 in (toggle_fis1, toggle_fis2) or send_to_oap_gauges:
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
            if send_to_oap_gauges and coolant != last_coolant and oap_api_is_connected:
                event_handler.update_to_api("getPidValue(2)", coolant, "coolant")
            last_coolant = coolant


@handle_errors
async def process_canid_461(msg):
    global up, down, select, back, nextbtn, prev, setup
    global toggle_fis1, toggle_fis2, pause_fis1, pause_fis2, camera_active

    if control_pi_by_rns_e_buttons:
        if msg == '373001004001':
            device.emit(uinput.KEY_1, 1)
            device.emit(uinput.KEY_1, 0)
        elif msg == '373001002001':
            device.emit(uinput.KEY_2, 1)
            device.emit(uinput.KEY_2, 0)
        elif msg == '373001400000':  # RNS-E: up button pressed
            up += 1
        elif msg == '373004400000' and up > 0:  # RNS-E: up button released
            if up <= 4:
                device.emit(uinput.KEY_UP, 1)
                device.emit(uinput.KEY_UP, 0)
                up = 0
            elif up > 4:
                if toggle_values_by_rnse_longpress:
                    if not show_label:
                        up = 0
                        if (send_on_canbus and can_functional) or toggle_fis1 == 0:
                            pause_fis1 = True
                            await block_show_value1()
                            await media_to_dis1()
                else:
                    device.emit(uinput.KEY_P, 1)
                    device.emit(uinput.KEY_P, 0)
            up = 0
        elif msg == '373001800000':  # RNS-E: down button pressed
            down += 1
        elif msg == '373004800000' and down > 0:  # RNS-E: down button released
            if down <= 4:
                device.emit(uinput.KEY_DOWN, 1)
                device.emit(uinput.KEY_DOWN, 0)
                down = 0
            elif 4 < down <= 16:
                if toggle_values_by_rnse_longpress:
                    down = 0
                    if (send_on_canbus and can_functional) or toggle_fis2 == 0:
                        pause_fis2 = True
                        await block_show_value2()
                        await media_to_dis2()
                else:
                    device.emit(uinput.KEY_F2, 1)
                    device.emit(uinput.KEY_F2, 0)
                    down = 0
            elif down > 16:
                if reversecamera_by_down_longpress:
                    camera_active = not camera_active
                    await toggle_camera()
                down = 0
        elif msg == '373001001000':  # RNS-E: wheel pressed
            select += 1
        elif msg == '373004001000' and select > 0:  # RNS-E: wheel released
            if select <= 4:
                device.emit(uinput.KEY_ENTER, 1)
                device.emit(uinput.KEY_ENTER, 0)
                select = 0
            elif select > 4:
                device.emit(uinput.KEY_B, 1)
                device.emit(uinput.KEY_B, 0)
                select = 0
        elif msg == '373001000200':  # RNS-E: return button pressed
            back += 1
        elif msg == '373004000200' and back > 0:  # RNS-E: return button released
            if back <= 4:
                device.emit(uinput.KEY_ESC, 1)
                device.emit(uinput.KEY_ESC, 0)
                back = 0
            elif back > 16:
                await candump()
                back = 0
        elif msg == '373001020000':  # RNS-E: next track button pressed
            nextbtn += 1
        elif msg == '373004020000' and nextbtn > 0:  # RNS-E: next track button released
            if nextbtn <= 4:
                device.emit(uinput.KEY_N, 1)
                device.emit(uinput.KEY_N, 0)
                nextbtn = 0
            elif nextbtn > 4:
                device.emit(uinput.KEY_LEFTCTRL, 1)
                device.emit(uinput.KEY_F3, 1)
                device.emit(uinput.KEY_LEFTCTRL, 0)
                device.emit(uinput.KEY_F3, 0)
                nextbtn = 0
        elif msg == '373001010000':  # RNS-E: previous track button pressed
            prev += 1
        elif msg == '373004010000' and prev > 0:  # RNS-E: previous track button released
            if prev <= 4:
                device.emit(uinput.KEY_V, 1)
                device.emit(uinput.KEY_V, 0)
                prev = 0
            elif prev > 4:
                device.emit(uinput.KEY_F12, 1)
                device.emit(uinput.KEY_F12, 0)
                prev = 0
        elif msg == '373001000100':  # RNS-E: setup button pressed
            setup += 1
        elif msg == '373004000100' and setup > 0:  # RNS-E: setup button released
            if setup <= 16:
                device.emit(uinput.KEY_M, 1)
                device.emit(uinput.KEY_M, 0)
                setup = 0
            elif setup > 16:
                command = 'sudo shutdown -h now'
                result = await run_command("sudo shutdown -h now", log_output=False)
                if result["stderr"]:
                    logger.error(f"Failed to shutdown Raspberry Pi: {result['stderr']}")
                setup = 0


@handle_errors
async def process_canid_5C3(msg):
    global press_mfsw, nextbtn, prev

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

    # instantly set on first received message
    first_message = last_msg_635 is None

    if first_message or msg != last_msg_635:
        light = int(msg[2:4], 16)

        new_light_status = 1 if light > 0 else 0

        if first_message or (new_light_status != light_status):
            if change_dark_mode_by_car_light and oap_api_is_connected:
                mode = 'night' if new_light_status == 1 else 'day'
                logger.info(f"Light status changed: Setting {mode} mode immediately.")
                event_handler.send_day_night(mode)

        light_status = new_light_status
        last_msg_635 = msg


@handle_errors
async def process_canid_65F(msg):
    global car_model_set, carmodel, FIS1, FIS2

    if msg[0:2] == '01' and car_model_set is None:
        carmodel = bytes.fromhex(msg[8:12]).decode()
        carmodelyear = await translate_caryear(bytes.fromhex(msg[14:16]).decode())
        car_models = {
            '8E': ('Audi A4', '265', '267'),
            '8J': ('Audi TT', '667', '66B'),
            '8L': ('Audi A3', '667', '66B'),
            '8P': ('Audi A3', '667', '66B'),
            '42': ('Audi R8', '265', '267'),
        }
        model_info = car_models.get(carmodel[0:2], ('Unbekanntes Modell', 'Unbekannt', 'Unbekannt'))
        carmodelfull, FIS1, FIS2 = model_info
        logger.info('The car model and car model year were successfully read from the CAN-Bus.')
        logger.info('CAR = %s %s %s', carmodelfull, carmodel, carmodelyear)
        logger.info('FIS1 = %s / FIS2 = %s', FIS1, FIS2)
        car_model_set = True


@handle_errors
async def process_canid_661(msg):
    global tv_mode_active, send_on_canbus, deactivate_overwrite_dis_content
    if msg in ['8101123700000000', '8301123700000000']:
        if tv_mode_active == 0:
            device.emit(uinput.KEY_X, 1)
            device.emit(uinput.KEY_X, 0)
            logger.info('RNS-E is (back) in TV mode - play media - Keyboard: "X" - OpenAuto: "play"')
            tv_mode_active = 1
            if only_send_if_radio_is_in_tv_mode:
                send_on_canbus = True
                deactivate_overwrite_dis_content = False
    else:
        if tv_mode_active == 1:
            device.emit(uinput.KEY_C, 1)
            device.emit(uinput.KEY_C, 0)
            logger.info('RNS-E is not in TV mode (anymore) - pause media - Keyboard: "C" - OpenAuto: "pause"')
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


asyncio.run(toggle_fis2_label())


@handle_errors
async def candump():
    global candump_process, pause_fis1, pause_fis2

    if candump_process:
        logger.info("Stopping candump now")
        result = await asyncio.create_subprocess_exec(
            'sudo', 'pkill', 'candump', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        if stderr:
            logger.error("Failed to stop candump: %s", stderr.decode())
        else:
            candump_process = False
            if send_on_canbus and can_functional:
                pause_fis1 = True
                pause_fis2 = True
                await clear_content(FIS1)
                await clear_content(FIS2)
                data1 = 'CANDUMP'
                data2 = 'STOP'
                await align_center(FIS1, data1)
                await align_center(FIS2, data2)
                await asyncio.sleep(2)  # Non-blocking delay
                await clear_content(FIS1)
                await clear_content(FIS2)
                pause_fis1 = False
                pause_fis2 = False
    else:
        logger.info("Starting candump now")
        candump_process = True
        now = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        log_file_path = f"{path}/candumps/{now}-candump-{can_interface}.txt"
        if not os.path.exists(f"{path}/candumps/"):
            os.makedirs(f"{path}/candumps/")
        command = ['sudo', 'candump', can_interface, '-tA']
        try:
            with open(log_file_path, 'w') as log_file:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=log_file,  # Redirect stdout to the file
                    stderr=log_file  # Redirection of stderr to the same file
                )
                await process.communicate()
            if process.returncode == 0:
                logger.info(f'candump output successfully written to {log_file_path}')
            else:
                logger.error(f'Failed to start candump with return code: {process.returncode}')
        except Exception as e:
            logger.error('Unexpected error while starting candump: %s', str(e))
        if send_on_canbus and can_functional:
            pause_fis1 = True
            pause_fis2 = True
            await clear_content(FIS1)
            await clear_content(FIS2)
            data1 = 'CANDUMP'
            data2 = 'START'
            await align_center(FIS1, data1)
            await align_center(FIS2, data2)
            await asyncio.sleep(3)  # Non-blocking delay
            await clear_content(FIS1)
            await clear_content(FIS2)
            pause_fis1 = False
            pause_fis2 = False


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
            if (not send_oap_api_mediadata_to_dashboard and toggle_fis in (1, 2, 3, 4, 5)) or (
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
                        if send_to_oap_gauges:
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
remote_server_shutdown_event = asyncio.Event()

async def stop_script():
    global stop_script_running, oap_api_is_connected, stop_flag, event_handler, server_socket, remote_task, remote_server_shutdown_event

    if stop_script_running:
        return
    stop_script_running = True
    try:
        logger.info("Stopping script...")

        if ENABLE_LOGGING:
            for task in asyncio.all_tasks():
                logger.info(f"Found running task: {task.get_coro().__name__}")

        # 🛑 first set stop_flag
        if ENABLE_LOGGING:
            logger.info("setting stop_flag and wait 2 seconfs to gently close running tasks")
        stop_flag = True
        await asyncio.sleep(2)

        if ENABLE_LOGGING:
            for task in asyncio.all_tasks():
                logger.info(f"Found running task: {task.get_coro().__name__}")

        if remote_task:
            logger.info("🚦 Triggering shutdown of remote_control")
            remote_server_shutdown_event.set()

            try:
                await asyncio.wait_for(remote_task, timeout=5)
                logger.info("✅ remote_control_task stopped cleanly.")
            except asyncio.TimeoutError:
                logger.warning("⚠️ remote_control_task did not stop in time.")
            except asyncio.CancelledError:
                logger.warning("⚠️ remote_control_task was force cancelled.")


        # 🎯 gathering all relevant tasks
        tasks = [task for task in asyncio.all_tasks() if
                 task is not asyncio.current_task() and
                 task._coro.__name__ in ["read_cpu", "receive_messages", "handle_client", "remote_control"]]

        if tasks:
            for task in tasks:
                task.cancel()
                try:
                    await task
                    logger.info(f"✅ Task {task.get_coro().__name__} was stopped.")
                except asyncio.CancelledError:
                    logger.info(f"✋ Task {task.get_coro().__name__} was cancelled.")
                except Exception as e:
                    logger.error(f"❌ Error while waiting for task {task.get_coro().__name__}: {e}")
        else:
            logger.info("ℹ️ No matching tasks were running or all already stopped.")

        # 🔌 diconnect OAP API
        if oap_api_is_connected:
            try:
                await asyncio.get_event_loop().run_in_executor(None, client.disconnect)
                oap_api_is_connected = False
                logger.info("Successfully disconnected from OAP API.")
            except Exception as e:
                logger.error("Error while disconnecting from OAP API.", exc_info=True)
    except asyncio.CancelledError:
        logger.info("Task stop_script was cancelled.")


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
        #general startup tasks:

        #write header to console
        await write_header_to_log_file(log_filename)
        #check if script is running with python3.13.5. If not, start install_python_3_13_5(), set_python3_alias(), restart_with_python_3_13_5()
        await ensure_python_3_13_5()
        #console output "Script is starting..."
        await script_starting()
        #check if the script is already running. If so, kill the old istance(s)
        await is_script_running(script_filename)
        #check importlib.metadata import with fallback
        version, PackageNotFoundError = await ensure_importlib_metadata(logger)
        #check and output python3 version and interpreter
        await check_python()
        #check for missing python3 (pip) packages. If missing, install them.
        installed_modules, missing_modules = await python_modules()
        await install_missing(missing_modules)
        #import the installed python3 modules. If not installed there is no import)
        import_modules(installed_modules)
        #check if can-utils ist already installed. If installed, check the version. If outdated remove and instal vi git.
        await check_can_utils()
        #check the uinput permissions, so that simulated keyboards controlls can work (Left, Right, Enter...)
        await uinput_permissions()
        #check the cpu powerplan and set to "ondemand" if it's in powersave mode
        await set_powerplan()
        #check if OpenAuto Pro API files are already downloaded. If not, download, extract and import.
        await check_import_oap_api()
        #check if the units (km/h / rpm and °C / °F) are set correctly in openauto_obd_gauges.ini for OAP Dashboard view.
        await oap_units_check(temp_unit, speed_unit, lower_speed, upper_speed)
        #start the remote control task to shutdown other running scripts on startup or via network controll website
        remote_task = asyncio.create_task(remote_control(), name="remote_control")
        tasks.append(remote_task)
        #test the can-interface and see if can-messages are getting received
        await test_can_interface()
        if send_on_canbus and can_functional:
            await welcome_message()

        script_started = True

        # conditional tasks
        #start oap api connection if oap api features are enabled
        if send_to_oap_gauges or (
                (send_oap_api_mediadata_to_dashboard or change_dark_mode_by_car_light) and bus is not None
        ):
            tasks.append(asyncio.create_task(oap_api_con(), name="receive_messages"))
        #if can-bus is runnung, start reading messages
        if bus and can_functional:
            tasks.append(asyncio.create_task(get_can_messages(), name="get_can_messages"))
            #initialise reversecamera if user has enabled this feature
            if reversecamera_by_reversegear or reversecamera_by_down_longpress:
                camera = await initialize_reverse_camera()
            #activate rns-e tv input if user has enabled this feature
            if activate_rnse_tv_input and send_on_canbus:
                tasks.append(asyncio.create_task(send_tv_input(), name="send_tv_input"))
            #if user has allowed sending on can-bus, check wich features are enabled
            if send_on_canbus:
                #if user has allowed sending values (speed, rpm,...) or sending mediadata (title, artist,...) to dashboard, start the function to overwrite dis/fis via hands free channel
                if send_values_to_dashboard or send_oap_api_mediadata_to_dashboard:
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
