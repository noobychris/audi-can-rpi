## Optional debugging tool for bench testing with virtual CAN (`vcan0`)

The `tools` directory contains an optional graphical CAN debugging utility:

```text
/home/pi/tools/
├── debugging_script.py
├── fis.png
└── rns-e.png
```

> [!NOTE]
> The debugging tool requires Raspberry Pi OS with Desktop or another active graphical session. A running X server is required.
>
> The tool uses the virtual SocketCAN interface `vcan0` and creates and activates it automatically if it does not already exist.


The script uses `fis.png` and `rns-e.png` from the same directory. It is not required for normal operation.

![Debugging script](docs/screenshots/debugging_script.png)

---
