# Audi CAN Bus to DIS/FIS via Raspberry Pi (with OpenAuto Pro support)

This project/script is designed to display CAN bus data (speed, RPM, etc.) on the DIS/FIS using a Raspberry Pi. It also allows controlling the Raspberry Pi using the RNS-E buttons in TV mode. It also includes functions for OpenAuto Pro, which has unfortunately been discontinued.

> **ℹ️ Note:**
> I'm not a professional developer – most of my knowledge is self-taught. I designed this script so that almost all functions can be activated or deactivated individually. My goal was to make the script install all missing components (e.g., Python modules) automatically so minimal prior knowledge is required.
> 
> This script is the result of years of work and runs very well in my own setup. However, I can only test it in my own car. In theory, it should work with other models as well. For this reason, I have created a table of compatible and tested models. Feedback after testing is highly appreciated so we can keep this table updated!

I use this script with a **Raspberry Pi 4** connected to an **RNS-E** in an **Audi A4 B6 (8E)**. It controls **OpenAuto Pro**. The OpenAuto Pro features in the script can be disabled, so the script will also run without it.

---

## To-do list

1. check compability with hudiy (https://hudiy.eu)
2. output content to middle dis/fis section like fis control does (https://fis-control.de/index_de.html)

---

## Installation Guide

1. Copy `read_from_canbus.py` to your Raspberry Pi, preferably in a new folder called `scripts`.

2. If you are using the reverse camera feature, also copy the file `lines.png` to the same folder.

3. Make the script executable:
   ```bash
   sudo chmod +x read_from_canbus.py
   ```

4. The files in the `.openauto/config` folder are only required if you want to send data like **Speed** or **RPM** to the OpenAuto Pro dashboard interface.

5. The first start of the script should be done **with an active internet connection** so that all required Python modules and dependencies can be installed automatically.

---


## Main Features


### 1. Driver Information System (DIS/FIS) Text Output
The script can write to the first two lines of the **DIS/FIS** (Driver Information System) in the instrument cluster. It can overwrite existing texts (e.g., from the radio or ima modules) via the telephone channel.

![Image](https://github.com/user-attachments/assets/f16e3018-3c32-4819-99f9-9a51ef2f099c) &nbsp;&nbsp;&nbsp; ![Image](https://github.com/user-attachments/assets/abfc84b0-341d-49f7-b662-1df58eaa0d3d)

| **CAN Bus Data (Infotainment Bus)** | **OpenAuto Pro Media Info** |
|-------------------------------------|-----------------------------|
| - Speed<br>- RPM<br>- Coolant temperature (A4 8E only)<br>- Outside temperature<br>- CPU usage & temperature<br>- Custom speed measurement value<br>- Blank line (no content) | - Title<br>- Artist<br>- Album<br>- Song position<br>- Song duration <br> <br> <br> |


**Alternative mode:**  
It can also display only a single value in the FIS/DIS with a custom title.

---

### 2. Additional Functions

- **Auto-Setup** – Installs all required packages on first start (incl. PiCan2/PiCan3).
- **Feature Control** – Enable/disable all features individually.
- **RNS-E Button Control** ([read_from_canbus_keymap.pdf](read_from_canbus_keymap.pdf))
  - Long press **UP/DOWN**: Cycle displayed FIS/DIS values
  - Very long press **RETURN**: Start/stop `candump`
  - Extreme long press **SETUP** (~5s): Shutdown Raspberry Pi
  - More functions in keymap file
- **MFSW (Multi Function Steering Wheel)** – Supported, but may conflict with hands-free control.  
  → Recommended: disable/remove hands-free hardware to avoid conflicts.
- **Reverse Camera**
  - Activate PiCamera when reverse gear detected via CAN Bus
  - Optional guidelines overlay
  - Toggle camera via very long press **DOWN**
- **Display Options** – Scrolling text or OEM-style 3s paging.
- **Theme Switching** – Day/night mode changes based on vehicle lights.
- **Speed Measurement**
  - Precision: 0.1s
  - Adjustable range (e.g., 0–100, 100–200 km/h)
  - Export results to file
- **System Integration**
  - Shutdown Pi on ignition off/key removal
  - Switch metric/imperial units (km/h ↔ mph, °C ↔ °F)
  - Debug logging for troubleshooting
- **RNS-E TV Mode Control**
  - Switch tv input activation format (PAL/NTSC)
  - Useful for custom video without IMA
  - Recommended firmware: [link](https://rnse.pcbbc.co.uk/index.php)
---

### 3. OpenAuto Pro Features

- Display media information in FIS/DIS  
- Change day/night mode based on car lighting  
- Send CAN Bus data to the OpenAuto Pro API for dashboard display  

<img width="400" height="240" alt="Image" src="https://github.com/user-attachments/assets/01535419-f95a-4656-b91c-9c4fa2c4af94" />  &nbsp;&nbsp;&nbsp; <img width="400" height="240" alt="Image" src="https://github.com/user-attachments/assets/4f004626-ab8b-4a4a-9990-ed9ffc31d536" />

---

### 4. Compabillity &nbsp;&nbsp;&nbsp; [![Report Compatibility](https://img.shields.io/badge/🚗%20Report%20Compatibility-orange)](https://github.com/noobychris/audi-can-rpi/issues/new?labels=compatibility&template=report_compatibility.yml)

| Model        | DIS/FIS Output | MFSW | Outside Temp | Note |
|--------------|----------------|------|--------------|------|
| Audi A4 8E   | ✅              | ⚠️    | ✅           |      |
| Audi A3 8L   | ⚠️              | ⚠️    | ⚠️           |      |
| Audi A3 8P   | ⚠️              | ⚠️    | ⚠️           |      |
| Audi TT 8J   | ⚠️              | ⚠️    | ⚠️           |      |
| Audi R8 42   | ⚠️              | ⚠️    | ⚠️           |      |

**Legend:**  
✅ = Tested and working  
⚠️ = Not yet tested / uncertain 

Note: The cars in the list should work based on candump analysis from this cars. It could work in more cars, if the can messages are the same.

---

## 📨 Feedback & Support

Found a bug? Have an idea or question?  
Your feedback helps improve this project!

[![Report Bug](https://img.shields.io/badge/🐞%20Report%20Bug-red)](https://github.com/noobychris/audi-can-rpi/issues/new?template=bug_report.yml&labels=bug)
[![Request Feature](https://img.shields.io/badge/💡%20Request%20Feature-blue)](https://github.com/noobychris/audi-can-rpi/issues/new?template=feature_request.yml&labels=enhancement)
[![Report Compatibility](https://img.shields.io/badge/🚗%20Report%20Compatibility-orange)](https://github.com/noobychris/audi-can-rpi/issues/new?labels=compatibility&template=report_compatibility.yml)
[![Ask a Question](https://img.shields.io/badge/❓%20Ask%20a%20Question-purple)](https://github.com/noobychris/audi-can-rpi/issues/new?template=question.yml&labels=question)
[![Join Discussions](https://img.shields.io/badge/💬%20Join%20Discussions-green)](https://github.com/noobychris/audi-can-rpi/discussions)

- **🐞 Report a bug** – Use the [Bug Report form](https://github.com/noobychris/audi-can-rpi/issues/new?template=bug_report.yml&labels=bug) to help us fix it quickly.
- **💡 Request a feature** – Share your idea via the [Feature Request form](https://github.com/noobychris/audi-can-rpi/issues/new?template=feature_request.yml&labels=enhancement).
- **🚗 Report compatibility** –  Report a (partially) compatible model via the [Compatibility-Formular](https://github.com/noobychris/audi-can-rpi/issues/new?labels=compatibility&template=report_compatibility.yml).
- **❓ Ask a question** – Use the [Question form](https://github.com/noobychris/audi-can-rpi/issues/new?template=question.yml&labels=question) for setup help, usage tips, or troubleshooting advice.
- **💬 Join discussions** – Ask broader questions, share test results, or brainstorm in the [Discussions area](https://github.com/noobychris/audi-can-rpi/discussions).

---

### My Setup

- Audi A4 B6 (8E) Avant 2001  
- Seat Exeo RNS-E 3R0 035 192  
- Raspberry Pi 4  
- PiCan 3  
- CarlinKit 2022 CPC200-AutoKit (Apple CarPlay)  
- VGA to RGB+Sync Converter – soldering required ([PCBWay Project](https://www.pcbway.com/project/shareproject/VGA_to_RGB_Sync_Converter_f202899d.html))  
- Twozoh Micro HDMI to VGA cable ([Amazon link](https://www.amazon.de/dp/B0CC9CVRDV))  
