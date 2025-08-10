# read_from_canbus.py

## Disclaimer
I am not a professional programmer – most of my knowledge is self-taught. I designed this script so that almost all functions can be activated or deactivated individually. My goal was to make the script install all missing components (e.g., Python modules) automatically so minimal prior knowledge is required.  

This script is the result of years of work and runs very well in my own setup. However, I can only test it in my own car. In theory, it should work with other models as well. For this reason, I have created a table of compatible and tested models. Feedback after testing is highly appreciated so we can keep this table updated! 

---

## Usage Context
I use this script with a **Raspberry Pi 4** connected to an **RNS-E** in an **Audi A4 B6 (8E)**.  
It controls **OpenAuto Pro** (which is unfortunately discontinued).  
The OpenAuto Pro features in the script can be disabled, so the script will also run without it.

---

## Main Features



### 1. Driver Information System (DIS/FIS) Text Output
The script can write to the first two lines of the **DIS/FIS** (Driver Information System) in the instrument cluster.  
It can overwrite existing text (e.g., from the radio) via the telephone channel.

![Image](https://github.com/user-attachments/assets/f16e3018-3c32-4819-99f9-9a51ef2f099c) &nbsp;&nbsp;&nbsp; ![Image](https://github.com/user-attachments/assets/abfc84b0-341d-49f7-b662-1df58eaa0d3d)

| **CAN Bus Data (Infotainment Bus)** | **OpenAuto Pro Media Info** |
|-------------------------------------|-----------------------------|
| - Speed<br>- RPM<br>- Coolant temperature (A4 8E only)<br>- Outside temperature<br>- CPU usage & temperature<br>- Custom speed measurement value<br>- Blank line (no content) | - Title<br>- Artist<br>- Album<br>- Song position<br>- Song duration <br> <br> <br> |


**Alternative mode:**  
It can also display only a single value in the FIS/DIS with a custom title.

---

### 2. Additional Functions

- Automatically install all required packages on first start  
  - Including PiCan2 and PiCan3 setup  
- Enable/disable all features individually  
- Control the Raspberry Pi using RNS-E radio buttons ([read_from_canbus_keymap.pdf](read_from_canbus_keymap.pdf))

**Long press button support:**  
  - Long press UP/DOWN to cycle displayed FIS/DIS values  
  - Very long press RETURN to start/stop `candump`
  - Extreme long press SETUP (about 5s) to shutdown the raspberry pi   
  - More functions in the keymap file  [read_from_canbus_keymap.pdf](read_from_canbus_keymap.pdf)

**MFSW (multi function steering wheel) support:**  
- The script does support MFSW support but:
  - MFSW is most likely in conflict with hands free control
  - I recommend to uncode and/or remove hands free hardware to use MFSW with this script
  - Otherwise you will controll the script and hands free at the same time

**Reverse camera support:**  
  - Activate PiCamera when reverse gear is detected via CAN Bus  
  - Optional guidelines overlay  
  - Optionally toggle the camera with a very long press of the DOWN button  

**Display settings:**  
  - Choose between scrolling text or OEM-style 3-second text paging  

**Light-based theme switching:**  
  - Switch OpenAuto Pro / Android Auto between day/night modes based on vehicle lights  

**Speed measurement tool:**  
  - 0.1 second precision  
  - Adjustable measurement range (e.g., 0–100 km/h or 100–200 km/h)  
  - Export results to a file  

**System integration:**  
  - Shut down the Raspberry Pi when ignition is turned off or the key is removed  
  - Switch between metric and imperial units (km/h ↔ mph, °C ↔ °F)  
  - Debug logging for troubleshooting  

**RNS-E TV mode control:**  
  - Change input format (PAL/NTSC)  
  - Useful if using a custom video source without an IMA  
  - Recommended: use [this firmware](https://rnse.pcbbc.co.uk/index.php) for a permanent TV input unlock

---

### 3. OpenAuto Pro Features

- Display media information in FIS/DIS  
- Change day/night mode based on car lighting  
- Send CAN Bus data to the OpenAuto Pro API for dashboard display  

<img width="400" height="240" alt="Image" src="https://github.com/user-attachments/assets/01535419-f95a-4656-b91c-9c4fa2c4af94" />  &nbsp;&nbsp;&nbsp; <img width="400" height="240" alt="Image" src="https://github.com/user-attachments/assets/4f004626-ab8b-4a4a-9990-ed9ffc31d536" />

---

### 4. Compabillity

| Model        | DIS/FIS Output | MFSW | Note |
|--------------|----------------|-------|------|
| Audi A4 8E   | ✅              | ⚠️     |      |
| Audi A3 8L   | ⚠️              | ⚠️     |      |
| Audi A3 8P   | ⚠️              | ⚠️     |      |
| Audi TT 8J   | ⚠️              | ⚠️     |      |
| Audi R8 42   | ⚠️              | ⚠️     |      |

**Legend:**  
✅ = Tested and working  
⚠️ = Not yet tested / uncertain 

Note: The cars in the list should work based on candump analysis from this cars. It could be work on more cars, if the can messages are the same.

---

### My Setup

- Audi A4 B6 (8E) Avant 2001  
- Seat Exeo RNS-E 3R0 035 192  
- Raspberry Pi 4  
- PiCan 3  
- CarlinKit 2022 CPC200-AutoKit (Apple CarPlay)  
- VGA to RGB+Sync Converter – soldering required ([PCBWay Project](https://www.pcbway.com/project/shareproject/VGA_to_RGB_Sync_Converter_f202899d.html))  
- Twozoh Micro HDMI to VGA cable ([Amazon link](https://www.amazon.de/dp/B0CC9CVRDV))  
