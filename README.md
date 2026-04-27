# Usage Guide for the project

Here we have built an agent, **complete automated pipeline** for deploying machine learning models to ESP32 microcontrollers:

```
CSV Data → Train → Quantize → Generate C Code → Simulate/Test → Deploy
```


## Prerequisites

- Windows 10/11 (instructions below are Windows‑oriented)
- Python 3.9+ installed and on PATH
- Git installed
- ESP32‑S3 development board
- ESP‑IDF (Espressif IoT Development Framework) installed for building and flashing firmware ((https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html))


## Clone the repository
```bash
git clone https://github.com/dipanshuthakurhp/ppg_to_ecg_agent.git
cd ppg_to_ecg_agent
```

## Activate the environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

## Install the dependencies
```bash
pip install -r requirements.txt
```

## Run the agent pipeline
```bash 
 python agent_cli.py test_pipeline --esp-project "D:\esp32_firmware" --csv-path "D:\physiofusion_agent_v2\data\ppg\ppg_2.csv" --col A1
```
## Prepare ESP32 firmware
Copy the esp_firmware folder from this repository to the D: drive and rename it if needed:
```bash
cd "D:\esp32_firmware"
```
Make sure ESP-IDF is installed on your device, and run:
```bash
idf.py fullclean
idf.py set-target esp32s3
idf.py build
idf.py flash
idf.py monitor
```

## Prepare the PPG-ECG dataset
Run the following command to prepare the segmennts. Make sure you have clear_segments.py in your repo. 
```bash
python create_ecg_ppg_segments.py
```

## Obtain the model specifications
To obtain the model specifications like architecture, size, number of paramaters, run:
```bash
python .\model_specifications.py    
```
