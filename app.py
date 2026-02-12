[     UTC     ] Logs for drone-detection-yolov11-psau.streamlit.app/

────────────────────────────────────────────────────────────────────────────────────────

[18:27:48] 🚀 Starting up repository: 'drone-detection-yolov11', branch: 'main', main module: 'app.py'

[18:27:48] 🐙 Cloning repository...

[18:27:50] 🐙 Cloning into '/mount/src/drone-detection-yolov11'...

[18:27:50] 🐙 Cloned repository!

[18:27:50] 🐙 Pulling code changes from Github...

[18:27:50] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.11.14 environment at /home/adminuser/venv

Resolved 82 packages in 883ms

Prepared 82 packages in 52.66s

Installed 82 packages in 600ms

 + altair==5.5.0

 + attrs==25.4.0

 + blinker==1.9.0

 + cachetools==5.5.2

 + certifi==2026.1.4

 + charset-normalizer==3.4.4

 +[2026-02-12 18:28:45.249695]  click==8.3.1

 + contourpy==1.3.3

 + cuda-bindings==12.9.4

 + cuda-pathfinder==1.3.4

 [2026-02-12 18:28:45.249914] + cycler==0.12.1

 + filelock==3.21.0

 + fonttools==4.61.1

 + fsspec==2026.2.0

 + gitdb==4.0.12

 + gitpython==3.1.46

 + idna==3.11

 + imageio-ffmpeg==0.6.0

 + jinja2==3.1.6

 + jsonschema==4.26.0

 + jsonschema-specifications==2025.9.1

 +[2026-02-12 18:28:45.250352]  kiwisolver==1.4.9

 + markdown-it-py==4.0.0

 + markupsafe==3.0.3

 + matplotlib==3.10.8

 + mdurl==0.1.2

 + mpmath==1.3.0

 + narwhals==2.16.0

 + networkx==3.6.1

 + numpy==1.26.4

 + nvidia-cublas-cu12==12.8.4.1

 + nvidia-cuda-cupti-cu12==12.8.90

 + nvidia-cuda-nvrtc-cu12==12.8.93[2026-02-12 18:28:45.250778] 

 + nvidia-cuda-runtime-cu12==12.8.90

 + nvidia-cudnn-cu12==9.10.2.21

 + nvidia-cufft-cu12==11.3.3.83

 + nvidia-cufile-cu12==1.13.1.3

 + nvidia-curand-cu12==10.3.9.90

 + nvidia-cusolver-cu12==11.7.3.90

 + nvidia-cusparse-cu12==12.5.8.93

 + nvidia-cusparselt-cu12==0.7.1

 + nvidia-nccl-cu12==2.27.5

 + nvidia-nvjitlink-cu12==[2026-02-12 18:28:45.251089] 12.8.93

 + nvidia-nvshmem-cu12==3.4.5

 + nvidia-nvtx-cu12==12.8.90

 + opencv-python==4.11.0.86

 + opencv-python-headless==[2026-02-12 18:28:45.251328] 4.10.0.84

 + packaging==24.2

 + pandas==2.3.3

 + pillow==11.3.0

 + polars==1.38.1

 + polars-runtime-32==1.38.1

 + protobuf==5.29.6

 + psutil==7.2.2

 + pyarrow==23.0.0

 + pydeck==0.9.1

 + pygments==2.19.2[2026-02-12 18:28:45.251528] 

 + pyparsing==3.3.2

 + python-dateutil==2.9.0.post0

 + pytz==2025.2

 + pyyaml==6.0.3

 +[2026-02-12 18:28:45.251729]  referencing==0.37.0

 + requests==2.32.5

 + rich==13.9.4

 + rpds-py==0.30.0

 + scipy==1.17.0

 [2026-02-12 18:28:45.251837] + six==1.17.0

 + smmap==5.0.2

 + streamlit==1.42.0

 + sympy==1.14.0

 + tenacity==9.1.4[2026-02-12 18:28:45.251940] 

 + toml==0.10.2

 + torch==2.10.0

 + torchvision==0.25.0

 + tornado==6.5.4

 +[2026-02-12 18:28:45.252082]  triton==3.6.0

 + typing-extensions==4.15.0

 + tzdata==2025.3

 + ultralytics==8.4.11

 +[2026-02-12 18:28:45.252275]  ultralytics-thop==2.0.18

 + urllib3==2.6.3

 + watchdog==6.0.0

Checking if Streamlit is installed

Found Streamlit version 1.42.0 in the environment

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.11.14 environment at /home/adminuser/venv

Audited 1 package in 4ms


────────────────────────────────────────────────────────────────────────────────────────


[18:28:47] 🐍 Python dependencies were installed from /mount/src/drone-detection-yolov11/requirements.txt using uv.

Check if streamlit is installed

Streamlit is already installed

[18:28:48] 📦 Processed dependencies!




────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:121 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:591 in code_to_exec                                     

                                                                                

  /mount/src/drone-detection-yolov11/app.py:6 in <module>                       

                                                                                

      3 import subprocess                                                       

      4                                                                         

      5 import streamlit as st                                                  

  ❱   6 from ultralytics import YOLO                                            

      7 import cv2                                                              

      8 import imageio_ffmpeg                                                   

      9                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/ultralytics/__init__.py:13  

  in <module>                                                                   

                                                                                

    10 if not os.environ.get("OMP_NUM_THREADS"):                                

    11 │   os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU util  

    12                                                                          

  ❱ 13 from ultralytics.utils import ASSETS, SETTINGS                           

    14 from ultralytics.utils.checks import check_yolo as checks                

    15 from ultralytics.utils.downloads import download                         

    16                                                                          

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/ultralytics/utils/__init__  

  .py:24 in <module>                                                            

                                                                                

      21 from types import SimpleNamespace                                      

      22 from urllib.parse import unquote                                       

      23                                                                        

  ❱   24 import cv2                                                             

      25 import numpy as np                                                     

      26 import torch                                                           

      27                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/cv2/__init__.py:181 in      

  <module>                                                                      

                                                                                

    178 │   if DEBUG: print('OpenCV loader: DONE')                              

    179                                                                         

    180                                                                         

  ❱ 181 bootstrap()                                                             

    182                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/cv2/__init__.py:153 in      

  bootstrap                                                                     

                                                                                

    150 │                                                                       

    151 │   py_module = sys.modules.pop("cv2")                                  

    152 │                                                                       

  ❱ 153 │   native_module = importlib.import_module("cv2")                      

    154 │                                                                       

    155 │   sys.modules["cv2"] = py_module                                      

    156 │   setattr(py_module, "_native", native_module)                        

                                                                                

  /usr/local/lib/python3.11/importlib/__init__.py:126 in import_module          

                                                                                

    123 │   │   │   if character != '.':                                        

    124 │   │   │   │   break                                                   

    125 │   │   │   level += 1                                                  

  ❱ 126 │   return _bootstrap._gcd_import(name[level:], package, level)         

    127                                                                         

    128                                                                         

    129 _RELOADING = {}                                                         

────────────────────────────────────────────────────────────────────────────────

ImportError: libGL.so.1: cannot open shared object file: No such file or 

directory

[18:30:18] 🐙 Pulling code changes from Github...

[18:30:19] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.11.14 environment at /home/adminuser/venv

Resolved 82 packages in 447ms

Audited 82 packages in 0.20ms

Checking if Streamlit is installed

Found Streamlit version 1.42.0 in the environment

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.11.14 environment at /home/adminuser/venv

Audited 1 package in 3ms


────────────────────────────────────────────────────────────────────────────────────────


[18:30:20] 🐍 Python dependencies were installed from /mount/src/drone-detection-yolov11/requirements.txt using uv.

[18:30:20] 📦 Processed dependencies!

  Stopping...




[18:30:23] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:121 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:591 in code_to_exec                                     

                                                                                

  /mount/src/drone-detection-yolov11/app.py:6 in <module>                       

                                                                                

      3 import subprocess                                                       

      4                                                                         

      5 import streamlit as st                                                  

  ❱   6 from ultralytics import YOLO                                            

      7 import cv2                                                              

      8 import imageio_ffmpeg                                                   

      9                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/ultralytics/__init__.py:13  

  in <module>                                                                   

                                                                                

    10 if not os.environ.get("OMP_NUM_THREADS"):                                

    11 │   os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU util  

    12                                                                          

  ❱ 13 from ultralytics.utils import ASSETS, SETTINGS                           

    14 from ultralytics.utils.checks import check_yolo as checks                

    15 from ultralytics.utils.downloads import download                         

    16                                                                          

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/ultralytics/utils/__init__  

  .py:24 in <module>                                                            

                                                                                

      21 from types import SimpleNamespace                                      

      22 from urllib.parse import unquote                                       

      23                                                                        

  ❱   24 import cv2                                                             

      25 import numpy as np                                                     

      26 import torch                                                           

      27                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/cv2/__init__.py:181 in      

  <module>                                                                      

                                                                                

    178 │   if DEBUG: print('OpenCV loader: DONE')                              

    179                                                                         

    180                                                                         

  ❱ 181 bootstrap()                                                             

    182                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/cv2/__init__.py:153 in      

  bootstrap                                                                     

                                                                                

    150 │                                                                       

    151 │   py_module = sys.modules.pop("cv2")                                  

    152 │                                                                       

  ❱ 153 │   native_module = importlib.import_module("cv2")                      

    154 │                                                                       

    155 │   sys.modules["cv2"] = py_module                                      

    156 │   setattr(py_module, "_native", native_module)                        

                                                                                

  /usr/local/lib/python3.11/importlib/__init__.py:126 in import_module          

                                                                                

    123 │   │   │   if character != '.':                                        

    124 │   │   │   │   break                                                   

    125 │   │   │   level += 1                                                  

  ❱ 126 │   return _bootstrap._gcd_import(name[level:], package, level)         

    127                                                                         

    128                                                                         

    129 _RELOADING = {}                                                         

────────────────────────────────────────────────────────────────────────────────

ImportError: libGL.so.1: cannot open shared object file: No such file or 

directory

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:121 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:591 in code_to_exec                                     

                                                                                

  /mount/src/drone-detection-yolov11/app.py:6 in <module>                       

                                                                                

      3 import subprocess                                                       

      4                                                                         

      5 import streamlit as st                                                  

  ❱   6 from ultralytics import YOLO                                            

      7 import cv2                                                              

      8 import imageio_ffmpeg                                                   

      9                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/ultralytics/__init__.py:13  

  in <module>                                                                   

                                                                                

    10 if not os.environ.get("OMP_NUM_THREADS"):                                

    11 │   os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU util  

    12                                                                          

  ❱ 13 from ultralytics.utils import ASSETS, SETTINGS                           

    14 from ultralytics.utils.checks import check_yolo as checks                

    15 from ultralytics.utils.downloads import download                         

    16                                                                          

                                                                                

  /home/adminuser/venv/lib/python3.11/site-packages/ultralytics/utils/__init__  

  .py:24 in <module>                                                            

                                                                                

      21 from types import SimpleNamespace                                      

      22 from urllib.parse import unquote                                       

      23                                                                        

  ❱   24 import cv2                                                             

      25 import numpy as np                                                     

      26 import torch                                                           

      27                                                                        

────────────────────────────────────────────────────────────────────────────────

ImportError: libGL.so.1: cannot open shared object file: No such file or 

directory

main
saadalhanif/drone-detection-yolov11/main/app.py
