#Hardware
  - Intel® Neural Compute Stick 1
  - Raspberry Pi 2/3 

# Raspbian OS  
  - Kernel: 5.10.52-v7+
  - Raspbian GNU/Linux 10 (buster)
  - https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2021-05-28/2021-05-07-raspios-buster-armhf-lite.zip  
  
# Raspberry Pi 2/3 ncsdk setup 

1. **Installing depedencies**    
    - sudo apt-get install -y libusb-1.0-0-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev git automake byacc lsb-release cmake libgflags-dev libgoogle-glog-dev liblmdb-dev swig3.0 graphviz libxslt-dev libxml2-dev gfortran python3-dev python-pip python3-pip python3-setuptools python3-markdown python3-pillow python3-yaml python3-pygraphviz python3-h5py python3-nose python3-lxml python3-matplotlib python3-numpy python3-protobuf python3-dateutil python3-skimage python3-scipy python3-six python3-networkx python3-tk
    - sudo apt install python-opencv libjasper-dev libqtgui4 libqt4-test
    - sudo pip3 install opencv-python

2. **Compiling sdk**     
   - cd /home/pi/movidius/ncsdk/api/src  
   - make  
   - sudo make install     

3. **Verifying USB Movidius neural stick functioning properly**    
python3 /home/pi/movidius/ncappzoo/apps/hello_ncs_py/hello_ncs.py

4. **Testing sample app**   
python3 /home/pi/movidius/ncappzoo/apps/object-detector/object-detector.py -i /home/pi/movidius/temp/test02.jpg   

5. **Pi 3 known issue**
   - Network connection will disconnect intermitently with 1st generation of Movidius USB   

## Openvino 2020.3 setup on Ubuntu 18 (if you want to use Ubuntu instead of Raspbian)

1. **Following Instruction below (movidius neural stick first generatuin only supported by openvino until 2020.3):**    
https://docs.openvinotoolkit.org/2020.3/_docs_install_guides_installing_openvino_linux.html   


2. **Test Sample App Step 1**    
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --list /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/models.lst

3. **Test Sample App Step 2 (please ensure the machine with web camera accessible via /dev/videoN)**     
    - python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m /opt/intel/openvino_2020.3.194/deployment_tools/intel/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml -i /dev/video2 

    - python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m /opt/intel/openvino_2020.3.194/deployment_tools/intel/vehicle-license-plate-detection-barrier-0106/FP16-INT8/vehicle-license-plate-detection-barrier-0106.xml -i /dev/video2

    - python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m  /opt/intel/openvino_2020.3.194/deployment_tools/intel/vehicle-detection-adas-binary-0001/FP32-INT1/vehicle-detection-adas-binary-0001.xml -i /dev/video2

    - python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m  /opt/intel/openvino_2020.3.194/deployment_tools/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -i /dev/video0 -d MYRIAD


## FAQ

1. ** Error on POPCNT feature in VBox VM **       
- cd C:\Program Files\Oracle\VirtualBox\VBoxManage.exe setextradata openvino VBoxInternal/CPUM/IsaExts/POPCNT 1

2. **Test run on openvino**    
    - cd ~/inference_engine_samples_build/intel64/Release     
   ./classification_sample_async -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml -d MYRIAD

    - cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites

    - python3 /opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/downloader.py  --list ./open_model_zoo/demos/python_demos/face_recognition_demo/models.lst

    - python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/face_recognition_demo/face_recognition_demo.py -m_fd /opt/intel/openvino_2020.3.194/deployment_tools/intel/face-detection-retail-0004/FP16-INT8/face-detection-retail-0004.xml -m_lm /opt/intel/openvino_2020.3.194/deployment_tools/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -m_reid /opt/intel/openvino_2020.3.194/deployment_tools/intel/face-reidentification-retail-0095/FP16-INT8/face-reidentification-retail-0095.xml -fg "/home/face_gallery"
