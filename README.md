# openvino

python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --list /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/models.lst



python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m /opt/intel/openvino_2020.3.194/deployment_tools/intel/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml -i /dev/video2 

python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m /opt/intel/openvino_2020.3.194/deployment_tools/intel/vehicle-license-plate-detection-barrier-0106/FP16-INT8/vehicle-license-plate-detection-barrier-0106.xml -i /dev/video2


python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m  /opt/intel/openvino_2020.3.194/deployment_tools/intel/vehicle-detection-adas-binary-0001/FP32-INT1/vehicle-detection-adas-binary-0001.xml -i /dev/video2


python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -m  /opt/intel/openvino_2020.3.194/deployment_tools/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -i /dev/video0 -d MYRIAD

=================================================================================================================================

cd C:\Program Files\Oracle\VirtualBox

VBoxManage.exe setextradata openvino VBoxInternal/CPUM/IsaExts/POPCNT 1


 cd ~/inference_engine_samples_build/intel64/Release
 ./classification_sample_async -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml -d MYRIAD



cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites



python3 /opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/downloader.py  --list ./open_model_zoo/demos/python_demos/face_recognition_demo/models.lst





python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/face_recognition_demo/face_recognition_demo.py \

-m_fd /opt/intel/openvino_2020.3.194/deployment_tools/intel/face-detection-retail-0004/FP16-INT8/face-detection-retail-0004.xml \

-m_lm /opt/intel/openvino_2020.3.194/deployment_tools/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml \

-m_reid /opt/intel/openvino_2020.3.194/deployment_tools/intel/face-reidentification-retail-0095/FP16-INT8/face-reidentification-retail-0095.xml \

-fg "/home/face_gallery"



 

=
