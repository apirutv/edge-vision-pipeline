# cam_capture
sudo systemctl start evp@services.cam_capture.main

# change_detector
sudo systemctl start evp@services.change_detector.main

# hailo_detector
sudo systemctl start evp@services.hailo_detector.main

# lvm_describer
sudo systemctl start evp@services.lvm_describer.main

# uploader_ingest
sudo systemctl start evp@services.uploader_ingest.main

# redis_dashboard (Pi dashboard)
sudo systemctl start evp@services.redis_dashboard.main
