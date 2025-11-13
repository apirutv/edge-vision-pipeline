MANIFEST='{
  "frame_id": "test123dfe",
  "camera_id": "lab_cam",
  "ts": "2025-11-12T10:59:00+00:00",
  "model": "hailo_yolo",
  "scene": "workbench",
  "person_present": false,
  "pet_present": false,
  "vehicles_present": false,
  "activities": []
}'

curl -v -X POST http://192.168.0.171:8000/api/ingest/frame \
  -F "manifest=${MANIFEST};type=application/json" \
  -F "frame=@/home/apirut/python_projects/edge-vision-pipeline/data/frames/dining/2025/11/12/11/1762945257514_e9740eceda35.jpg" \
  -F "tagged=@/home/apirut/python_projects/edge-vision-pipeline/outputs/dining/1762843277679_d7c16e65cd79_outframe_00001.jpg" \
  -F "detections=@/home/apirut/python_projects/edge-vision-pipeline/outputs/dining/1762857300967_51fc9991cbbc.json" \
  -F "description=@/home/apirut/python_projects/edge-vision-pipeline/outputs/dining/described/061b4a0c01a1.json"
