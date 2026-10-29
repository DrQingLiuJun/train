import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(YOLO_CONFIG_DIR=str(ROOT/'.settings'),YOLO_AUTOINSTALL='false',MPLCONFIGDIR=str(ROOT/'.settings/mpl'))
from ultralytics import YOLO
p=argparse.ArgumentParser();p.add_argument('--model',default='runs/main/weights/best.pt');p.add_argument('--imgsz',type=int,default=768);a=p.parse_args()
model=YOLO(str(ROOT/a.model));r=model.val(data=str(ROOT/'dataset/data.yaml'),imgsz=a.imgsz,batch=16,device=0,workers=4,conf=.001,iou=.45,max_det=300,plots=False,project=str(ROOT/'reports'),name='evaluation',exist_ok=True)
result={'model':a.model,'imgsz':a.imgsz,'map50':float(r.box.map50),'map50_95':float(r.box.map),'recall':float(r.box.mr),'precision':float(r.box.mp),'names':model.names}
(ROOT/'reports/evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
