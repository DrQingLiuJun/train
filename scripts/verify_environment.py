"""Small disposable GPU training/inference test. Does not start a full run."""
import gc,json,os,tempfile
from pathlib import Path
import train as project
import torch,torchvision,ultralytics,yaml
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
assert torch.cuda.is_available(),'CUDA unavailable'
report={'python':os.sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'vram_gib':round(torch.cuda.get_device_properties(0).total_memory/2**30,2),'models':{}}
image=Path((ROOT/'dataset/val.txt').read_text().splitlines()[0])
for path in [ROOT/'models/yolo26m.pt',*sorted((ROOT/'models/baseline').glob('*.pt'))]:
 m=YOLO(str(path));ckpt=m.ckpt or {};r=m.predict(str(image),device=0,imgsz=768,verbose=False,save=False)
 report['models'][path.name]={'names':m.names,'parameters':sum(p.numel() for p in m.model.parameters()),'scale':m.model.yaml.get('scale'),'checkpoint_epoch':ckpt.get('epoch'),'optimizer_present':ckpt.get('optimizer') is not None,'inference_ok':True,'sample_detections':len(r[0].boxes)}
 del m,r;gc.collect();torch.cuda.empty_cache()
with tempfile.TemporaryDirectory(prefix='train-smoke-',dir='/tmp') as td:
 td=Path(td)
 for s in ('train','val'):(td/f'{s}.txt').write_text('\n'.join((ROOT/f'dataset/{s}.txt').read_text().splitlines()[:4])+'\n')
 data=td/'data.yaml';data.write_text(yaml.safe_dump({'path':str(ROOT/'dataset'),'train':str(td/'train.txt'),'val':str(td/'val.txt'),'nc':4,'names':['person','car','cyclist','truck']}))
 model=YOLO(str(ROOT/'models/yolo26m.pt'));verified={}
 def before(t):verified.update(project.verify_initial_weights(t,ROOT/'models/yolo26m.pt',strict=False))
 model.add_callback('on_train_start',before);model.add_callback('on_model_save',project.preserve_resume)
 model.train(data=str(data),epochs=1,patience=0,imgsz=768,batch=2,workers=0,device=0,amp=False,cache=False,optimizer='AdamW',lr0=.0001,warmup_epochs=0,warmup_bias_lr=0.,plots=False,save=True,val=True,verbose=False,project=str(td),name='run',exist_ok=True,seed=20261027)
 checkpoint=torch.load(td/'run/weights/resume.pt',map_location='cpu',weights_only=False)
 assert checkpoint['optimizer'] is not None and checkpoint['epoch']==0,'Rolling full checkpoint is not resumable'
 report['fresh_model_transfer']=verified;report['gpu_forward_backward_optimizer_ok']=True;report['rolling_resume_has_optimizer']=True;report['rolling_resume_epoch']=checkpoint['epoch']
 report['smoke_max_allocated_gib']=round(torch.cuda.max_memory_allocated()/2**30,2)
 del checkpoint,model;gc.collect();torch.cuda.empty_cache()
(ROOT/'reports/environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('ENVIRONMENT_VERIFIED',json.dumps(report,ensure_ascii=False),flush=True)
