"""Explicit fresh / full-state resume / weight-only finetune. No silent fallback."""
import argparse,copy,json,os,shutil,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);os.environ.update(YOLO_CONFIG_DIR=str(ROOT/'.settings'),YOLO_AUTOINSTALL='false',MPLCONFIGDIR=str(ROOT/'.settings/mpl'),WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='4')
import torch,yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model
STATE=ROOT/'reports/training-state.json'
def state(**values):
 s=json.loads(STATE.read_text()) if STATE.exists() else {};s.update(values,pid=os.getpid(),updated=time.time());p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2));p.replace(STATE)
def interrupt(*_):raise KeyboardInterrupt

def verify_initial_weights(t,source,strict=True):
 ref=YOLO(str(source)).model.float().state_dict();actual=unwrap_model(t.model).state_dict();bad=[];checked=0;new_head=0
 head_prefix='model.'+str(len(unwrap_model(t.model).model)-1)+'.'
 for key,v in ref.items():
  target=actual.get(key)
  equal=target is not None and target.shape==v.shape and torch.equal(v.cpu(),target.detach().cpu().to(v.dtype))
  if equal:checked+=1
  elif not strict and key.startswith(head_prefix):new_head+=1
  else:bad.append(key)
 if bad:raise RuntimeError('Initial model differs from selected weights: '+str(bad[:8]))
 print('INITIAL_WEIGHTS_VERIFIED',checked,'fresh_head_tensors='+str(new_head),'strict='+str(strict),flush=True)
 return {'checked':checked,'fresh_head_tensors':new_head,'strict':strict}

def preserve_resume(t):
 # One rolling FULL checkpoint survives Ultralytics' final optimizer stripping.
 last=Path(t.last)
 if last.exists():
  temp=last.with_name('resume.tmp.pt');shutil.copyfile(last,temp);temp.replace(last.with_name('resume.pt'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fresh','resume','finetune'],default='fresh');p.add_argument('--model');p.add_argument('--name',default='main');p.add_argument('--epochs',type=int);p.add_argument('--batch',type=int);a=p.parse_args()
 if not a.name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run name')
 cfg=yaml.safe_load((ROOT/'config/train.yaml').read_text());out=ROOT/'runs'/a.name
 source=ROOT/(a.model or (str(Path('runs')/a.name/'weights/resume.pt') if a.mode=='resume' else cfg.pop('model')))
 cfg.pop('model',None);cfg['data']=str(ROOT/cfg['data'])
 if a.epochs is not None:cfg['epochs']=a.epochs
 if a.batch is not None:cfg['batch']=a.batch
 if not source.is_file():raise FileNotFoundError(source)
 if not Path(cfg['data']).is_file():raise RuntimeError('Dataset is not prepared')
 if a.mode!='resume' and (out/'weights/last.pt').exists():raise RuntimeError('Existing results: use --mode resume or a different --name')
 model=YOLO(str(source));ckpt=model.ckpt or {};can_resume=ckpt.get('epoch',-1)>=0 and ckpt.get('optimizer') is not None
 if a.mode=='resume' and not can_resume:raise RuntimeError('Checkpoint has no optimizer/epoch state. Use --mode finetune explicitly.')
 if a.mode=='finetune':cfg.update(lr0=.0002,warmup_epochs=0,warmup_bias_lr=0.)
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
 state(status='preparing',mode=a.mode,source=str(source),epoch=None,completed_epoch=None,epochs=cfg['epochs'],metrics={},error=None,output=str(out),gpu_memory_limit=None)
 def verify(t):verify_initial_weights(t,source,strict=a.mode!='fresh')
 in_epoch=False
 def start(t):
  nonlocal in_epoch
  in_epoch=True;state(status='training',epoch=t.epoch+1,epochs=t.epochs)
 def end(t):
  nonlocal in_epoch
  if not in_epoch:return  # final best-model validation is not another training epoch
  in_epoch=False
  m={k:float(v) for k,v in t.metrics.items()};state(status='training',epoch=t.epoch+1,completed_epoch=t.epoch+1,metrics=m)
  print('[PROGRESS]',json.dumps({'epoch':t.epoch+1,'epochs':t.epochs,**m}),flush=True)
 model.add_callback('on_train_start',verify);model.add_callback('on_train_epoch_start',start);model.add_callback('on_fit_epoch_end',end);model.add_callback('on_model_save',preserve_resume)
 print('TRAINING_MODE',a.mode,'SOURCE',source,flush=True)
 model.train(**cfg,resume=str(source) if a.mode=='resume' else False,device=0,project=str(ROOT/'runs'),name=a.name,exist_ok=True,save=True,save_period=-1,plots=True,val=True,verbose=False)
 state(status='complete',best=str(out/'weights/best.pt'),last=str(out/'weights/last.pt'));print('TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:main()
 except KeyboardInterrupt:state(status='interrupted');raise SystemExit(130)
 except Exception:state(status='failed',error=traceback.format_exc());raise
