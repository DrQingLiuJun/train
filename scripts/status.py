import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'reports/training-state.json'
if not p.exists():print('环境已准备；尚未启动正式训练。');raise SystemExit
s=json.loads(p.read_text());m=s.get('metrics',{})
print('状态:',s.get('status'),'| 模式:',s.get('mode'),'| PID:',s.get('pid'));print('权重来源:',s.get('source'));print('当前轮数: %s / %s；最近完成: %s'%(s.get('epoch'),s.get('epochs'),s.get('completed_epoch')))
for title,key in [('mAP50','metrics/mAP50(B)'),('mAP50–95','metrics/mAP50-95(B)'),('召回率','metrics/recall(B)'),('精确率','metrics/precision(B)')]:
 v=m.get(key);print(title+': '+('等待验证' if v is None else f'{v:.4f} ({v*100:.2f}%)'))
print('更新:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(s['updated'])));print('输出目录:',s.get('output'))
if s.get('error'):print(s['error'])
