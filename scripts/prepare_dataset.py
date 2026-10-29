"""Extract the selected train/val set, audit it, and prepare training copies only."""
import argparse,collections,hashlib,json,math,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'dataset';REPORTS=ROOT/'reports'
def conflict(rows):
 boxes=[(int(r[0]),r[1]-r[3]/2,r[2]-r[4]/2,r[1]+r[3]/2,r[2]+r[4]/2) for r in rows]
 for i,a in enumerate(boxes):
  for b in boxes[i+1:]:
   if a[0]==b[0]:continue
   inter=max(0,min(a[3],b[3])-max(a[1],b[1]))*max(0,min(a[4],b[4])-max(a[2],b[2]));union=(a[3]-a[1])*(a[4]-a[2])+(b[3]-b[1])*(b[4]-b[2])-inter
   if union>0 and inter/union>.8:return True
 return False

def main():
 archive=ROOT/'dataset.zip';sha=hashlib.sha256()
 with archive.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):sha.update(b)
 expected=(ROOT/'.dataset.sha256').read_text().split()[0]
 if sha.hexdigest()!=expected:raise RuntimeError('Uploaded dataset.zip checksum mismatch')
 reference=json.loads((ROOT/'.reference-images.json').read_text());groups=collections.defaultdict(list);extracted=0;verified=0
 with zipfile.ZipFile(archive) as z:
  for i in z.infolist():
   parts=i.filename.split('/')
   if len(parts)!=4 or parts[0]!='train' or parts[1] not in ('images','labels') or parts[2] not in ('train','val') or parts[3] in ('','.','..'):continue
   if parts[1]=='images' and not parts[3].lower().endswith(('.jpg','.jpeg','.png')):continue
   if parts[1]=='labels' and not parts[3].endswith('.txt'):continue
   rel='/'.join(parts[1:]);p=D/rel;p.parent.mkdir(parents=True,exist_ok=True);content=z.read(i)
   if parts[1]=='images':
    h=hashlib.sha256(content).hexdigest();groups[h].append(rel)
    if rel not in reference or reference[rel]!=h:raise RuntimeError('Dataset differs from prior audited images: '+rel)
    verified+=1
   p.write_bytes(content);extracted+=1
   if extracted%5000==0:print('EXTRACTED',extracted,flush=True)
 if verified!=len(reference):raise RuntimeError('Image set incomplete')
 leaks=[v for v in groups.values() if any('/train/' in x for x in v) and any('/val/' in x for x in v)]
 if leaks:raise RuntimeError('Exact image train/val leakage found')
 report={'archive_sha256':sha.hexdigest(),'images_verified_against_previous_project':verified,'cross_split_exact_duplicates':len(leaks),'class_names':['person','car','cyclist','truck'],'splits':{}}
 for split in ('train','val'):
  stats={'source_images':0,'used_images':0,'missing_labels':[],'invalid_rows':[],'duplicate_boxes':0,'ambiguous_images':[],'excluded_images':[],'clipped_boxes':0,'negative_images':0,'class_counts':[0]*4};paths=[]
  for image in sorted((D/'images'/split).iterdir()):
   stats['source_images']+=1;label=D/'labels'/split/(image.stem+'.txt')
   if not label.exists():stats['missing_labels'].append(image.name);continue
   rows=[];seen=set();invalid=False;raw=label.read_text(encoding='utf-8-sig')
   for num,line in enumerate(raw.splitlines(),1):
    if not line.strip():continue
    try:
     r=list(map(float,line.split()));assert len(r)==5
     c,x,y,w,h=r;assert all(math.isfinite(v) for v in r) and c.is_integer() and 0<=c<4 and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
    except (ValueError,AssertionError):stats['invalid_rows'].append([label.name,num,line]);invalid=True;continue
    if tuple(r) in seen:stats['duplicate_boxes']+=1;continue
    seen.add(tuple(r));rows.append(r)
   ambiguous=conflict(rows)
   if ambiguous:stats['ambiguous_images'].append(image.name)
   if split=='train' and (invalid or ambiguous):stats['excluded_images'].append(image.name);continue
   clean=[]
   for c,x,y,w,h in rows:
    if split=='train':
     x1=max(0,x-w/2);y1=max(0,y-h/2);x2=min(1,x+w/2);y2=min(1,y+h/2)
     if x1>=x2 or y1>=y2:continue
     if min(x-w/2,y-h/2)<0 or max(x+w/2,y+h/2)>1:stats['clipped_boxes']+=1
     x,y,w,h=(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1
    clean.append(f'{int(c)} {x:.10f} {y:.10f} {w:.10f} {h:.10f}');stats['class_counts'][int(c)]+=1
   if not clean:
    if raw.strip():stats['excluded_images'].append(image.name);continue
    stats['negative_images']+=1
   label.write_text('\n'.join(clean)+ ('\n' if clean else ''));paths.append(str(image))
  (D/f'{split}.txt').write_text('\n'.join(paths)+'\n');stats['used_images']=len(paths);report['splits'][split]=stats
  print('PREPARED',split,len(paths),stats['class_counts'],flush=True)
 (D/'data.yaml').write_text(f'path: {D}\ntrain: {D}/train.txt\nval: {D}/val.txt\nnc: 4\nnames: [person, car, cyclist, truck]\n')
 REPORTS.mkdir(exist_ok=True);(REPORTS/'dataset-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print('DATASET_READY',json.dumps({k:v['used_images'] for k,v in report['splits'].items()}),flush=True)
if __name__=='__main__':main()
