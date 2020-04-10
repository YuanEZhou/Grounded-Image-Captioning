echo 'Results Reported in Table 4.'
echo 'Eval Up-Down+XE'
python eval.py   --verbose_beam  0   --verbose_loss  0  --verbose  0   --model  './log/COCO-CE-r1024/model-best.pth'  --infos_path  './log/COCO-CE-r1024/infos_COCO-CE-r1024-best.pkl'  --split test  --beam_size  3    --att_supervise  0   --eval_att   0  --gt_grd_eval   0  --dataset  coco  --eval_scan  0
echo '================================================================================='
echo 
echo 
echo 
echo '================================================================================='
echo 'Eval Up-Down+XE+SCST(CIDEr)'
python eval.py   --verbose_beam  0   --verbose_loss  0  --verbose  0   --model  './log/sc-cider-COCO-CE-r1024/model-best.pth'  --infos_path  './log/sc-cider-COCO-CE-r1024/infos_sc-cider-COCO-CE-r1024-best.pkl'   --split  'test' --beam_size  3    --att_supervise  0   --eval_att   0  --gt_grd_eval   0  --dataset  coco  --eval_scan  0
echo '================================================================================='
echo 
echo 
echo 
echo '================================================================================='
echo 'Eval Up-Down+XE+SCST(CIDEr+SCAN)'
python eval.py   --verbose_beam  0   --verbose_loss  0  --verbose  0   --model  './log/sc-ground-COCO-CE-r1024/model-best.pth'  --infos_path  './log/sc-ground-COCO-CE-r1024/infos_sc-ground-COCO-CE-r1024-best.pkl'  --split  'test' --beam_size  3    --att_supervise  0   --eval_att   0  --gt_grd_eval   0  --dataset  coco  --eval_scan  0
echo '================================================================================='
echo 
echo 
echo 
echo '================================================================================='
echo 'Eval Up-Down+XE+KL(POS-SCAN)'
python eval.py   --verbose_beam  0   --verbose_loss  0  --verbose  0   --model  './log/COCO-CE-scan-sup-1kl-r1024/model-best.pth'  --infos_path  './log/COCO-CE-scan-sup-1kl-r1024/infos_COCO-CE-scan-sup-1kl-r1024-best.pkl'  --split  'test'  --beam_size  3    --att_supervise  0   --eval_att   0  --gt_grd_eval   0  --dataset  coco  --eval_scan  0
echo '================================================================================='
echo 
echo 
echo 
echo '================================================================================='
echo 'Eval Up-Down+XE+KL(POS-SCAN)+SCST(CIDEr)'
python eval.py   --verbose_beam  0   --verbose_loss  0  --verbose  0   --model  './log/sc-cider-COCO-CE-scan-sup-1kl-r1024/model-best.pth'  --infos_path  './log/sc-cider-COCO-CE-scan-sup-1kl-r1024/infos_sc-cider-COCO-CE-scan-sup-1kl-r1024-best.pkl'  --split  'test'  --beam_size  3    --att_supervise  0   --eval_att   0  --gt_grd_eval   0  --dataset  coco  --eval_scan  0
echo '================================================================================='
echo 
echo 
echo 
echo '================================================================================='
echo 'Eval Up-Down+XE+KL(POS-SCAN)+SCST(CIDEr+SCAN)'
python eval.py   --verbose_beam  0   --verbose_loss  0  --verbose  0   --model  './log/sc-ground-COCO-CE-scan-sup-1kl-r1024/model-best.pth'  --infos_path  './log/sc-ground-COCO-CE-scan-sup-1kl-r1024/infos_sc-ground-COCO-CE-scan-sup-1kl-r1024-best.pkl'  --split  'test'  --beam_size  3    --att_supervise  0   --eval_att   0  --gt_grd_eval   0  --dataset  coco  --eval_scan  0
echo '================================================================================='
echo 
echo 
echo 