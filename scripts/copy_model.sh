#!/bin/sh

cp -r CE-att-supervise-0.5  $1
cd $1
# mv optimizer-30000.pth  optimizer.pth
# mv model-30000.pth  model.pth
mv infos_CE-att-supervise-0.5-best.pkl infos_$1-best.pkl 
mv  infos_CE-att-supervise-0.5.pkl  infos_$1.pkl 
mv histories_CE-att-supervise-0.5.pkl  histories_$1.pkl
cd ../