#!/bin/bash


for i in {0..9}; do
  FLIP3D=1 SPEC=f python model_trainer.py $i
  BRIGHTNESS=1 SPEC=b python model_trainer.py $i
  SCALE=1 SPEC=s10 python model_trainer.py $i
  SCALE=2 SPEC=s20 python model_trainer.py $i
  ROTATION=12 SPEC=r15 python model_trainer.py $i
  ROTATION=6 SPEC=r30 python model_trainer.py $i
  ROTATION=3 SPEC=r60 python model_trainer.py $i
  ROTATION=2 SPEC=r90 python model_trainer.py $i
  ELASTIC=2 SPEC=e2 python model_trainer.py $i
  ELASTIC=4 SPEC=e4 python model_trainer.py $i
  ELASTIC=6 SPEC=e6 python model_trainer.py $i
  ELASTIC=8 SPEC=e8 python model_trainer.py $i
done
