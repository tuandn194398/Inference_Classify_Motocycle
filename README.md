# Inference_Classify_Motocycle
--Windows--
python .\segment\predict.py --weights .\gelan-c-seg.pt --source .\data\videos\24_4.mp4 --save-crop --classes 1 2 3 -cp E:\Users\Admin\Desktop\Motocycle-Detection-BKAI\src\result_13kbb4cls\resnet50-v9.ckpt --model_mp resnet50 --view-img


--Ubuntu--
python segment/predict.py --weights gelan-c-seg.pt --source data/images/ --save-crop --classes 1 2 3 --model_mp resnet50 -cp resnet50-v10.ckpt --conf-thres 0.6 --nosave --iou-thres 0.4 --retina-masks
