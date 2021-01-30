import torch
print(torch)

print(torch.cuda.is_available())

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
fdir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
imgs = [fdir + f for f in ('zidane.jpg', 'bus.jpg')]  # batched list of images

results = model(imgs)
results.print()  # or .show(), .save()