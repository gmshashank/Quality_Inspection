# Quality_Inspection
Quality Inspection of Casting defect using MobileNet 



# Dataset:
```
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
```

```
https://manufacturing-models.s3.eu-west-2.amazonaws.com/Castings.zip
```

# Model
```
https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
```
The MobileNet v2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input. MobileNet v2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, non-linearities in the narrow layers were removed in order to maintain representational power.


# Reference
MobileNet v2 - ```https://arxiv.org/abs/1801.04381```

AWS Lambda Deployment - ```https://www.philschmid.de/scaling-machine-learning-from-zero-to-hero```