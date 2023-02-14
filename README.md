# Metric Calculator

MetricCalculator is a tool for calculating common computer vision metrics, such as Mean Average Precision (mAP), Number of False Negatives (FN), and Mean Distance between Centers of Bounding Boxes (MDBC).

```bash
python3 main.py
```

## Evaluation metrics


### Mean average precision (mAP)

mAP is a metric for determining how good object detection model is. It calculates an average of average precisions around all given IOU thresholds.
These threshold are determined by three numbers after '@' eg. mAP@0.5:0.9:0.05, this means that the starting IOU threshold is 0.5, ending threshold 
equals 0.9 and step between them is equal to 0.05 so average precision will be calculated for thresholds (0.5, 0.55, 0.6, 0.65, ..., 0.85, 0.9).
Average precision is simply an area under precision-recall curve. To calculate this area in an efficient way we used 11 point interpolation method.
First the curve must be smoothed out, every precision value is set to the maxiumum precision value for recalls grater than the current recall value.
Then 11 points are evenly placed on recall axis and the average of correponding precision values is taken. The resulting number is an average precision.

### Mean center distance

Mean center distance is used to represent mean distance between centers of bounding boxes. First every prediction is matched with the closest 
ground truth bounding box. If distance between centers of these bounding boxes is smaller than given threshold (we used the value 20) it is
classified as true positive. Than we simply took an average of these distances as mean center distance metric.