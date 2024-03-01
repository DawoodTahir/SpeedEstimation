

###import libraries 
import argparse
from ultralytics import YOLO
import numpy as  np
import supervision as sv 
import cv2
from collections import defaultdict,deque


#### Coordinates setting for the scene of the video, it will vary for different scenes and can be modified.
SOURCE= np.array([[1252,787],[2298,803],[5039,2159],[-550,2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

###Target coordinates of the transform 
TARGET = np.array(
    [
        [0,0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    
    ]
)
###


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
       
       ###since Getperspective function need ppints in float32
       source = source.astype(np.float32)
       target = target.astype(np.float32)
       self.m = cv2.getPerspectiveTransform(source,target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        ##perspective transform needs in 3d space ,so the extra dimension
        reshape_points=points.reshape(-1,1,2).astype(np.float32)
        transformed_points= cv2.perspectiveTransform(reshape_points,self.m)
        return transformed_points.reshape(-1,2)

def parse_argument():
    parser=argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
   
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_argument()
    model = YOLO("yolov8x.pt")
    video_info=sv.VideoInfo.from_video_path(args.source_video_path)
    ###to track multiple objects we use Bytetrack
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    
    text_scale=sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)



    thickness=sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)
    
    bounding_box_annotator=sv.BoundingBoxAnnotator(thickness=4)
    label_annotator=sv.LabelAnnotator(text_scale=text_scale,text_thickness=thickness)
    trace_annotator=sv.TraceAnnotator(color=sv.Color.red(),trace_length=30,thickness=4)
    
    frame_generator=sv.get_video_frames_generator(args.source_video_path)
    polygon_zone = sv.PolygonZone(
        polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh
    )
    view_transformer=ViewTransformer(source=SOURCE,target=TARGET)
    ##to calculate speed , getting coordinates and 
    coordinates=defaultdict(lambda : deque(maxlen= video_info.fps))

    for frame in frame_generator :
        ##run model on each frame
        result = model(frame)[0]
        ###convert our results into supervision detection models
        detections= sv.Detections.from_ultralytics(result)
        detections=detections[polygon_zone.trigger(detections)] 
        detections =byte_track.update_with_detections(detections=detections)
        ##points collected from sv video source 
        points=detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        ##points converted through view transform method to find 
        points=view_transformer.transform_points(points=points).astype(int)
        
        labels=[]
           
        for tracker_id, [_,y] in zip(detections.tracker_id,points):
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2 :
                    labels.append(f'#{tracker_id}')
                else: 
                    coordinate_start=coordinates[tracker_id][-1]
                    coordinate_end=coordinates[tracker_id][0]
                    time = len(coordinates[tracker_id]) / video_info.fps
                    distance=abs(coordinate_start - coordinate_end)
                    speed = distance / time * 3.6
                    labels.append(f'#{tracker_id} {int(speed)} km/h')
        annotated_frame = frame.copy()
        annotated_frame=sv.draw_polygon(scene=annotated_frame,polygon=SOURCE,color=sv.Color.red())
        annotated_frame=bounding_box_annotator.annotate(scene=annotated_frame,detections=detections)
        annotated_frame=label_annotator.annotate(scene=annotated_frame,detections=detections,labels=labels)
        annotated_frame=trace_annotator.annotate(scene=annotated_frame,detections=detections)
       
        cv2.imshow("annotated_frame",annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.destroyAllWindows()


