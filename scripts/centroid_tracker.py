from scipy.spatial import distance as dist
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=20, iou_threshold=0.3):
        self.next_object_id = 0
        self.objects = {}  # objectID -> (bbox, class)
        self.disappeared = {}
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def register(self, bbox, cls):
        self.objects[self.next_object_id] = (bbox, cls)
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = [self._centroid(bbox)]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.tracks[object_id]

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, input_boxes, input_classes):
        if len(input_boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {k: self._centroid(v[0]) for k, v in self.objects.items()}

        if len(self.objects) == 0:
            for box, cls in zip(input_boxes, input_classes):
                self.register(box, cls)
        else:
            object_ids = list(self.objects.keys())
            object_data = list(self.objects.values())

            unmatched_input = set(range(len(input_boxes)))
            unmatched_existing = set(object_ids)

            matches = []

            for idx, (box, cls) in enumerate(zip(input_boxes, input_classes)):
                best_iou = 0
                best_id = None
                for oid in object_ids:
                    obox, ocls = self.objects[oid]
                    if ocls != cls:
                        continue
                    iou = self._iou(box, obox)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_id = oid

                if best_id is not None and best_id in unmatched_existing:
                    self.objects[best_id] = (box, cls)
                    self.disappeared[best_id] = 0
                    self.tracks[best_id].append(self._centroid(box))
                    self.tracks[best_id] = self.tracks[best_id][-15:]
                    unmatched_input.discard(idx)
                    unmatched_existing.discard(best_id)

            for oid in unmatched_existing:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for idx in unmatched_input:
                self.register(input_boxes[idx], input_classes[idx])

        return {k: self._centroid(v[0]) for k, v in self.objects.items()}
