from .video_record import VideoRecord
from datetime import timedelta
import time

from fvcore.common.config import CfgNode
import numpy as np


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, "%H:%M:%S.%f")
    sec = (
        float(timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds())
        + float(timestamp.split(".")[-1]) / 100
    )
    return sec


class EpicKitchensVideoRecordGRU(VideoRecord):
    def __init__(self, tup, cfg: CfgNode):
        self.cfg = cfg
        self._index = str(tup[0])
        self._series = tup[1]
        self._num_frames = self.cfg.DATA.NUM_FRAMES
        self._subclip_overlap = self.cfg.DATA.SUBCLIP_OVERLAP

    @property
    def participant(self):
        return self._series["participant_id"]

    @property
    def untrimmed_video_name(self):
        return self._series["video_id"]

    @property
    def start_frame(self):
        return int(round(timestamp_to_sec(self._series["start_timestamp"]) * self.fps))

    @property
    def end_frame(self):
        return int(round(timestamp_to_sec(self._series["stop_timestamp"]) * self.fps))

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split("_")[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def length_in_s(self):
        return self.num_frames / self.fps

    @property
    def label(self):
        return {
            "verb": self._series["verb_class"] if "verb_class" in self._series else -1,
            "noun": self._series["noun_class"] if "noun_class" in self._series else -1,
        }

    @property
    def num_subclips(self):
        subclip_length = self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE / self.fps
        return int(
            np.ceil(
                max(
                    1,
                    (self.length_in_s - self.cfg.DATA.SUBCLIP_OVERLAP)
                    / (subclip_length - self.cfg.DATA.SUBCLIP_OVERLAP),
                ),
            ),
        )

    @property
    def metadata(self):
        return {"narration_id": self._index}
