import cv2
import numpy as np
from PIL import Image

from bottle_qc.pipeline.sources import ImageFileSource, VideoFileSource


def test_image_source_is_single_item_and_keeps_each_image_identity(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (48, 32), "white").save(first)
    Image.new("RGB", (48, 32), "black").save(second)
    left = ImageFileSource(first, source_item_id="asset-left")
    right = ImageFileSource(second, source_item_id="asset-right")
    try:
        left_frame = left.read()
        right_frame = right.read()
        assert left_frame.index == right_frame.index == 0
        assert left_frame.source_item_id == "asset-left"
        assert right_frame.source_item_id == "asset-right"
        assert left.read() is None and right.read() is None
    finally:
        left.close()
        right.close()


def test_video_source_returns_every_frame_in_order_with_source_timestamps(tmp_path):
    video = tmp_path / "ordered.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for value in (20, 60, 100, 140):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    source = VideoFileSource(video, source_item_id="asset-video")
    try:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)
        assert [frame.index for frame in frames] == [0, 1, 2, 3]
        assert all(frame.source_item_id == f"asset-video:frame-{index:08d}" for index, frame in enumerate(frames))
        assert all(frame.source_timestamp_ms is not None for frame in frames)
        assert [frame.timestamp for frame in frames] == sorted(frame.timestamp for frame in frames)
        source_times = [frame.source_timestamp_ms for frame in frames]
        assert all(right > left for left, right in zip(source_times, source_times[1:]))
        assert [frame.timestamp * 1000 for frame in frames] == source_times
        assert source.frames_read == 4
    finally:
        source.close()
