import os, shutil, tempfile
import numpy as np
from PIL import Image


def segment_organoid(movie):
    CHECKPOINT = r"C:\Users\6331823\Downloads\sam2.1_hiera_base_plus.pt"
    CONFIG = r"C:\Users\6331823\Downloads\sam2.1_hiera_b+.yaml"
    sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)

    movie = movie.astype(np.uint8)

    # Temporary directory for saving frames
    temp_dir = tempfile.mkdtemp()

    # Itterate over movie frames
    for i, frame in enumerate(movie):
        # Convert image to RGB grayscale image
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Normalize the frame to 0-255 range
        frame = (frame / frame.max() * 255).astype(np.uint8)

        # Save the frame as a JPEG image
        Image.fromarray(frame).save(os.path.join(temp_dir, f"{i:05d}.jpeg"))

    # Get inference state
    inference_state = sam2_model.init_state(temp_dir)

    # Calculate center point where organoid should be
    center_point = np.array(
        [[movie.shape[1] // 2, movie.shape[2] // 2]], dtype=np.float32
    )
    _, object_ids, mask_logits = sam2_model.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=center_point,
        labels=np.array([1]),
    )

    all_masks = []

    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(
        inference_state
    ):
        masks = (mask_logits > 0.0).cpu().numpy()  # shape: (N, X, H, W)
        N, X, H, W = masks.shape
        masks = masks.reshape(N * X, H, W)
        all_masks.append(masks)

    shutil.rmtree(temp_dir)

    all_masks = np.array(all_masks)
    all_masks = all_masks[:, 0, :, :]

    return all_masks
