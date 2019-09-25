import torch
import cv2
import time
import argparse

import posenet

import subprocess 

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=100)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def main():
    save_file = ['ffmpeg',
            '-i','-',
            '-c:v', 'copy',
            '-hls_time', '10',
            '-hls_list_size', '3',
            '-hls_wrap', '3',
            '-hls_segment_type', 'fmp4',
            'vid.m3u8']
    stream_file = ['ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s','640x480',
            '-i','vid.m3u8',
            '-an',   # disable audio
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            # '-tune', 'zerolatency',
            # '-filter:v', 'minterpolate=\'mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=30\'',
            # '-b:v', '1M',
            # '-c:v', 'libx264',
            # '-maxrate', '1M',
            # '-bufsize', '2M',
            # '-strict','experimental',
            # '-vcodec','h264',
            # '-pix_fmt','yuv420p',
            # '-g', '50',
            # '-vb','1000k',
            # '-profile:v', 'baseline',
            # '-preset', 'ultrafast',
            # '-r', '10',
            # '-f', 'flv', 
            'rtmp://live-lax.twitch.tv/app/live_173288790_pEOfgLFUAfocVRZdAQ1D8bUubjL4OY']

    pipe = subprocess.Popen(save_file, stdin=subprocess.PIPE)
    stream = None

    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        pipe.stdin.write(overlay_image.tostring())
        if not stream:
            stream = subprocess.Popen(stream_file)

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()