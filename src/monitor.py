import subprocess as sp
import numpy as np
import cv2


class Monitor:
    """Records the environment using ffmpeg."""

    def __init__(self, width: int, height: int, saved_path: str):
        self.width = width
        self.height = height
        self.command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}X{height}",
            "-pix_fmt",
            "bgr24",  # [수정 1] rgb24 -> bgr24 (OpenCV 기본 포맷에 맞춤)
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "mpeg4",
            "-r",
            "60",
            "-pix_fmt",
            "yuv420p",
            saved_path,
        ]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            print("Error: ffmpeg not found. Video recording disabled.")
            self.pipe = None

    def record(self, env) -> None:
        frame = env.render()
        bgr_frame = None

        if isinstance(frame, np.ndarray):
            frame_np = np.asarray(frame)

            # 차원 및 채널 변환 로직
            if frame_np.ndim == 3 and frame_np.shape[2] in (3, 4):
                # RGB(A) HWC -> BGR
                code = (
                    cv2.COLOR_RGB2BGR if frame_np.shape[2] == 3 else cv2.COLOR_RGBA2BGR
                )
                bgr_frame = cv2.cvtColor(frame_np, code)
            elif frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
                # CHW -> HWC -> BGR
                transposed = np.transpose(frame_np, (1, 2, 0))
                code = (
                    cv2.COLOR_RGB2BGR
                    if transposed.shape[2] == 3
                    else cv2.COLOR_RGBA2BGR
                )
                bgr_frame = cv2.cvtColor(transposed, code)
            elif frame_np.ndim == 2:
                # Grayscale -> BGR
                bgr_frame = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            else:
                print(f"Skipping render: unexpected frame shape {frame_np.shape}")

            if bgr_frame is not None:
                # [수정 2] 해상도 강제 맞춤 (rawvideo 깨짐 방지)
                if (bgr_frame.shape[1] != self.width) or (
                    bgr_frame.shape[0] != self.height
                ):
                    bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

                try:
                    cv2.imshow("Super Mario Bros PPO", bgr_frame)
                    cv2.waitKey(1)  # [수정 3] GUI 이벤트 처리를 위해 필수
                except cv2.error as e:
                    print(f"Skipping render due to OpenCV error: {e}")
                    bgr_frame = None

        if self.pipe is not None and bgr_frame is not None:
            try:
                # [수정 4] 예외 처리 추가 및 데이터 타입 보장
                self.pipe.stdin.write(bgr_frame.astype(np.uint8).tobytes())
            except (BrokenPipeError, OSError):
                print("FFmpeg pipe closed. Stopping recording.")
                self.pipe = None

    # [추가 제안] 자원 해제를 위한 메서드
    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()
            self.pipe = None
        cv2.destroyAllWindows()
