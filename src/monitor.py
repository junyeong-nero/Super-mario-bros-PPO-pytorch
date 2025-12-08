import subprocess as sp
import numpy as np
import cv2


class Monitor:
    """Records the environment using ffmpeg."""

    def __init__(self, width: int, height: int, saved_path: str):
        # [설정] 동영상 인코딩을 위해 해상도를 짝수로 보정
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        self.width = width
        self.height = height
        self.saved_path = saved_path

        self.command = [
            "ffmpeg",
            "-y",  # 파일 덮어쓰기
            "-f",
            "rawvideo",  # 입력 포맷
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}X{height}",  # 입력 해상도
            "-pix_fmt",
            "bgr24",  # OpenCV 포맷 (BGR)
            "-r",
            "60",  # 입력 프레임레이트
            "-i",
            "-",  # 파이프에서 입력 받음
            "-an",  # 오디오 없음
            "-vcodec",
            "libx264",  # 호환성 좋은 H.264 코덱
            "-preset",
            "fast",  # 인코딩 속도
            "-pix_fmt",
            "yuv420p",  # 플레이어 호환성 포맷
            saved_path,
        ]

        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            print("Error: ffmpeg not found. Video recording disabled.")
            self.pipe = None

    def record_frame_stack(self, frame_stack) -> None:
        """
        LazyFrames 또는 (1, H, W, C) 형태의 입력을 받아 녹화합니다.
        Input: (1, 240, 256, 3) 등의 형태
        """
        # 1. LazyFrames -> Numpy Array 변환
        frame = np.array(frame_stack)

        # 2. 불필요한 차원 제거 (Batch 차원 제거)
        # 예: (1, 240, 256, 3) -> (240, 256, 3)
        frame = np.squeeze(frame)

        # 3. 데이터 타입 및 값 범위 보정 (Float 0~1 -> Uint8 0~255)
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        # 4. 차원 확인 및 BGR 변환
        bgr_frame = None

        # (H, W, C) 형태일 때
        if frame.ndim == 3:
            if frame.shape[2] == 3:  # RGB -> BGR
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 1:  # Gray(Channel last) -> BGR
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # 만약 FrameStack으로 인해 (H, W, 4) 처럼 4채널(흑백 4장)이 들어온다면
            # 가장 최근 프레임(보통 마지막)만 사용
            elif frame.shape[2] > 3:
                single_frame = frame[:, :, -1]  # 마지막 채널만 가져옴
                bgr_frame = cv2.cvtColor(single_frame, cv2.COLOR_GRAY2BGR)

        # (C, H, W) 형태일 때 (PyTorch 스타일)
        elif frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
            if frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = cv2.cvtColor(frame[:, :, -1], cv2.COLOR_GRAY2BGR)

        # 5. 녹화 및 화면 표시
        if bgr_frame is not None:
            self._write_frame(bgr_frame)

    def record(self, env) -> None:
        """기존 방식: env.render() 결과를 녹화"""
        try:
            frame = env.render()
        except Exception:
            frame = None

        if frame is None:
            try:
                frame = env.render(mode="rgb_array")
            except:
                pass

        if frame is None:
            return

        bgr_frame = None
        if isinstance(frame, np.ndarray):
            frame_np = np.asarray(frame)

            if frame_np.ndim == 3 and frame_np.shape[2] in (3, 4):
                code = (
                    cv2.COLOR_RGB2BGR if frame_np.shape[2] == 3 else cv2.COLOR_RGBA2BGR
                )
                bgr_frame = cv2.cvtColor(frame_np, code)
            elif frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
                transposed = np.transpose(frame_np, (1, 2, 0))
                code = (
                    cv2.COLOR_RGB2BGR
                    if transposed.shape[2] == 3
                    else cv2.COLOR_RGBA2BGR
                )
                bgr_frame = cv2.cvtColor(transposed, code)
            elif frame_np.ndim == 2:
                bgr_frame = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)

            if bgr_frame is not None:
                self._write_frame(bgr_frame)

    def _write_frame(self, bgr_frame):
        """내부 함수: 프레임 리사이즈, 표시, FFmpeg 쓰기 공통 로직"""
        # 리사이즈
        if (bgr_frame.shape[1] != self.width) or (bgr_frame.shape[0] != self.height):
            bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

        # 화면 표시 (Human 모드 효과)
        # try:
        #     cv2.imshow("Super Mario Bros PPO", bgr_frame)
        #     cv2.waitKey(1)
        # except:
        #     pass

        # FFmpeg 쓰기
        if self.pipe is not None:
            try:
                self.pipe.stdin.write(bgr_frame.astype(np.uint8).tobytes())
            except (BrokenPipeError, OSError):
                pass

    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()
            self.pipe = None
        cv2.destroyAllWindows()
