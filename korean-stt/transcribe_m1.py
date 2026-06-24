import os
import glob
import mlx_whisper
import time

# ==========================================
# ⚙️ 사용자 설정 (원하는 대로 수정하세요)
# ==========================================

# 1. 사용할 모델 선택 (MLX 최적화 모델)
# - "mlx-community/whisper-tiny"          : 가장 빠름, 정확도 낮음
# - "mlx-community/whisper-base"          : 빠름, 가벼움
# - "mlx-community/whisper-small"         : 속도/정확도 균형
# - "mlx-community/whisper-medium"        : 정확도 높음, 속도 느림
# - "mlx-community/whisper-large-v3"      : 최고 정확도, 가장 느림
# - "mlx-community/whisper-large-v3-turbo" : (추천) 매우 높은 정확도 + 준수한 속도
MODEL_NAME = "mlx-community/whisper-large-v3-turbo" 

# 2. 경로 설정
INPUT_DIR = "audio_input"    # 음성 파일이 들어있는 폴더
OUTPUT_DIR = "text_output"   # 결과 텍스트가 저장될 폴더

# 3. 언어 설정
LANGUAGE = "ko"  # 한국어: "ko", 영어: "en", 자동감지: None

# ==========================================

def transcribe_audio():
    # 폴더가 없으면 생성
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"'{INPUT_DIR}' 폴더가 생성되었습니다. 음성 파일을 넣어주세요.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 지원하는 오디오 확장자
    extensions = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac")
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not audio_files:
        print(f"'{INPUT_DIR}' 폴더에 오디오 파일이 없습니다.")
        return

    print(f"사용 모델: {MODEL_NAME}")
    print(f"총 {len(audio_files)}개의 파일을 찾았습니다. 변환을 시작합니다...")

    for audio_path in audio_files:
        file_name = os.path.basename(audio_path)
        base_name = os.path.splitext(file_name)[0]
        final_output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        print(f"처리 중: {file_name}...", end="", flush=True)
        start_time = time.time()

        try:
            # MLX를 사용하여 M1 GPU/Neural Engine 활용
            result = mlx_whisper.transcribe(
                audio_path, 
                path_or_hf_repo=MODEL_NAME,
                language=LANGUAGE
            )

            # 텍스트 파일 저장
            with open(final_output_path, "w", encoding="utf-8") as f:
                f.write(result["text"].strip())

            elapsed_time = time.time() - start_time
            print(f" 완료! ({elapsed_time:.2f}초)")

        except Exception as e:
            print(f"\n 오류 발생 ({file_name}): {e}")

if __name__ == "__main__":
    transcribe_audio()
