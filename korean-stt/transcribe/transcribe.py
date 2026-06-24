#!/usr/bin/env python3
"""
로컬 음성 텍스트 변환 (100% 오프라인, 외부 API 없음)
실행: python3 transcribe.py
"""

# ============================================================
#  여기에 경로를 입력하고 python3 transcribe.py 로 실행하세요
# ============================================================

PATH = "/Users/jsw/Desktop/무제 폴더/문학과영상예술"
# 파일 예시:  PATH = "/Users/jsw/Downloads/책 물리학 0421.m4a"
# 폴더 예시:  PATH = "/Users/jsw/Downloads/녹음폴더"

MODEL = "small"
# tiny  → 빠르지만 부정확  |  base → 기본값  |  small → 균형
# medium → 높은 정확도    |  large-v3 → 최고 정확도 (느림)

LANGUAGE = "ko"
# None → 자동 감지  |  "en" → 영어 고정  |  "ko" → 한국어 고정

TIMESTAMPS = False
# False → 타임스탬프 없음  |  True → [00:00 → 00:05] 형식으로 출력

# ============================================================

import argparse
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma",
                    ".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODEL_INFO = {
    "tiny":     "~75MB,  가장 빠름, 낮은 정확도",
    "base":     "~145MB, 빠름,     적당한 정확도",
    "small":    "~466MB, 보통,     좋은 정확도",
    "medium":   "~1.5GB, 느림,     높은 정확도",
    "large-v2": "~3GB,   매우 느림, 매우 높은 정확도",
    "large-v3": "~3GB,   매우 느림, 최고 정확도",
}


def check_dependencies():
    missing = []
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    try:
        import static_ffmpeg
    except ImportError:
        missing.append("static-ffmpeg")
    if missing:
        print("필요한 패키지가 없습니다. 아래 명령어로 설치하세요:", file=sys.stderr)
        print(f"  pip3 install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h > 0 else f"{m:02d}:{s:05.2f}"


def collect_audio_files(folder: Path) -> list[Path]:
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def transcribe_file(model, audio_path: Path, args) -> list[str]:
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=5,
        vad_filter=True,
    )
    print(f"  감지된 언어: {info.language} (신뢰도: {info.language_probability:.0%})", file=sys.stderr)

    lines = []
    for segment in segments:
        if args.timestamps:
            line = f"[{format_time(segment.start)} → {format_time(segment.end)}]  {segment.text.strip()}"
        else:
            line = segment.text.strip()
        if line:
            lines.append(line)
    return lines


def save_txt(lines: list[str], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    # 상단 설정값을 사용하되, 터미널에서 인자를 넘기면 그쪽이 우선
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--timestamps", action="store_true", default=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-h", "--help", action="store_true")
    cli = parser.parse_args()

    if cli.help:
        print(__doc__)
        print("상단 PATH, MODEL, LANGUAGE, TIMESTAMPS 변수를 수정한 뒤 실행하세요.")
        sys.exit(0)

    # 터미널 인자 > 코드 상단 설정값 순서로 적용
    path_str   = cli.path      or PATH
    model_str  = cli.model     or MODEL
    lang       = cli.language  or LANGUAGE
    timestamps = cli.timestamps or TIMESTAMPS
    device     = cli.device

    if not path_str:
        print("오류: PATH가 비어 있습니다.", file=sys.stderr)
        print("transcribe.py 상단의 PATH = \"...\" 에 경로를 입력하세요.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(path_str)

    class Args:
        pass

    args = Args()
    args.language   = lang
    args.timestamps = timestamps
    args.output     = cli.output
    args.device     = device

    if not input_path.exists():
        print(f"\n오류: 경로를 찾을 수 없습니다 → {path_str}", file=sys.stderr)
        sys.exit(1)

    check_dependencies()

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass

    from faster_whisper import WhisperModel

    print(f"\n[모델] {model_str}  ({MODEL_INFO[model_str]})", file=sys.stderr)
    print(f"[언어] {'자동 감지' if not args.language else args.language}", file=sys.stderr)
    print("모델 로딩 중...", file=sys.stderr)
    model = WhisperModel(model_str, device=device, compute_type="int8")

    # ── 폴더 모드 ──────────────────────────────────────────────
    if input_path.is_dir():
        audio_files = collect_audio_files(input_path)

        if not audio_files:
            print(f"\n오디오 파일이 없습니다 → {input_path}", file=sys.stderr)
            sys.exit(0)

        total = len(audio_files)
        print(f"\n폴더: {input_path}", file=sys.stderr)
        print(f"오디오 파일 {total}개 발견\n", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        ok, fail = 0, 0
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{total}] {audio_file.name}", file=sys.stderr)
            try:
                lines = transcribe_file(model, audio_file, args)
                output_path = audio_file.with_suffix(".txt")
                save_txt(lines, output_path)
                print(f"  저장 완료 → {output_path.name}", file=sys.stderr)
                ok += 1
            except Exception as e:
                print(f"  오류 발생: {e}", file=sys.stderr)
                fail += 1

        print("\n" + "=" * 60, file=sys.stderr)
        print(f"완료: 성공 {ok}개", file=sys.stderr)
        if fail:
            print(f"      실패 {fail}개", file=sys.stderr)

    # ── 단일 파일 모드 ─────────────────────────────────────────
    else:
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            print(f"\n경고: 지원하지 않는 확장자입니다 ({input_path.suffix})", file=sys.stderr)

        print(f"\n[파일] {input_path.name}", file=sys.stderr)
        print("변환 중...\n", file=sys.stderr)

        lines = transcribe_file(model, input_path, args)
        print("-" * 60, file=sys.stderr)

        for line in lines:
            print(line)

        if args.output:
            save_txt(lines, Path(args.output))
            print(f"\n저장 완료 → {args.output}", file=sys.stderr)
        else:
            # --output 없으면 파일과 같은 폴더에 자동 저장
            auto_out = input_path.with_suffix(".txt")
            save_txt(lines, auto_out)
            print(f"\n저장 완료 → {auto_out}", file=sys.stderr)

        print("완료!", file=sys.stderr)


if __name__ == "__main__":
    main()
