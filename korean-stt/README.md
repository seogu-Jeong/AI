# Korean STT (한국어 음성인식)

M1 Mac에서 한국어 음성을 텍스트로 변환하는 음성인식 도구입니다.

## 구성

### 메인 모듈 (korean_stt/)
- 한국어 특화 음성인식 파이프라인
- 오디오 입력 처리 및 텍스트 출력

### transcribe/
- `transcribe.py`: Whisper 기반 음성-텍스트 변환 스크립트
- 오디오 파일(.m4a, .mp3, .wav)을 텍스트로 변환
- 사용법은 `사용법.md` 참고

## 기술 스택
- Python
- OpenAI Whisper
- Apple M1 최적화 (Core ML)

## 실행 방법
```bash
# transcribe 폴더에서
python transcribe.py <오디오파일>
```
