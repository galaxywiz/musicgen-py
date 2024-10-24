# pip install -r requirements.txt 
# 갱신시 pip freeze > requirements.txt
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import sounddevice as sd
import soundfile as sf
import time
import os
from tqdm import tqdm

class MusicGenerator:
    def __init__(self, model_size='small'):
        """
        MusicGen 모델을 초기화합니다.
        
        Args:
            model_size (str): 'small', 'medium', 'large' 중 선택
        """
        print(f"MusicGen {model_size} 모델을 로딩중...")
        self.model = MusicGen.get_pretrained(model_size)
        self.sample_rate = self.model.sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"모델이 {self.device}에 로드되었습니다.")

    def generate_music(self, prompt, duration=10, num_samples=1):
        """
        프롬프트를 기반으로 음악을 생성합니다.
        
        Args:
            prompt (str): 음악 생성을 위한 설명
            duration (int): 생성할 음악의 길이(초)
            num_samples (int): 생성할 음악의 수
        
        Returns:
            torch.Tensor: 생성된 음악 데이터
        """
        print(f"\n'{prompt}' 기반으로 음악을 생성중...")
        
        self.model.set_generation_params(
            duration=duration,
            temperature=0.95,  # 창의성 조절 (0.5~1.0)
            top_k=250,        # 샘플링할 상위 토큰 수
            top_p=0.95,       # 누적 확률 임계값
        )
        
        with torch.no_grad():
            generated = self.model.generate(
                [prompt] * num_samples,
                progress=True  # 진행바 표시
            )
            
        return generated

    def save_music(self, wav_data, filename):
        """
        생성된 음악을 파일로 저장합니다.
        
        Args:
            wav_data (torch.Tensor): 음악 데이터
            filename (str): 저장할 파일 이름
        """
        # 저장 디렉토리 생성
        os.makedirs('generated_music', exist_ok=True)
        filepath = os.path.join('generated_music', filename)
        
        # 음악 저장
        audio_write(
            filepath, wav_data.cpu(), self.sample_rate,
            strategy="loudness",  # 음량 정규화
        )
        print(f"\n음악이 저장되었습니다: {filepath}")
        
    def play_music(self, wav_data):
        """
        생성된 음악을 재생합니다.
        
        Args:
            wav_data (torch.Tensor): 재생할 음악 데이터
        """
        # torch tensor를 numpy 배열로 변환
        audio_data = wav_data.cpu().numpy()[0]  # 첫 번째 채널 선택
        
        print("\n음악을 재생합니다... (재생이 끝날 때까지 기다려주세요)")
        sd.play(audio_data, self.sample_rate)
        sd.wait()  # 재생이 끝날 때까지 대기

def get_style_prompt(base_prompt, style):
    """
    선택된 스타일에 따라 프롬프트를 생성합니다.
    """
    style_prompts = {
        '1': "웅장한 오케스트라 음악, 장엄한 분위기, 영화 사운드트랙 스타일",
        '2': "잔잔한 피아노 선율, 감성적이고 서정적인 분위기",
        '3': "신나는 팝 음악, 경쾌한 리듬과 멜로디",
        '4': "전자음이 가미된 현대적인 사운드, EDM 스타일",
        '5': base_prompt  # 사용자 정의 프롬프트
    }
    return style_prompts.get(style, base_prompt)

def main():
    print("=== MusicGen 음악 생성기 ===")
    
    # 모델 크기 선택
    print("\n사용할 모델 크기를 선택하세요:")
    print("1. small (빠르지만 품질이 낮음)")
    print("2. medium (중간 품질)")
    print("3. large (고품질이지만 느림)")
    
    model_sizes = {'1': 'small', '2': 'medium', '3': 'large'}
    model_choice = input("선택 (1-3) [기본: 1]: ").strip() or '1'
    model_size = model_sizes.get(model_choice, 'small')
    
    # 모델 초기화
    generator = MusicGenerator(model_size)
    
    while True:
        print("\n=== 음악 스타일 선택 ===")
        print("1. 오케스트라 / 영화음악")
        print("2. 감성적인 피아노")
        print("3. 팝 음악")
        print("4. EDM")
        print("5. 직접 프롬프트 입력")
        
        style = input("\n스타일을 선택하세요 (1-5): ").strip()
        
        # 사용자 정의 프롬프트 입력 받기
        if style == '5':
            base_prompt = input("\n원하는 음악을 자세히 설명해주세요: ").strip()
        else:
            base_prompt = ""
            
        prompt = get_style_prompt(base_prompt, style)
        
        # 음악 길이 설정
        duration = int(input("\n음악 길이(초)를 입력하세요 [기본: 10]: ").strip() or "10")
        
        try:
            # 음악 생성
            wav = generator.generate_music(prompt, duration)
            
            # 파일명 생성 및 저장
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"music_{timestamp}.wav"
            generator.save_music(wav[0], filename)
            
            # 음악 재생
            generator.play_music(wav)
            
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")
            continue
            
        # 계속 생성할지 확인
        if input("\n다른 음악을 생성하시겠습니까? (y/n): ").lower() != 'y':
            break
            
    print("\n프로그램을 종료합니다. 생성된 음악은 'generated_music' 폴더에 저장되어 있습니다.")

if __name__ == "__main__":
    main()