import numpy as np
import matplotlib.pyplot as plt

class NumpyCNNExplorer:
    """
    Numpy만을 사용하여 CNN의 핵심 연산(Conv, ReLU, Pooling)을 구현하고 
    데이터의 변화를 시각적으로 추적하는 클래스입니다.
    """
    def __init__(self):
        # 샘플 데이터: 세로 줄무늬가 있는 7x7 이미지 (흑백)
        self.sample_image = np.array([
            [10, 10, 10, 0, 0, 0, 0],
            [10, 10, 10, 0, 0, 0, 0],
            [10, 10, 10, 0, 10, 10, 10],
            [10, 10, 10, 0, 10, 10, 10],
            [10, 10, 10, 0, 10, 10, 10],
            [10, 10, 10, 0, 0, 0, 0],
            [10, 10, 10, 0, 0, 0, 0]
        ], dtype=float)
        
        # 세로 엣지 검출용 Sobel Filter (3x3)
        self.vertical_filter = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])

    @staticmethod
    def convolution2d(image, kernel, stride=1, padding=0):
        """
        다양한 하이퍼파라미터를 지원하는 정교한 2D 합성곱 연산
        """
        # 패딩 적용
        if padding > 0:
            image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
            
        img_h, img_w = image.shape
        k_h, k_w = kernel.shape
        
        # 출력 사이즈 계산 (O = (I - K + 2P) / S + 1)
        out_h = (img_h - k_h) // stride + 1
        out_w = (img_w - k_w) // stride + 1
        
        output = np.zeros((out_h, out_w))
        
        for y in range(0, out_h):
            for x in range(0, out_w):
                # 슬라이딩 윈도우 추출
                y_start, x_start = y * stride, x * stride
                patch = image[y_start:y_start+k_h, x_start:x_start+k_w]
                # 원소별 곱셈 후 합산 (내적)
                output[y, x] = np.sum(patch * kernel)
                
        return output

    @staticmethod
    def relu(x):
        """비선형 활성화 함수: 음수 제거"""
        return np.maximum(0, x)

    @staticmethod
    def max_pooling2d(image, pool_size=2, stride=2):
        """최댓값 풀링: 공간 해상도 축소 및 특징 강조"""
        img_h, img_w = image.shape
        out_h = (img_h - pool_size) // stride + 1
        out_w = (img_w - pool_size) // stride + 1
        
        output = np.zeros((out_h, out_w))
        
        for y in range(out_h):
            for x in range(out_w):
                y_s, x_s = y * stride, x * stride
                patch = image[y_s:y_s+pool_size, x_s:x_s+pool_size]
                output[y, x] = np.max(patch)
                
        return output

    def run_pipeline(self):
        print("--- CNN Pipeline Scratch Execution ---")
        
        # 1. Convolution (Edge Detection)
        conv_out = self.convolution2d(self.sample_image, self.vertical_filter, padding=1)
        print(f"Conv Output Shape: {conv_out.shape}")
        
        # 2. Activation (Non-linearity)
        relu_out = self.relu(conv_out)
        
        # 3. Max Pooling (Downsampling)
        pool_out = self.max_pooling2d(relu_out, pool_size=2, stride=2)
        print(f"Pool Output Shape: {pool_out.shape}")

        # 결과 시각화
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        
        axes[0].imshow(self.sample_image, cmap='gray')
        axes[0].set_title("Original Image")
        
        axes[1].imshow(conv_out, cmap='gray')
        axes[1].set_title("Convolution (Edge)")
        
        axes[2].imshow(relu_out, cmap='gray')
        axes[2].set_title("ReLU Activation")
        
        axes[3].imshow(pool_out, cmap='gray')
        axes[3].set_title("Max Pooling")
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    explorer = NumpyCNNExplorer()
    explorer.run_pipeline()
