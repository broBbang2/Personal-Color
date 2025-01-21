import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# 모델 경로 및 클래스 라벨 설정
MODEL_PATH = "path/to/your/personal_color_model.h5"  # 사용자 모델 경로
CLASS_LABELS = ['봄웜톤', '여름쿨톤', '가을웜톤', '겨울쿨톤']

# 모델 로드
model = load_model(MODEL_PATH)
input_shape = model.input_shape  # 모델 입력 크기 확인
print(f"Model Input Shape: {input_shape}")

def predict_personal_color(image_path):
    """
    입력 이미지를 기반으로 퍼스널 컬러를 예측합니다.

    Args:
        image_path (str): 예측할 이미지 파일 경로

    Returns:
        tuple: (원본 이미지, 예측된 퍼스널 컬러, 신뢰도)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image cannot be loaded. Check the file path.")

    # 원본 이미지 유지 및 전처리
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[1], input_shape[2]))  # 모델 입력 크기
    image = np.array(image, dtype=np.float32) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가

    # 모델 예측
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    return original_image, CLASS_LABELS[predicted_class], confidence

def display_image_with_korean_text(image, result_text):
    """
    이미지 상단에 예측 결과를 한글로 표시합니다.

    Args:
        image (numpy.ndarray): 원본 이미지
        result_text (str): 표시할 텍스트 (예: "봄웜톤 (95.34%)")
    """
    # OpenCV 이미지를 PIL 이미지로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # 한글 폰트 설정 (MacOS 기본 폰트 경로)
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font = ImageFont.truetype(font_path, 40)

    # 텍스트 크기 계산
    text_bbox = font.getbbox(result_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (pil_image.width - text_width) // 2
    text_y = 10  # 텍스트 위치: 상단

    # 텍스트 배경 그리기
    draw.rectangle(
        [(text_x - 10, text_y - 10), (text_x + text_width + 10, text_y + text_height + 10)],
        fill=(0, 0, 0)
    )
    draw.text((text_x, text_y), result_text, font=font, fill=(255, 255, 255))

    # PIL 이미지를 다시 OpenCV 형식으로 변환
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 결과 이미지 출력
    cv2.imshow("Predicted Personal Color", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 테스트 실행
if __name__ == "__main__":
    TEST_IMAGE_PATH = "path/to/your/test_image.jpg"  # 테스트 이미지 경로

    try:
        # 퍼스널 컬러 예측
        original_image, color, confidence = predict_personal_color(TEST_IMAGE_PATH)
        result_text = f"{color} ({confidence:.2f}%)"

        # 결과 출력
        print("Prediction Result:")
        print(f"- Personal Color: {color}")
        print(f"- Confidence: {confidence:.2f}%")

        # 결과 이미지 표시
        display_image_with_korean_text(original_image, result_text)

    except ValueError as e:
        print(f"Error: {e}")

    except Exception as e:
        print(f"Unexpected Error: {e}")
