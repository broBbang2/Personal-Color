import os
import cv2
import dlib
import numpy as np

# Dlib 모델 로드
model_path = os.path.expanduser('~/Desktop/new/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# 결과 저장 폴더 생성
result_path = os.path.expanduser('~/Desktop/new/processed_results')
os.makedirs(result_path, exist_ok=True)

# 피부 영역 시각화 및 결과 저장 함수
def visualize_skin_region(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None, None

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
        skin_region = points[2:15]  # 피부 영역: 양쪽 뺨과 턱
        cv2.polylines(img, [skin_region], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, "Skin", tuple(skin_region[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(save_path, img)
    print(f"Processed and saved: {save_path}")
    return img, skin_region

# 이미지 처리 및 분류
target_path = os.path.expanduser('~/Desktop/generated_yellow-stylegan2')
test_images = [f for f in os.listdir(target_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

personal_colors = {
    "봄웜톤": [],
    "가을웜톤": [],
    "여름쿨톤": [],
    "겨울쿨톤": []
}

warm_tones = []
cool_tones = []
total_images = 0

for idx, img_file in enumerate(test_images):
    img_path = os.path.join(target_path, img_file)
    save_path = os.path.join(result_path, f"processed_{idx+1}.png")
    image, skin_region = visualize_skin_region(img_path, save_path)

    if image is None or skin_region is None:
        continue

    total_images += 1

    # 피부 영역 마스크 생성
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, skin_region, 255)
    skin_region_image = cv2.bitwise_and(image, image, mask=mask)

    # LAB 평균값 계산
    lab_image = cv2.cvtColor(skin_region_image, cv2.COLOR_BGR2Lab)
    l_mean = np.mean(lab_image[:, :, 0][mask == 255].astype(np.float32))
    a_mean = np.mean(lab_image[:, :, 1][mask == 255].astype(np.float32))
    b_mean = np.mean(lab_image[:, :, 2][mask == 255].astype(np.float32))

    # 웜톤/쿨톤 분류
    if b_mean > 147:
        warm_tones.append((img_file, image, l_mean, a_mean, b_mean))
        print(f"{img_file}: 웜톤으로 분류")
    else:
        cool_tones.append((img_file, image, l_mean, a_mean, b_mean))
        print(f"{img_file}: 쿨톤으로 분류")

# 세부 분류: 웜톤 (봄/가을)
for filename, image, l_mean, a_mean, b_mean in warm_tones:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv_image[:, :, 0])
    s_mean = np.mean(hsv_image[:, :, 1])
    if h_mean > 20 and s_mean > 60:
        personal_colors["봄웜톤"].append((filename, h_mean, s_mean, l_mean, a_mean, b_mean))
    else:
        personal_colors["가을웜톤"].append((filename, h_mean, s_mean, l_mean, a_mean, b_mean))

# 세부 분류: 쿨톤 (여름/겨울)
for filename, image, l_mean, a_mean, b_mean in cool_tones:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_mean = np.mean(hsv_image[:, :, 1])
    if s_mean < 65 and l_mean > 70:
        personal_colors["여름쿨톤"].append((filename, s_mean, l_mean, a_mean, b_mean))
    else:
        personal_colors["겨울쿨톤"].append((filename, s_mean, l_mean, a_mean, b_mean))

# 결과 출력
print(f"총 처리된 이미지 수: {total_images}")
for tone, images in personal_colors.items():
    count = len(images)
    print(f"{tone}: {count}장")
