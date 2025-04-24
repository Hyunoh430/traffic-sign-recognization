import os
import xml.etree.ElementTree as ET

# 경로 설정
annotations_dir = 'C:/Users/2019124074/traffic-sign/kaggle_dataset/annotations'
images_dir = 'C:/Users/2019124074/traffic-sign/kaggle_dataset/images'

# 삭제된 파일 목록 저장
deleted_files = []

# 'speedlimit' 태그만 포함된 파일만 남기기
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_dir, annotation_file))
        root = tree.getroot()
        
        # 태그가 speedlimit만 있는지 확인
        tags = [obj.find('name').text for obj in root.findall('object')]
        
        if len(tags) == 1 and tags[0] == 'speedlimit':
            # speedlimit만 있으면 그대로 둠
            continue
        else:
            # 다른 태그가 있으면 삭제할 파일을 기록
            deleted_files.append(annotation_file)
            
            # annotations 폴더에서 XML 파일 삭제
            os.remove(os.path.join(annotations_dir, annotation_file))
            
            # images 폴더에서 해당 이미지 파일 삭제
            image_file = annotation_file.replace('.xml', '.png')  # 이미지 파일은 png 확장자로 가정
            if os.path.exists(os.path.join(images_dir, image_file)):
                deleted_files.append(image_file)
                os.remove(os.path.join(images_dir, image_file))

# 삭제된 파일 출력
print("Deleted files:")
for file in deleted_files:
    print(file)