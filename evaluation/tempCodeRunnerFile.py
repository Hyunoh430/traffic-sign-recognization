    # 결과 시각화 (읽기 가능한 복사본 생성)
    annotated_frame = results.render()[0].copy()

    # FPS 표시
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)